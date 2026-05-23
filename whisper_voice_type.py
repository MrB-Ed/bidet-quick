"""
whisper_voice_type.py
=====================
Local Whisper-based voice-typing replacement for Windows H.

Hotkey: Ctrl+Shift+;  (toggle: press once to start, press again to stop)
Backup hotkey: Right Alt  (push-to-talk: hold while speaking)

When transcribed, the text is pasted into the focused field via clipboard + Ctrl+V.
Audio + transcript are saved to ~/whisper_corpus/<timestamp>/ for future XTTSv2 voice cloning.

Visual feedback layers (added 2026-05-22, extended PM):
  - System tray icon with 5 states: gray (idle), red (recording), yellow (transcribing), green flash (success), orange flash (error)
  - Native Windows toast notifications on error paths (no audio, too short, transcribe fail, model not loaded)
  - Toast on recording START (defensive depth so Mark never brain-dumps into the ether)
  - Soft chime on successful insert (PlaySound SystemAsterisk)
  - Distinct chime on recording START (PlaySound SystemQuestion) — different pitch from completion
  - Floating always-on-top indicator on the monitor where the mouse cursor lives — visible no matter which monitor Mark is looking at
  - Every state transition logged to logs/whisper_<date>.log
  Env vars to quiet down:
    BIDET_QUICK_QUIET=1     -> disable start chime + start toast + floating indicator (keeps tray + completion chime)
    BIDET_QUICK_NO_FLOAT=1  -> disable just the floating indicator (keep audio cues)
    BIDET_QUICK_NO_START_CHIME=1  -> disable just the start chime

Architecture:
  - faster-whisper large-v3 on CUDA float16 (RTX 4070, ~3.9 GB VRAM, ~9x real-time)
  - sounddevice 16 kHz mono PCM capture (Whisper-native rate)
  - keyboard module for global hotkey
  - pyperclip + keyboard.send('ctrl+v') for insertion (works in every Windows app)
  - pystray for system tray indicator
  - winotify (preferred) / win10toast (fallback) for native Windows toasts

Author: Claude (Apex Junior) for Mark Barnett, 2026-05-22
"""

from __future__ import annotations
import os
import sys
import time
import wave
import threading
import datetime
import logging
import traceback
from pathlib import Path

# Make CUDA libs (cuBLAS, cuDNN) shipped via pip packages discoverable
# BEFORE importing ctranslate2 (faster-whisper backend). ctranslate2 is a
# native extension that follows the standard Windows DLL search order, so we
# must mutate PATH itself (os.add_dll_directory is only honored by ctypes).
def _add_cuda_dlls():
    import site
    extra = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin", "cuda_runtime/bin"):
            p = Path(sp) / "nvidia" / sub
            if p.is_dir():
                extra.append(str(p))
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(p))
                    except OSError:
                        pass
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")
_add_cuda_dlls()

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
from PIL import Image, ImageDraw

# ---------------- Config ----------------
HOME = Path(os.path.expanduser("~"))
CORPUS_DIR = HOME / "whisper_corpus"
LOG_DIR = Path(__file__).parent / "logs"
MODEL_DIR = Path(__file__).parent / "models"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "en"          # forced English (Mark is American English)
SAMPLE_RATE = 16000      # Whisper native rate
CHANNELS = 1

TOGGLE_HOTKEY = "ctrl+shift+;"   # primary: toggle
PTT_HOTKEY = "right alt"          # secondary: push-to-talk (hold while speaking)

# Minimum audio duration before we accept it as a real recording
MIN_AUDIO_SECONDS = 0.5
# RMS threshold below which we treat the recording as silence
SILENCE_RMS_THRESHOLD = 0.003

# Quiet knobs (default = noisy: start chime + start toast + floating indicator ON)
QUIET_MODE = os.environ.get("BIDET_QUICK_QUIET", "0") == "1"
CHIME_ON_SUCCESS = True
CHIME_ON_START   = not QUIET_MODE and os.environ.get("BIDET_QUICK_NO_START_CHIME", "0") != "1"
TOAST_ON_START   = not QUIET_MODE and os.environ.get("BIDET_QUICK_NO_START_TOAST", "0") != "1"
FLOAT_INDICATOR  = not QUIET_MODE and os.environ.get("BIDET_QUICK_NO_FLOAT", "0") != "1"

# ---------------- Logging ----------------
LOG_FILE = LOG_DIR / f"whisper_{datetime.datetime.now():%Y-%m-%d}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("whisper_stt")

# ---------------- Toast notifications (winotify preferred, win10toast fallback) ----------------
_TOAST_BACKEND = None
_winotify_Notification = None
_winotify_audio = None
_win10toaster = None
try:
    from winotify import Notification as _winotify_Notification  # noqa: E402
    from winotify import audio as _winotify_audio  # noqa: E402
    _TOAST_BACKEND = "winotify"
    log.info("Toast backend: winotify")
except Exception:
    try:
        from win10toast import ToastNotifier  # noqa: E402
        _win10toaster = ToastNotifier()
        _TOAST_BACKEND = "win10toast"
        log.info("Toast backend: win10toast")
    except Exception:
        log.warning("No toast backend available (winotify / win10toast); errors will only hit the log file")


def _toast(title: str, body: str = "", urgent: bool = False) -> None:
    """Fire a native Windows toast. Non-blocking, swallow all errors."""
    log.info("TOAST [%s] %s :: %s", "!" if urgent else "i", title, body)
    if _TOAST_BACKEND is None:
        return
    def _fire():
        try:
            if _TOAST_BACKEND == "winotify":
                # winotify needs an app_id; "Whisper STT" shows as the source
                n = _winotify_Notification(
                    app_id="Whisper STT",
                    title=title,
                    msg=body or " ",
                    duration="short",
                )
                # Only ping the speakers on urgent toasts; success has its own chime
                if urgent:
                    n.set_audio(_winotify_audio.Reminder, loop=False)
                else:
                    n.set_audio(_winotify_audio.Silent, loop=False)
                n.show()
            elif _TOAST_BACKEND == "win10toast":
                _win10toaster.show_toast(
                    title,
                    body,
                    duration=4 if urgent else 3,
                    threaded=True,
                )
        except Exception as e:
            log.warning("Toast send failed: %s", e)
    threading.Thread(target=_fire, daemon=True).start()


# ---------------- Soft chime ----------------
def _chime():
    """Completion chime (Asterisk = soft ding)."""
    if not CHIME_ON_SUCCESS:
        return
    try:
        import winsound
        # SND_ASYNC = non-blocking, SND_ALIAS = play the named system event
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception as e:
        log.warning("Chime failed: %s", e)


def _chime_start():
    """Recording-START chime (Question = distinct double-pitch). Tells Mark the mic is hot
    even when his eyes are on the wrong monitor."""
    if not CHIME_ON_START:
        return
    try:
        import winsound
        winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception as e:
        log.warning("Start chime failed: %s", e)


# ---------------- Whisper model (loaded once, kept in VRAM) ----------------
log.info("Loading faster-whisper model=%s device=%s compute_type=%s ...", MODEL_NAME, DEVICE, COMPUTE_TYPE)
from faster_whisper import WhisperModel  # noqa: E402

_model_ready = False
_load_t0 = time.time()
try:
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=str(MODEL_DIR),
    )
    log.info("Model ready in %.1fs", time.time() - _load_t0)
    _model_ready = True
except Exception as e:
    log.error("Model load FAILED: %s", e)
    log.error(traceback.format_exc())
    model = None
    _toast("Whisper STT: model load failed", str(e)[:200], urgent=True)

# ---------------- Recorder ----------------
class Recorder:
    """Records 16 kHz mono PCM into an in-memory list of np.float32 chunks."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.recording = False
        self.lock = threading.Lock()

    def _callback(self, indata, _frames, _t, status):
        if status:
            log.warning("sounddevice status: %s", status)
        # mono float32
        self.frames.append(indata.copy().reshape(-1))

    def start(self):
        with self.lock:
            if self.recording:
                return
            self.frames = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
            self.recording = True
            log.info("Recording started")

    def stop(self) -> np.ndarray | None:
        with self.lock:
            if not self.recording:
                return None
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log.warning("Stream stop error: %s", e)
            self.stream = None
            self.recording = False
        if not self.frames:
            return None
        audio = np.concatenate(self.frames, axis=0)
        log.info("Recording stopped: %.2fs of audio", len(audio) / SAMPLE_RATE)
        return audio


recorder = Recorder()

# ---------------- System tray ----------------
import pystray  # noqa: E402

_tray_icon: pystray.Icon | None = None
_state = {"phase": "idle", "msg": "idle"}   # phase in {idle, recording, transcribing, success}
_flash_timer: threading.Timer | None = None

# Icon color palette (R, G, B)
_COLOR_IDLE        = (130, 130, 130)   # gray
_COLOR_RECORDING   = (220, 30, 30)     # red
_COLOR_TRANSCRIBING= (240, 200, 30)    # yellow
_COLOR_SUCCESS     = (40, 200, 70)     # green
_COLOR_ERROR       = (220, 100, 30)    # orange (used briefly before returning to idle)

def _make_icon_image(color: tuple) -> Image.Image:
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=color)
    # subtle dark outline for visibility on light taskbars
    d.ellipse((6, 6, 58, 58), outline=(20, 20, 20), width=2)
    return img


# ---------------- Floating always-on-top indicator ----------------
# Mark has two monitors. The tray icon lives on one — if he's looking at the
# other while recording, he can't see the state change and brain-dumps into
# the ether. This floating dot positions itself on the monitor where the
# CURSOR is, so it follows his attention. Click-through (won't block clicks
# behind it), small, semi-transparent, hidden when idle.
class _FloatingIndicator:
    """Small always-on-top dot that appears on the active monitor while recording
    or transcribing. Driven from any thread via show(state) / hide(); the actual
    tkinter calls hop onto the indicator's own thread via after()."""

    SIZE_PX = 56
    MARGIN_PX = 24

    _COLOR_HEX = {
        "recording":    "#dc1e1e",  # red
        "transcribing": "#f0c81e",  # yellow
        "success":      "#28c846",  # green
        "error":        "#dc6418",  # orange
    }

    def __init__(self):
        self._tk = None
        self._canvas = None
        self._dot = None
        self._ready = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Cache Win32 lookups
        self._user32 = None
        try:
            import ctypes
            self._user32 = ctypes.windll.user32
        except Exception as e:
            log.warning("Float indicator: ctypes unavailable: %s", e)

    # --- Win32 helpers (no extra deps) ---
    def _active_monitor_rect(self):
        """Return (left, top, right, bottom) of the work-area of the monitor
        under the cursor. Falls back to the primary monitor on failure."""
        try:
            import ctypes
            from ctypes import wintypes
            user32 = self._user32 or ctypes.windll.user32

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                            ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

            p = POINT()
            user32.GetCursorPos(ctypes.byref(p))
            mh = user32.MonitorFromPoint(p, 2)  # MONITOR_DEFAULTTONEAREST
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(mh, ctypes.byref(mi))
            r = mi.rcWork
            return (r.left, r.top, r.right, r.bottom)
        except Exception as e:
            log.warning("active_monitor_rect failed, falling back: %s", e)
            # Fall back to primary monitor screen size
            try:
                if self._tk is not None:
                    sw = self._tk.winfo_screenwidth()
                    sh = self._tk.winfo_screenheight()
                    return (0, 0, sw, sh)
            except Exception:
                pass
            return (0, 0, 1920, 1080)

    def _make_click_through(self):
        """Make the window pass clicks through (WS_EX_TRANSPARENT) AND not steal
        focus (WS_EX_NOACTIVATE) AND not appear in alt-tab/taskbar (WS_EX_TOOLWINDOW).
        We deliberately DO NOT mix this with -transparentcolor because that combo
        makes Tk render the dot invisibly on some Windows builds. Instead we use
        an opaque colored window at -alpha 0.82 — a small red square, clearly visible."""
        try:
            import ctypes
            from ctypes import wintypes
            user32 = self._user32 or ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW  = 0x00000080
            WS_EX_NOACTIVATE  = 0x08000000
            hwnd = self._tk.winfo_id()
            GetParent = user32.GetParent
            GetParent.restype = wintypes.HWND
            GetParent.argtypes = [wintypes.HWND]
            top = hwnd
            while True:
                parent = GetParent(top)
                if not parent:
                    break
                top = parent
            ex = user32.GetWindowLongW(top, GWL_EXSTYLE)
            # NOTE: do NOT add WS_EX_LAYERED here ourselves — Tk's -alpha already
            # sets it, and setting it twice with our own alpha-blend can confuse
            # the compositor and make the window invisible on some builds.
            ex |= WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(top, GWL_EXSTYLE, ex)
        except Exception as e:
            log.warning("Float indicator: click-through setup failed (will still be visible): %s", e)

    def _run(self):
        try:
            import tkinter as tk
            self._tk = tk.Tk()
            self._tk.overrideredirect(True)             # no title bar / borders
            self._tk.attributes("-topmost", True)        # always on top
            self._tk.attributes("-alpha", 0.82)          # semi-transparent (Tk sets WS_EX_LAYERED itself)
            # Opaque colored fill — the whole window is the dot. Square shape, but at
            # 56x56 with the recording-red color it reads as a clear "RECORDING" marker.
            # We deliberately skip -transparentcolor: combining it with WS_EX_TRANSPARENT
            # makes the dot invisible on some Win10/11 builds.
            self._tk.configure(bg=self._COLOR_HEX["recording"])
            self._tk.geometry(f"{self.SIZE_PX}x{self.SIZE_PX}+-1000+-1000")  # park offscreen until first show
            # Canvas inside the window for a rounded look via an oval painted on the colored bg
            self._canvas = tk.Canvas(
                self._tk,
                width=self.SIZE_PX, height=self.SIZE_PX,
                bg=self._COLOR_HEX["recording"],
                highlightthickness=0, bd=0,
            )
            self._canvas.pack(fill="both", expand=True)
            # An outline ring for visual polish — even on top of the colored window,
            # this gives the dot a defined edge against busy desktop backgrounds.
            self._canvas.create_oval(
                2, 2, self.SIZE_PX - 2, self.SIZE_PX - 2,
                outline="#202020", width=2,
            )
            # Hide initially; only appear when state demands it
            self._tk.withdraw()
            self._ready.set()
            self._tk.update_idletasks()
            self._make_click_through()
            self._tk.mainloop()
        except Exception as e:
            log.warning("Float indicator thread crashed: %s", e)
            log.warning(traceback.format_exc())
            self._ready.set()  # unblock anyone waiting

    def _do_show(self, state):
        if self._tk is None or self._canvas is None:
            return
        try:
            color = self._COLOR_HEX.get(state, self._COLOR_HEX["recording"])
            # Repaint both the window bg and the canvas bg so the whole 56x56 takes the color
            self._tk.configure(bg=color)
            self._canvas.configure(bg=color)
            l, t, r, b = self._active_monitor_rect()
            # Bottom-right of the active monitor's work area, with a margin
            x = r - self.SIZE_PX - self.MARGIN_PX
            y = b - self.SIZE_PX - self.MARGIN_PX
            self._tk.geometry(f"{self.SIZE_PX}x{self.SIZE_PX}+{x}+{y}")
            self._tk.deiconify()
            self._tk.lift()
            self._tk.attributes("-topmost", True)
            # Re-apply click-through (deiconify can reset some ex-styles on some builds)
            self._make_click_through()
        except Exception as e:
            log.warning("Float indicator show failed: %s", e)

    def _do_hide(self):
        if self._tk is None:
            return
        try:
            self._tk.withdraw()
        except Exception as e:
            log.warning("Float indicator hide failed: %s", e)

    def show(self, state: str):
        if not self._ready.wait(timeout=2.0) or self._tk is None:
            return
        try:
            self._tk.after(0, lambda: self._do_show(state))
        except Exception as e:
            log.warning("Float indicator show schedule failed: %s", e)

    def hide(self):
        if not self._ready.wait(timeout=2.0) or self._tk is None:
            return
        try:
            self._tk.after(0, self._do_hide)
        except Exception as e:
            log.warning("Float indicator hide schedule failed: %s", e)


_float_indicator: _FloatingIndicator | None = None
if FLOAT_INDICATOR:
    try:
        _float_indicator = _FloatingIndicator()
        log.info("Floating indicator: enabled")
    except Exception as e:
        log.warning("Floating indicator init failed (continuing without): %s", e)
        _float_indicator = None
else:
    log.info("Floating indicator: disabled (BIDET_QUICK_QUIET or BIDET_QUICK_NO_FLOAT set)")


def _set_phase(phase: str, msg: str | None = None):
    """Drive the tray icon + title from a single phase string."""
    global _flash_timer
    _state["phase"] = phase
    if msg is not None:
        _state["msg"] = msg
    if _tray_icon is None:
        return
    color_map = {
        "idle": _COLOR_IDLE,
        "recording": _COLOR_RECORDING,
        "transcribing": _COLOR_TRANSCRIBING,
        "success": _COLOR_SUCCESS,
        "error": _COLOR_ERROR,
    }
    color = color_map.get(phase, _COLOR_IDLE)
    titles = {
        "idle":         f"Whisper STT: idle ({_state['msg']})",
        "recording":    f"Whisper STT: RECORDING ({_state['msg']})",
        "transcribing": f"Whisper STT: transcribing... ({_state['msg']})",
        "success":      f"Whisper STT: done ({_state['msg']})",
        "error":        f"Whisper STT: error ({_state['msg']})",
    }
    try:
        _tray_icon.icon = _make_icon_image(color)
        _tray_icon.title = titles.get(phase, titles["idle"])
    except Exception as e:
        log.warning("Tray update failed: %s", e)
    log.info("STATE %s :: %s", phase, _state["msg"])

    # Drive the floating indicator (active-monitor dot) off the same phase
    if _float_indicator is not None:
        try:
            if phase == "idle":
                _float_indicator.hide()
            else:
                _float_indicator.show(phase)
        except Exception as e:
            log.warning("Float indicator update failed: %s", e)

    # Schedule a return to idle after a success/error flash
    if phase in ("success", "error"):
        if _flash_timer is not None:
            try:
                _flash_timer.cancel()
            except Exception:
                pass
        _flash_timer = threading.Timer(1.0, lambda: _set_phase("idle", "ready"))
        _flash_timer.daemon = True
        _flash_timer.start()


# ---------------- Transcribe + insert ----------------
def _save_corpus(audio: np.ndarray, transcript: str) -> Path:
    """Save WAV + transcript to ~/whisper_corpus/<ts>/."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = CORPUS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    # Convert float32 [-1, 1] to int16 PCM
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    log.info("Corpus saved: %s", out_dir)
    return out_dir


def _insert_text(text: str) -> None:
    """Paste text into focused field via clipboard + Ctrl+V."""
    if not text:
        return
    # Preserve clipboard contents
    prev = ""
    try:
        prev = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    # tiny delay so clipboard settles
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    # Restore clipboard after 1.5s (give Windows time to paste)
    def _restore():
        time.sleep(1.5)
        try:
            pyperclip.copy(prev)
        except Exception:
            pass
    threading.Thread(target=_restore, daemon=True).start()


def _audio_is_silent(audio: np.ndarray) -> bool:
    if audio is None or len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    log.info("Audio RMS=%.5f (threshold=%.5f)", rms, SILENCE_RMS_THRESHOLD)
    return rms < SILENCE_RMS_THRESHOLD


def _transcribe_and_insert(audio: np.ndarray | None) -> None:
    # Guard: model load may have failed
    if not _model_ready or model is None:
        log.error("Transcribe requested but model not loaded")
        _set_phase("error", "model not loaded")
        _toast("Whisper STT: model not loaded",
               "The Whisper model failed to load. Check logs/whisper_<date>.log.",
               urgent=True)
        return

    # Guard: nothing recorded
    if audio is None or len(audio) == 0:
        log.info("No audio frames captured")
        _set_phase("error", "no audio")
        _toast("Whisper STT: no audio detected",
               "Recording produced no audio. Check default mic in Windows Sound settings.",
               urgent=True)
        return

    audio_sec = len(audio) / SAMPLE_RATE

    # Guard: too short
    if audio_sec < MIN_AUDIO_SECONDS:
        log.info("Audio too short: %.2fs < %.2fs", audio_sec, MIN_AUDIO_SECONDS)
        _set_phase("error", f"too short ({audio_sec:.2f}s)")
        _toast("Whisper STT: no audio detected",
               f"Recording was only {audio_sec:.2f}s. Hold the hotkey while you speak.",
               urgent=True)
        return

    # Guard: silent (mic muted / dead)
    if _audio_is_silent(audio):
        log.info("Audio is silent (RMS below threshold)")
        _set_phase("error", "silent")
        _toast("Whisper STT: no audio detected",
               "Recording was silent. Is your mic muted or unplugged?",
               urgent=True)
        return

    # Move to transcribing state
    _set_phase("transcribing", f"{audio_sec:.1f}s")

    t0 = time.time()
    try:
        segments, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        log.error("Transcription failed: %s", e)
        log.error(traceback.format_exc())
        _set_phase("error", "transcribe failed")
        _toast("Whisper STT: transcription failed",
               f"{type(e).__name__}: {str(e)[:200]}",
               urgent=True)
        return

    dur = time.time() - t0
    rt = audio_sec / max(dur, 0.001)
    log.info("Transcribed %.2fs audio in %.2fs (%.1fx RT): %r",
             audio_sec, dur, rt, text)

    if not text:
        log.info("Empty transcript (likely silence after VAD)")
        _set_phase("error", "empty transcript")
        _toast("Whisper STT: transcription empty",
               "Whisper returned an empty result. Try speaking louder or closer to the mic.",
               urgent=True)
        return

    try:
        _save_corpus(audio, text)
    except Exception as e:
        log.warning("Corpus save failed (continuing with insert): %s", e)

    _insert_text(text)
    _chime()
    _set_phase("success", f"{audio_sec:.1f}s -> {len(text)}ch")


# ---------------- Hotkey handlers ----------------
_toggle_busy = threading.Lock()

def _on_toggle():
    """Press once to start, press again to stop + transcribe + insert."""
    if not _toggle_busy.acquire(blocking=False):
        return
    try:
        if not _model_ready:
            _toast("Whisper STT: model not loaded",
                   "Hotkey pressed before the model finished loading.",
                   urgent=True)
            return
        if not recorder.recording:
            recorder.start()
            _chime_start()
            if TOAST_ON_START:
                _toast("Whisper STT: recording started",
                       f"Speak now. Press {TOGGLE_HOTKEY} again to stop.",
                       urgent=False)
            _set_phase("recording", "toggle")
        else:
            audio = recorder.stop()
            # transcription runs off the hotkey thread; it sets its own phase
            threading.Thread(target=_transcribe_and_insert, args=(audio,), daemon=True).start()
    finally:
        _toggle_busy.release()


_ptt_active = {"on": False}

def _on_ptt_press(_e):
    if _ptt_active["on"]:
        return
    if not _model_ready:
        _ptt_active["on"] = False
        _toast("Whisper STT: model not loaded",
               "Hotkey pressed before the model finished loading.",
               urgent=True)
        return
    _ptt_active["on"] = True
    recorder.start()
    _chime_start()
    # PTT is brief by design — skip the start toast (would be noise), keep the chime + indicator
    _set_phase("recording", "ptt")


def _on_ptt_release(_e):
    if not _ptt_active["on"]:
        return
    _ptt_active["on"] = False
    audio = recorder.stop()
    threading.Thread(target=_transcribe_and_insert, args=(audio,), daemon=True).start()


def _register_hotkeys():
    keyboard.add_hotkey(TOGGLE_HOTKEY, _on_toggle, suppress=True)
    # right-alt as PTT: press + release events
    keyboard.on_press_key("right alt", _on_ptt_press, suppress=False)
    keyboard.on_release_key("right alt", _on_ptt_release, suppress=False)
    log.info("Hotkeys registered: toggle=%s  ptt=%s", TOGGLE_HOTKEY, PTT_HOTKEY)


def _on_quit(icon, _item):
    log.info("Quit requested from tray menu")
    icon.stop()
    os._exit(0)


def _start_tray():
    global _tray_icon
    menu = pystray.Menu(
        pystray.MenuItem("Quit", _on_quit),
    )
    _tray_icon = pystray.Icon(
        "whisper_stt",
        icon=_make_icon_image(_COLOR_IDLE),
        title="Whisper STT: idle",
        menu=menu,
    )
    # Once the tray is up, surface a brief ready toast (non-urgent, no chime)
    def _on_ready():
        _set_phase("idle", "ready")
        _toast("Whisper STT: ready",
               f"Toggle: {TOGGLE_HOTKEY}   |   Push-to-talk: {PTT_HOTKEY}",
               urgent=False)
    threading.Timer(0.5, _on_ready).start()
    _tray_icon.run()  # blocks; runs the Windows message loop


def main():
    _register_hotkeys()
    log.info("Ready. Toggle=%s, PTT=%s. Corpus=%s", TOGGLE_HOTKEY, PTT_HOTKEY, CORPUS_DIR)
    _start_tray()


if __name__ == "__main__":
    main()
