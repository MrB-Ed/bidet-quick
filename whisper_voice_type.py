"""
bidet_quick / whisper_voice_type.py
====================================
Local Whisper-based voice-typing for Windows. The free, offline replacement
for Windows H (and for paid dictation services).

Hotkey: Ctrl+Shift+;   (toggle: press once to start, press again to stop)
Backup hotkey: Right Alt   (push-to-talk: hold while speaking)

When transcribed, the text is pasted into the focused field via clipboard + Ctrl+V.
Audio + transcript are also saved to %USERPROFILE%\\whisper_corpus\\<timestamp>\\
for future voice-clone fine-tuning. Set BIDET_QUICK_CORPUS=0 in the environment
to disable corpus saving.

Visual feedback:
  - System tray icon with 5 states: gray (idle), red (recording),
    yellow (transcribing), green flash (success), orange flash (error)
  - Native Windows toast notifications on error paths
  - Soft "Asterisk" chime on successful insert
  - Every state transition logged to logs/whisper_<date>.log

Architecture:
  - faster-whisper large-v3 on CUDA float16 (NVIDIA GPU, ~3.9 GB VRAM, ~9x real-time)
    Falls back to CPU int8 if CUDA isn't available.
  - sounddevice 16 kHz mono PCM capture (Whisper-native rate)
  - keyboard module for global hotkey
  - pyperclip + keyboard.send('ctrl+v') for insertion (works in every Windows app)
  - pystray for system tray indicator
  - winotify (preferred) / win10toast (fallback) for native Windows toasts

Bidet Quick — built by Mark Barnett (@MrB-Ed). MIT license.
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

# Honor an opt-out env var. Set BIDET_QUICK_CORPUS=0 to disable corpus saving.
SAVE_CORPUS = os.environ.get("BIDET_QUICK_CORPUS", "1") != "0"

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
# Auto-detect GPU; fall back to CPU int8 if CUDA / a GPU isn't available.
_DEVICE_ENV = os.environ.get("WHISPER_DEVICE", "auto")
if _DEVICE_ENV == "auto":
    try:
        # ctranslate2 ships with faster-whisper; ask it whether CUDA works.
        import ctranslate2 as _ct2  # noqa: E402
        DEVICE = "cuda" if _ct2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        DEVICE = "cpu"
else:
    DEVICE = _DEVICE_ENV
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE",
                              "float16" if DEVICE == "cuda" else "int8")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
SAMPLE_RATE = 16000      # Whisper native rate
CHANNELS = 1

TOGGLE_HOTKEY = os.environ.get("WHISPER_TOGGLE_HOTKEY", "ctrl+shift+;")
PTT_HOTKEY = os.environ.get("WHISPER_PTT_HOTKEY", "right alt")

# Minimum audio duration before we accept it as a real recording
MIN_AUDIO_SECONDS = 0.5
# RMS threshold below which we treat the recording as silence
SILENCE_RMS_THRESHOLD = 0.003

# Chime on successful insert (set False to silence)
CHIME_ON_SUCCESS = os.environ.get("WHISPER_CHIME", "1") != "0"

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
log = logging.getLogger("bidet_quick")

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
                # winotify needs an app_id; "Bidet Quick" shows as the source
                n = _winotify_Notification(
                    app_id="Bidet Quick",
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
    if not CHIME_ON_SUCCESS:
        return
    try:
        import winsound
        # SND_ASYNC = non-blocking, SND_ALIAS = play the named system event
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception as e:
        log.warning("Chime failed: %s", e)


# ---------------- Whisper model (loaded once, kept in VRAM) ----------------
log.info("Loading faster-whisper model=%s device=%s compute_type=%s ...",
         MODEL_NAME, DEVICE, COMPUTE_TYPE)
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
    log.error("Model load FAILED on device=%s compute_type=%s: %s", DEVICE, COMPUTE_TYPE, e)
    log.error(traceback.format_exc())
    # If CUDA failed, try CPU int8 as a last resort so the user still gets a working tool.
    if DEVICE == "cuda":
        log.warning("Retrying model load on CPU int8 as fallback...")
        try:
            model = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )
            DEVICE = "cpu"
            COMPUTE_TYPE = "int8"
            _model_ready = True
            log.info("Model ready on CPU int8 fallback in %.1fs", time.time() - _load_t0)
            _toast("Bidet Quick: running on CPU",
                   "CUDA not available; using CPU int8 (slower but works).",
                   urgent=False)
        except Exception as e2:
            log.error("CPU fallback ALSO failed: %s", e2)
            model = None
            _toast("Bidet Quick: model load failed", str(e2)[:200], urgent=True)
    else:
        model = None
        _toast("Bidet Quick: model load failed", str(e)[:200], urgent=True)

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
_state = {"phase": "idle", "msg": "idle"}
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
        "idle":         f"Bidet Quick: idle ({_state['msg']})",
        "recording":    f"Bidet Quick: RECORDING ({_state['msg']})",
        "transcribing": f"Bidet Quick: transcribing... ({_state['msg']})",
        "success":      f"Bidet Quick: done ({_state['msg']})",
        "error":        f"Bidet Quick: error ({_state['msg']})",
    }
    try:
        _tray_icon.icon = _make_icon_image(color)
        _tray_icon.title = titles.get(phase, titles["idle"])
    except Exception as e:
        log.warning("Tray update failed: %s", e)
    log.info("STATE %s :: %s", phase, _state["msg"])

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
        _toast("Bidet Quick: model not loaded",
               "The Whisper model failed to load. Check logs/whisper_<date>.log.",
               urgent=True)
        return

    # Guard: nothing recorded
    if audio is None or len(audio) == 0:
        log.info("No audio frames captured")
        _set_phase("error", "no audio")
        _toast("Bidet Quick: no audio detected",
               "Recording produced no audio. Check default mic in Windows Sound settings.",
               urgent=True)
        return

    audio_sec = len(audio) / SAMPLE_RATE

    # Guard: too short
    if audio_sec < MIN_AUDIO_SECONDS:
        log.info("Audio too short: %.2fs < %.2fs", audio_sec, MIN_AUDIO_SECONDS)
        _set_phase("error", f"too short ({audio_sec:.2f}s)")
        _toast("Bidet Quick: no audio detected",
               f"Recording was only {audio_sec:.2f}s. Hold the hotkey while you speak.",
               urgent=True)
        return

    # Guard: silent (mic muted / dead)
    if _audio_is_silent(audio):
        log.info("Audio is silent (RMS below threshold)")
        _set_phase("error", "silent")
        _toast("Bidet Quick: no audio detected",
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
        _toast("Bidet Quick: transcription failed",
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
        _toast("Bidet Quick: transcription empty",
               "Whisper returned an empty result. Try speaking louder or closer to the mic.",
               urgent=True)
        return

    if SAVE_CORPUS:
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
            _toast("Bidet Quick: model not loaded",
                   "Hotkey pressed before the model finished loading.",
                   urgent=True)
            return
        if not recorder.recording:
            recorder.start()
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
        _toast("Bidet Quick: model not loaded",
               "Hotkey pressed before the model finished loading.",
               urgent=True)
        return
    _ptt_active["on"] = True
    recorder.start()
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
        "bidet_quick",
        icon=_make_icon_image(_COLOR_IDLE),
        title="Bidet Quick: idle",
        menu=menu,
    )
    # Once the tray is up, surface a brief ready toast (non-urgent, no chime)
    def _on_ready():
        _set_phase("idle", "ready")
        _toast("Bidet Quick: ready",
               f"Toggle: {TOGGLE_HOTKEY}   |   Push-to-talk: {PTT_HOTKEY}",
               urgent=False)
    threading.Timer(0.5, _on_ready).start()
    _tray_icon.run()  # blocks; runs the Windows message loop


def main():
    _register_hotkeys()
    log.info("Ready. Toggle=%s, PTT=%s. Corpus=%s (save=%s)",
             TOGGLE_HOTKEY, PTT_HOTKEY, CORPUS_DIR, SAVE_CORPUS)
    _start_tray()


if __name__ == "__main__":
    main()
