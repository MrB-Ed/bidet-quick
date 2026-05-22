# Bidet Quick

**Local Whisper voice typing for Windows. Press a hotkey, talk, release. Your words get typed into whatever you're focused on. Free, offline, GPU-accelerated.**

Built as a no-cost replacement for Windows Voice Typing (Win+H), Dragon, and paid dictation services. Runs entirely on your machine — your audio never leaves the laptop.

---

## What this is

A small Python app that sits in your system tray. Press a global hotkey from anywhere in Windows — Chrome, Cursor, Notepad, an email reply, a browser address bar — and it records your microphone. Press it again (or release the push-to-talk key) and it transcribes what you said with [OpenAI's Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) running locally on your GPU, then pastes the text into the focused field via the clipboard.

On an RTX 4070 it runs about **9x faster than real time** and produces transcripts indistinguishable from cloud services. On CPU it still works, just slower.

---

## Hardware requirements

- **Windows 10 or 11** (this app uses Windows-specific APIs for hotkeys, tray, and paste)
- **Python 3.10 or newer** ([download](https://www.python.org/downloads/windows/) — check "Add Python to PATH" during install)
- **An NVIDIA GPU is strongly recommended** but not required. The app auto-detects CUDA and falls back to CPU `int8` mode if no GPU is found. Expect roughly:
  - RTX 4070 / 4080 / 4090: ~9x real time, ~3.9 GB VRAM
  - RTX 30-series: similar
  - CPU only (modern i7/Ryzen 7): ~0.5-1x real time (workable for short bursts; sluggish for long dictation)
- **~5 GB free disk space** for the Whisper model and Python packages
- **A microphone** (built-in laptop mic is fine)

---

## Install (3 steps)

```bat
git clone https://github.com/MrB-Ed/bidet-quick.git
cd bidet-quick
install.bat
```

That's it. `install.bat` will:

1. Create a Python virtual environment in `.venv` inside the repo folder
2. Install all dependencies (this is the slow part — about 5-10 minutes on first run, mostly because the bundled CUDA DLL packages are ~700 MB)
3. Copy a Startup-folder shortcut so Bidet Quick launches every time you log in

When it's done it prints:

> Press `Ctrl+Shift+;` in any text field to start dictating.

To launch right now without rebooting, double-click `run_whisper_stt.vbs` (silent background) or `run_whisper_stt.bat` (with a debug console).

**First run downloads the Whisper model (~3 GB) from Hugging Face.** That happens once. After that, everything is offline.

---

## Hotkeys

| Hotkey | What it does |
|---|---|
| `Ctrl + Shift + ;` | **Toggle.** Press once to start recording, press again to stop, transcribe, and paste. |
| `Right Alt` (hold) | **Push-to-talk.** Hold while you speak; release to transcribe and paste. |

Both work in *any* focused text field — Chrome, Cursor, Notepad, Outlook, browser address bars, Discord, you name it.

Want different hotkeys? Set environment variables before launching, or edit the constants at the top of `whisper_voice_type.py`:

```python
TOGGLE_HOTKEY = "ctrl+shift+;"   # or env: WHISPER_TOGGLE_HOTKEY
PTT_HOTKEY    = "right alt"       # or env: WHISPER_PTT_HOTKEY
```

The [`keyboard`](https://github.com/boppreh/keyboard) library uses standard syntax: `"f12"`, `"ctrl+alt+v"`, `"win+space"`, etc.

---

## Tray icon — what the colors mean

Bidet Quick lives in your system tray (the up-arrow overflow area next to the clock). The dot changes color so you always know what state it's in:

| Color | State | What it means |
|---|---|---|
| Gray | Idle | Model loaded, hotkeys armed, waiting for you |
| Red | Recording | Mic is hot. Speak. |
| Yellow | Transcribing | Whisper is working on what you just said |
| Green flash | Success | Text pasted into the focused field |
| Orange flash | Error | Something went wrong — toast notification will explain |

Hover the icon for a status string like "Bidet Quick: transcribing... (12.4s)". Right-click → Quit to exit.

**Pro tip:** drag the gray dot out of the overflow area onto the always-visible portion of the taskbar so you never lose it.

---

## Voice corpus (this is the cool part)

Every successful transcription writes a matched pair to:

```
%USERPROFILE%\whisper_corpus\YYYY-MM-DD-HHMMSS\
    audio.wav        16 kHz mono PCM, the recording
    transcript.txt   what Whisper transcribed
```

This is exactly the format you need to **train a voice clone of yourself** — XTTSv2, Tortoise, Bark, and most modern neural TTS systems eat `(text, audio)` pairs in this shape.

The folder grows naturally as you use the app for everyday typing. After a few weeks of normal use you'll have hours of clean, in-domain training data — way better than a one-time staged recording session, because it's how you actually talk.

A future release will ship a one-click LoRA fine-tune script (Whisper-mark) that uses this corpus to bump accuracy on your specific voice and vocabulary, plus an opt-in path to a personal TTS voice clone.

**Don't want corpus saving?** Set the environment variable `BIDET_QUICK_CORPUS=0` before launch. Or delete the existing corpus any time:

```powershell
Remove-Item -Recurse -Force $env:USERPROFILE\whisper_corpus\*
```

The corpus folder is on `.gitignore` so it never accidentally ends up in a repo.

---

## Troubleshooting

**Nothing happens when I press the hotkey.**
Open Task Manager → Details and look for `pythonw.exe`. If it's not there, double-click `run_whisper_stt.bat` to see the actual error in a console window. Most often it's a missing dependency — re-run `install.bat`.

**"cublas64_12.dll is not found" or other CUDA DLL errors.**
The CUDA DLLs ship as pip packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) and the script puts them on PATH at startup. If you still get this:
```bat
.venv\Scripts\python.exe -m pip install --force-reinstall nvidia-cublas-cu12 nvidia-cudnn-cu12
```
If you don't have an NVIDIA GPU at all, force CPU mode:
```bat
set WHISPER_DEVICE=cpu
wscript run_whisper_stt.vbs
```

**No tray icon visible.**
Look in the up-arrow overflow next to the system clock. To pin it permanently: drag-and-drop the gray dot from the overflow onto the always-visible taskbar.

**Wrong microphone is being recorded.**
Right-click the speaker icon in your taskbar → Sound settings → Input. Set the device you want as default. Bidet Quick uses whatever Windows says is the default mic.

**Hotkey conflict with another app.**
Edit `whisper_voice_type.py` near the top:
```python
TOGGLE_HOTKEY = "ctrl+shift+;"
PTT_HOTKEY    = "right alt"
```
Standard [`keyboard` library syntax](https://github.com/boppreh/keyboard#keyboard.parse_hotkey_combinations) applies.

**Mic permission prompt on first use.**
Windows will pop a "Allow X to access your microphone?" prompt the very first time. Click Allow. After that it's permanent.

**Notification toasts not showing.**
On first use Windows may ask "Allow notifications from Bidet Quick?" — click Allow. Toasts also respect Focus Assist / Do Not Disturb; if those are on, toasts go straight to Action Center.

**It's slow / says "running on CPU".**
Either you don't have an NVIDIA GPU, or the script couldn't load CUDA. Check `logs\whisper_<date>.log` for the actual error. The CPU fallback is real (it works), just much slower — ~0.5x real time on a modern i7 vs ~9x on a 4070.

**I want a smaller / faster model.**
Set `WHISPER_MODEL` before launch. Options: `tiny`, `base`, `small`, `medium`, `large-v3`. `medium` is a good middle ground on CPU.

```bat
set WHISPER_MODEL=medium
wscript run_whisper_stt.vbs
```

---

## Uninstall

Run `uninstall.bat`. It removes the Startup shortcut and (with confirmation) the `.venv` directory. It does *not* delete the model cache, the corpus, or the logs — those stay so you don't lose your training data accidentally. Delete those folders manually if you want a full purge.

---

## What's coming

- **Whisper-mark fine-tuning** — one-click LoRA training script that uses your accumulated corpus to bump accuracy on your specific voice and vocabulary
- **Personal voice clone** — XTTSv2 / Tortoise fine-tune from the same corpus (opt-in)
- **Screenshots** — tray icon states will get pinned to the README once the screenshot run is done

---

## Built by

Mark Barnett ([@MrB-Ed](https://github.com/MrB-Ed)) — 57-year-old middle-school history teacher, self-described Digital Twin Architect. This is part of a larger personal-AI stack at [bidetai.thebarnetts.info](https://bidetai.thebarnetts.info).

Bidet AI is the web version (voice brain-dump → cleaned-up text and analysis). **Bidet Quick** is the dictation half — the part that types for you anywhere in Windows.

---

## License

MIT. See [LICENSE](LICENSE). Use it, fork it, ship it.

---

## Credit where it's due

- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — the CTranslate2-backed Whisper inference engine that makes this fast enough to feel instant
- [OpenAI Whisper](https://github.com/openai/whisper) — the model itself
- [`keyboard`](https://github.com/boppreh/keyboard), [`sounddevice`](https://python-sounddevice.readthedocs.io/), [`pystray`](https://github.com/moses-palmer/pystray), [`winotify`](https://github.com/versa-syahptr/winotify) — the bits that make it feel native
