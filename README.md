# FindMoments

Drag a Twitch/YouTube VOD onto a file, walk away, and get the **top 12 moments as raw, uncropped, unsubtitled clips** (~90 seconds each) plus a timestamp list. Meant for people who want to edit the clips themselves.

It's a small wrapper around [publikclip](https://github.com/Blueturboguy07/publikclip): publikclip does the transcription, speaker detection, and moment scoring, and FindMoments stops before its vertical crop / caption / zoom steps and cuts the original horizontal video instead. **Everything runs locally and free** (Ollama), no API keys needed.

> This repo does **not** include publikclip's code. You install publikclip yourself, and FindMoments uses your copy.

## What you get

For `MyStream.mp4` you get a `MyStream_moments` folder next to it:

```
MyStream_moments/
  01_score92_1-23-45.mp4     <- best moment first; score is publikclip's own rating
  02_score88_0-41-10.mp4
  ...                        <- 12 clips total
  moments.md                 <- readable list: timestamps, scores, why it was picked
  moments.json               <- same data for scripts
```

Clips are original resolution and aspect ratio (16:9 stays 16:9), with no captions and no zoom. Each clip includes extra room before and after the moment so you have footage to trim.

## Setup (Windows, one time, about 20 min)

### Step 1: Install Ollama (the free local AI that judges the moments)

Ollama runs an AI model on your own PC. publikclip asks it to rate each candidate moment.

1. Download the Windows installer from <https://ollama.com/download> and run it. No admin rights needed. It starts in the background (llama icon in the system tray).
2. Open **PowerShell** and check it works: `ollama --version`
3. Pick a model from the table below, then download it (one time, a few GB):
   ```
   ollama pull qwen2.5:7b
   ```
4. Quick test: `ollama run qwen2.5:7b "say hi"` (type `/bye` to exit).

**Which model for my PC?** Find your GPU memory: press `Ctrl+Shift+Esc`, open **Performance**, click your **GPU**, and read **Dedicated GPU memory** (or run `nvidia-smi` for NVIDIA cards).

| Your VRAM | Model to pull | Download size | Notes |
|---|---|---|---|
| No dedicated GPU / under 4 GB | `qwen2.5:3b` | ~2 GB | Runs on CPU. Works, rates moments less reliably. |
| 6 to 8 GB | `qwen2.5:7b` | ~4.7 GB | **Recommended default.** What this was built and tested with. |
| 12 to 16 GB | `qwen2.5:14b` | ~9 GB | Noticeably better judgment on jokes and context. |
| 24 GB or more | `qwen2.5:32b` | ~20 GB | Best local quality, slowest. |

Rule of thumb: the model file should fit in your VRAM with 1 to 2 GB to spare. If it doesn't fit, Ollama splits it onto your CPU and RAM, which still works but gets much slower. Sizes are approximate. If you have no GPU or want zero local load, use `--llm gemini` instead (see Options).

**How FindMoments uses Ollama:** transcription (the longest step) doesn't use Ollama at all. Ollama is only used at the scoring step, where publikclip sends it the transcript around each of ~35 candidate moments and gets back a score; the top 12 are cut. FindMoments checks Ollama is running when it starts and tries to launch it if it isn't. So Ollama just needs to be installed and the model pulled.

### Step 2: Install publikclip

Install publikclip (v0.2.6 tested) by following <https://github.com/Blueturboguy07/publikclip>. Open it once so it finishes setting itself up. You should end up with `C:\Users\<you>\.publikclip\env\Scripts\python.exe`. In publikclip's settings, choose **Ollama** as the LLM and enter the model you pulled (for example `qwen2.5:7b`).

### Step 3: Get FindMoments

Click the green **Code** button, then **Download ZIP**, and unzip. Keep `FindMoments.bat` and `findmoments.py` in the same folder.

## Use

**Drag your VOD file onto `FindMoments.bat`.** That's it. A window shows progress; when it says done, press any key.

Or from PowerShell / cmd:

```
FindMoments.bat "D:\VODs\MyStream.mp4"
FindMoments.bat "D:\VODs\MyStream.mp4" --length 120 --out "D:\clips"
```

### How long does it take?

Transcription is the slow step and runs on CPU: roughly real-time to 1.5x the VOD length on an average PC. Leave it overnight for a 3+ hour VOD. Progress is saved, so a crash or closed window is resumable (see below).

## Options

| Option | What it does |
|---|---|
| `--length 90` | Clip length in seconds (default 90). Use 120 for more room. |
| `--pad 3` | Minimum seconds kept before and after the moment itself. |
| `--out "C:\folder"` | Put the output folder somewhere else. |
| `--llm ollama` | `ollama` (default, free, local), `gemini` (needs a key in publikclip), or `publik`. |
| `--mark 1:23:45` | Also cut a moment you already know about. Use `T1-T2` for an exact range (`--mark 2:10:02-2:10:40`). Repeat the flag for more. |
| `--marks-only` | Skip all analysis and only cut your `--mark` moments (instant). |
| `--mark-before 12` / `--mark-after 18` | Seconds around a single-point mark. |
| `--all` | Also export the ~20 near-miss candidates into `candidates/`. |
| `--extras` | Also scan the audio for mic claps and "quiet beats" (silence right after something loud), into `extras/`. Off by default. |
| `--clap-sens 1.0` | Higher finds more claps and more false alarms. |
| `--fast` | Stream-copy instead of re-encode (instant, but cuts snap to keyframes). |
| `--resume JOB_ID` | Continue a stopped run. The job id is printed at the start of every run. |

## Troubleshooting

**"Stopped: Ollama isn't running."** FindMoments tries to start Ollama for you. If it can't, open the Ollama app yourself, then continue with `FindMoments.bat --resume <job id>`.

**Window closed or PC restarted mid-run.** Run `FindMoments.bat --resume <job id>` (the id looks like `20261015-124348-2b1d73`; it's also in `%USERPROFILE%\.publikclip\jobs`). Finished stages are not redone.

**"Run this with publikclip's Python".** Use `FindMoments.bat`, not plain `python findmoments.py`. The script needs the Python that publikclip installed. Manual form:
```
& "$env:USERPROFILE\.publikclip\env\Scripts\python.exe" findmoments.py "D:\VODs\MyStream.mp4"
```

**Don't delete a job folder under `.publikclip\jobs` while a run is in progress.** It holds that run's saved progress.

**Warnings about pyannote / torchaudio / whisperx.** Normal; ignore them unless the run actually stops.

## Optional: make the publikclip app itself render raw

`optional/RawRender.py` patches publikclip's own renderer so clips rendered **inside the app** are also horizontal with no subtitles. You don't need it for FindMoments. Run it with publikclip's Python; `--undo` restores the original. Re-run it after updating publikclip.

## Notes

- Scores come from publikclip's scorer using your local model (labelled "local-estimate"). Without a cloud key, publikclip skips its vision pass, so scores are text and audio based.
- Tested against publikclip 0.2.6. Later versions may rename internals; open an issue if it breaks.
- publikclip is AGPL-licensed and belongs to its authors. This repo is an independent add-on, MIT licensed, and isn't affiliated with publikclip.
