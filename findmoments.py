"""Find the best moments in a VOD with publikclip, then export them RAW.

Runs publikclip's analysis (ingest -> transcribe -> speakers -> listen ->
scan -> judge) and stops there: no camera direction, no vertical crop, no
captions, no zooms. Everything is cut straight from the original horizontal
VOD, plus moments.md / moments.json with timestamps, scores and summaries.

What you get in <vod name>_moments next to the VOD:
  NN_scoreS_h-mm-ss.mp4   the scored picks (publikclip's own ranking)
  extras/                 only with --extras: mic claps and "quiet beats"
                          (silence right after something loud) + .jpg sheets
  marked/                 moments YOU flag with --mark (always cut)
  candidates/             with --all: the other ~20 candidates that just
                          missed the top 12

Usage (or drag a VOD onto FindMoments.bat):
  python.exe findmoments.py "C:\\path\\vod.mp4"                    # Ollama
  python.exe findmoments.py "C:\\path\\vod.mp4" --llm gemini
  python.exe findmoments.py --resume <job_id>                     # continue a stopped run
  python.exe findmoments.py "C:\\path\\vod.mp4" --mark 1:23:45 --mark 2:10:02-2:10:40
  python.exe findmoments.py "C:\\path\\vod.mp4" --marks-only --mark 1:23:45
Options:
  --mark T            cut a moment you know about. T = H:MM:SS (or MM:SS, or seconds);
                      T1-T2 for an exact range. Repeat the flag for more.
  --marks-only        skip all analysis; just cut the --mark moments (instant)
  --mark-before 12 --mark-after 18   seconds around a single-point mark
  --all               also export the near-miss candidates
  --extras            also run the clap / quiet-beat scan (off by default)
  --clap-sens 1.0     >1 finds more claps (and more false alarms), <1 fewer
  --length 90         clip length in seconds around each moment (default 90 = 1.5 min)
  --pad 3             minimum seconds kept before/after each scored moment
  --out "C:\\folder"   output folder        --fast  stream-copy (keyframe-snapped)

Must run with publikclip's own Python:
  %USERPROFILE%\\.publikclip\\env\\Scripts\\python.exe
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from publikclip_pipeline import cli, config
    from publikclip_pipeline.jobs import queue
except ImportError:
    sys.exit("Run this with publikclip's Python: %USERPROFILE%\\.publikclip\\env\\Scripts\\python.exe")

import numpy as np

ANALYSIS_ONLY = {"ingest", "asr", "diarize", "events", "candidates", "score"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def tag(t: float) -> str:
    return hms(t).replace(":", "-")


def parse_time(s: str) -> float:
    t = 0.0
    for part in s.strip().split(":"):
        t = t * 60 + float(part)
    return t


def find_ffmpeg() -> str:
    from publikclip_pipeline.render import ffmpeg_bin
    return ffmpeg_bin.ffmpeg()


def cut(ff: str, media: Path, start: float, end: float, dest: Path, fast: bool) -> None:
    start = max(0.0, start)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-i", str(media), "-t", f"{end - start:.3f}"]
    cmd += (["-c", "copy", "-avoid_negative_ts", "make_zero"] if fast else
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
    subprocess.run(cmd + [str(dest)], check=True)


def contact_sheet(ff: str, media: Path, start: float, end: float, dest: Path) -> None:
    """8 frames across the window in one JPG, so a silent visual gag can be
    spotted without watching the clip. Never fatal."""
    start = max(0.0, start)
    dur = max(1.0, end - start)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(media),
           "-t", f"{dur:.3f}", "-vf", f"fps={8 / dur:.5f},scale=360:-2,tile=4x2",
           "-frames:v", "1", "-q:v", "4", str(dest)]
    if subprocess.run(cmd).returncode != 0:
        print(f"  (contact sheet failed for {dest.name}, clip is still there)", flush=True)


# --------------------------------------------------------------------------
# audio scan: claps and quiet beats (things publikclip's ranking can't see)
# --------------------------------------------------------------------------
def frame_stats(x, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS and high-frequency energy per 10 ms frame (block by block, so a
    3-hour file never has to fit in memory as float)."""
    hop = sr // 100
    n = len(x) // hop
    rms = np.empty(n, np.float32)
    hf = np.empty(n, np.float32)
    block = 6000  # frames = 60 s
    scale = 32768.0 if x.dtype.kind == "i" else 1.0
    for i in range(0, n, block):
        j = min(n, i + block)
        f = np.asarray(x[i * hop:j * hop], dtype=np.float32).reshape(-1, hop) / scale
        rms[i:j] = np.sqrt((f ** 2).mean(axis=1))
        hf[i:j] = np.sqrt((np.diff(f, axis=1) ** 2).mean(axis=1))
    return rms, hf


def detect_claps(rms: np.ndarray, hf: np.ndarray, sens: float, top: int) -> list[dict]:
    """Short, sharp, broadband, loud: a clap into the mic. Repeated claps within
    2.5 s are merged into one burst. Game gunshots / door slams can also match,
    which is why every hit comes with a contact sheet."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    if len(rms) < 100:
        return []
    base = uniform_filter1d(rms, size=200, mode="nearest")        # ~2 s average
    prom = rms / (base + 1e-6)
    floor = float(np.percentile(rms, 95)) * 0.5
    peak = (rms == maximum_filter1d(rms, size=9)) & (prom >= 6.0 / max(sens, 0.1)) & (rms >= floor)
    hits = []
    for i in np.nonzero(peak)[0]:
        if i < 6 or i + 20 >= len(rms):
            continue
        rise = rms[i] / (rms[i - 5:i - 2].mean() + 1e-6)          # sharp onset
        decay = rms[i + 15:i + 20].mean() / rms[i]                 # gone within ~150 ms
        ratio = float((hf[i - 1:i + 2] / (rms[i - 1:i + 2] + 1e-6)).max())  # broadband, not a voice
        if rise >= 3 and decay <= 0.5 and ratio >= 0.6:
            hits.append((i / 100.0, float(prom[i])))
    bursts: list[dict] = []
    for t, p in hits:
        if bursts and t - bursts[-1]["t_end"] <= 2.5:
            bursts[-1]["t_end"] = t
            bursts[-1]["count"] += 1
            bursts[-1]["score"] = max(bursts[-1]["score"], p)
        else:
            bursts.append({"kind": "clap", "t": t, "t_end": t, "count": 1, "score": p})
    bursts = sorted(bursts, key=lambda b: b["score"] * (1 + 0.2 * b["count"]), reverse=True)[:top]
    return sorted(bursts, key=lambda b: b["t"])


def detect_quiet_beats(rms: np.ndarray, top: int) -> list[dict]:
    """1.5-15 s of near-silence right after something loud: the comedic beat
    after a crash, a joke landing, a jump scare. Relative to THIS VOD's own
    loudness, not an absolute level."""
    g = 25  # 0.25 s cells
    m = len(rms) // g
    if m < 100:
        return []
    cell = np.sqrt((rms[:m * g].reshape(m, g) ** 2).mean(axis=1))
    lvl = 20 * np.log10(cell + 1e-6)
    loud_thr, quiet_thr = np.percentile(lvl, 85), np.percentile(lvl, 25)
    quiet = lvl < quiet_thr
    found, i = [], 0
    while i < m:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < m and quiet[j]:
            j += 1
        n = j - i
        if 6 <= n <= 60 and i >= 16:
            before = float(lvl[i - 16:i].max())
            if before >= loud_thr:
                score = before - float(lvl[i:j].mean()) + 0.5 * n * 0.25
                found.append({"kind": "quiet-beat", "t": i * 0.25, "t_end": j * 0.25,
                              "count": 1, "score": score})
        i = j
    found.sort(key=lambda b: b["score"], reverse=True)
    kept: list[dict] = []
    for b in found:
        if all(abs(b["t"] - k["t"]) >= 20 for k in kept):
            kept.append(b)
        if len(kept) >= top:
            break
    return sorted(kept, key=lambda b: b["t"])


def scan_audio(audio_path: str, sens: float) -> list[dict]:
    from scipy.io import wavfile
    sr, x = wavfile.read(audio_path, mmap=True)
    if x.ndim > 1:
        x = x[:, 0]
    print("[extras    ] scanning audio for claps and quiet beats…", flush=True)
    rms, hf = frame_stats(x, sr)
    events = detect_claps(rms, hf, sens, top=25) + detect_quiet_beats(rms, top=20)
    return sorted(events, key=lambda e: e["t"])


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def analyse(job: queue.Job) -> dict[str, dict]:
    emit = cli._progress_printer(False)
    ok, err = cli._ensure_pipeline_deps(False, emit)
    if not ok:
        sys.exit(f"publikclip's dependencies aren't installed: {err}")
    stages = [s for s in cli._stages() if s.name in ANALYSIS_ONLY]
    jdir = config.jobs_dir() / job.id
    if not jdir.exists():  # session folder was deleted (app cleanup?): checkpoints are gone, redo them
        print(f"note: session folder was missing, recreating it. Saved progress is gone, so every "
              f"stage will run again.", flush=True)
        jdir.mkdir(parents=True, exist_ok=True)
    return queue.run_stages(job, stages, emit)


def ensure_ollama(timeout: float = 45.0) -> None:
    """Make sure the Ollama server is up before the (slow) scoring stage needs it.
    If it isn't running, start `ollama serve` in the background and wait for it."""
    import os, shutil, time, urllib.request
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    url = host.rstrip("/") + "/api/tags"

    def up() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    if up():
        print("[ollama    ] running", flush=True)
        return
    exe = shutil.which("ollama")
    if not exe:
        guess = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(guess) if guess.exists() else None
    if not exe:
        sys.exit("Ollama isn't running and ollama.exe wasn't found. Open the Ollama app, then run this again.")
    print("[ollama    ] not running, starting it…", flush=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    end = time.time() + timeout
    while time.time() < end:
        if up():
            print("[ollama    ] started", flush=True)
            return
        time.sleep(1)
    sys.exit("Started Ollama but it didn't answer within 45s. Open the Ollama app, then run this again.")


def overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    return max(0.0, inter) / max(1e-6, a1 - a0)


class Report:
    def __init__(self, media: Path, out: Path, header: str = "") -> None:
        self.media, self.out, self.header = media, out, header
        self.sections: dict[str, list[dict]] = {}
        out.mkdir(parents=True, exist_ok=True)

    def add(self, section: str, entry: dict, md: str) -> None:
        entry["_md"] = md
        self.sections.setdefault(section, []).append(entry)

    def write(self) -> None:
        titles = {"scored": "Scored moments (publikclip's ranking)",
                  "extras": "Extras: claps and quiet beats (not ranked, review the .jpg sheets)",
                  "marked": "Marked by you", "candidates": "Near-miss candidates (not in the top 12)"}
        lines = [f"# Raw moments: {self.media.name}\n", self.header]
        manifest = {}
        for key, items in self.sections.items():
            lines.append(f"\n# {titles.get(key, key)}\n")
            manifest[key] = []
            for e in items:
                lines.append(e["_md"])
                manifest[key].append({k: v for k, v in e.items() if k != "_md"})
        (self.out / "moments.json").write_text(
            json.dumps({"source": str(self.media), **manifest}, indent=2), encoding="utf-8")
        (self.out / "moments.md").write_text("\n".join(lines), encoding="utf-8")
        total = sum(len(v) for v in self.sections.values())
        print(f"\nDone: {total} raw clips + moments.md in {self.out}", flush=True)


def widen(c_start: float, c_end: float, length: float, pad: float, total: float | None) -> tuple[float, float]:
    """Grow the window around the moment to `length` seconds (at least `pad` each side),
    centred on the moment and kept inside the VOD."""
    extra = max(2 * pad, length - (c_end - c_start))
    start, end = c_start - extra / 2, c_end + extra / 2
    if start < 0:
        end, start = end - start, 0.0
    if total and end > total:
        start, end = max(0.0, start - (end - total)), total
    return start, end


def export_scored(rep: Report, ff: str, clips: list[dict], pad: float, fast: bool,
                  length: float = 90.0, total: float | None = None) -> None:
    for i, c in enumerate(clips, 1):
        start, end = widen(c["start"], c["end"], length, pad, total)
        score = round(c.get("score", 0))
        name = f"{i:02d}_score{score}_{tag(c['start'])}.mp4"
        into = c["start"] - start
        print(f"[export    ] {i}/{len(clips)} {name}  ({hms(start)} - {hms(end)}, {end - start:.0f}s)", flush=True)
        cut(ff, rep.media, start, end, rep.out / name, fast)
        summary = c.get("summary", "")
        rep.add("scored", {"file": name, "rank": i, "score": score,
                           "source_start": round(c["start"], 2), "source_end": round(c["end"], 2),
                           "clip_starts_at_source": round(start, 2), "clip_ends_at_source": round(end, 2),
                           "moment_starts_in_clip": round(into, 2),
                           "best_platform": c.get("best_platform"), "summary": summary,
                           "confidence": c.get("confidence")},
                f"## {i:02d}. score {score} · moment {hms(c['start'])}–{hms(c['end'])} in the VOD\n"
                f"File: `{name}` · {end - start:.0f}s long · the moment starts {int(into) // 60}:{int(into) % 60:02d} into the clip\n\n{summary}\n")


def export_candidates(rep: Report, ff: str, cands: list[dict], clips: list[dict], fast: bool) -> None:
    sub = rep.out / "candidates"
    sub.mkdir(exist_ok=True)
    rest = [c for c in cands if all(overlap_frac(c["start"], c["end"], k["start"], k["end"]) < 0.5 for k in clips)]
    rest.sort(key=lambda c: c.get("curve_score", 0), reverse=True)
    for i, c in enumerate(rest, 1):
        name = f"C{i:02d}_curve{c.get('curve_score', 0):.2f}_{tag(c['start'])}.mp4"
        print(f"[candidates] {i}/{len(rest)} {name}", flush=True)
        cut(ff, rep.media, c["start"], c["end"], sub / name, fast)
        rep.add("candidates", {"file": f"candidates/{name}", "source_start": round(c["start"], 2),
                               "source_end": round(c["end"], 2), "curve_score": c.get("curve_score")},
                f"- `candidates/{name}` · {hms(c['start'])}–{hms(c['end'])} · interest-curve score "
                f"{c.get('curve_score', 0):.2f} (no judge score)")


def export_extras(rep: Report, ff: str, events: list[dict], clips: list[dict], fast: bool) -> None:
    sub = rep.out / "extras"
    sub.mkdir(exist_ok=True)
    for i, e in enumerate(events, 1):
        start, end = max(0.0, e["t"] - 8.0), e["t_end"] + 8.0
        name = f"X{i:02d}_{e['kind']}_{tag(e['t'])}"
        print(f"[extras    ] {i}/{len(events)} {name}", flush=True)
        cut(ff, rep.media, start, end, sub / f"{name}.mp4", fast)
        contact_sheet(ff, rep.media, start, end, sub / f"{name}.jpg")
        dup = next((j + 1 for j, k in enumerate(clips)
                    if overlap_frac(start, end, k["start"], k["end"]) > 0.5), None)
        what = (f"{e['count']} clap(s) into the mic" if e["kind"] == "clap"
                else f"{e['t_end'] - e['t']:.1f}s of quiet right after something loud")
        rep.add("extras", {"file": f"extras/{name}.mp4", "sheet": f"extras/{name}.jpg", "kind": e["kind"],
                           "at": round(e["t"], 2), "until": round(e["t_end"], 2), "strength": round(e["score"], 1),
                           "already_in_scored": dup},
                f"- `extras/{name}.mp4` · **{e['kind']}** at {hms(e['t'])} · {what} · strength {e['score']:.1f}"
                + (f" · overlaps scored #{dup}" if dup else ""))


def export_marks(rep: Report, ff: str, marks: list[str], before: float, after: float,
                 pad: float, fast: bool) -> None:
    sub = rep.out / "marked"
    sub.mkdir(exist_ok=True)
    for i, m in enumerate(marks, 1):
        if "-" in m:
            a, b = m.split("-", 1)
            t0, t1 = parse_time(a), parse_time(b)
            start, end, how = t0 - pad, t1 + pad, f"{hms(t0)}–{hms(t1)} (+{pad:g}s each side)"
        else:
            t0 = parse_time(m)
            start, end, how = t0 - before, t0 + after, f"{hms(t0)} (-{before:g}s / +{after:g}s)"
        name = f"M{i:02d}_{tag(t0)}"
        print(f"[marked    ] {i}/{len(marks)} {name}  {how}", flush=True)
        cut(ff, rep.media, start, end, sub / f"{name}.mp4", fast)
        contact_sheet(ff, rep.media, start, end, sub / f"{name}.jpg")
        rep.add("marked", {"file": f"marked/{name}.mp4", "sheet": f"marked/{name}.jpg", "mark": m,
                           "clip_starts_at_source": round(max(0.0, start), 2), "clip_ends_at_source": round(end, 2)},
                f"- `marked/{name}.mp4` · you flagged {how}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="VOD file to analyse")
    ap.add_argument("--resume", metavar="JOB_ID", help="continue an earlier run instead of starting one")
    ap.add_argument("--llm", choices=["ollama", "gemini", "publik"], default="ollama")
    ap.add_argument("--pad", type=float, default=3.0, help="minimum seconds kept before/after each moment (default 3)")
    ap.add_argument("--length", type=float, default=90.0, help="target clip length in seconds, centred on the moment (default 90)")
    ap.add_argument("--out", help="output folder (default: <vod name>_moments next to the VOD)")
    ap.add_argument("--fast", action="store_true", help="stream-copy: instant, but cuts snap to keyframes")
    ap.add_argument("--mark", action="append", default=[], metavar="T", help="always cut this moment (repeatable)")
    ap.add_argument("--marks-only", action="store_true", help="skip analysis, only cut the --mark moments")
    ap.add_argument("--mark-before", type=float, default=12.0)
    ap.add_argument("--mark-after", type=float, default=18.0)
    ap.add_argument("--all", action="store_true", help="also export the near-miss candidates")
    ap.add_argument("--extras", action="store_true", help="also run the clap / quiet-beat scan")
    ap.add_argument("--clap-sens", type=float, default=1.0)
    args = ap.parse_args()

    for m in args.mark:  # fail now, not after hours of transcription
        try:
            [parse_time(p) for p in m.split("-", 1)]
        except ValueError:
            sys.exit(f"Can't read --mark {m!r}. Use H:MM:SS, MM:SS, seconds, or a range like 1:23:45-1:24:30")

    if args.marks_only:
        if not args.source or not args.mark:
            ap.error("--marks-only needs a VOD file and at least one --mark")
        media = Path(args.source.strip('"')).expanduser().resolve()
        if not media.exists():
            sys.exit(f"File not found: {media}")
        out = Path(args.out) if args.out else media.parent / f"{media.stem}_moments"
        rep = Report(media, out)
        export_marks(rep, find_ffmpeg(), args.mark, args.mark_before, args.mark_after, args.pad, args.fast)
        rep.write()
        return

    if args.resume:
        job = queue.get_job(args.resume)
        if job is None:
            sys.exit(f"No session {args.resume}")
        settings = json.loads(job.settings_json)
        if settings.get("llm_mode") != args.llm:  # same mechanism as `publikclip resume --llm`
            settings["llm_mode"] = args.llm
            with queue._connect() as conn:  # noqa: SLF001
                conn.execute("UPDATE jobs SET settings_json = ? WHERE id = ?", (json.dumps(settings), job.id))
            job = queue.get_job(job.id)
    elif args.source:
        src = Path(args.source.strip('"')).expanduser().resolve()
        if not src.exists():
            sys.exit(f"File not found: {src}")
        settings = config.Settings()
        settings.llm_mode = args.llm
        job = queue.create_job("file", str(src), json.dumps(settings.to_json()))
    else:
        ap.error("give a VOD file, or --resume JOB_ID")

    print(f"session {job.id}  (brain: {args.llm})", flush=True)
    if args.llm == "ollama":
        ensure_ollama()
    try:
        results = analyse(job)
    except Exception as err:  # noqa: BLE001
        sys.exit(f"\nStopped: {err}\nFix the issue, then continue with:  findmoments.py --resume {job.id}")

    media = Path(results["ingest"]["media_path"])
    out = Path(args.out) if args.out else media.parent / f"{media.stem}_moments"
    clips = results["score"]["clips"]
    rep = Report(media, out, f"Scored by: {results['score'].get('llm_mode')} ({results['score'].get('model')})\n")
    ff = find_ffmpeg()

    total = (results["ingest"].get("probe") or {}).get("duration_sec")
    export_scored(rep, ff, clips, args.pad, args.fast, args.length, total)
    if args.mark:
        export_marks(rep, ff, args.mark, args.mark_before, args.mark_after, args.pad, args.fast)
    if args.extras:
        try:
            events = scan_audio(results["ingest"]["audio_path"], args.clap_sens)
            export_extras(rep, ff, events, clips, args.fast)
        except Exception as err:  # noqa: BLE001 — never lose the scored clips over the bonus scan
            print(f"[extras    ] skipped: {err}", flush=True)
    if args.all:
        export_candidates(rep, ff, results["candidates"]["candidates"], clips, args.fast)
    rep.write()


if __name__ == "__main__":
    main()
