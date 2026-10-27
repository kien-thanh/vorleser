"""Audio assembly: trim segment silence, insert pauses, build the final m4b with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def trim_silence(wav: np.ndarray, sr: int, threshold_db: float = -45.0, keep: float = 0.04) -> np.ndarray:
    if wav.size == 0:
        return wav
    frame = int(sr * 0.01)
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms / (np.max(np.abs(wav)) + 1e-9)) > threshold_db)[0]
    if loud.size == 0:
        return wav[:0]
    pad = int(keep * sr)
    return wav[max(0, loud[0] * frame - pad): min(len(wav), (loud[-1] + 1) * frame + pad)]


def speech_seconds(path: Path) -> float:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return len(trim_silence(wav, sr)) / sr


def join_segments(parts: list[tuple[Path, float]], out: Path):
    """parts: (wav path, pause after in seconds). Writes one mono wav."""
    chunks, sr = [], None
    for path, pause in parts:
        wav, rate = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr is None:
            sr = rate
        elif rate != sr:
            raise ValueError(f"sample rate mismatch in {path}")
        chunks.append(trim_silence(wav, sr))
        chunks.append(np.zeros(int(pause * sr), dtype=np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(out, audio, sr, subtype="PCM_16")
    return len(audio) / sr


def _ffmeta_escape(s: str) -> str:
    for ch in "\\=;#\n":
        s = s.replace(ch, "\\" + ch)
    return s


def build_m4b(chapter_wavs: list[tuple[str, Path]], out: Path, title: str, author: str,
              cover: Path | None, bitrate: str = "64k"):
    work = out.parent
    concat = work / f".{out.stem}.concat.txt"
    meta = work / f".{out.stem}.ffmeta"

    lines, start = [";FFMETADATA1", f"title={_ffmeta_escape(title)}", f"artist={_ffmeta_escape(author)}",
                    f"album={_ffmeta_escape(title)}", "genre=Audiobook"], 0
    for name, wav in chapter_wavs:
        ms = int(sf.info(wav).duration * 1000)
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={start + ms}",
                  f"title={_ffmeta_escape(name)}"]
        start += ms
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    concat.write_text("".join(f"file '{str(w).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                              for _, w in chapter_wavs), encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats",
           "-f", "concat", "-safe", "0", "-i", str(concat), "-i", str(meta)]
    if cover:
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "2:v", "-c:v", "copy",
                "-disposition:v", "attached_pic"]
    else:
        cmd += ["-map", "0:a"]
    cmd += ["-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", bitrate,
            "-movflags", "+faststart", "-f", "mp4", str(out)]
    subprocess.run(cmd, check=True)
    concat.unlink()
    meta.unlink()


def to_mp3(wav: Path, out: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "128k", str(out)], check=True)
