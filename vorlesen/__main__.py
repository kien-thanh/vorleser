"""vorlesen – turn a German EPUB into an audiobook (.m4b) with a local TTS model.

  python -m vorlesen list    BOOK.epub
  python -m vorlesen preview BOOK.epub --chapter 8
  python -m vorlesen sample  BOOK.epub --engines chatterbox,qwen --chapter 8
  python -m vorlesen render  BOOK.epub --engine chatterbox
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import audio
from .chunker import Segment, chapter_segments
from .engines import ENGINES, ROOT, Engine
from .epub import Book, read_epub

FRONT_MATTER = re.compile(r"^(cover|titel|title|impressum|copyright|inhalt|contents|das buch|der autor|die autorin"
                          r"|über (den|die) autor|widmung|danksagung|leseprobe)\b", re.I)


def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def _slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.lower()).strip("-")[:60]


def parse_selection(spec: str | None, book: Book) -> list[int]:
    """'7-12,15' (1-based, as shown by `list`) or default: everything but front matter."""
    if not spec:
        return [i for i, c in enumerate(book.chapters) if c.chars >= 100 and not FRONT_MATTER.match(c.title)]
    picked = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        picked += range(int(a) - 1, int(b or a))
    return [i for i in picked if 0 <= i < len(book.chapters)]


def engine_options(args) -> dict:
    opts = {}
    if getattr(args, "voice", None):
        voice = Path(args.voice).expanduser().resolve()
        opts["voice"] = str(voice)
        transcript = voice.with_suffix(".txt")
        if not getattr(args, "ref_text", None) and transcript.exists():
            opts["ref_text"] = transcript.read_text(encoding="utf-8").strip()
    if getattr(args, "ref_text", None):
        opts["ref_text"] = args.ref_text
    for kv in getattr(args, "opt", None) or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        opts[k] = v
    return opts


def chapter_title(book: Book, idx: int) -> str:
    """'2' -> 'Erstes Bild • Der Tempel allen Wissens – 2' so chapter lists are readable."""
    title = book.chapters[idx].title
    if re.fullmatch(r"\d+", title):
        for prev in reversed(book.chapters[:idx]):
            parent = prev.title.split(" – ")[0]
            if not re.fullmatch(r"\d+", parent):
                return f"{parent} – {title}"
    return title


CHARS_PER_SECOND = 16.0  # typical German narration speed of both models
MAX_ATTEMPTS = 3


def _duration_error(text: str, seconds: float) -> float:
    """How far off the plausible speaking time a segment is (0 = fine). Catches truncated
    output (too fast) and babbling / long hallucinated pauses (too slow)."""
    expected = 0.3 + len(text) / CHARS_PER_SECOND
    lo, hi = expected * 0.55, expected * 1.8 + 0.5
    if seconds < lo:
        return lo / max(seconds, 1e-3) - 1
    if seconds > hi:
        return seconds / hi - 1
    return 0.0


def synthesize_checked(engine: Engine, text: str, path: Path, qa_log: Path | None, first_done: bool = False):
    """first_done: attempt 0 already exists at .try0.wav (rendered as part of a batch)."""
    best = None  # (error, attempt path)
    for attempt in range(MAX_ATTEMPTS):
        tmp = path.with_suffix(f".try{attempt}.wav")
        if not (attempt == 0 and first_done):
            engine.synthesize(text, tmp, seed=None if attempt == 0 else 1000 + attempt)
        err = _duration_error(text, audio.speech_seconds(tmp))
        if best is None or err < best[0]:
            if best:
                best[1].unlink()
            best = (err, tmp)
        else:
            tmp.unlink()
        if err == 0:
            break
    err, tmp = best
    tmp.rename(path)
    if qa_log and (attempt > 0 or err > 0):
        with qa_log.open("a", encoding="utf-8") as f:
            status = "FIXED" if err == 0 else f"SUSPECT (off by {err:.0%})"
            f.write(f"{status} after {attempt + 1} tries  {path.name}  {text}\n")


def render_chapter(engine: Engine, segments: list[Segment], seg_dir: Path, progress,
                   qa_log: Path | None = None) -> list[tuple[Path, float]]:
    seg_dir.mkdir(parents=True, exist_ok=True)
    parts, todo = [], []
    for i, seg in enumerate(segments):
        path = seg_dir / f"{i:04d}-{_hash(seg.text)}.wav"
        if path.exists():
            progress(len(seg.text), fresh=False)
        else:
            todo.append((seg.text, path))
        parts.append((path, seg.pause_after))
    # Similar lengths per batch: a batch runs as long as its longest segment.
    todo.sort(key=lambda t: len(t[0]))
    for k in range(0, len(todo), engine.batch_size):
        group = todo[k:k + engine.batch_size]
        if len(group) > 1:
            engine.synthesize_batch([(text, path.with_suffix(".try0.wav")) for text, path in group])
        for text, path in group:  # QA + single-segment retries
            synthesize_checked(engine, text, path, qa_log, first_done=len(group) > 1)
            progress(len(text), fresh=True)
    return parts


class Progress:
    def __init__(self, total_chars: int):
        self.total, self.done, self.fresh_chars, self.t0 = total_chars, 0, 0, time.time()

    def __call__(self, n: int, fresh: bool):
        self.done += n
        if fresh:
            self.fresh_chars += n
        rate = self.fresh_chars / max(time.time() - self.t0, 1e-6)
        eta = (self.total - self.done) / rate if self.fresh_chars else 0
        print(f"\r  {self.done / self.total:6.1%}  {rate:5.0f} chars/s  ETA {eta / 3600:4.1f} h ",
              end="", file=sys.stderr, flush=True)


def cmd_list(args):
    book = read_epub(args.book)
    selected = set(parse_selection(None, book))
    print(f"{book.title} – {book.author} ({book.language})\n")
    for i, c in enumerate(book.chapters):
        mark = "*" if i in selected else " "
        print(f"{mark}{i + 1:3d}  {c.chars:7d} chars  {chapter_title(book, i)}")
    print("\n* = included by default")


def cmd_preview(args):
    book = read_epub(args.book)
    max_chars = ENGINES[args.engine].max_chars
    for idx in parse_selection(args.chapter, book):
        print(f"### {chapter_title(book, idx)}")
        for s in chapter_segments(book.chapters[idx], max_chars):
            print(f"[{s.pause_after:.2f}s] {s.text}")


def _sample_segments(book: Book, idx: int, max_chars: int, chars: int) -> list[Segment]:
    out, total = [], 0
    for s in chapter_segments(book.chapters[idx], max_chars):
        out.append(s)
        total += len(s.text)
        if total >= chars:
            break
    return out


def cmd_sample(args):
    book = read_epub(args.book)
    idx = parse_selection(args.chapter, book)[0]
    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)
    for name in args.engines.split(","):
        engine = Engine(name, engine_options(args), args.batch_size)
        segments = _sample_segments(book, idx, engine.spec.max_chars, args.chars)
        work = ROOT / "work" / _slug(book.title) / f"{name}-{_hash(engine.options)}" / f"ch{idx:03d}"
        print(f"{name}: {len(segments)} segments, options {engine.options}", file=sys.stderr)
        t0 = time.time()
        with engine:
            load_s = time.time() - t0
            t1 = time.time()
            parts = render_chapter(engine, segments, work, Progress(sum(len(s.text) for s in segments)),
                                   qa_log=out_dir / f"{_slug(book.title)}-{name}.qa.log")
            synth_s = time.time() - t1
        wav = out_dir / f"{_slug(book.title)}-{name}.wav"
        dur = audio.join_segments(parts, wav)
        audio.to_mp3(wav, wav.with_suffix(".mp3"))
        wav.unlink()
        print(f"\n  -> {wav.with_suffix('.mp3')}  ({dur:.0f}s audio, model load {load_s:.0f}s, "
              f"synthesis {synth_s:.0f}s, {dur / max(synth_s, 1e-6):.2f}x realtime)", file=sys.stderr)


DESIGN_TEXT = ("Es war einmal ein kleines Dorf am Rande eines grossen Waldes. Die Menschen dort lebten ruhig "
               "und bescheiden, bis eines Tages ein Fremder kam, der eine seltsame Geschichte zu erzählen hatte.")


def cmd_design_voice(args):
    """Create a reference voice from a description (Qwen VoiceDesign) for cloning with any engine."""
    out_dir = ROOT / "voices"
    out_dir.mkdir(exist_ok=True)
    wav = out_dir / f"{args.name}.wav"
    opts = {"design": args.description}
    if args.seed is not None:
        opts["seed"] = args.seed
    with Engine("qwen", opts) as engine:
        engine.synthesize(args.text, wav)
    wav.with_suffix(".txt").write_text(args.text + "\n", encoding="utf-8")
    print(f"-> {wav}  (transcript in {wav.with_suffix('.txt').name}; use with --voice {wav})", file=sys.stderr)


def cmd_render(args):
    book = read_epub(args.book)
    engine = Engine(args.engine, engine_options(args), args.batch_size)
    chapters = parse_selection(args.chapters, book)
    work = ROOT / "work" / _slug(book.title) / f"{args.engine}-{_hash(engine.options)}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "options.json").write_text(json.dumps(engine.options, indent=2, ensure_ascii=False))

    plan = [(idx, chapter_segments(book.chapters[idx], engine.spec.max_chars)) for idx in chapters]
    progress = Progress(sum(len(s.text) for _, segs in plan for s in segs))
    print(f"{book.title}: {len(plan)} chapters, {progress.total} chars, engine {args.engine} {engine.options}",
          file=sys.stderr)
    print(f"work dir: {work}  (safe to interrupt; rerun the same command to resume)", file=sys.stderr)

    chapter_wavs = []
    with engine:
        for idx, segments in plan:
            wav = work / f"ch{idx:03d}-{_hash([s.text for s in segments], [s.pause_after for s in segments])}.wav"
            if not wav.exists():
                parts = render_chapter(engine, segments, work / f"ch{idx:03d}", progress, qa_log=work / "qa.log")
                audio.join_segments(parts, wav)
            else:
                progress(sum(len(s.text) for s in segments), fresh=False)
            chapter_wavs.append((chapter_title(book, idx), wav))
    print(file=sys.stderr)

    out = Path(args.out).expanduser() if args.out else Path.home() / f"{book.title} - {book.author} ({args.engine}).m4b"
    cover = None
    if book.cover:
        cover = work / f"cover.{book.cover_ext}"
        cover.write_bytes(book.cover)
    print(f"building {out}", file=sys.stderr)
    audio.build_m4b(chapter_wavs, out, book.title, book.author, cover)
    print(f"done: {out}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vorlesen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def voice_args(sp):
        sp.add_argument("--voice", help="reference audio (wav, 10–20 s, clean speech) to clone")
        sp.add_argument("--ref-text", help="transcript of --voice (improves Qwen cloning)")
        sp.add_argument("--opt", action="append", metavar="KEY=VALUE", help="engine option, repeatable")
        sp.add_argument("--batch-size", type=int, help="segments per model call (default per engine; 1 = off)")

    sp = sub.add_parser("list", help="show chapters")
    sp.add_argument("book")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("preview", help="print the normalized text segments the model will read")
    sp.add_argument("book")
    sp.add_argument("--chapter", required=True, help="chapter number(s) from `list`")
    sp.add_argument("--engine", default="chatterbox", choices=ENGINES)
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("sample", help="render the start of one chapter with one or more engines")
    sp.add_argument("book")
    sp.add_argument("--engines", default=",".join(ENGINES))
    sp.add_argument("--chapter", help="chapter number from `list` (default: first included)")
    sp.add_argument("--chars", type=int, default=1800, help="approximate text length (1800 ≈ 2 min)")
    voice_args(sp)
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("design-voice", help="create a reference voice from a text description (Qwen VoiceDesign)")
    sp.add_argument("name", help="saved as voices/<name>.wav + .txt")
    sp.add_argument("description", help="e.g. 'Ruhiger deutscher Hörbuchsprecher, tiefe warme Männerstimme'")
    sp.add_argument("--text", default=DESIGN_TEXT, help="German sentence(s) the voice speaks (~15 s)")
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_design_voice)

    sp = sub.add_parser("render", help="render the whole book to .m4b (resumable)")
    sp.add_argument("book")
    sp.add_argument("--engine", required=True, choices=ENGINES)
    sp.add_argument("--chapters", help="e.g. 7-12,15 (default: all but front matter)")
    sp.add_argument("--out", help="output .m4b path (default: ~/<title> - <author> (<engine>).m4b)")
    voice_args(sp)
    sp.set_defaults(func=cmd_render)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
