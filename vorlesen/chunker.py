"""Turn a chapter's paragraphs into TTS-sized segments with pauses between them."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .epub import Chapter
from .normalize_de import ensure_terminal, merge_broken_paragraphs, normalize

PAUSE_SENTENCE = 0.25
PAUSE_PARAGRAPH = 0.6
PAUSE_HEADING = 1.2

# Split after sentence-final punctuation (optionally followed by a closing quote)
# when the next sentence starts with an uppercase letter, digit or opening quote.
_SENT_SPLIT = re.compile(r'(?<=[.!?])"?\s+(?=["A-ZÄÖÜ0-9])')


@dataclass
class Segment:
    text: str
    pause_after: float


def split_sentences(text: str) -> list[str]:
    parts, last = [], 0
    for m in _SENT_SPLIT.finditer(text):
        parts.append(text[last:m.end()].strip())
        last = m.end()
    parts.append(text[last:].strip())
    return [p for p in parts if p]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at spaces."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces, current = [], ""
    for clause in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + 1 + len(clause) > max_chars:
            pieces.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    pieces.append(current)
    out = []
    for p in pieces:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut])
            p = p[cut:].strip()
        out.append(p)
    return out


def chapter_segments(chapter: Chapter, max_chars: int) -> list[Segment]:
    segments: list[Segment] = []
    raw = merge_broken_paragraphs([p.text for p in chapter.paragraphs])
    short_heading_like = {p.text for p in chapter.paragraphs if p.heading}

    for i, para in enumerate(raw):
        is_title = para in short_heading_like or (len(para) < 60 and not re.search(r"[.!?…«\"]$", para))
        text = normalize(para)
        if not text:
            continue
        if is_title:
            # Consecutive title lines ("Erstes Bild." "Der Tempel…" "Eins.") are read as one
            # segment: very short inputs make some models mispronounce or babble.
            if segments and segments[-1].pause_after == PAUSE_HEADING and len(segments[-1].text) < 120:
                segments[-1].text += " " + ensure_terminal(text)
            else:
                segments.append(Segment(ensure_terminal(text), PAUSE_HEADING))
            continue

        sentences = [s for sent in split_sentences(text) for s in _split_long(sent, max_chars)]
        group = ""
        for sent in sentences:
            if group and len(group) + 1 + len(sent) > max_chars:
                segments.append(Segment(group, PAUSE_SENTENCE))
                group = sent
            else:
                group = f"{group} {sent}".strip()
        if group:
            segments.append(Segment(group, PAUSE_PARAGRAPH))
        elif segments:
            segments[-1].pause_after = PAUSE_PARAGRAPH
    return segments
