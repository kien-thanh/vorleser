"""German text normalization for TTS: spell out what voice models tend to garble."""
from __future__ import annotations

import re

from num2words import num2words

ABBREVIATIONS = {
    "z. B.": "zum Beispiel", "z.B.": "zum Beispiel",
    "d. h.": "das heisst", "d.h.": "das heisst",
    "u. a.": "unter anderem", "u.a.": "unter anderem",
    "o. ä.": "oder ähnliches", "o.ä.": "oder ähnliches",
    "u. U.": "unter Umständen", "u.U.": "unter Umständen",
    "s. o.": "siehe oben", "s. u.": "siehe unten",
    "v. a.": "vor allem", "v.a.": "vor allem",
    "bzw.": "beziehungsweise", "usw.": "und so weiter", "etc.": "et cetera",
    "ca.": "circa", "evtl.": "eventuell", "ggf.": "gegebenenfalls",
    "vgl.": "vergleiche", "bzgl.": "bezüglich", "inkl.": "inklusive",
    "sog.": "sogenannte", "Nr.": "Nummer", "Dr.": "Doktor", "Prof.": "Professor",
    "Hr.": "Herr", "Fr.": "Frau", "St.": "Sankt", "Str.": "Strasse",
    "Mio.": "Millionen", "Mrd.": "Milliarden", "Jh.": "Jahrhundert",
    "v. Chr.": "vor Christus", "n. Chr.": "nach Christus",
}
UNITS = {"%": "Prozent", "km": "Kilometer", "m": "Meter", "cm": "Zentimeter", "mm": "Millimeter",
         "kg": "Kilogramm", "g": "Gramm", "km/h": "Kilometer pro Stunde", "°C": "Grad Celsius",
         "°": "Grad", "€": "Euro", "$": "Dollar", "h": "Stunden", "min": "Minuten", "s": "Sekunden"}

# Articles/prepositions after which an ordinal takes the weak "-en" ending.
OBLIQUE = {"des", "dem", "den", "im", "am", "vom", "zum", "beim", "zur", "der", "einem", "eines", "einen"}
MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
DIGITS = "null eins zwei drei vier fünf sechs sieben acht neun".split()

_SENT_END = re.compile(r"[.!?…:;»«\"'”)\]]$")


def _cardinal(s: str) -> str:
    return num2words(int(s), lang="de")


def _ordinal(n: str, prev_word: str) -> str:
    word = num2words(int(n), lang="de", to="ordinal")
    return word + "n" if prev_word.lower() in OBLIQUE else word


def _digitwise(s: str) -> str:
    return " ".join(DIGITS[int(d)] for d in s if d.isdigit())


def _year(n: int) -> str:
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return num2words(hi, lang="de") + "hundert" + (num2words(lo, lang="de") if lo else "")
    return num2words(n, lang="de")


def _fix_caps(text: str) -> str:
    """Small-caps lead-ins and shouted names arrive as ALL CAPS; models then spell them out.
    Runs of 2+ caps words and single caps words of 4+ letters become Capitalized.
    Short isolated caps (KI, UN, DNA) are kept as acronyms."""
    caps = r"[A-ZÄÖÜ][A-ZÄÖÜß]+"

    def run(m):
        return re.sub(caps, lambda w: w.group(0).capitalize(), m.group(0))

    text = re.sub(rf"\b{caps}(?:[ ,\-]+{caps}\b)+", run, text)
    return re.sub(rf"\b[A-ZÄÖÜ][A-ZÄÖÜß]{{3,}}\b", lambda m: m.group(0).capitalize(), text)


def _numbers(text: str) -> str:
    # Codes: hyphenated digit groups or very long digit runs -> digit by digit.
    text = re.sub(r"\b\d+(?:-\d+)+\b", lambda m: ", ".join(_digitwise(g) for g in m.group(0).split("-")), text)
    text = re.sub(r"\b\d{7,}\b", lambda m: _digitwise(m.group(0)), text)
    # Dates "3. Mai", ordinals "des 191. Pantap"
    text = re.sub(rf"\b(\d{{1,2}})\.\s+({MONTHS})\b",
                  lambda m: f"{_ordinal(m.group(1), 'den')} {m.group(2)}", text)
    text = re.sub(r"(\b\w+\s+)?\b(\d{1,4})\.(?=\s+[A-ZÄÖÜa-zäöü])",
                  lambda m: (m.group(1) or "") + _ordinal(m.group(2), (m.group(1) or "").strip()), text)
    # Years after "im Jahr(e)" / "anno"
    text = re.sub(r"\b((?:im |das |dem )?Jahre? |anno )(1[1-9]\d\d)\b",
                  lambda m: m.group(1) + _year(int(m.group(2))), text)
    # Decimals "3,8" (no space after the comma, otherwise it's a list)
    text = re.sub(r"\b(\d+),(\d+)",
                  lambda m: f"{_cardinal(m.group(1))} Komma {_digitwise(m.group(2)) if len(m.group(2)) > 2 or m.group(2).startswith('0') else _cardinal(m.group(2))}",
                  text)
    # Thousands separators "12.000"
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: _cardinal(m.group(0).replace(".", "")), text)
    # Units after numbers
    for unit in sorted(UNITS, key=len, reverse=True):
        text = re.sub(rf"(\d)\s?{re.escape(unit)}(?![\wäöüß])", rf"\1 {UNITS[unit]}", text)
    # Remaining plain integers ("60B" -> "sechzig B", but "4fache" stays joined)
    text = re.sub(r"(\d)([A-ZÄÖÜ])(?![a-zäöüß])", r"\1 \2", text)
    return re.sub(r"\d+", lambda m: _cardinal(m.group(0)), text)


def normalize(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)  # _emphasis_
    text = re.sub(r"\*+(.+?)\*+", r"\1", text)
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        text = re.sub(rf"(?<![\wäöü]){re.escape(abbr)}", ABBREVIATIONS[abbr], text)
    text = _fix_caps(text)
    text = _numbers(text)
    # Quotes and punctuation the models handle inconsistently
    text = re.sub(r"[»«„“”]", '"', text)
    text = re.sub(r"[›‹‚‘]", "", text)
    text = text.replace("’", "'").replace("…", "...")
    text = re.sub(r"\s+[-–—]\s+", ", ", text)  # spaced dash = pause
    text = text.replace("•", ".")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return text


def merge_broken_paragraphs(paragraphs: list[str]) -> list[str]:
    """Rejoin paragraphs split mid-sentence (common in converted ebooks)."""
    out: list[str] = []
    for p in paragraphs:
        if out and not _SENT_END.search(out[-1]) and p[:1].islower():
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def ensure_terminal(text: str) -> str:
    """Titles like 'Das Wrack' get a period so the voice ends them cleanly."""
    text = text[:1].upper() + text[1:]
    return text if _SENT_END.search(text) else text + "."
