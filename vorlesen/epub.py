"""EPUB -> chapters of paragraphs, using only the standard library."""
from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote
from xml.etree import ElementTree as ET

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
}

BLOCK_TAGS = {"p", "div", "li", "blockquote", "section", "article", "tr", "dd", "dt",
              "h1", "h2", "h3", "h4", "h5", "h6", "pre", "figcaption"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {"script", "style", "head", "sup", "rt"}  # sup: footnote markers


@dataclass
class Paragraph:
    text: str
    heading: bool = False


@dataclass
class Chapter:
    title: str
    href: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p.text) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    language: str
    chapters: list[Chapter]
    cover: bytes | None = None
    cover_ext: str = "jpg"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[Paragraph] = []
        self._buf: list[str] = []
        self._heading_depth = 0
        self._skip_depth = 0
        self._in_heading_block = False

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf).replace("\xa0", " ")).strip()
        if text:
            self.paragraphs.append(Paragraph(text, heading=self._in_heading_block))
        self._buf = []
        self._in_heading_block = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._flush()
            if tag in HEADING_TAGS:
                self._heading_depth += 1
                self._in_heading_block = True
        elif tag == "br":
            self._buf.append(" ")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            if tag in HEADING_TAGS:
                self._in_heading_block = True
                self._flush()
                self._heading_depth = max(0, self._heading_depth - 1)
            else:
                self._flush()

    def handle_data(self, data):
        if not self._skip_depth:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def _html_paragraphs(html: str) -> list[Paragraph]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.paragraphs


def _toc_labels(zf: zipfile.ZipFile, opf_dir: str, manifest: dict, spine_toc: str | None) -> dict[str, list[str]]:
    """Map document href (without fragment) -> TOC labels pointing into it, in order."""
    labels: dict[str, list[str]] = {}

    def add(src: str, base: str, label: str):
        href = posixpath.normpath(posixpath.join(base, unquote(src.split("#")[0])))
        label = re.sub(r"\s+", " ", label).strip()
        if label:
            labels.setdefault(href, []).append(label)

    # EPUB 3 nav document
    for item in manifest.values():
        if "nav" in (item.get("properties") or "").split():
            path = posixpath.join(opf_dir, item["href"])
            root = ET.fromstring(zf.read(path))
            base = posixpath.dirname(path)
            for nav in root.iter(f"{{{NS['xhtml']}}}nav"):
                if nav.get(f"{{{NS['epub']}}}type") not in (None, "toc"):
                    continue
                for a in nav.iter(f"{{{NS['xhtml']}}}a"):
                    if a.get("href"):
                        add(a.get("href"), base, "".join(a.itertext()))
            if labels:
                return labels

    # EPUB 2 NCX
    ncx_item = manifest.get(spine_toc) if spine_toc else None
    if ncx_item is None:
        ncx_item = next((i for i in manifest.values() if i["media-type"] == "application/x-dtbncx+xml"), None)
    if ncx_item:
        path = posixpath.join(opf_dir, ncx_item["href"])
        root = ET.fromstring(zf.read(path))
        base = posixpath.dirname(path)
        for np in root.iter(f"{{{NS['ncx']}}}navPoint"):
            text = np.find("ncx:navLabel/ncx:text", NS)
            content = np.find("ncx:content", NS)
            if text is not None and content is not None:
                add(content.get("src", ""), base, text.text or "")
    return labels


def read_epub(path: str) -> Book:
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        opf_path = container.find(".//c:rootfile", NS).get("full-path")
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(zf.read(opf_path))

        def dc(name: str) -> str:
            el = opf.find(f".//dc:{name}", NS)
            return (el.text or "").strip() if el is not None else ""

        manifest = {}
        for item in opf.find("opf:manifest", NS):
            manifest[item.get("id")] = {
                "href": unquote(item.get("href")),
                "media-type": item.get("media-type"),
                "properties": item.get("properties"),
            }

        spine = opf.find("opf:spine", NS)
        labels = _toc_labels(zf, opf_dir, manifest, spine.get("toc"))

        chapters = []
        for ref in spine.findall("opf:itemref", NS):
            item = manifest.get(ref.get("idref"))
            if not item or "html" not in item["media-type"]:
                continue
            href = posixpath.normpath(posixpath.join(opf_dir, item["href"]))
            paragraphs = _html_paragraphs(zf.read(href).decode("utf-8", errors="replace"))
            title = " – ".join(labels.get(href, [])) or posixpath.splitext(posixpath.basename(href))[0]
            chapters.append(Chapter(title=title, href=href, paragraphs=paragraphs))

        cover, cover_ext = _find_cover(zf, opf, opf_dir, manifest)

    author = dc("creator")
    if "," in author:  # "Eschbach, Andreas" -> "Andreas Eschbach"
        last, first = [s.strip() for s in author.split(",", 1)]
        author = f"{first} {last}"
    return Book(title=dc("title"), author=author, language=dc("language") or "de",
                chapters=chapters, cover=cover, cover_ext=cover_ext)


def _find_cover(zf, opf, opf_dir, manifest):
    cover_id = None
    for meta in opf.iter(f"{{{NS['opf']}}}meta"):
        if meta.get("name") == "cover":
            cover_id = meta.get("content")
    candidates = []
    if cover_id in manifest:
        candidates.append(manifest[cover_id])
    candidates += [i for i in manifest.values() if "cover-image" in (i.get("properties") or "")]
    candidates += [i for i in manifest.values()
                   if i["media-type"].startswith("image/") and "cover" in i["href"].lower()]
    for item in candidates:
        if item["media-type"].startswith("image/"):
            data = zf.read(posixpath.join(opf_dir, item["href"]))
            ext = "png" if "png" in item["media-type"] else "jpg"
            return data, ext
    return None, "jpg"
