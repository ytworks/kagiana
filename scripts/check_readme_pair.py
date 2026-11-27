#!/usr/bin/env python3
"""Check that README.md and README.ja.md say the same thing.

The two cannot be compared word for word, so what can be compared is: the outline (the levels of
the headings from `##` down, in order), the commands and configuration in the code blocks (with
their comments removed, since the comments are translated), the links to files of the repository,
the numbers in the tables, and that each names the other. Every relative link and every anchor of
both must resolve, as GitHub resolves them. Run by the `lint` job of CI. Standard library only.
"""
from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
#: the one pair of documents: the English file, the Japanese file, and how each links the other
PAIRS = (("README.md", "README.ja.md", "README.md", "README.ja.md"),)
PAIR = PAIRS[0][:2]
#: a number: sign, digits, decimals
NUMBER = re.compile(r"[-+\u2212]?\d+(?:\.\d+)?")
#: a cell of a table that holds numbers only (a value, a value with its error or interval)
NUMERIC_CELL = re.compile(r"^[-+\u2212\u00b1\d.,\s\[\]()/%]*\d[-+\u2212\u00b1\d.,\s\[\]()/%]*$")

FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
INLINE_LINK = re.compile(r"\]\(\s*(<[^>\n]*>|(?:[^\s()]|\([^\s()]*\))+)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)")
REFERENCE_USE = re.compile(r"\[([^\]]+)\]\[([^\]]*)\]")
DEFINITION = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>\n]*>|\S+)")
HREF = re.compile(r"<(?:a|img)\s[^>]*(?<![-\w])(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
HTML_ANCHOR = re.compile(r"<\w+\s[^>]*(?<![-\w])(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
SHORTCUT_REFERENCE = re.compile(r"(?<![\]!])\[([^\]]+)\](?![\[(:])")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


OUTSIDE, OPENING, INSIDE, CLOSING = 0, 1, 2, 3


def fence_states(lines: list[str]) -> list[int]:
    """For every line its place in fenced code blocks (OUTSIDE, OPENING, INSIDE, CLOSING), as
    CommonMark reads fences: three or more backticks or tildes (indented, also inside list items);
    a backtick fence has no backtick in its info string; it is closed by a line of the same
    character, at least as long; an unclosed fence runs to the end of the document. Shared with
    scripts/check_progress_size.py."""
    states, fence = [], ""
    for line in lines:
        if not fence:
            m = FENCE_OPEN.match(line)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = m.group(1)
                states.append(OPENING)
            else:
                states.append(OUTSIDE)
        elif re.match(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$", line):
            fence = ""
            states.append(CLOSING)
        else:
            states.append(INSIDE)
    return states


def blocks_and_text(text: str) -> tuple[list[list[str]], list[str]]:
    """The fenced code blocks (their lines, without the fences) and the lines outside them, with
    HTML comments removed from the latter."""
    lines = text.splitlines()
    blocks, outside = [], []
    for line, state in zip(lines, fence_states(lines)):
        if state == OPENING:
            blocks.append([])
        elif state == INSIDE:
            blocks[-1].append(line)
        elif state == OUTSIDE:
            outside.append(line)
    if not outside:
        return blocks, []
    joined = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), "\n".join(outside), flags=re.DOTALL)
    return blocks, joined.split("\n")


def headings(text: str) -> list[tuple[int, str]]:
    """The headings (level, text), ATX and setext, outside code blocks."""
    _, lines = blocks_and_text(text)
    found = []
    for i, line in enumerate(lines):
        m = ATX.match(line)
        if m:
            found.append((len(m.group(1)), (m.group(2) or "").strip()))
            continue
        s = SETEXT.match(line)
        prev = lines[i - 1] if i else ""
        if s and prev.strip() and not ATX.match(prev) and not re.match(r"^ {0,3}([-*+]|\d+[.)])\s", prev) \
                and not SETEXT.match(prev):
            j = i - 1                            # the paragraph above, which may span lines
            while j > 0 and lines[j - 1].strip() and not ATX.match(lines[j - 1]) and not SETEXT.match(lines[j - 1]):
                j -= 1
            found.append((1 if s.group(1)[0] == "=" else 2, " ".join(ln.strip() for ln in lines[j:i])))
    return found


def outline(text: str) -> list[int]:
    """The levels of the headings from `##` down, in order."""
    return [level for level, _ in headings(text) if level >= 2]


def uncommented(block: list[str]) -> list[str]:
    """The lines of a code block without comments and blank lines (comments are translated). A `#`
    starts a comment at the start of a line or after a space, outside quotes."""
    kept = []
    for line in block:
        quote, cut = "", len(line)
        for i, c in enumerate(line):
            if quote:
                if c == quote:
                    quote = ""
            elif c == '"' or (c == "'" and (i == 0 or line[i - 1] in " \t=([{,")):
                quote = c
            elif c == "#" and (i == 0 or line[i - 1].isspace()):
                cut = i
                break
        line = line[:cut].rstrip()
        if line.strip():
            kept.append(line)
    return kept


def links(text: str) -> list[str]:
    """Link targets outside code blocks and code spans, in order: inline links and images,
    reference links with their definitions, and HTML links."""
    _, lines = blocks_and_text(text)
    prose = [CODE_SPAN.sub("", ln) for ln in lines]
    definitions = {}
    for ln in prose:
        m = DEFINITION.match(ln)
        if m:
            definitions[m.group(1).strip().lower()] = m.group(2)
    found = []
    for ln in prose:
        if DEFINITION.match(ln):
            continue
        found += [m.group(1) for m in INLINE_LINK.finditer(ln)]
        for m in REFERENCE_USE.finditer(ln):
            key = (m.group(2) or m.group(1)).strip().lower()
            found.append(definitions.get(key, f"[undefined reference {key}]"))
        rest = REFERENCE_USE.sub("", INLINE_LINK.sub("", ln))
        found += [definitions[m.group(1).strip().lower()] for m in SHORTCUT_REFERENCE.finditer(rest)
                  if m.group(1).strip().lower() in definitions]
        found += [m.group(1) for m in HREF.finditer(ln)]
    return [t[1:-1] if t.startswith("<") and t.endswith(">") else t for t in found]


def is_external(target: str) -> bool:
    return bool(EXTERNAL.match(target))


def slugify(heading: str) -> str:
    """The anchor GitHub gives a heading text: links and images by their text, code spans by their
    content, HTML tags dropped; lower case; letters, numbers, spaces, hyphens and underscores kept,
    everything else dropped; spaces to hyphens."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = CODE_SPAN.sub(r"\2", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip().lower()
    text = "".join(c for c in text if c in " -_" or unicodedata.category(c)[0] in "LNM")
    return text.replace(" ", "-")


def slug(heading: str) -> str:
    return slugify(heading)


def anchors(text: str) -> set[str]:
    """The anchors of a document: of every heading, numbered on repetition as github-slugger
    numbers them, and of every HTML element with an id or a name."""
    occurrences: dict[str, int] = {}
    found = set()
    for _, title in headings(text):
        base = result = slugify(title)
        while result in occurrences:
            occurrences[base] += 1
            result = f"{base}-{occurrences[base]}"
        occurrences[result] = 0
        found.add(result)
    _, lines = blocks_and_text(text)
    found |= {m.group(1) for ln in lines for m in HTML_ANCHOR.finditer(ln)}
    return found


def link_problems(name: str, text: str, root: Path) -> list[str]:
    """Relative links of a Markdown file whose file or anchor does not exist."""
    problems = []
    for target in links(text):
        if target.startswith("[undefined reference"):
            problems.append(f"{name}: {target[1:-1]}")
            continue
        if is_external(target):
            continue
        path, _, anchor = target.partition("#")
        path, anchor = path.partition("?")[0], unquote(anchor)
        if not path:                                   # an anchor of this file
            if anchor not in anchors(text):
                problems.append(f"{name}: link to {target}: no such heading")
            continue
        file = root / unquote(path).lstrip("/") if path.startswith("/") else (root / name).parent / unquote(path)
        if not file.exists():
            problems.append(f"{name}: link to {target}: no such file")
        elif anchor and file.suffix == ".md" and anchor not in anchors(file.read_text(encoding="utf-8")):
            problems.append(f"{name}: link to {target}: no such heading")
    return problems


def table_numbers(text: str) -> list[str]:
    """The numbers in the cells of the tables outside code blocks that hold numbers only, in order
    (cells of prose are translated and may count in words; a minus sign written as U+2212 counts as
    a hyphen-minus). Numbers in prose, in cells with words, units, code spans or emphasis are not
    compared: keep the numbers of a result in cells of their own."""
    _, lines = blocks_and_text(text)
    cells = [cell.strip() for line in lines if line.lstrip().startswith("|")
             for cell in line.strip().strip("|").split("|")]
    return [m.group(0).replace("\u2212", "-") for cell in cells if NUMERIC_CELL.match(cell)
            for m in NUMBER.finditer(cell)]


def pair_problems(english: str, japanese: str, root: Path = ROOT, pair: tuple = PAIRS[0]) -> list[str]:
    """Differences between the two versions of a document given as text (empty list: none);
    ``pair`` is one of :data:`PAIRS`."""
    en_name, ja_name, en_ref, ja_ref = pair
    problems = []
    if outline(english) != outline(japanese):
        problems.append(f"the headings differ: {en_name} has levels {outline(english)}, "
                        f"{ja_name} {outline(japanese)}")
    en_blocks = [uncommented(b) for b in blocks_and_text(english)[0]]
    ja_blocks = [uncommented(b) for b in blocks_and_text(japanese)[0]]
    if len(en_blocks) != len(ja_blocks):
        problems.append(f"{en_name} has {len(en_blocks)} code blocks, {ja_name} {len(ja_blocks)}")
    for i, (en, ja) in enumerate(zip(en_blocks, ja_blocks), start=1):
        if en != ja:
            first = next((a, b) for a, b in zip(en + [""], ja + [""]) if a != b)
            problems.append(f"code block {i} differs apart from comments: {first[0]!r} / {first[1]!r}")
    files = [[unquote(t.partition("#")[0]) for t in links(x) if not is_external(t) and not t.startswith(("#", "["))]
             for x in (english, japanese)]
    en_files, ja_files = (sorted(set(f) - {en_ref, ja_ref}) for f in files)
    if en_files != ja_files:
        problems.append(f"links to files differ: only in {en_name} {sorted(set(en_files) - set(ja_files))}, "
                        f"only in {ja_name} {sorted(set(ja_files) - set(en_files))}")
    targets = [{unquote(t.partition("#")[0]) for t in links(x)} for x in (english, japanese)]
    if ja_ref not in targets[0]:
        problems.append(f"{en_name} does not link to {ja_ref}")
    if en_ref not in targets[1]:
        problems.append(f"{ja_name} does not link to {en_ref}")
    en_numbers, ja_numbers = table_numbers(english), table_numbers(japanese)
    if en_numbers != ja_numbers:
        at = next((i for i, (a, b) in enumerate(zip(en_numbers, ja_numbers)) if a != b),
                  min(len(en_numbers), len(ja_numbers)))
        problems.append(f"the numbers in the tables differ: {en_name} has {len(en_numbers)}, {ja_name} "
                        f"{len(ja_numbers)}; the first difference is number {at + 1}: "
                        f"{(en_numbers + ['(none)'])[at]!r} / {(ja_numbers + ['(none)'])[at]!r}")
    problems += link_problems(en_name, english, root) + link_problems(ja_name, japanese, root)
    return problems


def main() -> int:
    status = 0
    for pair in PAIRS:
        texts = []
        for name in pair[:2]:
            path = ROOT / name
            if not path.is_file():
                print(f"check_readme_pair: {path} not found", file=sys.stderr)
                return 1
            texts.append(path.read_text(encoding="utf-8"))
        problems = pair_problems(*texts, ROOT, pair)
        if problems:
            print(f"{pair[0]} and {pair[1]} differ:\n- " + "\n- ".join(problems))
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
