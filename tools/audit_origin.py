#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def fragments(text: str) -> list[str]:
    return [f.strip() for f in re.split(r"\.\n|\n\n", text) if f.strip()]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("origin.txt")
    text = path.read_text(errors="ignore")
    frags = fragments(text)
    toks = words(text)

    starts = Counter()
    bigrams = Counter()
    for frag in frags:
        ws = words(frag)
        if ws:
            starts[" ".join(ws[:3])] += 1
        for i in range(len(ws) - 1):
            bigrams[(ws[i], ws[i + 1])] += 1

    print(f"path: {path}")
    print(f"chars: {len(text)}")
    print(f"words: {len(toks)}")
    print(f"fragments: {len(frags)}")

    print("\ntop fragment starts:")
    for phrase, n in starts.most_common(20):
        print(f"{n:>3}  {phrase}")

    print("\ntop bigrams:")
    for (a, b), n in bigrams.most_common(25):
        print(f"{n:>3}  {a} {b}")

    repetitive = [
        ("i am", bigrams[("i", "am")]),
        ("this is", bigrams[("this", "is")]),
        ("not a", bigrams[("not", "a")]),
        ("my field", bigrams[("my", "field")]),
    ]
    print("\nwatchlist:")
    for phrase, n in repetitive:
        print(f"{n:>3}  {phrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
