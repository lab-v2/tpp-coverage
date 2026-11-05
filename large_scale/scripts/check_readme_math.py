#!/usr/bin/env python
"""Lint the README's math for GitHub rendering, and fail before committing.

GitHub renders $...$ / $$...$$ with MathJax, but its markdown layer and macro
restrictions break several things that are valid LaTeX. This checks the rules we
have actually hit, so broken math is caught before it is committed:

  1. balanced $$ (display) and inline $ delimiters;
  2. inline $...$ content has no leading/trailing whitespace  (GitHub drops the
     math when a space sits next to a delimiter, e.g. "...\\rbrace $");
  3. an inline opening $ is not glued to a word char or '-'  (e.g.
     "holdout-$R^2$"), and a closing $ is not glued to a following word char;
  4. no raw \\{ or \\}  (markdown strips the backslash, so the braces vanish) ->
     use \\lbrace / \\rbrace;
  5. no GitHub-unsupported macros: \\operatorname, \\bigl, \\bigr, \\Bigl,
     \\Bigr, \\biggl, \\biggr, \\tfrac, and the backslash-punctuation spacers
     \\! \\, \\; \\:  (markdown strips the backslash, leaving literal
     punctuation, e.g. "\\," -> ",");
  6. balanced { } and \\left / \\right inside every span;
  7. no '}_' inside inline math (a subscript right after a closing brace):
     GitHub's markdown reads that _ (flanked by non-word chars) as emphasis and
     drops the whole span -- put such sub/superscripted terms in a $$ display
     block instead.

Usage: python3 scripts/check_readme_math.py   (exit 0 clean, 1 on any issue)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
BANNED = [r"\operatorname", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
          r"\biggl", r"\biggr", r"\tfrac", r"\!", r"\,", r"\;", r"\:"]


def main() -> int:
    text = README.read_text()
    errors: list[tuple[int, str]] = []

    def line_of(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    # 1. delimiter balance
    if text.count("$$") % 2:
        errors.append((0, f"odd number of $$ display delimiters ({text.count('$$')})"))
    singles = len(re.findall(r"(?<!\$)\$(?!\$)", text))
    if singles % 2:
        errors.append((0, f"odd number of inline $ delimiters ({singles})"))

    # extract display spans, then blank them so the inline scan is clean
    spans: list[tuple[str, str, int]] = []
    masked = list(text)
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        spans.append(("display", m.group(1), m.start()))
        for i in range(m.start(), m.end()):
            masked[i] = "\n" if text[i] == "\n" else " "
    masked = "".join(masked)

    for m in re.finditer(r"\$([^\$\n]+?)\$", masked):
        content = text[m.start(1):m.end(1)]
        spans.append(("inline", content, m.start()))
        ln = line_of(m.start())
        before = text[m.start() - 1] if m.start() > 0 else "\n"
        after = text[m.end()] if m.end() < len(text) else "\n"
        if re.match(r"[A-Za-z0-9\-]", before):
            errors.append((ln, f"inline $ opens glued to '{before}' (needs a space before it): "
                               f"...{text[max(0, m.start() - 14):m.end()]}"))
        if re.match(r"[A-Za-z0-9]", after):
            errors.append((ln, f"inline $ closes glued to '{after}' (needs a space/punctuation after)"))
        if content[:1].isspace() or content[-1:].isspace():
            errors.append((ln, f"inline math has edge whitespace (GitHub will not render it): ${content}$"))
        if "}_" in content:
            errors.append((ln, "inline '}_' (subscript right after a brace) -> GitHub reads _ as "
                               "emphasis and drops the span; move this term to a $$ display block"))

    # 4-6. per-span content checks
    for kind, content, start in spans:
        ln = line_of(start)
        if "\\{" in content or "\\}" in content:
            errors.append((ln, r"raw \{ or \} (markdown strips the backslash; use \lbrace / \rbrace)"))
        for b in BANNED:
            if b in content:
                errors.append((ln, f"GitHub-unsupported macro {b}"))
        if content.count("{") != content.count("}"):
            errors.append((ln, "unbalanced { } inside a math span"))
        if content.count("\\left") != content.count("\\right"):
            errors.append((ln, r"unbalanced \left / \right inside a math span"))

    if errors:
        seen = set()
        print("README math check FAILED:")
        for ln, msg in sorted(errors):
            if (ln, msg) not in seen:
                seen.add((ln, msg))
                print(f"  line {ln}: {msg}")
        return 1

    n_disp = sum(1 for k, _, _ in spans if k == "display")
    n_inl = sum(1 for k, _, _ in spans if k == "inline")
    print(f"README math OK: {n_disp} display + {n_inl} inline spans, no issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
