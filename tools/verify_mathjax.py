#!/usr/bin/env python3
"""Validate the shared MathJax setup and TeX delimiters in static pages."""

from html.parser import HTMLParser
from pathlib import Path


WEB = Path(__file__).resolve().parents[1]
MATHJAX_VERSION = "3.2.2"
CDN_URL = f"https://cdn.jsdelivr.net/npm/mathjax@{MATHJAX_VERSION}/es5/tex-svg.js"
IGNORED_TEXT_ELEMENTS = {"code", "pre", "script", "style"}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ignored_depth = 0
        self.text: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in IGNORED_TEXT_ELEMENTS:
            self.ignored_depth += 1
        if tag == "script":
            source = dict(attrs).get("src")
            if source:
                self.scripts.append(source)

    def handle_endtag(self, tag: str) -> None:
        if tag in IGNORED_TEXT_ELEMENTS:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.text.append(data)


def expected_config_path(page: Path) -> str:
    depth = len(page.relative_to(WEB).parents) - 1
    return f"{'../' * depth}assets/js/mathjax-config.js"


def validate_page(page: Path) -> list[str]:
    source = page.read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(source)
    visible_text = "".join(parser.text)
    has_math = any(token in visible_text for token in (r"\(", r"\)", r"\[", r"\]"))
    has_mathjax = any("mathjax" in script.lower() for script in parser.scripts)
    errors: list[str] = []

    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        if visible_text.count(opening) != visible_text.count(closing):
            errors.append(f"unbalanced {opening} {closing} delimiters")

    if has_math or has_mathjax:
        expected = expected_config_path(page)
        if parser.scripts.count(expected) != 1:
            errors.append(f"expected one shared config script: {expected}")
        if parser.scripts.count(CDN_URL) != 1:
            errors.append(f"expected one pinned MathJax script: {CDN_URL}")

    if "window.MathJax" in source:
        errors.append("inline MathJax configuration is not allowed")
    if "mathjax@3/" in source:
        errors.append("MathJax CDN version is not pinned")
    return errors


def main() -> int:
    failures: list[str] = []
    for page in WEB.rglob("*.html"):
        for error in validate_page(page):
            failures.append(f"{page.relative_to(WEB)}: {error}")

    config = (WEB / "assets/js/mathjax-config.js").read_text(encoding="utf-8")
    for literal in ('"\\\\("', '"\\\\)"', '"\\\\["', '"\\\\]"'):
        if literal not in config:
            failures.append(f"assets/js/mathjax-config.js: missing escaped delimiter {literal}")

    if failures:
        print("MathJax verification failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("MathJax verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
