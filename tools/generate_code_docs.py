#!/usr/bin/env python3
"""Generate the static HiPaCs source explorer from Doxygen XML."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
CORE = ROOT / "core"
DOXYFILE = Path(__file__).with_name("Doxyfile")
XML_DIR = WEB / ".generated" / "doxygen" / "xml"
CODE_DIR = WEB / "code"
DETAIL_DIR = CODE_DIR / "functions"
REPORT_PATH = WEB / ".generated" / "function-documentation.json"
SOURCE_SUFFIXES = {".h", ".cpp"}
FOLDER_DESCRIPTIONS = {
    "assembly": "Global sparsity construction and residual/tangent assembly.",
    "boundary": "Dirichlet and Neumann boundary-condition evaluation.",
    "hpx": "Automatic differentiation and local element-system utilities.",
    "io": "Input parsing, VTK/HDF5 output, and postprocessing.",
    "mesh": "Mesh topology, readers, domains, and element connectivity.",
    "multiscale": "RVE constraints, solves, homogenization, and output.",
    "solver": "Nonlinear time stepping and linear solver backends.",
    "space": "Field layouts, shape functions, kinematics, and weak forms.",
    "vem": "Virtual element geometry, projection, and stabilization.",
}


@dataclass
class Function:
    uid: str
    name: str
    signature: str
    brief: str
    detail: str
    group: str
    documented: bool
    file: str
    line: int
    end: int
    code: str
    declaration_file: str
    declaration_line: int

    def listing(self) -> dict[str, object]:
        return {
            "uid": self.uid, "name": self.name, "signature": self.signature,
            "brief": self.brief, "group": self.group, "line": self.line,
            "detailUrl": f"functions/{self.uid}.html",
        }


def text_content(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def run_doxygen() -> None:
    if not shutil.which("doxygen"):
        raise SystemExit("doxygen is required to generate the code explorer")
    (WEB / ".generated").mkdir(exist_ok=True)
    # Doxygen leaves XML for removed or renamed symbols behind between runs.
    if XML_DIR.exists():
        shutil.rmtree(XML_DIR)
    subprocess.run(["doxygen", str(DOXYFILE)], cwd=WEB, check=True)


def core_files() -> dict[str, list[str]]:
    return {
        folder.name: [path.name for path in sorted(folder.iterdir()) if path.suffix in SOURCE_SUFFIXES]
        for folder in sorted(path for path in CORE.iterdir() if path.is_dir())
    }


def relative_core(raw: str) -> Path | None:
    if not raw:
        return None
    try:
        relative = Path(raw).resolve().relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return None
    return relative if len(relative.parts) == 3 and relative.parts[0] == "core" else None


def source_slice(relative: Path, start: int, end: int) -> str:
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    start = max(1, start)
    end = max(start, end)
    return "\n".join(lines[start - 1 : min(len(lines), end)])


def generated_brief(name: str, group: str) -> str:
    readable = re.sub(r"_+", " ", name).strip()
    if name.startswith("operator"):
        return f"Implements the {name} overload."
    if name.startswith("~"):
        return f"Destroys a {name[1:]} instance."
    if group == "special":
        return f"Constructs or manages a {readable} instance."
    return f"Implements {readable}."


def nearby_line_comment(relative: Path, line: int) -> str:
    """Return a meaningful contiguous // comment immediately above a function."""
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    cursor = min(len(lines) - 1, max(0, line - 2))
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    comments: list[str] = []
    while cursor >= 0 and lines[cursor].lstrip().startswith("//"):
        value = lines[cursor].lstrip()[2:].strip(" -/")
        if value:
            comments.append(value)
        cursor -= 1
    comments.reverse()
    joined = " ".join(comments)
    return joined if len(joined) >= 12 else ""


CPP_TOKEN_RE = re.compile(
    r"(?P<comment>//[^\n]*|/\*.*?\*/)|"
    r"(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|"
    r"(?P<preprocessor>^[ \t]*\#[^\n]*)|"
    r"(?P<number>\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b)|"
    r"(?P<keyword>\b(?:alignas|alignof|auto|bool|break|case|catch|char|class|const|constexpr|continue|default|delete|do|double|else|enum|explicit|extern|false|float|for|friend|if|inline|int|long|namespace|new|nullptr|operator|private|protected|public|return|short|signed|sizeof|static|struct|switch|template|this|throw|true|try|typedef|typename|union|unsigned|using|virtual|void|volatile|while)\b)",
    re.MULTILINE | re.DOTALL,
)


def highlight_cpp(code: str, first_line: int) -> str:
    """Apply deterministic local C++ highlighting without runtime dependencies."""
    highlighted_lines: list[str] = []
    for offset, raw_line in enumerate(code.split("\n")):
        parts: list[str] = []
        cursor = 0
        for match in CPP_TOKEN_RE.finditer(raw_line):
            parts.append(html.escape(raw_line[cursor:match.start()]))
            token_class = match.lastgroup or ""
            parts.append(f'<span class="token {token_class}">{html.escape(match.group())}</span>')
            cursor = match.end()
        parts.append(html.escape(raw_line[cursor:]))
        highlighted_lines.append(
            f'<span class="source-line" data-line="{first_line + offset}">{"".join(parts) or " "}</span>'
        )
    return "\n".join(highlighted_lines)


def function_group(member: ET.Element, name: str, declaration: Path, body: Path) -> str:
    definition = text_content(member.find("definition"))
    owner = definition.rsplit("::", 1)[0].split()[-1] if "::" in definition else ""
    if name.startswith(("operator", "~")) or (owner and name == owner.rsplit("::", 1)[-1]):
        return "special"
    if member.get("static") == "yes" or member.get("prot") in {"private", "protected"}:
        return "internal"
    if declaration == body and body.suffix == ".cpp":
        return "internal"
    return "public"


def parse_functions(files: dict[str, list[str]]) -> tuple[dict[str, dict[str, list[Function]]], dict[str, Function]]:
    output = {folder: {name: [] for name in names} for folder, names in files.items()}
    registry: dict[str, Function] = {}
    associations: set[tuple[str, str]] = set()
    for xml_path in sorted(XML_DIR.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        for member in root.findall(".//memberdef[@kind='function']"):
            location = member.find("location")
            if location is None:
                continue
            declaration = relative_core(location.get("file") or "")
            body = relative_core(location.get("bodyfile") or "") or declaration
            if declaration is None and body is None:
                continue
            assert body is not None
            declaration = declaration or body
            name = text_content(member.find("name"))
            args = text_content(member.find("argsstring"))
            definition = text_content(member.find("definition"))
            signature = (definition + args).strip()
            start = int(location.get("bodystart") or location.get("line") or 1)
            end = int(location.get("bodyend") or start)
            declaration_line = int(location.get("line") or start)
            identity = f"{body}:{name}:{args}:{start}"
            uid = "fn-" + hashlib.sha1(identity.encode()).hexdigest()[:14]
            group = function_group(member, name, declaration, body)
            brief = text_content(member.find("briefdescription"))
            detail = text_content(member.find("detaileddescription"))
            detail = detail.replace("Included in the web code explorer.", "").strip()
            source_brief = nearby_line_comment(declaration, declaration_line) or nearby_line_comment(body, start)
            if not brief:
                brief = source_brief
            if uid not in registry:
                registry[uid] = Function(
                    uid=uid, name=name, signature=signature,
                    brief=brief or generated_brief(name, group), detail=detail,
                    group=group, documented=bool(brief), file=str(body), line=start, end=end,
                    code=source_slice(body, start, end), declaration_file=str(declaration),
                    declaration_line=declaration_line,
                )
            elif brief and not registry[uid].documented:
                registry[uid].brief = brief
                registry[uid].detail = detail
                registry[uid].documented = True

            # File XML may describe the implementation before the header entry.
            # Keep the real declaration when both describe the same function.
            if declaration.suffix == ".h" and Path(registry[uid].declaration_file).suffix != ".h":
                registry[uid].declaration_file = str(declaration)
                registry[uid].declaration_line = declaration_line
                registry[uid].group = group

            # A header shows declarations and a .cpp file shows implementations.
            for relative in {declaration, body}:
                folder, filename = relative.parts[1], relative.parts[2]
                key = (str(relative), uid)
                if folder in output and filename in output[folder] and key not in associations:
                    output[folder][filename].append(registry[uid])
                    associations.add(key)

    for folder in output.values():
        for functions in folder.values():
            functions.sort(key=lambda item: (item.group, item.line, item.name, item.signature))
    return output, registry


def nav(prefix: str) -> str:
    links = [
        ("Home", "index.html"), ("Main Page", "main.html"),
        ("Framework", "framework.html"), ("Code Structure", "architecture.html"),
        ("Applications", "applications.html"), ("Gallery", "gallery.html"),
        ("Install", "install.html"), ("Contact", "contact.html"),
    ]
    return "".join(f'<a href="{prefix}{href}">{label}</a>' for label, href in links)


def petsc_note() -> str:
    return """
<section class="petsc-overview" data-petsc-overview hidden>
  <div class="explorer-tabs" role="tablist" aria-label="PETSc solver modes">
    <button type="button" class="active" data-petsc-tab="direct" role="tab" aria-selected="true">Direct Solver</button>
    <button type="button" data-petsc-tab="iterative" role="tab" aria-selected="false">Iterative Solver</button>
  </div>
  <div class="petsc-panel active" data-petsc-panel="direct"><h3>PETSc direct path</h3><p>Uses direct factorization through PETSc KSP/PC configuration for robust reference solves.</p></div>
  <div class="petsc-panel" data-petsc-panel="iterative"><h3>Iterative solver and Schur approximation</h3>
    <p>Mechanical-pressure systems use PETSc FieldSplit. The exact pressure Schur complement is expensive, so the backend selects an approximation.</p>
    <div class="compact-equation">S = K<sub>pp</sub> − K<sub>pm</sub>K<sub>mm</sub><sup>−1</sup>K<sub>mp</sub></div>
    <ul class="clean-list"><li><strong>Factorization:</strong> lower, upper, or full block coupling.</li><li><strong>Matrix-based:</strong> A11, SELFP, or FULL.</li>
      <li><strong>Pressure operators:</strong> mass, Laplacian, or a weighted combination.</li><li><strong>PCD:</strong> an operator chain using M<sub>p</sub>, K<sub>p</sub>, and A<sub>p</sub>.</li></ul>
    <a class="text-link" href="../pages/framework-schur.html">Open the full Schur and PCD mathematical note →</a>
  </div>
</section>"""


def folder_page(folder: str, names: list[str], functions: dict[str, list[Function]]) -> str:
    payload = {
        "folder": folder,
        "files": [{"name": name, "path": f"core/{folder}/{name}", "functions": [fn.listing() for fn in functions[name]]} for name in names],
    }
    safe_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="HiPaCs core/{folder} source and function explorer."><title>core/{folder} · HiPaCs</title><link rel="stylesheet" href="../assets/css/styles.css"></head>
<body><header class="topbar"><a class="brand" href="../index.html"><span class="brand-mark">HC</span><span>HiPaCs</span></a><button class="menu-button" type="button" aria-label="Open navigation" aria-expanded="false"><span></span><span></span><span></span></button>
<nav class="top-links" aria-label="Primary navigation">{nav('../')}</nav></header><main id="top"><section class="subpage-hero section-wrap wide-page-wrap"><a class="back-button" href="../architecture.html">← Core Directory Explorer</a>
<p class="eyebrow">Core directory</p><h1>core/{folder}</h1><p class="lead">{FOLDER_DESCRIPTIONS.get(folder, '')}</p></section>
<section class="doc-section section-wrap wide-page-wrap code-explorer" data-code-explorer>
<div class="explorer-column explorer-files"><div class="explorer-column-heading"><span>Files</span><small>{len(names)} total</small></div><div data-file-list></div></div>
<div class="explorer-column explorer-functions"><div class="explorer-column-heading"><span>Functions</span><small data-function-count>all files</small></div>
<div class="function-search-wrap"><label for="function-search">Search functions</label><input id="function-search" type="search" placeholder="Function name or signature" data-function-search></div>
{petsc_note() if folder == 'solver' else ''}<div data-function-list></div></div></section></main>
<script id="code-explorer-data" type="application/json">{safe_payload}</script><script defer src="../assets/js/main.js"></script><script defer src="../assets/js/code-explorer.js"></script></body></html>"""


def detail_page(fn: Function, previous: Function | None, following: Function | None, commit: str, repo: str) -> str:
    folder = Path(fn.file).parts[1]
    filename = Path(fn.file).name
    revision = "main" if commit == "working-tree" else commit
    github = f"https://github.com/{repo}/blob/{revision}/{fn.file}#L{fn.line}"
    back = f"../{folder}.html?file={html.escape(filename, quote=True)}"
    prev_link = f'<a href="{previous.uid}.html">← {html.escape(previous.name)}()</a>' if previous else "<span></span>"
    next_link = f'<a href="{following.uid}.html">{html.escape(following.name)}() →</a>' if following else "<span></span>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{html.escape(fn.brief, quote=True)}"><title>{html.escape(fn.name)}() · HiPaCs</title>
<link rel="stylesheet" href="../../assets/css/styles.css"></head>
<body><header class="topbar"><a class="brand" href="../../index.html"><span class="brand-mark">HC</span><span>HiPaCs</span></a><button class="menu-button" type="button" aria-label="Open navigation" aria-expanded="false"><span></span><span></span><span></span></button><nav class="top-links" aria-label="Primary navigation">{nav('../../')}</nav></header>
<main><section class="subpage-hero section-wrap wide-page-wrap function-source-hero"><a class="back-button" href="{back}">← core/{folder}/{filename}</a>
<p class="eyebrow">Function detail</p><h1>{html.escape(fn.name)}()</h1><p class="lead">{html.escape(fn.brief)}</p></section>
<section class="doc-section section-wrap wide-page-wrap function-source-page"><div class="function-meta-card"><span class="tag">{html.escape(fn.group.title())}</span>
<h2>Signature</h2><p class="function-signature"><code>{html.escape(fn.signature)}</code></p>{f'<p>{html.escape(fn.detail)}</p>' if fn.detail else ''}
<dl class="source-location-list"><div><dt>Implementation</dt><dd>{html.escape(fn.file)}:{fn.line}</dd></div><div><dt>Declaration</dt><dd>{html.escape(fn.declaration_file)}:{fn.declaration_line}</dd></div></dl>
<a class="text-link" href="{github}" target="_blank" rel="noopener">View source on GitHub ↗</a></div>
<div class="source-code-card"><div class="source-code-heading"><h2>Full implementation</h2><span>lines {fn.line}–{fn.end}</span></div>
<pre class="source-code-light"><code class="language-cpp">{highlight_cpp(fn.code, fn.line)}</code></pre></div>
<nav class="function-sequence" aria-label="Adjacent functions">{prev_link}{next_link}</nav></section></main>
<script defer src="../../assets/js/main.js"></script></body></html>"""


def update_architecture(files: dict[str, list[str]], functions: dict[str, dict[str, list[Function]]]) -> None:
    path = WEB / "architecture.html"
    source = path.read_text(encoding="utf-8")
    source = source.replace("The code-structure page collects the implementation map, function maps, and file-oriented module view in one technical documentation page.", "Follow the implementation flow, then browse the source through the real core directory structure.")
    def folder_link(folder: str) -> str:
        return f'<a class="flow-folder-card" href="code/{folder}.html"><strong>{folder}</strong><small>{FOLDER_DESCRIPTIONS[folder]}</small></a>'

    flow = f'''<!-- GENERATED:CODE_FLOW:START -->
<div aria-label="Code structure flow" class="code-flow-map">
  <article class="code-flow-group setup-flow-group">
    <span class="group-label">Setup</span>
    <a class="flow-folder-card flow-setup-card" href="code/setup.html"><strong>setup</strong><small>Application entry points, mesh input, and runtime text configuration.</small></a>
  </article>
  <article class="code-flow-group finite-element-flow-group">
    <span class="group-label">Finite Element Core</span>
    <div class="flow-folder-grid">{''.join(folder_link(name) for name in ('assembly', 'boundary', 'hpx', 'mesh', 'space'))}</div>
  </article>
  <article class="code-flow-group solver-flow-group">
    <span class="group-label">Solver</span>
    {folder_link('solver')}
  </article>
  <article class="code-flow-group extensions-flow-group">
    <span class="group-label">Output &amp; Extensions</span>
    <div class="flow-folder-grid">{''.join(folder_link(name) for name in ('io', 'vem', 'multiscale'))}</div>
  </article>
</div>
<!-- GENERATED:CODE_FLOW:END -->'''
    flow_pattern = re.compile(r"<!-- GENERATED:CODE_FLOW:START -->.*?<!-- GENERATED:CODE_FLOW:END -->", re.S)
    if flow_pattern.search(source):
        source = flow_pattern.sub(flow, source)
    else:
        old_pattern = re.compile(
            r'<div aria-label="Code structure map".*?'
            r'<!-- GENERATED:CORE_DIRECTORY_EXPLORER:END -->',
            re.S,
        )
        source = old_pattern.sub(flow, source)
    path.write_text(source, encoding="utf-8")


def write_legacy_stubs() -> None:
    mapping = {"assembly":"assembly","autodiff":"hpx","backend":"solver","input":"io","kinematics":"space","mesh":"mesh","multiscale":"multiscale","postprocess":"io","solver":"solver","space":"space","vem":"vem","weakform":"space"}
    for legacy, folder in mapping.items():
        target = f"../code/{folder}.html"
        (WEB / "functions" / f"{legacy}.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><!-- TODO(legacy-removal): Delete this compatibility page after all inbound links use code/*.html. --><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0; url={target}"><title>Function map moved · HiPaCs</title><link rel="canonical" href="{target}"><link rel="stylesheet" href="../assets/css/styles.css"></head><body><main><section class="subpage-hero section-wrap"><p class="eyebrow">Compatibility page</p><h1>This function map has moved.</h1><p><a class="button primary" href="{target}">Open core/{folder} →</a></p></section></main></body></html>''', encoding="utf-8")


def update_sitemap(folders: list[str]) -> None:
    path = WEB / "sitemap.xml"
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"\s*<url>\s*<loc>[^<]+/(?:functions|code)/[^<]+</loc>.*?</url>", "", source, flags=re.S)
    entries = "".join(f"\n  <url><loc>https://yongbinchoi91.github.io/code/{folder}.html</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>" for folder in folders)
    path.write_text(source.replace("\n</urlset>", entries + "\n</urlset>"), encoding="utf-8")


def write_report(registry: dict[str, Function]) -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    report = {
        "total": len(registry), "documented": sum(fn.documented for fn in registry.values()),
        "generated": sum(not fn.documented for fn in registry.values()),
        "functions": [{"file": fn.file, "line": fn.line, "name": fn.name, "group": fn.group, "documented": fn.documented} for fn in sorted(registry.values(), key=lambda item: (item.file, item.line, item.name))],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-doxygen", action="store_true")
    args = parser.parse_args()
    if not args.skip_doxygen:
        run_doxygen()
    files = core_files()
    functions, registry = parse_functions(files)
    CODE_DIR.mkdir(exist_ok=True)
    if DETAIL_DIR.exists():
        for stale in DETAIL_DIR.glob("*.html"):
            stale.unlink()
    DETAIL_DIR.mkdir(exist_ok=True)
    commit = os.environ.get("GITHUB_SHA", "working-tree")
    repo = os.environ.get("HIPACS_GITHUB_REPOSITORY", "YongbinChoi91/HiPaCs")
    for folder, names in files.items():
        (CODE_DIR / f"{folder}.html").write_text(folder_page(folder, names, functions[folder]), encoding="utf-8")
    ordered = sorted(registry.values(), key=lambda fn: (fn.file, fn.line, fn.name, fn.signature))
    for index, fn in enumerate(ordered):
        previous = ordered[index - 1] if index and ordered[index - 1].file == fn.file else None
        following = ordered[index + 1] if index + 1 < len(ordered) and ordered[index + 1].file == fn.file else None
        (DETAIL_DIR / f"{fn.uid}.html").write_text(detail_page(fn, previous, following, commit, repo), encoding="utf-8")
    update_architecture(files, functions)
    write_legacy_stubs()
    update_sitemap(["setup", *files])
    write_report(registry)
    print(f"Generated {len(files)} directory pages and {len(registry)} function detail pages.")
    print(f"Descriptions: {sum(fn.documented for fn in registry.values())} documented, {sum(not fn.documented for fn in registry.values())} generated.")


if __name__ == "__main__":
    main()
