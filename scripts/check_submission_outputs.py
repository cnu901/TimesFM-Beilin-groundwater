"""Audit figure files and separated manuscript deliverables."""

from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    figures = sorted((ROOT / "figures").glob("figure*/Figure*.png"))
    print("PNG audit")
    if len(figures) != 8:
        raise SystemExit(f"expected 8 PNG figures, found {len(figures)}")
    for path in figures:
        image = Image.open(path)
        dpi = image.info.get("dpi", (0.0, 0.0))
        print(f"{path.parent.name}: {image.size}, dpi={dpi}, mode={image.mode}")
        if abs(dpi[0] - 600) > 2 or abs(dpi[1] - 600) > 2:
            raise SystemExit(f"invalid PNG dpi: {path}")

    print("Vector audit")
    for suffix in ("pdf", "svg"):
        files = sorted((ROOT / "figures").glob(f"figure*/Figure*.{suffix}"))
        if len(files) != 8:
            raise SystemExit(f"expected 8 {suffix.upper()} figures, found {len(files)}")
        for path in files:
            if suffix == "svg":
                text = path.read_text(encoding="utf-8")
                if "viewBox=" not in text or "<text" not in text:
                    raise SystemExit(f"SVG lacks editable geometry/text: {path}")
            else:
                if path.stat().st_size < 1000:
                    raise SystemExit(f"PDF is unexpectedly small: {path}")
        print(f"{suffix.upper()}: {len(files)} files")

    print("DOCX separation audit")
    for name in ("revised_manuscript_highlighted.docx", "supplementary_material.docx", "response_to_reviewers.docx"):
        path = ROOT / "manuscript" / name
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
        yellow = xml.count('w:highlight w:val="yellow"')
        print(f"{name}: bytes={len(xml)}, yellow_runs={yellow}, tables={xml.count('<w:tbl>')}")
        if name == "revised_manuscript_highlighted.docx" and yellow == 0:
            raise SystemExit("highlighted manuscript contains no yellow runs")
    print("PASS")


if __name__ == "__main__":
    main()
