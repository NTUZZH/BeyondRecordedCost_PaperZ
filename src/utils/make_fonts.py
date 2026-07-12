"""Build TrueType copies of TeX Gyre Termes for the figure pipeline.

The manuscript body (newtx) is set in TeX Gyre Termes, the Times New Roman
metric-compatible face shipped with TeX, so the figures should use the same
face. matplotlib can only embed TrueType outlines cleanly (pdf.fonttype 42);
handing it the stock OTF (CFF outlines) produces a PDF whose font descriptor
says TrueType while the payload is CFF, which trips PDF preflight. Converting
the curves once to quadratics fixes that at no visible cost.

Run once: python src/utils/make_fonts.py  -> writes assets/fonts/*.ttf
"""

from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from cu2qu.pens import Cu2QuPen

ROOT = Path(__file__).resolve().parents[2]
SRC = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
OUT = ROOT / "assets" / "fonts"
MAX_ERR = 1.0  # units per em (1/1000 em): visually lossless


def otf_to_ttf(src: Path, dst: Path) -> None:
    font = TTFont(str(src))
    glyph_set = font.getGlyphSet()
    glyf_pen_glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, MAX_ERR, reverse_direction=True))
        glyf_pen_glyphs[name] = pen.glyph()

    from fontTools.ttLib.tables._g_l_y_f import table__g_l_y_f
    glyf = table__g_l_y_f()
    glyf.glyphOrder = font.getGlyphOrder()
    glyf.glyphs = glyf_pen_glyphs
    font["glyf"] = glyf

    from fontTools.ttLib.tables._l_o_c_a import table__l_o_c_a
    font["loca"] = table__l_o_c_a()

    # maxp must advertise TrueType (0x00010000) before compiling glyf.
    font["maxp"].tableVersion = 0x00010000
    font["maxp"].maxZones = 1
    font["maxp"].maxTwilightPoints = 0
    font["maxp"].maxStorage = 0
    font["maxp"].maxFunctionDefs = 0
    font["maxp"].maxInstructionDefs = 0
    font["maxp"].maxStackElements = 0
    font["maxp"].maxSizeOfInstructions = 0
    font["maxp"].maxComponentElements = max(
        (len(g.components) if hasattr(g, "components") else 0)
        for g in glyf_pen_glyphs.values())
    font["head"].indexToLocFormat = 0
    font.sfntVersion = "\000\001\000\000"
    for tag in ("CFF ", "VORG"):
        if tag in font:
            del font[tag]
    dst.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(dst))
    print(f"wrote {dst.relative_to(ROOT)}")


def main() -> None:
    for style in ("regular", "bold", "italic", "bolditalic"):
        src = SRC / f"texgyretermes-{style}.otf"
        if not src.exists():
            raise SystemExit(f"missing {src}; install the TeX Gyre fonts")
        otf_to_ttf(src, OUT / f"TeXGyreTermes-{style}.ttf")


if __name__ == "__main__":
    main()
