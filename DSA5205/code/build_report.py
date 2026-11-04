"""Build the PDF report: report.md -> HTML (pandoc) -> PDF (xhtml2pdf)."""

import hashlib
import os
import re
import subprocess
import sys
from io import BytesIO


def _patch_pml_image_reader_str():
    """Work around an xhtml2pdf 0.2.23 + reportlab 5 image-dedup bug.

    PmlImageReader.__str__ derives the image cache key from the *unread tail*
    of the PNG stream; every PNG ends with the same IEND CRC bytes, so all
    images hash identically and reportlab dedups every figure into the first
    one. Key by the md5 of the full stream instead.
    """
    from xhtml2pdf import xhtml2pdf_reportlab as xr

    def __str__(self):
        fn = self.fileName
        if isinstance(fn, BytesIO):
            pos = fn.tell()
            fn.seek(0)
            data = fn.read()
            fn.seek(pos)
            return "PmlImageObject_" + hashlib.md5(data).hexdigest()
        return str(fn or id(self))

    xr.PmlImageReader.__str__ = __str__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "report.md")
HTML = os.path.join(ROOT, "output", "report.html")
PDF = os.path.join(ROOT, "Group_X_Report.pdf")

CSS_TPL = """
@font-face { font-family: DejaVu; src: url('FONTDIR/DejaVuSans.ttf'); }
@font-face { font-family: DejaVu; font-weight: bold;
               src: url('FONTDIR/DejaVuSans-Bold.ttf'); }
@font-face { font-family: DejaVu; font-style: italic;
               src: url('FONTDIR/DejaVuSans-Oblique.ttf'); }
@page { size: a4; margin: 2cm; }
body { font-family: DejaVu, Helvetica, Arial, sans-serif; font-size: 9.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 4px; }
h2 { font-size: 12pt; margin-top: 18px; }
h3 { font-size: 10.5pt; }
img { max-width: 100%%; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 8px 0; }
th, td { border: 1px solid #888; padding: 3px 6px; text-align: center; }
code { font-family: Courier New, monospace; font-size: 8.5pt;
       background: #f2f2f2; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6px;
      font-size: 8pt; }
.titleblock { text-align: center; margin-top: 120px; margin-bottom: 60px; }
.titleblock h1 { border: none; font-size: 20pt; }
blockquote { border-left: 3px solid #ccc; padding-left: 10px;
             color: #444; }
"""


def main():
    import matplotlib
    import shutil
    fdir = os.path.join(ROOT, "output", "fonts")
    os.makedirs(fdir, exist_ok=True)
    ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for f in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"]:
        shutil.copy(os.path.join(ttf, f), os.path.join(fdir, f))
    subprocess.run(["pandoc", MD, "-f", "markdown", "-t", "html5",
                    "-s", "-o", HTML, "--metadata",
                    "title=DSA5205 Project 2 - Group_X"],
                   check=True, cwd=ROOT)
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    html = re.sub(r"<html[^>]*>", "<html>", html)  # xhtml2pdf chokes on lang=
    # xhtml2pdf resolves images relative to the HTML file location; rewrite
    # figure paths to absolute file paths.
    html = html.replace('src="output/figures/',
                        'src="' + os.path.join(ROOT, "output", "figures").replace("\\", "/") + "/")
    css = CSS_TPL.replace("FONTDIR", fdir.replace("\\", "/")).replace("%%", "%")
    html = html.replace("</head>", f"<style>{css}</style></head>")
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    _patch_pml_image_reader_str()
    from xhtml2pdf import pisa
    with open(PDF, "wb") as out:
        status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if status.err:
        sys.exit("PDF conversion reported errors")
    print("wrote", PDF)


if __name__ == "__main__":
    main()
