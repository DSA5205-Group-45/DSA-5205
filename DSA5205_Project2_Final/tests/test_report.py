"""Academic report deliverables: PDF, LaTeX source, pagination and consistency."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "report"


def _skip_if_missing():
    if not (REPORT / "Project2_Report.pdf").exists():
        pytest.skip("Report PDF has not been generated in this checkout")


def test_report_pdf_exists_and_page_count():
    _skip_if_missing()
    from pypdf import PdfReader

    reader = PdfReader(str(REPORT / "Project2_Report.pdf"))
    pages = len(reader.pages)
    assert 15 <= pages <= 25, pages


def test_latex_source_is_present_and_substantial():
    _skip_if_missing()
    tex = REPORT / "Project2_Report.tex"
    assert tex.exists()
    text = tex.read_text()
    assert len(text) > 50_000
    assert text.count("\\section{") >= 15
    assert "\\begin{document}" in text and "\\end{document}" in text


def test_six_required_figures_exist():
    _skip_if_missing()
    figures = PROJECT_ROOT / "figures"
    expected = [
        "fig_score_to_position.png",
        "fig_four_model.png",
        "fig_ridge_benchmarks.png",
        "fig_eight_summary.png",
        "fig_drawdowns.png",
        "fig_annual.png",
        "fig_risk_adjusted.png",
        "fig_correlation.png",
        "fig_stability.png",
    ]
    for name in expected:
        path = figures / name
        assert path.exists(), name
        assert path.stat().st_size > 20_000


def test_pdf_has_no_broken_references_or_placeholders():
    _skip_if_missing()
    from pypdf import PdfReader

    reader = PdfReader(str(REPORT / "Project2_Report.pdf"))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "??" not in text
    for placeholder in ("TODO", "TBD", "Lorem ipsum", "[insert"):
        assert placeholder.lower() not in text.lower()


def test_report_contains_key_empirical_numbers():
    _skip_if_missing()
    from pypdf import PdfReader

    reader = PdfReader(str(REPORT / "Project2_Report.pdf"))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    oos = pd.read_csv(PROJECT_ROOT / "results/oos/eight_strategy/metrics.csv").set_index("strategy")
    expected = [
        f"{oos.loc['ridge', 'annualized_return'] * 100:.1f}",
        f"{oos.loc['B1_Momentum10', 'annualized_return'] * 100:.1f}",
        f"{oos.loc['B3_MomentumDoubleSort10', 'annualized_return'] * 100:.1f}",
    ]
    for value in expected:
        assert value in text, value
    stability = pd.read_csv(PROJECT_ROOT / "results/diagnostics/coefficient_stability_summary.csv").set_index("model")
    assert f"{stability.loc['ridge', 'mean_relative_l2_change']:.3f}" in text
    assert f"{stability.loc['lasso', 'mean_relative_l2_change']:.3f}" in text


def test_report_references_all_main_figures():
    _skip_if_missing()
    from pypdf import PdfReader

    reader = PdfReader(str(REPORT / "Project2_Report.pdf"))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for number in range(1, 8):
        assert f"Figure {number}" in text, number
    assert "Figure A1" in text and "Figure A2" in text


def test_report_discloses_designation_chronology():
    _skip_if_missing()
    from pypdf import PdfReader

    reader = PdfReader(str(REPORT / "Project2_Report.pdf"))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "not a clean pre-registration" in text
    assert "lasso" in text
    assert "ridge" in text


def test_compliance_checklist_covers_all_required_sections():
    path = PROJECT_ROOT / "COMPLIANCE_CHECKLIST.md"
    assert path.exists()
    text = path.read_text()
    for section in [f"§4.{i}" for i in range(1, 10)]:
        assert section in text, section
    assert ("Figures 1–7" in text) or ("Figures 1--7" in text)


def test_report_figure_and_table_numbering():
    _skip_if_missing()
    tex = (REPORT / "Project2_Report.tex").read_text()
    # Figure 1 is score-to-position, 2-7 OOS, 8 R4 reconciliation, A1-A2 diagnostics.
    for label in ("fig:score", "fig:fourmodel", "fig:benchmarks", "fig:summary", "fig:drawdowns", "fig:annual", "fig:risk"):
        assert f"\\label{{{label}}}" in tex
    assert "\\setcounter{figure}{0}" in tex
    tables = tex.count("\\begin{table}")
    tables += sum(path.read_text().count("\\begin{table}") for path in (REPORT / "generated").glob("table_*.tex"))
    assert tables >= 8
