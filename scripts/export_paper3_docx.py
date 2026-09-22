"""Create a publication-style DOCX from the evidence-corrected manuscript."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(r"E:\Nodel\ExerciseProject\AutoResearchClaw\artifacts\paper3_arc23\recovery_deliverables")
OUT = ROOT / "SCAFFOLD_Clinical_evidence_corrected.docx"

def set_font(run, name="Calibri", size=11, bold=False, color=None):
    run.font.name = name; run._element.rPr.rFonts.set(qn("w:ascii"), name); run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size); run.bold = bold
    if color: run.font.color.rgb = RGBColor(*color)

def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill); tc_pr.append(shd)

def widths(table, values):
    table.autofit = False
    grid = table._tbl.tblGrid.gridCol_lst
    for col, width in zip(grid, values): col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, values):
            cell.width = Inches(width / 1440)

def para(doc, text, style=None, italic=False):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text); set_font(r, size=11); r.italic = italic
    p.paragraph_format.space_after = Pt(8); p.paragraph_format.line_spacing = 1.25
    return p

def main():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    section.header_distance = section.footer_distance = Inches(0.492)
    normal = doc.styles["Normal"]; normal.font.name = "Calibri"; normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); normal.font.size = Pt(11)
    for name, size, before, after, color in [("Heading 1",16,16,8,(46,116,181)), ("Heading 2",13,12,6,(46,116,181))]:
        style = doc.styles[name]; style.font.name="Calibri"; style._element.rPr.rFonts.set(qn("w:ascii"),"Calibri"); style.font.size=Pt(size); style.font.color.rgb=RGBColor(*color); style.paragraph_format.space_before=Pt(before); style.paragraph_format.space_after=Pt(after)
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r=title.add_run("SCAFFOLD-Clinical: an outcome-blind audit contract for foundation-model assessment of recorded bleeding events"); set_font(r,size=18,bold=True,color=(11,37,69)); title.paragraph_format.space_after=Pt(8)
    sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=sub.add_run("Evidence-corrected exploratory medical-informatics report"); set_font(r,size=10,color=(89,89,89)); sub.paragraph_format.space_after=Pt(18)
    for heading, body in [
        ("Abstract", "Foundation-model analyses of structured clinical records are vulnerable to outcome leakage and non-reproducible reporting. We evaluated SCAFFOLD-Clinical, a research-only audit contract that freezes a baseline state card, records a write-ahead hash for each model request and response, and connects outcomes only after scoring. In a de-identified retrospective cohort of 86 encounters, 16 had a recorded post-anticoagulation bleeding flag. MiniMax-M3 produced one outcome-blind, locked score per encounter. The locked score's ROC AUC was 0.642 (bootstrap 95% CI 0.475-0.796); its average precision was 0.298. These exploratory results do not establish calibration, clinical superiority, or clinical utility."),
        ("Introduction", "Risk scoring with general-purpose language models can be misleading when the endpoint, follow-up documentation, or identifiers enter the prompt. SCAFFOLD-Clinical requires a predefined field allow-list, schema-constrained JSON, evidence-field pointers, and an append-only score lock before outcome linkage. Recorded bleeding was treated as an administrative exploratory endpoint, not as adjudicated major bleeding; the source data do not establish ISTH major-bleeding criteria."),
        ("Methods", "The source CSV and SPSS dictionary contained 86 de-identified encounters. State cards used only a predeclared baseline allow-list; no endpoint, discharge field, identifier, dose unit, or unverified laboratory reference interval was included. MiniMax-M3 was called in 15 batches. For every batch, the audit trail stores state cards, raw response text, UTC timestamp, and SHA-256 request and response hashes. Each score was then locked before a separate process joined the recorded endpoint. ROC AUC used 2,000 nonparametric bootstrap resamples. HAS-BLED and Caprini were fixed-score comparators, not refitted models."),
    ]:
        doc.add_paragraph(heading, style="Heading 1"); para(doc, body)
    doc.add_paragraph("Results", style="Heading 1")
    para(doc, "All 86 lock records and all 86 outcome-join records were present; every join hash matched its lock hash. The endpoint string was absent from every state-card batch and write-ahead lock. Three seeds vary bootstrap resampling only, not model inference.")
    table = doc.add_table(rows=1, cols=4); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.style="Table Grid"; widths(table,[3200,1300,2900,1960])
    hdr=["Condition","ROC AUC","Bootstrap 95% CI","Average precision"]
    for c,t in zip(table.rows[0].cells,hdr):
        c.text=t; shade(c,"E8EEF5"); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in c.paragraphs[0].runs: set_font(run,size=9,bold=True)
    rows=[("SCAFFOLD locked MiniMax-M3 score","0.642","0.474-0.790","0.298"),("HAS-BLED source score","0.775","0.647-0.883","0.424"),("Caprini source score","0.551","0.410-0.692","0.212"),("Prevalence constant","0.500","0.500-0.500","0.186"),("Rank/IQR computational check","0.634","0.467-0.785","0.287")]
    for data in rows:
        cells=table.add_row().cells
        for c,t in zip(cells,data):
            c.text=t; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in c.paragraphs[0].runs: set_font(run,size=9)
    note=doc.add_paragraph(); note.paragraph_format.space_before=Pt(4); note.paragraph_format.space_after=Pt(8); r=note.add_run("Note. Fixed clinical scores are not interpreted as probabilities; their Brier scores are therefore not used for calibration claims."); set_font(r,size=9); r.italic=True
    doc.add_paragraph("Discussion", style="Heading 1")
    para(doc, "The central result is procedural: an outcome-blind score can be traced from state card to response hash, score lock, and independent endpoint connection. The observed performance was exploratory and lower than the source HAS-BLED score in this cohort. A retrospective cohort with 16 events cannot support model selection, calibration assessment, causal claims, clinical deployment, or a claim suitable for a top-quartile clinical journal.")
    doc.add_paragraph("Limitations and safeguards", style="Heading 1")
    para(doc, "The audit does not resolve source-field measurement error, ambiguous units, missing context, a non-adjudicated endpoint, or absent temporal and external validation. The underlying artifacts contain de-identified research data and remain access-controlled. No API credential is stored in the deliverables.")
    doc.add_paragraph("References", style="Heading 1")
    para(doc, "International Society on Thrombosis and Haemostasis. Definitions of bleeding outcomes. https://www.isth.org/page/Definitions", italic=False)
    footer = section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=footer.add_run("Exploratory research report - not for clinical decision-making"); set_font(r,size=8,color=(89,89,89))
    doc.core_properties.title="SCAFFOLD-Clinical evidence-corrected report"; doc.core_properties.author="AutoResearchClaw"
    doc.save(OUT); print(OUT)

if __name__ == "__main__": main()
