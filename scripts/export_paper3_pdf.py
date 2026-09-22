"""Generate the evidence-corrected report as a standalone PDF."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT = Path(r"E:\Nodel\ExerciseProject\AutoResearchClaw\artifacts\paper3_arc23\recovery_deliverables\SCAFFOLD_Clinical_evidence_corrected.pdf")

def main():
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0B2545"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#595959"), spaceAfter=18))
    styles.add(ParagraphStyle(name="PaperBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="PaperHead", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#2E74B5"), spaceBefore=12, spaceAfter=6))
    doc=SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch, title="SCAFFOLD-Clinical evidence-corrected report", author="AutoResearchClaw")
    story=[Paragraph("SCAFFOLD-Clinical: an outcome-blind audit contract for foundation-model assessment of recorded bleeding events",styles["PaperTitle"]),Paragraph("Evidence-corrected exploratory medical-informatics report",styles["Subtitle"])]
    sections=[
      ("Abstract","Foundation-model analyses of structured clinical records are vulnerable to outcome leakage and non-reproducible reporting. We evaluated SCAFFOLD-Clinical, a research-only audit contract that freezes a baseline state card, records a write-ahead hash for each model request and response, and connects outcomes only after scoring. In a de-identified retrospective cohort of 86 encounters, 16 had a recorded post-anticoagulation bleeding flag. MiniMax-M3 produced one outcome-blind, locked score per encounter. The locked score's ROC AUC was 0.642 (bootstrap 95% CI 0.475-0.796); its average precision was 0.298. These exploratory results do not establish calibration, clinical superiority, or clinical utility."),
      ("Introduction","Risk scoring with general-purpose language models can be misleading when the endpoint, follow-up documentation, or identifiers enter the prompt. SCAFFOLD-Clinical requires a predefined field allow-list, schema-constrained JSON, evidence-field pointers, and an append-only score lock before outcome linkage. Recorded bleeding was treated as an administrative exploratory endpoint, not as adjudicated major bleeding."),
      ("Methods","The source CSV and SPSS dictionary contained 86 de-identified encounters. State cards used only a predeclared baseline allow-list; no endpoint, discharge field, identifier, dose unit, or unverified laboratory reference interval was included. MiniMax-M3 was called in 15 batches. For every batch, the audit trail stores state cards, raw response text, UTC timestamp, and SHA-256 request and response hashes. Each score was locked before a separate process joined the recorded endpoint. ROC AUC used 2,000 nonparametric bootstrap resamples."),
    ]
    for h,b in sections: story += [Paragraph(h,styles["PaperHead"]),Paragraph(b,styles["PaperBody"])]
    story += [Paragraph("Results",styles["PaperHead"]),Paragraph("All 86 lock records and all 86 outcome-join records were present; every join hash matched its lock hash. The endpoint string was absent from every state-card batch and write-ahead lock. Three seeds vary bootstrap resampling only, not model inference.",styles["PaperBody"])]
    data=[["Condition","ROC AUC","Bootstrap 95% CI","Average precision"],["SCAFFOLD locked MiniMax-M3 score","0.642","0.474-0.790","0.298"],["HAS-BLED source score","0.775","0.647-0.883","0.424"],["Caprini source score","0.551","0.410-0.692","0.212"],["Prevalence constant","0.500","0.500-0.500","0.186"],["Rank/IQR computational check","0.634","0.467-0.785","0.287"]]
    t=Table(data,colWidths=[2.25*inch,.8*inch,1.7*inch,1.25*inch],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E8EEF5")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8.5),("LEADING",(0,0),(-1,-1),11),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#AAB7C4")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])); story += [t,Spacer(1,6),Paragraph("Note. Fixed clinical scores are not interpreted as probabilities; their Brier scores are therefore not used for calibration claims.",styles["PaperBody"])]
    for h,b in [("Discussion","The central result is procedural: an outcome-blind score can be traced from state card to response hash, score lock, and independent endpoint connection. The observed performance was exploratory and lower than the source HAS-BLED score in this cohort. A retrospective cohort with 16 events cannot support model selection, calibration assessment, causal claims, clinical deployment, or a claim suitable for a top-quartile clinical journal."),("Limitations and safeguards","The audit does not resolve source-field measurement error, ambiguous units, missing context, a non-adjudicated endpoint, or absent temporal and external validation. The underlying artifacts contain de-identified research data and remain access-controlled. No API credential is stored in the deliverables."),("Reference","International Society on Thrombosis and Haemostasis. Definitions of bleeding outcomes. https://www.isth.org/page/Definitions")]: story += [Paragraph(h,styles["PaperHead"]),Paragraph(b,styles["PaperBody"])]
    doc.build(story); print(OUT)
if __name__ == "__main__": main()
