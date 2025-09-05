import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

def ai_act_report(csv_path: str, out_pdf: str, meta: dict = None):
    meta = meta or {}
    df = pd.read_csv(csv_path)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("ERA – AI Act Readiness Report", styles['Title']))
    story.append(Spacer(1, 12))
    info = [
        ['Baseline model', meta.get('baseline_model','N/A')],
        ['Candidate model', meta.get('candidate_model','N/A')],
        ['CSI Alert threshold', str(meta.get('csi_alert','N/A'))],
    ]
    t = Table(info, colWidths=[150, 350])
    t.setStyle(TableStyle([('BOX',(0,0),(-1,-1),1,colors.black),
                           ('INNERGRID',(0,0),(-1,-1),0.5,colors.black)]))
    story.append(t); story.append(Spacer(1, 12))

    story.append(Paragraph("Per-concept metrics", styles['Heading2']))
    rows = [['Domain','Concept','EmbDist','ProbShift','CSI','Alert']]
    for _, r in df.iterrows():
        rows.append([r['domain'], r['concept'], f"{r['embedding_distance']:.3f}", f"{r['prob_shift']:.3f}", f"{r['CSI']:.3f}", "YES" if r['alert'] else "NO"])
    t2 = Table(rows); t2.setStyle(TableStyle([('BOX',(0,0),(-1,-1),1,colors.black),('INNERGRID',(0,0),(-1,-1),0.25,colors.black)]))
    story.append(t2)

    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    doc.build(story)
    return out_pdf
