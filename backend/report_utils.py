import os
import csv
from fpdf import FPDF

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'uploads', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_csv_report(brand_name, analysis):
    filename = os.path.join(REPORTS_DIR, f'{brand_name}_analysis.csv')
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for k, v in analysis['metrics'].items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(['Recommendation Priority', 'Issue', 'Action', 'Rationale'])
        for rec in analysis['recommendations']:
            writer.writerow([rec['priority'], rec['issue'], rec['action'], rec['rationale']])
    return filename

def generate_pdf_report(brand_name, analysis):
    filename = os.path.join(REPORTS_DIR, f'{brand_name}_analysis.pdf')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Brand Analysis Report: {brand_name}', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Metrics:', ln=True)
    for k, v in analysis['metrics'].items():
        pdf.cell(0, 10, f'{k}: {v}', ln=True)
    pdf.cell(0, 10, '', ln=True)
    pdf.cell(0, 10, 'Recommendations:', ln=True)
    for rec in analysis['recommendations']:
        pdf.multi_cell(0, 10, f"[{rec['priority'].upper()}] {rec['issue']}: {rec['action']}\nRationale: {rec['rationale']}")
        pdf.cell(0, 5, '', ln=True)
    pdf.output(filename)
    return filename 