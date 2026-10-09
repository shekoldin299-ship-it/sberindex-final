"""Render the authoritative Markdown report without network dependencies."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('D', str(FONT_DIR / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DB', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('D', normal='D', bold='DB', italic='D', boldItalic='DB')
styles = {
    'body': ParagraphStyle('body', fontName='D', fontSize=9.1, leading=13.1, spaceAfter=8, textColor=colors.HexColor('#25352e')),
    'h1': ParagraphStyle('h1', fontName='DB', fontSize=24, leading=29, spaceAfter=18, textColor=colors.HexColor('#125b36')),
    'h2': ParagraphStyle('h2', fontName='DB', fontSize=14, leading=18, spaceBefore=9, spaceAfter=10, textColor=colors.HexColor('#125b36')),
    'cell': ParagraphStyle('cell', fontName='D', fontSize=8, leading=11, alignment=TA_LEFT),
}


def para(text, style='body'):
    return Paragraph(escape(text), styles[style])


def render():
    lines = (ROOT / 'REPORT.md').read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', s) for s in cells):
                    rows.append([para(s, 'cell') for s in cells])
                i += 1
            table = Table(rows, colWidths=[487 / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e5f1e8')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f7f5')]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor('#8faf98')),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        elif line.startswith('!['):
            target = re.search(r'\]\((.*?)\)', line).group(1)
            path = ROOT / target
            width, height = ImageReader(str(path)).getSize()
            factor = min(487 / width, 250 / height)
            story.extend([Image(str(path), width=width * factor, height=height * factor), Spacer(1, 10)])
        elif line.startswith('# '):
            story.append(para(line[2:], 'h1'))
        elif line.startswith('## '):
            story.append(para(line[3:], 'h2'))
        else:
            story.append(para(line))
        i += 1

    def footer(canvas, doc):
        canvas.setFont('D', 8)
        canvas.setFillColor(colors.HexColor('#66796b'))
        canvas.drawString(54, 27, 'СберИндекс / прогноз и мониторинг / 09.10.2026')
        canvas.drawRightString(541, 27, str(doc.page))
    doc = SimpleDocTemplate(str(ROOT / 'REPORT.pdf'), pagesize=(595,842),
        leftMargin=54, rightMargin=54, topMargin=42, bottomMargin=44,
        title='СберИндекс: прогноз потребления и мониторинг изменений', author='Щеколдин Ярослав Данилович')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(ROOT / 'REPORT.pdf')


if __name__ == '__main__':
    render()
