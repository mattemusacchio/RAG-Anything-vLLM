from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, height - 72, "RAG-Anything w/ vLLM Validation")

    # Body text
    c.setFont("Helvetica", 12)
    text = """
    This is a validation document for the RAG-Anything system running with vLLM.
    
    RAG-Anything is a framework that uses a Dual-Graph approach:
    1. Semantic Graph: Connects concepts.
    2. Structural Graph: Connects document layout elements.

    The backend being used right now is vLLM, which provides an OpenAI-compatible API.
    If you are reading this, the parsing step by Mineru was successful.
    """
    
    y = height - 120
    for line in text.split('\n'):
        c.drawString(72, y, line.strip())
        y -= 20

    # Add a simple table-like drawing to verify vision/layout
    y -= 50
    c.setStrokeColor(colors.black)
    c.line(72, y, 400, y)
    c.drawString(72, y - 15, "Component")
    c.drawString(200, y - 15, "Status")
    c.line(72, y - 20, 400, y - 20)
    c.drawString(72, y - 35, "MinerU Parser")
    c.drawString(200, y - 35, "Active")
    c.drawString(72, y - 55, "vLLM Backend")
    c.drawString(200, y - 55, "Active")
    c.line(72, y - 60, 400, y - 60)

    c.save()

if __name__ == "__main__":
    create_pdf("demo_data/test_doc.pdf")
    print("Created demo_data/test_doc.pdf")
