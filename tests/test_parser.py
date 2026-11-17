
import pymupdf
import pytest

from src.resume_parser import extract_text_from_pdf


def create_test_pdf():
    """Create a small PDF in memory for testing."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Anvesha Sharma\nPython Developer\nSkills: Python, SQL, Machine Learning"
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_extract_text_from_pdf():
    pdf_bytes = create_test_pdf()

    text = extract_text_from_pdf(pdf_bytes)

    assert "Anvesha Sharma" in text
    assert "Python Developer" in text
    assert "Machine Learning" in text


def test_empty_pdf():
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_pdf(b"")


def test_invalid_pdf():
    with pytest.raises(ValueError):
        extract_text_from_pdf(b"This is not a PDF")
