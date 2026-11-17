
import pymupdf


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract selectable text from an uploaded PDF resume.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.

    Returns:
        Extracted and cleaned text.

    Raises:
        ValueError: If the PDF is invalid or contains no extractable text.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        pages = []
        for page in document:
            pages.append(page.get_text())

        document.close()

        text = "\n".join(pages).strip()
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        if not text:
            raise ValueError(
                "No selectable text found. This may be a scanned PDF."
            )

        return text

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not read the PDF: {error}") from error
