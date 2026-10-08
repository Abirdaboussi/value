from pathlib import Path
import fitz  # PyMuPDF
import tiktoken
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_REPORTS_DIR = PROJECT_ROOT / "data" / "raw" / "reports"
RAW_HIGHLIGHTS_DIR = PROJECT_ROOT / "data" / "raw" / "highlights"

REPORTS_TXT_DIR = PROJECT_ROOT / "data" / "processed" / "reports_txt"
REFERENCES_TXT_DIR = PROJECT_ROOT / "data" / "processed" / "references_txt"

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

REPORTS_TXT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_TXT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """
    Extract all text from a PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    tuple[str, int]
        (full_text, page_count)
    """
    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        pages.append(page.get_text())

    text = "\n".join(pages)
    page_count = len(document)

    document.close()

    return text, page_count

def save_text(text: str, output_path: Path):
    """
    Save text content into a txt file.

    Parameters
    ----------
    text : str
        Text to save.

    output_path : Path
        Destination txt file.
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken.
    """

    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    return len(tokens)



def count_words(text: str) -> int:
    """
    Count words in a text.

    Parameters
    ----------
    text : str
        Text whose words should be counted.

    Returns
    -------
    int
        Number of words.
    """
    return len(text.split())


def count_words(text: str) -> int:
    """
    Count words in a text.

    Parameters
    ----------
    text : str
        Text whose words should be counted.

    Returns
    -------
    int
        Number of words.
    """
    return len(text.split())


def process_reports(reports_dir: Path, output_dir: Path) -> list[dict]:
    """
    Process all report PDFs.

    Extracts text from each PDF, saves it as a .txt file,
    and collects metadata for the manifest.

    Parameters
    ----------
    reports_dir : Path
        Directory containing report PDFs.

    output_dir : Path
        Directory where extracted report text will be saved.

    Returns
    -------
    list[dict]
        Metadata for each processed report.
    """

    metadata = []

    pdf_files = sorted(reports_dir.glob("*.pdf"))

    for pdf_path in pdf_files:

        doc_id = pdf_path.stem

        text, page_count = extract_pdf_text(pdf_path)

        output_path = output_dir / f"{doc_id}.txt"

        save_text(text, output_path)

        token_count = count_tokens(text)

        metadata.append({
            "doc_id": doc_id,
            "report_path": str(output_path),
            "report_page_count": page_count,
            "report_token_count": token_count
        })

    return metadata


def process_highlights(highlights_dir: Path, output_dir: Path) -> list[dict]:
    """
    Process all highlight/reference PDFs.

    Extracts text from each highlight PDF, saves it as a .txt file,
    and collects metadata for the manifest.

    Parameters
    ----------
    highlights_dir : Path
        Directory containing highlight PDFs.

    output_dir : Path
        Directory where extracted highlight text will be saved.

    Returns
    -------
    list[dict]
        Metadata for each processed highlight.
    """

    metadata = []

    pdf_files = sorted(highlights_dir.glob("*.pdf"))

    for pdf_path in pdf_files:

        doc_id = pdf_path.stem.removesuffix("-highlights")

        text, _ = extract_pdf_text(pdf_path)

        output_path = output_dir / f"{doc_id}.txt"

        save_text(text, output_path)

        word_count = count_words(text)

        metadata.append({
            "doc_id": doc_id,
            "highlight_path": str(output_path),
            "highlight_word_count": word_count
        })

    return metadata


def build_manifest(
    report_metadata: list[dict],
    highlight_metadata: list[dict],
    manifest_path: Path
):
    """
    Build the benchmark manifest by combining report and highlight metadata.

    Parameters
    ----------
    report_metadata : list[dict]
        Metadata returned by process_reports().

    highlight_metadata : list[dict]
        Metadata returned by process_highlights().

    manifest_path : Path
        Path where manifest.csv will be written.
    """

    highlights_by_id = {
        item["doc_id"]: item
        for item in highlight_metadata
    }

    rows = []

    for report in report_metadata:

        doc_id = report["doc_id"]

        highlight = highlights_by_id.get(doc_id)

        if highlight is not None:
            highlight_path = highlight["highlight_path"]
            highlight_word_count = highlight["highlight_word_count"]
            ok = True
        else:
            highlight_path = ""
            highlight_word_count = 0
            ok = False

        rows.append({
            "doc_id": doc_id,
            "report_path": report["report_path"],
            "highlight_path": highlight_path,
            "report_page_count": report["report_page_count"],
            "report_token_count": report["report_token_count"],
            "highlight_word_count": highlight_word_count,
            "ok": ok
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_id",
        "report_path",
        "highlight_path",
        "report_page_count",
        "report_token_count",
        "highlight_word_count",
        "ok"
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)
        

def main():
    """
    Run the complete data ingestion pipeline.
    """

    print("Processing reports...")

    report_metadata = process_reports(
        RAW_REPORTS_DIR,
        REPORTS_TXT_DIR
    )

    print(f"Processed {len(report_metadata)} reports.")

    print("Processing highlights...")

    highlight_metadata = process_highlights(
        RAW_HIGHLIGHTS_DIR,
        REFERENCES_TXT_DIR
    )

    print(f"Processed {len(highlight_metadata)} highlights.")

    print("Building manifest...")

    build_manifest(
        report_metadata,
        highlight_metadata,
        MANIFEST_PATH
    )

    print(f"Manifest created: {MANIFEST_PATH}")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()