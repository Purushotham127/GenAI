from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
import json
from .dev_decorators import calc_latency

@calc_latency
def handle_docling_doc(filepath: str):
    docling_converter = DocumentConverter()
    doc_data = docling_converter.convert(filepath)

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(filepath).name}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(doc_data.document.export_to_dict(), fp, indent=2)

@calc_latency
def handle_pdf(filepath : str) -> str:
    '''Handling the pdf, reads the data and returns the data string.'''
    if(os.path.exists(filepath)):
        pdf_loader = PyPDFLoader(filepath)
        data_extracted = pdf_loader.load()
    else:
        raise(FileNotFoundError)
    return data_extracted

@calc_latency
def handle_doc(filepath : str) -> str:
    '''Handling the doc, reads the data and returns the data string.'''
    if(os.path.exists(filepath)):
        doc_loader = Docx2txtLoader(filepath)
        data_extracted = doc_loader.load()
    else:
        raise(FileNotFoundError)
    return data_extracted

@calc_latency
def handle_text(filepath : str) -> list[Document]:
    '''Read text documents before the caller removes the temporary file.'''
    if(os.path.exists(filepath)):
        text_loader = TextLoader(filepath)
        data_extracted = text_loader.load()
    else:
        raise(FileNotFoundError)
    return data_extracted