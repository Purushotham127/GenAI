from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
import os

def handle_pdf(filepath : str) -> str:
    '''Handling the pdf, reads the data and returns the data string.'''

    if(os.path.exists(filepath)):
        pdf_loader = PyPDFLoader(filepath)
        data_extracted = pdf_loader.load()
    else:
        raise(FileNotFoundError)
    return data_extracted

def handle_doc(filepath : str) -> str:
    '''Handling the doc, reads the data and returns the data string.'''

    if(os.path.exists(filepath)):
        doc_loader = Docx2txtLoader(filepath)
        data_extracted = doc_loader.load()
    else:
        raise(FileNotFoundError)
    return data_extracted

def handle_text(filepath : str) -> str:
    '''Handling the text file, reads the data and returns the data string.'''

    if(os.path.exists(filepath)):
        text_loader = TextLoader(filepath)
        data_extracted = text_loader.lazy_load()
    else:
        raise(FileNotFoundError)
    return data_extracted