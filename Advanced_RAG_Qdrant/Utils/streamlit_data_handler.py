import os
from .data_util import handle_doc, handle_pdf, handle_text
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import sys

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOTDIR))
from Backend.rag_backend import chunk_uploaded_docs

load_dotenv()
data_upload_path = os.environ['DATA_UPLOAD_PATH']

def handle_uploaded_files(files):
    '''This method will handle files and store them in /Uploaded_Data folder'''
    try:
        chunks = []
        for file in files:
            filename = file.name
            file_ext = filename.split('.')[-1]
            file_name_without_ext = filename[:-(len(file_ext)+1)]
            data = file.read()
            tmp_filepath = ''
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
                tmp.write(data)
                tmp_filepath = tmp.name
            if(file_ext == 'pdf'):
                file_data = handle_pdf(f'{tmp_filepath}')
            elif(file_ext == 'docx'):
                file_data = handle_doc(f'{tmp_filepath}')
            elif(file_ext == 'txt'):
                file_data = handle_text(f'{tmp_filepath}')
            os.remove(tmp_filepath)
            if file_data:
                # parsed_data =  "\n".join([document.page_content for document in file_data])
                # with open(f'{data_upload_path}/{file_name_without_ext}.txt','w') as f:
                #     f.write(parsed_data)

                #Calling chunking method to chunk the data
                chunks = chunk_uploaded_docs(file_data)
        print("File data read and Chunking completed!!")
    except Exception as e:
        print(f"Exception occured in handling uploading files method : {e}")