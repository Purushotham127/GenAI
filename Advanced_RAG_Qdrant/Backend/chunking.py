from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()

TOKENIZER_MODEL = os.environ['TOKENIZER_MODEL']

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

# Splitting documents data by tokens not with chars
def recursive_char_split_chunking(documents, chunk_size=500, chunk_overlap=100):
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",
                "\n# ",
                "\n\n",
                "\n",
                ". ",
                " ",""
            ]
            )
        chunks = text_splitter.split_documents(documents)
        # print(f"No.of chunks created for the document : {len(chunks)}")
        # print("First chunk : \n", chunks[0])
        for chunk in chunks:
            no_of_chunks = len(tokenizer.encode(chunk.page_content))
            print(f"Chunk size(no.of tokens) : {no_of_chunks}.")
        return chunks
    except Exception as e:
        raise RuntimeError(f"Exception occured in recursive text splitting. Exception : {e}.") from e