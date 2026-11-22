from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
from pathlib import Path
import sys
ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOTDIR))
from Utils.logging_util import logger
from Utils.dev_decorators import calc_latency

TOKENIZER_MODEL = os.environ['TOKENIZER_MODEL']
CHUNK_SIZE = int(os.environ['CHUNK_SIZE'])
CHUNK_OVERLAP = int(os.environ['CHUNK_OVERLAP'])
DENSE_EMBEDDING_MODEL = os.environ['DENSE_EMBEDDING_MODEL']
SPARSE_EMBEDDING_MODEL = os.environ['SPARSE_EMBEDDING_MODEL']
RERANKER_MODEL = os.environ['RERANKER_MODEL']
LLM_MODEL_TEMPERATURE = float(os.environ['LLM_MODEL_TEMPERATURE'])
LLM_MODEL_NAME = os.environ['LLM_MODEL_NAME']
MONITORING_MODEL = os.getenv("MONITORING_MODEL", LLM_MODEL_NAME)
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://127.0.0.1:4000").rstrip("/")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")

if not LITELLM_MASTER_KEY:
    raise RuntimeError(
        "The LiteLLM monitoring proxy is not configured. Start the application with main.py."
    )

def load_model_with_fallback(loader, model_name, **kwargs):
    """Load from the local cache if present; otherwise download once so later runs use the cache.
    Set HF_HUB_OFFLINE=1 to forbid downloads entirely."""
    try:
        return loader(model_name, local_files_only=True, **kwargs)
    except Exception as local_error:
        print(f"Model '{model_name}' not found in local cache ({local_error}). Downloading...")
    return loader(model_name, local_files_only=False, **kwargs)

dense_embedding_model = load_model_with_fallback(SentenceTransformer, DENSE_EMBEDDING_MODEL)
sparse_embedding_model = load_model_with_fallback(SparseTextEmbedding, SPARSE_EMBEDDING_MODEL)
reranker_model = load_model_with_fallback(CrossEncoder, RERANKER_MODEL)

tokenizer = load_model_with_fallback(AutoTokenizer.from_pretrained, TOKENIZER_MODEL)

llm = ChatOpenAI(
    model=MONITORING_MODEL,
    base_url=f"{LITELLM_API_BASE}/v1",
    api_key=LITELLM_MASTER_KEY,
    default_headers={"X-Request-Source": "rag-app"},
    temperature=LLM_MODEL_TEMPERATURE,
    streaming=True,
)

system_prompt = """
    You are an enterprise document assistant.

    Instructions:
    1. Answer only using the provided context.
    2. Do not make up information.
    3. If the answer is not found in the context, say:
    "I could not find sufficient information in the provided documents."
    4. Cite the source document name when available.
    5. Provide a concise and factual answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
"""

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
        for chunk in chunks:
            no_of_chunks = len(tokenizer.encode(chunk.page_content))
            print(f"Chunk size(no.of tokens) : {no_of_chunks}.")
        return chunks
    except Exception as e:
        raise RuntimeError(f"Exception occured in recursive text splitting. Exception : {e}.") from e

@calc_latency
def chunk_uploaded_docs(documents):
    chunks = recursive_char_split_chunking(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    document_source = documents[0].metadata.get('source', 'Unknown') if documents else 'Unknown'
    logger.info(f"Document source : {document_source}, no.of chunks created : {len(chunks)}.")
    return chunks

@calc_latency
def create_dense_embeddings(chunks) -> list:
    chunks_data = [chunk.page_content for chunk in chunks]
    chunks_dense_embeddings = dense_embedding_model.encode(
        chunks_data,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return list(chunks_dense_embeddings)

@calc_latency
def create_sparse_embeddings(chunks) -> list:
    chunks_data = [chunk.page_content for chunk in chunks]
    chunks_sparse_embeddings = sparse_embedding_model.embed(
        chunks_data,
        batch_size=32
    )
    return list(chunks_sparse_embeddings)

def create_prompt_template(system_prompt):
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    return prompt_template

def create_prompt_chain(system_prompt, llm):
    prompt_template = create_prompt_template(system_prompt)
    prompt_chain = prompt_template | llm
    return prompt_chain