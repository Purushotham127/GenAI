import os
from uuid import uuid5, NAMESPACE_DNS
# from collections import Counter
from .data_util import handle_doc, handle_pdf, handle_text, handle_docling_doc
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import sys
ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOTDIR))
from Utils.logging_util import logger
from Utils.dev_decorators import calc_latency
from Backend.rag_util import chunk_uploaded_docs, create_dense_embeddings, create_sparse_embeddings, create_prompt_chain, system_prompt, llm
from Backend.rag_backend import qdrant_db

load_dotenv()
qdrant_collection = os.environ['QDRANT_COLLECTION_NAME']
vectors_size = int(os.environ["VECTORS_SIZE"])
topk = int(os.environ["TOPK"])
topn = int(os.environ["TOPN"])

def handle_uploaded_files(files):
    '''This method will handle files and ingest data into Qdrant DB'''
    try:
        for count, file in enumerate(files):
            filename = file.name
            logger.info(f"Processing file {count+1} : {filename}")
            file_ext = filename.split('.')[-1]
            data = file.read()
            tmp_filepath = ''
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
                tmp.write(data)
                tmp_filepath = tmp.name
            file_data = ""
            if(file_ext == 'pdf'):
                file_data = handle_pdf(f'{tmp_filepath}')
                handle_docling_doc(tmp_filepath)
            elif(file_ext == 'docx'):
                file_data = handle_doc(f'{tmp_filepath}')
            elif(file_ext == 'txt'):
                file_data = handle_text(f'{tmp_filepath}')
            os.remove(tmp_filepath)
            if file_data:
                #Calling ingestion method to insert the data into qdrant vectordb
                ingest_data_into_qdrant_db(file_data, filename)
        logger.info("File data read and insertion into qdrant is completed!!")
    except Exception as e:
        logger.error(f"Exception occured in handling uploading files method : {e}", exc_info=True)

def create_chunk_hash(chunk, hash_preppend_str:str = "App:v1"):
    return str(uuid5(NAMESPACE_DNS, hash_preppend_str + chunk))

@calc_latency
def preprocess_chunks(chunks, filename: str) -> list:
    chunks_content = [chunk.page_content for chunk in chunks]
    chunks_hash_ids = [create_chunk_hash(chunk) for chunk in chunks_content]
    distinct_ids = set(chunks_hash_ids)
    if (len(chunks_hash_ids) != len(distinct_ids)):
        # duplicate_counters = Counter(chunks_hash_ids)
        indices_to_delete = []
        updated_chunk_ids = []
        # first_occurence_index_map = {}
        # duplicate_chunk_ids = [chunk_id if duplicate_counters[chunk_id] > 1 for chunk_id in distinct_ids]

        for index, (chunk_id, chunk) in enumerate(zip(chunks_hash_ids, chunks)):
            if chunk_id not in updated_chunk_ids:
                updated_chunk_ids.append(chunk_id)
                # first_occurence_index_map[chunk_id] = index
                chunk.metadata["filename"] = filename
            else:
                indices_to_delete.append(index)
        indices_to_delete.sort(reverse=True)
        for index in indices_to_delete:
            chunks.pop(index)

        logger.info(f"Deleted {len(indices_to_delete)} found duplicates in Pre-Processing.")
        return updated_chunk_ids
    else:
        for chunk in chunks:
            chunk.metadata["filename"] = filename
        return chunks_hash_ids

def ingest_data_into_qdrant_db(file_data, filename):
    try:
        chunks = chunk_uploaded_docs(file_data)
        if qdrant_db.client.collection_exists(collection_name=qdrant_collection):
            logger.info(f"Collection {qdrant_collection} is available in Qdrant DB.")
            setattr(qdrant_db, "collection_name", qdrant_collection)
        else:
            qdrant_db.create_collection(qdrant_collection, vectors_size)
            print(f"New collection {qdrant_collection} created!")
            logger.info(f"New collection {qdrant_collection} created!")

        chunks_hash_ids = preprocess_chunks(chunks, filename)
        datapoint_map, matched_ids = qdrant_db.seggregate_chunks_ids(chunks_hash_ids)

        deleted_indices = []
        # Removing the duplicate chunks if chunks exist in vectorDB
        if matched_ids:
            deleted_indices = qdrant_db.eliminate_duplicate_chunks(chunks, datapoint_map, chunks_hash_ids, matched_ids)

        if chunks:
            chunks_dense_embeddings = create_dense_embeddings(chunks)
            chunks_sparse_embeddings = create_sparse_embeddings(chunks)
            logger.info(f" {len(chunks_sparse_embeddings)} Sparse embeddings and {len(chunks_dense_embeddings)} Dense embeddings created.")
            data_points = qdrant_db.create_data_points(chunks, chunks_dense_embeddings, chunks_sparse_embeddings)
            qdrant_db.insert_data_points(data_points)
            logger.info(f"{len(data_points)} Data point(s) inserted, Skipping {len(deleted_indices)} found duplicated chunk(s).")
    except Exception as e:
        raise(RuntimeError(f"Exception : {str(e)}"))

def get_topk_results(query):
    if hasattr(qdrant_db, "collection_name"):
        ...
    else:
        setattr(qdrant_db, "collection_name", qdrant_collection)
    results = qdrant_db.get_topk_chunks(query, topk, topn)
    return results

@calc_latency
def get_llm_response(user_query, results):
    prompt_chain = create_prompt_chain(system_prompt, llm)
    results_data = "\n\n".join([result.get("page_content", "") for result in results])
    logger.info("\n".join(
        [f"Chunk{id+1}, semantic score: {result.get("semantic_score", "NaN")}, relevance_score: {result.get("relevancy_score(reranker)", "NaN")}"
            for id, result in enumerate(results)
        ]
    ))
    response = prompt_chain.invoke({
        "context" : results_data,
        "question" : user_query
    })
    return results_data, response