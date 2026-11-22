from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SparseVectorParams, SparseVector, Distance, Prefetch, FusionQuery, SetPayloadOperation, SetPayload
from uuid import uuid5, NAMESPACE_DNS
from typing import List
from datetime import datetime, timezone
from .rag_util import dense_embedding_model, sparse_embedding_model, reranker_model, CrossEncoder
from pathlib import Path
import sys
ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOTDIR))
from Utils.logging_util import logger
from Utils.dev_decorators import calc_latency

class QdrantVectorDB:
    '''Custom Qdrant vector database class'''
    def __init__(self, host, port, grpc_port = 6334):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client = self.create_client()

    def create_client(self):
        client = QdrantClient(
            host=self.host, port=self.port
        )
        return client

    def create_collection(self, collection_name, vector_size):
        self.collection_name = collection_name
        self.dense_vector_config = {
            "dense": VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
        }
        self.sparse_vector_config = {
            "sparse": SparseVectorParams()
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.dense_vector_config,
            sparse_vectors_config=self.sparse_vector_config
        )

    def get_collections(self):
        return self.client.get_collections()

    @calc_latency
    def update_datapoints_with_payload(self, matched_metadata: dict):
        payload_operations = []
        for datapoint_id, payload in matched_metadata.items():
            payload_operations.append(
                SetPayloadOperation(
                    set_payload=SetPayload(
                        payload={"filename" : payload["filename"]},
                        points=[datapoint_id]
                    )
                )
            )
        if payload_operations:
            self.client.batch_update_points(
                collection_name=self.collection_name,
                update_operations=payload_operations,
                wait=True
            )
        logger.info(f" {len(payload_operations)} datapoints updated with payload.")

    def seggregate_chunks_ids(self, chunks_hash_ids: list) -> tuple:
        datapoints = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunks_hash_ids,
            with_payload=True
        )
        if not datapoints:
            return ({}, [])
        else:
            matched_ids = []
            # unmatched_ids = []
            datapoint_map = {}
            for datapoint in datapoints:
                datapoint_id = datapoint.id
                datapoint_map[datapoint_id] = datapoint.payload or {}
                if datapoint_id in chunks_hash_ids:
                    matched_ids.append(datapoint_id)
                # else:
                #     unmatched_ids.append(datapoint_id)
            # logger.info(f"Out of distinct {len(matched_ids)+len(unmatched_ids)}, {len(matched_ids)} chunks already exists in the {self.collection_name} collection.")
            return (datapoint_map, matched_ids)

    @calc_latency
    def eliminate_duplicate_chunks(self, chunks, datapoint_map: dict, chunks_hash_ids: list, matched_ids: list) -> list:
        matched_metadata = {}
        matched_duplicate_indices = {}
        for idx, (chunk_hash_id, chunk) in enumerate(zip(chunks_hash_ids, chunks)):
            if chunk_hash_id not in matched_ids:
                continue
            updated_metadata = {}

            # For tracing
            if chunk_hash_id in matched_duplicate_indices:
                matched_duplicate_indices[chunk_hash_id].append(idx)
            else:
                matched_duplicate_indices[chunk_hash_id] = [idx]
            
            filename = chunk.metadata["filename"]
            if isinstance(filename, str):
                filename = [filename]
            datapoint_payload = datapoint_map.get(chunk_hash_id, {})
            datapoint_filename = datapoint_payload.get("filename", "")
            datapoint_filename = datapoint_filename if isinstance(datapoint_filename, list) else [datapoint_filename]
            updated_metadata["filename"] = datapoint_filename + filename
            matched_metadata[chunk_hash_id] = updated_metadata

        # Updating existing datapoints with latest payload
        self.update_datapoints_with_payload(matched_metadata)

        # Removing chunks using their indices in descending order to avoid index issues 
        final_indices = []
        for hash_id, indices in matched_duplicate_indices.items():
            final_indices += indices
        final_indices.sort(reverse=True)
        for index in final_indices:
            chunks.pop(index)

        logger.info(f" {len(final_indices)} duplicate chunks are eliminated. Matched IDs : {matched_ids}. Indices : {final_indices}")
        return final_indices

    @calc_latency
    def create_data_points(self, chunks, chunks_dense_embeddings, chunks_sparse_embeddings, hash_preppend_str:str = "App:v1"):
        data_points: List[PointStruct] = []

        for idx, (chunk, dense_embedding, sparse_embedding) in enumerate(zip(chunks, chunks_dense_embeddings, chunks_sparse_embeddings)):
            now_utc = datetime.now(timezone.utc)
            data_point = PointStruct(
                id=str(uuid5(NAMESPACE_DNS, hash_preppend_str + chunk.page_content)),
                vector = {
                    "dense": dense_embedding.tolist(),
                    "sparse": {
                        "indices": sparse_embedding.indices.tolist(),
                        "values": sparse_embedding.values.tolist()
                    }
                },
                payload={
                    "page_content": chunk.page_content,
                    **chunk.metadata,
                    "chunk_id":idx,
                    "created_at": now_utc.isoformat()
                }
            )
            data_points.append(data_point)
        return data_points

    @calc_latency
    def insert_data_points(self, points):
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=16,
            parallel=2,
            max_retries=3,
            wait=True
        )

    @calc_latency
    def rerank_chunks(self, query, chunks, reranker_model: CrossEncoder):
        if not chunks.points:
            return
        query_chunk_pairs = [(query, chunk.payload.get("page_content","")) for chunk in chunks.points]
        relevancy_scores = reranker_model.predict(
            inputs=query_chunk_pairs,
            batch_size=16,
        )
        reranked_chunks = sorted(
            zip(chunks.points, relevancy_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return reranked_chunks

    # Hybrid search with BM25(Keyword search with sparse vectors) and vector search(semantic similarity with dense vectors)
    @calc_latency
    def get_topk_chunks(self, search_query, top_k, top_n, score_threshold = None):
        dense_query_embedding = dense_embedding_model.encode(search_query)
        sparse_query_embedding = next(sparse_embedding_model.query_embed(search_query))
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query_embedding.tolist(),
                    using="dense",
                    limit=top_n
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query_embedding.indices.tolist(),
                        values=sparse_query_embedding.values.tolist()
                    ),
                    using="sparse",
                    limit=top_n
                )
            ],
            query=FusionQuery(fusion='rrf'),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold
        )

        # Reranking the chunks according to their relevancy score using cross-encoder reranker
        reranked_results = self.rerank_chunks(search_query, search_results, reranker_model)

        outputs = []

        for result in reranked_results:
            data_point, relevancy_score = result
            outputs.append({
                "id": data_point.id,
                "page_content": data_point.payload.get("page_content",""),
                "semantic_score": data_point.score,
                "relevancy_score(reranker)": relevancy_score,
                "metadata": {
                    k: v for k,v in data_point.payload.items() if k != "page_content"
                }
            })
        return outputs