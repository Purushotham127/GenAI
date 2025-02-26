# -*- coding: utf-8 -*-
"""RAG_load_retrieve.ipynb

RAG implementation using ollama model(custom model without openai endpoint)

Original file is located at
    https://colab.research.google.com/drive/1QJVqxZvlDo-nUxyj25Me36yAJZdt6gn9
"""
# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

def get_pdf_data(pdf_path):
  pdf_loader = PyPDFLoader(pdf_path)
  pdf_text = pdf_loader.load()
  text_data = [doc.page_content for doc in pdf_text]
  return text_data

def ingest_data(text_data,vector_store_path):
  text = " ".join(text_data)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
  chunks = text_splitter.split_text(text)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #example model : Metric-AI/armenian-text-embeddings-1
  if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(vector_store_path,embeddings=embeddings,allow_dangerous_deserialization=True)
    vector_store.add_texts(chunks,embeddings=embeddings)
    vector_store.save_local(vector_store_path)
  else:
    vector_store = FAISS.from_texts(chunks,embedding=embeddings)
    vector_store.save_local(vector_store_path)

def chunks_for_query(query):
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vector_store = FAISS.load_local(vector_store_path,embeddings=embeddings,allow_dangerous_deserialization=True)
  retrieved_chunks = vector_store.similarity_search(query,k=5)
  return retrieved_chunks

def get_ollama_llm():
  return Ollama()
  # bert_pipeline = pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")
  # return HuggingFacePipeline(pipeline=bert_pipeline)
def get_conversational_chain():
  rag_prompt = """
  Give me the correct response for the question asked using provided context.
  Context:{context}
  question: {question}
  answer:
  """
  final_rag_prompt = PromptTemplate(template=rag_prompt,input_variables=["context","question"])
  chain = load_qa_chain(get_ollama_llm(),chain_type="stuff",prompt=final_rag_prompt)
  return chain

pdf_path = "temp_data\\temp1.pdf"
vector_store_path = "faiss_index"

input_value = input("Enter 1 for ingest, 2 for query : ")
if input_value == "1":
  text_data = get_pdf_data(pdf_path)
  ingest_data(text_data,vector_store_path)
elif input_value == "2":
  query = input("Enter your query: ")
  retrieved_chunks = chunks_for_query(query)
  # for index in range(len(retrieved_chunks)):
  #   print(f'chunk{index+1}: {retrieved_chunks[index]}')
  chain = get_conversational_chain()
  response = chain({"question": query, "input_documents": retrieved_chunks}, return_only_outputs=True)
  print(response)
else:
  print("Invalid input")
