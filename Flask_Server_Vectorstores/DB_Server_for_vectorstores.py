from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from flask import Flask, request, jsonify
app = Flask(__name__)

def chunks_for_query(query, database_name, no_of_chunks):
    database_path = os.path.join(os.curdir,database_name)
    if(os.path.exists(database_path) and len(os.listdir(database_path))>1):
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
        chunks = vectorstore.similarity_search(query,k=no_of_chunks)
        chunk_dicts = [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
        return chunk_dicts
    else:
        return []

@app.route('/process', methods=['GET'])
def process_strings():
    SSK = request.args.get('SSK')
    database_name = request.args.get('database_name')
    no_of_chunks = int(request.args.get('no_of_chunks'))
    print(f"database: {database_name} is selected.")
    result_list = chunks_for_query(SSK, database_name, no_of_chunks)
    return jsonify(result_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)