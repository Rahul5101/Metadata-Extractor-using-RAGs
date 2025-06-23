from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from loaders.document_loaders import load_documents_from_folder
# from splitters.text_splitter import split_documents


def build_faiss_vector(docs: List[Document], persist_dir = "vectorstore") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(persist_dir)
    return vectordb


def load_faiss_vector(persist_dir = "vectorstore") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)


# if __name__ == "__main__":
#     docs = load_documents_from_folder("data/test")
#     chunks = split_documents(docs)
    
#     print("Building FAISS vector store...")
#     build_faiss_vectorstore(chunks, persist_dir="vectorstore")


#     print("Loading FAISS vector store...")
#     vectordb = load_faiss_vectorstore("vectorstore")

#     query = "What is the agreement start date?"

#     results = vectordb.similarity_search(query, k=3)


#     print(f"\nTop 3 results for query: {query}")
#     for i, res in enumerate(results):
#         print(f"\n--- Result {i+1} ---\n{res.page_content[:300]}...")
