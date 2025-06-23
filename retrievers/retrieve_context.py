from langchain_community.vectorstores import FAISS


def get_top_k_context(query: str, vectordb: FAISS, k: int = 3):
    return vectordb.similarity_search(query, k=k)
