# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# def compressed_context(query: str, vectordb: FAISS, k: int = 3):
#     base_retriever = vectordb.as_retriever(search_kwargs={"k": k})

#     compressor_llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     google_api_key=os.getenv("GOOGLE_API_KEY"))

#     compressor= LLMChainExtractor.from_llm(compressor_llm)

#     compressed_retriever= ContextualCompressionRetriever(
#         base_compressor =compressor,
#         base_retriever= base_retriever
#     )

#     return compressed_retriever.invoke(query)

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.llms import HuggingFaceHub
import os
from langchain_huggingface import HuggingFaceEndpoint

def compressed_context(query: str, vectordb: FAISS, k: int = 3):
    base_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    

    compressor_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


    compressor = LLMChainExtractor.from_llm(compressor_llm)

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compressed_retriever.invoke(query)

