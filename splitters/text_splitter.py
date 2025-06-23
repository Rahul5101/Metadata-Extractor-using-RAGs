from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from loaders.document_loaders import load_documents_from_folder

def split_documents(documents: List[Document], chunk_size: int = 550, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)



# if __name__ == "__main__":

#     docs = load_documents_from_folder("data/test")
#     chunks = split_documents(docs)

#     print(f"total chunks: {len(chunks)}")
#     # print(chunk[0].page_content)
#     # print(chunk[0].metadata)
    
#     for i, chunk in enumerate(chunks[10:21]): # Displaying first 10 chunks
#         print(f"\n--- Chunk {i+1} ---\n{chunk.page_content[:300]}...")