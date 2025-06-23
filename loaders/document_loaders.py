import os
from typing import List
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.schema import Document
from PIL import Image
import pytesseract


def load_docx_file(path) -> List[Document]:
    loader = Docx2txtLoader(path)
    return loader.load()


def load_png_file(path) -> List[Document]:
    # text = pytesseract.image_to_string(Image.open(path))
    # return [Document(page_content=text, metadata={"source":path})]
    loader = UnstructuredImageLoader(path)
    return loader.load()



def load_documents_from_folder(folder_path) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".docx"):
            documents.extend(load_docx_file(file_path))
        elif filename.endswith(".png"):
            documents.extend(load_png_file(file_path))
    return documents


if __name__ == "__main__":
    folder_path = "data/test"  # or give full path if needed
    docs = load_documents_from_folder(folder_path)
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(doc.page_content)