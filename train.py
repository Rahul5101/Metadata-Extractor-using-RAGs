

from loaders.document_loaders import load_documents_from_folder
from splitters.text_splitter import split_documents
from vectorstore.build_vectorstore import build_faiss_vector


def store_train(train_folder):
    docs = load_documents_from_folder(train_folder) # we  docs which has all text from docs and png
    chunks = split_documents(docs) # and here we split the document into smaller chunks
    vectordb = build_faiss_vector(chunks) # we generate the embeddings and build the vectorstores from the chunks
   


if __name__ == "__main__":
    store_train("data/train")
