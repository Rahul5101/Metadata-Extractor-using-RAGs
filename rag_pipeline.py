from loaders.document_loaders import load_documents_from_folder
from splitters.text_splitter import split_documents
from vectorstore.build_vectorstore import load_faiss_vector
from retrievers.retrieve_context import compressed_context
from chains.llm_chain import extract_metadata


def run_testing(test_folder):
    test_docs = load_documents_from_folder(test_folder)
    vectordb = load_faiss_vector()

    results = []
    for doc in test_docs:
        chunks = split_documents([doc])
        
        BATCH_SIZE = 10   # number of chunks to combine per LLM call

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunk = chunks[i:i + BATCH_SIZE]
            combined_text = "\n\n".join([chunk.page_content for chunk in batch_chunk])

            top_context = compressed_context(combined_text, vectordb)
            prediction = extract_metadata(top_context)
            results.append(prediction)

    return results


if __name__ == "__main__":
    import json
    results = run_testing("data/test")
    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=4)
