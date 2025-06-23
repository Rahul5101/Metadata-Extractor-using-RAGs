
# Metadata Extraction with LangChain + RAG

## Description
This project extracts metadata from .docx and .png agreement documents using LangChain, RAG architecture, and structured prompting.

## The AI/ML system should be able to extract the following fields from the documents
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two


## Solution Apprach

### 1. **Document Loading** (`document_loader.py`)
- For `.docx` files: Text extraction using `python-docx`
- For `.png` files: used UnstructuredImageLoader to extract text in document format

### 2. **Text Splitting** (`text_splitter.py`)
- Document chunking for vector storage

### 3. **Vector Store Management** (`build_vectorstore.py`)
- Embedding generation using HuggingFace models
- FAISS vector database for similarity search
- Context retrieval for RAG

### 4. **Metadata Extraction** (`llm_chain.py`)
- use promptemplate and output parser to define the schema
- use gemini-1.5-flash-latest model for efficient processing
- RAG approach with context-aware extraction
- Fallback to pattern-based extraction

### 5. **Evaluation** (`evaluate.py`)
- Comprehensive calculation of csv file and json file
- Evaulate the par field Recall
- Recall = (true)/(true + false)

### 6. **Pipeline Orchestration** (`rag_pipeline.py`)
- Main pipeline coordination
- Metadata stored in prediction.json file

### **RESULT**
 **Per-field Recall Scores:**
   - agreement_start_date: 0.25
   - agreement_end_date: 0.5
   - renewal_notice_days: 0.75
   - party_one: 0.25
   - party_two: 0.25


 **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
### **RESULT**
-  **Per-field Recall Scores:**
   agreement_start_date: 0.25
   agreement_end_date: 0.5
   renewal_notice_days: 0.75
   party_one: 0.25
   party_two: 0.25


## How to Run
1. Install requirements
2. Run `train.py` to store the embedding
3. Run `rag_pipeline.py` to extract the metadata from test folder

This will:
1. Load training data and create the vector store
2. Process test documents
3. Extract metadata using BERT QA models
4. Calculate and display evaluation metrics
5. Save results to `predictions.json`
