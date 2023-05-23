from pathlib import Path
from typing import List, Tuple

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain

# Constants
local_path = "./models/gpt4all-converted.bin"
model_path = "./models/ggml-model-q4_0.bin"
docs_folder = "./docs"
index_path = "./index-thing"

def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(docs_folder, file_extension=".txt")
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path, n_ctx=2048, verbose=True)

embeddings = initialize_embeddings()

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)

# Chatbot loop
chat_history = []
print(".")
while True:
    query = input("Please enter your question: ")
    
    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])