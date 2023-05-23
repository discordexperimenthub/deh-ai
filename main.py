from pathlib import Path
from typing import List, Tuple

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Constants
local_path = "./models/gpt4all-converted.bin"
model_path = "./models/ggml-model-q4_0.bin"
docs_folder = "./docs"
index_path = "./index-thing"

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(docs_folder, file_extension=".txt")
    return loader.load()

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def main():
    embeddings = initialize_embeddings()
    sources = load_documents()
    chunks = split_chunks(sources)

    vectorstore = generate_index(chunks, embeddings)
    vectorstore.save_local(index_path)

    llm = LLMChain()

    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), max_tokens_limit=400)

    chat_history = []
    print("Welcome Cum Experiment Hub")
    while True:
        query = input("Give me ur seme sorry sorry just give ur question for discord api ")

        if query.lower() == 'exit':
            break

        result = qa({"question": query, "chat_history": chat_history})

        print("Answer:", result['answer'])

if __name__ == "__main__":
    main()
