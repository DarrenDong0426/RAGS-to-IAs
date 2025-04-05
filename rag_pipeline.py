import os
from typing import Optional, List

from pdfminer.high_level import extract_text
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import google.generativeai as genai

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GOOGLE_API_KEY = "AIzaSyDuDWi3ZMs4p2G08HzlIX0aMjAEtinUWP8"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash-001')

def chunk_and_retrieve():
    RAW_KNOWLEDGE_BASE = []
    data_paths = ["487w25-syllabus.pdf"]

    for data_path in data_paths:
        RAW_KNOWLEDGE_BASE.append(LangchainDocument(extract_text(data_path)))

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"

    def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique


    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    return embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed

def retrieve_info(user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed):
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    return retrieved_docs

def generate_rag_response(query, retrieved_docs, model):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
    You are an assistant that answers questions based on the provided information. 
    Use only the following context to answer the question. If you don't know the answer based on the context, 
    say "I don't have enough information to answer this question."

    CONTEXT: {context}

    QUESTION: {query}

    ANSWER:
    """
    config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 1,
        "max_output_tokens": 100
    }
    
    response = model.generate_content(prompt, generation_config=config)
    return response.text