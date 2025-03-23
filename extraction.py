from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextBoxHorizontal, LTFigure, LTImage
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
import re
from doc2vec import doc2vec
from sbert import sbert
from nltk.corpus import stopwords



import spacy

nlp = spacy.load("en_core_web_lg")
def chunk_text_semantic(text, max_length=500):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        if len(current_chunk) + len(sent.text) <= max_length:
            current_chunk += " " + sent.text
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent.text
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



path = "487w25-syllabus.pdf"
text = extract_text(path)

# print(repr(text))
# text = re.sub(r'\s+', ' ', text).strip()
# # Remove special characters (optional)
# text = re.sub(r'[^\w\s.,;:!?]', '', text)
# # print(chunk_text_semantic(text))
# print(len(chunk_text_semantic(text)))
# print(chunk_text_semantic(text)[0])
# print("\n\n\n\n\n")
# print(chunk_text_semantic(text)[1])
# print("\n\n\n\n\n")
# print(chunk_text_semantic(text)[2])
# print("\n\n\n\n\n")
# print(chunk_text_semantic(text)[3])
# print("\n\n\n\n\n")
# print(chunk_text_semantic(text)[4])
# print("\n\n\n\n\n")
# print(chunk_text_semantic(text)[5])
# print(word_tokenize(chunk_text_semantic(text)[5]))

# print(chunk_text_semantic(chunk_text_semantic(text)[4]))/

# pdf_files = []
# for root, dirs, files in os.walk("."):
#     for file in files:
#         if file.endswith(".pdf"):
#             pdf_files.append(os.path.join(root, file))

# chunks = []
# for file in pdf_files:
#     text = extract_text(file)
#     chunks.extend(chunk_text_semantic(text))
# stopwords = set(stopwords.words('english'))


chunks = chunk_text_semantic(text)
query = nlp("who is the professor")
best_score = -1
best_chunk = None
for chunk in chunks:
    score = (nlp(chunk)).similarity(query)
    if score > best_score:
        best_score = score
        best_chunk = chunk

print("Best Score: ", best_score, " | Best Chunk ", best_chunk)
# new_chunks = []
# for chunk in chunks:
#     preprocess = []
#     for word in chunk.split():
#         if word.lower() not in stopwords:
#             preprocess.append(word.lower())
#     new_chunks.append(" ".join(preprocess))


# # chunks = [chunk.lower() for chunk in chunks]

# sbert(new_chunks, ["who is the professor"])


# model = doc2vec(chunks)
# print(model.dv)

# query = ["what is the purpose of this course"]

# inferred_vec = model.infer_vector(query)

# most_sim = model.dv.most_similar([inferred_vec], topn=10)
# print(most_sim)

# for idx, _ in most_sim:
#     print(chunks[int(idx)])
#     print("\n\n\n\n")


# from sklearn.metrics.pairwise import cosine_similarity
# from tfidf import tfidf

# def find_top_k_similar(query, chunks, model, vectorizer, k=5):
#     # Transform the query into its TF-IDF representation
#     query_vec = vectorizer.transform([query])
    
#     # Compute cosine similarity between the query and each chunk
#     similarities = cosine_similarity(query_vec, model).flatten()
    
#     # Get the indices of the top K most similar chunks
#     top_k_indices = similarities.argsort()[-k:][::-1]
    
#     # Return the top K most similar chunks and their similarity scores
#     top_k_chunks = [(chunks[i], similarities[i]) for i in top_k_indices]
    
#     return top_k_chunks
# # Get the TF-IDF model and vectorizer
# model, vectorizer = tfidf(chunks)

# # Define the query
# query = "who is the professor"

# # Find the top 2 most similar chunks
# top_k_similar = find_top_k_similar(query, chunks, model, vectorizer, k=10)

# # Print the results
# for chunk, score in top_k_similar:
#     print(f"Chunk: {chunk}\nSimilarity Score: {score}\n")