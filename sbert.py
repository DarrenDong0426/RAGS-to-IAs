from sentence_transformers import SentenceTransformer
import numpy as np


def sbert(chunks, query):
    model = SentenceTransformer("all-MiniLM-L6-v2")


    # Compute embeddings for both lists
    embeddings1 = model.encode(chunks)
    embeddings2 = model.encode(query)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    print(similarities.shape)
    print(similarities)

    # similarities[0] = sorted(similarities[0], reverse=True)

    idx = np.argmax(similarities)
    print(chunks[idx])
    print(similarities[idx])

    # # Output the pairs with their score
    # for idx_i, sentence1 in enumerate(chunks):
    #     print(sentence1)
    #     for idx_j, sentence2 in enumerate(query):
    #         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")