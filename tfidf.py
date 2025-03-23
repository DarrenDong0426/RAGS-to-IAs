from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(chunks):
    vectorizer = TfidfVectorizer()
    model = vectorizer.fit_transform(chunks)
    return model, vectorizer
