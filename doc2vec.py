from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

def doc2vec(chunks):
    # Preprocess documents
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) 
                for i, doc in enumerate(chunks)]

    # Train Doc2Vec model
    model = Doc2Vec(vector_size=300, min_count=1, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model

