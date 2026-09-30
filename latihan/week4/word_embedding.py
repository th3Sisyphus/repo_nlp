from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

raw_corpus = [
    "Pada mulanya adalah Firman; Firman itu bersama-sama dengan Allah dan Firman itu adalah Allah.",
    "Ia pada mulanya bersama-sama dengan Allah.",
    "Segala sesuatu dijadikan oleh Dia dan tanpa Dia tidak ada suatupun yang telah jadi dari segala yang telah dijadikan."]
#Praprosess (lowercase, tokenize, remove punctuation)
sentences = [simple_preprocess(doc) for doc in raw_corpus]
# Inisiasi model dan melakukan pelatihan (traiining)
model = Word2Vec(sentences, vector_size=20,
                window=5, min_count=1, workers=2)
word_vector = model.wv['firman']
print(word_vector)
