import os, nltk, re
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec

root = r"D:\Private\Coding\repo_nlp\latihan\dokumen_contoh"
docs = dict()
sentences = []  # List untuk menampung token per dokumen untuk Word2Vec
stop_words = set(stopwords.words('indonesian'))

# Preprocessing and read files
for path, subdirs, files in os.walk(root):
    for name in files:
        fname = os.path.join(path, name)
        fname_1 = fname.split("\\")[-1]  # ambil nama file

        with open(fname, 'r', encoding='utf-8') as f:
            fcontent = f.read()
            fcontent = re.sub(r'[\W\s(0-9)]+', ' ', fcontent.lower())
            tokens = nltk.word_tokenize(fcontent)
            clean_text = [w for w in tokens if not w in stop_words]

            # 1. Simpan FreqDist di dict docs (jika masih butuh untuk perhitungan frekuensi)
            term = nltk.FreqDist(clean_text)
            docs[fname_1] = term

            # 2. Simpan list of words ke sentences jika dokumen tidak kosong
            if clean_text:
                sentences.append(clean_text)

print("Jumlah dokumen terproses:", len(sentences))

# --- WORD2VEC ---
# Masukkan 'sentences' (list of lists) ke Word2Vec
model = Word2Vec(sentences=sentences, vector_size=20, window=5, min_count=1, workers=2)

# Cek apakah kata ada di dalam vocab sebelum mengambil vector
target_word = input('target kata: ')
if target_word in model.wv:
    word_vector = model.wv[target_word]
    print(f"\nVector untuk kata '{target_word}':\n", word_vector)
else:
    print(f"\nKata '{target_word}' tidak ditemukan dalam dataset dokumen.")
