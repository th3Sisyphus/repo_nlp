import os, nltk, re
import pandas as pd
from nltk.corpus import stopwords

root = r"D:\Private\Coding\repo_nlp\latihan\dokumen_contoh"
docs = dict()
stop_words = set(stopwords.words('indonesian'))

# Preprocessing and read files
for path, subdirs, files in os.walk(root):
    for name in files :
        fname = os.path.join(path, name)
        fname_1 = fname.split("\\")[-1] # ambil nama file
        # print(fname)
        with open(fname, 'r', encoding = 'utf-8') as f:
            fcontent = f.read()
            fcontent = re.sub(r'[\W\s(0-9)]+', ' ', fcontent.lower())
            tokens = nltk.word_tokenize(fcontent)
            clean_text=[w for w in tokens if not w in stop_words]
            term = nltk.FreqDist(clean_text)
            docs[fname_1]= term
            tokens, clean_text, term =[],[], []

print(docs)

# Visualisasi
# import matplotlib.pyplot as plt
# for filename in docs:
#     print(filename, " :\n")
#     keys, values = [], []
#     # menampilkan hasil dgn sortir term berdasarkan ranking tertinggi
#     for key, value in docs[filename].most_common(25):
#         print(u'{}:{}'. format(key, value)) # pengaturan format cetak
#         keys.append(key)
#         values.append(value)
#     plt.plot(keys, values)
#     plt.xticks(rotation=65)
#     plt.suptitle(filename)
#     plt.show()
#     keys, values = [], []

# WORD2VEC
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

model = Word2Vec(docs, vector_size=20,
                window=5, min_count=1, workers=2)
word_vector = model.wv['gibran']
print(word_vector)
