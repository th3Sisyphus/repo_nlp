from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
idn_stemmer = factory.create_stemmer()

corpus = [
    'Arsitektur Dalam Perubahan Kebudayaan',
    'perubahan arsitektur sebagai bagian dari perubahan kebudayaan pada masyarakat tertent',
    'Arsitektur sebagai artefak kebudayaan berperan penting dalam memberikan ciri kebudayaan setempat'
]

clean_corpus = []
for dok in corpus:
    content = idn_stemmer.stem(str.lower(dok))
    clean_corpus.append(content)

count_vec = CountVectorizer()

one_HEVect = count_vec.fit_transform(clean_corpus)

# 2. Perbaiki get_feature_names() menjadi get_feature_names_out()
tok = count_vec.get_feature_names_out()

# 3. Perbaiki tanda petik miring pada index
df_countvect = pd.DataFrame(
    data=one_HEVect.toarray(),
    index=['Doc1', 'Doc2', 'Doc3'],
    columns=tok
)

print("One-Hot / Bag-of-Words Encoded Vector:\n")
print(df_countvect)
