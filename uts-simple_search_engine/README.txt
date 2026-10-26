TUGAS PEMROGRAMAN KELOMPOK II
Mesin Pencari Sederhana - Korpus Artikel "Automated Short Answer Grading" (ASAG)

ANGGOTA KELOMPOK
1. NIM: 71230994  Nama: Putu Gde Kenzie Carlen Mataram
2. NIM: 71230975  Nama: Thomas Aquinas Ryan Wisnu Adi
3. NIM: 71231023  Nama: Yehezkiel Darren Putra Wardoyo

--------------------------------------------------------------------
KEBUTUHAN DAN CARA MENJALANKAN
--------------------------------------------------------------------
Python 3.9 atau lebih baru.

1. Pasang pustaka:
    pip install requirements.txt
2. Stopword NLTK diunduh otomatis saat program pertama kali berjalan. Bila gagal, jalankan manual:
    python -c "import nltk; nltk.download('stopwords')"
3. Letakkan 15 artikel di folder corpus/ (sejajar dengan mesin_pencari_asag.py).
4. Jalankan:
    python mesin_pencari_asag.py                 mode terminal, interaktif
    python mesin_pencari_asag.py --kueri "semantic similarity"
    python mesin_pencari_asag.py --gui           antarmuka grafis (bonus)

Luaran:
- Terminal: daftar dokumen, tabel frekuensi + DF, matriks tf-idf (transpose),
  hasil pencarian terurut berdasarkan Cosine Similarity.
- frekuensi_term.csv  : frekuensi mentah tiap term per dokumen + DF.
- tfidf_transpose.csv : matriks tf-idf, baris = term, kolom = dokumen.

--------------------------------------------------------------------
DAFTAR JURNAL YANG DIGUNAKAN (judul artikel - link jurnal)
Bahasa korpus: Inggris (15 artikel)
--------------------------------------------------------------------
01. Computer-Assisted Short Answer Grading Using Large Language Models and Rubrics - https://dl.gi.de/handle/20.500.12116/45094
02. Using BERT and XLNET for the Automatic Short Answer Grading Task - https://www.scitepress.org/Papers/2020/94224/94224.pdf
03. Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading - https://arxiv.org/pdf/2309.09338
04. Indonesian automated short-answer grading using transformers-based semantic similarity - https://ijict.iaescore.com/index.php/IJICT/article/download/21350/13218
05. Short answer grading with sentence similarity and a few given grades - https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.124/2025.EDM.short-papers.124.pdf
06. Automatic Short Answer Grading with Rubric-based Semantic Embedding Optimization - https://ipsj.ixsq.nii.ac.jp/record/216464/files/IPSJ-ICS22205011.pdf
07. A Comparative Evaluation of Embedding Techniques from Language Models for Automatic Grading of Short Answer Questions - https://journals.ui.edu.ng/index.php/uijslictr/article/download/1563/1265
08. Automatic Short-Answer Grading System (ASAGS) - https://arxiv.org/pdf/1011.1742
09. Recent Trends in Answer Script Evaluation - A Literature Survey - https://www.atlantis-press.com/article/125960808.pdf
10. Grade Guard: A Smart System for Short Answer Automated Grading - https://arxiv.org/pdf/2504.01253
11. Wisdom of Students: A Consistent Automatic Short Answer Grading Technique - https://aclanthology.org/W16-6324/
12. Get IT Scored Using AutoSAS - An Automated System for Scoring Short Answers - https://arxiv.org/pdf/2012.11243
13. Domain-Specific Hybrid BERT based System for Automatic Short Answer Grading - https://asianssr.org/index.php/ajct/article/view/1225/903
14. Fast and Easy Short Answer Grading with High Accuracy - https://aclanthology.org/N16-1123/
15. Japanese Short Answer Grading for Japanese Language Learners Using the Contextual Representation of BERT - https://doaj.org/article/df3e04cf39464a0da6bb2202e1b888ef