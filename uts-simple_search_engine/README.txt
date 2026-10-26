TUGAS PEMROGRAMAN KELOMPOK II - NATURAL LANGUAGE PROCESSING (TI6073)
Mesin Pencari Sederhana - Korpus Artikel "Automated Short Answer Grading" (ASAG)

ANGGOTA KELOMPOK
1. NIM: ........  Nama: ........................
2. NIM: ........  Nama: ........................
3. NIM: ........  Nama: ........................

BAHASA KORPUS : Inggris (semua 15 artikel berbahasa Inggris)

--------------------------------------------------------------------
SUMBER ARTIKEL (URL)
Simpan tiap artikel sebagai .txt di folder corpus/ (boleh juga .pdf).
Kode dokumen (D01, D02, ...) dibuat otomatis berdasarkan urutan nama berkas.
Anggota: tulis di kolom terakhir siapa yang mengumpulkan (5 artikel per anggota).
--------------------------------------------------------------------
01. Kara - Unveiling the Landscape: Studies on Automated Short Answer Evaluation
    https://asianjde.com/ojs/index.php/AsianJDE/article/download/768/427
02. Casalino dkk - Framing automatic grading techniques for open-ended questionnaires responses. A short survey
    https://ceur-ws.org/Vol-3025/paper5.pdf
03. Literature survey ASAG (Atlantis Press)
    https://www.atlantis-press.com/article/125960808.pdf
04. Henkel dkk - Can LLMs Grade Short-Answer Reading Comprehension Questions
    https://arxiv.org/abs/2310.18373
05. Selvi & Banerjee - Automatic Short-Answer Grading System (ASAGS)
    https://arxiv.org/pdf/1011.1742
06. Automatic Short Answer Grading Using Paragraph Vectors (journals.kau.edu.sa)
    https://journals.kau.edu.sa/index.php/CITS/article/view/171/186
07. Odumuyiwa dkk - A Comparative Evaluation of Embedding Techniques from Language Models for Automatic Grading of Short Answer Questions
    https://journals.ui.edu.ng/index.php/uijslictr/article/download/1563/1265
08. Automatic Short Answer Grading with Rubric-based Semantic Embedding Optimization (IPSJ)
    https://ipsj.ixsq.nii.ac.jp/record/216464/files/IPSJ-ICS22205011.pdf
09. Short answer grading with sentence similarity and a few given grades (EDM 2025)
    https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.124/2025.EDM.short-papers.124.pdf
10. Semi-supervised clustering for input in automated short answer grading (Inderscience, open access)
    https://inderscience.com/filter.php?aid=124160
11. Situmeang dkk - Indonesian automated short-answer grading using transformers-based semantic similarity
    https://ijict.iaescore.com/index.php/IJICT/article/download/21350/13218
12. Kortemeyer - Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading
    https://arxiv.org/pdf/2309.09338
13. Thakkar - Finetuning Transformer Models to Build ASAG System
    https://arxiv.org/pdf/2109.12300
14. Using BERT and XLNET for the Automatic Short Answer Grading Task (SciTePress)
    https://www.scitepress.org/Papers/2020/94224/94224.pdf
15. Metzler dkk - Computer-Assisted Short Answer Grading Using Large Language Models and Rubrics
    https://dl.gi.de/handle/20.500.12116/45094

Cadangan bila ada tautan yang gagal dibuka:
- Luhurkinanti dkk - Japanese Short Answer Grading ... Contextual Representation of BERT (IEEE Access 2025)
  https://doi.org/10.1109/ACCESS.2025.3532659
- Lu dkk - Survey and analysis for the challenges in computer science to the automation of grading systems (ACM CSUR 2025)
  https://wrap.warwick.ac.uk/193086

--------------------------------------------------------------------
KEBUTUHAN DAN CARA MENJALANKAN
--------------------------------------------------------------------
Python 3.9 atau lebih baru.

1. Pasang pustaka:
     pip install nltk pandas numpy scikit-learn pypdf
   (pypdf hanya perlu bila korpus berupa .pdf)
   Untuk korpus bahasa Indonesia, tambah: pip install PySastrawi
2. Stopword NLTK diunduh otomatis saat program pertama kali berjalan.
   Bila gagal, jalankan manual:
     python -c "import nltk; nltk.download('stopwords')"
3. Letakkan 15 artikel di folder corpus/ (sejajar dengan mesin_pencari_asag.py).
4. Jalankan:
     python mesin_pencari_asag.py                 mode terminal, interaktif
     python mesin_pencari_asag.py --kueri "semantic similarity"
     python mesin_pencari_asag.py --gui           antarmuka grafis (bonus)
     python mesin_pencari_asag.py --lang id       bila korpus berbahasa Indonesia

Luaran:
- Terminal: daftar dokumen, tabel frekuensi + DF, matriks tf-idf (transpose),
  hasil pencarian terurut berdasarkan Cosine Similarity.
- frekuensi_term.csv  : frekuensi mentah tiap term per dokumen + DF.
- tfidf_transpose.csv : matriks tf-idf, baris = term, kolom = dokumen.

--------------------------------------------------------------------
CATATAN METODE
--------------------------------------------------------------------
Praproses  : case folding -> tokenisasi (RegexpTokenizer, huruf saja) -> hapus
             stopword (NLTK + daftar tambahan istilah artikel) -> stemming
             (Porter untuk Inggris, Sastrawi untuk Indonesia).
Pembobotan : tf = frekuensi mentah, idf = ln((1+N)/(1+df)) + 1, vektor dokumen
             dinormalisasi L2 (sama dengan default TfidfVectorizer sklearn).
Indeks     : dictionary term -> {df, idf, postings{kode dokumen: tf}}.
Kueri      : frasa pengguna dipraproses dengan jalur yang sama, dibobot dengan
             tf kueri * idf korpus, lalu dibandingkan dengan Cosine Similarity.
             Term kueri yang tidak ada di korpus diabaikan.
