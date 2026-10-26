"""
Cara menjalankan:
  python mesin_pencari_asag.py                      # mode terminal, korpus di ./corpus
  python mesin_pencari_asag.py --folder corpus --lang en
  python mesin_pencari_asag.py --kueri "semantic similarity grading"
  streamlit run app_streamlit.py                    # antarmuka web Streamlit (bonus)
"""

import argparse
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Kata yang sering muncul di artikel ilmiah tetapi tidak membawa makna topik.
EXTRA_STOPWORDS = {
    "http", "https", "www", "doi", "org", "com", "pdf", "et", "al", "fig",
    "figure", "table", "pp", "vol", "no", "arxiv", "preprint", "ieee", "acm",
    "springer", "elsevier", "copyright", "license", "creative", "commons",
}

# pypdf mencetak banyak peringatan format PDF yang tidak memengaruhi hasil ekstraksi.
logging.getLogger("pypdf").setLevel(logging.ERROR)

def baca_berkas(path):
    """Baca satu berkas menjadi satu string panjang. Kembalikan None jika tipe tidak didukung."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[lewati] {os.path.basename(path)}: pasang pypdf (pip install pypdf) "
                  "atau ubah dulu ke .txt")
            return None
        reader = PdfReader(path)
        return "\n".join((hal.extract_text() or "") for hal in reader.pages)
    return None


def muat_korpus(folder):
    """Kembalikan list dokumen: kode (D01, D02, ...), nama berkas, judul, teks."""
    if not os.path.isdir(folder):
        sys.exit(f"Folder korpus '{folder}' tidak ditemukan.")
    nama_berkas = sorted(
        n for n in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, n)) and n.lower() != "readme.txt"
    )
    korpus = []
    for nama in nama_berkas:
        teks = baca_berkas(os.path.join(folder, nama))
        if not teks or not teks.strip():
            continue
        baris_pertama = next((b.strip() for b in teks.splitlines() if b.strip()), nama)
        korpus.append({
            "kode": f"D{len(korpus) + 1:02d}",
            "nama": nama,
            "judul": baris_pertama[:90],
            "teks": teks,
        })
    if not korpus:
        sys.exit(f"Tidak ada berkas .txt/.pdf yang terbaca di '{folder}'.")
    return korpus

def siapkan_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def buat_stemmer(lang):
    if lang == "id":
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # pip install PySastrawi
        return StemmerFactory().create_stemmer().stem
    return PorterStemmer().stem


class Praproses:
    """Objek yang bisa dipanggil: teks (str) -> list token bersih."""

    def __init__(self, lang="en"):
        # \w minus angka dan underscore = huruf unicode saja (angka dan tanda baca terbuang)
        self.tokenizer = RegexpTokenizer(r"[^\W\d_]+")
        nama_stop = "indonesian" if lang == "id" else "english"
        self.stop = set(stopwords.words(nama_stop)) | EXTRA_STOPWORDS
        self._stem = lru_cache(maxsize=None)(buat_stemmer(lang))  # cache: stemming mahal

    def __call__(self, teks):
        teks = teks.lower()                                   # case folding
        token = self.tokenizer.tokenize(teks)                 # tokenisasi + buang tanda baca
        token = [t for t in token if len(t) > 2 and t not in self.stop]  # stopword
        return [self._stem(t) for t in token]                 # stemming

@dataclass
class Indeks:
    kode: list            # urutan kode dokumen, mis. ['D01', 'D02', ...]
    vocab: list           # term terurut alfabet
    pos: dict             # term -> posisi kolom
    tf: dict              # kode -> Counter(term -> frekuensi mentah)
    df: Counter           # term -> jumlah dokumen yang memuat term
    idf: dict             # term -> bobot idf
    X: np.ndarray         # matriks tf-idf ternormalisasi, bentuk (dokumen, term)
    inverted: dict        # term -> {'df', 'idf', 'postings': {kode: tf}}


def bangun_indeks(token_dok):
    """token_dok: dict kode -> list token. Hitung tf, df, idf, tf-idf, lalu susun indeks."""
    kode = list(token_dok)
    n_dok = len(kode)

    tf = {k: Counter(t) for k, t in token_dok.items()}        # frekuensi term per dokumen
    df = Counter()                                            # document frequency
    for c in tf.values():
        df.update(c.keys())

    vocab = sorted(df)
    pos = {t: i for i, t in enumerate(vocab)}
    # idf versi "smooth": ln((1+N)/(1+df)) + 1, sama dengan default TfidfVectorizer
    idf = {t: math.log((1 + n_dok) / (1 + df[t])) + 1 for t in vocab}

    X = np.zeros((n_dok, len(vocab)))
    for i, k in enumerate(kode):
        for t, c in tf[k].items():
            X[i, pos[t]] = c * idf[t]                         # tf-idf = tf mentah * idf
    norma = np.linalg.norm(X, axis=1, keepdims=True)
    norma[norma == 0] = 1
    X = X / norma                                             # normalisasi L2 per dokumen

    inverted = {
        t: {"df": df[t], "idf": idf[t],
            "postings": {k: tf[k][t] for k in kode if t in tf[k]}}
        for t in vocab
    }
    return Indeks(kode, vocab, pos, tf, df, idf, X, inverted)


def tabel_frekuensi(idx):
    """DataFrame term x dokumen berisi frekuensi mentah, ditambah kolom DF."""
    tabel = pd.DataFrame(
        {k: dict(c) for k, c in idx.tf.items()}
    ).fillna(0).astype(int).reindex(idx.vocab)
    tabel["DF"] = [idx.df[t] for t in tabel.index]
    return tabel


def tabel_tfidf_transpose(idx):
    """Matriks tf-idf dengan dokumen sebagai kolom dan term sebagai baris."""
    dok_x_term = pd.DataFrame(idx.X, index=idx.kode, columns=idx.vocab)
    return dok_x_term.T                                       # transpose

def vektor_kueri(token, idx):
    """Bobot tf-idf kueri memakai idf korpus. Term yang tidak ada di korpus diabaikan."""
    q = np.zeros(len(idx.vocab))
    for t, c in Counter(token).items():
        j = idx.pos.get(t)
        if j is not None:
            q[j] = c * idx.idf[t]
    norma = np.linalg.norm(q)
    return q / norma if norma else q


def cari(kueri, idx, praproses):
    """Kembalikan (token kueri, [(kode, skor), ...]) terurut dari skor tertinggi."""
    token = praproses(kueri)
    q = vektor_kueri(token, idx)
    if not q.any():
        return token, []
    skor = cosine_similarity(q.reshape(1, -1), idx.X)[0]
    urut = np.argsort(-skor, kind="stable")
    return token, [(idx.kode[i], float(skor[i])) for i in urut if skor[i] > 0]

def cetak_hasil(hasil, info, topn):
    if not hasil:
        print("Tidak ada dokumen yang cocok. Coba kata kunci lain.\n")
        return
    print(f"{'Rank':<5}{'Kode':<6}{'Cosine':<10}Judul")
    print("-" * 80)
    for r, (kode, skor) in enumerate(hasil[:topn], start=1):
        print(f"{r:<5}{kode:<6}{skor:<10.4f}{info[kode]['judul']}")
    print()


def mode_terminal(idx, praproses, info, topn):
    print("Ketik kueri lalu Enter. Kosong atau 'keluar' untuk berhenti.\n")
    while True:
        try:
            kueri = input("Kueri> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not kueri or kueri.lower() in {"keluar", "exit", "quit"}:
            break
        token, hasil = cari(kueri, idx, praproses)
        print(f"Token kueri setelah praproses: {token}")
        cetak_hasil(hasil, info, topn)

# ---------------------------------------------------------------------------
# Program utama
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mesin pencari sederhana korpus artikel ASAG")
    ap.add_argument("--folder", default="corpus", help="folder berisi artikel (.txt/.pdf)")
    ap.add_argument("--lang", choices=["en", "id"], default="en",
                    help="bahasa artikel: en (default) atau id")
    ap.add_argument("--kueri", help="jalankan satu kueri lalu selesai")
    ap.add_argument("--topn", type=int, default=15, help="jumlah hasil yang ditampilkan")
    args = ap.parse_args()

    siapkan_nltk()
    korpus = muat_korpus(args.folder)
    print(f"Terbaca {len(korpus)} dokumen dari '{args.folder}'", end="")
    print("" if len(korpus) == 15 else "  (peringatan: tugas meminta 15 dokumen)")

    praproses = Praproses(args.lang)
    token_dok = {d["kode"]: praproses(d["teks"]) for d in korpus}
    info = {d["kode"]: d for d in korpus}

    print("\nDaftar dokumen:")
    for d in korpus:
        print(f"  {d['kode']}  {d['nama']}  ({len(token_dok[d['kode']])} token bersih)")

    idx = bangun_indeks(token_dok)
    print(f"\nIndeks selesai: {len(idx.kode)} dokumen, {len(idx.vocab)} term unik.")

    # Frekuensi + DF
    frek = tabel_frekuensi(idx)
    frek.to_csv("frekuensi_term.csv", encoding="utf-8-sig")
    print("\nFrekuensi term per dokumen (20 term dengan DF dan total tertinggi):")
    urut = frek.assign(TOTAL=frek.drop(columns="DF").sum(axis=1)) \
               .sort_values(["DF", "TOTAL"], ascending=False).head(20)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(urut.drop(columns="TOTAL"))

    # tf-idf (transpose: term = baris, dokumen = kolom)
    tfidf_t = tabel_tfidf_transpose(idx)
    tfidf_t.to_csv("tfidf_transpose.csv", encoding="utf-8-sig")
    print("\nMatriks tf-idf (transpose: baris = term, kolom = dokumen), 20 term berbobot total tertinggi:")
    teratas = tfidf_t.loc[tfidf_t.sum(axis=1).sort_values(ascending=False).head(20).index]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(teratas)
    print("\nMatriks lengkap tersimpan di frekuensi_term.csv dan tfidf_transpose.csv\n")

    if args.kueri:
        token, hasil = cari(args.kueri, idx, praproses)
        print(f"Token kueri setelah praproses: {token}")
        cetak_hasil(hasil, info, args.topn)
    else:
        mode_terminal(idx, praproses, info, args.topn)


if __name__ == "__main__":
    main()
