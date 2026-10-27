"""
Antarmuka web (Streamlit) untuk mesin pencari korpus artikel ilmiah ASAG.
Menggunakan Vector Space Model, pembobotan TF-IDF, dan Cosine Similarity.
"""

import html
import os
import random
import re
import sys
import urllib.parse

import altair as alt
import pandas as pd
import streamlit as st

from mesin_pencari_asag import (
    Praproses, bangun_indeks, cari, muat_korpus, siapkan_nltk,
    tabel_frekuensi, tabel_tfidf_transpose,
)

FOLDER_INI = os.path.dirname(os.path.abspath(__file__))
BAHASA = "en"  # artikel pada korpus berbahasa Inggris
ANGGOTA = [
    ("71230994", "Putu Gde Kenzie Carlen Mataram"),
    ("71230975", "Thomas Aquinas Ryan Wisnu Adi"),
    ("71231023", "Yehezkiel Darren Putra Wardoyo"),
]

def pastikan_folder_static():
    """Memastikan folder static tersedia dan terhubung ke corpus untuk static file serving Streamlit."""
    folder_static = os.path.join(FOLDER_INI, "static")
    folder_corpus = os.path.join(FOLDER_INI, "corpus")

    if not os.path.exists(folder_static):
        try:
            if sys.platform == "win32":
                import subprocess
                subprocess.run(
                    f'cmd /c mklink /J "{folder_static}" "{folder_corpus}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.symlink(folder_corpus, folder_static, target_is_directory=True)
        except Exception:
            pass

        # Fallback jika symlink/junction tidak diizinkan oleh OS: salin berkas
        if not os.path.exists(folder_static):
            try:
                import shutil
                shutil.copytree(folder_corpus, folder_static)
            except Exception:
                pass

pastikan_folder_static()

# Pemetaan metadata resmi artikel (judul baku dan tautan publikasi ke jurnal/arXiv)
DOKUMEN_INFO = {
    "1011.1742v1.pdf": {
        "judul": "Automatic Short-Answer Grading System (ASAGS)",
        "url": "https://arxiv.org/abs/1011.1742",
    },
    "1225-Article Text-2315-1-10-20220818.pdf": {
        "judul": "Domain-Specific Hybrid BERT based System for Automatic Short Answer Grading",
        "url": "https://asianssr.org/index.php/ajct/article/view/1225/903",
    },
    "125960808.pdf": {
        "judul": "Recent Trends in Answer Script Evaluation - A Literature Survey",
        "url": "https://www.atlantis-press.com/article/125960808.pdf",
    },
    "2012.11243v1.pdf": {
        "judul": "Get IT Scored Using AutoSAS - An Automated System for Scoring Short Answers",
        "url": "https://arxiv.org/abs/2012.11243",
    },
    "2025.EDM.short-papers.124.pdf": {
        "judul": "Short answer grading with sentence similarity and a few given grades",
        "url": "https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.124/2025.EDM.short-papers.124.pdf",
    },
    "21350-42242-1-PB.pdf": {
        "judul": "Indonesian automated short-answer grading using transformers-based semantic similarity",
        "url": "https://ijict.iaescore.com/index.php/IJICT/article/download/21350/13218",
    },
    "2309.09338v1.pdf": {
        "judul": "Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading",
        "url": "https://arxiv.org/abs/2309.09338",
    },
    "2504.01253v1.pdf": {
        "judul": "Grade Guard: A Smart System for Short Answer Automated Grading",
        "url": "https://arxiv.org/abs/2504.01253",
    },
    "94224.pdf": {
        "judul": "Using BERT and XLNET for the Automatic Short Answer Grading Task",
        "url": "https://www.scitepress.org/Papers/2020/94224/94224.pdf",
    },
    "IPSJ-ICS22205011.pdf": {
        "judul": "Automatic Short Answer Grading with Rubric-based Semantic Embedding Optimization",
        "url": "https://ipsj.ixsq.nii.ac.jp/record/216464/files/IPSJ-ICS22205011.pdf",
    },
    "Japanese_Short_Answer_Grading_for_Japanese_Language_Learners_Using_the_Contextual_Representation_of_BERT.pdf": {
        "judul": "Japanese Short Answer Grading for Japanese Language Learners Using the Contextual Representation of BERT",
        "url": "https://doaj.org/article/df3e04cf39464a0da6bb2202e1b888ef",
    },
    "Metzler_et_al_Computer-Assisted_Short_Answer_Grading.pdf": {
        "judul": "Computer-Assisted Short Answer Grading Using Large Language Models and Rubrics",
        "url": "https://dl.gi.de/handle/20.500.12116/45094",
    },
    "N16-1123.pdf": {
        "judul": "Fast and Easy Short Answer Grading with High Accuracy",
        "url": "https://aclanthology.org/N16-1123/",
    },
    "UIJSLICTR+Vol+13_1_164_177.pdf": {
        "judul": "A Comparative Evaluation of Embedding Techniques from Language Models for Automatic Grading of Short Answer Questions",
        "url": "https://journals.ui.edu.ng/index.php/uijslictr/article/download/1563/1265",
    },
    "W16-6324.pdf": {
        "judul": "Wisdom of Students: A Consistent Automatic Short Answer Grading Technique",
        "url": "https://aclanthology.org/W16-6324/",
    },
}

st.set_page_config(page_title="ASAG Search - Mesin Pencari Dokumen Ilmiah", layout="wide")

# ---------------------------------------------------------------------------
# Desain CSS: Tema Gelap Profesional, Hairline Border, Tanpa Ikon/Emoji
# ---------------------------------------------------------------------------
st.html("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap");

:root {
  --ungu: #a78bfa;
  --pink: #f472b6;
  --cyan: #22d3ee;
  --latar-kartu: rgba(255, 255, 255, 0.035);
  --garis: rgba(255, 255, 255, 0.08);
  --teks-utama: #f8fafc;
  --teks-redup: #94a3b8;
  --teks-muted: #64748b;
}

html, body, [class*="css"], .stMarkdown, input, button {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(167, 139, 250, 0.18), transparent 35%),
    radial-gradient(circle at 85% 10%, rgba(244, 114, 182, 0.12), transparent 35%),
    radial-gradient(circle at 50% 100%, rgba(34, 211, 238, 0.08), transparent 40%),
    #09090f;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* Hero Section */
.hero { text-align: center; margin-bottom: 1.6rem; }
.hero .badge {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 600;
  border: 1px solid var(--garis);
  background: var(--latar-kartu);
  color: var(--ungu);
}
.hero h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: clamp(2.3rem, 5.5vw, 3.8rem);
  font-weight: 700;
  margin: 0.6rem 0 0.4rem;
  letter-spacing: -0.03em;
  line-height: 1.1;
  background: linear-gradient(90deg, #f8fafc 10%, var(--ungu) 60%, var(--cyan) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero p {
  color: var(--teks-redup);
  font-size: 1.02rem;
  margin: 0;
  line-height: 1.5;
}

/* Kartu Statistik Ringkas (KPI) */
.grid-stat {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin: 1.2rem 0 1.6rem;
}
@media (max-width: 760px) {
  .grid-stat { grid-template-columns: repeat(2, 1fr); }
}
.stat {
  background: var(--latar-kartu);
  border: 1px solid var(--garis);
  border-radius: 16px;
  padding: 16px 18px;
  backdrop-filter: blur(12px);
  transition: border-color 0.2s ease;
}
.stat:hover {
  border-color: rgba(167, 139, 250, 0.3);
}
.stat .angka {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--teks-utama);
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
.stat .label {
  color: var(--teks-redup);
  font-size: 0.8rem;
  margin-top: 4px;
  font-weight: 500;
}

/* Input Pencarian */
div[data-testid="stTextInputRootElement"] {
  border-radius: 999px !important;
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid var(--garis) !important;
  overflow: hidden;
  transition: all 0.2s ease !important;
}
div[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--ungu) !important;
  box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
}
div[data-testid="stTextInput"] input {
  padding: 13px 22px !important;
  font-size: 1rem !important;
  color: var(--teks-utama) !important;
  background: transparent !important;
}
div[data-testid="InputInstructions"] { display: none; }

/* Rekomendasi Kata Kunci (Pills) */
.rekomendasi-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 0.6rem 0 1.2rem;
  font-size: 0.82rem;
}
.rekomendasi-label {
  color: var(--teks-muted);
  font-weight: 500;
  margin-right: 4px;
}

/* Token Hasil Pemrosesan */
.token-wrap {
  margin: 0.6rem 0 1.2rem;
  color: var(--teks-redup);
  font-size: 0.85rem;
}
.chip {
  display: inline-block;
  margin: 2px 4px 2px 0;
  padding: 3px 10px;
  border-radius: 6px;
  background: rgba(167, 139, 250, 0.1);
  border: 1px solid rgba(167, 139, 250, 0.25);
  color: #ddd6fe;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.82rem;
}

/* Kartu Hasil Dokumen */
.hasil {
  position: relative;
  background: var(--latar-kartu);
  border: 1px solid var(--garis);
  border-radius: 18px;
  padding: 18px 20px 16px 74px;
  margin-bottom: 8px;
  transition: border-color 0.2s, transform 0.15s;
}
.hasil:hover {
  transform: translateY(-1px);
  border-color: rgba(167, 139, 250, 0.35);
}
.rank {
  position: absolute;
  left: 18px;
  top: 18px;
  width: 38px;
  height: 38px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: #ececf5;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--garis);
}
.rank.top {
  color: #fbcfe8;
  background: rgba(244, 114, 182, 0.12);
  border-color: rgba(244, 114, 182, 0.45);
}
.hasil .header-baris {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.hasil .judul {
  color: var(--teks-utama);
  font-weight: 600;
  font-size: 1.04rem;
  line-height: 1.4;
  flex: 1;
}
.btn-jurnal {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 7px;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--ungu);
  background: rgba(167, 139, 250, 0.08);
  border: 1px solid rgba(167, 139, 250, 0.25);
  text-decoration: none;
  white-space: nowrap;
  transition: all 0.2s ease;
  flex-shrink: 0;
}
.btn-jurnal:hover {
  background: rgba(167, 139, 250, 0.2);
  border-color: rgba(167, 139, 250, 0.5);
  color: #ffffff;
}

.hasil .meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  color: var(--teks-redup);
  font-size: 0.78rem;
  margin: 6px 0 10px;
}
.hasil .meta .kode-badge {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  color: var(--cyan);
  background: rgba(34, 211, 238, 0.08);
  border-radius: 5px;
  padding: 1px 6px;
  border: 1px solid rgba(34, 211, 238, 0.2);
}

/* Meter Skor Kemiripan */
.meter {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 6px;
}
.meter .rel {
  flex: 1;
  height: 4px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
}
.meter .isi {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--ungu), var(--pink));
}
.meter .skor {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  color: var(--teks-utama);
  font-size: 0.88rem;
  min-width: 58px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.badge-rel {
  font-size: 0.72rem;
  padding: 2px 7px;
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-rel.tinggi {
  background: rgba(34, 211, 238, 0.1);
  color: #67e8f9;
  border: 1px solid rgba(34, 211, 238, 0.25);
}
.badge-rel.cukup {
  background: rgba(167, 139, 250, 0.1);
  color: #c4b5fd;
  border: 1px solid rgba(167, 139, 250, 0.25);
}
.badge-rel.rendah {
  background: rgba(148, 163, 184, 0.08);
  color: #94a3b8;
  border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Cuplikan Teks & Sorotan */
.cuplikan {
  color: #cbd5e1;
  font-size: 0.88rem;
  margin-top: 10px;
  line-height: 1.6;
}
.cuplikan mark {
  background: rgba(34, 211, 238, 0.14);
  color: #67e8f9;
  padding: 1px 5px;
  border-radius: 4px;
  font-weight: 600;
  border-bottom: 1px solid rgba(34, 211, 238, 0.35);
}

/* Expander Dokumen */
div[data-testid="stExpander"] {
  border: 1px solid var(--garis) !important;
  border-radius: 12px !important;
  background: rgba(255, 255, 255, 0.015) !important;
  margin-top: 4px;
  margin-bottom: 14px;
}
div[data-testid="stExpander"] summary {
  color: var(--teks-redup) !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  padding: 8px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
  color: var(--teks-utama) !important;
}
.isi-dok {
  color: #cbd5e1;
  font-size: 0.85rem;
  line-height: 1.6;
  white-space: pre-wrap;
  max-height: 360px;
  overflow-y: auto;
  padding: 4px 6px;
}

/* Pesan Kosong / Informasi */
.pesan-status {
  text-align: center;
  padding: 40px 20px;
  border: 1px dashed var(--garis);
  border-radius: 18px;
  color: var(--teks-redup);
  font-size: 0.95rem;
  background: rgba(255, 255, 255, 0.01);
}

/* Sidebar & Anggota */
.anggota {
  background: var(--latar-kartu);
  border: 1px solid var(--garis);
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 8px;
}
.anggota .nama {
  color: var(--teks-utama);
  font-weight: 600;
  font-size: 0.86rem;
  line-height: 1.3;
}
.anggota .nim {
  color: var(--ungu);
  font-size: 0.78rem;
  margin-top: 2px;
  font-family: 'Space Grotesk', sans-serif;
  font-variant-numeric: tabular-nums;
}

/* Grid Dokumen (Tab Korpus) */
.grid-dok {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}
.dok {
  background: var(--latar-kartu);
  border: 1px solid var(--garis);
  border-radius: 16px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: border-color 0.2s;
}
.dok:hover {
  border-color: rgba(167, 139, 250, 0.3);
}
.dok .kode {
  display: inline-block;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--cyan);
  background: rgba(34, 211, 238, 0.08);
  border-radius: 6px;
  padding: 2px 7px;
  border: 1px solid rgba(34, 211, 238, 0.2);
  align-self: flex-start;
}
.dok .judul {
  color: var(--teks-utama);
  font-weight: 600;
  font-size: 0.9rem;
  margin: 10px 0 6px;
  line-height: 1.4;
}
.dok .meta {
  color: var(--teks-redup);
  font-size: 0.76rem;
  overflow-wrap: anywhere;
  line-height: 1.4;
}
.dok .tautan {
  margin-top: 12px;
}

/* Tombol Unduh CSV */
div.stDownloadButton > button {
  background: rgba(167, 139, 250, 0.08) !important;
  border: 1px solid rgba(167, 139, 250, 0.25) !important;
  color: #ddd6fe !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  transition: all 0.2s ease !important;
}
div.stDownloadButton > button:hover {
  background: rgba(167, 139, 250, 0.18) !important;
  border-color: rgba(167, 139, 250, 0.5) !important;
  color: #ffffff !important;
}

/* 1. Tombol Rekomendasi Kata Kunci (Kolom 1 - 5): Rounded Oval Penuh, Abu-Abu Gelap Redup */
div[data-testid="stColumn"]:not(:last-child) button,
div[data-testid="stColumn"]:not(:last-child) button[data-testid="baseButton-secondary"],
div[data-testid="stColumn"]:not(:last-child) .stButton > button {
  border-radius: 999px !important;
  font-size: 0.8rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #94a3b8 !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 10px !important;
  transition: all 0.15s ease !important;
}
div[data-testid="stColumn"]:not(:last-child) button:hover {
  background: rgba(255, 255, 255, 0.09) !important;
  border-color: rgba(255, 255, 255, 0.25) !important;
  color: #f8fafc !important;
}

/* 2. Tombol Acak Kata: Sama-sama Rounded Oval Penuh (999px), Warna Abu-Abu Solid Terang & Teks Putih */
div[data-testid="stColumn"]:last-child button,
div[data-testid="stColumn"]:last-child button[data-testid="baseButton-primary"],
div[data-testid="stColumn"]:last-child .stButton > button,
button[data-testid="baseButton-primary"] {
  border-radius: 999px !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  background: #3e4c5e !important;
  border: 1px solid #7a889b !important;
  color: #ffffff !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 12px !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
  transition: all 0.18s ease !important;
}
div[data-testid="stColumn"]:last-child button:hover,
button[data-testid="baseButton-primary"]:hover {
  background: #52637a !important;
  border-color: #a0aec0 !important;
  color: #ffffff !important;
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
}
</style>
""")

# ---------------------------------------------------------------------------
# Data & Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Mempersiapkan korpus dan membangun indeks dokumen...")
def siapkan(folder, lang):
    siapkan_nltk()
    korpus = muat_korpus(folder)
    praproses = Praproses(lang)
    token_dok = {d["kode"]: praproses(d["teks"]) for d in korpus}
    idx = bangun_indeks(token_dok)
    return korpus, praproses, token_dok, idx


@st.cache_data
def tabel_tabel(folder, lang):
    _, _, _, idx = siapkan(folder, lang)
    return tabel_frekuensi(idx), tabel_tfidf_transpose(idx)


def dapatkan_info_dokumen(nama_berkas, judul_fallback):
    """Mengembalikan judul baku dan URL jurnal dari daftar resmi jika tersedia."""
    if nama_berkas in DOKUMEN_INFO:
        return DOKUMEN_INFO[nama_berkas]["judul"], DOKUMEN_INFO[nama_berkas]["url"]
    return judul_fallback, None


def dapatkan_pool_rekomendasi(idx, tfidf_t, batas_pool=40):
    """Menghasilkan kumpulan kata kunci berbobot TF-IDF tinggi dengan daya pembeda baik."""
    total_tfidf = tfidf_t.sum(axis=1)
    kandidat = [
        t for t in total_tfidf.sort_values(ascending=False).index
        if 2 <= idx.df[t] <= (len(idx.kode) - 2) and len(t) > 3
    ]
    return kandidat[:batas_pool]


def tentukan_label_relevansi(skor):
    """Mengelompokkan tingkat relevansi berbasis skor cosine similarity."""
    if skor >= 0.35:
        return "tinggi", "Relevansi Tinggi"
    if skor >= 0.15:
        return "cukup", "Relevansi Cukup"
    return "rendah", "Relevansi Rendah"


def cuplikan(teks, token, lebar=150):
    """Menghasilkan potongan kalimat di sekitar term kueri dengan penyorotan."""
    teks = re.sub(r"\s+", " ", teks)
    pola = re.compile(r"\b(" + "|".join(re.escape(t) for t in token) + r")\w*", re.IGNORECASE)
    m = pola.search(teks)
    if not m:
        return html.escape(teks[:lebar * 2]) + "…"
    awal = max(0, m.start() - lebar)
    potong = teks[awal:m.end() + lebar]
    aman = html.escape(potong)
    aman = pola.sub(lambda x: f"<mark>{x.group(0)}</mark>", aman)
    return ("…" if awal else "") + aman + "…"


# ---------------------------------------------------------------------------
# Sidebar (Pengaturan & Identitas Kelompok)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pengaturan")
    topn = st.slider("Jumlah Hasil", 1, 15, 8)
    min_skor = st.slider("Skor Cosine Minimum", 0.0, 0.5, 0.0, 0.01)
    st.markdown("---")
    st.markdown("### Identitas Kelompok")
    st.html("".join(
        f'<div class="anggota"><div class="nama">{nama}</div><div class="nim">NIM {nim}</div></div>'
        for nim, nama in ANGGOTA))
    st.caption("Vector Space Model · Pembobotan TF-IDF · Cosine Similarity")

folder = os.path.join(FOLDER_INI, "corpus")
korpus, praproses, token_dok, idx = siapkan(folder, BAHASA)
frek, tfidf_t = tabel_tabel(folder, BAHASA)
info = {d["kode"]: d for d in korpus}
total_token = sum(len(t) for t in token_dok.values())

# Pool kata kunci rekomendasi dan state acak
pool_rekomendasi = dapatkan_pool_rekomendasi(idx, tfidf_t, batas_pool=40)
if "rekomendasi_aktif" not in st.session_state:
    st.session_state["rekomendasi_aktif"] = random.sample(pool_rekomendasi, min(5, len(pool_rekomendasi)))

if "input_teks_kueri" not in st.session_state:
    st.session_state["input_teks_kueri"] = ""

# ---------------------------------------------------------------------------
# Header (Hero) & Kartu Metrik
# ---------------------------------------------------------------------------
st.html(f"""
<div class="hero">
  <span class="badge">Mesin Pencari Dokumen Ilmiah · Korpus ASAG</span>
  <h1>ASAG Search</h1>
  <p>Pencarian artikel ilmiah seputar Automated Short Answer Grading berbasis kemiripan teks.</p>
</div>
<div class="grid-stat">
  <div class="stat"><div class="angka">{len(korpus)}</div><div class="label">Total Dokumen</div></div>
  <div class="stat"><div class="angka">{len(idx.vocab):,}</div><div class="label">Kata Unik (Kosakata)</div></div>
  <div class="stat"><div class="angka">{total_token:,}</div><div class="label">Total Kata Bersih</div></div>
  <div class="stat"><div class="angka">{total_token // len(korpus):,}</div><div class="label">Rata-rata Kata per Dokumen</div></div>
</div>
""")

tab_cari, tab_stat, tab_matriks, tab_korpus = st.tabs(
    ["Pencarian", "Statistik Korpus", "Matriks Pembobotan", "Daftar Korpus"])

# ---------------------------------------------------------------------------
# Tab 1: Pencarian
# ---------------------------------------------------------------------------
# Callback helper untuk tombol kata kunci (menghindari error Streamlit session state)
def pilih_kata_kunci(kata):
    st.session_state["input_teks_kueri"] = kata

def acak_kata_kunci():
    st.session_state["rekomendasi_aktif"] = random.sample(pool_rekomendasi, min(5, len(pool_rekomendasi)))

with tab_cari:
    kueri_input = st.text_input(
        "Kueri",
        key="input_teks_kueri",
        label_visibility="collapsed",
        placeholder="Ketik kata kunci pencarian...",
    )

    # Baris rekomendasi kata kunci dan tombol acak sejajar (1 baris)
    st.html('<div class="rekomendasi-wrap" style="margin: 0.5rem 0 0.3rem;"><span class="rekomendasi-label">Rekomendasi kata kunci:</span></div>')
    cols_rek = st.columns([1, 1, 1, 1, 1, 0.85])
    for i, kata in enumerate(st.session_state["rekomendasi_aktif"]):
        cols_rek[i].button(
            kata,
            key=f"btn_rek_{i}_{kata}",
            use_container_width=True,
            on_click=pilih_kata_kunci,
            args=(kata,),
        )

    cols_rek[5].button(
        "↻ Acak",
        key="btn_acak_kata",
        type="primary",
        use_container_width=True,
        on_click=acak_kata_kunci,
        help="Undi 5 kata kunci baru secara acak dari korpus",
    )

    kueri = kueri_input.strip()
    if not kueri:
        st.html('<div class="pesan-status">Masukkan kata kunci pada kolom di atas untuk mencari dokumen.</div>')
    else:
        token, hasil = cari(kueri, idx, praproses)
        hasil = [(k, s) for k, s in hasil if s >= min_skor][:topn]
        chip = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in token) or "—"
        st.html(f'<div class="token-wrap">Kata kunci hasil pemrosesan: {chip}</div>')

        if not hasil:
            st.html('<div class="pesan-status">'
                    'Tidak ada dokumen yang cocok dengan kata kunci ini. Silakan coba kata kunci lain.</div>')
        else:
            skor_maks = hasil[0][1]
            for r, (kode, skor) in enumerate(hasil, start=1):
                d = info[kode]
                judul_resmi, _ = dapatkan_info_dokumen(d["nama"], d["judul"])
                lebar = (100 * skor / skor_maks) if skor_maks > 0 else 0
                kelas_rel, teks_rel = tentukan_label_relevansi(skor)
                url_pdf_lokal = f"/app/static/{urllib.parse.quote(d['nama'])}"
                tombol_jurnal_html = (
                    f'<a class="btn-jurnal" href="{url_pdf_lokal}" target="_blank" rel="noopener noreferrer">Buka Dokumen PDF ↗</a>'
                )

                st.html(f"""
<div class="hasil">
  <div class="rank {'top' if r == 1 else ''}">{r:02d}</div>
  <div class="header-baris">
    <div class="judul">{html.escape(judul_resmi)}</div>
    {tombol_jurnal_html}
  </div>
  <div class="meta">
    <span class="kode-badge">{kode}</span>
    <span>{html.escape(d['nama'])}</span>
    <span>·</span>
    <span>{len(token_dok[kode]):,} kata</span>
  </div>
  <div class="meter">
    <div class="rel"><div class="isi" style="width:{lebar:.1f}%"></div></div>
    <span class="badge-rel {kelas_rel}">{teks_rel}</span>
    <div class="skor">{skor:.4f}</div>
  </div>
  <div class="cuplikan">{cuplikan(d['teks'], token)}</div>
</div>""")
                with st.expander(f"Lihat Isi Dokumen [{kode}]"):
                    st.html('<div class="isi-dok">' + html.escape(d["teks"][:3000]) + '</div>')

# ---------------------------------------------------------------------------
# Tab 2: Statistik Korpus
# ---------------------------------------------------------------------------
with tab_stat:
    kiri, kanan = st.columns(2)
    total = frek.drop(columns="DF").sum(axis=1)

    with kiri:
        st.markdown("##### 15 Kata Paling Sering Muncul")
        data_frek = total.sort_values(ascending=False).head(15).rename("frekuensi") \
                         .rename_axis("term").reset_index()

        base_frek = alt.Chart(data_frek).encode(
            y=alt.Y("term:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12)),
            x=alt.X("frekuensi:Q", title="Total Frekuensi Kemunculan",
                    axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", grid=False)),
            tooltip=["term", "frekuensi"],
        )
        bar_frek = base_frek.mark_bar(cornerRadiusEnd=3, color="#a78bfa")
        teks_frek = base_frek.mark_text(
            align="left", baseline="middle", dx=6, color="#f1f5f9", fontSize=11, font="Space Grotesk"
        ).encode(text=alt.Text("frekuensi:Q", format=",d"))

        grafik_kiri = (bar_frek + teks_frek).properties(height=430) \
                                            .configure(background="transparent") \
                                            .configure_view(stroke=None)
        st.altair_chart(grafik_kiri, width="stretch")

    with kanan:
        st.markdown("##### 15 Kata Paling Signifikan (TF-IDF)")
        data_tfidf = tfidf_t.max(axis=1).sort_values(ascending=False).head(15).rename("tfidf") \
                            .rename_axis("term").reset_index()
        data_tfidf["dokumen"] = [tfidf_t.loc[t].idxmax() for t in data_tfidf["term"]]

        base_tfidf = alt.Chart(data_tfidf).encode(
            y=alt.Y("term:N", sort="-x", title=None,
                    axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12)),
            x=alt.X("tfidf:Q", title="Bobot TF-IDF Maksimum",
                    axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8", grid=False)),
            tooltip=["term", "dokumen", alt.Tooltip("tfidf:Q", format=".3f")],
        )
        bar_tfidf = base_tfidf.mark_bar(cornerRadiusEnd=3, color="#22d3ee")
        teks_tfidf = base_tfidf.mark_text(
            align="left", baseline="middle", dx=6, color="#f1f5f9", fontSize=11, font="Space Grotesk"
        ).encode(text=alt.Text("tfidf:Q", format=".3f"))

        grafik_kanan = (bar_tfidf + teks_tfidf).properties(height=430) \
                                               .configure(background="transparent") \
                                               .configure_view(stroke=None)
        st.altair_chart(grafik_kanan, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: Matriks Pembobotan
# ---------------------------------------------------------------------------
with tab_matriks:
    cari_term = st.text_input("Cari Kata", placeholder="Ketik kata untuk memfilter tabel...")
    saring = frek.index.str.contains(cari_term.strip().lower(), regex=False) if cari_term else slice(None)

    st.markdown("##### Matriks Frekuensi Term per Dokumen dan Nilai DF")
    st.dataframe(frek.loc[saring], width="stretch", height=320)
    st.download_button(
        "Unduh Matriks Frekuensi (CSV)",
        frek.to_csv().encode("utf-8-sig"),
        "frekuensi_term.csv",
        "text/csv",
    )

    st.markdown("##### Matriks TF-IDF Transposisi (Baris: Kata, Kolom: Dokumen)")
    st.dataframe(
        tfidf_t.loc[saring],
        width="stretch",
        height=320,
        column_config={k: st.column_config.NumberColumn(format="%.3f") for k in tfidf_t.columns},
    )
    st.download_button(
        "Unduh Matriks TF-IDF (CSV)",
        tfidf_t.to_csv().encode("utf-8-sig"),
        "tfidf_transpose.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 4: Daftar Korpus
# ---------------------------------------------------------------------------
with tab_korpus:
    kartu_list = []
    for d in korpus:
        judul_resmi, _ = dapatkan_info_dokumen(d["nama"], d["judul"])
        url_pdf_lokal = f"/app/static/{urllib.parse.quote(d['nama'])}"
        tautan_korpus = (
            f'<div class="tautan"><a class="btn-jurnal" href="{url_pdf_lokal}" target="_blank" rel="noopener noreferrer">Buka Dokumen PDF ↗</a></div>'
        )
        kartu_list.append(f"""
<div class="dok">
  <div>
    <span class="kode">{d['kode']}</span>
    <div class="judul">{html.escape(judul_resmi)}</div>
    <div class="meta">{html.escape(d['nama'])} · {len(token_dok[d['kode']]):,} kata</div>
  </div>
  {tautan_korpus}
</div>""")

    st.html(f'<div class="grid-dok">{"".join(kartu_list)}</div>')
