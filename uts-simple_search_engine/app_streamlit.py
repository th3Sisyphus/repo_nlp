"""
Antarmuka web (Streamlit) untuk mesin pencari korpus ASAG.

Cara menjalankan (dari folder ini):
  streamlit run app_streamlit.py
"""

import html
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from mesin_pencari_asag import (
    Praproses, bangun_indeks, cari, muat_korpus, siapkan_nltk,
    tabel_frekuensi, tabel_tfidf_transpose,
)

FOLDER_INI = os.path.dirname(os.path.abspath(__file__))
BAHASA = "en"  # semua artikel di korpus berbahasa Inggris
ANGGOTA = [
    ("71230994", "Putu Gde Kenzie Carlen Mataram"),
    ("71230975", "Thomas Aquinas Ryan Wisnu Adi"),
    ("71231023", "Yehezkiel Darren Putra Wardoyo"),
]

st.set_page_config(page_title="ASAG Search", layout="wide")

# ---------------------------------------------------------------------------
# Gaya: latar gelap bergradasi, kartu "glass", aksen ungu-pink-cyan
# ---------------------------------------------------------------------------
st.html("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap");
:root {
  --ungu: #a78bfa; --pink: #f472b6; --cyan: #22d3ee;
  --kartu: rgba(255,255,255,0.05); --garis: rgba(255,255,255,0.10);
  --teks: #ececf5; --redup: #9b9bb5;
}
html, body, [class*="css"], .stMarkdown, input, button { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp {
  background:
    radial-gradient(circle at 10% 0%, rgba(167,139,250,0.22), transparent 40%),
    radial-gradient(circle at 95% 15%, rgba(244,114,182,0.16), transparent 38%),
    radial-gradient(circle at 50% 100%, rgba(34,211,238,0.10), transparent 45%),
    #0b0b14;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }
.hero { text-align: center; margin-bottom: 1.4rem; }
.hero .badge {
  display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: .8rem;
  border: 1px solid var(--garis); background: var(--kartu); color: var(--redup);
}
.hero h1 {
  font-family: 'Space Grotesk', sans-serif !important; font-size: clamp(2.4rem, 6vw, 4rem);
  font-weight: 700; margin: .5rem 0 .2rem; letter-spacing: -.03em; line-height: 1.05;
  background: linear-gradient(90deg, var(--ungu), var(--pink) 55%, var(--cyan));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--redup); font-size: 1.05rem; margin: 0; }
.grid-stat { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1rem 0 1.6rem; }
@media (max-width: 760px) { .grid-stat { grid-template-columns: repeat(2, 1fr); } }
.stat {
  background: var(--kartu); border: 1px solid var(--garis); border-radius: 20px;
  padding: 16px 18px; backdrop-filter: blur(12px);
}
.stat .angka { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--teks); }
.stat .label { color: var(--redup); font-size: .82rem; }
div[data-testid="stTextInputRootElement"] {
  border-radius: 999px !important; background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--garis) !important; overflow: hidden;
}
div[data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--ungu) !important; box-shadow: 0 0 0 3px rgba(167,139,250,.25) !important;
}
div[data-testid="stTextInput"] input { padding: 14px 22px !important; font-size: 1.05rem !important; background: transparent !important; }
div[role="tab"] { font-weight: 600; }
.token-wrap { margin: .4rem 0 1rem; color: var(--redup); font-size: .88rem; }
.chip {
  display: inline-block; margin: 3px 4px 3px 0; padding: 4px 12px; border-radius: 999px;
  background: rgba(167,139,250,.14); border: 1px solid rgba(167,139,250,.35); color: #ddd6fe;
  font-family: 'Space Grotesk', sans-serif; font-size: .85rem;
}
.hasil {
  position: relative; background: var(--kartu); border: 1px solid var(--garis); border-radius: 22px;
  padding: 18px 20px 16px 76px; margin-bottom: 12px; transition: transform .15s, border-color .15s;
}
.hasil:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.45); }
.rank {
  position: absolute; left: 18px; top: 18px; width: 42px; height: 42px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #0b0b14;
  background: linear-gradient(135deg, var(--ungu), var(--pink));
}
.rank.top { background: linear-gradient(135deg, #fde68a, #f472b6); box-shadow: 0 0 22px rgba(244,114,182,.45); }
.hasil .judul { color: var(--teks); font-weight: 700; font-size: 1.05rem; line-height: 1.35; }
.hasil .meta { color: var(--redup); font-size: .8rem; margin: 3px 0 10px; }
.meter { display: flex; align-items: center; gap: 10px; }
.meter .rel { flex: 1; height: 8px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }
.meter .isi { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--ungu), var(--pink)); }
.meter .skor { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--teks); font-size: .9rem; min-width: 64px; text-align: right; font-variant-numeric: tabular-nums; }
.cuplikan { color: #c9c9dc; font-size: .9rem; margin-top: 10px; line-height: 1.55; }
.cuplikan mark { background: rgba(34,211,238,.18); color: #a5f3fc; padding: 0 3px; border-radius: 5px; }
div[data-testid="InputInstructions"] { display: none; }
.anggota { background: var(--kartu); border: 1px solid var(--garis); border-radius: 14px; padding: 10px 12px; margin-bottom: 8px; }
.anggota .nama { color: var(--teks); font-weight: 600; font-size: .88rem; line-height: 1.3; }
.anggota .nim { color: var(--ungu); font-size: .78rem; margin-top: 2px; font-variant-numeric: tabular-nums; }
.kosong {
  text-align: center; padding: 38px 20px; border: 1px dashed var(--garis); border-radius: 22px; color: var(--redup);
}
.isi-dok { color: #c9c9dc; font-size: .88rem; line-height: 1.6; white-space: pre-wrap; max-height: 360px; overflow-y: auto; }
.grid-dok { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.dok { background: var(--kartu); border: 1px solid var(--garis); border-radius: 18px; padding: 14px 16px; }
.dok .kode { display: inline-block; font-family: 'Space Grotesk', sans-serif; font-size: .75rem; font-weight: 700;
  color: var(--cyan); background: rgba(34,211,238,.10); border-radius: 8px; padding: 2px 8px; }
.dok .judul { color: var(--teks); font-weight: 600; font-size: .92rem; margin: 8px 0 4px; line-height: 1.35; }
.dok .meta { color: var(--redup); font-size: .76rem; overflow-wrap: anywhere; }
</style>
""")


# ---------------------------------------------------------------------------
# Data: indeks dibangun sekali lalu disimpan di cache Streamlit
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Lagi baca 15 paper & bangun indeks...")
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


def cuplikan(teks, token, lebar=150):
    """Potongan teks di sekitar kemunculan pertama term kueri, dengan term disorot."""
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pengaturan")
    topn = st.slider("Jumlah hasil", 1, 15, 8)
    min_skor = st.slider("Skor cosine minimum", 0.0, 0.5, 0.0, 0.01)
    st.markdown("---")
    st.markdown("### Anggota Kelompok")
    st.html("".join(
        f'<div class="anggota"><div class="nama">{nama}</div><div class="nim">NIM {nim}</div></div>'
        for nim, nama in ANGGOTA))
    st.caption("TF-IDF + Cosine Similarity · Tugas Kelompok II NLP")

folder = os.path.join(FOLDER_INI, "corpus")
korpus, praproses, token_dok, idx = siapkan(folder, BAHASA)
info = {d["kode"]: d for d in korpus}
total_token = sum(len(t) for t in token_dok.values())

# ---------------------------------------------------------------------------
# Hero + widget statistik
# ---------------------------------------------------------------------------
st.html(f"""
<div class="hero">
  <span class="badge">Automated Short Answer Grading · mini search engine</span>
  <h1>ASAG Search</h1>
  <p>Ketik topik, langsung ketemu paper yang paling nyambung.</p>
</div>
<div class="grid-stat">
  <div class="stat"><div class="angka">{len(korpus)}</div><div class="label">paper di korpus</div></div>
  <div class="stat"><div class="angka">{len(idx.vocab):,}</div><div class="label">term unik</div></div>
  <div class="stat"><div class="angka">{total_token:,}</div><div class="label">token bersih</div></div>
  <div class="stat"><div class="angka">{total_token // len(korpus):,}</div><div class="label">token / paper</div></div>
</div>
""")

tab_cari, tab_stat, tab_matriks, tab_korpus = st.tabs(
    ["Cari", "Statistik", "Matriks", "Korpus"])

# ---------------------------------------------------------------------------
# Tab Cari
# ---------------------------------------------------------------------------
with tab_cari:
    kueri = st.text_input("Kueri", key="kueri", label_visibility="collapsed",
                          placeholder="Masukkan Query")
    kueri = kueri.strip()
    if not kueri:
        st.html('<div class="kosong">Mulai ketik kueri di kotak pencarian di atas.</div>')
    else:
        token, hasil = cari(kueri, idx, praproses)
        hasil = [(k, s) for k, s in hasil if s >= min_skor][:topn]
        chip = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in token) or "—"
        st.html(f'<div class="token-wrap">Token setelah praproses: {chip}</div>')

        if not hasil:
            st.html('<div class="kosong">'
                    'Yah, belum ada paper yang cocok. Coba kata kunci lain ya.</div>')
        else:
            skor_maks = hasil[0][1]
            for r, (kode, skor) in enumerate(hasil, start=1):
                d = info[kode]
                lebar = 100 * skor / skor_maks
                st.html(f"""
<div class="hasil">
  <div class="rank {'top' if r == 1 else ''}">{r}</div>
  <div class="judul">{html.escape(d['judul'])}</div>
  <div class="meta">{kode} · {html.escape(d['nama'])} · {len(token_dok[kode]):,} token</div>
  <div class="meter"><div class="rel"><div class="isi" style="width:{lebar:.1f}%"></div></div>
  <div class="skor">{skor:.4f}</div></div>
  <div class="cuplikan">{cuplikan(d['teks'], token)}</div>
</div>""")
                with st.expander(f"Baca isi {kode}"):
                    st.html('<div class="isi-dok">' + html.escape(d["teks"][:3000]) + '</div>')

# ---------------------------------------------------------------------------
# Tab Statistik
# ---------------------------------------------------------------------------
frek, tfidf_t = tabel_tabel(folder, BAHASA)

with tab_stat:
    kiri, kanan = st.columns(2)
    total = frek.drop(columns="DF").sum(axis=1)

    with kiri:
        st.markdown("##### 15 term paling sering muncul")
        data = total.sort_values(ascending=False).head(15).rename("frekuensi") \
                    .rename_axis("term").reset_index()
        grafik = alt.Chart(data).mark_bar(cornerRadiusEnd=4, color="#a78bfa").encode(
            x=alt.X("frekuensi:Q", title="Frekuensi total"),
            y=alt.Y("term:N", sort="-x", title=None),
            tooltip=["term", "frekuensi"],
        ).properties(height=420).configure(background="transparent").configure_view(stroke=None)
        st.altair_chart(grafik, width="stretch")

    with kanan:
        st.markdown("##### 15 term paling khas (bobot tf-idf tertinggi)")
        data = tfidf_t.max(axis=1).sort_values(ascending=False).head(15).rename("tfidf") \
                      .rename_axis("term").reset_index()
        data["dokumen"] = [tfidf_t.loc[t].idxmax() for t in data["term"]]
        grafik = alt.Chart(data).mark_bar(cornerRadiusEnd=4, color="#22d3ee").encode(
            x=alt.X("tfidf:Q", title="Bobot tf-idf maksimum"),
            y=alt.Y("term:N", sort="-x", title=None),
            tooltip=["term", "dokumen", alt.Tooltip("tfidf:Q", format=".3f")],
        ).properties(height=420).configure(background="transparent").configure_view(stroke=None)
        st.altair_chart(grafik, width="stretch")

# ---------------------------------------------------------------------------
# Tab Matriks
# ---------------------------------------------------------------------------
with tab_matriks:
    cari_term = st.text_input("Filter term", placeholder="Masukkan Kata")
    saring = frek.index.str.contains(cari_term.strip().lower(), regex=False) if cari_term else slice(None)

    st.markdown("##### Frekuensi term per dokumen + DF")
    st.dataframe(frek.loc[saring], width="stretch", height=320)
    st.download_button("Unduh frekuensi_term.csv", frek.to_csv().encode("utf-8-sig"),
                       "frekuensi_term.csv", "text/csv")

    st.markdown("##### Matriks tf-idf (transpose: baris = term, kolom = dokumen)")
    st.dataframe(tfidf_t.loc[saring], width="stretch", height=320,
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in tfidf_t.columns})
    st.download_button("Unduh tfidf_transpose.csv", tfidf_t.to_csv().encode("utf-8-sig"),
                       "tfidf_transpose.csv", "text/csv")

# ---------------------------------------------------------------------------
# Tab Korpus
# ---------------------------------------------------------------------------
with tab_korpus:
    kartu = "".join(f"""
<div class="dok"><span class="kode">{d['kode']}</span>
<div class="judul">{html.escape(d['judul'])}</div>
<div class="meta">{html.escape(d['nama'])} · {len(token_dok[d['kode']]):,} token</div></div>"""
                    for d in korpus)
    st.html(f'<div class="grid-dok">{kartu}</div>')
