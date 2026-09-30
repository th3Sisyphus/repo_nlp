'''
Tugas Kelompok C-BOW
Anggota:
Thomas Aquinas Ryan Wisnu Adi / 71230975
Putu Gde Kenzie Carlen Mataram / 71230994
Yehezkiel Darren Putra Wardoyo / 71231023
'''

import os
import re
from collections import Counter

doc_files = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]
all_sentences = []

def process_document(text):
    """
    Memproses teks dokumen dengan urutan:
    Paragraf -> Kalimat -> Normalisasi -> Tambah Penanda '_' -> Tokenisasi
    """
    tokenized_sentences = []
    
    # 1. Bagi berdasarkan Paragraf (split "\n")
    paragraphs = text.split("\n")
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # 2. Bagi berdasarkan Kalimat (split ". ")
        sentences = p.split(". ")
        
        for s in sentences:
            s = s.strip()
            if not s:
                continue
                
            # 3. Normalisasi (lowercasing & hapus punctuation/karakter non-huruf)
            s_clean = s.lower()
            s_clean = re.sub(r"[^a-z\s]", " ", s_clean)
            s_clean = re.sub(r"\s+", " ", s_clean).strip()
            
            if not s_clean:
                continue
                
            # 4. Tambahkan "_" pada awal dan akhir setiap kalimat
            s_marked = f"_ {s_clean} _"
            
            # 5. Tokenisasi per kata
            tokens = s_marked.split()
            
            tokenized_sentences.append(tokens)
            
    return tokenized_sentences

def build_cbow(sentences, window_size):
    """
    Membentuk pasangan (context_tuple, target_word) dari daftar token kalimat.
    """
    pairs = []
    for tokens in sentences:
        # Kata target hanya diambil dari kata asli (indeks 1 hingga len-2)
        # Indeks 0 dan -1 adalah penanda '_'
        for i in range(1, len(tokens) - 1):
            target = tokens[i]
            
            # Ambil konteks kiri dan kanan dari list tokens
            left = tokens[max(0, i - window_size) : i]
            right = tokens[i + 1 : min(len(tokens), i + 1 + window_size)]
            
            # Jika konteks kiri/kanan kurang dari window_size, lengkapi dengan '_'
            while len(left) < window_size:
                left.insert(0, "_")
            while len(right) < window_size:
                right.append("_")
                
            context = left + right
            pairs.append((tuple(context), target))
            
    return pairs

def count_frequency(pairs):
    return Counter(pairs)

for f in doc_files:
    # Cek direktori utama atau folder 'dokumen/'
    path = f if os.path.exists(f) else os.path.join("dokumen", f)
    
    if not os.path.exists(path):
        print(f"Peringatan: File {f} tidak ditemukan di '{path}'. Dilewati.")
        continue

    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
        
    sentences = process_document(raw)
    all_sentences.extend(sentences)
    
print(f"Total kalimat yang berhasil diekstrak dan ditokenisasi: {len(all_sentences)} kalimat.")
print("\nContoh 2 hasil tokenisasi kalimat pertama:")
for sample in all_sentences[:2]:
    print(sample)
    
for W in [1, 2]:
    print(f"\n{'='*60}")
    print(f"Hasil Ekstraksi CBOW dengan Window Size W = {W}")
    print('='*60)
    
    pairs = build_cbow(all_sentences, W)
    freq = count_frequency(pairs)
    
    print(f"Total pasangan (konteks, target) yang terbentuk: {len(pairs)}")
    print(f"Total pasangan unik: {len(freq)}")
    print(f"\n20 Pasangan Terbanyak:\n")
    
    for (context, target), count in sorted(freq.items(), key=lambda x: -x[1])[:20]:
        context_str = str(list(context))
        print(f"  context={context_str:<42} target='{target}'\tfreq={count}")