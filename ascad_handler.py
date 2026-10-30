# =============================================================================
# FISIERUL 2: ascad_handler.py
# =============================================================================
"""
Modul pentru manipularea datelor hardware (h5py).
Extrage Fereastra de Interes (POI) din baza ASCAD sau, daca fisierul lipseste,
genereaza date sintetice realiste pentru demonstratie.

Detalii importante despre ASCAD (baza publica, microcontroler ATmega8515):
  * Fisierul ASCAD.h5 contine deja fereastra de 700 de esantioane extrasa
    exact in jurul operatiei S-Box pentru al 3-lea octet de stare (INDEX 2).
    Prin urmare atacul tinteste octetul de cheie index 2, nu index 0.
  * Implementarea este MASCATA (boolean masking). Valoarea care scurge in aceasta
    fereastra nu este Sbox(p^k), ci Sbox(p^k) XOR masca_de_iesire.
    In metadata, masca de iesire a S-Box-ului octetului 2 este coloana masks[:, 0].
    Fara aceasta masca, un atac CPA de ordinul 1 esueaza (corelatie la nivel de zgomot).
"""

import numpy as np
import os
import zipfile
try:
    import h5py
except ImportError:
    h5py = None

# --- Configuratie atac ASCAD ---
ASCAD_TARGET_BYTE = 2   # fereastra ASCAD este centrata pe octetul de stare index 2
ASCAD_MASK_COL = 0      # masks[:, 0] = masca de iesire a S-Box-ului pentru octetul 2

# Tabela S-Box pentru AES-128
AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

# Greutatea Hamming pre-calculata pentru toti octetii (0..255)
HW_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.float64)


def hw(value):
    """Greutatea Hamming (numarul de biti 1) a unui octet."""
    return bin(int(value)).count("1")


def hw_vec(arr):
    """Greutatea Hamming vectorizata pe un array de octeti."""
    return HW_TABLE[np.asarray(arr, dtype=np.uint8)]


def generate_synthetic_data(num_traces, window_size):
    """
    Date sintetice de rezerva (fara ASCAD.h5).

    Modeleaza cazul real pe care SVD il rezolva: un MACRO-SEMNAL de ceas care
    variaza de la urma la urma (jitter de amplitudine) si care ingroapa
    scurgerea criptografica. Pentru ca macro-semnalul variaza intre urme, el NU
    dispare la centrare si domina atacul CPA brut -> atacul brut esueaza.
    Eliminand componenta singulara dominanta (sigma_1), SVD recupereaza
    scurgerea si atacul reuseste. Datele sintetice sunt NEMASCATE (atacam octetul 0).
    """
    print(f"[FALLBACK] Generez dataset sintetic realist ({num_traces}x{window_size})...")
    rng = np.random.default_rng(1337)

    plaintexts = rng.integers(0, 256, size=(num_traces, 16), dtype=np.uint8)
    real_key = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
                         0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], dtype=np.uint8)

    # Macro-ceas cu forma fixa, dar amplitudine (gain) aleatoare pe fiecare urma.
    t_axis = np.linspace(0, 8 * np.pi, window_size)
    macro = np.sin(t_axis)
    gain = rng.normal(40.0, 14.0, size=(num_traces, 1))
    traces = gain * macro[None, :]

    # Zgomot termic
    traces += 2.0 * rng.standard_normal((num_traces, window_size))

    # Scurgere S-Box slaba, injectata unde macro-semnalul este puternic (un varf al sinusului),
    # ca sa fie efectiv ingropata in urma bruta.
    target_time = window_size // 16
    for i in range(num_traces):
        leakage = HW_TABLE[AES_SBOX[plaintexts[i, 0] ^ real_key[0]]] - 4.0
        traces[i, target_time - 2:target_time + 3] += leakage * 1.0

    masks = np.zeros((num_traces, 16), dtype=np.uint8)  # date nemascate
    info = {"source": "synthetic", "target_byte": 0, "mask_col": None}
    return traces.astype(np.float64), plaintexts, real_key, masks, info


def extract_poi(filepath, num_traces=2000, time_window=(0, 700)):
    """
    Extrage Fereastra de Interes (POI) din baza ASCAD.
    Daca fisierul nu exista sau h5py lipseste, foloseste fallback-ul sintetic.

    Intoarce: traces, plaintexts, real_key, masks, info
      info = {"source", "target_byte", "mask_col"}
    """
    window_size = time_window[1] - time_window[0]

    # Baza reala e comisa in repo ca arhiva (ASCAD.zip). O dezarhivam automat
    # la prima rulare, ca proiectul sa mearga direct dupa un clone.
    if not os.path.exists(filepath) and os.path.exists("ASCAD.zip") and h5py is not None:
        try:
            print("[INFO] Dezarhivez ASCAD.zip -> ASCAD.h5 ...")
            with zipfile.ZipFile("ASCAD.zip", "r") as z:
                z.extractall(".")
        except Exception as e:
            print(f"[AVERTISMENT] Dezarhivarea ASCAD.zip a esuat: {e}")

    if not os.path.exists(filepath) or h5py is None:
        print(f"[AVERTISMENT] {filepath} nu a fost gasit sau h5py nu e instalat.")
        return generate_synthetic_data(num_traces, window_size)

    try:
        print(f"[INFO] Citesc date reale din {filepath}...")
        with h5py.File(filepath, "r") as in_file:
            grp = in_file["Profiling_traces"]
            traces = grp["traces"][0:num_traces, time_window[0]:time_window[1]].astype(np.float64)
            plaintexts = np.array(grp["metadata"]["plaintext"][0:num_traces])
            keys = np.array(grp["metadata"]["key"][0:num_traces])
            masks = np.array(grp["metadata"]["masks"][0:num_traces])
            real_key = keys[0]  # cheia este fixa in ASCAD

        info = {"source": "ascad",
                "target_byte": ASCAD_TARGET_BYTE,
                "mask_col": ASCAD_MASK_COL}
        return traces, plaintexts, real_key, masks, info
    except Exception as e:
        print(f"[EROARE] Citirea ASCAD a esuat: {e}")
        return generate_synthetic_data(num_traces, window_size)
