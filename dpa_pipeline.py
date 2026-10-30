# =============================================================================
# FISIERUL 3: dpa_pipeline.py
# =============================================================================
"""
Pipeline-ul complet pentru Atacul DPA cu Truncated SVD.
Ruleaza benchmark-ul SVD, filtreaza datele prin Truncated SVD (eliminarea
componentei singulare dominante = macro-zgomot), ataca AES prin CPA (corelatie
Pearson) si genereaza raportul grafic.

Model de scurgere:
  - Date reale ASCAD (mascate): HW( Sbox(p[b] ^ k) XOR masca_iesire ), octet b=2.
  - Date sintetice (nemascate): HW( Sbox(p[b] ^ k) ), octet b=0.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend fara display (ruleaza si pe server)
import matplotlib.pyplot as plt

from svd_core import benchmark_svd, SVD_Custom
from ascad_handler import extract_poi, AES_SBOX, hw_vec

# Cate componente singulare pastram dupa eliminarea macro-zgomotului (sigma_1).
KEEP_FROM = 1     # aruncam componenta 0 (macro-ceasul dominant)
KEEP_TO = 30      # pastram componentele 1..29 (variatiile fine = scurgerea)


def calculate_cpa(traces, plaintexts, target_byte, mask):
    """
    Atac CPA (Correlation Power Analysis) pe octetul 'target_byte'.
    'mask' este vectorul mastii de iesire (0 pentru date nemascate).
    Intoarce matricea de corelatii (256 x num_samples).
    """
    num_traces, num_samples = traces.shape
    correlations = np.zeros((256, num_samples))

    # Centram si normalizam urmele o singura data
    traces_centered = traces - np.mean(traces, axis=0)
    traces_std = np.std(traces_centered, axis=0)
    traces_std[traces_std == 0] = 1.0

    pt_col = plaintexts[:, target_byte].astype(np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)

    for key_guess in range(256):
        # Model ipotetic: HW al iesirii S-Box (mascate, daca e cazul)
        sbox_out = AES_SBOX[(pt_col ^ key_guess).astype(np.uint8)]
        hyp = hw_vec((sbox_out ^ mask).astype(np.uint8))

        hyp_centered = hyp - np.mean(hyp)
        hyp_std = np.std(hyp_centered)
        if hyp_std > 0:
            cov = np.sum(traces_centered * hyp_centered[:, np.newaxis], axis=0)
            correlations[key_guess, :] = cov / (num_traces * traces_std * hyp_std)

    return correlations


def summarize(corr, real_key_byte, label):
    """Afiseaza cel mai bun candidat, rangul cheii corecte si marja de corelatie."""
    peaks = np.max(np.abs(corr), axis=1)
    best = int(np.argmax(peaks))
    rank = int((peaks > peaks[real_key_byte]).sum())  # 0 = cheia corecta e pe primul loc
    second = np.sort(peaks)[-2]
    margin = peaks[best] - second
    ok = (best == real_key_byte)
    print(f"  [{label}] ghicit={hex(best):>4}  corect={hex(real_key_byte):>4}  "
          f"rang_cheie={rank:3d}  varf={peaks[best]:.3f}  marja={margin:+.3f}  "
          f"{'-> SUCCES' if ok else '-> esec'}")
    return best, ok


def main():
    # ------------------------------------------------------------------
    # PASUL 1: Validare SVD de la zero (barem)
    # ------------------------------------------------------------------
    print(">>> PASUL 1: Validare SVD de la zero")
    test_mat = np.random.default_rng(0).standard_normal((100, 50))
    _ = benchmark_svd(test_mat)

    # ------------------------------------------------------------------
    # PASUL 2: Incarcare date
    # ------------------------------------------------------------------
    print(">>> PASUL 2: Incarcare date hardware")
    ASCAD_PATH = "ASCAD.h5"
    traces, plaintexts, real_key, masks, info = extract_poi(
        ASCAD_PATH, num_traces=2000, time_window=(0, 700))

    target_byte = info["target_byte"]
    mask_col = info["mask_col"]
    mask = masks[:, mask_col] if mask_col is not None else np.zeros(len(traces), dtype=np.uint8)
    real_key_byte = int(real_key[target_byte])

    print(f"    Sursa date         : {info['source']}")
    print(f"    Octet tintit       : {target_byte}")
    print(f"    Cheia reala (byte {target_byte}): {hex(real_key_byte)}")
    print(f"    Mascare            : {'da (masks[:, %d])' % mask_col if mask_col is not None else 'nu'}")

    # ------------------------------------------------------------------
    # PASUL 3: Filtrare Truncated SVD (de la zero)
    # ------------------------------------------------------------------
    print("\n>>> PASUL 3: Filtrare Truncated SVD (eliminarea macro-zgomotului)")
    U, S, Vt = SVD_Custom(traces)

    keep_to = min(KEEP_TO, len(S))
    print(f"    Pastram componentele singulare {KEEP_FROM}..{keep_to - 1} "
          f"(aruncam sigma_0 = {S[0]:.1f}, macro-ceasul dominant)")

    traces_centered = traces - np.mean(traces, axis=0)
    A_clean = np.zeros_like(traces, dtype=np.float64)
    for idx in range(KEEP_FROM, keep_to):
        A_clean += S[idx] * np.outer(U[:, idx], Vt[idx, :])

    # ------------------------------------------------------------------
    # PASUL 4: Atac CPA pe date brute vs. filtrate
    # ------------------------------------------------------------------
    print("\n>>> PASUL 4: Executie atac CPA (Pearson)")
    corr_raw = calculate_cpa(traces_centered, plaintexts, target_byte, mask)
    corr_clean = calculate_cpa(A_clean, plaintexts, target_byte, mask)

    _, _ = summarize(corr_raw, real_key_byte, "BRUT ")
    best_clean, ok_clean = summarize(corr_clean, real_key_byte, "SVD  ")

    if ok_clean:
        print(f"\n[SUCCES] Cheia AES (octet {target_byte}) a fost extrasa: {hex(best_clean)}")
    else:
        print(f"\n[ESEC] Atacul nu a izolat cheia (necesita mai multe urme).")

    # ------------------------------------------------------------------
    # PASUL 5: Raport grafic
    # ------------------------------------------------------------------
    print("\n>>> PASUL 5: Generare raport grafic")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.set_facecolor("#0a0a0a")
    for ax in axes:
        ax.set_facecolor("#0f0f14")
        ax.grid(color="#1a1a2e", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panou 1: urma bruta vs. filtrata
    axes[0].plot(traces_centered[0], color="#ff4455", alpha=0.6, label="Urma bruta (centrata)")
    axes[0].plot(A_clean[0], color="#22dd88", linewidth=1.5, label="Urma filtrata (Truncated SVD)")
    axes[0].set_title("Efectul SVD asupra unei urme", color="white")
    axes[0].legend(loc="upper right")

    # Panou 2: spectrul valorilor singulare (rosu = taiat, verde = pastrat)
    n_display = min(30, len(S))
    kept = set(range(KEEP_FROM, keep_to))
    colors = ["#04bd07" if i in kept else "#b2000f" for i in range(n_display)]
    axes[1].bar(range(n_display), S[:n_display], color=colors, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_title("Spectrul valorilor singulare (rosu = taiat, verde = pastrat)", color="white")
    axes[1].set_ylabel("Magnitudine (log)")

    # Panou 3: CPA pe datele filtrate, cheia corecta evidentiata
    for k in range(256):
        if k == real_key_byte:
            continue
        axes[2].plot(np.abs(corr_clean[k]), color="#555566", linewidth=0.4, alpha=0.5)
    axes[2].plot(np.abs(corr_clean[real_key_byte]), color="#ff4455", linewidth=2.5,
                 zorder=10, label=f"Cheia corecta ({hex(real_key_byte)})")
    axes[2].set_title("Rezultatul atacului CPA pe $A_{clean}$", color="white")
    axes[2].set_xlabel("Esantion de timp")
    axes[2].set_ylabel("Corelatie absoluta")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    out = "raport_dpa.png"
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    print(f"    Grafic salvat in '{out}'. Pipeline finalizat.")


if __name__ == "__main__":
    main()
