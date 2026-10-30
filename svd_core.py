# =============================================================================
# FISIERUL 1: svd_core.py
# =============================================================================
"""
Modul Core pentru Analiza Numerica.
Implementare SVD (Descompunerea la Valori Singulare) FARA functii predefinite
de descompunere (fara np.linalg.svd / eig).

Metoda:
  1. Formam matricea simetrica M = A^T A.
  2. Reducem M la forma tridiagonala prin reflexii Householder.
  3. Diagonalizam tridiagonala prin iteratia QR cu shift Wilkinson si deflatie.
  4. Valorile singulare sunt sqrt(valorilor proprii), iar vectorii singulari
     stangi U se obtin din U_i = (A v_i) / s_i.
"""

import numpy as np
import time


def tridiag_householder(A):
    """
    Reduce o matrice SIMETRICA A la forma tridiagonala T prin reflexii Householder.
    Intoarce (T, Q) cu Q ortogonala astfel incat Q^T A Q = T.
    Reflexiile se aplica pe submatrici (nu formam niciodata matricea H completa),
    deci costul ramane O(n^3), nu O(n^4).
    """
    n = A.shape[0]
    T = A.astype(np.float64).copy()
    Q = np.eye(n)
    for k in range(n - 2):
        x = T[k + 1:, k]
        nx = np.linalg.norm(x)
        if nx < 1e-300:
            continue
        # vectorul Householder v: reflecta x pe axa e1
        alpha = -np.sign(x[0]) * nx if x[0] != 0.0 else -nx
        v = x.copy()
        v[0] -= alpha
        nv = np.linalg.norm(v)
        if nv < 1e-300:
            continue
        v /= nv
        # T <- H T H, cu H = I - 2 v v^T, aplicat doar pe liniile/coloanele k+1..n-1
        w = T[k + 1:, :].T @ v
        T[k + 1:, :] -= 2.0 * np.outer(v, w)
        w = T[:, k + 1:] @ v
        T[:, k + 1:] -= 2.0 * np.outer(w, v)
        # acumulam transformarea in Q
        Q[:, k + 1:] -= 2.0 * np.outer(Q[:, k + 1:] @ v, v)
    return T, Q


def eigh_symmetric(A, tol=1e-13, max_sweeps=60):
    """
    Descompunere spectrala a unei matrici SIMETRICE, de la zero.
    Pas 1: tridiagonalizare Householder.
    Pas 2: iteratia QR (explicita) cu shift Wilkinson si deflatie, folosind
           rotatii Givens; vectorii proprii se acumuleaza in V.
    Intoarce (w, V) cu A ~= V diag(w) V^T, coloanele lui V ortonormate.
    """
    n = A.shape[0]
    T, V = tridiag_householder(A)
    hi = n - 1
    guard = 0
    guard_max = max_sweeps * n * n
    while hi > 0:
        # deflatie: daca sub-diagonala curenta e neglijabila, blocam valoarea proprie
        if abs(T[hi, hi - 1]) <= tol * (abs(T[hi, hi]) + abs(T[hi - 1, hi - 1])) + 1e-300:
            T[hi, hi - 1] = 0.0
            T[hi - 1, hi] = 0.0
            hi -= 1
            continue
        guard += 1
        if guard > guard_max:
            break
        # shift Wilkinson: valoarea proprie a blocului 2x2 din colt mai apropiata de T[hi,hi]
        a = T[hi - 1, hi - 1]
        b = T[hi, hi]
        c = T[hi, hi - 1]
        delta = (a - b) / 2.0
        if delta == 0.0:
            mu = b - abs(c)
        else:
            mu = b - (c * c) / (delta + np.sign(delta) * np.hypot(delta, c))
        # un pas QR explicit pe blocul activ 0..hi din (T - mu I), apoi T <- R Q + mu I
        idx = np.arange(hi + 1)
        T[idx, idx] -= mu
        cs = np.empty(hi)
        sn = np.empty(hi)
        # Q^T (T - mu I) = R  -- rotatii Givens pe linii
        for i in range(hi):
            r = np.hypot(T[i, i], T[i + 1, i])
            if r < 1e-300:
                cs[i] = 1.0
                sn[i] = 0.0
                continue
            c_ = T[i, i] / r
            s_ = T[i + 1, i] / r
            cs[i] = c_
            sn[i] = s_
            row_i = T[i, i:hi + 1].copy()
            row_j = T[i + 1, i:hi + 1].copy()
            T[i, i:hi + 1] = c_ * row_i + s_ * row_j
            T[i + 1, i:hi + 1] = -s_ * row_i + c_ * row_j
        # R Q  -- aceleasi rotatii pe coloane, acumulate si in V
        for i in range(hi):
            c_ = cs[i]
            s_ = sn[i]
            col_i = T[:hi + 1, i].copy()
            col_j = T[:hi + 1, i + 1].copy()
            T[:hi + 1, i] = c_ * col_i + s_ * col_j
            T[:hi + 1, i + 1] = -s_ * col_i + c_ * col_j
            vi = V[:, i].copy()
            vj = V[:, i + 1].copy()
            V[:, i] = c_ * vi + s_ * vj
            V[:, i + 1] = -s_ * vi + c_ * vj
        T[idx, idx] += mu
    w = np.diag(T).copy()
    return w, V


def SVD_Custom(A):
    """
    SVD de la zero.
    Metoda: diagonalizarea matricei de covarianta M = A^T A (simetrica, pozitiv
    semidefinita) cu solver-ul propriu de valori proprii.
    Returneaza: U (vectori singulari stangi), S (valori singulare descrescator),
                Vt (vectori singulari drepti, transpusi).
    """
    A = A.astype(np.float64)
    m, n = A.shape

    M = A.T @ A
    eigvals, V = eigh_symmetric(M)

    # erori FP pot produce valori proprii usor negative -> le taiem la 0
    eigvals[eigvals < 0] = 0.0

    # sortare descrescatoare
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = V[:, order]

    S = np.sqrt(eigvals)

    # U_i = (A v_i) / s_i
    U = np.zeros((m, n))
    for i in range(n):
        if S[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / S[i]
    return U, S, V.T


def benchmark_svd(A):
    """
    Validare academica a implementarii: compara timpul si precizia de
    reconstructie fata de numpy.linalg.svd (referinta LAPACK).
    """
    print("\n" + "=" * 50)
    print(" BAREM - BENCHMARKING SVD CUSTOM VS NUMPY")
    print("=" * 50)

    start = time.time()
    U_c, S_c, Vt_c = SVD_Custom(A)
    time_custom = time.time() - start
    A_rec_c = U_c @ np.diag(S_c) @ Vt_c
    err_custom = np.linalg.norm(A - A_rec_c, ord="fro")

    start = time.time()
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    time_np = time.time() - start
    A_rec_np = U_np @ np.diag(S_np) @ Vt_np
    err_np = np.linalg.norm(A - A_rec_np, ord="fro")

    # diferenta intre valorile singulare (semnatura corectitudinii)
    k = min(len(S_c), len(S_np))
    s_diff = np.max(np.abs(np.sort(S_c)[::-1][:k] - S_np[:k]))

    print(f"{'Metoda':<15} | {'Timp (s)':<10} | {'Eroare Reconstructie (Frob)'}")
    print("-" * 50)
    print(f"{'Custom (QR)':<15} | {time_custom:<10.4f} | {err_custom:.4e}")
    print(f"{'Numpy (LAPACK)':<15} | {time_np:<10.4f} | {err_np:.4e}")
    print(f"Diferenta maxima intre valorile singulare: {s_diff:.4e}")
    print("=" * 50 + "\n")

    return U_c, S_c, Vt_c
