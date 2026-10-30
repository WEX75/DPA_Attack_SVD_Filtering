# Atac Diferential de Putere (DPA/CPA) cu filtrare Truncated SVD

Proiect de Calcul Numeric: un atac de canal secundar de tip CPA (Correlation Power
Analysis) asupra AES-128, care foloseste o filtrare prin **Truncated SVD** pentru a
curata urmele de consum de putere. Punctul central numeric este o implementare a
descompunerii SVD **de la zero** (reflexii Householder + iteratia QR cu shift
Wilkinson), fara `numpy.linalg.svd` sau `eig`.

Baza de date folosita este **ASCAD** (publica, academica), capturata de pe un
microcontroler ATmega8515 care executa AES-128.

## Fisierele proiectului

* **`svd_core.py`** - nucleul numeric. SVD implementat de la zero prin:
  tridiagonalizare Householder a matricei de covarianta `M = A^T A`, apoi iteratia
  QR cu shift Wilkinson si deflatie (rotatii Givens) pentru diagonalizare. Contine
  si `benchmark_svd`, care compara precizia si timpul fata de `numpy.linalg.svd`.
* **`ascad_handler.py`** - gestionarea datelor. Citeste fereastra de interes (POI)
  din ASCAD; daca baza lipseste, genereaza date sintetice realiste.
* **`dpa_pipeline.py`** - orchestratorul. Ruleaza benchmark-ul, filtreaza prin
  Truncated SVD, executa atacul CPA pe datele brute si filtrate si salveaza raportul
  grafic `raport_dpa.png`.
* **`ASCAD.zip`** - baza de date reala (arhivata). Se dezarhiveaza automat la prima
  rulare in `ASCAD.h5`.

## Cum se ruleaza

```bash
pip install -r requirements.txt
python dpa_pipeline.py
```

La prima rulare, `ASCAD.zip` este dezarhivat automat in `ASCAD.h5`. Daca `h5py`
lipseste sau baza nu e gasita, pipeline-ul cade pe datele sintetice.

## Detalii importante despre ASCAD (de ce esueaza un atac naiv)

Varianta pre-procesata `ASCAD.h5` contine deja fereastra de 700 de esantioane
extrasa in jurul operatiei S-Box pentru **al 3-lea octet de stare (index 2)**. Doua
consecinte practice, pe care implementarea le trateaza explicit:

1. **Octetul tintit este index 2**, nu index 0. Fereastra nu contine scurgerea
   octetului 0, deci un atac pe octetul 0 nu are ce gasi.
2. **Implementarea este mascata** (boolean masking). Valoarea care scurge in aceasta
   fereastra este `Sbox(p[2] ^ k[2]) XOR masca_de_iesire`, nu `Sbox(p[2] ^ k[2])`.
   Masca de iesire a S-Box-ului pentru octetul 2 este coloana `masks[:, 0]` din
   metadata. Un CPA de ordinul 1 care ignora masca ramane la nivelul zgomotului
   (corelatie ~0.05). Folosind masca, cheia corecta iese clar (corelatie ~0.84).

Folosirea mastii cunoscute din metadata este o validare de ordinul 1 standard pentru
ASCAD: scopul proiectului este filtrarea SVD, nu spargerea contramasurii de mascare.

## Ce arata rezultatele

* **Date reale ASCAD**: atacul recupereaza octetul de cheie (byte 2). Truncated SVD
  (eliminarea componentei singulare dominante `sigma_0`, macro-zgomotul) aduce o
  imbunatatire a marjei de corelatie fata de atacul brut.
* **Date sintetice**: macro-zgomotul variaza de la urma la urma si ingroapa
  scurgerea, deci atacul **brut esueaza**. Dupa eliminarea lui `sigma_0` prin
  Truncated SVD, atacul **reuseste** - demonstratia clara a rolului filtrarii SVD.

## Concepte

**CPA (Correlation Power Analysis).** Pentru fiecare dintre cele 256 de valori
posibile ale unui octet de cheie se construieste un model ipotetic de consum
(greutatea Hamming a iesirii S-Box) si se calculeaza corelatia Pearson cu consumul
real masurat. Ipoteza cu cel mai mare scor absolut de corelatie este cheia corecta.

**De ce SVD.** Urmele reale contin un macro-zgomot (ceas, drift de amplitudine)
mult mai energic decat scurgerea criptografica. Acesta se concentreaza in prima
valoare singulara `sigma_0`. Eliminand aceasta componenta si reconstruind matricea
din componentele de rang mic, raportul semnal/zgomot creste si atacul CPA devine mai
robust.
