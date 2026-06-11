"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta. 
"""



"""
Tesla_Scalar_1.py  —  SLW motor (cista fizika)

Simulira skalarno-longitudinalni talas (SLW):
- skalarno polje S(x, t) koje se prostire u +x pravcu (d'Alamberova jednacina),
- uzduzno polje E_x = -dS/dx  (gradient-driven, u pravcu prostiranja).

Teorija (Tesla scalar / longitudinalni talasi)
Ima recenziran rad o Extended Electrodynamics (EED)
koji formalno predviđa scalar-longitudinal wave (SLW)
— talas sa električnim poljem u pravcu prostiranja (MDPI Symmetry, 2020).
Čak postoji i američki patent (9,306,527) za antene za takve talase.
Dakle, nije „samo pseudonauka" — ima ozbiljnih pokušaja formalizacije.
"""


from pathlib import Path

import numpy as np
import pandas as pd

SEED = 39

# Tezine za kombinovani skor (talas vs prava frekvencija).
W_TALAS = 0.7
W_FREQ = 0.3

# Korak 2: primena talasa na CSV.
CSV_PATH = Path("/data/loto7hh_4630_k46.csv")
MIN_BROJ = 1
MAX_BROJ = 39
KOLONE = [f"Num{i}" for i in range(1, 8)]
OUTPUT_DIR = Path(__file__).resolve().parent


def simuliraj_slw(
    duzina=200.0,     # duzina domena (proizvoljne jedinice)
    nx=4630,          # broj tacaka po prostoru (= broj kombinacija u CSV)
    c=1.0,            # brzina prostiranja talasa
    cfl=0.5,          # Courant broj (stabilnost: <= 1)
    koraka=600,       # broj vremenskih koraka
    centar=40.0,      # pocetni polozaj pulsa
    sirina=6.0,       # sirina pocetnog Gaussovog pulsa
):
    """Vrati (x, S, E_x): polozaj, skalarno polje i uzduzno polje na kraju simulacije."""
    dx = duzina / (nx - 1)
    dt = cfl * dx / c
    x = np.linspace(0.0, duzina, nx)

    # Pocetni Gaussov puls koji se krece u +x (zadajem S i njegov pomak unazad).
    def gauss(xx):
        return np.exp(-((xx - centar) ** 2) / (2.0 * sirina ** 2))

    S_prev = gauss(x)
    S_curr = gauss(x - c * dt)

    r2 = (c * dt / dx) ** 2
    for _ in range(koraka):
        S_next = np.empty_like(S_curr)
        S_next[1:-1] = (
            2.0 * S_curr[1:-1]
            - S_prev[1:-1]
            + r2 * (S_curr[2:] - 2.0 * S_curr[1:-1] + S_curr[:-2])
        )

        # Otvorene granice: prost prenos susedne vrednosti da puls ne eksplodira na ivici.
        S_next[0] = S_next[1]
        S_next[-1] = S_next[-2]

        S_prev, S_curr = S_curr, S_next

    E_x = -np.gradient(S_curr, dx)
    return x, S_curr, E_x


def glavne_mere(S, E_x):
    """Vrati osnovne mere polja: maksimum S, maksimum |E_x| i energijsku gustinu."""
    gustina_energije = 0.5 * (S ** 2 + E_x ** 2)
    return {
        "max_S": float(np.max(S)),
        "max_abs_E_x": float(np.max(np.abs(E_x))),
        "ukupna_gustina_energije": float(np.sum(gustina_energije)),
    }


# ---------------------------------------------------------------------------
# Korak 2: primena SLW talasa na loto CSV (ne-frekvencijski)
# ---------------------------------------------------------------------------
def ucitaj_izvlacenja(csv_path=CSV_PATH):
    """Ucitaj CSV i vrati listu izvlacenja (svako je lista od 7 brojeva), hronoloski."""
    df = pd.read_csv(csv_path)
    df = df[KOLONE].astype(int)
    return df.to_numpy().tolist()


def ne_frekvencijski_skor(izvlacenja, energija):
    """Za svaki broj 1..39 vrati PROSECNU energiju talasa na mestima gde se pojavio.

    Nije frekvencija (koliko puta), nego gde u talasu broj "lezi" (rezonanca).
    """
    zbir = {b: 0.0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    pojave = {b: 0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for i, red in enumerate(izvlacenja):
        for b in red:
            zbir[b] += energija[i]
            pojave[b] += 1
    skor = {b: (zbir[b] / pojave[b] if pojave[b] > 0 else 0.0) for b in zbir}
    return skor, pojave


def frekvencija_brojeva(izvlacenja):
    """Prava frekvencija: koliko se svaki broj 1..39 pojavio nad svim izvlacenjima.

    Vraca (udeo, pojave): udeo = relativna frekvencija (zbir = 1), pojave = sirovi broj.
    """
    pojave = {b: 0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for red in izvlacenja:
        for b in red:
            pojave[b] += 1
    ukupno = sum(pojave.values())
    udeo = {b: (pojave[b] / ukupno if ukupno > 0 else 0.0) for b in pojave}
    return udeo, pojave


def _na_0_1(vrednosti):
    """Min-max skaliranje recnika na opseg 0..1 (ravno na 0 ako su sve jednake)."""
    v = np.array(list(vrednosti.values()), dtype=float)
    raspon = v.max() - v.min()
    if raspon <= 0:
        return {k: 0.0 for k in vrednosti}
    return {k: (vrednosti[k] - v.min()) / raspon for k in vrednosti}


def kombinovani_skor(talas_skor, udeo, w_talas=W_TALAS, w_freq=W_FREQ):
    """Tezinska kombinacija: normalizuj obe komponente na 0..1, pa ponderisan zbir."""
    t = _na_0_1(talas_skor)
    f = _na_0_1(udeo)
    return {b: w_talas * t[b] + w_freq * f[b] for b in talas_skor}


def izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED):
    """Izaberi 7/39 kombinacije, ponderisano ne-frekvencijskim skorom."""
    rng = np.random.default_rng(seed)
    brojevi = np.array(list(skor.keys()))
    tezine = np.array([skor[b] for b in brojevi], dtype=float)
    if tezine.sum() <= 0:
        tezine = np.ones_like(tezine)
    p = tezine / tezine.sum()

    kombinacije, vidjeno, pokusaji = [], set(), 0
    while len(kombinacije) < broj_kombinacija and pokusaji < broj_kombinacija * 300:
        pokusaji += 1
        izbor = tuple(sorted(rng.choice(brojevi, size=7, replace=False, p=p).tolist()))
        if izbor not in vidjeno:
            vidjeno.add(izbor)
            kombinacije.append(izbor)
    return kombinacije


def skor_kombinacije(kombinacija, skor):
    """Skor kombinacije je zbir skorova njenih 7 brojeva."""
    return float(sum(skor[b] for b in kombinacija))


def main():
    # --- Korak 1: SLW motor ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_slw(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / SLW motor - korak 1")
    print("Talas: skalarno polje S(x,t), prostiranje u +x pravcu")
    print("Uzduzno polje: E_x = -dS/dx")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    # Opadajuce po pravoj frekvenciji, pa po velicini broja.
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )

    with open(OUTPUT_DIR / "tesla_scalar_1.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - korak 2 (tezinski: talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

    print()
    print("\nTesla Scalar - korak 2 (tezinski: talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")
    
    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")
    
    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / 'tesla_scalar_1.txt'}")
    print()


if __name__ == "__main__":
    main()



"""

Tesla Scalar / SLW motor - korak 1
Talas: skalarno polje S(x,t), prostiranje u +x pravcu
Uzduzno polje: E_x = -dS/dx

broj tacaka: 4630
max S: 0.9999985007
max |E_x|: 0.1010865969
ukupna gustina energije: 124.7797101951



Tesla Scalar - korak 2 (tezinski: talas + prava frekvencija)
CSV: /data/loto7hh_4630_k46.csv | Izvlacenja: 4630 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  34  skor=0.9229166667  freq=0.02694  (pojava=873)
   x  skor=0.8187868171  freq=0.02601  (pojava=843)
  08  skor=0.7620207698  freq=0.02808  (pojava=910)
   y  skor=0.6944234425  freq=0.02549  (pojava=826)
  13  skor=0.6710366593  freq=0.02555  (pojava=828)
   z  skor=0.6134945057  freq=0.02561  (pojava=830)
  33  skor=0.6122469155  freq=0.02635  (pojava=854)
  37  skor=0.5777269684  freq=0.02654  (pojava=860)
  03  skor=0.5608625538  freq=0.02546  (pojava=825)
  25  skor=0.5588051505  freq=0.02589  (pojava=839)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   910  | 0.02808
    x  |   905  | 0.02792
   34  |   873  | 0.02694
    y  |   869  | 0.02681
   37  |   860  | 0.02654
    z  |   860  | 0.02654
   32  |   857  | 0.02644
    x  |   854  | 0.02635
   22  |   851  | 0.02626
    y  |   849  | 0.02620
   29  |   848  | 0.02616
    z  |   845  | 0.02607
   35  |   843  | 0.02601
   09  |   843  | 0.02601
    x  |   842  | 0.02598
   07  |   842  | 0.02598
    y  |   840  | 0.02592
   25  |   839  | 0.02589
    z  |   837  | 0.02583
   31  |   830  | 0.02561
   13  |   828  | 0.02555
   05  |   828  | 0.02555
   21  |   826  | 0.02549
   03  |   825  | 0.02546
   02  |   824  | 0.02542
   28  |   820  | 0.02530
   18  |   820  | 0.02530
   06  |   816  | 0.02518
   19  |   813  | 0.02508
   04  |   812  | 0.02505
   12  |   810  | 0.02499
   14  |   809  | 0.02496
   15  |   797  | 0.02459
   27  |   788  | 0.02431
   01  |   788  | 0.02431
   30  |   787  | 0.02428
   36  |   786  | 0.02425
   20  |   770  | 0.02376
   17  |   766  | 0.02363
  ukupno pojava: 32410


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 06 x 22 y 34 z 39  skor_komb=4.1828743121
  02. 07 x 13 y 24 z 34  skor_komb=4.1724896390
  03. 08 x 17 y 23 z 34  skor_komb=3.9744379737
  04. 02 x 16 y 26 z 39  skor_komb=3.7826283876
  05. 17 x 28 y 31 z 34  skor_komb=3.7645748441
  06. 10 x 14 y 23 z 34  skor_komb=3.5593504018
  07. 04 x 21 y 30 z 37  skor_komb=3.4756849693
  08. 06 x 13 y 32 z 39  skor_komb=3.4360797056
  09. 02 x 08 y 19 z 39  skor_komb=3.1304011568
  10. 09 x 11 y 30 z 38  skor_komb=2.8350171823

Sacuvano: /Tesla/tesla_scalar_1.txt

"""






"""
čist SLW motor — skalarni talas koji se prostire u +x pravcu (d'Alamberov talas), 
skalarno polje S(x,t) koje se prostire u +x pravcu 
plus uzdužno polje E_x = -∂S/∂x (gradient-driven, kako EED i opisuje). 
osnovne mere: max S, max |E_x|, ukupna gustina energije
Bez frekvencijske logike, čista fizika.

Ključna ideja koraka 2 (ne-frekvencijska): 
SLW talas iz koraka 1 prostire se preko 4630 izvlačenja (1 tačka = 1 izvlačenje). 
Za svaki broj 1-39 računa ne-frekvencijski skor =  prosečna energija talasa na pozicijama gde se taj broj pojavio 
— dakle ne koliko puta (frekvencija), nego gde u talasu leži (rezonanca sa poljem).
ne_frekvencijski_skor() — prosečna energija talasa po broju (ne frekvencija)
Biram 10 kombinacija ponderisano tim skorom (seed 39).
izaberi_kombinacije() — 10 kombinacija ponderisano tim skorom (seed 39)

skorovi su međusobno vrlo blizu (0.0285-0.0329) 
— talas trenutno slabo razdvaja brojeve. 
Razlog: uzimam prosečnu energiju po broju, 
a kako se svaki broj pojavljuje na ~800+ 
različitih pozicija duž celog talasa, 
proseci se izravnaju i ispadnu skoro isti za sve.
To znači da, iako jeste ne-frekvencijski po definiciji, 
talas još nema dovoljno „oštrine" da napravi jasnu razliku.
Zato model koristi pravu frekvenciju pojavljivanja brojeva nad svih 4630 izvlačenja.
frekvencija_brojeva(izvlacenja) — broji koliko se svaki broj 1-39 pojavio i vraća relativnu frekvenciju (udeo, zbir = 1) i sirov broj (pojave).
kombinovani_skor na normalizovanu težinsku kombinaciju (svaka komponenta skalirana na 0-1, pa ponderisan zbir).
Obe komponente se skaliraju na 0-1 (_na_0_1), pa ponderisan zbir: skor = W_TALAS·talas + W_FREQ·freq.
Težine na vrhu fajla: W_TALAS = 0.7, W_FREQ = 0.3 (lako menjam).

Logika: 
cela ideja je ne-frekvencijska (Tesla talas nosi razliku), 
a kod 7/39 nad 4630 izvlačenja frekvencija je skoro ravna (svaki broj ~ isti očekivani broj pojava) 
— pa frekvencija služi samo kao blagi stabilizator, ne kao glavni signal. 
Zato talas treba da dominira.

Postavljeno: 
talas 0.7 / freq 0.3 — talas dominira (ne-frekvencijska ideja), frekvencija samo blago koriguje. 

Frekvencija je realna: 
ukupno ima 4630 x 7 = 32410 pojavljivanja, očekivano po broju je oko 831. 

Zato su ove vrednosti normalne:
08 = 910 je stvarno jači po frekvenciji.
x = 873 nije najjači po frekvenciji, ali je prvi po skoru, znači talas ga je podigao.
y = 826, z = 828, 31 = 830 su skoro prosečni po frekvenciji, ali su visoko, opet znači da talas radi svoj deo.
Zaključak: odnos 0.7 talas / 0.3 freq nije ubio Teslinu ideju. 
Frekvencija samo stabilizuje, a talas i dalje odlučuje.

broj 34 se previše ponavlja u kombinacijama. 
Sledeći korak mozda da bude kontrola raznovrsnosti kombinacija, da top broj ne upada u pola liste.

Tabela pravih frekvencija je sortirana opadajuće — primarno po frekvenciji (pojava), a kod istog broja pojava po većem broju prvo. Isto u txt i u konzoli.

Sad imam dobru osnovu: 
SLW motor radi na 4630 tačaka.
Talas je primenjen na CSV.
Prava frekvencija je izračunata i prikazana.
Kombinovani skor je težinski 0.7 talas / 0.3 freq.
Txt izlaz je pregledan i potpun.

Svaku kombinaciju ocenim (npr. zbir/prosek skorova njenih 7 brojeva) i sortiramo opadajuće. 
Svaka kombinacija dobije svoj skor i da lista bude sortirana od najjače ka slabijoj, pa će prva zaista biti favorit.

10 kombinacija se prvo generiše, zatim se svaka ocenjuje sa skor_komb = zbir skorova 7 brojeva i sortira opadajuće. 
Zato je kombinacija 01 sada favorit, a deseta je najslabija od ponuđenih.

Kombinacije su sad sortirane opadajuće po skor_komb: od 4.183 (prva) do 2.835 (deseta).
Favorit je 01: 06 x 22 y 34 z 39 (skor_komb=4.1828).
Najslabija ponuđena je 10: 09 x 11 y 30 z 38 (2.8350).

Logika je konzistentna: 
prva kombinacija ima najviše jakih brojeva po kombinovanom skoru 
(34, x, 25, y, 39 su svi visoko u top listi), 
a deseta ima brojeve nižeg ranga.
"""
