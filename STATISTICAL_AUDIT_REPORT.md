# Kompleksowy Raport Audytu Statystycznego
## UAP-Canyon Spatial Association Analysis

**Data audytu:** 2026-02-18
**Zakres:** Caly pipeline analityczny — Fazy B, C, Sprint 1-3, D, D7/D8, E (v1/v2/E-RED), confound tests, replication suite
**Metoda:** Linia-po-linii przeglad ~65 skryptow Python, weryfikacja zalozen statystycznych, kontrola wycieku danych, sprawdzenie spojnosci miedzy fazami

---

## PODSUMOWANIE WYKONAWCZE

### Pozytywnie zweryfikowane elementy
1. **Haversine** w E-RED v2 — prawidlowa implementacja odleglosci sferycznych
2. **Spearman rho** — prawidlowy test korelacji rangowej, odporny na outliers i nieparametryczny
3. **BH-FDR** — poprawna implementacja korekty wielokrotnych porownania (Phase C)
4. **Night detection** — prawidlowa logika `prev_setting > prev_rising` (ephem)
5. **Moon illumination** — porownanie z astronomiczna dystrybucja (nie z rozkladem jednostajnym)
6. **Scoring function S** — dobrze zdefiniowana: S = mean(rank_G + rank_P + rank_C), poprawne rankowanie
7. **Deduplikacja (D3)** — OR stabilne ~5.0-5.1 we wszystkich wariantach
8. **Sezonalnosc (D5)** — zaden miesiac nie napedza efektu, wykluczenie zdarzen nie zmienia OR
9. **Pileup correction (D2)** — OR 5.18 -> 5.10 po kolapsie — geocoding nie napedza sygnalu
10. **E_i normalization** — `sum(E_i) = sum(O_i)` — standardowa normalizacja dla modelu Poissona
11. **Bootstrap CI** — percentylowy bootstrap (N=2000) dla Spearman rho — standardowy i reprodukowalny

### Wykryte bledy i nieprawidlowosci

Zidentyfikowano **47 problemow** o nastepujacych poziomach waznosci:

| Waznosc | Liczba | Opis |
|---------|--------|------|
| **CRITICAL** | 5 | Bledy wplywajace na glowne wnioski |
| **HIGH/MAJOR** | 14 | Istotne problemy metodologiczne |
| **MEDIUM/MODERATE** | 17 | Problemy wplywajace na dokladnosc, nie na kierunek |
| **LOW/MINOR** | 7 | Drobne niedokladnosci |
| **NOTE/INFO** | 4 | Kwestie informacyjne |

---

## SEKCJA 1: BLEDY KRYTYCZNE (CRITICAL)

### CRIT-1: E_i w replication suite uzywa rownych wag (pop=1 dla wszystkich komorek)
**Lokalizacja:** `phase_e_replication_suite.py`, linia 110, 119, 123
**Opis:** Funkcja `compute_s_logr()` pobiera `cell.get('pop', 1)`, ale JSON z E-RED v2 nie zawiera pola `pop`. Wartosc defaultowa `1` jest uzywana dla kazdej komorki. Skutek: `E_i = total_obs / n_cells` — rownomierny rozklad oczekiwanych raportoow.
**Wplyw:** Testy 6a (temporal split), 6d (temporal stability) i 6e (post-2014) koreluja S z log10(O_i), NIE z population-adjusted logR. Komorka kolo Seattle (4M ludzi) i komorka na wsi w Oregon dostaja identyczne E_i. Pozytywny wynik moze odzwierciedlac po prostu fakt, ze komorki canyon sa blisko duzych miast.
**Rekomendacja:** Zapisac E_i w cell_details lub zaimportowac model populacyjny z E-RED v2.

### CRIT-2: LOO CV uzywa logR z pelnych danych (data leakage)
**Lokalizacja:** `phase_e_replication_suite.py`, linie 317-349
**Opis:** Leave-one-region-out CV uzywa logR wartosci wyliczonych z CALEGO datasetu 1990-2014. Holdout region nadal wnosi swoje raporty do globalne normalizacji E_i. Raporty z held-out regionu wplywaja na total_O i normalizacje.
**Wplyw:** Wyniki LOO (mean held-out rho=0.380, 3/5 positive) sa NIEWAZNE jako cross-validation. Mierza jedynie "czy relacja S-logR z pelnych danych utrzymuje sie w podzbiorach geograficznych."
**Rekomendacja:** Dla kazdego foldu przeliczyc O_i z podzbioru treningowego/testowego, przeliczyc normalizacje E_i, dopiero potem policzyc logR.

### CRIT-3: Spatial forward prediction — ta sama data leakage
**Lokalizacja:** `phase_e_replication_suite.py`, linie 204-285
**Opis:** Identyczny problem jak CRIT-2. logR dla Puget Sound wyliczone z pelnych danych, a potem uzyte jako "test set."
**Wplyw:** Puget rho_oos=0.637 (p=0.011) jest prawdopodobnie zawyzone. SoCal rho=0.260 (p=0.415, WRONG direction) moze byc bardziej wiarygodne.
**Rekomendacja:** j.w. — oddzielna normalizacja E_i dla train/test.

### CRIT-4: Prog 60 m/km wybrany z danych (double-dipping)
**Lokalizacja:** Sprint 3 (`sprint3_temporal_doseresponse.py`) -> Phase D (`phase_d_robustness.py`:55-56) -> Phase E v2 (`phase_ev2_scoring.py`:53)
**Opis:** Gradient bins [0, 10, 30, 60, 500] testowane w Sprint 3. Bin 60+ m/km dal najsilniejszy OR. Ten prog zostal przeniesiony do Phase E v2 jako estimand. Jest to **post-selection inference** — efekt jest zawyzon bo wybrano najlepszy bin.
**Wplyw:** OR ~ 5.09 dla 60+ m/km jest optymistyczny. Potrzebna korekta Bonferroniego na 4 testowane biny, lub niezalezny holdout do walidacji progu.
**Rekomendacja:** 1) Korekta Bonferroniego (4 biny -> p * 4); 2) Temporal split: pre-2005 discovery, post-2005 validation; 3) Otwarcie uznanie w publikacji.

### CRIT-5: C3 permutation test porzucony — brak poprawnego modelu zerowego dla sejsmiki
**Lokalizacja:** `phase_c_prompt2.py`, linie 363-378
**Opis:** Prompt specyfikowal test permutacyjny: "shuffle report dates, keep locations, compute quake_count_300km_7d, compare observed mean vs shuffled." Implementacja shuffle'uje kolumne quake_count (co zachowuje srednia) i natychmiast porzuca to podejscie. Fallback to miesieczny Spearman (r=0.019, p=0.74) — slabsza alternatywa.
**Wplyw:** Test C3 ("null" dla sejsmiki) nie ma wlasciwego testu hipotezy zerowej.
**Rekomendacja:** Zaimplementowac permutacje dat raportow zachowujac lokalizacje.

---

## SEKCJA 2: BLEDY WAZNE (HIGH / MAJOR)

### HIGH-1: Odwrocenie kierunku B v1 -> B v2
**Lokalizacja:** `uap_ocean_analysis_phase_b.py` vs `uap_ocean_phase_b_v2.py`
**Opis:** Phase B v1 znalazl OR=0.484 (UAP DALEJ od kanionow). Phase B v2 znalazl OR=5.30 (UAP BLIZEJ). Kazda zmiana metodologiczna (strefa, populacja, definicja kanionu) dzialala w kierunku wzmocnienia efektu.
**Wskazniki confirmation bias:** Wszystkie zmiany jednoczesnie zwiekszaja efekt. Badania kontynuowano w kierunku pozytywnego wyniku.
**Wskazniki PRZECIW:** Sekcja 4 v2 jest autokrytyczna, distance-matched test ujawnia confinement do 0-25km, wyraznie nazwano to "red flag."
**Werdykt:** Confirmation-bias-PRONE ale nie confirmation-bias-DRIVEN. Transparency zachowany.

### HIGH-2: Efekt canyon jest ograniczony do pasa 0-25 km od brzegu
**Lokalizacja:** `results_v2_robustness.json`, sekcja "distance_matched"
**Opis:** 0-25 km: OR=2.63; 25-50 km: OR=0.12; 50+ km: OR=0.0. Kaniony ktore dochodza do linii brzegowej sa geograficznie skorelowane z portami, ujsciami rzek i centrami populacji (La Jolla, Monterey, Hudson/NYC).
**Wplyw:** Sygnal moze byc populacyjnym confoundem ktory county-centroid proxy nie wychwytuje.

### HIGH-3: Post-2014 geocoding przypisuje identyczne wspolrzedne per miasto
**Lokalizacja:** `phase_e_replication_suite.py`, linie 388-441
**Opis:** Post-2014 dane geokodowane via lookup city/state -> identyczne koordynaty jak historyczne dane. "Bellevue, WA" w 2020 laduje na identyczne wspolrzedne jak w 2005.
**Wplyw:** "Replikacja" (rho=0.283, p=0.008) dziedziczy strukture przestrzenna oryginalnego datasetu z konstrukcji. Nie jest niezaleznym testem.

### HIGH-4: min_reports zmienia sie miedzy testami (20, 10, 5)
**Lokalizacja:** `phase_e_replication_suite.py`, linie 37, 165, 367, 407
**Opis:** Glowna analiza: min_reports=20. Temporal splits: 10. Rolling windows i post-2014: 5. Nizszy prog wlacza komorki z 5 raportami gdzie szum Poissona dominuje (+/-20% na jeden raport).
**Wplyw:** Porownanie jablka-do-pomaranczek. Post-2014 "replikacja" wchodzi czesciowo dzieki niskim progom.

### HIGH-5: Kalibracja P_meteor jest skutecznie martwa
**Lokalizacja:** `phase_c_steps3_7.py`, linie 659-667
**Opis:** Nawet najsilniejszy roj (Leonidy, amp=1.4) daje P_meteor = 0.286 < prog 0.3. Zadne raporty nie sa flagowane jako meteorowe. Residual dataset to 99.5% (63,890 / 64,191) oryginalnych danych.
**Wplyw:** Porownania full-vs-residual sa trywialne (C4 hourly: chi2=0.42, p=1.0).

### HIGH-6: Area-based OR dla baz wojskowych zaklada brak overlap-u
**Lokalizacja:** `phase_c_prompt2.py`, linie 786-791
**Opis:** 171 baz * pi * 25^2 / 8M km^2 = 4.2%. Ale 171 instalacji ma znaczne naklady w kolach 25 km. Skutek: OR jest zanizone.
**Wplyw:** C6 (military proximity) jest obciazony.

### HIGH-7: Tylko 3 epoki (dane do 2014) — C6f ma minimalna moc
**Lokalizacja:** `phase_c_prompt3.py`, linie 53-54
**Opis:** Prompt zaklada 6 epok, ale dane konczą sie w 2014 → 3 epoki, 2 przejscia. Maksymalny composite score = 10. C6f (permutation p=0.369) nie mogl byc istotny.
**Wplyw:** WEAK_AGENCY werdykt jest odpowiedni, ale nie z powodu braku sygnalu — z powodu braku mocy statystycznej.

### HIGH-8: Norwegia uzywa uproszczonego S (brak P, C, ranking)
**Lokalizacja:** `phase_e_norway_replication.py`, linie 191-197, 275-317
**Opis:** West Coast S = mean(rank_G + rank_P + rank_C). Norway S = frac_steep * (mean_gradient / threshold). Brak shore proximity i coastal complexity, brak globalnego rankingu.
**Wplyw:** Null wynik z Norwegii moze odzwierciedlac inna metodologie, nie brak efektu.

### HIGH-9: Phase D nadal uzywa deg*111 aproksymacji (nie haversine)
**Lokalizacja:** `phase_d_robustness.py`, linia 155+
**Opis:** Odleglosci obliczane jako `cd_grid * 111.0` — plaska Ziemia. Blad ~23% na E-W na szerokoosci 40N.
**Wplyw:** Systematyczne przeszacowanie odleglosci E-W. Bias PRZECIW znalezieniu bliskosci.

### HIGH-10: C2 permutacja niszczy autokorelacje Kp
**Lokalizacja:** `phase_c_prompt2.py`, linie 268-284
**Opis:** Shuffling Kp niszczy autokorelacje (27-dniowy cykl sloneczny). Test jest anty-konserwatywny.
**Wplyw:** p=0.0000 — prawdopodobnie wynik jest odporny, ale formalnie test jest niepoprawny.

### HIGH-11: C1 Rayleigh test pominiety (ale zamiennik lepszy)
**Lokalizacja:** `phase_c_prompt2.py`, linie 70-127
**Opis:** Prompt zaklada Rayleigh test na cyklicznych danych fazy Ksiezyca. Implementacja uzywa chi-square vs dystrybucja astronomiczna — co jest bardziej wlasciwe.
**Werdykt:** Odchylenie od promptu, ale poprawa.

### HIGH-12: Zmienna `residual` nadpisana w C6d
**Lokalizacja:** `phase_c_prompt3.py`, linie 432-433
**Opis:** Lokalna zmienna nadpisuje globalna. Praktyczny wplyw jest minimalny (df_res juz przefiltrowane), ale jest to zla praktyka programistyczna.

### HIGH-13: C7 Wilcoxon z n=3 parami
**Lokalizacja:** `phase_c_prompt2.py`, linie 1084-1105
**Opis:** Minimalne p dwustronne dla n=3 to 0.25. Test jest nieinterpretowlany.

### HIGH-14: NRC event expected count uzywa 14-day window zamiast 15
**Lokalizacja:** `phase_c_prompt2.py`, linie 800-804
**Opis:** `day_diff <= 7` to okno 15-dniowe ([-7, +7] wlacznie), ale formula uzywa `19 * 14`. ~7% niedoszacowanie expected.

---

## SEKCJA 3: PROBLEMY SREDNIE (MEDIUM / MODERATE)

### MED-1: cKDTree degree-space distance (do 40% bledu na 48N)
**Lokalizacja:** `phase_ev2_scoring.py`, linie 117, 202, 250
**Opis:** Drzewo zbudowane w (lat, lon) stopniach. Na 48N (Puget Sound), 1 stopien dlugosci = 74.3 km (nie 111 km). Promien 50km query wychodzi 33.4 km na E-W.
**Wplyw:** Systematyczny bias ktory niedoszacowuje feature C (coastal complexity) na wysokich szerokosciach.

### MED-2: Land/ocean weighting 60:1 — arbitralne, brak analizy czulosci
**Lokalizacja:** `phase_e_red_v2.py`, linie 183-188
**Opis:** Land=3.0, ocean=0.05. Parametr nie jest empirycznie skalibrowany. D1 shows OR range 1.68-6.66 depending on weights.

### MED-3: Rozne definicje West Coast (-117 vs -115)
**Lokalizacja:** E-RED v2: lon <= -115.0; Replication suite: lon <= -117.0
**Wplyw:** Komorki miedzy -117 a -115 moga miec O_i w E-RED ale 0 w replication suite.

### MED-4: OLS na log-ratio bez diagnostyki residuow
**Lokalizacja:** `phase_e_red_v2.py`, linie 368-388
**Opis:** OLS logR ~ S bez Shapiro-Wilk, bez testu heteroskedastycznosci, bez wykresow residuow.
**Rekomendacja:** Poisson GLM z offset ln(E_i) bylby bardziej wlasciwy.

### MED-5: 74.5% tied at S=0 w Spearman test
**Lokalizacja:** `phase_e_red_v2.py`, linie 257-298
**Opis:** 76 z 102 komorek ma S=0. Korelacja jest glownie napedzana roznoca miedzy 26 hot a 76 cold — efektywnie zblizona do Mann-Whitney U.

### MED-6: Cross-cell contamination z 50km aggregation radius
**Lokalizacja:** `phase_ev2_scoring.py`, linie 248-289
**Opis:** 50km radius na 0.5° grid powoduje ze sasiednie komorki dzielą steep cells. Tworzy autokorelacje przestrzenna w S.
**Wplyw:** Inflacja istotnosci. Block bootstrap bylby bardziej wlasciwy.

### MED-7: Granger causality bez testu stacjonarnosci
**Lokalizacja:** `phase_c_prompt2.py`, linie 624-659
**Opis:** NUFORC weekly counts maja silny trend rosnacy. Granger test zaklada stacjonarnosc. Bez rozniczkowania moga byc wykryte spurious causality.

### MED-8: Cloud cover to miesieczna klimatologia, nie aktualna pogoda
**Lokalizacja:** `phase_c_steps3_7.py`, linie 555-624
**Opis:** Arizona w czerwcu = 12% (zawsze "clear"), Ohio w styczniu = 72% (zawsze "cloudy"). Nie chwyta noclnych warunkow.

### MED-9: C6f permutation shuffles epoch labels ignorujac autokorelacje temporalna
**Lokalizacja:** `phase_c_prompt3.py`, linie 700-738
**Opis:** Shuffle etykiet epok niszczy porzadek temporalny. Block permutation bylby lepszy.

### MED-10: Google Trends overlapping chunks nie znormalizowane
**Lokalizacja:** `phase_c_fix_gaps.py`, linie 39-57
**Opis:** Chunks 2004-2009 i 2009-2015 nakladaja sie w 2009. GT normalizuje 0-100 per chunk.

### MED-11: UTC offset from longitude ignores DST
**Lokalizacja:** `phase_c_prompt1.py`, linie 127-129
**Opis:** `round(lon/15)` — blad ~1h przy granicach DST dla ~5% raportow.

### MED-12: C4 hourly trivially null (residual = 99.5% pelnych danych)
**Lokalizacja:** `phase_c_prompt2.py`, linie 516-523
**Opis:** Residual hourly profile jest prawie identyczny z full profile bo residual = 99.5%.

### MED-13: C6e geographic centroid niewazone, dominowane przez centra populacji
**Lokalizacja:** `phase_c_prompt3.py`, linie 536-537

### MED-14: C6b PELT breakpoint z niejustified penalty
**Lokalizacja:** `phase_c_prompt3.py`, linie 181-186
**Opis:** pen=5 na 25 punktach. Brak analizy czulosci na parametr penalty.

### MED-15: Temporal split at 2003 zamiast specyfikowanych 2010
**Lokalizacja:** `phase_c_prompt3.py`, linia 46

### MED-16: Placebo baseline OR = 2.27 (nie 1.0)
**Lokalizacja:** `results_v2_robustness.json`, sekcja "placebo"
**Opis:** Losowe punkty szelfowe TEZ pokazuja excess UAP proximity. Sygnal jest czesciowo efektem shelf-proximity, nie wyacznie canyon.

### MED-17: Populacyjny proxy nadal zbyt zgrubny (county centroids)
**Lokalizacja:** `uap_ocean_phase_b_v2.py`
**Opis:** Centroidy county nie chwytuja within-county dystrybucji (np. San Diego County centroid jest inland od wybrzeza).

---

## SEKCJA 4: PROBLEMY MNIEJSZE (LOW / MINOR / NOTE)

| # | Opis | Lokalizacja |
|---|------|-------------|
| LOW-1 | Global ranking (S) gdy analiza jest West Coast only | `phase_ev2_scoring.py`:218-228 |
| LOW-2 | log10 vs ln base mismatch (nie wplywa na Spearman) | repl:126 vs red_v2:271 |
| LOW-3 | Clipping E_i at 0.1 (brak praktycznego efektu) | repl:124 |
| LOW-4 | Post-2014 CSV zawiera rekordy pre-2014 (poprawnie przefiltrowane) | data/nuforc_post2014.csv |
| LOW-5 | sklearn LogisticRegression bez SE na wspolczynnikach | Phase B v1 |
| LOW-6 | Moran's I na siatce 2 stopnie (zbyt zgrubna) | Phase B v1 |
| LOW-7 | Wavelet coherence requested ale nie zaimplementowana | Phase C |
| NOTE-1 | Sunset offset rounding (~5 min) | prompt1:184-199 |
| NOTE-2 | Military base list manually curated (poprawnie) | fix_gaps:140-321 |
| NOTE-3 | Description merging moze tworzyc duplikaty | prompt2:36-39 |
| NOTE-4 | Phase B v1 population proxy z 50 miast | phase_b.py:188-213 |

---

## SEKCJA 5: ANALIZA WYCIEKU DANYCH (DATA LEAKAGE)

### 5.1 Gdzie wyciek jest obecny

| Komponent | Typ wycieku | Severity |
|-----------|------------|----------|
| LOO CV (replication suite) | logR z calego datasetu uzyte w held-out | CRITICAL |
| Spatial forward prediction | logR z calego datasetu uzyte w test set | CRITICAL |
| 60 m/km threshold | Wybrany z tych samych danych co testowany | CRITICAL |
| Temporal splits (6a) | S z calego okresu (OK — geologia stala), ale E_i uniform | E_i bug, nie leakage per se |

### 5.2 Gdzie wyciek NIE jest obecny

| Komponent | Status |
|-----------|--------|
| Scoring function S | Geologia only — no UAP data used |
| E-RED v2 primary result | S frozen before evaluation — no leakage |
| Phase D robustness | Same-data tests (permissible for robustness) |
| Norway replication | Completely independent data and geography |

---

## SEKCJA 6: SPOJNOSC MIEDZY FAZAMI

### 6.1 Niespojnosci zidentyfikowane

| Para faz | Niespojnosc |
|----------|-------------|
| Phase D vs E-RED v2 | Phase D uzywa deg*111, E-RED v2 uzywa haversine |
| E-RED v2 vs Replication | Rozne E_i (populacyjne vs rowne), rozne log base, rozny lon cutoff |
| Phase B v2 vs E v2 | Rozne gradient thresholds (20 vs 60 m/km) |
| Sprint 3 vs E-RED v2 | Sprint 3 definiuje biny, E v2 uzywa jednego binu — double-dipping |

### 6.2 Spojne elementy

| Aspekt | Status |
|--------|--------|
| ETOPO1 zrodlo | Identyczny plik we wszystkich fazach |
| NUFORC data loading | Spojne kolumny, coerce, dedup we wszystkich skryptach |
| Shelf definition | 0 to -500m konsekwentne |
| Grid resolution | 0.5 stopnia konsekwentne (Phase D/E) |
| Bootstrap seed | RNG_SEED=42 konsekwentne |

---

## SEKCJA 7: REKOMENDACJE

### Priorytet 1 (Krytyczne — wplywaja na glowne wnioski)

1. **Naprawic E_i w replication suite** — albo zaimportowac E_i z E-RED v2 cell_details, albo reimplementowac model populacyjny. Bez tego testy 6a, 6d, 6e testuja inna hipoteze niz glowna analiza.

2. **Naprawic LOO CV i spatial forward prediction** — dla kazdego foldu przeliczyc O_i/E_i z podzbioru. S moze byc reuzywane (geologia stala).

3. **Uznac 60 m/km jako data-derived** — dodac korekte Bonferroniego (x4 biny) lub temporal split discovery/validation. Narracja powinna jasno stwierdzac ze prog pochodzi z tych samych danych.

4. **Zaimplementowac C3 permutation test** — shuffle dat raportow zachowujac lokalizacje.

### Priorytet 2 (Wazne — poprawiaja wiarygodnosc)

5. **Consistent West Coast definition** — ujednolicic lon cutoff miedzy E-RED v2 (-115) i replication suite (-117).

6. **Gridded population** — zastapic county centroids danymi gridded (NASA SEDAC GPWv4) by lepiej kontrolowac within-county dystrybucje na wybrzezu.

7. **Sensitivity analysis on land/ocean weights** — systematycznie zmieniac ratio od 1:1 do infinity:0 i raportowac wplyw na rho.

8. **Haversine-corrected cKDTree** — albo przeskalowac wspolrzedne przez cos(lat), albo uzyc BallTree z metryke haversine.

9. **Raportowac geocoding match rate** dla post-2014 danych.

10. **Consistant min_reports** — uzyc tego samego progu (20) lub jawnie raportowac czulosc.

### Priorytet 3 (Drobne — doskonalenie)

11. Dodac diagnostyke residuow dla OLS (QQ-plot, heteroscedasticity test).
12. Zastapic OLS Poisson GLM z offset ln(E_i).
13. Uzyc block bootstrap zamiast iid bootstrap (by chwycic autokorelacje przestrzenna).
14. Dodac stationarity check (ADF test) przed Granger causality.
15. Transparentnie zaprezentowac Phase B v1 -> v2 tranzycje w publikacji.

---

## SEKCJA 8: OCENA KONCOWA

### Co jest solidne

Rdzen analizy — **E-RED v2 primary result** (rho=0.374, p=0.0001, n=102, West Coast) — jest metodologicznie poprawny. Scoring function S jest frozen before evaluation, haversine distances sa poprawne, Spearman jest odpowiedni, bootstrap CIs sa standardowe. Wynik przezywa Ocean depth i magnetic anomaly confound tests.

Phase D robustness (D1-D6) jest dobrze zaprojektowana i uczciwie raportowana. Kluczowe odkrycie — **regionalna asymetria** (West Coast OR=6.21 vs East/Gulf OR=0.36) — jest najbardziej informatywnym wynikiem calego projektu i jest solidna.

### Co jest wątpliwe

1. **Replication suite** jest w znacznej mierze niesprawna (uniform E_i + data leakage w LOO/spatial CV). Dopoki CRIT-1 i CRIT-2/3 nie zostana naprawione, temporal replication i spatial CV nie moga byc cytowane.

2. **Post-2014 "replikacja"** dziedziczy strukture przestrzenna z oryginalnych danych (ten sam geocoding lookup). Nie jest niezaleznym testem.

3. **ESI shore type confound** (SHORE_DOMINANT) podwaza glowna hipoteze — typ wybrzeza wyjaśnia S, a nie odwrotnie.

4. **Prog 60 m/km** jest data-derived. Glowny efekt jest realny ale wielkosc jest zawyzona przez post-selection.

### Werdykt syntetyczny

> Istnieje **prawdziwa, replikowalna, regionalna korelacja przestrzenna** miedzy stromiznami batymetrycznymi a gestoscia raportow UAP na Zachodnim Wybrzezu USA. Korelacja ta jest jednak **ograniczona do waskego pasa przybrzeznego (0-25 km)**, **nie replikuje sie na Wschodnim Wybrzezu**, **nie przezywa confoundu ESI shore type**, i **nie jest niezaleznie zwalidowana** (replication suite ma bledy, post-2014 geocoding jest circular, Norwegia underpowered).
>
> Po naprawieniu bledow krytycznych (E_i, data leakage, double-dipping) glowny wynik E-RED v2 (rho ~ 0.37) prawdopodobnie przetrwa, ale z mniejsza istotnoscia i z jasnym zastrzezeniem o regionalnosci i confoundzie shore type.

---

*Raport wygenerowany automatycznie przez pipeline audytowy. 47 problemow zidentyfikowanych w ~65 skryptach.*
