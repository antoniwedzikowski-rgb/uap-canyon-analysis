# PROMPT DLA CLAUDE CODE: UAP-Ocean Geomorphology Correlation Analysis (Phase B)

Wklej poniższy tekst jako prompt w Claude Code:

---

Przeprowadź kompletną analizę statystyczną korelacji między lokalizacjami raportów UAP (Unidentified Anomalous Phenomena) a strukturami geologicznymi dna oceanu. To jest test falsyfikacyjny hipotezy kryptoziemskiej (CTH) — czy hotspoty UAP korelują z głębokimi strukturami oceanicznymi bardziej niż wynika z gęstości zaludnienia i obecności baz wojskowych?

## DANE DO POBRANIA

1. **NUFORC sightings** — geokodowane raporty UAP z lat/lon:
   - Główne źródło: https://github.com/planetsig/ufo-reports → `csv-data/ufo-scrubbed-geocoded-time-standardized.csv` (~80,000 raportów)
   - Alternatywnie Kaggle: https://www.kaggle.com/datasets/NUFORC/ufo-sightings (`scrubbed.csv`)
   - Kolumny zawierają: datetime, city, state, country, shape, duration, comments, latitude, longitude
   - Wyczyść: usuń (0,0), NaN, lat/lon poza zakresem, duplikaty

2. **GEBCO bathymetry** — głębokość oceanu:
   - https://www.gebco.net/data_and_products/gridded_bathymetry_data/
   - Pobierz GEBCO_2024 Grid (netCDF). Jeśli pełny plik za duży (>7GB), użyj podzestawu dla regionu USA: lat 15-55°N, lon 130-60°W
   - Alternatywnie: ETOPO1 z NOAA (mniejszy): https://www.ngdc.noaa.gov/mgg/global/
   - Potrzebujesz: dla każdej współrzędnej UAP → głębokość dna oceanu w najbliższym punkcie morskim (< 0m)

3. **Gęstość zaludnienia** — kontrola population bias:
   - NASA SEDAC GPWv4: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11
   - Lub WorldPop: https://www.worldpop.org/
   - Lub prostsze: US Census population per county (census.gov)
   - Potrzebujesz: gęstość zaludnienia w punkcie każdego raportu UAP

4. **Bazy wojskowe USA** — kontrola military bias:
   - Lokalizacje baz: https://catalog.data.gov/dataset/military-installations-ranges-and-training-areas
   - Lub hardcode listę ~400 baz z lat/lon z publicznych źródeł
   - Potrzebujesz: odległość każdego raportu UAP od najbliższej bazy wojskowej

## ANALIZA DO PRZEPROWADZENIA

### Krok 1: Przygotowanie danych
- Załaduj NUFORC CSV, wyczyść współrzędne
- Dla każdego raportu UAP oblicz:
  - `depth_nearest_ocean`: głębokość GEBCO w najbliższym punkcie morskim (wartość < 0)
  - `dist_to_coast_km`: odległość do najbliższej linii brzegowej
  - `dist_to_trench_km`: odległość do najbliższego rowu oceanicznego (depth < -6000m)
  - `dist_to_canyon_km`: odległość do najbliższego kanionu podmorskiego (gradient głębokości)
  - `pop_density`: gęstość zaludnienia w punkcie raportu
  - `dist_to_military_km`: odległość do najbliższej bazy wojskowej
- Ogranicz analizę do raportów kontynentalnych USA (najlepsza jakość danych)

### Krok 2: Model NULL — czego oczekujemy bez korelacji z oceanem
- Wygeneruj 10,000 syntetycznych punktów z rozkładem proporcjonalnym do gęstości zaludnienia USA
- Oblicz te same metryki (depth, dist_coast, dist_trench, dist_canyon) dla syntetycznych punktów
- To jest Twoja kontrola — "tak wyglądałyby raporty gdyby ludzie zgłaszali proporcjonalnie do tego gdzie mieszkają"

### Krok 3: Kontrola baz wojskowych
- Osobna kontrola: wygeneruj punkty z rozkładem proporcjonalnym do odległości od baz wojskowych
- Pytanie: czy rezydualny sygnał UAP-ocean istnieje PO usunięciu efektu "bazy wojskowe przyciągają raporty"?

### Krok 4: Testy statystyczne
Wszystkie testy dwustronne chyba że zaznaczono inaczej:

a) **Kolmogorov-Smirnov**: rozkład `dist_to_coast` dla UAP vs kontrola populacyjna
b) **Mann-Whitney U**: `depth_nearest_ocean` dla UAP vs kontrola — czy raporty UAP są bliżej głębszych struktur?
c) **Regresja logistyczna**: zmienna zależna = UAP report (1) vs control point (0), zmienne niezależne = depth_nearest, dist_coast, pop_density, dist_military. Pytanie: czy depth/dist_coast są istotne PO kontroli pop_density i dist_military?
d) **Permutation test** (n=10,000): losowo permutuj etykiety UAP/kontrola i oblicz statystykę testową. Porównaj z obserwowaną.
e) **Spatial autocorrelation (Moran's I)**: czy raporty UAP są bardziej przestrzennie skupione wokół struktur oceanicznych niż wynika z populacji?

### Krok 5: Analiza subgrup
- Podziel raporty UAP po kolumnie "shape": czy sfery/orby/cylinder (typowe USO shapes) korelują z oceanem silniej niż trójkąty/światła?
- Podziel raporty po dekadach: czy korelacja się zmienia w czasie?
- Wyodrębnij raporty przybrzeżne (<100km od oceanu): jakie struktury dna dominują?

### Krok 6: Wizualizacje
Zapisz wszystkie jako PNG w wysokiej rozdzielczości:

a) Mapa: raporty UAP na tle batymetrii GEBCO (heatmap + kontury głębokości)
b) Histogram: rozkład `dist_to_coast` dla UAP vs kontrola populacyjna
c) Scatter: `depth_nearest_ocean` vs `dist_to_coast` z kolorami wg gęstości raportów
d) Box plot: odległość do struktury oceanicznej per typ kształtu UAP
e) Mapa rezydualna: gęstość raportów UAP MINUS oczekiwana gęstość z populacji — co zostaje? Czy residua skupiają się nad kanionami/basenami?

### Krok 7: Raport
Zapisz jako `uap_ocean_phase_b_results.md` z:
- Executive summary (3 zdania)
- Metodologia
- Wyniki każdego testu z p-value i effect size
- Wizualizacje inline
- Dyskusja: co wyniki znaczą dla CTH, jakie alternatywne wyjaśnienia, co trzeba jeszcze sprawdzić
- Surowe dane w CSV: `uap_ocean_full_data.csv`

## PRE-REGISTERED HYPOTHESES (deklarowane PRZED analizą)

Następujące hipotezy są sformułowane a priori. Wyniki które je potwierdzają LUB obalają są równie wartościowe:

- **H1**: Gęstość raportów UAP (znormalizowana per capita) jest istotnie wyższa w promieniu 50km od kanionów podmorskich niż oczekiwana z modelu populacyjnego. (Próg 50km wybrany a priori jako ~1h drogi łodzią podwodną przy 50 węzłach)
- **H2**: Raporty UAP o kształtach typowych dla USO (sphere, oval, cylinder, cigar, disk) korelują z bliskością struktur oceanicznych silniej niż raporty o kształtach nietypowych (triangle, light, chevron, formation)
- **H3**: Rezydua gęstości UAP (po usunięciu efektu populacji + baz wojskowych) wykazują spatial clustering wokół struktur podmorskich (Moran's I > 0, p < 0.01)
- **H_null**: Rozkład raportów UAP jest w pełni wyjaśniony przez gęstość zaludnienia + bliskość baz wojskowych + bliskość wybrzeża (bez niezależnego efektu geomorfologii dna)

## KOREKTY STATYSTYCZNE

- **Multiple comparisons**: zastosuj Benjamini-Hochberg FDR correction (q < 0.05) na wszystkie testy. Raportuj ZARÓWNO raw p-values jak i adjusted q-values.
- **Spatial autocorrelation w danych wejściowych**: raporty UAP nie są niezależne — jeden incydent medialny generuje klaster raportów. PRZED testowaniem: zastosuj DBSCAN clustering (eps=0.5°, min_samples=3) i redukuj klastry do centroidów. Testuj na centroidach ORAZ na surowych danych — raportuj oba wyniki.
- **Reporting bias proxy**: jako dodatkową zmienną kontrolną oblicz "sky visibility index" — odwrotność light pollution w punkcie raportu. Użyj VIIRS nighttime lights (https://eogdata.mines.edu/products/vnl/) lub prostszy proxy: odległość od miasta >100k mieszkańców. Dodaj jako covariate do regresji logistycznej.
- **Sensitivity analysis**: dla KAŻDEGO progu odległościowego (np. "blisko kanionu") przeprowadź analizę dla wartości 25, 50, 100, 200, 500 km i raportuj jak zmienia się wynik. Przedstaw jako wykres p-value vs próg.
- **Effect size**: dla każdego testu raportuj Cohen's d (lub odpowiednik nieparametryczny: rank-biserial correlation). P-value bez effect size jest bezwartościowe.
- **Spatial bootstrap**: jako alternatywę dla permutation testu, przeprowadź block bootstrap (bloki 2°x2°) zachowujący lokalną strukturę przestrzenną. Porównaj wyniki z naiwnym bootstrapem.

## WAŻNE ZASTRZEŻENIA

- NIE wybieraj ręcznie hotspotów — użyj WSZYSTKICH raportów z datasetu
- NIE wybieraj ręcznie struktur oceanicznych — użyj pełnej siatki GEBCO
- Każdy arbitralny próg (np. "500km") MUSI być uzasadniony a priori lub przetestowany dla wielu wartości (sensitivity analysis)
- Raportuj WSZYSTKIE wyniki, w tym null results — nie ukrywaj p-values > 0.05
- Effect size (Cohen's d lub equivalent) jest ważniejszy niż p-value

## INSTALACJA PAKIETÓW

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn netCDF4 xarray cartopy geopandas shapely pyproj
```

Jeśli GEBCO netCDF jest za duży do RAM, użyj xarray z chunked loading:
```python
import xarray as xr
ds = xr.open_dataset('gebco_2024.nc', chunks={'lat': 1000, 'lon': 1000})
```

## OCZEKIWANY OUTPUT

1. `uap_ocean_phase_b_results.md` — pełny raport
2. `uap_ocean_full_data.csv` — surowe dane z obliczonymi metrykami
3. `figures/` — folder z wizualizacjami PNG
4. `uap_ocean_analysis_phase_b.py` — kompletny, powtarzalny skrypt

Zacznij od pobrania danych, potem analiza. Jeśli jakieś źródło danych jest niedostępne, użyj alternatywnego i odnotuj to w raporcie.
