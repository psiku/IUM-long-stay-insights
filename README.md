long-stay-insights
==============================

# Autorzy
Bartosz Psik <br>
Maksymilian Baj


# Kontekst projektu

W ramach projektu wcielamy się w rolę **analityka danych pracującego dla portalu Nocarz** – serwisu, w którym klienci mogą wyszukiwać i rezerwować noclegi oferowane przez zewnętrznych dostawców.

Nasze zadanie nie jest trywialne – otrzymujemy je w formie **ogólnego opisu**, a do nas należy jego **doprecyzowanie i wdrożenie**. Oznacza to konieczność:
- zrozumienia problemu biznesowego,
- eksploracji i analizy danych,
- a czasem także negocjacji z interesariuszami.

Oprócz wytrenowania odpowiednich modeli predykcyjnych, musimy również przygotować je do **potencjalnego wdrożenia produkcyjnego**, zakładając, że w przyszłości będą pojawiać się kolejne ich wersje, które będzie można rozwijać i testować.

---

# Dostępne dane

Jak każda nowoczesna firma internetowa, Nocarz zbiera dane dotyczące swojej działalności. Jako analitycy, możemy wnioskować o dostęp do następujących informacji:

- **Szczegółowe dane o dostępnych lokalach** – ich parametry, lokalizacja, wyposażenie itp.
- **Recenzje i oceny lokali** – teksty oraz metryki przyznawane przez użytkowników.
- **Kalendarz dostępności i cen** – informacje o terminach możliwych rezerwacji i kosztach.
- **Baza klientów i sesji** – dane o użytkownikach oraz ich interakcjach z platformą.

**Problem : “Nie do końca rozumiemy jakimi kryteriami kierują się klienci, którzy rezerwują dłuższe noclegi. Taka informacja bardzo pomogłaby naszym konsultantom.”**



# Organizacja Projektu
```text
long-stay-insights/
├── LICENSE                  # Licencja projektu
├── Makefile                # Polecenia pomocnicze (np. make data, make lint, make jupyter)
├── README.md               # Ten plik – opis projektu
├── pyproject.toml          # Główny plik konfiguracyjny dla Poetry
├── poetry.lock             # Zamrożone wersje zależności (wygenerowane przez Poetry)
│
├── .gitignore              # Pliki/ścieżki ignorowane przez Git
├── .env                    # Zmienne środowiskowe (jeśli używane)
│
├── data/
│   ├── raw/                # Surowe, niezmienione dane źródłowe
│   ├── interim/            # Dane po wstępnym przetworzeniu
│   ├── processed/          # Dane gotowe do modelowania/analizy
│   └── external/           # Dane pochodzące z zewnętrznych źródeł
│
├── notebooks/              # Notebooki Jupyter z analizami (np. 1.0-ab-opis.ipynb)
│
├── models/                 # Zapisane modele, predykcje, metryki
│
├── reports/
│   └── figures/            # Wygenerowane wykresy, do raportów i prezentacji
│
├── src/                    # Kod źródłowy projektu
│   ├── __init__.py         # Plik inicjalizujący moduł Pythonowy
│   ├── data/               # Skrypty do pobierania/przetwarzania danych (make_dataset.py)
│   ├── features/           # Tworzenie cech do modelowania (build_features.py)
│   ├── models/             # Trenowanie i predykcje modeli ML (train_model.py, predict_model.py)
│   └── visualization/      # Wykresy i wizualizacje wyników (visualize.py)
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
