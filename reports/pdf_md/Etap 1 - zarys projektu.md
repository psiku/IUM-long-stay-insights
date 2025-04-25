# Etap 1 - zarys projektu

# Autorzy

### Maksymilian Baj

### Bartosz Psik

---

# Definicja problemu biznesowego

> “Nie do końca rozumiemy jakimi kryteriami kierują się klienci, którzy rezerwują dłuższe noclegi. Taka informacja bardzo pomogłaby naszym konsultantom”
> 

**Kontekst :** Klient nie ma jasności, jakie czynniki wpływają na decyzję użytkowników o rezerwacji **dłuższych noclegów**. Brak tej wiedzy utrudnia konsultantom skuteczne dopasowanie ofert do preferencji klientów. 

**Zadanie biznesowe :** Zrozumienie kryteriów, które sprzyjają dłuższym pobytom, pozwoliłoby nie tylko poprawić jakość rekomendacji, ale również zwiększyć udział długoterminowych rezerwacji, które generują wyższe przychody.

**Cel analizy/modelowania:**

Celem jest opracowanie modelu analitycznego, który pomoże zidentyfikować cechy ofert i zachowania użytkowników skorelowane z dłuższymi rezerwacjami. Po przeprowadzeniu analizy chcemy móc przekazać klientowi wnioski takie jak:

- Cena wpływa na długość rezerwacji pobytu
- Użytkownicy wolą rezerwować na dłużej oferty z lokacji XYZ
- Obecność WiFI w ofercie pozytywnie wpływa na długość rezerwacji

# Zadania modelowania

Planujemy wykorzystać następujące zadania modelowania : 

1. **Klasyfikacja binarna,** która dla danych wejściowych będzie przewidywała czy dana oferta zostanie wynajęta przez użytkownika na długo (tj. 5+ dni) czy na krótko (<5 dni)
2. **Regresja Linowa** , która na podstawie cech oferty i użytkownika będzie prognozować długość pobytu (w dniach)
3. **Grupowanie,** które będzie grupowało oferty w grupy określająca na jak długo są one wynajmowane, aczkolwiek to zadanie modelowania sprawdzimy tylko jeśli wystarczy nam czasu

### Dane do modelowania :

- logi z informacjami o tym, jakie oferty są przeglądane i rezerwowane przez użytkowników,
- informacje o dostępnych ofertach, takie jak dane gospodarza, wyposażenie, rozkład pomieszczeń,
- kalendarz dostępności ofert (które dni są zajęte),
- podstawowe dane o użytkownikach: imię, nazwisko, miejsce zamieszkania,
- recenzje użytkowników na temat danych ofert.

# Kryteria sukcesu

### Biznesowe kryterium sukcesu

- Zwiększenie ilości wyświetleń ofert, które są bookowane najdłużej
- Zwiększenie liczby rezerwacji obejmujących co najmniej 5 dni

### Analityczne kryteria sukcesu

**Dla zadania klasyfikacji :** 

- Dokonywać klasyfikacji z dokładnością średnią klasową około 90% (uwzględniamy średnią klasową, żeby oferty wynajmowane na krótko, które stanowią większość danych nie zdominowały wyniku)
- Uzyskanie recallu na poziomie 90% dla klasy „długi pobyt”, aby minimalizować liczbę błędnych klasyfikacji tej klasy

**Dla zadane regresji :** stworzyć model, który będzie przewidywał ilość dni na jakie zostanie wynajęta dana oferta z dokładnością do 1 dnia (Błąd MAE w okolicach 1)

# Założenia

1. Za długi pobyt określamy taki, który trwa 5 dni lub dłużej
2. Długość pobytu liczona jest jako liczba kolejnych zarezerwowanych nocy.
3. Model będzie trenowany na danych obejmujących rezerwacje od grudnia 2024 do grudnia 2025 roku