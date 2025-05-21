# Wykrywanie zapalenia płuc ze zdjęć rentgenowskich przy użyciu modelu CNN (Python + TensorFlow + Keras)

Projekt oparty na danych medycznych przedstawiających zdjęcia rentgenowskie klatki piersiowej, którego celem jest zbudowanie modelu sztucznej inteligencji zdolnego do klasyfikacji obrazów jako przedstawiające przypadki **normalne** lub **z zapaleniem płuc**.

## 📁 Dane

### Źródło danych
Dane pochodzą z [Kaggle Hub](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) i są automatycznie pobierane z wykorzystaniem biblioteki `kagglehub`.

**Struktura zbioru danych:**
- `train/`
- `val/`
- `test/`
Każdy z podzbiorów zawiera podfoldery `NORMAL/` i `PNEUMONIA/`.

---

## 🧪 Przetwarzanie danych

### 1. Przenoszenie danych
Z pierwotnego zbioru treningowego przenoszonych jest 120 obrazów z każdej klasy (`NORMAL`, `PNEUMONIA`) do zbioru walidacyjnego, aby zbalansować dane.

### 2. Normalizacja
Wszystkie obrazy są przeskalowywane do zakresu `0–1` poprzez podzielenie wartości pikseli przez 255.

### 3. Augmentacja (rozszerzenie danych)

Użyto warstw `tf.keras.layers`:
- `RandomZoom(0.11)` – losowe przybliżenie/oddalenie do 11%
- `RandomContrast(0.15)` – losowe zmiany kontrastu do 15%

Augmentacja stosowana jest **wyłącznie** do zbioru treningowego.

#### 🔧 Możliwości modyfikacji
Parametry augmentacji można łatwo zmieniać w poniższym fragmencie kodu:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.11),
    tf.keras.layers.RandomContrast(0.15)
])
```

---

## 🧠 Architektura modelu

Model stworzono z użyciem **Keras Sequential API**:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### ⚙️ Kompilacja modelu
- Optymalizator: `adam`
- Funkcja straty: `binary_crossentropy`
- Metryka: `accuracy`

---

## 🔁 Proces treningowy

Model uczony był w **dwóch fazach**:

1. **Faza 1**: 5 epok z oryginalnymi danymi treningowymi  
2. **Faza 2**: 5 kolejnych epok z danymi poddanymi augmentacji

```python
# Faza 1
model.fit(train_df, epochs=5, validation_data=validation_df)

# Faza 2
model.fit(train_df2, epochs=5, validation_data=validation_df)
```

---

## 📊 Walidacja i testowanie

- **Zbiór walidacyjny**: `/PSI/val`
- **Zbiór testowy**: `/PSI/test`

Na końcu przeprowadzono predykcję na zbiorze testowym oraz oceniono skuteczność modelu:

# Ewaluacja i predykcja na zbiorze testowym

## 1. Predykcja modelu

### Model przetwarza cały zbiór testowy i zwraca prawdoposobieństwo przynależności do klasy PNEUMONIA
```python
y_prob = model.predict(test_df).squeeze()  # prawdopodobieństwa
threshold = 0.7
y_pred = (y_prob > threshold).astype(int)  # etykiety 0/1
```
- Próg `0.7` można dostosować (trade‑off czułość ↔ precyzja).

## 2. Etykiety rzeczywiste
Pobierane są prawdziwe etykiety z przetworzonego test_df
```python
y_true = np.concatenate([y for _, y in test_df])
```

## 3. Wynikowy DataFrame
| kolumna       | opis                                         |
|---------------|----------------------------------------------|
| `file`        | nazwa pliku w `/PSI/test`                    |
| `true_label`  | prawdziwa klasa (`NORMAL`, `PNEUMONIA`)      |
| `pred_label`  | klasa przewidziana przez model               |
| `probability` | prawdopodobieństwo zapalenia płuc (0–1)      |

Każdy rekord zawiera:
nazwę pliku (np. person123_bacteria_1.jpeg),
etykietę rzeczywistą (true_label),
etykietę przewidzianą (pred_label),
prawdopodobieństwo wykrycia zapalenia płuc (probability).

## 4. Podsumowanie skuteczności
```python
correct = (results_df['pred_label'] == results_df['true_label']).sum()
total = len(results_df)
accuracy = correct / total
```
- Wyświetla podsumowanie: `Trafienia: {correct}/{total} (accuracy = {accuracy:.2%})`.

```python
accuracy = (liczba trafnych predykcji / liczba wszystkich próbek)
```
---

## 📈 Przykładowy wynik

Przy zastosowanych parametrach uzyskano skuteczność:
```
✔️ Trafienia: 580/620 (accuracy ≈ 93.5%)
```

---

## 📦 Technologie użyte w projekcie

- Python 3
- TensorFlow + Keras
- NumPy, Pandas, Matplotlib
- KaggleHub (do pobierania danych)

---

## 📌 Uruchamianie projektu

1. Upewnij się, że masz zainstalowane:
    ```bash
    pip install tensorflow kagglehub matplotlib pandas
    ```

2. Uruchom notebook `PSI_MODEL.ipynb` w środowisku Jupyter lub Google Colab.

---

## 📬 Kontakt

Autor: Michał Pieniek 

