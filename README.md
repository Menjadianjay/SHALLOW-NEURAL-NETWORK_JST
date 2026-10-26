# Shallow Neural Network — Klasifikasi Digit MNIST

<div align="center">

![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-orange?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-96.53%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-MNIST_784-blueviolet)

**Implementasi Shallow Neural Network dari Scratch untuk Klasifikasi Digit Tulisan Tangan**

</div>

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Shallow Neural Network](#arsitektur-shallow-neural-network)
- [Cara Kerja: Tahapan Lengkap](#cara-kerja-tahapan-lengkap)
  - [1. Persiapan Data](#1-persiapan-data)
  - [2. Inisialisasi Parameter](#2-inisialisasi-parameter)
  - [3. Forward Propagation](#3-forward-propagation)
  - [4. Fungsi Cost](#4-fungsi-cost)
  - [5. Backward Propagation](#5-backward-propagation)
  - [6. Update Parameter](#6-update-parameter)
- [Matematika di Balik SNN](#matematika-di-balik-snn)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
- [Pengujian](#pengujian)
- [Lisensi](#lisensi)
- [Referensi](#referensi)

---

## Gambaran Umum

Proyek ini merupakan implementasi **Shallow Neural Network (SNN)** yang dibangun **dari awal** (*from scratch*) menggunakan Python dan NumPy, **tanpa** framework deep learning seperti TensorFlow atau PyTorch.

### Tujuan Proyek
- Membangun neural network satu hidden layer secara manual
- Mengklasifikasikan digit tulisan tangan (0–9) dari dataset MNIST
- Memahami secara mendalam mekanisme forward & backward propagation

### Dataset: MNIST

| Properti | Detail |
|----------|--------|
| **Sumber** | `sklearn.datasets.fetch_openml('mnist_784')` |
| **Total Data** | 70.000 gambar |
| **Dimensi** | 28×28 piksel = **784 fitur** per gambar |
| **Kelas** | 10 digit (0 hingga 9) |
| **Train / Test Split** | 56.000 / 14.000 (80:20) |
| **Normalisasi** | Nilai piksel dibagi 255 → rentang [0, 1] |

---

## Arsitektur Shallow Neural Network

Jaringan yang dibangun terdiri dari **3 lapisan**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INPUT LAYER        HIDDEN LAYER          OUTPUT LAYER         │
│   (784 neuron)       (128 neuron)          (10 neuron)          │
│                       [ReLU]               [Softmax]            │
│                                                                 │
│   x₁  ──────┐                          ┌──► P(kelas=0)         │
│   x₂  ──────┤                          ├──► P(kelas=1)         │
│   x₃  ──────┤   W¹(128×784)  W²(10×128)├──► P(kelas=2)        │
│   ...  ─────┼───────────────────────►──┤   ...                 │
│   x₇₈₃──────┤                          ├──► P(kelas=8)         │
│   x₇₈₄──────┘                          └──► P(kelas=9)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Spesifikasi Lapisan

| Lapisan | Neuron | Aktivasi | Fungsi |
|---------|--------|----------|--------|
| **Input Layer** | 784 | — | Menerima piksel gambar (28×28) yang telah dinormalisasi |
| **Hidden Layer** | 128 | **ReLU** | Mengekstrak fitur dan pola tersembunyi dari input |
| **Output Layer** | 10 | **Softmax** | Menghasilkan distribusi probabilitas untuk 10 kelas digit |

### Dimensi Matriks Parameter

```
W¹  : (128 × 784)   → Bobot koneksi Input → Hidden
b¹  : (128 × 1)     → Bias Hidden Layer
W²  : (10  × 128)   → Bobot koneksi Hidden → Output
b²  : (10  × 1)     → Bias Output Layer
```

---

## Cara Kerja: Tahapan Lengkap

### 1. Persiapan Data

```python
# Load dataset MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0   # Normalisasi piksel ke [0, 1]
Y = mnist.target.astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# One-Hot Encoding label
# Contoh: digit "3" → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))  # Shape: (56000, 10)
y_test_oh  = encoder.transform(y_test.reshape(-1, 1))       # Shape: (14000, 10)

# Transpose untuk komputasi matrix
X_train_T = np.array(X_train).T   # (784, 56000)
y_train_T = y_train_oh.T          # (10,  56000)
```

**Mengapa One-Hot Encoding?**
Neural network tidak bisa memproses label kategorik langsung. One-hot encoding mengubah label integer menjadi vektor biner sehingga output Softmax bisa dibandingkan langsung.

---

### 2. Inisialisasi Parameter

```python
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01   # Shape: (128, 784)
    b1 = np.zeros((n_h, 1))                  # Shape: (128, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01   # Shape: (10, 128)
    b2 = np.zeros((n_y, 1))                  # Shape: (10, 1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
```

> **Mengapa bobot dikali 0.01 (bukan nol)?**
> - **Symmetry Breaking**: Jika semua bobot sama (nol), semua neuron akan belajar hal yang identik sehingga jaringan tidak berkembang.
> - **Nilai Kecil**: Mencegah saturasi pada fungsi aktivasi di awal training, sehingga gradien bisa mengalir dengan baik.

---

### 3. Forward Propagation

Proses mengalirkan data dari input hingga output untuk menghasilkan prediksi.

```python
def relu(nilai_z):
    return np.maximum(0, nilai_z)

def softmax(nilai_Z):
    # Numerically stable softmax (dikurangi max untuk mencegah overflow)
    expZ = np.exp(nilai_Z - np.max(nilai_Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1   # Linear: (128, m)
    A1 = relu(Z1)              # ReLU:   (128, m)
    Z2 = np.dot(W2, A1) + b2  # Linear: (10, m)
    A2 = softmax(Z2)           # Softmax:(10, m)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache
```

**Alur Forward Propagation:**

```
Input X (784, m)
    │
    ▼  Z¹ = W¹·X + b¹
   [Z1] (128, m)
    │
    ▼  A¹ = ReLU(Z¹) = max(0, Z¹)
   [A1] (128, m)   ← Representasi Fitur Tersembunyi
    │
    ▼  Z² = W²·A¹ + b²
   [Z2] (10, m)
    │
    ▼  A² = Softmax(Z²)
   [A2] (10, m)   ← Probabilitas Tiap Kelas
```

**Fungsi Aktivasi:**

| Fungsi | Formula | Digunakan di | Keunggulan |
|--------|---------|--------------|------------|
| **ReLU** | `max(0, z)` | Hidden Layer | Mengatasi *vanishing gradient*, komputasi cepat |
| **Softmax** | `eᶻⁱ / Σeᶻʲ` | Output Layer | Menghasilkan distribusi probabilitas (jumlah = 1) |

---

### 4. Fungsi Cost

Mengukur seberapa jauh prediksi model dari label sebenarnya menggunakan **Cross-Entropy Loss**:

```python
def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.log(A2 + 1e-8)    # +1e-8 mencegah log(0) = -inf
    cost = -np.sum(Y * logprobs) / m
    return cost
```

**Formula Cross-Entropy:**

```
L = -(1/m) · Σᵢ Σⱼ Y[j,i] · log(A2[j,i] + ε)
```

Di mana:
- `m` = jumlah sampel
- `Y` = label one-hot (ground truth)
- `A2` = output softmax (prediksi)
- `ε = 1e-8` = epsilon untuk stabilitas numerik

> Loss dimulai sekitar **2.303** (mendekati log(10) — sangat acak) dan turun menjadi ~**0.098** pada iterasi ke-900, nilai terakhir yang dicetak notebook (pencetakan setiap 100 iterasi; training berjalan 1000 iterasi).

---

### 5. Backward Propagation

Menghitung gradien dari setiap parameter menggunakan **chain rule** untuk memperbarui bobot.

```python
def relu_derivative(Z):
    return (Z > 0).astype(float)   # 1 jika Z > 0, else 0

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]

    # Gradien Output Layer
    dZ2 = A2 - Y                                           # (10, m)
    dW2 = (1/m) * np.dot(dZ2, A1.T)                      # (10, 128)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)     # (10, 1)

    # Gradien Hidden Layer
    dA1 = np.dot(W2.T, dZ2)                               # (128, m)
    dZ1 = dA1 * relu_derivative(Z1)                       # (128, m)
    dW1 = (1/m) * np.dot(dZ1, X.T)                       # (128, 784)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)     # (128, 1)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
```

**Alur Backward Propagation (Chain Rule):**

```
Loss (L)
    │
    ▼  dZ2 = A² - Y  ← (Softmax + CrossEntropy gradient yang disederhanakan)
    │
    ├──► dW2 = (1/m) · dZ2 · A1ᵀ
    ├──► db2 = (1/m) · Σ dZ2
    │
    ▼  dA1 = W2ᵀ · dZ2
    │
    ▼  dZ1 = dA1 ⊙ ReLU'(Z1)  ← ReLU'(z) = 1 jika z > 0, else 0
    │
    ├──► dW1 = (1/m) · dZ1 · Xᵀ
    └──► db1 = (1/m) · Σ dZ1
```

> **Turunan Softmax + CrossEntropy** menyederhanakan menjadi `A2 - Y`, yang sangat elegan secara matematis.

---

### 6. Update Parameter

Memperbarui bobot menggunakan **Gradient Descent**:

```python
def update_parameters(parameters, grads, learning_rate=0.5):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters
```

**Formula Update:**

```
W := W - α · ∂L/∂W
b := b - α · ∂L/∂b
```

Di mana `α = 0.5` adalah *learning rate* (kecepatan pembaruan bobot).

---

### Fungsi Model (Keseluruhan Training Loop)

```python
def model(X, Y, n_h, iterasi=1000, learning_rate=0.5, print_cost=False):
    np.random.seed(3)   # catatan: initialize_parameters() langsung menimpanya dengan seed 2
    n_x, n_y = X.shape[0], Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(iterasi):
        A2, cache    = forward_propagation(X, parameters)                    # Step 1: Prediksi
        cost         = compute_cost(A2, Y)                                    # Step 2: Hitung Loss
        grads        = backward_propagation(parameters, cache, X, Y)          # Step 3: Gradien
        parameters   = update_parameters(parameters, grads, learning_rate)    # Step 4: Update

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)

    return parameters, costs
```

---

## Matematika di Balik SNN

### Notasi

| Simbol | Deskripsi |
|--------|-----------|
| `m` | Jumlah sampel data |
| `nₓ = 784` | Jumlah fitur input |
| `nₕ = 128` | Jumlah neuron hidden layer |
| `nᵧ = 10` | Jumlah kelas output |
| `X ∈ ℝ^(784×m)` | Matriks input |
| `W¹ ∈ ℝ^(128×784)` | Bobot layer pertama |
| `b¹ ∈ ℝ^(128×1)` | Bias layer pertama |
| `A¹ ∈ ℝ^(128×m)` | Aktivasi hidden layer |
| `W² ∈ ℝ^(10×128)` | Bobot layer kedua |
| `b² ∈ ℝ^(10×1)` | Bias layer kedua |
| `A² ∈ ℝ^(10×m)` | Output (probabilitas tiap kelas) |

### Forward Pass

```
Z¹ = W¹ · X + b¹
A¹ = ReLU(Z¹) = max(0, Z¹)

Z² = W² · A¹ + b²
A² = Softmax(Z²) = exp(Z²ᵢ) / Σⱼ exp(Z²ⱼ)

L = -(1/m) Σᵢ Σⱼ Yⱼᵢ · log(A²ⱼᵢ + ε)
```

### Backward Pass

```
∂L/∂Z² = A² - Y                          [dZ2]
∂L/∂W² = (1/m) · dZ2 · A¹ᵀ              [dW2]
∂L/∂b² = (1/m) · Σ dZ2                   [db2]

∂L/∂A¹ = W²ᵀ · dZ2                       [dA1]
∂L/∂Z¹ = dA1 ⊙ 𝟙[Z¹ > 0]               [dZ1]
∂L/∂W¹ = (1/m) · dZ1 · Xᵀ               [dW1]
∂L/∂b¹ = (1/m) · Σ dZ1                   [db1]
```

---

## Hasil Eksperimen

### Penurunan Loss (Cost) Selama Training

| Iterasi | Cost |
|---------|------|
| 0 | 2.3033 |
| 100 | 0.3302 |
| 200 | 0.2483 |
| 300 | 0.2037 |
| 400 | 0.1736 |
| 500 | 0.1511 |
| 600 | 0.1335 |
| 700 | 0.1193 |
| 800 | 0.1078 |
| 900 | 0.0981 |

### Akurasi Akhir

| Metrik | Nilai |
|--------|-------|
| **Train Accuracy** | **97.52%** |
| **Test Accuracy** | **96.53%** |

### Classification Report (Test Set)

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.98 | 0.98 | 0.98 | 1343 |
| 1 | 0.97 | 0.98 | 0.98 | 1600 |
| 2 | 0.97 | 0.96 | 0.96 | 1380 |
| 3 | 0.96 | 0.96 | 0.96 | 1433 |
| 4 | 0.96 | 0.96 | 0.96 | 1295 |
| 5 | 0.96 | 0.96 | 0.96 | 1273 |
| 6 | 0.97 | 0.98 | 0.98 | 1396 |
| 7 | 0.97 | 0.97 | 0.97 | 1503 |
| 8 | 0.96 | 0.95 | 0.95 | 1357 |
| 9 | 0.95 | 0.95 | 0.95 | 1420 |
| **Accuracy** | | | **0.97** | **14000** |

> Model berhasil mencapai **96.53% akurasi** pada data test — performa luar biasa untuk jaringan satu hidden layer yang dibangun dari nol tanpa framework!

### Penjelasan Metrik Evaluasi

- **Accuracy** — Proporsi prediksi yang benar dari seluruh data. Mengukur performa model secara keseluruhan.
- **Precision** — Dari semua prediksi suatu kelas, berapa yang benar-benar kelas itu. `TP / (TP + FP)`
- **Recall** — Dari semua data suatu kelas, berapa yang berhasil ditemukan model. `TP / (TP + FN)`
- **F1-Score** — Rata-rata harmonis precision dan recall. Berguna saat kelas tidak seimbang.

---

## Struktur Proyek

```
SHALLOW-NEURAL-NETWORK_JST/
├── README.md
├── UTS_225314118.ipynb         ← Notebook utama (seluruh kode)
└── DATA_MNIST/MNIST/raw/       ← File MNIST mentah (idx + .gz, ≈ 66 MB); TIDAK dibaca notebook
```

Notebook memuat data lewat `fetch_openml`, bukan dari folder `DATA_MNIST/`. Sebagai pengecekan, kode notebook yang sama dijalankan ulang dengan data dari `DATA_MNIST/` (gambar train lalu test, digabung, split `random_state=42`) dan menghasilkan angka identik: Train 97.52%, Test 96.53%, serta cost dan classification report yang sama. Isi notebook (`UTS_225314118.ipynb`) berurutan sebagai berikut:

```
Import Library          numpy, sklearn, matplotlib
Load & Eksplorasi Data  MNIST 70.000 gambar (784 fitur)
Split & Preprocessing   Normalisasi + One-Hot Encoding
initialize_parameters() Inisialisasi W1, b1, W2, b2
forward_propagation()   ReLU (hidden) + Softmax (output)
compute_cost()          Cross-Entropy Loss
backward_propagation()  Gradien via Chain Rule
update_parameters()     Gradient Descent
model()                 Training Loop (1000 iterasi)
predict()               Prediksi dengan argmax
Visualisasi             Contoh prediksi + Confusion Matrix
```

---

## Instalasi & Cara Menjalankan

### Prasyarat

- Python 3. Notebook tersimpan dari Python 3.13.2; versi minimum tidak dinyatakan di repositori. `OneHotEncoder(sparse_output=...)` membutuhkan scikit-learn 1.2 atau lebih baru.
- Koneksi internet saat pertama kali menjalankan: `fetch_openml('mnist_784')` mengunduh dataset dari OpenML (lalu disimpan di cache scikit-learn).
- `pandas` wajib ada: `fetch_openml` mengembalikan DataFrame untuk data ini dan scikit-learn akan error jika pandas tidak terpasang (diperiksa dari kode scikit-learn 1.9.1; belum dijalankan terhadap OpenML).

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

Repositori tidak memiliki `requirements.txt`, sehingga versi paket tidak dikunci.

### Menjalankan Notebook

```bash
# Clone repository
git clone https://github.com/Menjadianjay/SHALLOW-NEURAL-NETWORK_JST.git
cd SHALLOW-NEURAL-NETWORK_JST

# Jalankan Jupyter Notebook
jupyter notebook UTS_225314118.ipynb
```

### Hyperparameter Model

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `n_h` | 128 | Jumlah neuron di hidden layer |
| `iterasi` | 1000 | Jumlah epoch training |
| `learning_rate` | 0.5 | Kecepatan pembaruan bobot |
| `random_seed` | 2 | Seed yang efektif untuk bobot awal. `model()` memanggil `np.random.seed(3)`, tetapi `initialize_parameters()` langsung menimpanya dengan `np.random.seed(2)` |

Pembagian data memakai `random_state=42`. Seluruh angka pada bagian Hasil Eksperimen diambil dari output yang tersimpan di notebook; training penuh (1000 iterasi pada 56.000 gambar) berjalan di CPU; pada satu kali uji di mesin 2-core tanpa GPU memakan sekitar 9 menit (562 detik).

---

## Pengujian

Tidak ada tes otomatis maupun konfigurasi CI di repositori ini. Cara memeriksa hasil adalah menjalankan seluruh sel notebook dan membandingkan Train/Test Accuracy dengan nilai pada bagian Hasil Eksperimen.

## Lisensi

Repositori ini tidak memiliki file lisensi, sehingga tidak ada lisensi yang diberikan secara default. Pemilik repositori perlu memilih dan menambahkannya. Dataset MNIST memiliki ketentuan penggunaannya sendiri (lihat referensi di bawah).

---

## Referensi

- **Dataset**: LeCun, Y., Cortes, C., & Burges, C. (2010). [MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/)
- **Deep Learning Textbook**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- **Backpropagation**: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- **ReLU Activation**: Nair, V., & Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines. *ICML*.
- **Softmax & Cross-Entropy**: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
