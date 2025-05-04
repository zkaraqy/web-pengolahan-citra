# Web Pengolahan Citra

Aplikasi web untuk melakukan pengolahan citra digital menggunakan berbagai metode seperti Thresholding dan Edge Detection. Dibuat untuk menyelesaikan tugas mata kuliah Pengolahan Citra.

## Fitur

- **Image Thresholding**: Konversi gambar ke biner dengan threshold yang dapat disesuaikan
- **Edge Detection**: Deteksi tepi pada gambar dengan berbagai operator:
  - Sobel
  - Prewitt
  - Roberts
  - Canny (dengan parameter threshold yang bisa disesuaikan)

## Persyaratan Sistem

- Python 3.10 atau lebih tinggi
- Browser web modern (Chrome, Firefox, Edge, dll)

## Cara Menjalankan Website di Localhost

### 1. Buat Virtual Environment

```bash
python -m venv venv
```

### 2. Aktifkan Virtual Environment

Pada Windows:
```bash
venv\Scripts\activate
```

Pada macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r dev-requirements.txt
```

### 4. Jalankan Aplikasi Flask

```bash
python app.py
```

### 5. Buka Website di Browser

Buka browser dan akses:
```
http://127.0.0.1:5000/
```

## Struktur Projek

```
web-pengolahan-citra/
├── app.py                 # File utama aplikasi Flask
├── static/
│   ├── css/               # File CSS termasuk Bootstrap
│   ├── js/                # File JavaScript termasuk Bootstrap
│   ├── img/               # Gambar statis (logo, dll)
│   ├── uploads/           # Folder penyimpanan gambar yang diunggah
│   └── processed/         # Folder penyimpanan gambar hasil olahan
├── templates/
│   ├── base.html          # Template dasar dengan layout umum
│   ├── home.html          # Halaman beranda
│   ├── thresholding.html  # Halaman metode thresholding
│   └── edge_detection.html # Halaman metode edge detection
├── requirements.txt       # Daftar dependensi utama
└── dev-requirements.txt   # Daftar dependensi termasuk untuk development
```

## Cara Penggunaan

1. Buka halaman beranda aplikasi
2. Pilih metode pengolahan citra yang ingin digunakan (Thresholding atau Edge Detection)
3. Unggah gambar
4. Atur parameter sesuai kebutuhan
5. Klik "Proses Gambar" untuk memproses gambar
6. Lihat hasil pengolahan citra yang ditampilkan
7. Anda dapat mengunduh gambar hasil pengolahan dengan mengklik tombol "Unduh Gambar"

## Developer

- Muhammad Azka Raki (2311016110005)

---

Dibuat dengan Python, Flask, OpenCV, dan Bootstrap.