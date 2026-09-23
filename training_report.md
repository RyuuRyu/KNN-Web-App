# Laporan Training Model KNN

**Dibuat:** 2026-09-23 15:03:34 +0700

## Ringkasan

| Metrik | Hasil |
| --- | ---: |
| Dataset | `data/Dataset IPK Mahasiswa.xlsx` |
| Jumlah baris setelah pembersihan | 282 |
| Jumlah fitur | 11 |
| Akurasi rata-rata cross-validation | 97.16% |
| Akurasi data testing | 94.74% |

## Akurasi Cross-Validation

Skor setiap fold: 96.49%, 98.25%, 96.43%, 96.43%, 98.21%

## Classification Report

| Kelas | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| TIDAK LULUS | 90.91% | 95.24% | 93.02% | 21 |
| LULUS | 97.14% | 94.44% | 95.77% | 36 |
| **Accuracy** |  |  | **94.74%** | 57 |
| **Macro average** | 94.03% | 94.84% | 94.40% | 57 |
| **Weighted average** | 94.85% | 94.74% | 94.76% | 57 |

## Distribusi Kelas Data Training

| Kelas | Sebelum SMOTE | Sesudah SMOTE |
| --- | ---: | ---: |
| TIDAK LULUS (0) | 81 | 144 |
| LULUS (1) | 144 | 144 |

## Fitur Model

- `SEMESTER 1`
- `SEMESTER 2`
- `SEMESTER 3`
- `SEMESTER 4`
- `SEMESTER 5`
- `SEMESTER 6`
- `SEMESTER 7`
- `STDEV_IPK`
- `TREND_IPK`
- `SLOPE_IPK`
- `WEIGHTED_AVG_IPK`
