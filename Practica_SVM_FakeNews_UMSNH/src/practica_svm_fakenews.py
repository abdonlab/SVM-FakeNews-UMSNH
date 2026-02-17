"""
Práctica V4 (OK): Detección de Fake News UMSNH con SVM (TF-IDF + LinearSVC)

Autor: Gilberto Abdón Chávez Olivos
Materia: Temas Selectos en Sistemas Inteligentes
Docente: Dr. Juan Carlos González Ibarra
Institución: ISU Universidad
Fecha: 16 de febrero de 2026

FIX V4 (el que te está fallando)
--------------------------------
Tu error indica que el script intenta escribir en:
    ...\resultados\reporte_metricas.txt

Pero en tu proyecto la carpeta correcta es:
    resultados\   (con "s")

Esta versión fuerza SIEMPRE el uso de **resultados/** (con "s")
y crea el directorio antes de escribir cualquier archivo.

Ejecución (PowerShell, desde la RAÍZ del proyecto)
-------------------------------------------------
python .\src\practica_svm_fakenews.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def detectar_raiz_proyecto() -> Path:
    """Detecta la raíz del proyecto (Windows/OneDrive friendly)."""
    cwd = Path.cwd()
    if (cwd / "datos").is_dir() and (cwd / "src").is_dir():
        return cwd.resolve()
    if cwd.name.lower() == "src" and (cwd.parent / "datos").is_dir():
        return cwd.parent.resolve()
    return Path(__file__).resolve().parents[1]


BASE = detectar_raiz_proyecto()
DATA_PATH = (BASE / "datos" / "dataset_fakenews_umsnh.csv").resolve()

# ✅ Forzar SIEMPRE "resultados" (con s)
OUT_DIR = (BASE / "resultados").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en:\n  {DATA_PATH}\n"
        "Asegúrate de tener: datos/dataset_fakenews_umsnh.csv"
    )

print("=== Diagnóstico ===")
print("CWD    :", Path.cwd())
print("BASE   :", BASE)
print("DATA   :", DATA_PATH)
print("OUTDIR :", OUT_DIR)
print("===================\n")


def limpiar_texto(texto: str) -> str:
    """Limpieza básica de texto."""
    t = str(texto).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^a-záéíóúñü\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


print("=== Cargando dataset ===")
df = pd.read_csv(DATA_PATH, encoding="utf-8")

required_cols = {"texto", "etiqueta"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"El CSV debe contener columnas {required_cols}. "
        f"Columnas actuales: {set(df.columns)}"
    )

df["texto_limpio"] = df["texto"].apply(limpiar_texto)

print("=== Vectorización TF-IDF ===")
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["texto_limpio"])
y = df["etiqueta"].astype(int)

print("=== División train/test ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("=== Entrenando SVM (LinearSVC) ===")
model = LinearSVC(C=1.0, max_iter=5000, random_state=42)
model.fit(X_train, y_train)

print("=== Evaluación ===")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

reporte = classification_report(
    y_test, y_pred,
    target_names=["Fake News (0)", "Verificada (1)"],
    digits=4,
    zero_division=0
)

print(f"Accuracy: {acc:.4f}\n")
print(reporte)
print("Matriz de confusión:\n", cm)

# ✅ Re-asegurar antes de escribir (por si OneDrive sincroniza lento)
OUT_DIR.mkdir(parents=True, exist_ok=True)

metric_path = OUT_DIR / "reporte_metricas.txt"
metric_path.parent.mkdir(parents=True, exist_ok=True)
with open(metric_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(reporte)
    f.write("\nMatriz de confusión:\n")
    f.write(np.array2string(cm))

img_path = OUT_DIR / "matriz_confusion_svm.png"
img_path.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Matriz de Confusión - SVM Fake News UMSNH")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.xticks([0, 1], ["Fake (0)", "Verificada (1)"])
plt.yticks([0, 1], ["Fake (0)", "Verificada (1)"])
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.tight_layout()
plt.savefig(img_path, dpi=300)
plt.close()

terms_path = OUT_DIR / "terminos_discriminativos.txt"
terms_path.parent.mkdir(parents=True, exist_ok=True)
coef = model.coef_.ravel()
terms = np.array(vectorizer.get_feature_names_out())

top_fake = np.argsort(coef)[:10]
top_real = np.argsort(coef)[-10:][::-1]

with open(terms_path, "w", encoding="utf-8") as f:
    f.write("Top términos asociados a FAKE NEWS (clase 0):\n")
    for idx in top_fake:
        f.write(f"- {terms[idx]}: {coef[idx]:.4f}\n")
    f.write("\nTop términos asociados a INFORMACIÓN VERIFICADA (clase 1):\n")
    for idx in top_real:
        f.write(f"- {terms[idx]}: {coef[idx]:.4f}\n")

print("\n✅ LISTO. Archivos generados:")
print(f" - {metric_path}")
print(f" - {img_path}")
print(f" - {terms_path}")
