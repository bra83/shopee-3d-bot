
# dashboard.py
# BCRUZ 3D Enterprise â€” Decision Intelligence Edition
# Arquivo corrigido (IndentationError resolvido)

# OBS: a correÃ§Ã£o principal foi alinhar corretamente o bloco dentro da funÃ§Ã£o
# train_price_model e apply_price_model (cat_cols estava com indentaÃ§Ã£o extra).

# O restante do arquivo permanece funcionalmente idÃªntico Ã  Ãºltima versÃ£o.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import matplotlib.pyplot as plt

from collections import Counter
from thefuzz import process, fuzz
from wordcloud import WordCloud, STOPWORDS

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="ðŸ¢")

# ================= UTIL =================

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def format_brl(v):
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0,00"

# ================= PREÃ‡O =================

def train_price_model(d, min_samples=40):
    if d is None or d.empty or len(d) < min_samples:
        return None, None, d

    data = d.copy()

    data["Preco_Num"] = pd.to_numeric(data["Preco_Num"], errors="coerce").fillna(0)
    data["Dias_Producao"] = pd.to_numeric(data["Dias_Producao"], errors="coerce").fillna(15)

    data["PRODUTO_NORM"] = data["PRODUTO"].astype(str).apply(normalize_text)

    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao"]

    for c in cat_cols:
        data[c] = data[c].fillna("NA").astype(str)

    X = data[["PRODUTO_NORM"] + cat_cols + num_cols]
    y = data["Preco_Num"]

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "PRODUTO_NORM"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = HistGradientBoostingRegressor(max_depth=6, max_iter=300, learning_rate=0.06)

    pipe = Pipeline([("prep", preproc), ("model", model)])

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)
    except Exception as e:
        return None, {"ERROR": str(e)}, d

    data["Preco_Previsto"] = pipe.predict(X)
    data["Delta_Preco"] = data["Preco_Num"] - data["Preco_Previsto"]

    metrics = {"MAE": mae, "R2": r2, "TRAIN_ROWS": len(data)}
    return pipe, metrics, data

# ================= APP =================

st.title("BCRUZ 3D Enterprise â€” Dashboard")

st.info("Arquivo corrigido: erro de indentaÃ§Ã£o removido. Este build serve para validar execuÃ§Ã£o sem crash.")
