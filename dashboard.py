# -*- coding: utf-8 -*-
# dashboard.py
# BCRUZ 3D Enterprise — Decision Intelligence Edition
# (Mantém seu dashboard original e adiciona módulos de decisão + ML.
#  Inclui: contagem de itens válidos por etapa, e modelo de preço aplicando no filtro.)
#
# Correções incluídas (2026-02-07):
# - Problema de caracteres "DiagnÃ³stico": garantir arquivo em UTF-8 + header coding utf-8
# - Erro ML "Sparse data was passed for X, but dense data is required": adiciona transformer ToDense antes do HistGradientBoostingRegressor
# - Sanitização forte (NaN/strings em numérico) antes do treino e antes de prever
# - Mostra o erro real do treino no sidebar

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Paleta de cores (contraste) ---
COLOR_MAP_FONTE = {
    "Shopee": "#ff7f0e",  # laranja
    "Elo7": "#2ca02c",    # verde
}
try:
    px.defaults.color_discrete_sequence = px.colors.qualitative.Safe
except Exception:
    pass
import numpy as np
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import time

# SciPy (opcional) para o Otimizador
try:
    from scipy.optimize import linprog
except Exception:
    linprog = None
from thefuzz import process, fuzz
import re
from collections import Counter

# --- ML extra ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, TransformerMixin


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    """
    Converte matriz esparsa (scipy sparse) em densa (numpy array).
    Necessário porque HistGradientBoostingRegressor exige entrada densa.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        # reduz memória e acelera um pouco
        try:
            return np.asarray(X, dtype=np.float32)
        except Exception:
            return X


# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# -----------------------------
# Memória de custos (persistência entre abas)
# -----------------------------
if "cost_config" not in st.session_state:
    st.session_state["cost_config"] = {
        "custo_hora": 8.0,
        "custo_grama": 0.12,
        "taxa_mkt": 0.14,
        "taxa_falha": 0.06,
        "embalagem": 4.0,
        "gramas_base": 60,
    }


# -----------------------------
# Robustez: carregamento com retry (Google Sheets CSV)
# -----------------------------
def load_csv_retry(url: str, attempts: int = 3, sleep_s: float = 1.0, **read_csv_kwargs):
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return pd.read_csv(url, **read_csv_kwargs)
        except Exception as e:
            last_err = e
            if i < attempts:
                time.sleep(sleep_s)
    # última falha
    raise last_err


# --- 2. LINKS ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- 3. LIMPEZA DE PREÇO AGRESSIVA (ANTI-OUTLIER) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0

    if isinstance(valor, (int, float)):
        val = float(valor)
    else:
        texto = str(valor).upper().strip()
        texto = re.sub(r"[^\d,.]", "", texto)

        try:
            if "," in texto:
                texto = texto.replace(".", "")
                texto = texto.replace(",", ".")
            elif texto.count(".") == 1:
                partes = texto.split(".")
                if len(partes[1]) == 3:
                    texto = texto.replace(".", "")
            val = float(texto)
        except Exception:
            return 0.0

    if val > 1500.0:
        return 0.0

    return val


# -----------------------------
# UTILITÁRIOS NOVOS (ADD-ON)
# -----------------------------
def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_features_from_title(title: str) -> dict:
    t = normalize_text(title)

    is_kit = int(bool(re.search(r"\bkit\b|\bconjunto\b|\bcombo\b", t)))
    is_personalizado = int(bool(re.search(r"\bpersonaliz", t)))
    is_pronta_entrega = int(bool(re.search(r"\bpronta\b|\bpronto\b|\bimediat", t)))
    is_decor = int(bool(re.search(r"\bdecor\b|\bdecora", t)))
    is_organizador = int(bool(re.search(r"\borganiz", t)))
    is_suporte = int(bool(re.search(r"\bsuporte\b|\bstand\b|\bbase\b", t)))
    is_vaso = int(bool(re.search(r"\bvaso\b|\bplant", t)))
    is_action = int(bool(re.search(r"\bfigure\b|\baction\b|\bminiatura\b|\bstatue\b|\bestatua\b", t)))
    is_gamer = int(bool(re.search(r"\bgamer\b|\bplaystation\b|\bxbox\b|\bnintendo\b|\bpc\b", t)))

    nums = re.findall(r"\b(\d{1,3})\s?(cm|mm)?\b", t)
    size_num = 0
    for n, unit in nums:
        try:
            v = int(n)
            if unit == "mm":
                v = int(round(v / 10))
            size_num = max(size_num, v)
        except Exception:
            pass

    premium = int(bool(re.search(r"\bpremium\b|\bdeluxe\b|\bmetal\b|\bvelvet\b|\babs\b|\bpetg\b|\bresina\b", t)))

    return dict(
        is_kit=is_kit,
        is_personalizado=is_personalizado,
        is_pronta_entrega=is_pronta_entrega,
        is_decor=is_decor,
        is_organizador=is_organizador,
        is_suporte=is_suporte,
        is_vaso=is_vaso,
        is_action=is_action,
        is_gamer=is_gamer,
        size_num=size_num,
        premium=premium,
        title_len=len(t),
        word_count=len(t.split()),
    )


def format_brl(v: float) -> str:
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


# -----------------------------
# CARREGAMENTO + CONTAGEM POR ETAPA
# -----------------------------
@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]

    stats = []
    per_source = []

    for f in fontes:
        try:
            temp_df = load_csv_retry(f["url"], on_bad_lines="skip", dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]

            raw_n = int(len(temp_df))
            if temp_df.empty:
                per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": 0, "erro": ""})
                continue

            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), "Geral")
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), "#")
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)

            temp_df = temp_df.rename(columns={col_prod: "PRODUTO"})
            temp_df["FONTE"] = f["nome"]
            temp_df["CATEGORIA"] = temp_df[col_cat] if col_cat in temp_df.columns else "Geral"
            temp_df["LINK"] = temp_df[col_link] if col_link in temp_df.columns else "#"

            if col_preco:
                temp_df["Preco_Num"] = temp_df[col_preco].apply(limpar_preco)
            else:
                temp_df["Preco_Num"] = 0.0

            after_price_n = int(len(temp_df))

            if col_prazo:
                temp_df["Prazo_Txt"] = temp_df[col_prazo].fillna("Normal")

                def get_days(t):
                    t = str(t).upper()
                    if "IMEDIATO" in t or "PRONTA" in t:
                        return 1
                    m = re.search(r"(\d+)", t)
                    return int(m.group(1)) if m else 15

                temp_df["Dias_Producao"] = temp_df["Prazo_Txt"].apply(get_days)
            else:
                temp_df["Dias_Producao"] = 15

            temp_df["Logistica"] = temp_df["Dias_Producao"].apply(lambda x: "⚡ FLASH" if x <= 2 else "📦 NORMAL")

            cols = ["PRODUTO", "Preco_Num", "FONTE", "CATEGORIA", "LINK", "Logistica", "Dias_Producao"]
            for c in cols:
                if c not in temp_df.columns:
                    temp_df[c] = ""

            temp_df = temp_df[temp_df["Preco_Num"] > 0.1].copy()
            after_valid_n = int(len(temp_df))

            stats.append({"raw": raw_n, "after_price": after_price_n, "after_valid": after_valid_n})
            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": after_valid_n, "erro": ""})

            dfs.append(temp_df[cols])

        except Exception as e:
            per_source.append({"fonte": f["nome"], "raw": 0, "validos": 0, "erro": str(e)})
            continue

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not final_df.empty:
        corte_superior = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= corte_superior].copy()

    final_df.attrs["rows_raw_total"] = int(sum(x.get("raw", 0) for x in stats))
    final_df.attrs["rows_after_price_clean"] = int(sum(x.get("after_price", 0) for x in stats))
    final_df.attrs["rows_after_filter_valid"] = int(sum(x.get("after_valid", 0) for x in stats))
    final_df.attrs["per_source"] = per_source

    return final_df


df = carregar_dados()


# -----------------------------------------
# ENRIQUECIMENTO DE DADOS
# -----------------------------------------
@st.cache_data(ttl=300)
def enrich_df(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return base_df

    d = base_df.copy()

    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)
    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    d["Preco_Num"] = pd.to_numeric(d["Preco_Num"], errors="coerce").fillna(0).astype(float)

    try:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        d["anomaly_iso"] = iso.fit_predict(d[["Preco_Num"]])
        d["is_anomaly_iso"] = (d["anomaly_iso"] == -1).astype(int)
    except Exception:
        d["is_anomaly_iso"] = 0

    try:
        n_neighbors = min(35, max(5, len(d) // 20))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof_pred = lof.fit_predict(d[["Preco_Num"]])
        d["is_anomaly_lof"] = (lof_pred == -1).astype(int)
    except Exception:
        d["is_anomaly_lof"] = 0

    d["is_anomaly"] = ((d["is_anomaly_iso"] + d["is_anomaly_lof"]) > 0).astype(int)
    return d


# -----------------------------------------
# VETORES DE TEXTO (SBERT opcional, fallback TF-IDF+SVD)
# -----------------------------------------
@st.cache_data(ttl=300)
def compute_text_vectors(texts: pd.Series, method: str = "auto", max_features: int = 4000):
    texts = texts.fillna("").astype(str).tolist()

    if method in ("auto", "sbert"):
        try:
            from sentence_transformers import SentenceTransformer  # opcional

            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            X = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.array(X), "SBERT (MiniLM multilingual)"
        except Exception:
            if method == "sbert":
                pass

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(texts)
    n_comp = int(min(128, max(8, X_tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, f"TF-IDF(1-2gram) + SVD({n_comp})"


# -----------------------------------------
# CANONICALIZAÇÃO / DEDUP
# -----------------------------------------
@st.cache_data(ttl=300)
def canonicalize_products(d: pd.DataFrame, max_groups: int = 250):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"], method="auto")

    n = len(out)
    k = int(np.clip(np.sqrt(n), 10, max_groups))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    except Exception:
        labels = np.zeros(n, dtype=int)

    out["GROUP_ID"] = labels

    canon = []
    for gid, grp in out.groupby("GROUP_ID"):
        counts = grp["PRODUTO"].astype(str).value_counts()
        best = counts.index[0] if len(counts) else grp["PRODUTO"].astype(str).iloc[0]
        candidates = counts.index.tolist()[:10]
        if candidates:
            best = sorted(candidates, key=lambda s: (len(s), s.lower()))[0]
        canon.append((gid, best))

    canon_map = dict(canon)
    out["PRODUTO_CANON"] = out["GROUP_ID"].map(canon_map).fillna(out["PRODUTO"].astype(str))
    out.attrs["vectorizer"] = vec_name
    out.attrs["k_groups"] = int(out["GROUP_ID"].nunique())
    return out


# -----------------------------------------
# CLUSTERING DE MERCADO
# -----------------------------------------
@st.cache_data(ttl=300)
def market_clusters(d: pd.DataFrame, n_clusters: int = 18):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"], method="auto")

    n = len(out)
    k = int(np.clip(n_clusters, 6, min(40, max(6, int(np.sqrt(n) * 1.2)))))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        out["CLUSTER_MKT"] = km.fit_predict(X)
    except Exception:
        out["CLUSTER_MKT"] = 0

    try:
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_t = tfidf.fit_transform(out["PRODUTO_NORM"].fillna("").astype(str))
        vocab = np.array(tfidf.get_feature_names_out())

        cluster_names = {}
        for cid in sorted(out["CLUSTER_MKT"].unique()):
            idx = np.where(out["CLUSTER_MKT"].values == cid)[0]
            if len(idx) == 0:
                cluster_names[cid] = f"Cluster {cid}"
                continue
            mean_vec = X_t[idx].mean(axis=0)
            mean_vec = np.asarray(mean_vec).ravel()
            top_idx = mean_vec.argsort()[-4:][::-1]
            top_terms = [vocab[i] for i in top_idx if mean_vec[i] > 0]
            cluster_names[cid] = " / ".join(top_terms) if top_terms else f"Cluster {cid}"

        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].map(cluster_names).fillna(out["CLUSTER_MKT"].astype(str))
    except Exception:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    out.attrs["mkt_vectorizer"] = vec_name
    out.attrs["mkt_k"] = int(out["CLUSTER_MKT"].nunique())
    return out


# -----------------------------------------
# SANITIZAÇÃO FORTE PARA O MODELO DE PREÇO
# -----------------------------------------
def sanitize_for_price_model(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    if "PRODUTO" not in d.columns:
        d["PRODUTO"] = ""
    if "PRODUTO_NORM" not in d.columns:
        d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)

    d["PRODUTO_NORM"] = d["PRODUTO_NORM"].fillna("").astype(str)

    if "Preco_Num" in d.columns:
        d["Preco_Num"] = pd.to_numeric(d["Preco_Num"], errors="coerce").fillna(0).astype(float)

    if "Dias_Producao" not in d.columns:
        d["Dias_Producao"] = 15
    d["Dias_Producao"] = pd.to_numeric(d["Dias_Producao"], errors="coerce").fillna(15).astype(float)

    for c in ["FONTE", "Logistica", "CATEGORIA"]:
        if c not in d.columns:
            d[c] = "NA"
        d[c] = d[c].fillna("NA").astype(str)

    num_cols = ["size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]
    for c in num_cols:
        if c not in d.columns:
            d[c] = 0
    d[num_cols] = d[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    return d


# -----------------------------------------
# MODELO DE PREÇO
# -----------------------------------------
@st.cache_data(ttl=300)
def train_price_model(d: pd.DataFrame, min_samples: int = 40):
    if d is None or d.empty or len(d) < min_samples:
        return None, {"ERROR": f"Dados insuficientes: {0 if d is None else len(d)} < {min_samples}"}, d

    data = sanitize_for_price_model(d)
    data = data[data["Preco_Num"] > 0].copy()
    if len(data) < min_samples:
        return None, {"ERROR": f"Poucos itens com preço > 0: {len(data)} < {min_samples}"}, d

    y = data["Preco_Num"].astype(float)

    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    X = data[[text_col] + cat_cols + num_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2, max_df=0.95), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_depth=6,
        max_iter=350,
        random_state=42,
        l2_regularization=0.2,
    )

    # 🔥 FIX DO ERRO: converte sparse -> dense antes do model
    pipe = Pipeline(steps=[("prep", preproc), ("dense", ToDenseTransformer()), ("model", model)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
    except Exception as e:
        return None, {"ERROR": str(e), "TRAIN_ROWS": int(len(data)), "MIN_SAMPLES": int(min_samples)}, d

    try:
        data["Preco_Previsto"] = pipe.predict(X)
        data["Delta_Preco"] = data["Preco_Num"] - data["Preco_Previsto"]
        data["Faixa_Min"] = np.maximum(0.0, data["Preco_Previsto"] - mae)
        data["Faixa_Max"] = data["Preco_Previsto"] + mae
    except Exception:
        data["Preco_Previsto"] = np.nan
        data["Delta_Preco"] = np.nan
        data["Faixa_Min"] = np.nan
        data["Faixa_Max"] = np.nan

    metrics = {"MAE": mae, "R2": r2, "MIN_SAMPLES": int(min_samples), "TRAIN_ROWS": int(len(data))}
    return pipe, metrics, data


def apply_price_model(model_pipe, d: pd.DataFrame):
    if model_pipe is None or d is None or d.empty:
        return d

    data = sanitize_for_price_model(d)

    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    X = data[[text_col] + cat_cols + num_cols]
    try:
        data["Preco_Previsto"] = model_pipe.predict(X)
    except Exception:
        return d

    return data


# -----------------------------------------
# SHAP (OPCIONAL)
# -----------------------------------------
def try_shap_explain(model_pipe, df_sample: pd.DataFrame):
    if model_pipe is None or df_sample is None or df_sample.empty:
        return None
    try:
        import shap  # noqa

        sample = df_sample.sample(min(250, len(df_sample)), random_state=42)

        num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]
        for c in num_cols:
            if c not in sample.columns:
                sample[c] = 0

        sample = sanitize_for_price_model(sample)
        num_only = sample[num_cols].astype(float)

        try:
            inner_model = model_pipe.named_steps["model"]
            explainer = shap.Explainer(inner_model, num_only)
            sv = explainer(num_only)
            imp = np.abs(sv.values).mean(axis=0)
            return pd.DataFrame({"feature": num_cols, "importance": imp}).sort_values("importance", ascending=False)
        except Exception:
            return None
    except Exception:
        return None


# -----------------------------------------
# GAP FINDER
# -----------------------------------------
@st.cache_data(ttl=300)
def gap_finder(d: pd.DataFrame):
    if d is None or d.empty:
        return pd.DataFrame()

    g = d.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
        itens=("PRODUTO", "count"),
        ticket=("Preco_Num", "mean"),
        mediana=("Preco_Num", "median"),
        flash_share=("Logistica", lambda s: float((s == "⚡ FLASH").mean())),
        fonte_div=("FONTE", lambda s: int(pd.Series(s).nunique())),
    ).reset_index()

    g["score_base"] = (
        (g["ticket"] / (g["ticket"].median() + 1e-9)) * 0.55
        + (1.0 - (g["itens"] / (g["itens"].max() + 1e-9))) * 0.30
        + (g["flash_share"]) * 0.15
    )

    examples = []
    for _, row in g.sort_values("score_base", ascending=False).head(30).iterrows():
        cid = int(row["CLUSTER_MKT"])
        ex = d[d["CLUSTER_MKT"] == cid].sort_values("Preco_Num", ascending=False).head(3)
        examples.append({
            "CLUSTER_MKT": cid,
            "EX1": ex["PRODUTO"].iloc[0] if len(ex) > 0 else "",
            "EX2": ex["PRODUTO"].iloc[1] if len(ex) > 1 else "",
            "EX3": ex["PRODUTO"].iloc[2] if len(ex) > 2 else "",
        })
    exdf = pd.DataFrame(examples)
    out = g.merge(exdf, on="CLUSTER_MKT", how="left")
    return out.sort_values("score_base", ascending=False)


# -----------------------------------------
# SIMULADOR (lucro/hora)
# -----------------------------------------
def estimate_print_hours(row, base_hours=2.0):
    try:
        days = float(row.get("Dias_Producao", 15))
    except Exception:
        days = 15.0
    try:
        size = float(row.get("size_num", 0))
    except Exception:
        size = 0.0
    logist = str(row.get("Logistica", "📦 NORMAL"))

    h = base_hours
    if size > 0:
        h += min(6.0, size / 25.0)
    h += min(8.0, max(0.0, (days - 2.0)) / 6.0)
    if "FLASH" in logist:
        h *= 0.75
    return float(np.clip(h, 0.4, 18.0))


def compute_profit(d: pd.DataFrame, custo_hora=8.0, custo_grama=0.12, gramas_base=60, taxa_falha=0.06, taxa_marketplace=0.14, embalagem=4.0):
    if d is None or d.empty:
        return d

    out = d.copy()
    out["Preco_Num"] = pd.to_numeric(out.get("Preco_Num", 0), errors="coerce").fillna(0).astype(float)

    size = pd.to_numeric(out.get("size_num", 0), errors="coerce").fillna(0).astype(float)
    kit = pd.to_numeric(out.get("is_kit", 0), errors="coerce").fillna(0).astype(int)
    prem = pd.to_numeric(out.get("premium", 0), errors="coerce").fillna(0).astype(int)

    out["Gramagem_Estimada"] = (float(gramas_base) + (size * 2.2) + (kit * 40) + (prem * 25)).clip(lower=20, upper=600)
    out["Horas_Estimadas"] = out.apply(lambda r: estimate_print_hours(r, base_hours=2.0), axis=1)

    out["Custo_Material"] = out["Gramagem_Estimada"] * float(custo_grama)
    out["Custo_Maquina"] = out["Horas_Estimadas"] * float(custo_hora)
    out["Custo_Embalagem"] = float(embalagem)
    out["Fee_Marketplace"] = out["Preco_Num"] * float(taxa_marketplace)

    out["Custo_Total_Estimado"] = (out["Custo_Material"] + out["Custo_Maquina"] + out["Custo_Embalagem"] + out["Fee_Marketplace"]) * (1.0 + float(taxa_falha))
    out["Lucro_Estimado"] = out["Preco_Num"] - out["Custo_Total_Estimado"]
    out["Lucro_por_Hora"] = out["Lucro_Estimado"] / (out["Horas_Estimadas"] + 1e-9)
    return out


# -----------------------------------------
# OTIMIZADOR DE FÁBRICA (linprog)
# -----------------------------------------
def run_factory_optimizer(df_sim: pd.DataFrame, hours_avail: float = 40.0):
    """Sugere mix de produção por CLUSTER_NOME maximizando lucro total com limite de horas.
    Requer colunas: CLUSTER_NOME, Lucro_Estimado, Horas_Estimadas.
    """
    if df_sim is None or df_sim.empty:
        return pd.DataFrame()

    if linprog is None:
        raise RuntimeError("SciPy não está instalado (linprog indisponível). Adicione 'scipy' ao requirements.txt.")

    g = df_sim.groupby("CLUSTER_NOME", dropna=False).agg(
        itens=("PRODUTO", "count"),
        lucro_medio=("Lucro_Estimado", "mean"),
        horas_media=("Horas_Estimadas", "mean"),
        preco_medio=("Preco_Num", "mean"),
    ).reset_index()

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["lucro_medio", "horas_media"])
    g = g[g["horas_media"] > 0].copy()
    if g.empty:
        return pd.DataFrame()

    # Variáveis: q_i >= 0 (quantidade por cluster)
    c = -g["lucro_medio"].astype(float).values  # maximizar lucro => minimizar -lucro
    A_ub = np.array([g["horas_media"].astype(float).values])
    b_ub = np.array([float(hours_avail)])

    bounds = [(0, None) for _ in range(len(g))]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Otimização falhou: {res.message}")

    g["qtd_sugerida"] = np.maximum(0.0, res.x)
    g["horas_usadas"] = g["qtd_sugerida"] * g["horas_media"]
    g["lucro_total"] = g["qtd_sugerida"] * g["lucro_medio"]

    g = g.sort_values("lucro_total", ascending=False)

    # arredondamento amigável (sem perder o valor real)
    g["qtd_sugerida_round"] = g["qtd_sugerida"].round(2)
    g["horas_usadas_round"] = g["horas_usadas"].round(2)
    g["lucro_total_round"] = g["lucro_total"].round(2)

    return g


# -----------------------------------------
# MATRIZ BCG (portfólio por cluster)
# -----------------------------------------
def build_bcg_matrix(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Retorna tabela por CLUSTER_NOME com volume e margem de lucro % para matriz BCG."""
    if df_sim is None or df_sim.empty:
        return pd.DataFrame()

    d = df_sim.copy()
    # margem % baseada no lucro estimado
    d["margem_pct"] = np.where(d["Preco_Num"] > 0, (d["Lucro_Estimado"] / d["Preco_Num"]) * 100.0, np.nan)

    g = d.groupby("CLUSTER_NOME", dropna=False).agg(
        volume=("PRODUTO", "count"),
        margem_pct=("margem_pct", "mean"),
        lucro_h=("Lucro_por_Hora", "mean"),
        ticket=("Preco_Num", "mean"),
    ).reset_index()

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["margem_pct", "volume"])
    if g.empty:
        return g

    vol_cut = g["volume"].median()
    mar_cut = g["margem_pct"].median()

    def classify(row):
        high_vol = row["volume"] >= vol_cut
        high_mar = row["margem_pct"] >= mar_cut
        if high_vol and high_mar:
            return "Estrela"
        if high_vol and not high_mar:
            return "Vaca Leiteira"
        if (not high_vol) and high_mar:
            return "Aposta"
        return "Abacaxi"

    g["tipo_bcg"] = g.apply(classify, axis=1)
    g.attrs["vol_cut"] = float(vol_cut)
    g.attrs["mar_cut"] = float(mar_cut)
    return g


# -----------------------------------------
# MODO CEO (resumo)
# -----------------------------------------
def build_ceo_summary(d: pd.DataFrame, gap: pd.DataFrame):
    if d is None or d.empty:
        return ["Sem dados para gerar decisões."]

    msgs = []
    top_price = d.sort_values("Preco_Num", ascending=False).head(1)
    if len(top_price):
        msgs.append(f"Maior ticket no filtro: **{format_brl(top_price['Preco_Num'].iloc[0])}** — {top_price['PRODUTO'].iloc[0]} ({top_price['FONTE'].iloc[0]}).")

    if "Delta_Preco" in d.columns and d["Delta_Preco"].notna().any():
        under = d.sort_values("Delta_Preco", ascending=True).head(3)
        if len(under):
            msgs.append("Top 3 prováveis **subprecificados** (abaixo do esperado):")
            for _, r in under.iterrows():
                msgs.append(f"- {r['PRODUTO']} | real {format_brl(r['Preco_Num'])} vs esperado {format_brl(r.get('Preco_Previsto', 0))}")

        over = d.sort_values("Delta_Preco", ascending=False).head(3)
        if len(over):
            msgs.append("Top 3 prováveis **caros demais** (acima do esperado):")
            for _, r in over.iterrows():
                msgs.append(f"- {r['PRODUTO']} | real {format_brl(r['Preco_Num'])} vs esperado {format_brl(r.get('Preco_Previsto', 0))}")

    if gap is not None and not gap.empty:
        top = gap.head(3)
        msgs.append("Top 3 **oportunidades por cluster** (alto ticket + baixa competição relativa + flash):")
        for _, r in top.iterrows():
            msgs.append(f"- **{r['CLUSTER_NOME']}** | score {r['score_base']:.2f} | ticket {format_brl(r['ticket'])} | itens {int(r['itens'])}")

    if "is_anomaly" in d.columns and pd.to_numeric(d["is_anomaly"], errors="coerce").fillna(0).sum() > 0:
        msgs.append(f"⚠️ **{int(pd.to_numeric(d['is_anomaly'], errors='coerce').fillna(0).sum())} anomalias** detectadas no filtro. Veja a aba Alertas.")

    return msgs


# ============================================================
# PIPELINE GLOBAL
# ============================================================
df_enriched = enrich_df(df)
df_enriched = canonicalize_products(df_enriched)
df_enriched = market_clusters(df_enriched)

price_model, price_metrics, df_enriched = train_price_model(df_enriched, min_samples=40)

# ============================================================
# UI
# ============================================================
st.sidebar.title("🎛️ Centro de Comando")

# --- Custos (persistentes) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Custos (persistentes)")
cc = st.session_state["cost_config"]
cc["custo_hora"] = st.sidebar.number_input("Custo/hora máquina (R$)", min_value=0.0, value=float(cc.get("custo_hora", 8.0)), step=0.5, key="cc_custo_hora")
cc["custo_grama"] = st.sidebar.number_input("Custo/grama filamento (R$)", min_value=0.0, value=float(cc.get("custo_grama", 0.12)), step=0.01, format="%.2f", key="cc_custo_grama")
cc["taxa_mkt"] = st.sidebar.number_input("Taxa marketplace", min_value=0.0, max_value=0.8, value=float(cc.get("taxa_mkt", 0.14)), step=0.01, format="%.2f", key="cc_taxa_mkt")
cc["taxa_falha"] = st.sidebar.number_input("Taxa falha/refugo", min_value=0.0, max_value=0.8, value=float(cc.get("taxa_falha", 0.06)), step=0.01, format="%.2f", key="cc_taxa_falha")
cc["embalagem"] = st.sidebar.number_input("Embalagem (R$)", min_value=0.0, value=float(cc.get("embalagem", 4.0)), step=0.5, key="cc_embalagem")
cc["gramas_base"] = st.sidebar.number_input("Gramas base (proxy)", min_value=10, value=int(cc.get("gramas_base", 60)), step=5, key="cc_gramas_base")
st.session_state["cost_config"] = cc


if not df_enriched.empty:
    st.sidebar.markdown("### ✅ Saúde dos Dados")

    raw_total = df.attrs.get("rows_raw_total", "—")
    after_price = df.attrs.get("rows_after_price_clean", "—")
    valid_total = df.attrs.get("rows_after_filter_valid", "—")

    st.sidebar.caption(f"Raw total: {raw_total} | Pós preço: {after_price} | Válidos: {valid_total}")

    per_source = df.attrs.get("per_source", [])
    if per_source:
        psrc = pd.DataFrame(per_source)
        if not psrc.empty:
            for _, r in psrc.iterrows():
                st.sidebar.caption(f"{r['fonte']}: raw {int(r['raw'])} → válidos {int(r['validos'])}")

    st.sidebar.metric("Itens válidos (TOTAL pós-limpeza)", int(len(df_enriched)))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Status do Modelo de Preço")
    if isinstance(price_metrics, dict) and "ERROR" in price_metrics:
        st.sidebar.error(f"Treino falhou: {price_metrics['ERROR']}")
        st.sidebar.caption(f"Linhas treino: {price_metrics.get('TRAIN_ROWS','—')} | mínimo: {price_metrics.get('MIN_SAMPLES','—')}")
    else:
        if isinstance(price_metrics, dict):
            st.sidebar.success(f"Ativo • MAE {format_brl(price_metrics.get('MAE', 0))} • R² {price_metrics.get('R2', 0):.3f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filtro de Ticket")

    max_val = float(df_enriched["Preco_Num"].max())
    preco_max = st.sidebar.slider("Teto de Preço (R$)", 0.0, max_val, min(500.0, max_val))
    fontes_sel = st.sidebar.multiselect("Fontes", df_enriched["FONTE"].unique(), default=df_enriched["FONTE"].unique())

    df_filtered = df_enriched[
        (df_enriched["FONTE"].isin(fontes_sel)) &
        (df_enriched["Preco_Num"] <= preco_max)
    ].copy()

    cats = st.sidebar.multiselect("Categorias", df_filtered["CATEGORIA"].unique())
    if cats:
        df_filtered = df_filtered[df_filtered["CATEGORIA"].isin(cats)].copy()

    st.sidebar.metric("Itens válidos (filtro atual)", int(len(df_filtered)))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Ajustes de ML")
    n_clusters = st.sidebar.slider("Clusters de Mercado (aprox.)", 6, 40, 18)
    gap_weight_ticket = st.sidebar.slider("Peso: Ticket", 0.0, 1.0, 0.55)
    gap_weight_comp = st.sidebar.slider("Peso: Baixa Competição", 0.0, 1.0, 0.30)
    gap_weight_flash = st.sidebar.slider("Peso: Flash", 0.0, 1.0, 0.15)

    @st.cache_data(ttl=300)
    def rerun_market_clusters(d, k):
        return market_clusters(d, n_clusters=k)

    df_filtered = rerun_market_clusters(df_filtered, n_clusters)

    gap_df = gap_finder(df_filtered)
    if not gap_df.empty:
        gap_df["score_base"] = (
            (gap_df["ticket"] / (gap_df["ticket"].median() + 1e-9)) * gap_weight_ticket
            + (1.0 - (gap_df["itens"] / (gap_df["itens"].max() + 1e-9))) * gap_weight_comp
            + (gap_df["flash_share"]) * gap_weight_flash
        )
        gap_df = gap_df.sort_values("score_base", ascending=False)

    df_filtered = apply_price_model(price_model, df_filtered)
    if isinstance(price_metrics, dict) and "MAE" in price_metrics and "Preco_Previsto" in df_filtered.columns:
        mae_global = float(price_metrics["MAE"])
        df_filtered["Delta_Preco"] = df_filtered["Preco_Num"] - df_filtered["Preco_Previsto"]
        df_filtered["Faixa_Min"] = np.maximum(0.0, df_filtered["Preco_Previsto"] - mae_global)
        df_filtered["Faixa_Max"] = df_filtered["Preco_Previsto"] + mae_global

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab_otimizador, tab_bcg = st.tabs([
        "📊 Visão Geral",
        "⚔️ Comparador",
        "🧠 IA & Insights",
        "🧪 Laboratório",
        "💡 Criador",
        "📂 Dados",
        "🧩 Mercado & Clusters",
        "💸 Precificação ML",
        "🚨 Alertas",
        "🏭 Simulador",
        "🧭 Recomendador",
        "📈 Forecast",
        "🏗️ Otimizador",
        "🧩 Matriz BCG"
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))
        media = pd.to_numeric(df_filtered["Preco_Num"], errors="coerce").fillna(0).mean()
        c2.metric("Ticket Médio", format_brl(media))
        c3.metric("Fontes", len(df_filtered["FONTE"].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered["Logistica"] == "⚡ FLASH"]))

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE, title="Distribuição de Preços (Limpa)"),
                            use_container_width=True)
        with col_g2:
            st.plotly_chart(px.pie(df_filtered, names="CATEGORIA", title="Share de Categorias"), use_container_width=True)

        st.markdown("---")
        
        st.markdown("---")
        st.subheader("☁️ Nuvens de Palavras (Visão Geral)")
        excluir = st.text_input("Palavras para excluir (separe por vírgula)", value="", key="wc_excluir_overview")
        extras = set([normalize_text(x).strip() for x in excluir.split(",") if normalize_text(x).strip()])

        sw_local = set(STOPWORDS)
        sw_local.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "mm", "peças", "pecas", "und", "unid"])
        sw_local.update(extras)

        c_cloud1, c_cloud2 = st.columns(2)
        with c_cloud1:
            st.caption("📌 MAIS FREQUENTES (o que mais aparece nos títulos)")
            texto_geral = " ".join(df_filtered["PRODUTO"].astype(str)) if "PRODUTO" in df_filtered.columns else ""
            try:
                if texto_geral.strip():
                    wc1 = WordCloud(width=500, height=360, background_color="white", stopwords=sw_local).generate(texto_geral)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(wc1)
                    ax1.axis("off")
                    st.pyplot(fig1)
                else:
                    st.info("Sem texto suficiente para gerar a nuvem.")
            except Exception as e:
                st.warning(f"Não consegui gerar a nuvem: {e}")

        with c_cloud2:
            st.caption("💰 MAIOR VALOR (palavras associadas a preços mais altos)")
            try:
                word_prices = {}
                if ("PRODUTO" in df_filtered.columns) and ("Preco_Num" in df_filtered.columns):
                    for _, row in df_filtered.iterrows():
                        palavras = str(row["PRODUTO"]).lower().split()
                        for pz in palavras:
                            if pz not in sw_local and len(pz) > 3:
                                word_prices.setdefault(pz, []).append(float(row["Preco_Num"]))
                if word_prices:
                    avg_prices = {k: sum(v)/len(v) for k, v in word_prices.items() if len(v) > 1}
                    if avg_prices:
                        wc2 = WordCloud(width=500, height=360, background_color="#222", max_words=60).generate_from_frequencies(avg_prices)
                        fig2, ax2 = plt.subplots()
                        ax2.imshow(wc2)
                        ax2.axis("off")
                        st.pyplot(fig2)
                    else:
                        st.info("Dados insuficientes para calcular valor por palavra.")
                else:
                    st.info("Sem dados suficientes para gerar a nuvem por valor.")
            except Exception as e:
                st.warning(f"Não consegui gerar a nuvem por valor: {e}")

        st.subheader("🧠 Modo CEO — decisões")
        for m in build_ceo_summary(df_filtered, gap_df)[:12]:
            st.write(m)

    with tab2:
        st.header("⚔️ Comparador de Preços")
        col_input, col_check = st.columns([3, 1])
        with col_input:
            termo = st.text_input("Filtrar Produto:", placeholder="Ex: Vaso Robert")
        with col_check:
            st.write("")
            st.write("")
            mostrar_tudo = st.checkbox("Ver Todos", value=False)

        df_comp = pd.DataFrame()
        if mostrar_tudo:
            df_comp = df_filtered
        elif termo:
            prods = df_filtered["PRODUTO"].unique()
            matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 40]
            df_comp = df_filtered[df_filtered["PRODUTO"].isin(similares)]

        if not df_comp.empty:
            cols_metrics = st.columns(len(df_comp["FONTE"].unique()) + 1)
            for i, fonte in enumerate(df_comp["FONTE"].unique()):
                media_local = pd.to_numeric(df_comp[df_comp["FONTE"] == fonte]["Preco_Num"], errors="coerce").fillna(0).mean()
                cols_metrics[i].metric(f"Média {fonte}", format_brl(media_local))

            fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE, size="Preco_Num",
                                  hover_data=["PRODUTO"], title="Comparativo de Preços")
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "LINK"]], hide_index=True, use_container_width=True)
        else:
            if not mostrar_tudo:
                st.info("Busque um produto acima.")

    with tab3:
        st.subheader("Nuvens de Inteligência")

        excluir_wc = st.text_input("Palavras para excluir (separe por vírgula)", value="", key="wc_excluir_tab3")
        extras_wc = set([normalize_text(x).strip() for x in excluir_wc.split(",") if normalize_text(x).strip()])

        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "peças"])
        sw.update(extras_wc)

        c_cloud1, c_cloud2 = st.columns(2)
        with c_cloud1:
            st.caption("☁️ MAIS FREQUENTES (O que todos vendem)")
            texto_geral = " ".join(df_filtered["PRODUTO"].astype(str))
            try:
                wc1 = WordCloud(width=400, height=300, background_color="white", stopwords=sw, colormap="Blues").generate(texto_geral)
                fig1, ax1 = plt.subplots()
                ax1.imshow(wc1)
                ax1.axis("off")
                st.pyplot(fig1)
            except Exception:
                st.warning("Sem dados.")

        with c_cloud2:
            st.caption("💰 MAIOR VALOR AGREGADO (O que custa caro)")
            word_prices = {}
            for _, row in df_filtered.iterrows():
                palavras = str(row["PRODUTO"]).lower().split()
                for p in palavras:
                    if p not in sw and len(p) > 3:
                        word_prices.setdefault(p, []).append(float(row["Preco_Num"]))
            if word_prices:
                avg_prices = {k: sum(v) / len(v) for k, v in word_prices.items() if len(v) > 1}
                if avg_prices:
                    wc2 = WordCloud(width=400, height=300, background_color="#222", colormap="Wistia", max_words=50).generate_from_frequencies(avg_prices)
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(wc2)
                    ax2.axis("off")
                    st.pyplot(fig2)
                else:
                    st.warning("Dados insuficientes.")

        st.markdown("---")
        st.subheader("🔎 Diagnóstico ML (rápido)")
        dA, dB, dC, dD = st.columns(4)
        dA.metric("Itens válidos (filtro)", int(len(df_filtered)))
        dB.metric("Clusters Mercado (filtro)", int(df_filtered["CLUSTER_MKT"].nunique()) if "CLUSTER_MKT" in df_filtered.columns else 0)
        dC.metric("Anomalias (filtro)", int(pd.to_numeric(df_filtered.get("is_anomaly", 0), errors="coerce").fillna(0).sum()))
        if isinstance(price_metrics, dict) and "MAE" in price_metrics:
            dD.metric("Modelo Preço", f"MAE {format_brl(price_metrics['MAE'])}")
        else:
            dD.metric("Modelo Preço", "Sem treino")
            if isinstance(price_metrics, dict) and "ERROR" in price_metrics:
                st.error(f"Erro do treino: {price_metrics['ERROR']}")

    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1:
            cx = st.selectbox("Eixo X", df_filtered.columns)
        with c2:
            cy = st.selectbox("Eixo Y", ["Preco_Num", "Dias_Producao"])
        with c3:
            tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])
        if tp == "Barras":
            st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP_FONTE), use_container_width=True)
        elif tp == "Dispersão":
            st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP_FONTE), use_container_width=True)
        elif tp == "Boxplot":
            st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP_FONTE), use_container_width=True)

    with tab5:
        st.header("Gerador de Títulos SEO")
        keyword = st.text_input("Produto:", "Vaso")
        if keyword:
            df_c = df[df["PRODUTO"].str.contains(keyword, case=False, na=False)]
            if not df_c.empty:
                txt = " ".join(df_c["PRODUTO"].astype(str))
                pals = [p for p in re.findall(r"\w+", txt.lower()) if p not in sw and len(p) > 2]
                top = [x[0].title() for x in Counter(pals).most_common(5)]
                st.success(f"Palavras-chave: {', '.join(top)}")
                st.code(f"{keyword.title()} 3D {' '.join(top[:2])} - Alta Qualidade")
            else:
                st.warning("Sem dados.")

    with tab6:
        st.dataframe(df_filtered, use_container_width=True)

    with tab7:
        st.header("🧩 Mercado & Clusters")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Mercado", int(df_filtered["CLUSTER_MKT"].nunique()))
        c2.metric("Grupos (Dedup)", int(df_filtered["GROUP_ID"].nunique()) if "GROUP_ID" in df_filtered.columns else 0)
        c3.metric("Fonte Diversa (média)", f"{df_filtered.groupby('CLUSTER_MKT')['FONTE'].nunique().mean():.1f}" if len(df_filtered) else "0")
        c4.metric("Vectorização", df_filtered.attrs.get("mkt_vectorizer", "auto"))

        st.markdown("---")
        cluster_table = df_filtered.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
            itens=("PRODUTO", "count"),
            ticket=("Preco_Num", "mean"),
            mediana=("Preco_Num", "median"),
            flash_share=("Logistica", lambda s: float((s == "⚡ FLASH").mean())),
            fonte_div=("FONTE", lambda s: int(pd.Series(s).nunique())),
        ).reset_index().sort_values("ticket", ascending=False)

        cluster_table["ticket_fmt"] = cluster_table["ticket"].apply(format_brl)
        cluster_table["mediana_fmt"] = cluster_table["mediana"].apply(format_brl)
        cluster_table["flash_%"] = (cluster_table["flash_share"] * 100).round(1)

        st.subheader("Mapa do mercado (por cluster)")
        st.dataframe(cluster_table[["CLUSTER_MKT", "CLUSTER_NOME", "itens", "ticket_fmt", "mediana_fmt", "flash_%", "fonte_div"]],
                     hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("🕳️ Gap Finder (oportunidades)")
        if gap_df is not None and not gap_df.empty:
            show_gap = gap_df.head(25).copy()
            show_gap["ticket_fmt"] = show_gap["ticket"].apply(format_brl)
            show_gap["flash_%"] = (show_gap["flash_share"] * 100).round(1)
            st.dataframe(show_gap[["CLUSTER_MKT", "CLUSTER_NOME", "score_base", "itens", "ticket_fmt", "flash_%", "EX1", "EX2", "EX3"]],
                         hide_index=True, use_container_width=True)
        else:
            st.info("Sem dados suficientes para gap finder no filtro atual.")

    with tab8:
        st.header("💸 Precificação ML")
        st.caption(f"Itens válidos TOTAL pós-limpeza: {len(df_enriched)} | mínimo para treinar: {price_metrics.get('MIN_SAMPLES', 40) if isinstance(price_metrics, dict) else 40}")

        if price_model is None or not isinstance(price_metrics, dict) or "MAE" not in price_metrics or "Preco_Previsto" not in df_filtered.columns:
            st.warning("Modelo de preço não está ativo.")
            if isinstance(price_metrics, dict) and "ERROR" in price_metrics:
                st.error(f"Erro do treino: {price_metrics['ERROR']}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE (global)", format_brl(price_metrics["MAE"]))
            c2.metric("R² (global)", f"{price_metrics['R2']:.3f}")
            c3.metric("Treino (linhas)", int(price_metrics["TRAIN_ROWS"]))
            c4.metric("Filtro (linhas)", int(len(df_filtered)))

            st.markdown("---")
            st.subheader("Mapa: Real vs Esperado (no filtro)")
            fig = px.scatter(df_filtered, x="Preco_Previsto", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE,
                             hover_data=["PRODUTO", "CATEGORIA", "Logistica"],
                             title="Real vs Esperado (acima: caros; abaixo: baratos)")
            try:
                mn = float(min(df_filtered["Preco_Previsto"].min(), df_filtered["Preco_Num"].min()))
                mx = float(max(df_filtered["Preco_Previsto"].max(), df_filtered["Preco_Num"].max()))
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Subprecificados / Caros demais (Top 20)")
            colL, colR = st.columns(2)

            with colL:
                under = df_filtered.sort_values("Delta_Preco", ascending=True).head(20).copy()
                under["real"] = under["Preco_Num"].apply(format_brl)
                under["esperado"] = under["Preco_Previsto"].apply(format_brl)
                under["faixa"] = under.apply(lambda r: f"{format_brl(r['Faixa_Min'])} ~ {format_brl(r['Faixa_Max'])}", axis=1)
                st.caption("⬇️ Abaixo do esperado")
                st.dataframe(under[["FONTE", "PRODUTO", "real", "esperado", "faixa", "LINK"]], hide_index=True, use_container_width=True)

            with colR:
                over = df_filtered.sort_values("Delta_Preco", ascending=False).head(20).copy()
                over["real"] = over["Preco_Num"].apply(format_brl)
                over["esperado"] = over["Preco_Previsto"].apply(format_brl)
                over["faixa"] = over.apply(lambda r: f"{format_brl(r['Faixa_Min'])} ~ {format_brl(r['Faixa_Max'])}", axis=1)
                st.caption("⬆️ Acima do esperado")
                st.dataframe(over[["FONTE", "PRODUTO", "real", "esperado", "faixa", "LINK"]], hide_index=True, use_container_width=True)

    with tab9:
        st.header("🚨 Alertas & Anomalias")
        if "is_anomaly" in df_filtered.columns:
            anom = df_filtered[pd.to_numeric(df_filtered["is_anomaly"], errors="coerce").fillna(0).astype(int) == 1].copy()
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomalias detectadas", int(len(anom)))
            c2.metric("Anomalias (%)", f"{(len(anom) / max(1, len(df_filtered)) * 100):.1f}%")
            c3.metric("Maior anomalia (preço)", format_brl(anom["Preco_Num"].max()) if len(anom) else "R$ 0,00")
            st.markdown("---")
            if len(anom):
                anom = anom.sort_values("Preco_Num", ascending=False)
                st.dataframe(anom[["FONTE", "PRODUTO", "Preco_Num", "CATEGORIA", "Logistica", "LINK"]], hide_index=True, use_container_width=True)
            else:
                st.success("Sem anomalias no filtro atual.")
        else:
            st.info("Sem colunas de anomalia (algo impediu o cálculo).")

    with tab10:
        st.header("🏭 Simulador Operacional (Lucro / Hora)")
        st.caption("Simulador paramétrico para decisão (FDM). Ajuste custos e priorize o que paga a impressora mais rápido.")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        custo_hora = col1.number_input("Custo/hora máquina (R$)", min_value=0.0, value=8.0, step=0.5)
        custo_grama = col2.number_input("Custo/grama filamento (R$)", min_value=0.0, value=0.12, step=0.01, format="%.2f")
        gramas_base = col3.number_input("Gramas base (proxy)", min_value=10, value=60, step=5)
        taxa_falha = col4.number_input("Taxa falha/refugo", min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.2f")
        taxa_market = col5.number_input("Taxa marketplace", min_value=0.0, max_value=0.5, value=0.14, step=0.01, format="%.2f")
        embalagem = col6.number_input("Embalagem (R$)", min_value=0.0, value=4.0, step=0.5)

        sim_df = compute_profit(df_filtered, custo_hora=custo_hora, custo_grama=custo_grama,
                                gramas_base=gramas_base, taxa_falha=taxa_falha,
                                taxa_marketplace=taxa_market, embalagem=embalagem)

        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Lucro médio (estimado)", format_brl(sim_df["Lucro_Estimado"].mean()))
        cB.metric("Lucro/hora médio", format_brl(sim_df["Lucro_por_Hora"].mean()))
        cC.metric("Top lucro/hora", format_brl(sim_df["Lucro_por_Hora"].max()))
        cD.metric("Itens com lucro negativo", int((sim_df["Lucro_Estimado"] < 0).sum()))

        st.markdown("---")
        st.subheader("Ranking: melhor uso de máquina (Top 30)")
        top = sim_df.sort_values("Lucro_por_Hora", ascending=False).head(30).copy()
        top["Lucro"] = top["Lucro_Estimado"].apply(format_brl)
        top["Lucro/H"] = top["Lucro_por_Hora"].apply(format_brl)
        top["Preço"] = top["Preco_Num"].apply(format_brl)
        st.dataframe(top[["FONTE", "PRODUTO", "Preço", "Lucro", "Lucro/H", "Horas_Estimadas", "Gramagem_Estimada", "LINK"]],
                     hide_index=True, use_container_width=True)

    with tab11:
        st.header("🧭 Recomendador (o que listar / produzir)")
        st.caption("Combina: Gap Finder + preço esperado (se ativo) + flash + (opcional) penaliza anomalia.")

        base = df_filtered.copy()
        has_price = ("Delta_Preco" in base.columns) and base["Delta_Preco"].notna().any()

        cluster_score_map = {}
        if gap_df is not None and not gap_df.empty:
            cluster_score_map = dict(zip(gap_df["CLUSTER_MKT"].astype(int), gap_df["score_base"].astype(float)))

        base["cluster_score"] = base["CLUSTER_MKT"].astype(int).map(cluster_score_map).fillna(0.0)
        base["flash_flag"] = (base["Logistica"] == "⚡ FLASH").astype(int)
        base["anom_penalty"] = pd.to_numeric(base.get("is_anomaly", 0), errors="coerce").fillna(0).astype(int)

        if has_price:
            base["under_score"] = (-base["Delta_Preco"]).clip(lower=0)
            base["under_score"] = base["under_score"] / (base["under_score"].max() + 1e-9)
        else:
            base["under_score"] = 0.0

        base["score_rec"] = (base["cluster_score"] * 0.55 + base["under_score"] * 0.25 + base["flash_flag"] * 0.15 - base["anom_penalty"] * 0.10)
        rec = base.sort_values("score_rec", ascending=False).head(40).copy()

        view = rec[["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preco_Num", "Logistica", "LINK"]].copy()
        view["Preço"] = view["Preco_Num"].apply(format_brl)
        st.dataframe(view[["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preço", "Logistica", "LINK"]],
                     hide_index=True, use_container_width=True)

    with tab12:
        st.header("📈 Forecast (se houver data no dataset)")
        st.caption("Se sua planilha tiver coluna de data/hora, aqui entra previsão real. Sem data, mostra aviso.")

        date_col = None
        for c in df.columns:
            if any(k in c.upper() for k in ["DATA", "DATE", "DIA", "HORA", "TIMESTAMP"]):
                date_col = c
                break

        if date_col is None:
            st.info("Não encontrei coluna de data/hora no CSV atual. Se você adicionar no Google Sheets (ex: Data/Hora), essa aba vira previsão real.")
        else:
            try:
                tmp = df.copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                if tmp.empty:
                    st.warning("Coluna de data existe, mas não consegui parsear as datas.")
                else:
                    tmp["dia"] = tmp[date_col].dt.date
                    ts = tmp.groupby("dia")["Preco_Num"].mean().reset_index()
                    ts["dia"] = pd.to_datetime(ts["dia"])

                    st.subheader("Série: ticket médio diário")
                    st.plotly_chart(px.line(ts, x="dia", y="Preco_Num", title="Ticket médio diário (observado)"), use_container_width=True)

                    st.subheader("Previsão (proxy): média móvel + tendência linear")
                    ts = ts.sort_values("dia")
                    ts["mm7"] = ts["Preco_Num"].rolling(7, min_periods=3).mean()
                    coef = np.polyfit(np.arange(len(ts)), ts["Preco_Num"].values, deg=1)
                    ts["trend"] = coef[0] * np.arange(len(ts)) + coef[1]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts["dia"], y=ts["Preco_Num"], mode="lines+markers", name="observado"))
                    fig.add_trace(go.Scatter(x=ts["dia"], y=ts["mm7"], mode="lines", name="mm7"))
                    fig.add_trace(go.Scatter(x=ts["dia"], y=ts["trend"], mode="lines", name="tendência"))
                    fig.update_layout(title="Ticket: observado vs suavização vs tendência")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Não consegui gerar forecast: {e}")


    # 13. OTIMIZADOR
    with tab_otimizador:
        st.header("🏗️ Otimizador de Fábrica (mix de produção)")
        st.caption("Usa matemática (linprog) para sugerir quantidades por cluster, maximizando lucro total com limite de horas semanais.")

        hours_avail = st.number_input("Horas disponíveis na semana", min_value=1.0, value=40.0, step=1.0)
        st.caption("Dica: instale `scipy` no requirements.txt para habilitar o otimizador.")

        # Recalcula simulador usando custos persistentes (sidebar)
        cc = st.session_state["cost_config"]
        sim_df_opt = compute_profit(
            df_filtered,
            custo_hora=float(cc.get("custo_hora", 8.0)),
            custo_grama=float(cc.get("custo_grama", 0.12)),
            gramas_base=int(cc.get("gramas_base", 60)),
            taxa_falha=float(cc.get("taxa_falha", 0.06)),
            taxa_marketplace=float(cc.get("taxa_mkt", 0.14)),
            embalagem=float(cc.get("embalagem", 4.0)),
        )

        if st.button("Rodar otimizador"):
            try:
                plan = run_factory_optimizer(sim_df_opt, hours_avail=float(hours_avail))
                if plan.empty:
                    st.info("Sem dados suficientes para otimizar.")
                else:
                    plan_view = plan.copy()
                    plan_view["lucro_medio_fmt"] = plan_view["lucro_medio"].apply(format_brl)
                    plan_view["preco_medio_fmt"] = plan_view["preco_medio"].apply(format_brl)
                    plan_view["lucro_total_fmt"] = plan_view["lucro_total"].apply(format_brl)
                    st.dataframe(
                        plan_view[["CLUSTER_NOME","itens","preco_medio_fmt","lucro_medio_fmt","horas_media","qtd_sugerida_round","horas_usadas_round","lucro_total_fmt"]],
                        hide_index=True,
                        use_container_width=True,
                    )

                    csv = plan_view.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Baixar plano (CSV)", data=csv, file_name="otimizador_plano.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Otimizador indisponível/erro: {e}")

    # 14. MATRIZ BCG
    with tab_bcg:
        st.header("🧩 Matriz BCG (Volume x Margem)")
        st.caption("Classifica clusters em: Estrela, Vaca Leiteira, Aposta e Abacaxi (thresholds = medianas do filtro).")

        cc = st.session_state["cost_config"]
        sim_df_bcg = compute_profit(
            df_filtered,
            custo_hora=float(cc.get("custo_hora", 8.0)),
            custo_grama=float(cc.get("custo_grama", 0.12)),
            gramas_base=int(cc.get("gramas_base", 60)),
            taxa_falha=float(cc.get("taxa_falha", 0.06)),
            taxa_marketplace=float(cc.get("taxa_mkt", 0.14)),
            embalagem=float(cc.get("embalagem", 4.0)),
        )

        bcg = build_bcg_matrix(sim_df_bcg)
        if bcg.empty:
            st.info("Sem dados suficientes para montar a Matriz BCG.")
        else:
            vol_cut = bcg.attrs.get("vol_cut", bcg["volume"].median())
            mar_cut = bcg.attrs.get("mar_cut", bcg["margem_pct"].median())

            fig = px.scatter(
                bcg,
                x="volume",
                y="margem_pct",
                color="tipo_bcg",
                hover_data=["CLUSTER_NOME", "ticket", "lucro_h"],
                title="Matriz BCG por Cluster (Volume x Margem %)",
            )
            # linhas de corte
            fig.add_shape(type="line", x0=vol_cut, y0=bcg["margem_pct"].min(), x1=vol_cut, y1=bcg["margem_pct"].max())
            fig.add_shape(type="line", x0=bcg["volume"].min(), y0=mar_cut, x1=bcg["volume"].max(), y1=mar_cut)
            st.plotly_chart(fig, use_container_width=True)

            bcg_view = bcg.copy()
            bcg_view["ticket_fmt"] = bcg_view["ticket"].apply(format_brl)
            bcg_view["margem_%"] = bcg_view["margem_pct"].round(1)
            st.dataframe(bcg_view[["CLUSTER_NOME","tipo_bcg","volume","margem_%","ticket_fmt","lucro_h"]], hide_index=True, use_container_width=True)

else:
    st.error("⚠️ Erro ao carregar dados. Verifique o Google Sheets (ou uma oscilacao de rede).")
per_source = df.attrs.get("per_source", []) if isinstance(df, pd.DataFrame) else []
if per_source:
    st.subheader("Diagnostico de carga (por fonte)")
    st.dataframe(pd.DataFrame(per_source), use_container_width=True)
    st.caption("Dica: como agora existe cache (ttl=600s), recarregar a pagina deve voltar mais rapido.")

