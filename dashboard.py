# -*- coding: utf-8 -*-
# dashboard.py
# BCRUZ 3D Enterprise - Decision Intelligence Edition
# Mantém o dashboard original e adiciona módulos de decisão + ML.
# Inclui: contagem de itens válidos por etapa, agrupamento, clusters, precificação ML,
# MAE por cluster, recomendador, alertas, simulador e busca na aba Dados.
# Cores fixas: Shopee=laranja, Elo7=verde.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from thefuzz import process, fuzz

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

# -----------------------------
# 1) CONFIG
# -----------------------------
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

COLOR_MAP = {
    "Shopee": "#ff7a00",  # laranja
    "Elo7": "#1db954",    # verde
}

# -----------------------------
# 2) LINKS
# -----------------------------
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# -----------------------------
# 3) LIMPEZA DE PREÇO
# -----------------------------
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
        except:
            return 0.0

    if val > 1500.0:
        return 0.0

    return val


# -----------------------------
# UTIL
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

    def has(pat): 
        return int(bool(re.search(pat, t)))

    is_kit = has(r"\bkit\b|\bconjunto\b|\bcombo\b")
    is_personalizado = has(r"\bpersonaliz")
    is_pronta_entrega = has(r"\bpronta\b|\bpronto\b|\bimediat")
    is_decor = has(r"\bdecor\b|\bdecora")
    is_organizador = has(r"\borganiz")
    is_suporte = has(r"\bsuporte\b|\bstand\b|\bbase\b")
    is_vaso = has(r"\bvaso\b|\bplant")
    is_action = has(r"\bfigure\b|\baction\b|\bminiatura\b|\bstatue\b|\bestatua\b")
    is_gamer = has(r"\bgamer\b|\bplaystation\b|\bxbox\b|\bnintendo\b|\bpc\b")

    nums = re.findall(r"\b(\d{1,3})\s?(cm|mm)?\b", t)
    size_num = 0
    for n, unit in nums:
        try:
            v = int(n)
            if unit == "mm":
                v = int(round(v / 10))
            size_num = max(size_num, v)
        except:
            pass

    premium = has(r"\bpremium\b|\bdeluxe\b|\bmetal\b|\bvelvet\b|\babs\b|\bpetg\b|\bresina\b")

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
    except:
        return "R$ 0,00"


# -----------------------------
# CARREGAMENTO + CONTAGEM POR ETAPA
# -----------------------------
@st.cache_data(ttl=120)
def carregar_dados():
    dfs = []
    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]
    per_source = []

    raw_total = 0
    after_price_total = 0
    valid_total = 0

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines="skip", dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]

            raw_n = int(len(temp_df))
            raw_total += raw_n
            if temp_df.empty:
                per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": 0})
                continue

            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRECO", "PRICE"])), None)
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

            after_price_total += int(len(temp_df))

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
            valid_n = int(len(temp_df))
            valid_total += valid_n

            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": valid_n})

            dfs.append(temp_df[cols])

        except:
            per_source.append({"fonte": f["nome"], "raw": 0, "validos": 0})

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not final_df.empty:
        corte_superior = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= corte_superior].copy()

    final_df.attrs["rows_raw_total"] = int(raw_total)
    final_df.attrs["rows_after_price_clean"] = int(after_price_total)
    final_df.attrs["rows_after_filter_valid"] = int(valid_total)
    final_df.attrs["per_source"] = per_source

    return final_df


# -----------------------------
# ENRIQUECIMENTO
# -----------------------------
@st.cache_data(ttl=300)
def enrich_df(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return base_df

    d = base_df.copy()
    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)
    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # Anomalias: usar apenas preço (denso)
    try:
        iso = IsolationForest(n_estimators=250, contamination=0.05, random_state=42)
        d["is_anomaly_iso"] = (iso.fit_predict(d[["Preco_Num"]].astype(float)) == -1).astype(int)
    except:
        d["is_anomaly_iso"] = 0

    try:
        nn = int(min(35, max(10, len(d) // 25)))
        lof = LocalOutlierFactor(n_neighbors=nn)
        d["is_anomaly_lof"] = (lof.fit_predict(d[["Preco_Num"]].astype(float)) == -1).astype(int)
    except:
        d["is_anomaly_lof"] = 0

    d["is_anomaly"] = ((d["is_anomaly_iso"] + d["is_anomaly_lof"]) > 0).astype(int)
    return d


# -----------------------------
# VETORES DE TEXTO (SBERT opcional)
# -----------------------------
@st.cache_data(ttl=300)
def compute_text_vectors(texts: pd.Series, method: str = "auto", max_features: int = 4000):
    texts = texts.fillna("").astype(str).tolist()

    if method in ("auto", "sbert"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            X = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.array(X), "SBERT MiniLM multilingual"
        except:
            if method == "sbert":
                pass

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(texts)
    n_comp = int(min(128, max(16, X_tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, f"TF-IDF + SVD({n_comp})"


# -----------------------------
# DEDUP (canonicalização)
# -----------------------------
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
        out["GROUP_ID"] = km.fit_predict(X)
    except:
        out["GROUP_ID"] = 0

    canon_map = {}
    for gid, grp in out.groupby("GROUP_ID"):
        counts = grp["PRODUTO"].astype(str).value_counts()
        best = counts.index[0] if len(counts) else grp["PRODUTO"].astype(str).iloc[0]
        candidates = counts.index.tolist()[:10]
        if candidates:
            best = sorted(candidates, key=lambda s: (len(s), s.lower()))[0]
        canon_map[int(gid)] = best

    out["PRODUTO_CANON"] = out["GROUP_ID"].map(canon_map).fillna(out["PRODUTO"].astype(str))
    out.attrs["vectorizer"] = vec_name
    out.attrs["k_groups"] = int(out["GROUP_ID"].nunique())
    return out


# -----------------------------
# CLUSTER DE MERCADO
# -----------------------------
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
    except:
        out["CLUSTER_MKT"] = 0

    # nome do cluster (top termos)
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
            mean_vec = np.asarray(X_t[idx].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[-4:][::-1]
            top_terms = [vocab[i] for i in top_idx if mean_vec[i] > 0]
            cluster_names[cid] = " / ".join(top_terms) if top_terms else f"Cluster {cid}"

        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].map(cluster_names).fillna(out["CLUSTER_MKT"].astype(str))
    except:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    out.attrs["mkt_vectorizer"] = vec_name
    out.attrs["mkt_k"] = int(out["CLUSTER_MKT"].nunique())
    return out


# -----------------------------
# GAP FINDER
# -----------------------------
@st.cache_data(ttl=300)
def gap_finder(d: pd.DataFrame):
    if d is None or d.empty or "CLUSTER_MKT" not in d.columns:
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
    return g.merge(exdf, on="CLUSTER_MKT", how="left").sort_values("score_base", ascending=False)


# -----------------------------
# MODELO DE PREÇO (corrigido: suporta sparse)
# -----------------------------
def _to_dense(X):
    # ColumnTransformer geralmente retorna matriz esparsa
    return X.toarray() if hasattr(X, "toarray") else X

@st.cache_data(ttl=300)
def train_price_model(d: pd.DataFrame, min_samples: int = 80):
    """
    Treina regressor pra prever Preco_Num no dataset TOTAL.
    Corrige o erro: 'Sparse data ... dense required' convertendo pra denso antes do modelo.
    """
    if d is None or d.empty or len(d) < min_samples:
        return None, None, d

    data = d.copy()
    data["Preco_Num"] = pd.to_numeric(data["Preco_Num"], errors="coerce").fillna(0.0).astype(float)

    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    for c in cat_cols:
        if c not in data.columns:
            data[c] = "NA"
        data[c] = data[c].fillna("NA").astype(str)

    for c in num_cols:
        if c not in data.columns:
            data[c] = 0
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0).astype(float)

    if text_col not in data.columns:
        data[text_col] = data["PRODUTO"].astype(str).apply(normalize_text)

    X = data[[text_col] + cat_cols + num_cols]
    y = data["Preco_Num"].astype(float)

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=2, max_df=0.95), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.9,
    )

    # Modelo que é forte e rápido; Ridge lida bem com alta dimensionalidade.
    model = Ridge(alpha=1.5, random_state=42)

    pipe = Pipeline(steps=[
        ("prep", preproc),
        ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
        ("model", model),
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
    except Exception as e:
        metrics = {"ERROR": str(e), "TRAIN_ROWS": int(len(data))}
        return None, metrics, d

    # prever no dataset completo
    data["Preco_Previsto"] = pipe.predict(X)
    data["Delta_Preco"] = data["Preco_Num"] - data["Preco_Previsto"]
    data["Faixa_Min"] = np.maximum(0.0, data["Preco_Previsto"] - mae)
    data["Faixa_Max"] = data["Preco_Previsto"] + mae

    metrics = {"MAE": mae, "R2": r2, "MIN_SAMPLES": int(min_samples), "TRAIN_ROWS": int(len(data))}
    return pipe, metrics, data


def apply_price_model(model_pipe, d: pd.DataFrame):
    if model_pipe is None or d is None or d.empty:
        return d

    data = d.copy()
    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    for c in cat_cols:
        if c not in data.columns:
            data[c] = "NA"
        data[c] = data[c].fillna("NA").astype(str)

    for c in num_cols:
        if c not in data.columns:
            data[c] = 0
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0).astype(float)

    if text_col not in data.columns:
        data[text_col] = data["PRODUTO"].astype(str).apply(normalize_text)

    X = data[[text_col] + cat_cols + num_cols]
    try:
        data["Preco_Previsto"] = model_pipe.predict(X)
    except:
        return d

    return data


@st.cache_data(ttl=300)
def mae_por_cluster(d: pd.DataFrame):
    if d is None or d.empty:
        return pd.DataFrame()
    if "Preco_Previsto" not in d.columns or "CLUSTER_MKT" not in d.columns:
        return pd.DataFrame()

    tmp = d.dropna(subset=["Preco_Num", "Preco_Previsto"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["abs_err"] = (tmp["Preco_Num"].astype(float) - tmp["Preco_Previsto"].astype(float)).abs()
    g = tmp.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
        itens=("PRODUTO", "count"),
        mae=("abs_err", "mean"),
        ticket=("Preco_Num", "mean"),
    ).reset_index().sort_values("mae")
    g["mae_fmt"] = g["mae"].apply(format_brl)
    g["ticket_fmt"] = g["ticket"].apply(format_brl)
    return g


# -----------------------------
# SIMULADOR (lucro/hora)
# -----------------------------
def estimate_print_hours(row, base_hours=2.0):
    days = float(row.get("Dias_Producao", 15))
    size = float(row.get("size_num", 0))
    logist = str(row.get("Logistica", "📦 NORMAL"))

    h = base_hours
    if size > 0:
        h += min(6.0, size / 25.0)
    h += min(8.0, max(0.0, (days - 2.0)) / 6.0)
    if "FLASH" in logist:
        h *= 0.75

    return float(np.clip(h, 0.4, 18.0))


def compute_profit(d: pd.DataFrame, custo_hora=8.0, custo_grama=0.12, gramas_base=60,
                   taxa_falha=0.06, taxa_marketplace=0.14, embalagem=4.0):
    if d is None or d.empty:
        return d

    out = d.copy()
    size = out.get("size_num", pd.Series([0] * len(out))).fillna(0).astype(float)
    kit = out.get("is_kit", pd.Series([0] * len(out))).fillna(0).astype(int)
    prem = out.get("premium", pd.Series([0] * len(out))).fillna(0).astype(int)

    out["Gramagem_Estimada"] = (gramas_base + (size * 2.2) + (kit * 40) + (prem * 25)).clip(lower=20, upper=600)
    out["Horas_Estimadas"] = out.apply(lambda r: estimate_print_hours(r, base_hours=2.0), axis=1)

    out["Custo_Material"] = out["Gramagem_Estimada"] * float(custo_grama)
    out["Custo_Maquina"] = out["Horas_Estimadas"] * float(custo_hora)
    out["Custo_Embalagem"] = float(embalagem)
    out["Fee_Marketplace"] = out["Preco_Num"].astype(float) * float(taxa_marketplace)

    out["Custo_Total_Estimado"] = (out["Custo_Material"] + out["Custo_Maquina"] + out["Custo_Embalagem"] + out["Fee_Marketplace"]) * (1.0 + float(taxa_falha))
    out["Lucro_Estimado"] = out["Preco_Num"].astype(float) - out["Custo_Total_Estimado"]
    out["Lucro_por_Hora"] = out["Lucro_Estimado"] / (out["Horas_Estimadas"] + 1e-9)

    return out


# -----------------------------
# MODO CEO (resumo)
# -----------------------------
def build_ceo_summary(d: pd.DataFrame, gap: pd.DataFrame):
    if d is None or d.empty:
        return ["Sem dados para gerar decisões."]

    msgs = []

    top_price = d.sort_values("Preco_Num", ascending=False).head(1)
    if len(top_price):
        msgs.append(f"Maior ticket no filtro: {format_brl(top_price['Preco_Num'].iloc[0])} - {top_price['PRODUTO'].iloc[0]} ({top_price['FONTE'].iloc[0]}).")

    if "Delta_Preco" in d.columns and d["Delta_Preco"].notna().any():
        under = d.sort_values("Delta_Preco", ascending=True).head(3)
        if len(under):
            msgs.append("Top 3 prováveis subprecificados (abaixo do esperado):")
            for _, r in under.iterrows():
                msgs.append(f"- {r['PRODUTO']} | real {format_brl(r['Preco_Num'])} vs esperado {format_brl(r.get('Preco_Previsto', 0))}")

        over = d.sort_values("Delta_Preco", ascending=False).head(3)
        if len(over):
            msgs.append("Top 3 prováveis caros demais (acima do esperado):")
            for _, r in over.iterrows():
                msgs.append(f"- {r['PRODUTO']} | real {format_brl(r['Preco_Num'])} vs esperado {format_brl(r.get('Preco_Previsto', 0))}")

    if gap is not None and not gap.empty:
        top = gap.head(3)
        msgs.append("Top 3 oportunidades por cluster (alto ticket + baixa competição relativa + flash):")
        for _, r in top.iterrows():
            msgs.append(f"- {r['CLUSTER_NOME']} | score {r['score_base']:.2f} | ticket {format_brl(r['ticket'])} | itens {int(r['itens'])}")

    if "is_anomaly" in d.columns and int(d["is_anomaly"].sum()) > 0:
        msgs.append(f"Atenção: {int(d['is_anomaly'].sum())} anomalias detectadas no filtro. Veja a aba Alertas.")

    return msgs


# ============================================================
# PIPELINE GLOBAL
# ============================================================
df = carregar_dados()
df_enriched = enrich_df(df)
df_enriched = canonicalize_products(df_enriched)
df_enriched = market_clusters(df_enriched)

# Treina no TOTAL pós-limpeza
price_model, price_metrics, df_enriched = train_price_model(df_enriched, min_samples=80)

# ============================================================
# UI SIDEBAR
# ============================================================
st.sidebar.title("🎛️ Centro de Comando")

if df_enriched is None or df_enriched.empty:
    st.error("⚠️ Erro ao carregar dados. Verifique o Google Sheets.")
    st.stop()

st.sidebar.markdown("### ✅ Saúde dos Dados")
raw_total = df.attrs.get("rows_raw_total", "—")
after_price = df.attrs.get("rows_after_price_clean", "—")
valid_total = df.attrs.get("rows_after_filter_valid", "—")
st.sidebar.caption(f"Raw: {raw_total} | Pós preço: {after_price} | Válidos: {valid_total}")

per_source = df.attrs.get("per_source", [])
if per_source:
    psrc = pd.DataFrame(per_source)
    for _, r in psrc.iterrows():
        st.sidebar.caption(f"{r['fonte']}: raw {int(r['raw'])} → válidos {int(r['validos'])}")

st.sidebar.metric("Itens válidos (TOTAL pós-limpeza)", int(len(df_enriched)))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtro de Ticket")
max_val = float(df_enriched["Preco_Num"].max())
preco_max = st.sidebar.slider("Teto de Preço (R$)", 0.0, max_val, min(500.0, max_val))

fontes_sel = st.sidebar.multiselect("Fontes", df_enriched["FONTE"].unique(), default=list(df_enriched["FONTE"].unique()))
df_filtered = df_enriched[(df_enriched["FONTE"].isin(fontes_sel)) & (df_enriched["Preco_Num"] <= preco_max)].copy()

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
if price_metrics and "Preco_Previsto" in df_filtered.columns and "MAE" in price_metrics:
    mae_global = float(price_metrics["MAE"])
    df_filtered["Delta_Preco"] = df_filtered["Preco_Num"].astype(float) - df_filtered["Preco_Previsto"].astype(float)
    df_filtered["Faixa_Min"] = np.maximum(0.0, df_filtered["Preco_Previsto"] - mae_global)
    df_filtered["Faixa_Max"] = df_filtered["Preco_Previsto"] + mae_global

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "📊 Visão Geral", "⚔️ Comparador", "🧠 IA & Insights", "🧪 Laboratório",
    "💡 Criador", "📂 Dados", "🧩 Mercado & Clusters", "💸 Precificação ML",
    "🚨 Alertas", "🏭 Simulador", "🧭 Recomendador"
])

# 1) Visão Geral
with tabs[0]:
    st.title("BCRUZ 3D Enterprise - Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Produtos", len(df_filtered))
    media = df_filtered["Preco_Num"].astype(float).mean() if len(df_filtered) else 0.0
    c2.metric("Ticket Médio", format_brl(media))
    c3.metric("Fontes", int(df_filtered["FONTE"].nunique()))
    c4.metric("Itens Flash", int((df_filtered["Logistica"] == "⚡ FLASH").sum()))

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE",
                     color_discrete_map=COLOR_MAP, title="Distribuição de Preços (Limpa)")
        st.plotly_chart(fig, use_container_width=True)
    with col_g2:
        fig = px.pie(df_filtered, names="CATEGORIA", title="Share de Categorias")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Modo CEO - decisões")
    for m in build_ceo_summary(df_filtered, gap_df)[:12]:
        st.write(m)

# 2) Comparador
with tabs[1]:
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
            media_local = df_comp[df_comp["FONTE"] == fonte]["Preco_Num"].astype(float).mean()
            cols_metrics[i].metric(f"Média {fonte}", format_brl(media_local))

        fig = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num",
                         hover_data=["PRODUTO"], title="Comparativo de Preços",
                         color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "LINK"]], hide_index=True, use_container_width=True)
    else:
        if not mostrar_tudo:
            st.info("Busque um produto acima.")

# 3) IA & Insights
with tabs[2]:
    st.subheader("Nuvens de Inteligência")

    sw = set(STOPWORDS)
    sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "peças", "pecas"])

    c_cloud1, c_cloud2 = st.columns(2)

    with c_cloud1:
        st.caption("☁️ MAIS FREQUENTES (o que todos vendem)")
        texto_geral = " ".join(df_filtered["PRODUTO"].astype(str))
        try:
            wc1 = WordCloud(width=400, height=300, background_color="white", stopwords=sw).generate(texto_geral)
            fig1, ax1 = plt.subplots()
            ax1.imshow(wc1)
            ax1.axis("off")
            st.pyplot(fig1)
        except:
            st.warning("Sem dados.")

    with c_cloud2:
        st.caption("💰 MAIOR VALOR AGREGADO (o que custa caro)")
        word_prices = {}
        for _, row in df_filtered.iterrows():
            palavras = str(row["PRODUTO"]).lower().split()
            for p in palavras:
                if p not in sw and len(p) > 3:
                    word_prices.setdefault(p, []).append(float(row["Preco_Num"]))

        if word_prices:
            avg_prices = {k: sum(v) / len(v) for k, v in word_prices.items() if len(v) > 1}
            if avg_prices:
                wc2 = WordCloud(width=400, height=300, background_color="#222", max_words=50).generate_from_frequencies(avg_prices)
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
    dC.metric("Anomalias (filtro)", int(df_filtered["is_anomaly"].sum()) if "is_anomaly" in df_filtered.columns else 0)

    if price_metrics and "MAE" in price_metrics and "R2" in price_metrics and not price_metrics.get("ERROR"):
        dD.metric("Modelo Preço (MAE global)", format_brl(price_metrics["MAE"]))
        st.caption(f"Treinado no TOTAL com {price_metrics['TRAIN_ROWS']} itens (mínimo: {price_metrics.get('MIN_SAMPLES', '—')}).")
    else:
        dD.metric("Modelo Preço", "Sem treino")
        if price_metrics and price_metrics.get("ERROR"):
            st.error(f"Erro do treino: {price_metrics['ERROR']}")

# 4) Laboratório
with tabs[3]:
    c1, c2, c3 = st.columns(3)
    with c1:
        cx = st.selectbox("Eixo X", df_filtered.columns)
    with c2:
        cy = st.selectbox("Eixo Y", ["Preco_Num", "Dias_Producao"])
    with c3:
        tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])

    if tp == "Barras":
        fig = px.bar(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    elif tp == "Dispersão":
        fig = px.scatter(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)

# 5) Criador
with tabs[4]:
    st.header("Gerador de Títulos SEO")
    keyword = st.text_input("Produto:", "Vaso")

    if keyword:
        df_c = df_enriched[df_enriched["PRODUTO"].str.contains(keyword, case=False, na=False)]
        if not df_c.empty:
            txt = " ".join(df_c["PRODUTO"].astype(str))
            pals = [p for p in re.findall(r"\w+", txt.lower()) if p not in sw and len(p) > 2]
            top = [x[0].title() for x in Counter(pals).most_common(7)]
            st.success(f"Palavras-chave: {', '.join(top[:7])}")
            st.code(f"{keyword.title()} 3D {' '.join(top[:2])} - Alta Qualidade")
        else:
            st.warning("Sem dados.")

# 6) Dados (com busca)
with tabs[5]:
    st.header("📂 Dados (com busca)")
    q = st.text_input("Buscar produto (contém):", "")
    view = df_filtered.copy()
    if q.strip():
        qq = q.strip().lower()
        view = view[view["PRODUTO"].astype(str).str.lower().str.contains(qq, na=False)].copy()
    st.caption(f"Itens mostrados: {len(view)}")
    st.dataframe(view, use_container_width=True)

# 7) Mercado & Clusters
with tabs[6]:
    st.header("🧩 Mercado & Clusters")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters Mercado", int(df_filtered["CLUSTER_MKT"].nunique()))
    c2.metric("Grupos (Dedup)", int(df_filtered["GROUP_ID"].nunique()) if "GROUP_ID" in df_filtered.columns else 0)
    c3.metric("Fonte Diversa (média)", f"{df_filtered.groupby('CLUSTER_MKT')['FONTE'].nunique().mean():.1f}" if len(df_filtered) else "0.0")
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
    st.dataframe(
        cluster_table[["CLUSTER_MKT", "CLUSTER_NOME", "itens", "ticket_fmt", "mediana_fmt", "flash_%", "fonte_div"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        fig = px.scatter(
            cluster_table,
            x="itens",
            y="ticket",
            size="itens",
            hover_data=["CLUSTER_NOME", "flash_%", "fonte_div"],
            title="Cluster: Ticket vs Competição (itens)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        topN = cluster_table.head(15).copy()
        fig = px.bar(
            topN.sort_values("ticket"),
            x="ticket",
            y="CLUSTER_NOME",
            orientation="h",
            title="Top 15 Clusters por Ticket Médio",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🕳️ Gap Finder (oportunidades)")
    if not gap_df.empty:
        show_gap = gap_df.head(25).copy()
        show_gap["ticket_fmt"] = show_gap["ticket"].apply(format_brl)
        show_gap["flash_%"] = (show_gap["flash_share"] * 100).round(1)
        st.dataframe(
            show_gap[["CLUSTER_MKT", "CLUSTER_NOME", "score_base", "itens", "ticket_fmt", "flash_%", "EX1", "EX2", "EX3"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Sem dados suficientes para gap finder no filtro atual.")

# 8) Precificação ML
with tabs[7]:
    st.header("💸 Precificação ML")

    st.caption(f"Itens válidos TOTAL pós-limpeza: {len(df_enriched)} | mínimo para treinar: {price_metrics.get('MIN_SAMPLES', 80) if price_metrics else 80}")

    if price_model is None or price_metrics is None or price_metrics.get("ERROR") or "Preco_Previsto" not in df_filtered.columns:
        st.warning("Modelo de preço não está ativo.")
        if price_metrics and price_metrics.get("ERROR"):
            st.error(f"Erro do treino: {price_metrics['ERROR']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (global)", format_brl(price_metrics["MAE"]))
        c2.metric("R² (global)", f"{price_metrics['R2']:.3f}")
        c3.metric("Treino (linhas)", int(price_metrics["TRAIN_ROWS"]))
        c4.metric("Filtro (linhas)", int(len(df_filtered)))

        st.markdown("---")
        st.subheader("MAE por cluster (onde o modelo erra menos/mais)")
        mae_cl = mae_por_cluster(df_filtered)
        if not mae_cl.empty:
            st.dataframe(mae_cl[["CLUSTER_MKT", "CLUSTER_NOME", "itens", "mae_fmt", "ticket_fmt"]], hide_index=True, use_container_width=True)
        else:
            st.info("Sem dados para MAE por cluster no filtro atual.")

        st.markdown("---")
        st.subheader("Mapa: Real vs Esperado (no filtro)")
        fig = px.scatter(
            df_filtered,
            x="Preco_Previsto",
            y="Preco_Num",
            color="FONTE",
            color_discrete_map=COLOR_MAP,
            hover_data=["PRODUTO", "CATEGORIA", "Logistica"],
            title="Real vs Esperado (acima: caros; abaixo: baratos)",
        )
        try:
            mn = float(min(df_filtered["Preco_Previsto"].min(), df_filtered["Preco_Num"].min()))
            mx = float(max(df_filtered["Preco_Previsto"].max(), df_filtered["Preco_Num"].max()))
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
        except:
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

# 9) Alertas
with tabs[8]:
    st.header("🚨 Alertas & Anomalias")

    if "is_anomaly" in df_filtered.columns:
        anom = df_filtered[df_filtered["is_anomaly"] == 1].copy()
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

    st.markdown("---")
    st.subheader("Distribuição por fonte")
    by_source = df_filtered.groupby("FONTE").agg(
        itens=("PRODUTO", "count"),
        ticket=("Preco_Num", "mean"),
        mediana=("Preco_Num", "median"),
        p90=("Preco_Num", lambda s: float(pd.Series(s).quantile(0.90))),
    ).reset_index()
    by_source["ticket_fmt"] = by_source["ticket"].apply(format_brl)
    by_source["p90_fmt"] = by_source["p90"].apply(format_brl)
    st.dataframe(by_source[["FONTE", "itens", "ticket_fmt", "p90_fmt"]], hide_index=True, use_container_width=True)
    fig = px.bar(by_source, x="FONTE", y="p90", title="P90 de preço por fonte (sinal de teto)", color="FONTE", color_discrete_map=COLOR_MAP)
    st.plotly_chart(fig, use_container_width=True)

# 10) Simulador
with tabs[9]:
    st.header("🏭 Simulador Operacional (Lucro / Hora)")
    st.caption("Simulador paramétrico para decisão (FDM). Ajuste custos e priorize o que paga a impressora mais rápido.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    custo_hora = col1.number_input("Custo/hora máquina (R$)", min_value=0.0, value=8.0, step=0.5)
    custo_grama = col2.number_input("Custo/grama filamento (R$)", min_value=0.0, value=0.12, step=0.01, format="%.2f")
    gramas_base = col3.number_input("Gramas base (proxy)", min_value=10, value=60, step=5)
    taxa_falha = col4.number_input("Taxa falha/refugo", min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.2f")
    taxa_market = col5.number_input("Taxa marketplace", min_value=0.0, max_value=0.5, value=0.14, step=0.01, format="%.2f")
    embalagem = col6.number_input("Embalagem (R$)", min_value=0.0, value=4.0, step=0.5)

    sim_df = compute_profit(
        df_filtered,
        custo_hora=custo_hora,
        custo_grama=custo_grama,
        gramas_base=gramas_base,
        taxa_falha=taxa_falha,
        taxa_marketplace=taxa_market,
        embalagem=embalagem,
    )

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

# 11) Recomendador
with tabs[10]:
    st.header("🧭 Recomendador (o que listar / produzir)")
    st.caption("Combina: Gap Finder + preço esperado (se ativo) + flash + penaliza anomalia.")

    base = df_filtered.copy()
    has_price = ("Delta_Preco" in base.columns) and base["Delta_Preco"].notna().any()

    cluster_score_map = {}
    if not gap_df.empty:
        cluster_score_map = dict(zip(gap_df["CLUSTER_MKT"].astype(int), gap_df["score_base"].astype(float)))

    base["cluster_score"] = base["CLUSTER_MKT"].astype(int).map(cluster_score_map).fillna(0.0)
    base["flash_flag"] = (base["Logistica"] == "⚡ FLASH").astype(int)
    base["anom_penalty"] = base.get("is_anomaly", 0).astype(int)

    if has_price:
        base["under_score"] = (-base["Delta_Preco"].astype(float)).clip(lower=0)
        base["under_score"] = base["under_score"] / (base["under_score"].max() + 1e-9)
    else:
        base["under_score"] = 0.0

    base["score_rec"] = (
        base["cluster_score"] * 0.55
        + base["under_score"] * 0.25
        + base["flash_flag"] * 0.15
        - base["anom_penalty"] * 0.10
    )

    rec = base.sort_values("score_rec", ascending=False).head(50).copy()
    view_cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preco_Num", "Logistica", "LINK"]
    if has_price:
        view_cols += ["Preco_Previsto", "Faixa_Min", "Faixa_Max", "Delta_Preco"]

    view = rec[view_cols].copy()
    view["Preço"] = view["Preco_Num"].apply(format_brl)
    if has_price:
        view["Esperado"] = view["Preco_Previsto"].apply(format_brl)
        view["Faixa"] = view.apply(lambda r: f"{format_brl(r['Faixa_Min'])} ~ {format_brl(r['Faixa_Max'])}", axis=1)
        view["Delta"] = view["Delta_Preco"].apply(lambda x: format_brl(x))

    show_cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preço", "Logistica", "LINK"]
    if has_price:
        show_cols += ["Esperado", "Faixa", "Delta"]

    st.dataframe(view[show_cols], hide_index=True, use_container_width=True)
