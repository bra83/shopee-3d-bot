# -*- coding: utf-8 -*-
# dashboard.py
# BCRUZ 3D Enterprise - Decision Intelligence Edition
# Full version: keeps your original dashboard and reintroduces all ML modules.
# Includes Gemini IA explanations (on-demand buttons) for selected rows and chart summaries.
#
# IMPORTANT requirements.txt additions (besides your existing):
#   google-genai
#   requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from thefuzz import process, fuzz

# --- ML ---
from sklearn.cluster import KMeans
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

# =====================
# CONFIG (ONLY ONCE)
# =====================
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="")

# =====================
# COLORS (Shopee orange, Elo7 green)
# =====================
COLOR_MAP = {
    "Shopee": "#ff7a00",
    "Elo7": "#1db954",
}

# =====================
# GEMINI (SAFE INIT + MODEL AUTODETECT)
# =====================
@st.cache_resource
def _init_gemini_client():
    try:
        from google import genai
        key = st.secrets.get("GEMINI_API_KEY")
        if not key:
            return None
        return genai.Client(api_key=key)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def _list_models_rest(api_key: str):
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("models", []) or []
    except Exception:
        return []

def _normalize_model_name(name: str) -> str:
    name = str(name or "")
    return name.replace("models/", "").strip()

@st.cache_data(ttl=3600)
def _pick_model_id():
    key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        return None

    override = st.secrets.get("GEMINI_MODEL") or st.secrets.get("GEMINI_MODEL_ID")
    if override:
        return str(override).strip()

    prefer = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    models = _list_models_rest(key)
    supported = []
    for m in models:
        n = _normalize_model_name(m.get("name"))
        methods = m.get("supportedGenerationMethods") or []
        ok = ("generateContent" in methods) or (not methods)
        if n and ok:
            supported.append(n)

    for p in prefer:
        if p in supported:
            return p
    if supported:
        return supported[0]

    # fallback
    return "gemini-pro"

_gemini_client = _init_gemini_client()

def gemini_explain(prompt: str) -> str:
    if _gemini_client is None:
        return "IA indisponivel. Verifique GEMINI_API_KEY e google-genai no requirements.txt."

    model_id = _pick_model_id()
    if not model_id:
        return "IA indisponivel. GEMINI_API_KEY nao encontrada."

    last_err = ""
    tried = []
    for mid in [model_id, f"models/{model_id}"]:
        try:
            tried.append(mid)
            resp = _gemini_client.models.generate_content(model=mid, contents=prompt)
            return getattr(resp, "text", "") or "IA respondeu vazio."
        except Exception as e:
            last_err = str(e)

    return "Erro ao consultar IA (model tried: " + ", ".join(tried) + "): " + last_err

# =====================
# LINKS
# =====================
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# =====================
# HELPERS
# =====================
def format_brl(v: float) -> str:
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_features_from_title(title: str) -> dict:
    t = normalize_text(title)

    def has(pat: str) -> int:
        return int(bool(re.search(pat, t)))

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

    return {
        "is_kit": has(r"\bkit\b|\bconjunto\b|\bcombo\b"),
        "is_personalizado": has(r"\bpersonaliz"),
        "is_pronta_entrega": has(r"\bpronta\b|\bpronto\b|\bimediat"),
        "is_decor": has(r"\bdecor\b|\bdecora"),
        "is_organizador": has(r"\borganiz"),
        "is_suporte": has(r"\bsuporte\b|\bstand\b|\bbase\b"),
        "is_vaso": has(r"\bvaso\b|\bplant"),
        "is_action": has(r"\bfigure\b|\baction\b|\bminiatura\b|\bstatue\b|\bestatua\b"),
        "is_gamer": has(r"\bgamer\b|\bplaystation\b|\bxbox\b|\bnintendo\b|\bpc\b"),
        "premium": has(r"\bpremium\b|\bdeluxe\b|\bmetal\b|\bvelvet\b|\babs\b|\bpetg\b|\bresina\b"),
        "size_num": int(size_num),
        "title_len": int(len(t)),
        "word_count": int(len(t.split())),
    }

# =====================
# PRICE CLEAN
# =====================
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

    return float(val)

# =====================
# LOAD DATA + STATS
# =====================
@st.cache_data(ttl=120)
def carregar_dados():
    dfs = []
    per_source = []
    raw_total = 0
    valid_total = 0

    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines="skip", dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            raw_n = int(len(temp_df))
            raw_total += raw_n

            if temp_df.empty:
                per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": 0})
                continue

            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "TÍTULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRECO", "PRICE", "R$"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), "GERAL")
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

            temp_df["Logistica"] = temp_df["Dias_Producao"].apply(lambda x: "FLASH" if int(x) <= 2 else "NORMAL")

            cols = ["PRODUTO", "Preco_Num", "FONTE", "CATEGORIA", "LINK", "Logistica", "Dias_Producao"]
            for c in cols:
                if c not in temp_df.columns:
                    temp_df[c] = ""

            temp_df = temp_df[temp_df["Preco_Num"] > 0.1].copy()
            valid_n = int(len(temp_df))
            valid_total += valid_n

            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": valid_n})
            dfs.append(temp_df[cols])

        except Exception:
            per_source.append({"fonte": f["nome"], "raw": 0, "validos": 0})

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not final_df.empty:
        cut = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= cut].copy()

    final_df.attrs["rows_raw_total"] = int(raw_total)
    final_df.attrs["rows_valid_total"] = int(valid_total)
    final_df.attrs["per_source"] = per_source
    return final_df

# =====================
# ENRICH
# =====================
@st.cache_data(ttl=600)
def enrich_df(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return base_df

    d = base_df.copy()
    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)

    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # anomalies using only price (robust)
    try:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        d["is_anomaly_iso"] = (iso.fit_predict(d[["Preco_Num"]]) == -1).astype(int)
    except Exception:
        d["is_anomaly_iso"] = 0

    try:
        nnb = int(min(35, max(5, len(d) // 20)))
        lof = LocalOutlierFactor(n_neighbors=nnb)
        d["is_anomaly_lof"] = (lof.fit_predict(d[["Preco_Num"]]) == -1).astype(int)
    except Exception:
        d["is_anomaly_lof"] = 0

    d["is_anomaly"] = ((d["is_anomaly_iso"] + d["is_anomaly_lof"]) > 0).astype(int)
    return d

# =====================
# TEXT VECTORS (TFIDF + SVD)
# =====================
@st.cache_data(ttl=600)
def compute_text_vectors(texts: pd.Series, max_features: int = 4000):
    texts = texts.fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(texts)

    n_comp = int(min(128, max(8, X_tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, f"tfidf+svd({n_comp})"

# =====================
# DEDUP / CANONICAL
# =====================
@st.cache_data(ttl=600)
def canonicalize_products(d: pd.DataFrame, max_groups: int = 250):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"])

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

# =====================
# MARKET CLUSTERS + NAMES
# =====================
@st.cache_data(ttl=600)
def market_clusters(d: pd.DataFrame, n_clusters: int = 18):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"])

    n = len(out)
    k = int(np.clip(n_clusters, 6, min(40, max(6, int(np.sqrt(n) * 1.2)))))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        out["CLUSTER_MKT"] = km.fit_predict(X)
    except Exception:
        out["CLUSTER_MKT"] = 0

    # name clusters using tfidf terms
    try:
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_t = tfidf.fit_transform(out["PRODUTO_NORM"].fillna("").astype(str))
        vocab = np.array(tfidf.get_feature_names_out())

        cluster_names = {}
        for cid in sorted(out["CLUSTER_MKT"].unique()):
            idx = np.where(out["CLUSTER_MKT"].values == cid)[0]
            if len(idx) == 0:
                cluster_names[cid] = f"cluster_{cid}"
                continue
            mean_vec = np.asarray(X_t[idx].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[-4:][::-1]
            top_terms = [vocab[i] for i in top_idx if mean_vec[i] > 0]
            cluster_names[cid] = " / ".join(top_terms) if top_terms else f"cluster_{cid}"

        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].map(cluster_names).fillna(out["CLUSTER_MKT"].astype(str))
    except Exception:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    out.attrs["mkt_vectorizer"] = vec_name
    out.attrs["mkt_k"] = int(out["CLUSTER_MKT"].nunique())
    return out

# =====================
# PRICE MODEL
# =====================
@st.cache_data(ttl=900)
def train_price_model(d: pd.DataFrame, min_samples: int = 80):
    if d is None or d.empty or len(d) < min_samples:
        return None, None, d

    data = d.copy()
    y = data["Preco_Num"].astype(float)

    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    for c in cat_cols:
        if c not in data.columns:
            data[c] = "NA"
    for c in num_cols:
        if c not in data.columns:
            data[c] = 0
    if text_col not in data.columns:
        data[text_col] = data["PRODUTO"].astype(str).apply(normalize_text)

    X = data[[text_col] + cat_cols + num_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=2, max_df=0.95), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_depth=7,
        max_iter=450,
        random_state=42,
        l2_regularization=0.2,
    )

    pipe = Pipeline(steps=[("prep", preproc), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    # fit full predictions
    data["Preco_Previsto"] = pipe.predict(X)
    data["Delta_Preco"] = data["Preco_Num"] - data["Preco_Previsto"]
    data["Faixa_Min"] = np.maximum(0.0, data["Preco_Previsto"] - mae)
    data["Faixa_Max"] = data["Preco_Previsto"] + mae

    metrics = {"MAE": mae, "R2": r2, "TRAIN_ROWS": int(len(data)), "MIN_SAMPLES": int(min_samples)}
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
    for c in num_cols:
        if c not in data.columns:
            data[c] = 0
    if text_col not in data.columns:
        data[text_col] = data["PRODUTO"].astype(str).apply(normalize_text)

    X = data[[text_col] + cat_cols + num_cols]
    try:
        data["Preco_Previsto"] = model_pipe.predict(X)
    except Exception:
        return d
    return data

# =====================
# GAP FINDER
# =====================
@st.cache_data(ttl=600)
def gap_finder(d: pd.DataFrame):
    if d is None or d.empty:
        return pd.DataFrame()

    g = d.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
        itens=("PRODUTO", "count"),
        ticket=("Preco_Num", "mean"),
        mediana=("Preco_Num", "median"),
        flash_share=("Logistica", lambda s: float((pd.Series(s) == "FLASH").mean())),
        fonte_div=("FONTE", lambda s: int(pd.Series(s).nunique())),
    ).reset_index()

    g["score_base"] = (
        (g["ticket"] / (g["ticket"].median() + 1e-9)) * 0.55
        + (1.0 - (g["itens"] / (g["itens"].max() + 1e-9))) * 0.30
        + (g["flash_share"]) * 0.15
    )

    # examples
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

# =====================
# SIMULATOR (profit per hour proxy for FDM)
# =====================
def estimate_print_hours(row, base_hours=2.0):
    days = float(row.get("Dias_Producao", 15))
    size = float(row.get("size_num", 0))
    logist = str(row.get("Logistica", "NORMAL"))
    h = float(base_hours)
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

    size = out.get("size_num", pd.Series([0] * len(out))).fillna(0).astype(float)
    kit = out.get("is_kit", pd.Series([0] * len(out))).fillna(0).astype(int)
    prem = out.get("premium", pd.Series([0] * len(out))).fillna(0).astype(int)

    out["Gramagem_Estimada"] = (gramas_base + (size * 2.2) + (kit * 40) + (prem * 25)).clip(lower=20, upper=600)
    out["Horas_Estimadas"] = out.apply(lambda r: estimate_print_hours(r, base_hours=2.0), axis=1)

    out["Custo_Material"] = out["Gramagem_Estimada"] * float(custo_grama)
    out["Custo_Maquina"] = out["Horas_Estimadas"] * float(custo_hora)
    out["Custo_Embalagem"] = float(embalagem)
    out["Fee_Marketplace"] = out["Preco_Num"] * float(taxa_marketplace)

    out["Custo_Total_Estimado"] = (out["Custo_Material"] + out["Custo_Maquina"] + out["Custo_Embalagem"] + out["Fee_Marketplace"]) * (1.0 + float(taxa_falha))
    out["Lucro_Estimado"] = out["Preco_Num"] - out["Custo_Total_Estimado"]
    out["Lucro_por_Hora"] = out["Lucro_Estimado"] / (out["Horas_Estimadas"] + 1e-9)

    return out

# =====================
# LOAD + PIPELINE
# =====================
df = carregar_dados()
df_enriched = enrich_df(df)
df_enriched = canonicalize_products(df_enriched)
df_enriched = market_clusters(df_enriched)

# train on TOTAL (post-clean)
price_model, price_metrics, df_enriched = train_price_model(df_enriched, min_samples=80)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("Centro de Comando")

# Gemini status
st.sidebar.subheader("IA (Gemini)")
if st.sidebar.button("Testar IA"):
    st.sidebar.write(gemini_explain("Responda apenas OK"))

if _gemini_client is None or not st.secrets.get("GEMINI_API_KEY"):
    st.sidebar.caption("Status: IA indisponivel")
else:
    st.sidebar.caption("Status: IA ativa | modelo: " + str(_pick_model_id()))

st.sidebar.caption('Dica: se der 404, use GEMINI_MODEL="<id>" nos Secrets e reinicie.')

st.sidebar.markdown("---")

if df_enriched is None or df_enriched.empty:
    st.error("Erro ao carregar dados. Verifique o Google Sheets.")
    st.stop()

# Data health
st.sidebar.subheader("Saude dos dados")
st.sidebar.caption("Raw total: " + str(df.attrs.get("rows_raw_total", "-")))
st.sidebar.caption("Validos (pos limpeza): " + str(len(df_enriched)))
for r in df.attrs.get("per_source", []):
    try:
        st.sidebar.caption(f"{r['fonte']}: raw {int(r['raw'])} -> validos {int(r['validos'])}")
    except Exception:
        pass

st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("Filtros")
max_val = float(df_enriched["Preco_Num"].max())
preco_max = st.sidebar.slider("Teto de preco (R$)", 0.0, max_val, float(min(500.0, max_val)))

fontes_sel = st.sidebar.multiselect("Fontes", df_enriched["FONTE"].unique().tolist(), default=df_enriched["FONTE"].unique().tolist())
df_filtered = df_enriched[(df_enriched["FONTE"].isin(fontes_sel)) & (df_enriched["Preco_Num"] <= preco_max)].copy()

cats = st.sidebar.multiselect("Categorias", df_filtered["CATEGORIA"].unique().tolist())
if cats:
    df_filtered = df_filtered[df_filtered["CATEGORIA"].isin(cats)].copy()

st.sidebar.metric("Itens validos (filtro atual)", int(len(df_filtered)))

st.sidebar.markdown("---")
st.sidebar.subheader("ML")
n_clusters = st.sidebar.slider("Clusters mercado (aprox.)", 6, 40, 18)

# re-cluster on filter for responsiveness
df_filtered = market_clusters(df_filtered, n_clusters=n_clusters)

# apply price model on filter
df_filtered = apply_price_model(price_model, df_filtered)
if price_metrics is not None and "Preco_Previsto" in df_filtered.columns:
    mae_global = float(price_metrics["MAE"])
    df_filtered["Delta_Preco"] = df_filtered["Preco_Num"] - df_filtered["Preco_Previsto"]
    df_filtered["Faixa_Min"] = np.maximum(0.0, df_filtered["Preco_Previsto"] - mae_global)
    df_filtered["Faixa_Max"] = df_filtered["Preco_Previsto"] + mae_global

gap_df = gap_finder(df_filtered)

# =====================
# UI TABS (full)
# =====================
tabs = st.tabs([
    "Visao Geral",
    "Comparador",
    "IA e Insights",
    "Laboratorio",
    "Criador",
    "Mercado e Clusters",
    "Precificacao ML",
    "Alertas",
    "Simulador",
    "Recomendador",
    "Forecast",
    "Dados",
])

# =====================
# TAB: Visao Geral
# =====================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total produtos", int(len(df_filtered)))
    c2.metric("Ticket medio", format_brl(float(df_filtered["Preco_Num"].mean())))
    c3.metric("Fontes", int(df_filtered["FONTE"].nunique()))
    c4.metric("Itens FLASH", int((df_filtered["Logistica"] == "FLASH").sum()))

    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        fig_box = px.box(
            df_filtered, x="FONTE", y="Preco_Num", color="FONTE",
            color_discrete_map=COLOR_MAP, title="Distribuicao de precos (limpa)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        if st.button("IA: explicar grafico de distribuicao"):
            prompt = (
                "Analise a distribuicao de precos por fonte (Shopee vs Elo7). "
                "Resuma em bullets: concorrencia, teto de preco e estrategia de posicionamento para FDM."
            )
            st.write(gemini_explain(prompt))

    with colB:
        fig_pie = px.pie(df_filtered, names="CATEGORIA", title="Share de categorias")
        st.plotly_chart(fig_pie, use_container_width=True)

        if st.button("IA: explicar categorias"):
            top = df_filtered["CATEGORIA"].value_counts().head(10).to_dict()
            prompt = "Explique o mix de categorias e o que priorizar. Top categorias: " + str(top)
            st.write(gemini_explain(prompt))

# =====================
# TAB: Comparador
# =====================
with tabs[1]:
    st.subheader("Comparador de precos")
    col_input, col_check = st.columns([3, 1])
    with col_input:
        termo = st.text_input("Filtrar produto (fuzzy):", placeholder="Ex: pokebola, hueforge, boneco...")
    with col_check:
        mostrar_tudo = st.checkbox("Ver todos", value=False)

    df_comp = pd.DataFrame()
    if mostrar_tudo:
        df_comp = df_filtered
    elif termo:
        prods = df_filtered["PRODUTO"].dropna().astype(str).unique().tolist()
        matches = process.extract(termo, prods, limit=80, scorer=fuzz.token_set_ratio)
        similares = [x[0] for x in matches if x[1] > 40]
        df_comp = df_filtered[df_filtered["PRODUTO"].isin(similares)]

    if df_comp.empty:
        st.info("Busque um produto acima ou marque Ver todos.")
    else:
        fig = px.scatter(
            df_comp, x="FONTE", y="Preco_Num", color="FONTE",
            color_discrete_map=COLOR_MAP, size="Preco_Num",
            hover_data=["PRODUTO"], title="Comparativo de precos"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "CATEGORIA", "Logistica", "LINK"]], hide_index=True, use_container_width=True)

        if st.button("IA: explicar comparacao atual"):
            s = df_comp.groupby("FONTE")["Preco_Num"].describe().to_dict()
            prompt = "Explique esta comparacao de precos entre fontes. Estatisticas: " + str(s)
            st.write(gemini_explain(prompt))

# =====================
# TAB: IA e Insights (wordclouds + diagnostics + row explain)
# =====================
with tabs[2]:
    st.subheader("Nuvens e diagnostico")

    sw = set(STOPWORDS)
    sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "pecas"])

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Mais frequentes")
        txt = " ".join(df_filtered["PRODUTO"].astype(str))
        try:
            wc = WordCloud(width=500, height=350, background_color="white", stopwords=sw).generate(txt)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
        except Exception:
            st.warning("Sem dados para wordcloud.")

    with c2:
        st.caption("Maior valor agregado (proxy)")
        word_prices = {}
        for _, r in df_filtered.iterrows():
            words = str(r["PRODUTO"]).lower().split()
            for w in words:
                if w not in sw and len(w) > 3:
                    word_prices.setdefault(w, []).append(float(r["Preco_Num"]))
        if word_prices:
            avg_prices = {k: sum(v) / len(v) for k, v in word_prices.items() if len(v) > 2}
            if avg_prices:
                wc2 = WordCloud(width=500, height=350, background_color="black", max_words=60).generate_from_frequencies(avg_prices)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc2)
                ax2.axis("off")
                st.pyplot(fig2)
            else:
                st.info("Poucos dados para valor agregado.")
        else:
            st.info("Sem dados.")

    st.markdown("---")
    dA, dB, dC, dD = st.columns(4)
    dA.metric("Itens (filtro)", int(len(df_filtered)))
    dB.metric("Clusters (filtro)", int(df_filtered["CLUSTER_MKT"].nunique()) if "CLUSTER_MKT" in df_filtered.columns else 0)
    dC.metric("Anomalias", int(df_filtered["is_anomaly"].sum()) if "is_anomaly" in df_filtered.columns else 0)
    if price_metrics:
        dD.metric("Modelo preco MAE", format_brl(float(price_metrics["MAE"])))
        st.caption("Treinado no total: " + str(price_metrics["TRAIN_ROWS"]) + " itens | R2: " + f"{price_metrics['R2']:.3f}")
    else:
        dD.metric("Modelo preco", "sem treino")

    st.markdown("---")
    st.subheader("IA: analisar um produto do filtro")
    sel = st.selectbox("Escolha um produto", df_filtered["PRODUTO"].dropna().astype(str).unique().tolist())
    if st.button("IA: explicar produto selecionado"):
        row = df_filtered[df_filtered["PRODUTO"] == sel].iloc[0].to_dict()
        prompt = (
            "Voce e um analista de mercado para impressao 3D FDM. "
            "Explique este anuncio e o que eu devo fazer. "
            "Responda em bullets curtos.\n\n"
            f"Produto: {row.get('PRODUTO','')}\n"
            f"Fonte: {row.get('FONTE','')}\n"
            f"Preco: {row.get('Preco_Num',0)}\n"
            f"Categoria: {row.get('CATEGORIA','')}\n"
            f"Logistica: {row.get('Logistica','')}\n"
        )
        if "Preco_Previsto" in row:
            prompt += f"Preco esperado (modelo): {row.get('Preco_Previsto','')}\n"
        st.write(gemini_explain(prompt))

# =====================
# TAB: Laboratorio
# =====================
with tabs[3]:
    st.subheader("Laboratorio")
    c1, c2, c3 = st.columns(3)
    with c1:
        cx = st.selectbox("Eixo X", df_filtered.columns.tolist(), index=0)
    with c2:
        cy = st.selectbox("Eixo Y", ["Preco_Num", "Dias_Producao"], index=0)
    with c3:
        tp = st.selectbox("Tipo", ["Barras", "Dispersao", "Boxplot"], index=1)

    if tp == "Barras":
        st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)
    elif tp == "Dispersao":
        st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)
    else:
        st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)

    if st.button("IA: explicar grafico do laboratorio"):
        prompt = (
            "Explique este grafico para decisao de impressao 3D FDM. "
            f"Eixo X: {cx}. Eixo Y: {cy}. Tipo: {tp}. "
            "Diga o que eu devo observar e como agir."
        )
        st.write(gemini_explain(prompt))

# =====================
# TAB: Criador (SEO)
# =====================
with tabs[4]:
    st.subheader("Gerador de titulos SEO")
    keyword = st.text_input("Produto base:", value="Vaso")
    if keyword:
        df_c = df_enriched[df_enriched["PRODUTO"].astype(str).str.contains(keyword, case=False, na=False)]
        if not df_c.empty:
            txt = " ".join(df_c["PRODUTO"].astype(str))
            pals = [p for p in re.findall(r"\w+", txt.lower()) if p not in sw and len(p) > 2]
            top = [x[0].title() for x in Counter(pals).most_common(6)]
            st.success("Palavras chave: " + ", ".join(top))
            st.code(f"{keyword.title()} 3D " + " ".join(top[:2]) + " - Alta Qualidade")
        else:
            st.warning("Sem dados para essa palavra.")

# =====================
# TAB: Mercado e Clusters
# =====================
with tabs[5]:
    st.subheader("Mercado e clusters")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters", int(df_filtered["CLUSTER_MKT"].nunique()))
    c2.metric("Grupos (dedup)", int(df_filtered["GROUP_ID"].nunique()) if "GROUP_ID" in df_filtered.columns else 0)
    c3.metric("Vectorizacao", str(df_filtered.attrs.get("mkt_vectorizer", "tfidf+svd")))
    c4.metric("Itens", int(len(df_filtered)))

    cluster_table = df_filtered.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
        itens=("PRODUTO", "count"),
        ticket=("Preco_Num", "mean"),
        mediana=("Preco_Num", "median"),
        flash_share=("Logistica", lambda s: float((pd.Series(s) == "FLASH").mean())),
        fonte_div=("FONTE", lambda s: int(pd.Series(s).nunique())),
    ).reset_index().sort_values("ticket", ascending=False)

    cluster_table["ticket_fmt"] = cluster_table["ticket"].apply(format_brl)
    cluster_table["mediana_fmt"] = cluster_table["mediana"].apply(format_brl)
    cluster_table["flash_pct"] = (cluster_table["flash_share"] * 100).round(1)

    st.dataframe(cluster_table[["CLUSTER_MKT", "CLUSTER_NOME", "itens", "ticket_fmt", "mediana_fmt", "flash_pct", "fonte_div"]],
                 hide_index=True, use_container_width=True)

    fig = px.scatter(
        cluster_table, x="itens", y="ticket", size="itens",
        hover_data=["CLUSTER_NOME", "flash_pct", "fonte_div"],
        title="Mapa: ticket vs competicao (itens)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Gap finder")
    if gap_df is not None and not gap_df.empty:
        show_gap = gap_df.head(25).copy()
        show_gap["ticket_fmt"] = show_gap["ticket"].apply(format_brl)
        show_gap["flash_pct"] = (show_gap["flash_share"] * 100).round(1)
        st.dataframe(show_gap[["CLUSTER_MKT", "CLUSTER_NOME", "score_base", "itens", "ticket_fmt", "flash_pct", "EX1", "EX2", "EX3"]],
                     hide_index=True, use_container_width=True)

        pick_cluster = st.selectbox("IA: escolher cluster para explicar", cluster_table["CLUSTER_NOME"].unique().tolist())
        if st.button("IA: explicar cluster selecionado"):
            sub = df_filtered[df_filtered["CLUSTER_NOME"] == pick_cluster]
            stats = {
                "itens": int(len(sub)),
                "ticket_medio": float(sub["Preco_Num"].mean()) if len(sub) else 0.0,
                "flash_share": float((sub["Logistica"] == "FLASH").mean()) if len(sub) else 0.0,
                "fontes": int(sub["FONTE"].nunique()) if len(sub) else 0,
            }
            prompt = (
                "Explique este cluster para decisao de impressao 3D FDM. "
                "Diga se parece oportunidade ou guerra de preco, e como diferenciar.\n\n"
                f"Cluster: {pick_cluster}\n"
                f"Stats: {stats}\n"
                "Inclua: produto exemplo, preco alvo e estrategia (SEO, kit, personalizacao)."
            )
            st.write(gemini_explain(prompt))
    else:
        st.info("Sem dados suficientes para gap finder.")

# =====================
# TAB: Precificacao ML
# =====================
with tabs[6]:
    st.subheader("Precificacao ML")
    if price_model is None or price_metrics is None or "Preco_Previsto" not in df_filtered.columns:
        st.warning("Modelo de preco nao ativo. (precisa de dados suficientes no total).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE global", format_brl(float(price_metrics["MAE"])))
        c2.metric("R2 global", f"{price_metrics['R2']:.3f}")
        c3.metric("Treino (linhas)", int(price_metrics["TRAIN_ROWS"]))
        c4.metric("Filtro (linhas)", int(len(df_filtered)))

        fig = px.scatter(
            df_filtered, x="Preco_Previsto", y="Preco_Num", color="FONTE",
            color_discrete_map=COLOR_MAP,
            hover_data=["PRODUTO", "CATEGORIA", "Logistica"],
            title="Mapa: real vs esperado (acima: caro; abaixo: barato)"
        )
        try:
            mn = float(min(df_filtered["Preco_Previsto"].min(), df_filtered["Preco_Num"].min()))
            mx = float(max(df_filtered["Preco_Previsto"].max(), df_filtered["Preco_Num"].max()))
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

        if st.button("IA: explicar grafico real vs esperado"):
            s = {
                "mae": float(price_metrics["MAE"]),
                "r2": float(price_metrics["R2"]),
                "media_delta": float(df_filtered["Delta_Preco"].mean()) if "Delta_Preco" in df_filtered.columns else 0.0,
            }
            prompt = "Explique o grafico real vs esperado e como usar para precificar. Contexto: " + str(s)
            st.write(gemini_explain(prompt))

        st.markdown("---")
        colL, colR = st.columns(2)
        with colL:
            under = df_filtered.sort_values("Delta_Preco", ascending=True).head(20).copy()
            under["real"] = under["Preco_Num"].apply(format_brl)
            under["esperado"] = under["Preco_Previsto"].apply(format_brl)
            st.caption("Abaixo do esperado (possivel subprecificado)")
            st.dataframe(under[["FONTE", "PRODUTO", "real", "esperado", "LINK"]], hide_index=True, use_container_width=True)

        with colR:
            over = df_filtered.sort_values("Delta_Preco", ascending=False).head(20).copy()
            over["real"] = over["Preco_Num"].apply(format_brl)
            over["esperado"] = over["Preco_Previsto"].apply(format_brl)
            st.caption("Acima do esperado (possivel caro)")
            st.dataframe(over[["FONTE", "PRODUTO", "real", "esperado", "LINK"]], hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("MAE por cluster")
        mae_cluster = df_filtered.dropna(subset=["Preco_Previsto"]).copy()
        if not mae_cluster.empty:
            mae_cluster["abs_err"] = (mae_cluster["Preco_Num"] - mae_cluster["Preco_Previsto"]).abs()
            mtab = mae_cluster.groupby("CLUSTER_NOME").agg(
                itens=("PRODUTO", "count"),
                mae=("abs_err", "mean"),
                ticket=("Preco_Num", "mean")
            ).reset_index().sort_values("mae", ascending=False)
            mtab["mae_fmt"] = mtab["mae"].apply(format_brl)
            mtab["ticket_fmt"] = mtab["ticket"].apply(format_brl)
            st.dataframe(mtab[["CLUSTER_NOME", "itens", "ticket_fmt", "mae_fmt"]], hide_index=True, use_container_width=True)

            if st.button("IA: explicar MAE por cluster"):
                topbad = mtab.head(8).to_dict(orient="records")
                prompt = "Explique porque alguns clusters tem MAE alto e como melhorar o agrupamento/entrada. Top clusters: " + str(topbad)
                st.write(gemini_explain(prompt))
        else:
            st.info("Sem dados para MAE por cluster.")

# =====================
# TAB: Alertas
# =====================
with tabs[7]:
    st.subheader("Alertas e anomalias")
    anom = df_filtered[df_filtered.get("is_anomaly", 0) == 1].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalias", int(len(anom)))
    c2.metric("Anomalias %", f"{(len(anom)/max(1,len(df_filtered))*100):.1f}%")
    c3.metric("Maior preco (anom)", format_brl(float(anom["Preco_Num"].max())) if len(anom) else "0")

    if len(anom):
        st.dataframe(anom[["FONTE", "PRODUTO", "Preco_Num", "CATEGORIA", "Logistica", "LINK"]], hide_index=True, use_container_width=True)
        if st.button("IA: explicar anomalias"):
            examples = anom.sort_values("Preco_Num", ascending=False).head(10)[["PRODUTO", "Preco_Num", "FONTE"]].to_dict(orient="records")
            prompt = "Explique essas anomalias de preco (possiveis erros, bundles, itens fora do nicho). Exemplos: " + str(examples)
            st.write(gemini_explain(prompt))
    else:
        st.success("Sem anomalias no filtro atual.")

# =====================
# TAB: Simulador
# =====================
with tabs[8]:
    st.subheader("Simulador operacional (lucro por hora) - proxy FDM")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    custo_hora = col1.number_input("Custo/hora maquina (R$)", min_value=0.0, value=8.0, step=0.5)
    custo_grama = col2.number_input("Custo/grama filamento (R$)", min_value=0.0, value=0.12, step=0.01, format="%.2f")
    gramas_base = col3.number_input("Gramas base (proxy)", min_value=10, value=60, step=5)
    taxa_falha = col4.number_input("Taxa falha/refugo", min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.2f")
    taxa_market = col5.number_input("Taxa marketplace", min_value=0.0, max_value=0.5, value=0.14, step=0.01, format="%.2f")
    embalagem = col6.number_input("Embalagem (R$)", min_value=0.0, value=4.0, step=0.5)

    sim_df = compute_profit(df_filtered, custo_hora=custo_hora, custo_grama=custo_grama, gramas_base=gramas_base,
                            taxa_falha=taxa_falha, taxa_marketplace=taxa_market, embalagem=embalagem)

    if sim_df is None or sim_df.empty:
        st.info("Sem dados no filtro.")
    else:
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Lucro medio (estimado)", format_brl(float(sim_df["Lucro_Estimado"].mean())))
        cB.metric("Lucro/hora medio", format_brl(float(sim_df["Lucro_por_Hora"].mean())))
        cC.metric("Top lucro/hora", format_brl(float(sim_df["Lucro_por_Hora"].max())))
        cD.metric("Lucro negativo", int((sim_df["Lucro_Estimado"] < 0).sum()))

        top = sim_df.sort_values("Lucro_por_Hora", ascending=False).head(30).copy()
        top["Preco"] = top["Preco_Num"].apply(format_brl)
        top["Lucro"] = top["Lucro_Estimado"].apply(format_brl)
        top["LucroH"] = top["Lucro_por_Hora"].apply(format_brl)
        st.dataframe(top[["FONTE", "PRODUTO", "Preco", "Lucro", "LucroH", "Horas_Estimadas", "Gramagem_Estimada", "LINK"]],
                     hide_index=True, use_container_width=True)

        if st.button("IA: explicar ranking de lucro/hora"):
            examples = top.head(10)[["PRODUTO", "Preco_Num", "Lucro_por_Hora", "FONTE"]].to_dict(orient="records")
            prompt = "Explique o ranking de lucro/hora e como transformar isso em plano de producao. Exemplos: " + str(examples)
            st.write(gemini_explain(prompt))

# =====================
# TAB: Recomendador
# =====================
with tabs[9]:
    st.subheader("Recomendador (o que listar / produzir)")
    base = df_filtered.copy()
    has_price = ("Delta_Preco" in base.columns) and base["Delta_Preco"].notna().any()

    cluster_score_map = {}
    if gap_df is not None and not gap_df.empty:
        cluster_score_map = dict(zip(gap_df["CLUSTER_MKT"].astype(int), gap_df["score_base"].astype(float)))

    base["cluster_score"] = base["CLUSTER_MKT"].astype(int).map(cluster_score_map).fillna(0.0)
    base["flash_flag"] = (base["Logistica"] == "FLASH").astype(int)
    base["anom_penalty"] = base.get("is_anomaly", 0).astype(int)

    if has_price:
        base["under_score"] = (-base["Delta_Preco"]).clip(lower=0)
        base["under_score"] = base["under_score"] / (base["under_score"].max() + 1e-9)
    else:
        base["under_score"] = 0.0

    base["score_rec"] = (
        base["cluster_score"] * 0.55
        + base["under_score"] * 0.25
        + base["flash_flag"] * 0.15
        - base["anom_penalty"] * 0.10
    )

    rec = base.sort_values("score_rec", ascending=False).head(60).copy()
    view_cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preco_Num", "Logistica", "LINK"]
    if has_price:
        view_cols += ["Preco_Previsto", "Faixa_Min", "Faixa_Max", "Delta_Preco"]
    show = rec[view_cols].copy()
    show["Preco"] = show["Preco_Num"].apply(format_brl)
    st.dataframe(show, hide_index=True, use_container_width=True)

    if st.button("IA: explicar recomendador"):
        top = rec.head(12)[["PRODUTO", "FONTE", "Preco_Num", "CLUSTER_NOME", "score_rec"]].to_dict(orient="records")
        prompt = "Explique por que estes itens sao recomendados e o que eu devo fazer primeiro. Top: " + str(top)
        st.write(gemini_explain(prompt))

# =====================
# TAB: Forecast
# =====================
with tabs[10]:
    st.subheader("Forecast (se houver data no dataset)")
    date_col = None
    for c in df.columns.tolist():
        cu = str(c).upper()
        if any(k in cu for k in ["DATA", "DATE", "DIA", "HORA", "TIMESTAMP"]):
            date_col = c
            break

    if date_col is None:
        st.info("Nao encontrei coluna de data/hora no CSV. Se adicionar uma coluna Data/Hora na planilha, esta aba vira previsao real.")
    else:
        try:
            tmp = df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if tmp.empty:
                st.warning("Coluna de data existe, mas nao consegui parsear as datas.")
            else:
                tmp["dia"] = tmp[date_col].dt.date
                ts = tmp.groupby("dia")["Preco_Num"].mean().reset_index()
                ts["dia"] = pd.to_datetime(ts["dia"])
                st.plotly_chart(px.line(ts, x="dia", y="Preco_Num", title="Ticket medio diario (observado)"), use_container_width=True)

                ts = ts.sort_values("dia")
                ts["mm7"] = ts["Preco_Num"].rolling(7, min_periods=3).mean()
                coef = np.polyfit(np.arange(len(ts)), ts["Preco_Num"].values, deg=1)
                ts["trend"] = coef[0] * np.arange(len(ts)) + coef[1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["Preco_Num"], mode="lines+markers", name="observado"))
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["mm7"], mode="lines", name="mm7"))
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["trend"], mode="lines", name="tendencia"))
                fig.update_layout(title="Ticket: observado vs suavizacao vs tendencia")
                st.plotly_chart(fig, use_container_width=True)

                if st.button("IA: explicar forecast"):
                    prompt = "Explique a tendencia de ticket no tempo e como usar isso para decidir o que imprimir e quando."
                    st.write(gemini_explain(prompt))
        except Exception as e:
            st.warning("Nao consegui gerar forecast: " + str(e))

# =====================
# TAB: Dados (with search)
# =====================
with tabs[11]:
    st.subheader("Dados (filtro atual)")
    q = st.text_input("Buscar produto na tabela", placeholder="Digite parte do nome...")
    view = df_filtered.copy()
    if q:
        view = view[view["PRODUTO"].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)

    if st.button("IA: resumir tabela atual"):
        sample = view.head(40)[["PRODUTO", "Preco_Num", "FONTE", "CATEGORIA"]].to_dict(orient="records")
        prompt = "Resuma os dados do filtro atual em 8 bullets e diga 3 proximas acoes. Amostra: " + str(sample)
        st.write(gemini_explain(prompt))
