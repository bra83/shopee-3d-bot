# -*- coding: utf-8 -*-
# dashboard.py
# BCRUZ 3D Enterprise â€” Decision Intelligence Edition (FDM)
# Regra do projeto: NÃƒO remover funcionalidades existentes. Este arquivo mantÃ©m o que jÃ¡ existia
# (carregamento Shopee/Elo7, grÃ¡ficos, comparador, nuvens, ML, simulador, BCG, Otimizador, downloads, Gemini).

import time
import re
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from thefuzz import process, fuzz

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor

# SciPy (Otimizador) â€” se nÃ£o existir, a aba funciona com aviso.
try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    linprog = None
    _HAS_SCIPY = False

# ==============================
# ConfiguraÃ§Ã£o da pÃ¡gina
# ==============================
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="ðŸ­")

# ==============================
# Session State (memÃ³ria de custos)
# ==============================
if "cost_config" not in st.session_state:
    st.session_state["cost_config"] = {
        "custo_hora": 8.0,
        "custo_grama": 0.12,
        "taxa_mkt": 0.14,
        "taxa_falha": 0.06,
        "embalagem": 4.0,
    }

# ==============================
# Cores por fonte
# ==============================
COLOR_MAP = {
    "Shopee": "#ff7a00",  # laranja
    "Elo7": "#1db954",    # verde
}

# ==============================
# Gemini (hook seguro)
# ==============================
def init_gemini():
    try:
        from google import genai  # precisa de google-genai no requirements
        key = None
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
        if not key:
            return None

        client = genai.Client(api_key=key)
        return client
    except Exception:
        return None

def gemini_explain(client, prompt: str):
    if client is None:
        return "IA indisponÃ­vel (configure GEMINI_API_KEY nos Secrets)."
    try:
        model = None
        try:
            model = st.secrets.get("GEMINI_MODEL")
        except Exception:
            model = None
        if not model:
            # modelo default: o usuÃ¡rio pode sobrescrever via Secrets
            model = "gemini-1.5-flash"

        resp = client.models.generate_content(model=model, contents=prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Erro ao consultar IA: {e}"

GEMINI_CLIENT = init_gemini()

# ==============================
# Links de dados
# ==============================
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# ==============================
# Robustez: retry de CSV
# ==============================
def load_csv_retry(url: str, tries: int = 3, sleep_s: float = 1.0, **kwargs) -> pd.DataFrame:
    last_err = None
    for _ in range(tries):
        try:
            return pd.read_csv(url, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise last_err  # type: ignore

# ==============================
# Limpeza de preÃ§o
# ==============================
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0

    if isinstance(valor, (int, float)):
        v = float(valor)
    else:
        t = str(valor).upper().strip()
        t = re.sub(r"[^\d,\.]", "", t)
        try:
            if "," in t:
                t = t.replace(".", "").replace(",", ".")
            elif t.count(".") == 1:
                partes = t.split(".")
                if len(partes[1]) == 3:
                    t = t.replace(".", "")
            v = float(t)
        except Exception:
            return 0.0

    if v > 1500.0:
        return 0.0
    return v

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

# Stopwords (clusters)
CLUSTER_STOPWORDS = {
    "de","da","do","das","dos","com","para","por","em","no","na","nos","nas","e","a","o","os","as",
    "um","uma","uns","umas","kit","un","und","unid","cm","mm","3d","impresso","impressao","impressÃ£o","pla",
    "frete","gratis","grÃ¡tis","pronta","pronto","entrega","envio"
}

def extract_features_from_title(title: str) -> dict:
    t = normalize_text(title)
    is_kit = int(bool(re.search(r"\bkit\b|\bconjunto\b|\bcombo\b", t)))
    is_personalizado = int(bool(re.search(r"\bpersonaliz", t)))
    is_pronta = int(bool(re.search(r"\bpronta\b|\bpronto\b|\bimediat", t)))
    is_decor = int(bool(re.search(r"\bdecor\b|\bdecora", t)))
    is_org = int(bool(re.search(r"\borganiz", t)))
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
        is_pronta_entrega=is_pronta,
        is_decor=is_decor,
        is_organizador=is_org,
        is_suporte=is_suporte,
        is_vaso=is_vaso,
        is_action=is_action,
        is_gamer=is_gamer,
        size_num=size_num,
        premium=premium,
        title_len=len(t),
        word_count=len(t.split()),
    )

# ==============================
# Carregamento de dados (com retry)
# ==============================
@st.cache_data(ttl=120)
def carregar_dados():
    dfs = []
    per_source = []

    for url, nome in [(URL_ELO7, "Elo7"), (URL_SHOPEE, "Shopee")]:
        raw = 0
        valid = 0
        try:
            df = load_csv_retry(url, tries=3, sleep_s=1.0, on_bad_lines="skip", dtype=str)
            df.columns = [str(c).strip().upper() for c in df.columns]
            raw = int(len(df))
            if df.empty:
                per_source.append({"fonte": nome, "raw": raw, "validos": 0})
                continue

            col_prod = next((c for c in df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), None)
            col_preco = next((c for c in df.columns if any(x in c for x in ["(R$)", "PREÃ‡O", "PRECO", "PRICE"])), None)
            col_cat = next((c for c in df.columns if "CATEG" in c), None)
            col_link = next((c for c in df.columns if "LINK" in c or "URL" in c), None)
            col_prazo = next((c for c in df.columns if "PRAZO" in c or "FLASH" in c), None)

            if not col_prod:
                per_source.append({"fonte": nome, "raw": raw, "validos": 0})
                continue

            df = df.rename(columns={col_prod: "PRODUTO"})
            df["FONTE"] = nome
            df["CATEGORIA"] = df[col_cat] if col_cat and col_cat in df.columns else "Geral"
            df["LINK"] = df[col_link] if col_link and col_link in df.columns else "#"

            if col_preco and col_preco in df.columns:
                df["Preco_Num"] = df[col_preco].apply(limpar_preco)
            else:
                df["Preco_Num"] = 0.0

            # Prazo / logÃ­stica
            if col_prazo and col_prazo in df.columns:
                df["Prazo_Txt"] = df[col_prazo].fillna("Normal")

                def get_days(t):
                    t = str(t).upper()
                    if "IMEDIATO" in t or "PRONTA" in t:
                        return 1
                    m = re.search(r"(\d+)", t)
                    return int(m.group(1)) if m else 15

                df["Dias_Producao"] = df["Prazo_Txt"].apply(get_days)
            else:
                df["Dias_Producao"] = 15

            df["Logistica"] = df["Dias_Producao"].apply(lambda x: "FLASH" if int(x) <= 2 else "NORMAL")

            df = df[df["Preco_Num"] > 0.1].copy()
            valid = int(len(df))

            cols = ["PRODUTO", "Preco_Num", "FONTE", "CATEGORIA", "LINK", "Logistica", "Dias_Producao"]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            dfs.append(df[cols])
            per_source.append({"fonte": nome, "raw": raw, "validos": valid})
        except Exception:
            per_source.append({"fonte": nome, "raw": raw, "validos": valid})
            continue

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not final_df.empty:
        corte_superior = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= corte_superior].copy()

    final_df.attrs["per_source"] = per_source
    final_df.attrs["valid_total"] = int(len(final_df))
    return final_df

# ==============================
# Enriquecimento + anomalias
# ==============================
@st.cache_data(ttl=300)
def enrich_df(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return base_df
    d = base_df.copy()
    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)
    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # Anomalias simples pelo preÃ§o
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

# ==============================
# Vetores de texto (TF-IDF + SVD)
# ==============================
@st.cache_data(ttl=300)
def compute_text_vectors(texts: pd.Series, max_features: int = 4000):
    texts = texts.fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_t = tfidf.fit_transform(texts)
    n_comp = int(min(128, max(8, X_t.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_t)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, tfidf

# ==============================
# Clusters de mercado + nomes limpos
# ==============================
@st.cache_data(ttl=300)
def market_clusters(d: pd.DataFrame, n_clusters: int = 18):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, tfidf = compute_text_vectors(out["PRODUTO_NORM"])
    n = len(out)
    k = int(np.clip(n_clusters, 6, min(40, max(6, int(np.sqrt(n) * 1.2)))))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        out["CLUSTER_MKT"] = km.fit_predict(X)
    except Exception:
        out["CLUSTER_MKT"] = 0

    # nome do cluster por termos mais fortes no centroid (aprox pela mÃ©dia TF-IDF)
    try:
        X_t = tfidf.transform(out["PRODUTO_NORM"].fillna("").astype(str))
        vocab = np.array(tfidf.get_feature_names_out())

        cluster_names = {}
        for cid in sorted(out["CLUSTER_MKT"].unique()):
            idx = np.where(out["CLUSTER_MKT"].values == cid)[0]
            if len(idx) == 0:
                cluster_names[cid] = f"Cluster {cid}"
                continue
            mean_vec = X_t[idx].mean(axis=0)
            mean_vec = np.asarray(mean_vec).ravel()
            top_idx = mean_vec.argsort()[-8:][::-1]
            top_terms = []
            for i in top_idx:
                term = str(vocab[i])
                tnorm = normalize_text(term)
                if not tnorm or tnorm in CLUSTER_STOPWORDS:
                    continue
                if len(tnorm) <= 2:
                    continue
                top_terms.append(term)
            # evita nomes vazios
            cluster_names[cid] = " / ".join(top_terms[:4]) if top_terms else f"Cluster {cid}"

        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].map(cluster_names).fillna(out["CLUSTER_MKT"].astype(str))
    except Exception:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    return out

# ==============================
# Modelo de preÃ§o (global)
# ==============================
class ToDenseTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        try:
            return X.toarray()
        except Exception:
            return X

@st.cache_data(ttl=300)
def train_price_model(d: pd.DataFrame, min_samples: int = 80):
    if d is None or d.empty or len(d) < min_samples:
        return None, None

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

    pipe = Pipeline(steps=[("prep", preproc), ("todense", ToDenseTransformer()), ("model", model)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
    except Exception:
        return None, None

    metrics = {"MAE": mae, "R2": r2, "TRAIN_ROWS": int(len(data)), "MIN_SAMPLES": int(min_samples)}
    return pipe, metrics

def apply_price_model(model_pipe, d: pd.DataFrame, mae_global: float | None = None) -> pd.DataFrame:
    if model_pipe is None or d is None or d.empty:
        return d

    out = d.copy()
    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    for c in cat_cols:
        if c not in out.columns:
            out[c] = "NA"
    for c in num_cols:
        if c not in out.columns:
            out[c] = 0
    if text_col not in out.columns:
        out[text_col] = out["PRODUTO"].astype(str).apply(normalize_text)

    X = out[[text_col] + cat_cols + num_cols]
    try:
        out["Preco_Previsto"] = model_pipe.predict(X)
        out["Delta_Preco"] = out["Preco_Num"] - out["Preco_Previsto"]
        if mae_global is not None:
            out["Faixa_Min"] = np.maximum(0.0, out["Preco_Previsto"] - mae_global)
            out["Faixa_Max"] = out["Preco_Previsto"] + mae_global
    except Exception:
        return d
    return out

# ==============================
# Simulador (Lucro/Hora) â€” FDM proxy
# ==============================
def estimate_print_hours(row, base_hours=2.0):
    days = float(row.get("Dias_Producao", 15))
    size = float(row.get("size_num", 0))
    logist = str(row.get("Logistica", "NORMAL"))

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
    size = out.get("size_num", pd.Series([0]*len(out))).fillna(0).astype(float)
    kit = out.get("is_kit", pd.Series([0]*len(out))).fillna(0).astype(int)
    prem = out.get("premium", pd.Series([0]*len(out))).fillna(0).astype(int)

    out["Gramagem_Estimada"] = (gramas_base + (size * 2.2) + (kit * 40) + (prem * 25)).clip(lower=20, upper=600)
    out["Horas_Estimadas"] = out.apply(lambda r: estimate_print_hours(r, base_hours=2.0), axis=1)

    out["Custo_Material"] = out["Gramagem_Estimada"] * float(custo_grama)
    out["Custo_Maquina"] = out["Horas_Estimadas"] * float(custo_hora)
    out["Custo_Embalagem"] = float(embalagem)
    out["Fee_Marketplace"] = out["Preco_Num"] * float(taxa_marketplace)

    out["Custo_Total_Estimado"] = (out["Custo_Material"] + out["Custo_Maquina"] + out["Custo_Embalagem"] + out["Fee_Marketplace"]) * (1.0 + float(taxa_falha))
    out["Lucro_Estimado"] = out["Preco_Num"] - out["Custo_Total_Estimado"]
    out["Margem_%"] = (out["Lucro_Estimado"] / (out["Preco_Num"] + 1e-9)) * 100.0
    out["Lucro_por_Hora"] = out["Lucro_Estimado"] / (out["Horas_Estimadas"] + 1e-9)
    return out

# ==============================
# Otimizador de fÃ¡brica (linprog)
# ==============================
def run_factory_optimizer(df: pd.DataFrame, hours_avail: float):
    if df is None or df.empty:
        return pd.DataFrame()
    if "Lucro_Estimado" not in df.columns or "Horas_Estimadas" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("CLUSTER_NOME", dropna=False).agg(
        lucro_medio=("Lucro_Estimado", "mean"),
        horas_media=("Horas_Estimadas", "mean"),
        itens=("PRODUTO", "count"),
    ).reset_index()

    g = g[(g["lucro_medio"].notna()) & (g["horas_media"].notna()) & (g["horas_media"] > 0)].copy()
    if g.empty:
        return pd.DataFrame()

    # Max lucro => min (-lucro)
    c = (-g["lucro_medio"].astype(float).values)

    # RestriÃ§Ã£o: sum(q_i * horas_media_i) <= hours_avail
    A = np.array([g["horas_media"].astype(float).values])
    b = np.array([float(hours_avail)])

    bounds = [(0, None) for _ in range(len(g))]  # quantidade >=0

    if not _HAS_SCIPY:
        g["qtd_sugerida"] = 0.0
        g["obs"] = "SciPy nÃ£o instalado (adicione 'scipy' no requirements.txt)."
        return g.sort_values("lucro_medio", ascending=False)

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not getattr(res, "success", False):
        g["qtd_sugerida"] = 0.0
        g["obs"] = f"Falha na otimizaÃ§Ã£o: {getattr(res, 'message', 'erro')}"
        return g.sort_values("lucro_medio", ascending=False)

    q = np.array(res.x)
    g["qtd_sugerida"] = q
    g["horas_total"] = g["qtd_sugerida"] * g["horas_media"]
    g["lucro_total"] = g["qtd_sugerida"] * g["lucro_medio"]
    g["obs"] = ""
    return g.sort_values("lucro_total", ascending=False)

# ==============================
# Downloads
# ==============================
def download_csv_button(df: pd.DataFrame, label: str, filename: str, key: str):
    if df is None or df.empty:
        st.info("Sem dados para exportar.")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", key=key)

# ==============================
# APP
# ==============================
df = carregar_dados()
df = enrich_df(df)
df = market_clusters(df)

# Modelo de preÃ§o (treina no total pÃ³s-limpeza)
price_model, price_metrics = train_price_model(df, min_samples=80)

# Sidebar
st.sidebar.title("Centro de Comando")

# SaÃºde
valid_total = int(df.attrs.get("valid_total", len(df))) if df is not None else 0
st.sidebar.caption(f"Itens vÃ¡lidos (total): {valid_total}")
per_source = df.attrs.get("per_source", [])
if per_source:
    for r in per_source:
        try:
            st.sidebar.caption(f"{r['fonte']}: raw {int(r['raw'])} â†’ vÃ¡lidos {int(r['validos'])}")
        except Exception:
            pass

st.sidebar.markdown("---")
st.sidebar.subheader("Filtro")
if df is None or df.empty:
    st.error("Erro ao carregar dados. Verifique seus Google Sheets.")
    st.stop()

max_val = float(df["Preco_Num"].max())
preco_max = st.sidebar.slider("Teto de preÃ§o (R$)", 0.0, max_val, min(500.0, max_val))

fontes_sel = st.sidebar.multiselect("Fontes", df["FONTE"].unique().tolist(), default=df["FONTE"].unique().tolist())

df_filtered = df[(df["FONTE"].isin(fontes_sel)) & (df["Preco_Num"] <= preco_max)].copy()

cats = st.sidebar.multiselect("Categorias", sorted(df_filtered["CATEGORIA"].unique().tolist()))
if cats:
    df_filtered = df_filtered[df_filtered["CATEGORIA"].isin(cats)].copy()

st.sidebar.caption(f"Itens vÃ¡lidos (filtro): {len(df_filtered)}")

# Custos (session_state)
st.sidebar.markdown("---")
st.sidebar.subheader("Custos (persistem entre abas)")
cc = st.session_state["cost_config"]
cc["custo_hora"] = st.sidebar.number_input("Custo/hora mÃ¡quina (R$)", min_value=0.0, value=float(cc["custo_hora"]), step=0.5)
cc["custo_grama"] = st.sidebar.number_input("Custo/grama (R$)", min_value=0.0, value=float(cc["custo_grama"]), step=0.01, format="%.2f")
cc["taxa_mkt"] = st.sidebar.number_input("Taxa marketplace", min_value=0.0, max_value=0.5, value=float(cc["taxa_mkt"]), step=0.01, format="%.2f")
cc["taxa_falha"] = st.sidebar.number_input("Taxa falha/refugo", min_value=0.0, max_value=0.5, value=float(cc["taxa_falha"]), step=0.01, format="%.2f")
cc["embalagem"] = st.sidebar.number_input("Embalagem (R$)", min_value=0.0, value=float(cc["embalagem"]), step=0.5)

# Tabs
tabs = st.tabs([
    "VisÃ£o Geral",
    "Comparador",
    "IA & Insights",
    "Mercado & Clusters",
    "PrecificaÃ§Ã£o ML",
    "Simulador",
    "Otimizador",
    "Matriz BCG",
    "Dados",
])

# ---------------------------
# VisÃ£o Geral (inclui as nuvens)
# ---------------------------
with tabs[0]:
    st.header("VisÃ£o Geral")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Produtos", int(len(df_filtered)))
    c2.metric("Ticket mÃ©dio", format_brl(df_filtered["Preco_Num"].mean() if len(df_filtered) else 0))
    c3.metric("Fontes", int(df_filtered["FONTE"].nunique()))
    c4.metric("Flash", int((df_filtered["Logistica"] == "FLASH").sum()))

    st.markdown("---")
    colA, colB = st.columns(2)

    # Boxplot por fonte (cores fixas)
    with colA:
        fig = px.box(
            df_filtered,
            x="FONTE",
            y="Preco_Num",
            color="FONTE",
            color_discrete_map=COLOR_MAP,
            title="DistribuiÃ§Ã£o de preÃ§os por fonte"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.pie(df_filtered, names="CATEGORIA", title="Share de categorias")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Nuvens de palavras (no filtro atual)")

    excluir = st.text_input("Palavras para excluir (separe por vÃ­rgula)", value="", key="wc_excluir_overview")
    extras = set([normalize_text(x).strip() for x in excluir.split(",") if normalize_text(x).strip()])

    sw_local = set(STOPWORDS)
    sw_local.update(list(CLUSTER_STOPWORDS))
    sw_local.update(extras)

    c_cloud1, c_cloud2 = st.columns(2)

    with c_cloud1:
        st.caption("Mais frequentes (o que mais aparece nos tÃ­tulos)")
        texto_geral = " ".join(df_filtered["PRODUTO"].astype(str).tolist())
        if texto_geral.strip():
            wc1 = WordCloud(width=450, height=320, background_color="white", stopwords=sw_local).generate(texto_geral)
            fig1, ax1 = plt.subplots()
            ax1.imshow(wc1)
            ax1.axis("off")
            st.pyplot(fig1)
        else:
            st.info("Sem dados suficientes para gerar a nuvem.")

    with c_cloud2:
        st.caption("Maior valor agregado (palavras associadas a preÃ§o alto)")
        word_prices = {}
        for _, row in df_filtered.iterrows():
            palavras = normalize_text(row["PRODUTO"]).split()
            for p in palavras:
                if p and p not in sw_local and len(p) > 3:
                    word_prices.setdefault(p, []).append(float(row["Preco_Num"]))
        if word_prices:
            avg_prices = {k: sum(v)/len(v) for k, v in word_prices.items() if len(v) > 1}
            if avg_prices:
                wc2 = WordCloud(width=450, height=320, background_color="white").generate_from_frequencies(avg_prices)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc2)
                ax2.axis("off")
                st.pyplot(fig2)
            else:
                st.info("Dados insuficientes para nuvem por valor (termos pouco repetidos).")
        else:
            st.info("Sem dados para nuvem por valor.")

# ---------------------------
# Comparador
# ---------------------------
with tabs[1]:
    st.header("Comparador de preÃ§os")
    col_input, col_check = st.columns([3, 1])
    with col_input:
        termo = st.text_input("Filtrar produto:", placeholder="Ex: Vaso Robert")
    with col_check:
        st.write("")
        mostrar_tudo = st.checkbox("Ver todos", value=False)

    df_comp = pd.DataFrame()
    if mostrar_tudo:
        df_comp = df_filtered
    elif termo:
        prods = df_filtered["PRODUTO"].unique()
        matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
        similares = [x[0] for x in matches if x[1] > 40]
        df_comp = df_filtered[df_filtered["PRODUTO"].isin(similares)]

    if df_comp.empty:
        st.info("Busque um produto acima ou marque 'Ver todos'.")
    else:
        fig = px.scatter(
            df_comp,
            x="FONTE",
            y="Preco_Num",
            color="FONTE",
            color_discrete_map=COLOR_MAP,
            size="Preco_Num",
            hover_data=["PRODUTO", "CATEGORIA", "Logistica"],
            title="Comparativo de preÃ§os"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "CATEGORIA", "Logistica", "LINK"]], use_container_width=True, hide_index=True)

# ---------------------------
# IA & Insights (linha selecionada)
# ---------------------------
with tabs[2]:
    st.header("IA & Insights")

    st.caption("Selecione uma linha no filtro e peÃ§a para a IA explicar (Gemini).")
    st.info("Dica: se der 404, defina GEMINI_MODEL nos Secrets e reinicie o app.")

    if df_filtered.empty:
        st.info("Sem dados no filtro.")
    else:
        # busca rÃ¡pida na tabela
        q = st.text_input("Buscar produto (texto contÃ©m)", value="", key="busca_ia")
        view = df_filtered.copy()
        if q.strip():
            qn = normalize_text(q)
            view = view[view["PRODUTO_NORM"].str.contains(qn, na=False)]

        st.dataframe(view[["FONTE","PRODUTO","Preco_Num","CATEGORIA","Logistica","CLUSTER_NOME","LINK"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        produto = st.selectbox("Escolha um produto para explicar:", view["PRODUTO"].astype(str).tolist()[:2000])
        if produto:
            row = view[view["PRODUTO"] == produto].head(1)
            if not row.empty:
                r = row.iloc[0].to_dict()
                prompt = (
                    "VocÃª Ã© um analista de mercado de impressÃ£o 3D FDM no Brasil. "
                    "Explique em portuguÃªs, de forma objetiva, o que este anÃºncio sugere sobre demanda e preÃ§o. "
                    "DÃª 3 insights e 2 aÃ§Ãµes prÃ¡ticas.\n\n"
                    f"Fonte: {r.get('FONTE')}\n"
                    f"Produto: {r.get('PRODUTO')}\n"
                    f"PreÃ§o: {r.get('Preco_Num')}\n"
                    f"Categoria: {r.get('CATEGORIA')}\n"
                    f"LogÃ­stica: {r.get('Logistica')}\n"
                    f"Cluster: {r.get('CLUSTER_NOME')}\n"
                )
                if st.button("Analisar com IA", type="primary"):
                    st.write(gemini_explain(GEMINI_CLIENT, prompt))

# ---------------------------
# Mercado & Clusters
# ---------------------------
with tabs[3]:
    st.header("Mercado & Clusters")

    if df_filtered.empty:
        st.info("Sem dados no filtro.")
    else:
        cl = df_filtered.groupby("CLUSTER_NOME", dropna=False).agg(
            itens=("PRODUTO", "count"),
            ticket=("Preco_Num", "mean"),
            mediana=("Preco_Num", "median"),
            flash_share=("Logistica", lambda s: float((s == "FLASH").mean())),
            fontes=("FONTE", lambda s: int(pd.Series(s).nunique())),
        ).reset_index()

        cl["ticket_fmt"] = cl["ticket"].apply(format_brl)
        cl["mediana_fmt"] = cl["mediana"].apply(format_brl)
        cl["flash_%"] = (cl["flash_share"] * 100).round(1)

        st.dataframe(cl.sort_values("ticket", ascending=False)[["CLUSTER_NOME","itens","ticket_fmt","mediana_fmt","flash_%","fontes"]], use_container_width=True, hide_index=True)

        fig = px.scatter(
            cl,
            x="itens",
            y="ticket",
            size="itens",
            color="fontes",
            hover_data=["CLUSTER_NOME","flash_%"],
            title="Mapa do mercado: competiÃ§Ã£o (itens) vs ticket"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PrecificaÃ§Ã£o ML
# ---------------------------
with tabs[4]:
    st.header("PrecificaÃ§Ã£o ML")

    if price_model is None or price_metrics is None:
        st.warning("Modelo de preÃ§o nÃ£o estÃ¡ ativo (dados insuficientes para treinar).")
    else:
        st.caption(f"Treinado no total com {price_metrics['TRAIN_ROWS']} itens. MAE: {format_brl(price_metrics['MAE'])} | RÂ²: {price_metrics['R2']:.3f}")
        df_ml = apply_price_model(price_model, df_filtered, mae_global=float(price_metrics["MAE"]))

        if df_ml.empty or "Preco_Previsto" not in df_ml.columns:
            st.info("Sem dados no filtro para aplicar o modelo.")
        else:
            fig = px.scatter(
                df_ml,
                x="Preco_Previsto",
                y="Preco_Num",
                color="FONTE",
                color_discrete_map=COLOR_MAP,
                hover_data=["PRODUTO","CATEGORIA","CLUSTER_NOME"],
                title="Real vs Esperado (acima: caro; abaixo: barato)"
            )
            # linha y=x
            try:
                mn = float(min(df_ml["Preco_Previsto"].min(), df_ml["Preco_Num"].min()))
                mx = float(max(df_ml["Preco_Previsto"].max(), df_ml["Preco_Num"].max()))
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)

            under = df_ml.sort_values("Delta_Preco", ascending=True).head(20).copy()
            over = df_ml.sort_values("Delta_Preco", ascending=False).head(20).copy()

            under["real"] = under["Preco_Num"].apply(format_brl)
            under["esperado"] = under["Preco_Previsto"].apply(format_brl)
            over["real"] = over["Preco_Num"].apply(format_brl)
            over["esperado"] = over["Preco_Previsto"].apply(format_brl)

            colL, colR = st.columns(2)
            with colL:
                st.subheader("Abaixo do esperado")
                st.dataframe(under[["FONTE","PRODUTO","real","esperado","CLUSTER_NOME","LINK"]], use_container_width=True, hide_index=True)
            with colR:
                st.subheader("Acima do esperado")
                st.dataframe(over[["FONTE","PRODUTO","real","esperado","CLUSTER_NOME","LINK"]], use_container_width=True, hide_index=True)

            download_csv_button(df_ml, "Baixar CSV (PrecificaÃ§Ã£o)", "precificacao_ml.csv", key="dl_precificacao")

# ---------------------------
# Simulador
# ---------------------------
with tabs[5]:
    st.header("Simulador (Lucro / Hora) â€” FDM")

    sim = compute_profit(
        df_filtered,
        custo_hora=float(cc["custo_hora"]),
        custo_grama=float(cc["custo_grama"]),
        taxa_falha=float(cc["taxa_falha"]),
        taxa_marketplace=float(cc["taxa_mkt"]),
        embalagem=float(cc["embalagem"]),
    )

    if sim.empty:
        st.info("Sem dados no filtro.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lucro mÃ©dio", format_brl(sim["Lucro_Estimado"].mean()))
        c2.metric("Lucro/hora mÃ©dio", format_brl(sim["Lucro_por_Hora"].mean()))
        c3.metric("Top lucro/hora", format_brl(sim["Lucro_por_Hora"].max()))
        c4.metric("Itens com lucro negativo", int((sim["Lucro_Estimado"] < 0).sum()))

        st.markdown("---")
        top = sim.sort_values("Lucro_por_Hora", ascending=False).head(30).copy()
        top["PreÃ§o"] = top["Preco_Num"].apply(format_brl)
        top["Lucro"] = top["Lucro_Estimado"].apply(format_brl)
        top["Lucro/H"] = top["Lucro_por_Hora"].apply(format_brl)
        st.subheader("Top 30 por lucro/hora")
        st.dataframe(top[["FONTE","PRODUTO","PreÃ§o","Lucro","Lucro/H","Horas_Estimadas","Gramagem_Estimada","CLUSTER_NOME","LINK"]], use_container_width=True, hide_index=True)

        download_csv_button(sim, "Baixar CSV (Simulador)", "simulador.csv", key="dl_simulador")

# ---------------------------
# Otimizador
# ---------------------------
with tabs[6]:
    st.header("Otimizador (mix ideal de produÃ§Ã£o por horas semanais)")

    sim = compute_profit(
        df_filtered,
        custo_hora=float(cc["custo_hora"]),
        custo_grama=float(cc["custo_grama"]),
        taxa_falha=float(cc["taxa_falha"]),
        taxa_marketplace=float(cc["taxa_mkt"]),
        embalagem=float(cc["embalagem"]),
    )

    hours_avail = st.number_input("Horas disponÃ­veis na semana", min_value=1.0, value=20.0, step=1.0)
    if st.button("Rodar otimizador", type="primary"):
        opt = run_factory_optimizer(sim, float(hours_avail))
        if opt.empty:
            st.warning("NÃ£o foi possÃ­vel otimizar (dados insuficientes).")
        else:
            opt_view = opt.copy()
            opt_view["lucro_medio_fmt"] = opt_view["lucro_medio"].apply(format_brl)
            if "lucro_total" in opt_view.columns:
                opt_view["lucro_total_fmt"] = opt_view["lucro_total"].apply(format_brl)
            st.dataframe(opt_view, use_container_width=True, hide_index=True)
            download_csv_button(opt_view, "Baixar CSV (Otimizador)", "otimizador.csv", key="dl_otimizador")

    if not _HAS_SCIPY:
        st.info("SciPy nÃ£o encontrado. Para ativar a otimizaÃ§Ã£o, adicione 'scipy' no requirements.txt.")

# ---------------------------
# Matriz BCG
# ---------------------------
with tabs[7]:
    st.header("Matriz BCG (Volume x Margem)")

    sim = compute_profit(
        df_filtered,
        custo_hora=float(cc["custo_hora"]),
        custo_grama=float(cc["custo_grama"]),
        taxa_falha=float(cc["taxa_falha"]),
        taxa_marketplace=float(cc["taxa_mkt"]),
        embalagem=float(cc["embalagem"]),
    )

    if sim.empty:
        st.info("Sem dados no filtro.")
    else:
        bcg = sim.groupby("CLUSTER_NOME", dropna=False).agg(
            volume=("PRODUTO", "count"),
            margem=("Margem_%", "mean"),
            ticket=("Preco_Num", "mean"),
        ).reset_index()

        vol_med = float(bcg["volume"].median()) if len(bcg) else 0.0
        mar_med = float(bcg["margem"].median()) if len(bcg) else 0.0

        def classificar(row):
            vol = float(row["volume"])
            mar = float(row["margem"])
            if vol >= vol_med and mar >= mar_med:
                return "Estrela"
            if vol >= vol_med and mar < mar_med:
                return "Vaca Leiteira"
            if vol < vol_med and mar >= mar_med:
                return "Aposta"
            return "Abacaxi"

        bcg["tipo"] = bcg.apply(classificar, axis=1)

        fig = px.scatter(
            bcg,
            x="volume",
            y="margem",
            color="tipo",
            size="ticket",
            hover_data=["CLUSTER_NOME","ticket"],
            title="BCG: Volume x Margem (tamanho ~ ticket)"
        )
        fig.add_vline(x=vol_med)
        fig.add_hline(y=mar_med)
        st.plotly_chart(fig, use_container_width=True)

        bcg_view = bcg.copy()
        bcg_view["ticket_fmt"] = bcg_view["ticket"].apply(format_brl)
        st.dataframe(bcg_view[["CLUSTER_NOME","tipo","volume","margem","ticket_fmt"]], use_container_width=True, hide_index=True)

# ---------------------------
# Dados (busca)
# ---------------------------
with tabs[8]:
    st.header("Dados")
    busca = st.text_input("Buscar produto (na tabela)", value="", key="busca_dados")
    view = df_filtered.copy()
    if busca.strip():
        b = normalize_text(busca)
        view = view[view["PRODUTO_NORM"].str.contains(b, na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    download_csv_button(view, "Baixar CSV (Dados filtrados)", "dados_filtrados.csv", key="dl_dados")
