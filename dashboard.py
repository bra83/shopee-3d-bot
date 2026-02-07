# dashboard.py
# BCRUZ 3D Enterprise — Decision Intelligence Dashboard
# Clean build with Gemini integration hooks (no UI break, no encoding issues)

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

# =====================
# CONFIG (ONLY ONCE)
# =====================
st.set_page_config(
    page_title="BCRUZ 3D Enterprise",
    layout="wide",
    page_icon="🏢"
)

# =====================
# COLORS
# =====================
COLOR_MAP = {
    "Shopee": "#ff7a00",  # laranja
    "Elo7": "#1db954",    # verde
}

# =====================
# GEMINI (SAFE INIT)
# =====================
def init_gemini():
    try:
        from google import genai
        key = st.secrets.get("GEMINI_API_KEY")
        if not key:
            return None
        client = genai.Client(api_key=key)
        return client
    except Exception:
        return None

gemini_client = init_gemini()

def gemini_explain(prompt: str):
    if gemini_client is None:
        return "IA indisponível (verifique GEMINI_API_KEY)."
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"Erro ao consultar IA: {e}"

# =====================
# DATA LINKS
# =====================
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# =====================
# PRICE CLEAN
# =====================
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0
    if isinstance(valor, (int, float)):
        v = float(valor)
    else:
        t = re.sub(r"[^\d,\.]", "", str(valor))
        try:
            if "," in t:
                t = t.replace(".", "").replace(",", ".")
            v = float(t)
        except:
            return 0.0
    if v > 1500:
        return 0.0
    return v

# =====================
# LOAD DATA
# =====================
@st.cache_data(ttl=120)
def carregar_dados():
    dfs = []
    for url, nome in [(URL_ELO7, "Elo7"), (URL_SHOPEE, "Shopee")]:
        try:
            df = pd.read_csv(url, dtype=str)
            df.columns = [c.upper().strip() for c in df.columns]
            col_prod = next((c for c in df.columns if "PROD" in c or "TIT" in c or "NOME" in c), None)
            col_price = next((c for c in df.columns if "PRE" in c or "R$" in c), None)
            if not col_prod or not col_price:
                continue
            df = df.rename(columns={col_prod: "PRODUTO"})
            df["Preco_Num"] = df[col_price].apply(limpar_preco)
            df = df[df["Preco_Num"] > 0]
            df["FONTE"] = nome
            dfs.append(df[["PRODUTO", "Preco_Num", "FONTE"]])
        except:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

df = carregar_dados()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🎛️ Controles")

if df.empty:
    st.error("Erro ao carregar dados.")
    st.stop()

preco_max = st.sidebar.slider(
    "Preço máximo (R$)",
    float(df["Preco_Num"].min()),
    float(df["Preco_Num"].max()),
    float(df["Preco_Num"].quantile(0.9))
)

fontes = st.sidebar.multiselect(
    "Fontes",
    df["FONTE"].unique().tolist(),
    default=df["FONTE"].unique().tolist()
)

df_f = df[(df["Preco_Num"] <= preco_max) & (df["FONTE"].isin(fontes))].copy()

# =====================
# TABS
# =====================
tabs = st.tabs([
    "📊 Visão Geral",
    "⚔️ Comparador",
    "🧠 IA Explicativa",
    "📂 Dados"
])

# =====================
# TAB 1 — GERAL
# =====================
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Itens", len(df_f))
    c2.metric("Preço médio", f"R$ {df_f['Preco_Num'].mean():.2f}")
    c3.metric("Fontes", df_f["FONTE"].nunique())

    fig = px.box(
        df_f,
        x="FONTE",
        y="Preco_Num",
        color="FONTE",
        color_discrete_map=COLOR_MAP,
        title="Distribuição de preços"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2 — COMPARADOR
# =====================
with tabs[1]:
    termo = st.text_input("Buscar produto")
    df_c = df_f.copy()
    if termo:
        matches = process.extract(termo, df_f["PRODUTO"].unique(), limit=50, scorer=fuzz.token_set_ratio)
        similares = [m[0] for m in matches if m[1] > 40]
        df_c = df_f[df_f["PRODUTO"].isin(similares)]

    fig = px.scatter(
        df_c,
        x="FONTE",
        y="Preco_Num",
        color="FONTE",
        color_discrete_map=COLOR_MAP,
        hover_data=["PRODUTO"],
        title="Comparação de preços"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_c, use_container_width=True)

# =====================
# TAB 3 — IA EXPLICATIVA
# =====================
with tabs[2]:
    st.subheader("Explique um produto com IA")
    sel = st.selectbox("Escolha um produto", df_f["PRODUTO"].unique().tolist())
    if st.button("🧠 Explicar"):
        row = df_f[df_f["PRODUTO"] == sel].iloc[0]
        prompt = (
            "Você é um analista de mercado de impressão 3D FDM.\n"
            "Explique este anúncio de forma prática para tomada de decisão:\n\n"
            f"Produto: {row['PRODUTO']}\n"
            f"Fonte: {row['FONTE']}\n"
            f"Preço: R$ {row['Preco_Num']:.2f}\n\n"
            "Explique:\n"
            "- se o preço parece baixo, médio ou alto\n"
            "- o que isso sugere sobre concorrência\n"
            "- se vale a pena competir nesse item"
        )
        st.write(gemini_explain(prompt))

# =====================
# TAB 4 — DADOS
# =====================
with tabs[3]:
    q = st.text_input("Buscar na tabela")
    dview = df_f.copy()
    if q:
        dview = dview[dview["PRODUTO"].str.contains(q, case=False, na=False)]
    st.dataframe(dview, use_container_width=True)
