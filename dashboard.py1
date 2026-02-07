# dashboard.py
# BCRUZ 3D Enterprise — Decision Intelligence Dashboard
# Clean build + Gemini integration (auto-detect model IDs to avoid 404)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from thefuzz import process, fuzz

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
# GEMINI (SAFE INIT + MODEL AUTODETECT)
# =====================
@st.cache_resource
def init_gemini_client():
    try:
        from google import genai
        key = st.secrets.get("GEMINI_API_KEY")
        if not key:
            return None
        return genai.Client(api_key=key)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def gemini_list_models_rest(api_key: str):
    """
    Fallback: lista modelos via REST.
    Retorna lista de dicts com name/displayName/supportedGenerationMethods (quando disponível).
    """
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

def pick_working_model_id(api_key: str, client):
    """
    Escolhe um model id que realmente existe para sua key.
    - 1) respeita GEMINI_MODEL se você setar
    - 2) tenta listar modelos e pega um que suporte generateContent
    - 3) fallback: tenta uma lista de candidatos comuns
    """
    # (1) override opcional
    override = st.secrets.get("GEMINI_MODEL") or st.secrets.get("GEMINI_MODEL_ID")
    if override:
        return override

    # (2) lista modelos (REST é mais compatível)
    models = gemini_list_models_rest(api_key)
    # preferências por custo/velocidade
    prefer = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    # Alguns retornam name como "models/gemini-..."
    def normalize_name(n):
        n = str(n or "")
        return n.replace("models/", "")

    # escolhe o melhor que suporte generateContent, se esse campo existir
    if models:
        supported = []
        for m in models:
            name = normalize_name(m.get("name"))
            methods = m.get("supportedGenerationMethods") or []
            # se o campo não existir, ainda tentamos pelo nome
            ok = ("generateContent" in methods) or (not methods)
            if name and ok:
                supported.append(name)

        # escolhe por preferência
        for p in prefer:
            if p in supported:
                return p
        # se não achou preferência, pega o primeiro suportado
        if supported:
            return supported[0]

    # (3) fallback bruto
    for cand in prefer:
        return cand

    return "gemini-pro"

gemini_client = init_gemini_client()

@st.cache_data(ttl=3600)
def get_gemini_model_id():
    key = st.secrets.get("GEMINI_API_KEY")
    if not key or gemini_client is None:
        return None
    return pick_working_model_id(key, gemini_client)

def gemini_explain(prompt: str):
    if gemini_client is None:
        return "IA indisponível (instale `google-genai` e configure GEMINI_API_KEY nos Secrets)."

    model_id = get_gemini_model_id()
    if not model_id:
        return "IA indisponível (GEMINI_API_KEY não encontrado)."

    # tenta 2 formatos: "gemini-..." e "models/gemini-..."
    tried = []
    for mid in [model_id, f"models/{model_id}"]:
        try:
            tried.append(mid)
            resp = gemini_client.models.generate_content(
                model=mid,
                contents=prompt
            )
            return resp.text
        except Exception as e:
            msg = str(e)
            # se for 404, tenta o outro formato; se continuar, mostra erro limpo
            last_err = msg

    return f"Erro ao consultar IA (model tried: {', '.join(tried)}): {last_err}"

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
            df = pd.read_csv(url, dtype=str, on_bad_lines="skip")
            df.columns = [c.upper().strip() for c in df.columns]

            col_prod = next((c for c in df.columns if any(x in c for x in ["PRODUT", "TITULO", "TÍTULO", "NOME"])), None)
            col_price = next((c for c in df.columns if any(x in c for x in ["PREÇO", "PRECO", "PRICE", "(R$)", "R$"])), None)

            if not col_prod or not col_price:
                continue

            df = df.rename(columns={col_prod: "PRODUTO"})
            df["Preco_Num"] = df[col_price].apply(limpar_preco)
            df = df[df["Preco_Num"] > 0].copy()
            df["FONTE"] = nome

            dfs.append(df[["PRODUTO", "Preco_Num", "FONTE"]])
        except:
            pass

    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        # filtro leve outlier (2% topo)
        cut = out["Preco_Num"].quantile(0.98)
        out = out[out["Preco_Num"] <= cut].copy()
        return out

    return pd.DataFrame()

df = carregar_dados()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🎛️ Controles")

# Gemini status + test
st.sidebar.markdown("### 🤖 IA (Gemini)")
if st.sidebar.button("Testar Gemini"):
    test = gemini_explain("Responda apenas: OK")
    st.sidebar.write(test)

if gemini_client is None or not st.secrets.get("GEMINI_API_KEY"):
    st.sidebar.caption("Status: ❌ IA indisponível")
else:
    st.sidebar.caption(f"Status: ✅ ativo | modelo: `{get_gemini_model_id()}`")
    st.sidebar.caption("Dica: se der 404, use `GEMINI_MODEL="<id>"` nos Secrets e reinicie.")

st.sidebar.markdown("---")

if df.empty:
    st.error("⚠️ Erro ao carregar dados. Verifique o Google Sheets.")
    st.stop()

preco_max = st.sidebar.slider(
    "Preço máximo (R$)",
    float(df["Preco_Num"].min()),
    float(df["Preco_Num"].max()),
    float(min(df["Preco_Num"].quantile(0.90), df["Preco_Num"].max()))
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
    c1.metric("Itens", int(len(df_f)))
    c2.metric("Preço médio", f"R$ {df_f['Preco_Num'].mean():.2f}".replace(".", ","))
    c3.metric("Fontes", int(df_f["FONTE"].nunique()))

    fig = px.box(
        df_f,
        x="FONTE",
        y="Preco_Num",
        color="FONTE",
        color_discrete_map=COLOR_MAP,
        title="Distribuição de preços (limpa)"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# TAB 2 — COMPARADOR
# =====================
with tabs[1]:
    termo = st.text_input("Buscar produto", placeholder="Ex: pokebola, vaso, carimbo...")
    df_c = df_f.copy()
    if termo:
        prods = df_f["PRODUTO"].dropna().astype(str).unique().tolist()
        matches = process.extract(termo, prods, limit=60, scorer=fuzz.token_set_ratio)
        similares = [m[0] for m in matches if m[1] > 40]
        df_c = df_f[df_f["PRODUTO"].isin(similares)]

    fig = px.scatter(
        df_c,
        x="FONTE",
        y="Preco_Num",
        color="FONTE",
        color_discrete_map=COLOR_MAP,
        hover_data=["PRODUTO"],
        title="Comparação de preços (por fonte)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_c, use_container_width=True, hide_index=True)

# =====================
# TAB 3 — IA EXPLICATIVA
# =====================
with tabs[2]:
    st.subheader("Explique um produto com IA")

    if df_f.empty:
        st.info("Sem dados no filtro atual.")
    else:
        sel = st.selectbox("Escolha um produto", df_f["PRODUTO"].dropna().astype(str).unique().tolist())
        if st.button("🧠 Explicar"):
            row = df_f[df_f["PRODUTO"] == sel].iloc[0]
            prompt = (
                "Você é um analista de mercado para impressão 3D FDM no Brasil.\n"
                "Explique este anúncio de forma objetiva para tomada de decisão.\n\n"
                f"Produto: {row['PRODUTO']}\n"
                f"Fonte: {row['FONTE']}\n"
                f"Preço: R$ {row['Preco_Num']:.2f}\n\n"
                "Responda em tópicos curtos:\n"
                "1) preço: baixo/médio/alto e por quê\n"
                "2) o que isso sugere sobre concorrência\n"
                "3) se vale a pena competir e como (diferenciação)\n"
            )
            st.write(gemini_explain(prompt))

# =====================
# TAB 4 — DADOS (com busca)
# =====================
with tabs[3]:
    q = st.text_input("Buscar na tabela", placeholder="Digite parte do nome...")
    dview = df_f.copy()
    if q:
        dview = dview[dview["PRODUTO"].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(dview, use_container_width=True, hide_index=True)
