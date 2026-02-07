# dashboard.py
# BCRUZ 3D Enterprise — Decision Intelligence Edition
# (Mantém seu dashboard original e adiciona módulos de decisão + ML.
#  Inclui: contagem de itens válidos por etapa, e modelo de preço aplicando no filtro.)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
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

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# --- 2. LINKS ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- 3. LIMPEZA DE PREÇO AGRESSIVA (ANTI-OUTLIER) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0

    # Se já for número
    if isinstance(valor, (int, float)):
        val = float(valor)
    else:
        # Tratamento de String (A selva dos formatos)
        texto = str(valor).upper().strip()
        # Limpa tudo que não é número, ponto ou vírgula
        texto = re.sub(r"[^\d,.]", "", texto)

        try:
            # LÓGICA DE DETECÇÃO DE FORMATO
            if "," in texto:
                # Formato BR (39,90 ou 1.200,00)
                texto = texto.replace(".", "")   # Remove milhar (1.200 -> 1200)
                texto = texto.replace(",", ".")  # Vírgula vira ponto (39,90 -> 39.90)
            elif texto.count(".") == 1:
                # Formato Misto (Pode ser 39.90 OU 1.200)
                partes = texto.split(".")
                if len(partes[1]) == 3:
                    # Se tem 3 casas decimais (1.200), é milhar
                    texto = texto.replace(".", "")
                # Se tem 2 casas (39.90), deixa o ponto quieto

            val = float(texto)
        except:
            return 0.0

    # --- GUILHOTINA DE ERROS ÓBVIOS ---
    # Se for > 1500, ignora (provável impressora / bundle)
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
    s = re.sub(r"[^\w\s\-\/]", " ", s)  # remove ruído
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_features_from_title(title: str) -> dict:
    """
    Extrai atributos simples do título.
    """
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
        except:
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
    except:
        return "R$ 0,00"


# -----------------------------
# CARREGAMENTO + CONTAGEM POR ETAPA (NOVO)
# -----------------------------
@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]

    stats = []
    per_source = []

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines="skip", dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]

            raw_n = int(len(temp_df))
            if temp_df.empty:
                per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": 0})
                continue

            # Mapeamento
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

            # Lógica de Prazo
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

            # FILTRO LIMPO: Remove preços zerados ou inválidos
            temp_df = temp_df[temp_df["Preco_Num"] > 0.1].copy()
            after_valid_n = int(len(temp_df))

            stats.append({"raw": raw_n, "after_price": after_price_n, "after_valid": after_valid_n})
            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": after_valid_n})

            dfs.append(temp_df[cols])

        except:
            per_source.append({"fonte": f["nome"], "raw": 0, "validos": 0})
            continue

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # --- FILTRO ESTATÍSTICO AUTOMÁTICO ---
    if not final_df.empty:
        corte_superior = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= corte_superior].copy()

    # salva métricas (visíveis depois)
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

    # Normalização + features do título
    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)
    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # Anomalias
    try:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        d["anomaly_iso"] = iso.fit_predict(d[["Preco_Num"]])
        d["is_anomaly_iso"] = (d["anomaly_iso"] == -1).astype(int)
    except:
        d["is_anomaly_iso"] = 0

    try:
        lof = LocalOutlierFactor(n_neighbors=min(35, max(5, len(d) // 20)))
        lof_pred = lof.fit_predict(d[["Preco_Num"]])
        d["is_anomaly_lof"] = (lof_pred == -1).astype(int)
    except:
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
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            X = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.array(X), "SBERT (MiniLM multilingual)"
        except:
            if method == "sbert":
                pass

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
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
    except:
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
    except:
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
    except:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    out.attrs["mkt_vectorizer"] = vec_name
    out.attrs["mkt_k"] = int(out["CLUSTER_MKT"].nunique())
    return out


# -----------------------------------------
# MODELO DE PREÇO (TREINA NO TOTAL, PREVÊ NO TOTAL)
# -----------------------------------------
@st.cache_data(ttl=300)
def train_price_model(d: pd.DataFrame, min_samples: int = 40):
    """
    Treina regressor pra prever Preco_Num no dataset TOTAL.
    Se tiver poucos dados válidos, retorna None.
    """
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

    pipe = Pipeline(steps=[("prep", preproc), ("model", model)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
    except:
        return None, None, d

    # prever no dataset completo
    try:
        data["Preco_Previsto"] = pipe.predict(X)
        data["Delta_Preco"] = data["Preco_Num"] - data["Preco_Previsto"]
        data["Faixa_Min"] = np.maximum(0.0, data["Preco_Previsto"] - mae)
        data["Faixa_Max"] = data["Preco_Previsto"] + mae
    except:
        data["Preco_Previsto"] = np.nan
        data["Delta_Preco"] = np.nan
        data["Faixa_Min"] = np.nan
        data["Faixa_Max"] = np.nan

    metrics = {"MAE": mae, "R2": r2, "MIN_SAMPLES": int(min_samples), "TRAIN_ROWS": int(len(data))}
    return pipe, metrics, data


def apply_price_model(model_pipe, d: pd.DataFrame):
    """
    Aplica o modelo no filtro atual (mesmo se tiver poucos itens).
    """
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
    except:
        return d

    # Se o modelo foi treinado no total, a banda/MAE fica salva em price_metrics (usamos depois na UI).
    return data


# -----------------------------------------
# SHAP (OPCIONAL)
# -----------------------------------------
def try_shap_explain(model_pipe, df_sample: pd.DataFrame):
    if model_pipe is None or df_sample is None or df_sample.empty:
        return None

    try:
        import shap

        sample = df_sample.sample(min(250, len(df_sample)), random_state=42)

        num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]
        for c in num_cols:
            if c not in sample.columns:
                sample[c] = 0
        num_only = sample[num_cols].astype(float)

        try:
            inner_model = model_pipe.named_steps["model"]
            explainer = shap.Explainer(inner_model, num_only)
            sv = explainer(num_only)
            imp = np.abs(sv.values).mean(axis=0)
            return pd.DataFrame({"feature": num_cols, "importance": imp}).sort_values("importance", ascending=False)
        except:
            return None
    except:
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

    if "is_anomaly" in d.columns and d["is_anomaly"].sum() > 0:
        msgs.append(f"⚠️ **{int(d['is_anomaly'].sum())} anomalias** detectadas no filtro. Veja a aba Alertas.")

    return msgs


# ============================================================
# PIPELINE GLOBAL (sem filtros): enrich + dedup + clusters + modelo
# ============================================================
df_enriched = enrich_df(df)
df_enriched = canonicalize_products(df_enriched)
df_enriched = market_clusters(df_enriched)

# Treina no TOTAL pós-limpeza (importante!)
price_model, price_metrics, df_enriched = train_price_model(df_enriched, min_samples=40)

# ============================================================
# UI
# ============================================================
st.sidebar.title("🎛️ Centro de Comando")

if not df_enriched.empty:
    # --- Saúde dos dados (NOVO) ---
    st.sidebar.markdown("### ✅ Saúde dos Dados")

    raw_total = df.attrs.get("rows_raw_total", "—")
    after_price = df.attrs.get("rows_after_price_clean", "—")
    valid_total = df.attrs.get("rows_after_filter_valid", "—")

    st.sidebar.caption(f"Raw total: {raw_total} | Pós preço: {after_price} | Válidos: {valid_total}")

    per_source = df.attrs.get("per_source", [])
    if per_source:
        psrc = pd.DataFrame(per_source)
        if not psrc.empty:
            # mostra um resumo enxuto por fonte
            for _, r in psrc.iterrows():
                st.sidebar.caption(f"{r['fonte']}: raw {int(r['raw'])} → válidos {int(r['validos'])}")

    st.sidebar.metric("Itens válidos (TOTAL pós-limpeza)", int(len(df_enriched)))

    # --- Filtros originais ---
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

    # --- Ajustes ML ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Ajustes de ML")
    n_clusters = st.sidebar.slider("Clusters de Mercado (aprox.)", 6, 40, 18)
    gap_weight_ticket = st.sidebar.slider("Peso: Ticket", 0.0, 1.0, 0.55)
    gap_weight_comp = st.sidebar.slider("Peso: Baixa Competição", 0.0, 1.0, 0.30)
    gap_weight_flash = st.sidebar.slider("Peso: Flash", 0.0, 1.0, 0.15)

    # re-cluster no filtro (rápido)
    @st.cache_data(ttl=300)
    def rerun_market_clusters(d, k):
        return market_clusters(d, n_clusters=k)

    df_filtered = rerun_market_clusters(df_filtered, n_clusters)

    # Gap
    gap_df = gap_finder(df_filtered)
    if not gap_df.empty:
        gap_df["score_base"] = (
            (gap_df["ticket"] / (gap_df["ticket"].median() + 1e-9)) * gap_weight_ticket
            + (1.0 - (gap_df["itens"] / (gap_df["itens"].max() + 1e-9))) * gap_weight_comp
            + (gap_df["flash_share"]) * gap_weight_flash
        )
        gap_df = gap_df.sort_values("score_base", ascending=False)

    # Aplica modelo no filtro (MESMO SE FILTRO TIVER POUCOS ITENS)
    df_filtered = apply_price_model(price_model, df_filtered)
    if price_metrics and "Preco_Previsto" in df_filtered.columns:
        mae_global = float(price_metrics["MAE"])
        df_filtered["Delta_Preco"] = df_filtered["Preco_Num"] - df_filtered["Preco_Previsto"]
        df_filtered["Faixa_Min"] = np.maximum(0.0, df_filtered["Preco_Previsto"] - mae_global)
        df_filtered["Faixa_Max"] = df_filtered["Preco_Previsto"] + mae_global

    # --- Layout / Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
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
    ])

    # 1. GERAL (SEU ORIGINAL + CEO)
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))

        media = df_filtered["Preco_Num"].mean()
        c2.metric("Ticket Médio", f"R$ {media:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        c3.metric("Fontes", len(df_filtered["FONTE"].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered["Logistica"] == "⚡ FLASH"]))

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", title="Distribuição de Preços (Limpa)"),
                            use_container_width=True)
        with col_g2:
            st.plotly_chart(px.pie(df_filtered, names="CATEGORIA", title="Share de Categorias"),
                            use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Modo CEO — decisões")
        ceo_msgs = build_ceo_summary(df_filtered, gap_df)
        for m in ceo_msgs[:12]:
            st.write(m)

    # 2. COMPARADOR (SEU ORIGINAL)
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
                media_local = df_comp[df_comp["FONTE"] == fonte]["Preco_Num"].mean()
                fmt = f"R$ {media_local:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                cols_metrics[i].metric(f"Média {fonte}", fmt)

            fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num",
                                  hover_data=["PRODUTO"], title="Comparativo de Preços")
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "LINK"]], hide_index=True, use_container_width=True)
        else:
            if not mostrar_tudo:
                st.info("Busque um produto acima.")

    # 3. NUVENS (SEU ORIGINAL + diagnóstico rápido)
    with tab3:
        st.subheader("Nuvens de Inteligência")

        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "peças"])

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
            except:
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
        dC.metric("Anomalias (filtro)", int(df_filtered["is_anomaly"].sum()) if "is_anomaly" in df_filtered.columns else 0)
        if price_metrics:
            dD.metric("Modelo Preço (MAE global)", format_brl(price_metrics["MAE"]))
            st.caption(f"Treinado no TOTAL com {price_metrics['TRAIN_ROWS']} itens (mínimo exigido: {price_metrics['MIN_SAMPLES']}).")
        else:
            dD.metric("Modelo Preço", "Sem treino (dados totais insuficientes)")

    # 4. LAB (SEU ORIGINAL)
    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1:
            cx = st.selectbox("Eixo X", df_filtered.columns)
        with c2:
            cy = st.selectbox("Eixo Y", ["Preco_Num", "Dias_Producao"])
        with c3:
            tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])
        if tp == "Barras":
            st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Dispersão":
            st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Boxplot":
            st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)

    # 5. CRIADOR (SEU ORIGINAL + SEO valioso)
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

        st.markdown("---")
        st.subheader("🧠 SEO valioso (frequente x caro)")
        if not df_filtered.empty:
            txt_all = df_filtered["PRODUTO_NORM"].fillna("").astype(str).tolist()
            tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, max_df=0.95)
            X = tfidf.fit_transform(txt_all)
            vocab = np.array(tfidf.get_feature_names_out())
            prices = df_filtered["Preco_Num"].astype(float).values.reshape(-1, 1)
            score = (X.multiply(prices)).mean(axis=0)
            score = np.asarray(score).ravel()
            top_idx = score.argsort()[-15:][::-1]
            top_terms = [(vocab[i], float(score[i])) for i in top_idx if score[i] > 0]
            if top_terms:
                st.dataframe(pd.DataFrame(top_terms, columns=["termo", "score_valor_proxy"]), hide_index=True, use_container_width=True)
            else:
                st.info("Poucos dados para SEO valioso nesse filtro.")
        else:
            st.info("Sem dados no filtro atual.")

    # 6. DADOS (SEU ORIGINAL)
    with tab6:
        st.dataframe(df_filtered, use_container_width=True)

    # 7. MERCADO & CLUSTERS
    with tab7:
        st.header("🧩 Mercado & Clusters")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters Mercado", int(df_filtered["CLUSTER_MKT"].nunique()))
        c2.metric("Grupos (Dedup)", int(df_filtered["GROUP_ID"].nunique()) if "GROUP_ID" in df_filtered.columns else 0)
        c3.metric("Fonte Diversa (média)", f"{df_filtered.groupby('CLUSTER_MKT')['FONTE'].nunique().mean():.1f}")
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
            st.plotly_chart(
                px.scatter(
                    cluster_table,
                    x="itens",
                    y="ticket",
                    size="itens",
                    hover_data=["CLUSTER_NOME", "flash_%", "fonte_div"],
                    title="Cluster: Ticket vs Competição (itens)",
                ),
                use_container_width=True,
            )
        with colB:
            topN = cluster_table.head(15).copy()
            st.plotly_chart(
                px.bar(
                    topN.sort_values("ticket"),
                    x="ticket",
                    y="CLUSTER_NOME",
                    orientation="h",
                    title="Top 15 Clusters por Ticket Médio",
                ),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("🕳️ Gap Finder (oportunidades)")
        if gap_df is not None and not gap_df.empty:
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

    # 8. PRECIFICAÇÃO ML
    with tab8:
        st.header("💸 Precificação ML")

        # Mostra contagem e critério de treino (NOVO, pra você nunca ficar no escuro)
        st.caption(f"Itens válidos TOTAL pós-limpeza: {len(df_enriched)} | "
                   f"mínimo para treinar: {price_metrics['MIN_SAMPLES'] if price_metrics else 40}")

        if price_model is None or price_metrics is None or "Preco_Previsto" not in df_filtered.columns:
            st.warning("Modelo de preço não está ativo. Se o TOTAL pós-limpeza for menor que o mínimo, ele não treina.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE (global)", format_brl(price_metrics["MAE"]))
            c2.metric("R² (global)", f"{price_metrics['R2']:.3f}")
            c3.metric("Treino (linhas)", int(price_metrics["TRAIN_ROWS"]))
            c4.metric("Filtro (linhas)", int(len(df_filtered)))

            st.markdown("---")
            st.subheader("Mapa: Real vs Esperado (no filtro)")
            fig = px.scatter(
                df_filtered,
                x="Preco_Previsto",
                y="Preco_Num",
                color="FONTE",
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

            st.markdown("---")
            st.subheader("Explicabilidade (SHAP / opcional)")
            shap_df = try_shap_explain(price_model, df_filtered)
            if shap_df is not None and not shap_df.empty:
                st.dataframe(shap_df, hide_index=True, use_container_width=True)
            else:
                st.info("SHAP não disponível (ou não conseguiu explicar o pipeline). Se instalar `shap`, este painel melhora.")

    # 9. ALERTAS
    with tab9:
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
        st.subheader("Alertas de distribuição por fonte")
        if not df_filtered.empty:
            by_source = df_filtered.groupby("FONTE").agg(
                itens=("PRODUTO", "count"),
                ticket=("Preco_Num", "mean"),
                mediana=("Preco_Num", "median"),
                p90=("Preco_Num", lambda s: float(pd.Series(s).quantile(0.90))),
            ).reset_index()
            by_source["ticket_fmt"] = by_source["ticket"].apply(format_brl)
            by_source["p90_fmt"] = by_source["p90"].apply(format_brl)

            st.dataframe(by_source[["FONTE", "itens", "ticket_fmt", "p90_fmt"]], hide_index=True, use_container_width=True)
            st.plotly_chart(px.bar(by_source, x="FONTE", y="p90", title="P90 de preço por fonte (sinal de teto)"), use_container_width=True)

    # 10. SIMULADOR
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

        st.markdown("---")
        st.subheader("Lucro por cluster (priorização)")
        cl = sim_df.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
            itens=("PRODUTO", "count"),
            lucro_medio=("Lucro_Estimado", "mean"),
            lucro_h_medio=("Lucro_por_Hora", "mean"),
            preco_medio=("Preco_Num", "mean"),
        ).reset_index().sort_values("lucro_h_medio", ascending=False)
        cl["lucro_h_fmt"] = cl["lucro_h_medio"].apply(format_brl)
        cl["preco_fmt"] = cl["preco_medio"].apply(format_brl)
        st.dataframe(cl[["CLUSTER_NOME", "itens", "preco_fmt", "lucro_h_fmt"]], hide_index=True, use_container_width=True)

    # 11. RECOMENDADOR
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

        rec = base.sort_values("score_rec", ascending=False).head(40).copy()

        cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preco_Num", "Logistica", "LINK"]
        if has_price:
            cols += ["Preco_Previsto", "Faixa_Min", "Faixa_Max", "Delta_Preco"]

        view = rec[cols].copy()
        view["Preço"] = view["Preco_Num"].apply(format_brl)
        if has_price:
            view["Esperado"] = view["Preco_Previsto"].apply(format_brl)
            view["Faixa"] = view.apply(lambda r: f"{format_brl(r['Faixa_Min'])} ~ {format_brl(r['Faixa_Max'])}", axis=1)
            view["Delta"] = view["Delta_Preco"].apply(lambda x: format_brl(x))

        show_cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Preço", "Logistica", "LINK"]
        if has_price:
            show_cols += ["Esperado", "Faixa", "Delta"]

        st.dataframe(view[show_cols], hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Templates de título (por cluster)")
        pick_cluster = st.selectbox("Escolha um cluster para gerar template:", sorted(df_filtered["CLUSTER_NOME"].unique().tolist()))
        if pick_cluster:
            sub = df_filtered[df_filtered["CLUSTER_NOME"] == pick_cluster]
            txt = " ".join(sub["PRODUTO_NORM"].astype(str).tolist())
            pals = [p for p in re.findall(r"\w+", txt.lower()) if p not in sw and len(p) > 2]
            top_terms = [x[0] for x in Counter(pals).most_common(8)]
            st.write("Termos frequentes no cluster:", ", ".join(top_terms[:8]))
            st.code(f"[Produto Principal] 3D | {pick_cluster} | {top_terms[0] if top_terms else ''} {top_terms[1] if len(top_terms)>1 else ''} | Pronta Entrega / Personalizado")

    # 12. FORECAST
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

else:
    st.error("⚠️ Erro ao carregar dados. Verifique o Google Sheets.")
