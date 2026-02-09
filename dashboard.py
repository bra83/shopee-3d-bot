# dashboard.py
# BCRUZ 3D Enterprise - Decision Intelligence Edition
# Build goals:
# - Keep original dashboard capabilities and add decision + ML modules
# - ASCII-only strings (no emojis, no accented chars) to avoid encoding issues on Streamlit Cloud
# - Shopee = orange, Elo7 = green
# - Price model: sparse-safe (TF-IDF + OneHot + Ridge with log1p), plus per-cluster models
# - Optional Gemini AI explanations (guarded, will not break if unavailable)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

from thefuzz import process, fuzz

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

# Optional visuals
try:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
except Exception:
    WordCloud = None
    STOPWORDS = set()
    plt = None



# -----------------------------
# CONFIG
# -----------------------------

# Fallback seguro para STOPWORDS
try:
    _STOPWORDS_FALLBACK = set(STOPWORDS)
except Exception:
    _STOPWORDS_FALLBACK = set()

st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide")

COLOR_MAP = {
    "Shopee": "#ff7a00",  # orange
    "Elo7": "#1db954",    # green
}

# -----------------------------
# OPTIONAL GEMINI AI
# -----------------------------
@st.cache_resource
def _init_gemini_client():
    """
    Uses google-genai (pip: google-genai).
    Returns (client, model_id) or (None, None).
    """
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        return None, None

    # Allow override from secrets
    try:
        forced_model = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        forced_model = ""

    try:
        from google import genai  # google-genai
        client = genai.Client(api_key=api_key)

        if forced_model:
            return client, forced_model

        # Try to list models and pick a usable one
        model_id = None
        try:
            models = list(client.models.list())
            # Prefer newer flash/pro/1.5/2.0 style; pick first that supports generate_content
            prefer = []
            for m in models:
                mid = getattr(m, "name", None) or getattr(m, "model", None)
                if not mid:
                    continue
                s = str(mid)
                # Heuristic ordering
                score = 0
                if "gemini" in s:
                    score += 10
                if "pro" in s:
                    score += 5
                if "flash" in s:
                    score += 4
                if "1.5" in s:
                    score += 3
                if "2.0" in s:
                    score += 6
                prefer.append((score, s))
            prefer.sort(reverse=True)
            if prefer:
                model_id = prefer[0][1]
        except Exception:
            model_id = None

        # Last resort
        if not model_id:
            model_id = "gemini-pro"

        return client, model_id
    except Exception:
        return None, None


def ai_available() -> bool:
    client, model_id = _init_gemini_client()
    return client is not None and bool(model_id)


def ai_ask(prompt: str) -> str:
    client, model_id = _init_gemini_client()
    if client is None or not model_id:
        return "AI unavailable. Set GEMINI_API_KEY in Streamlit Secrets."
    try:
        # google-genai: generate_content
        resp = client.models.generate_content(model=model_id, contents=prompt)
        txt = getattr(resp, "text", None)
        if txt:
            return txt
        # Fallback: try to extract
        return str(resp)
    except Exception as e:
        return "AI error: " + str(e)


# -----------------------------
# LINKS
# -----------------------------
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"


# -----------------------------
# PRICE CLEAN
# -----------------------------
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0

    # If already numeric
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


def format_brl(v):
    try:
        return ("R$ {:,.2f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


def render_nuvens_palavras(df_base, key_prefix="wc"):
    """Renderiza as duas nuvens: frequência e valor agregado. Usa input para excluir palavras."""
    if df_base is None or df_base.empty:
        st.info("Sem dados para nuvens de palavras.")
        return

    sw = set(STOPWORDS)
    sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "peças"])

    excluir_txt = st.text_input("Palavras para excluir (separe por vírgula)", "", key=f"{key_prefix}_excluir")
    if excluir_txt:
        extras = [x.strip().lower() for x in excluir_txt.split(",") if x.strip()]
        sw.update(extras)

    c1, c2 = st.columns(2)

    with c1:
        st.caption("☁️ MAIS FREQUENTES (o que mais aparece)")
        texto_geral = " ".join(df_base["PRODUTO"].astype(str).tolist())
        try:
            wc1 = WordCloud(width=500, height=360, background_color="white", stopwords=sw).generate(texto_geral)
            fig1, ax1 = plt.subplots()
            ax1.imshow(wc1)
            ax1.axis("off")
            st.pyplot(fig1)
        except Exception as e:
            st.warning(f"Não consegui gerar a nuvem de frequência: {e}")

    with c2:
        st.caption("💰 MAIOR VALOR AGREGADO (palavras ligadas a preço)")
        word_prices = {}
        for _, row in df_base.iterrows():
            palavras = str(row["PRODUTO"]).lower().split()
            for p in palavras:
                if p not in sw and len(p) > 3:
                    word_prices.setdefault(p, []).append(float(row["Preco_Num"]))
        try:
            avg_prices = {k: sum(v)/len(v) for k, v in word_prices.items() if len(v) > 1}
            if avg_prices:
                wc2 = WordCloud(width=500, height=360, background_color="#222", max_words=60).generate_from_frequencies(avg_prices)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc2)
                ax2.axis("off")
                st.pyplot(fig2)
            else:
                st.info("Dados insuficientes para nuvem por valor.")
        except Exception as e:
            st.warning(f"Não consegui gerar a nuvem por valor: {e}")

# -----------------------------
# TEXT / FEATURES
# -----------------------------
def normalize_text(s):
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_features_from_title(title):
    t = normalize_text(title)

    def has(rx):
        return int(bool(re.search(rx, t)))

    is_kit = has(r"\bkit\b|\bconjunto\b|\bcombo\b")
    is_personalizado = has(r"\bpersonaliz")
    is_pronta = has(r"\bpronta\b|\bpronto\b|\bimediat")
    is_decor = has(r"\bdecor\b|\bdecora")
    is_organizador = has(r"\borganiz")
    is_suporte = has(r"\bsuporte\b|\bstand\b|\bbase\b")
    is_vaso = has(r"\bvaso\b|\bplant")
    is_action = has(r"\bfigure\b|\baction\b|\bminiatura\b|\bstatue\b|\bestatua\b")
    is_gamer = has(r"\bgamer\b|\bplaystation\b|\bxbox\b|\bnintendo\b|\bpc\b")
    premium = has(r"\bpremium\b|\bdeluxe\b|\bmetal\b|\bvelvet\b|\babs\b|\bpetg\b|\bresina\b")

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
        "is_kit": is_kit,
        "is_personalizado": is_personalizado,
        "is_pronta_entrega": is_pronta,
        "is_decor": is_decor,
        "is_organizador": is_organizador,
        "is_suporte": is_suporte,
        "is_vaso": is_vaso,
        "is_action": is_action,
        "is_gamer": is_gamer,
        "size_num": size_num,
        "premium": premium,
        "title_len": len(t),
        "word_count": len(t.split()),
    }


# -----------------------------
# LOAD DATA + STATS
# -----------------------------
@st.cache_data(ttl=120)
def carregar_dados():
    dfs = []
    per_source = []
    stats = {"raw_total": 0, "valid_total": 0}

    fontes = [
        {"url": URL_ELO7, "nome": "Elo7"},
        {"url": URL_SHOPEE, "nome": "Shopee"},
    ]

    for f in fontes:
        raw_n = 0
        valid_n = 0
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines="skip", dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            raw_n = int(len(temp_df))
            stats["raw_total"] += raw_n

            if temp_df.empty:
                per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": 0})
                continue

            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PRECO", "PRECO", "PRICE"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)

            temp_df = temp_df.rename(columns={col_prod: "PRODUTO"})
            temp_df["FONTE"] = f["nome"]
            temp_df["CATEGORIA"] = temp_df[col_cat] if (col_cat and col_cat in temp_df.columns) else "Geral"
            temp_df["LINK"] = temp_df[col_link] if (col_link and col_link in temp_df.columns) else "#"

            if col_preco and col_preco in temp_df.columns:
                temp_df["Preco_Num"] = temp_df[col_preco].apply(limpar_preco)
            else:
                temp_df["Preco_Num"] = 0.0

            # prazo -> dias
            if col_prazo and col_prazo in temp_df.columns:
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
            stats["valid_total"] += valid_n

            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": valid_n})
            dfs.append(temp_df[cols])
        except Exception:
            per_source.append({"fonte": f["nome"], "raw": raw_n, "validos": valid_n})

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not final_df.empty:
        corte = final_df["Preco_Num"].quantile(0.98)
        final_df = final_df[final_df["Preco_Num"] <= corte].copy()

    final_df.attrs["stats"] = stats
    final_df.attrs["per_source"] = per_source
    return final_df


# -----------------------------
# ENRICH + ANOMALIES
# -----------------------------
@st.cache_data(ttl=600)
def enrich_df(base_df):
    if base_df is None or base_df.empty:
        return base_df

    d = base_df.copy()
    d["PRODUTO_NORM"] = d["PRODUTO"].astype(str).apply(normalize_text)

    feats = d["PRODUTO"].astype(str).apply(extract_features_from_title)
    feats_df = pd.DataFrame(list(feats))
    d = pd.concat([d.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # anomalies (price-only)
    try:
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        d["is_anomaly_iso"] = (iso.fit_predict(d[["Preco_Num"]]) == -1).astype(int)
    except Exception:
        d["is_anomaly_iso"] = 0

    try:
        nn = int(min(35, max(5, len(d) // 20)))
        lof = LocalOutlierFactor(n_neighbors=nn)
        d["is_anomaly_lof"] = (lof.fit_predict(d[["Preco_Num"]]) == -1).astype(int)
    except Exception:
        d["is_anomaly_lof"] = 0

    d["is_anomaly"] = ((d["is_anomaly_iso"] + d["is_anomaly_lof"]) > 0).astype(int)
    return d


# -----------------------------
# TEXT VECTORS (TFIDF+SVD)
# -----------------------------
@st.cache_data(ttl=600)
def compute_text_vectors(texts, max_features=4000):
    texts = pd.Series(texts).fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(texts)
    n_comp = int(min(128, max(8, X_tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, f"TFIDF+SVD({n_comp})"


# -----------------------------
# CANONICALIZE (DE-DUP)
# -----------------------------
@st.cache_data(ttl=600)
def canonicalize_products(d, max_groups=250):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"], max_features=4000)

    n = len(out)
    k = int(np.clip(np.sqrt(n), 10, max_groups))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        out["GROUP_ID"] = km.fit_predict(X)
    except Exception:
        out["GROUP_ID"] = 0

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


# -----------------------------
# MARKET CLUSTERS
# -----------------------------
@st.cache_data(ttl=600)

# --- Stopwords específicas para NOMES de CLUSTER (evita poluição com de/com/para etc.) ---
CLUSTER_STOPWORDS = {
    "de","da","do","das","dos","com","para","por","sem","em","no","na","nos","nas","a","o","as","os","e","ou","um","uma","uns","umas",
    "ao","à","aos","às","pra","pro","p/","c/","d","p",
    "3d","impresso","impressao","impressão","filamento","pla","petg","abs","resina","kit","conjunto","combo",
    "un","und","unid","pc","pcs","peça","peças","cm","mm","tamanho","tam","novo","nova","produto","personalizado","personalizada",
    "frete","gratis","grátis","envio","original","pronta","entrega","pronta-entrega","pronto","imediato",
}

def _filtrar_termos_cluster(termos):
    """Remove termos comuns/ruído dos rótulos de cluster e deduplica."""
    limpos = []
    for t in termos:
        tt = str(t).strip().lower()
        if not tt:
            continue
        # termo pode vir como bigrama ("mario bros") — checa cada token
        tokens = re.split(r"[\s/_-]+", tt)
        # se todos tokens são stopwords, descarta
        if all((tok in CLUSTER_STOPWORDS or len(tok) <= 2) for tok in tokens):
            continue
        # se termo inteiro é stopword, descarta
        if tt in CLUSTER_STOPWORDS:
            continue
        limpos.append(t)

    seen=set()
    out=[]
    for t in limpos:
        k=str(t).strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def market_clusters(d, n_clusters=18):
    if d is None or d.empty:
        return d

    out = d.copy()
    X, vec_name = compute_text_vectors(out["PRODUTO_NORM"], max_features=5000)

    n = len(out)
    k = int(np.clip(n_clusters, 6, min(40, max(6, int(np.sqrt(n) * 1.2)))))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        out["CLUSTER_MKT"] = km.fit_predict(X)
    except Exception:
        out["CLUSTER_MKT"] = 0

    # Name clusters via tfidf top terms
    try:
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_t = tfidf.fit_transform(out["PRODUTO_NORM"].fillna("").astype(str))
        vocab = np.array(tfidf.get_feature_names_out())
        names = {}
        for cid in sorted(out["CLUSTER_MKT"].unique()):
            idx = np.where(out["CLUSTER_MKT"].values == cid)[0]
            if len(idx) == 0:
                names[cid] = f"cluster {cid}"
                continue
            mean_vec = np.asarray(X_t[idx].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[-4:][::-1]
            top_terms = _filtrar_termos_cluster([vocab[i] for i in top_idx if mean_vec[i] > 0])
            names[cid] = " / ".join(top_terms) if top_terms else f"cluster {cid}"
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].map(names).fillna(out["CLUSTER_MKT"].astype(str))
    except Exception:
        out["CLUSTER_NOME"] = out["CLUSTER_MKT"].astype(str)

    out.attrs["mkt_vectorizer"] = vec_name
    out.attrs["mkt_k"] = int(out["CLUSTER_MKT"].nunique())
    return out


# -----------------------------
# PRICE MODEL (GLOBAL + PER CLUSTER)
# -----------------------------
def _build_price_pipeline():
    text_col = "PRODUTO_NORM"
    cat_cols = ["FONTE", "Logistica", "CATEGORIA"]
    num_cols = ["Dias_Producao", "size_num", "premium", "is_kit", "is_personalizado", "word_count", "title_len"]

    preproc = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=9000, ngram_range=(1, 2), min_df=2, max_df=0.95), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    ridge = Ridge(alpha=2.0, random_state=42)
    model = TransformedTargetRegressor(regressor=ridge, func=np.log1p, inverse_func=np.expm1)

    pipe = Pipeline(steps=[("prep", preproc), ("model", model)])
    return pipe, (text_col, cat_cols, num_cols)


def _ensure_price_cols(data, text_col, cat_cols, num_cols):
    for c in cat_cols:
        if c not in data.columns:
            data[c] = "NA"
    for c in num_cols:
        if c not in data.columns:
            data[c] = 0
    if text_col not in data.columns:
        data[text_col] = data["PRODUTO"].astype(str).apply(normalize_text)
    return data


@st.cache_resource
def train_price_models(all_data, min_samples_global=120, min_samples_cluster=80):
    """
    Returns:
      global_model, global_metrics, cluster_models(dict), cluster_metrics_df
    """
    if all_data is None or all_data.empty:
        return None, None, {}, pd.DataFrame()

    pipe, (text_col, cat_cols, num_cols) = _build_price_pipeline()
    data = all_data.copy()
    data = _ensure_price_cols(data, text_col, cat_cols, num_cols)

    # GLOBAL
    global_model = None
    global_metrics = None
    if len(data) >= int(min_samples_global):
        X = data[[text_col] + cat_cols + num_cols]
        y = data["Preco_Num"].astype(float)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.18, random_state=42)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        global_model = pipe
        global_metrics = {
            "MAE": float(mean_absolute_error(yte, pred)),
            "R2": float(r2_score(yte, pred)),
            "TRAIN_ROWS": int(len(data)),
            "MODEL": "Ridge(log1p)-global",
        }

    # PER CLUSTER
    cluster_models = {}
    rows = []
    if "CLUSTER_MKT" in data.columns:
        for cid, grp in data.groupby("CLUSTER_MKT"):
            if len(grp) < int(min_samples_cluster):
                rows.append({
                    "cluster": int(cid),
                    "name": str(grp["CLUSTER_NOME"].iloc[0]) if "CLUSTER_NOME" in grp.columns else str(cid),
                    "n": int(len(grp)),
                    "mae": np.nan,
                    "r2": np.nan,
                    "trained": 0,
                })
                continue

            p, (tc, cc, nc) = _build_price_pipeline()
            g = grp.copy()
            g = _ensure_price_cols(g, tc, cc, nc)

            Xg = g[[tc] + cc + nc]
            yg = g["Preco_Num"].astype(float)

            try:
                Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.22, random_state=42)
                p.fit(Xtr, ytr)
                pr = p.predict(Xte)
                mae = float(mean_absolute_error(yte, pr))
                r2 = float(r2_score(yte, pr))
                cluster_models[int(cid)] = p
                rows.append({
                    "cluster": int(cid),
                    "name": str(g["CLUSTER_NOME"].iloc[0]) if "CLUSTER_NOME" in g.columns else str(cid),
                    "n": int(len(g)),
                    "mae": mae,
                    "r2": r2,
                    "trained": 1,
                })
            except Exception:
                rows.append({
                    "cluster": int(cid),
                    "name": str(g["CLUSTER_NOME"].iloc[0]) if "CLUSTER_NOME" in g.columns else str(cid),
                    "n": int(len(g)),
                    "mae": np.nan,
                    "r2": np.nan,
                    "trained": 0,
                })

    cluster_metrics_df = pd.DataFrame(rows)
    return global_model, global_metrics, cluster_models, cluster_metrics_df


def apply_price_models(df_in, global_model, cluster_models, global_mae=None):
    if df_in is None or df_in.empty:
        return df_in

    out = df_in.copy()
    pipe_tmp, (text_col, cat_cols, num_cols) = _build_price_pipeline()
    out = _ensure_price_cols(out, text_col, cat_cols, num_cols)
    X = out[[text_col] + cat_cols + num_cols]

    # Choose best model: per cluster if available, else global
    preds = np.full(len(out), np.nan, dtype=float)

    if "CLUSTER_MKT" in out.columns and cluster_models:
        for cid, idx in out.groupby("CLUSTER_MKT").groups.items():
            idx = list(idx)
            model = cluster_models.get(int(cid), None)
            if model is not None:
                preds[idx] = model.predict(X.iloc[idx])
    if global_model is not None:
        missing = np.isnan(preds)
        if missing.any():
            preds[missing] = global_model.predict(X.iloc[np.where(missing)[0]])

    out["Preco_Previsto"] = preds
    out["Delta_Preco"] = out["Preco_Num"] - out["Preco_Previsto"]

    # Bands using global MAE as fallback
    if global_mae is not None and np.isfinite(global_mae):
        out["Faixa_Min"] = np.maximum(0.0, out["Preco_Previsto"] - float(global_mae))
        out["Faixa_Max"] = out["Preco_Previsto"] + float(global_mae)
    else:
        out["Faixa_Min"] = np.nan
        out["Faixa_Max"] = np.nan

    return out


# -----------------------------
# GAP FINDER
# -----------------------------
@st.cache_data(ttl=600)
def gap_finder(d):
    if d is None or d.empty or "CLUSTER_MKT" not in d.columns:
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
    g = g.sort_values("score_base", ascending=False)

    examples = []
    for _, row in g.head(30).iterrows():
        cid = int(row["CLUSTER_MKT"])
        ex = d[d["CLUSTER_MKT"] == cid].sort_values("Preco_Num", ascending=False).head(3)
        examples.append({
            "CLUSTER_MKT": cid,
            "EX1": ex["PRODUTO"].iloc[0] if len(ex) > 0 else "",
            "EX2": ex["PRODUTO"].iloc[1] if len(ex) > 1 else "",
            "EX3": ex["PRODUTO"].iloc[2] if len(ex) > 2 else "",
        })
    exdf = pd.DataFrame(examples)
    out = g.merge(exdf, left_on="CLUSTER_MKT", right_on="CLUSTER_MKT", how="left")
    return out


# -----------------------------
# SIMULATOR (profit/hour)
# -----------------------------
def estimate_print_hours(row, base_hours=2.0):
    days = float(row.get("Dias_Producao", 15))
    size = float(row.get("size_num", 0))
    logist = str(row.get("Logistica", "NORMAL"))

    h = base_hours
    if size > 0:
        h += min(6.0, size / 25.0)
    h += min(8.0, max(0.0, (days - 2.0)) / 6.0)
    if logist == "FLASH":
        h *= 0.75
    return float(np.clip(h, 0.4, 18.0))


def compute_profit(d, custo_hora=8.0, custo_grama=0.12, gramas_base=60, taxa_falha=0.06, taxa_marketplace=0.14, embalagem=4.0):
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


# -----------------------------
# CEO SUMMARY
# -----------------------------
def build_ceo_summary(d, gap):
    if d is None or d.empty:
        return ["No data for decisions."]
    msgs = []

    top_price = d.sort_values("Preco_Num", ascending=False).head(1)
    if len(top_price):
        msgs.append("Highest ticket in current filter: " + format_brl(top_price["Preco_Num"].iloc[0]) + " - " + str(top_price["PRODUTO"].iloc[0]) + " (" + str(top_price["FONTE"].iloc[0]) + ")")

    if "Delta_Preco" in d.columns and d["Delta_Preco"].notna().any():
        under = d.sort_values("Delta_Preco", ascending=True).head(3)
        if len(under):
            msgs.append("Top 3 likely underpriced (below expected):")
            for _, r in under.iterrows():
                msgs.append("- " + str(r["PRODUTO"]) + " | real " + format_brl(r["Preco_Num"]) + " vs expected " + format_brl(r.get("Preco_Previsto", 0)))

        over = d.sort_values("Delta_Preco", ascending=False).head(3)
        if len(over):
            msgs.append("Top 3 likely overpriced (above expected):")
            for _, r in over.iterrows():
                msgs.append("- " + str(r["PRODUTO"]) + " | real " + format_brl(r["Preco_Num"]) + " vs expected " + format_brl(r.get("Preco_Previsto", 0)))

    if gap is not None and not gap.empty:
        top = gap.head(3)
        msgs.append("Top 3 cluster opportunities (high ticket + lower competition + flash):")
        for _, r in top.iterrows():
            msgs.append("- " + str(r["CLUSTER_NOME"]) + " | score " + str(round(float(r["score_base"]), 2)) + " | ticket " + format_brl(r["ticket"]) + " | items " + str(int(r["itens"])))

    if "is_anomaly" in d.columns and int(d["is_anomaly"].sum()) > 0:
        msgs.append("Warning: " + str(int(d["is_anomaly"].sum())) + " anomalies detected in current filter. See Alerts tab.")

    return msgs


# ============================================================
# PIPELINE
# ============================================================
df_raw = carregar_dados()

if df_raw is None or df_raw.empty:
    st.error("Error loading data. Check Google Sheets publish CSV.")
    st.stop()

df_enriched = enrich_df(df_raw)
df_enriched = canonicalize_products(df_enriched)
df_enriched = market_clusters(df_enriched)

# Train models (global + per cluster)
global_model, global_metrics, cluster_models, cluster_metrics_df = train_price_models(
    df_enriched, min_samples_global=120, min_samples_cluster=80
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Control Center")

# Data health
stats = df_raw.attrs.get("stats", {})
per_source = df_raw.attrs.get("per_source", [])

st.sidebar.markdown("Data Health")
st.sidebar.caption("Raw total: " + str(stats.get("raw_total", "-")) + " | Valid total: " + str(stats.get("valid_total", "-")))
if per_source:
    for r in per_source:
        st.sidebar.caption(str(r.get("fonte")) + ": raw " + str(int(r.get("raw", 0))) + " -> valid " + str(int(r.get("validos", 0))))

st.sidebar.metric("Valid items (total after clean)", int(len(df_enriched)))

# Filters
st.sidebar.markdown("---")
st.sidebar.markdown("Ticket Filter")
max_val = float(df_enriched["Preco_Num"].max())
preco_max = st.sidebar.slider("Max price (BRL)", 0.0, max_val, min(500.0, max_val))
fontes_sel = st.sidebar.multiselect("Sources", df_enriched["FONTE"].unique().tolist(), default=df_enriched["FONTE"].unique().tolist())

df_filtered = df_enriched[(df_enriched["FONTE"].isin(fontes_sel)) & (df_enriched["Preco_Num"] <= preco_max)].copy()

cats = st.sidebar.multiselect("Categories", sorted(df_filtered["CATEGORIA"].unique().tolist()))
if cats:
    df_filtered = df_filtered[df_filtered["CATEGORIA"].isin(cats)].copy()

st.sidebar.metric("Valid items (current filter)", int(len(df_filtered)))

# Clusters slider
st.sidebar.markdown("---")
st.sidebar.markdown("ML Settings")
n_clusters = st.sidebar.slider("Market clusters (approx)", 6, 40, 18)

@st.cache_data(ttl=600)
def rerun_market_clusters(d, k):
    return market_clusters(d, n_clusters=k)

df_filtered = rerun_market_clusters(df_filtered, n_clusters)

# Gap finder
gap_df = gap_finder(df_filtered)

# Apply price models to current filter
global_mae = float(global_metrics["MAE"]) if global_metrics and "MAE" in global_metrics else None
df_filtered = apply_price_models(df_filtered, global_model, cluster_models, global_mae=global_mae)

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "Overview",
    "Comparator",
    "AI Insights",
    "Lab",
    "Title Builder",
    "Data",
    "Data Analysis",
    "Market Clusters",
    "Score",
    "Pricing ML",
    "Alerts",
    "Simulator",
    "Recommender",
    "Forecast",
])

# -----------------------------
# TAB 1 OVERVIEW
# -----------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", len(df_filtered))
    c2.metric("Avg ticket", format_brl(df_filtered["Preco_Num"].mean() if len(df_filtered) else 0))
    c3.metric("Sources", int(df_filtered["FONTE"].nunique()))
    c4.metric("Flash items", int((df_filtered["Logistica"] == "FLASH").sum()))

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE, title="Price distribution by source",
                     color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    with col_g2:
        fig = px.pie(df_filtered, names="CATEGORIA", title="Category share")
        st.plotly_chart(fig, use_container_width=True)


    # Word clouds (frequency vs value-weighted)
    st.markdown("---")
    st.subheader("Word clouds")
    if WordCloud is None or plt is None:
        st.info("WordCloud/matplotlib not available. Add 'wordcloud' and 'matplotlib' to requirements.txt.")
    elif df_filtered.empty:
        st.info("No data in current filter.")
    else:
        sw_local = set(STOPWORDS) if isinstance(STOPWORDS, (set, frozenset)) else set()
        sw_local.update([
            "de","da","do","das","dos","para","com","sem","em","no","na","nos","nas",
            "3d","pla","petg","abs","resina","kit","un","cm","mm","peca","pecas","novo","nova",
            "personalizado","personalizada","pronta","entrega","imediato","imediata",
        ])

        cA, cB = st.columns(2)

        with cA:
            st.caption("Most frequent terms")
            texto = " ".join(df_filtered["PRODUTO"].astype(str).tolist())
            try:
                wc = WordCloud(width=520, height=320, background_color="white", stopwords=sw_local, max_words=80).generate(texto)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc)
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            except Exception as e:
                st.warning(f"Could not render word cloud: {e}")

        with cB:
            st.caption("Higher value terms (avg price proxy)")
            word_prices = {}
            for _, row in df_filtered.iterrows():
                words = re.findall(r"[a-zA-Z0-9_]+", str(row.get("PRODUTO","")).lower())
                for w in words:
                    if w in sw_local or len(w) < 4:
                        continue
                    word_prices.setdefault(w, []).append(float(row.get("Preco_Num", 0.0) or 0.0))
            try:
                avg_prices = {k: (sum(v)/len(v)) for k, v in word_prices.items() if len(v) >= 2}
                if avg_prices:
                    wc2 = WordCloud(width=520, height=320, background_color="white", stopwords=sw_local, max_words=80).generate_from_frequencies(avg_prices)
                    fig_wc2, ax_wc2 = plt.subplots()
                    ax_wc2.imshow(wc2)
                    ax_wc2.axis("off")
                    st.pyplot(fig_wc2)
                else:
                    st.info("Not enough repeated terms for value-weighted cloud.")
            except Exception as e:
                st.warning(f"Could not render value cloud: {e}")

    st.markdown("---")
    st.subheader("CEO Mode - decisions")
    for m in build_ceo_summary(df_filtered, gap_df)[:12]:
        st.write(m)

    if ai_available():
        with st.expander("AI - explain current overview"):
            if st.button("Ask AI about this filter", key="ai_overview"):
                prompt = "You are a business analyst for FDM 3D printing. Summarize key decisions for this filter.\n"
                prompt += "Stats:\n"
                prompt += "- items: " + str(len(df_filtered)) + "\n"
                prompt += "- avg_price: " + str(df_filtered["Preco_Num"].mean() if len(df_filtered) else 0) + "\n"
                prompt += "- sources: " + ", ".join(sorted(df_filtered["FONTE"].unique().tolist())) + "\n"
                if gap_df is not None and not gap_df.empty:
                    top = gap_df.head(5)[["CLUSTER_NOME", "itens", "ticket", "flash_share", "score_base"]].to_dict("records")
                    prompt += "Top opportunities:\n" + str(top) + "\n"
                st.write(ai_ask(prompt))


    st.markdown("---")
    st.subheader("☁️ Nuvens de palavras (filtro atual)")
    render_nuvens_palavras(df_filtered, key_prefix="wc_overview")

# -----------------------------
# TAB 2 COMPARATOR
# -----------------------------
with tabs[1]:
    st.subheader("Price Comparator")
    col_input, col_check = st.columns([3, 1])
    with col_input:
        termo = st.text_input("Search product (fuzzy)", placeholder="Example: vaso robert")
    with col_check:
        st.write("")
        show_all = st.checkbox("Show all", value=False)

    df_comp = pd.DataFrame()
    if show_all:
        df_comp = df_filtered
    elif termo:
        prods = df_filtered["PRODUTO"].unique().tolist()
        matches = process.extract(termo, prods, limit=60, scorer=fuzz.token_set_ratio)
        similares = [x[0] for x in matches if x[1] > 40]
        df_comp = df_filtered[df_filtered["PRODUTO"].isin(similares)]

    if not df_comp.empty:
        fig = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE, size="Preco_Num",
                         hover_data=["PRODUTO"], title="Price comparison",
                         color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_comp[["FONTE", "PRODUTO", "Preco_Num", "LINK"]], hide_index=True, use_container_width=True)
    else:
        st.info("Type a search term or enable show all.")

# -----------------------------
# TAB 3 AI INSIGHTS (row + chart + cluster)
# -----------------------------
with tabs[2]:
    st.subheader("AI Insights (optional)")
    st.caption("This does not auto-read chart clicks. Select an item or a cluster and ask AI.")

    if not ai_available():
        st.warning("AI unavailable. Configure GEMINI_API_KEY in Streamlit Secrets.")
    else:
        colA, colB = st.columns(2)

        with colA:
            st.markdown("Explain a product row")
            if len(df_filtered):
                pick = st.selectbox("Pick a product", df_filtered["PRODUTO"].unique().tolist(), key="ai_pick_prod")
                row = df_filtered[df_filtered["PRODUTO"] == pick].head(1)
                if st.button("Explain this row", key="ai_row_btn") and not row.empty:
                    r = row.iloc[0].to_dict()
                    prompt = "You are a senior marketplace analyst for FDM 3D printing.\n"
                    prompt += "Explain this listing and pricing position. Give 3 actions.\n"
                    prompt += "Row: " + str(r) + "\n"
                    st.write(ai_ask(prompt))
            else:
                st.info("No items in current filter.")

        with colB:
            st.markdown("Explain a cluster")
            if "CLUSTER_NOME" in df_filtered.columns and len(df_filtered):
                clusters = sorted(df_filtered["CLUSTER_NOME"].unique().tolist())
                pickc = st.selectbox("Pick a cluster", clusters, key="ai_pick_cluster")
                sub = df_filtered[df_filtered["CLUSTER_NOME"] == pickc].copy()
                summ = {
                    "cluster": pickc,
                    "n": int(len(sub)),
                    "avg_price": float(sub["Preco_Num"].mean()) if len(sub) else 0.0,
                    "median_price": float(sub["Preco_Num"].median()) if len(sub) else 0.0,
                    "sources": sorted(sub["FONTE"].unique().tolist()),
                    "flash_share": float((sub["Logistica"] == "FLASH").mean()) if len(sub) else 0.0,
                }
                if st.button("Explain this cluster", key="ai_cluster_btn"):
                    prompt = "You are a strategic advisor. Explain the market cluster and how to win with FDM printing.\n"
                    prompt += "Cluster summary: " + str(summ) + "\n"
                    prompt += "Give: (1) product angle, (2) pricing strategy, (3) differentiation.\n"
                    st.write(ai_ask(prompt))
            else:
                st.info("No cluster data in filter.")

        st.markdown("---")
        st.markdown("Explain pricing map (real vs expected)")
        if st.button("Explain pricing distribution", key="ai_price_map"):
            summ = {
                "n": int(len(df_filtered)),
                "mae_global": float(global_metrics["MAE"]) if global_metrics else None,
                "r2_global": float(global_metrics["R2"]) if global_metrics else None,
                "underpriced_examples": df_filtered.sort_values("Delta_Preco", ascending=True).head(5)[["PRODUTO", "FONTE", "Preco_Num", "Preco_Previsto", "Delta_Preco"]].to_dict("records") if "Delta_Preco" in df_filtered.columns else [],
                "overpriced_examples": df_filtered.sort_values("Delta_Preco", ascending=False).head(5)[["PRODUTO", "FONTE", "Preco_Num", "Preco_Previsto", "Delta_Preco"]].to_dict("records") if "Delta_Preco" in df_filtered.columns else [],
            }
            prompt = "Explain the pricing map and what actions to take.\n"
            prompt += "Data: " + str(summ) + "\n"
            prompt += "Be concise, tactical, and business-focused.\n"
            st.write(ai_ask(prompt))

# -----------------------------
# TAB 4 LAB
# -----------------------------
with tabs[3]:
    st.subheader("Lab")
    c1, c2, c3 = st.columns(3)
    with c1:
        cx = st.selectbox("X axis", df_filtered.columns.tolist())
    with c2:
        cy = st.selectbox("Y axis", ["Preco_Num", "Dias_Producao"])
    with c3:
        tp = st.selectbox("Chart type", ["Bar", "Scatter", "Box"])

    if tp == "Bar":
        st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)
    elif tp == "Scatter":
        st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)
    else:
        st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE", color_discrete_map=COLOR_MAP), use_container_width=True)

# -----------------------------
# TAB 5 TITLE BUILDER
# -----------------------------
with tabs[4]:
    st.subheader("SEO Title Builder")
    keyword = st.text_input("Keyword", "vaso")
    if keyword:
        df_c = df_enriched[df_enriched["PRODUTO"].str.contains(keyword, case=False, na=False)]
        if not df_c.empty:
            txt = " ".join(df_c["PRODUTO"].astype(str))
            pals = [p for p in re.findall(r"\w+", txt.lower()) if len(p) > 2]
            top = [x[0].title() for x in Counter(pals).most_common(8)]
            st.success("Top keywords: " + ", ".join(top[:6]))
            st.code(keyword.title() + " 3D " + " ".join(top[:2]) + " - Alta Qualidade")
        else:
            st.warning("No data for this keyword in current dataset.")

# -----------------------------
# TAB 6 DATA (with search)
# -----------------------------
with tabs[5]:
    st.subheader("Data")
    q = st.text_input("Search in table", "")
    view = df_filtered.copy()
    if q.strip():
        view = view[view["PRODUTO"].str.contains(q, case=False, na=False)].copy()
    st.dataframe(view, use_container_width=True)

# -----------------------------
# TAB 7 MARKET CLUSTERS
# -----------------------------

# -----------------------------
# DATA ANALYSIS (stats + qualitative + quantitative)
# -----------------------------
with tabs[6]:
    st.subheader("Data Analysis")
    st.caption("Statistical, quantitative and qualitative analysis of the current filter.")

    if df_filtered is None or df_filtered.empty:
        st.info("No data in current filter.")
    else:
        st.markdown("### Summary statistics (current filter)")
        s = df_filtered["Preco_Num"].astype(float)
        desc = s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame().T
        desc["mean_fmt"] = desc["mean"].apply(format_brl)
        desc["p50_fmt"] = desc["50%"].apply(format_brl)
        desc["min_fmt"] = desc["min"].apply(format_brl)
        desc["max_fmt"] = desc["max"].apply(format_brl)
        st.dataframe(desc[["count", "mean_fmt", "p50_fmt", "min_fmt", "max_fmt"]], hide_index=True, use_container_width=True)

        st.markdown("### By source")
        by_source = df_filtered.groupby("FONTE").agg(
            items=("PRODUTO", "count"),
            avg_price=("Preco_Num", "mean"),
            med_price=("Preco_Num", "median"),
            p90=("Preco_Num", lambda x: float(pd.Series(x).quantile(0.90))),
            flash_share=("Logistica", lambda x: float((pd.Series(x) == "FLASH").mean())),
        ).reset_index()
        by_source["avg_fmt"] = by_source["avg_price"].apply(format_brl)
        by_source["med_fmt"] = by_source["med_price"].apply(format_brl)
        by_source["p90_fmt"] = by_source["p90"].apply(format_brl)
        by_source["flash_pct"] = (by_source["flash_share"] * 100).round(1)
        st.dataframe(by_source[["FONTE", "items", "avg_fmt", "med_fmt", "p90_fmt", "flash_pct"]], hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.bar(by_source, x="FONTE", y="avg_price", color="FONTE", color_discrete_map=COLOR_MAP, title="Average price by source"),
            use_container_width=True,
        )

        st.markdown("### By category (top 25)")
        by_cat = df_filtered.groupby("CATEGORIA").agg(
            items=("PRODUTO", "count"),
            avg_price=("Preco_Num", "mean"),
            med_price=("Preco_Num", "median"),
        ).reset_index().sort_values(["items", "avg_price"], ascending=False).head(25)
        by_cat["avg_fmt"] = by_cat["avg_price"].apply(format_brl)
        st.dataframe(by_cat[["CATEGORIA", "items", "avg_fmt"]], hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.scatter(by_cat, x="items", y="avg_price", size="items", hover_data=["CATEGORIA"], title="Category: items vs average price"),
            use_container_width=True,
        )

        st.markdown("### Qualitative: frequent terms in titles")
        txt_all = " ".join(df_filtered["PRODUTO_NORM"].fillna("").astype(str).tolist())
        words = [w for w in re.findall(r"\w+", txt_all.lower()) if len(w) >= 4]
        top_words = Counter(words).most_common(30)
        st.dataframe(pd.DataFrame(top_words, columns=["term", "count"]), hide_index=True, use_container_width=True)

        st.markdown("### Extremes (price)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Top 15 most expensive")
            hi = df_filtered.sort_values("Preco_Num", ascending=False).head(15).copy()
            hi["price"] = hi["Preco_Num"].apply(format_brl)
            st.dataframe(hi[["FONTE", "PRODUTO", "price", "CATEGORIA", "LINK"]], hide_index=True, use_container_width=True)
        with c2:
            st.caption("Top 15 cheapest")
            lo = df_filtered.sort_values("Preco_Num", ascending=True).head(15).copy()
            lo["price"] = lo["Preco_Num"].apply(format_brl)
            st.dataframe(lo[["FONTE", "PRODUTO", "price", "CATEGORIA", "LINK"]], hide_index=True, use_container_width=True)

        if ai_available():
            with st.expander("AI summary of this analysis"):
                if st.button("Generate AI summary", key="ai_data_analysis"):
                    prompt = "You are a senior data analyst for a small FDM 3D print business.\n"
                    prompt += "Summarize key insights from the current filter dataset: pricing, categories, sources, and opportunities.\n"
                    prompt += "Provide 6 bullet points and 3 actions.\n"
                    prompt += "Summary stats: " + str(desc.to_dict("records")) + "\n"
                    prompt += "By source: " + str(by_source.to_dict("records")) + "\n"
                    prompt += "Top terms: " + str(top_words[:15]) + "\n"
                    st.write(ai_ask(prompt))

with tabs[7]:
    st.subheader("Market Clusters")
    if "CLUSTER_MKT" not in df_filtered.columns:
        st.info("No cluster info.")
    else:
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

        colA, colB = st.columns(2)
        with colA:
            fig = px.scatter(cluster_table, x="itens", y="ticket", size="itens",
                             hover_data=["CLUSTER_NOME", "flash_pct", "fonte_div"],
                             title="Ticket vs competition (items)")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            topN = cluster_table.head(15).copy().sort_values("ticket")
            fig = px.bar(topN, x="ticket", y="CLUSTER_NOME", orientation="h", title="Top clusters by avg ticket")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Gap Finder (opportunities)")
        if gap_df is not None and not gap_df.empty:
            show = gap_df.head(25).copy()
            show["ticket_fmt"] = show["ticket"].apply(format_brl)
            show["flash_pct"] = (show["flash_share"] * 100).round(1)
            st.dataframe(show[["CLUSTER_MKT", "CLUSTER_NOME", "score_base", "itens", "ticket_fmt", "flash_pct", "EX1", "EX2", "EX3"]],
                         hide_index=True, use_container_width=True)
        else:
            st.info("Not enough data for gap finder in this filter.")

# -----------------------------
# TAB 8 PRICING ML
# -----------------------------

# -----------------------------
# SCORE (explain + calculations + FDM weights + profit/hour score)
# -----------------------------
with tabs[8]:
    st.subheader("Score (meaning and calculation)")

    st.markdown("### Why cluster names look like 'mario / kit / festa'?")
    st.write(
        "Cluster names are created automatically from the most frequent and distinctive words (TF-IDF) in product titles. "
        "So 'mario / kit / festa' means those words often appear together in that cluster's titles. "
        "It is a label for the pattern, not a recommendation to copy licensed IP."
    )

    st.markdown("### FDM opportunity score (radar)")
    st.write(
        "This score ranks clusters for investigation using: avg price (value), low competition inside the cluster, "
        "FLASH share (urgency), and an anomaly penalty."
    )

    # FDM default weights (balanced for time + value): ticket 0.45, low-competition 0.35, flash 0.10, anomalies 0.10
    col1, col2, col3, col4 = st.columns(4)
    w_ticket = col1.slider("Weight: ticket", 0.0, 1.0, 0.45, 0.05)
    w_comp = col2.slider("Weight: low competition", 0.0, 1.0, 0.35, 0.05)
    w_flash = col3.slider("Weight: flash/urgency", 0.0, 1.0, 0.10, 0.05)
    w_anom = col4.slider("Penalty: anomalies", 0.0, 1.0, 0.10, 0.05)

    # Compute per-cluster metrics
    if "is_anomaly" not in df_filtered.columns:
        df_filtered["is_anomaly"] = 0

    cl = df_filtered.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
        items=("PRODUTO", "count"),
        avg_price=("Preco_Num", "mean"),
        med_price=("Preco_Num", "median"),
        flash_share=("Logistica", lambda x: float((pd.Series(x) == "FLASH").mean())),
        anom_share=("is_anomaly", "mean"),
    ).reset_index()

    def norm01(x):
        x = pd.Series(x).astype(float)
        lo = float(x.quantile(0.10))
        hi = float(x.quantile(0.90))
        if hi - lo < 1e-9:
            return pd.Series([0.5] * len(x))
        return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)

    ticket_n = norm01(cl["avg_price"])
    low_comp_n = 1.0 - norm01(cl["items"])
    flash_n = cl["flash_share"].fillna(0.0).clip(0, 1)
    anom_pen = cl["anom_share"].fillna(0.0).clip(0, 1)

    cl["score_fdm"] = (
        ticket_n * float(w_ticket)
        + low_comp_n * float(w_comp)
        + flash_n * float(w_flash)
        - anom_pen * float(w_anom)
    )

    cl = cl.sort_values("score_fdm", ascending=False)
    show = cl.head(30).copy()
    show["avg_price_fmt"] = show["avg_price"].apply(format_brl)
    show["flash_pct"] = (show["flash_share"] * 100).round(1)
    show["anom_pct"] = (show["anom_share"] * 100).round(1)

    st.dataframe(
        show[["CLUSTER_MKT", "CLUSTER_NOME", "items", "avg_price_fmt", "flash_pct", "anom_pct", "score_fdm"]],
        hide_index=True,
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(show.sort_values("score_fdm"), x="score_fdm", y="CLUSTER_NOME", orientation="h", title="Top clusters by FDM score"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Alternative score: profit/hour")
    st.write("Uses simulator assumptions to estimate profit/hour and ranks clusters by average profit/hour.")

    a, b, c, d, e, f = st.columns(6)
    custo_hora = a.number_input("Cost/hour (BRL)", min_value=0.0, value=8.0, step=0.5, key="score_cost_hour")
    custo_grama = b.number_input("Cost/gram (BRL)", min_value=0.0, value=0.12, step=0.01, format="%.2f", key="score_cost_g")
    gramas_base = c.number_input("Base grams", min_value=10, value=60, step=5, key="score_base_g")
    taxa_falha = d.number_input("Failure rate", min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.2f", key="score_fail")
    taxa_market = e.number_input("Marketplace fee", min_value=0.0, max_value=0.5, value=0.14, step=0.01, format="%.2f", key="score_fee")
    embalagem = f.number_input("Packaging (BRL)", min_value=0.0, value=4.0, step=0.5, key="score_pack")

    sim = compute_profit(
        df_filtered,
        custo_hora=custo_hora,
        custo_grama=custo_grama,
        gramas_base=gramas_base,
        taxa_falha=taxa_falha,
        taxa_marketplace=taxa_market,
        embalagem=embalagem,
    )

    if sim is None or sim.empty or "Lucro_por_Hora" not in sim.columns:
        st.info("Not enough data to compute profit/hour score.")
    else:
        clp = sim.groupby(["CLUSTER_MKT", "CLUSTER_NOME"], dropna=False).agg(
            items=("PRODUTO", "count"),
            avg_price=("Preco_Num", "mean"),
            avg_profit_h=("Lucro_por_Hora", "mean"),
            pct_negative=("Lucro_Estimado", lambda s: float((pd.Series(s) < 0).mean())),
            flash_share=("Logistica", lambda x: float((pd.Series(x) == "FLASH").mean())),
        ).reset_index()

        x = clp["avg_profit_h"].astype(float)
        lo = float(x.quantile(0.10))
        hi = float(x.quantile(0.90))
        if hi - lo < 1e-9:
            clp["score_profit"] = 0.5
        else:
            clp["score_profit"] = ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)
        clp["score_profit"] = (clp["score_profit"] * (1.0 - clp["pct_negative"].fillna(0.0).clip(0, 1))).clip(0, 1)

        clp = clp.sort_values("score_profit", ascending=False)
        sp = clp.head(30).copy()
        sp["avg_price_fmt"] = sp["avg_price"].apply(format_brl)
        sp["avg_profit_h_fmt"] = sp["avg_profit_h"].apply(format_brl)
        sp["neg_pct"] = (sp["pct_negative"] * 100).round(1)
        st.dataframe(
            sp[["CLUSTER_MKT", "CLUSTER_NOME", "items", "avg_price_fmt", "avg_profit_h_fmt", "neg_pct", "score_profit"]],
            hide_index=True,
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(sp.sort_values("score_profit"), x="score_profit", y="CLUSTER_NOME", orientation="h", title="Top clusters by profit/hour score"),
            use_container_width=True,
        )

    if ai_available():
        with st.expander("AI: explain a cluster"):
            clusters = sorted(df_filtered["CLUSTER_NOME"].unique().tolist())
            pickc = st.selectbox("Pick a cluster", clusters, key="ai_score_cluster")
            if st.button("Explain", key="ai_score_btn"):
                sub = df_filtered[df_filtered["CLUSTER_NOME"] == pickc].copy()
                summ = {
                    "cluster": pickc,
                    "n": int(len(sub)),
                    "avg_price": float(sub["Preco_Num"].mean()) if len(sub) else 0.0,
                    "flash_share": float((sub["Logistica"] == "FLASH").mean()) if len(sub) else 0.0,
                }
                prompt = "Explain this cluster pattern for an FDM 3D print seller.\n"
                prompt += "Cluster summary: " + str(summ) + "\n"
                prompt += "Give 3 product angles that do NOT require licensed IP.\n"
                st.write(ai_ask(prompt))

with tabs[9]:
    st.subheader("Pricing ML")

    if global_metrics is None or "Preco_Previsto" not in df_filtered.columns:
        st.warning("Price model not active.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Global MAE", format_brl(global_metrics.get("MAE", 0)))
        c2.metric("Global R2", "{:.3f}".format(float(global_metrics.get("R2", 0))))
        c3.metric("Training rows", int(global_metrics.get("TRAIN_ROWS", 0)))
        c4.metric("Filter rows", int(len(df_filtered)))

        st.markdown("---")
        st.subheader("Real vs Expected (filter)")
        fig = px.scatter(df_filtered, x="Preco_Previsto", y="Preco_Num", color="FONTE", color_discrete_map=COLOR_MAP_FONTE,
                         hover_data=["PRODUTO", "CATEGORIA", "Logistica"],
                         title="Real vs Expected (above line = expensive; below = cheap)",
                         color_discrete_map=COLOR_MAP)
        try:
            mn = float(min(df_filtered["Preco_Previsto"].min(), df_filtered["Preco_Num"].min()))
            mx = float(max(df_filtered["Preco_Previsto"].max(), df_filtered["Preco_Num"].max()))
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Underpriced vs Overpriced (top 20)")
        colL, colR = st.columns(2)
        with colL:
            under = df_filtered.sort_values("Delta_Preco", ascending=True).head(20).copy()
            under["real"] = under["Preco_Num"].apply(format_brl)
            under["expected"] = under["Preco_Previsto"].apply(format_brl)
            st.caption("Underpriced")
            st.dataframe(under[["FONTE", "PRODUTO", "real", "expected", "LINK"]], hide_index=True, use_container_width=True)
        with colR:
            over = df_filtered.sort_values("Delta_Preco", ascending=False).head(20).copy()
            over["real"] = over["Preco_Num"].apply(format_brl)
            over["expected"] = over["Preco_Previsto"].apply(format_brl)
            st.caption("Overpriced")
            st.dataframe(over[["FONTE", "PRODUTO", "real", "expected", "LINK"]], hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("MAE by cluster (models)")
        if cluster_metrics_df is None or cluster_metrics_df.empty:
            st.info("No per-cluster metrics.")
        else:
            cm = cluster_metrics_df.copy()
            cm["mae_fmt"] = cm["mae"].apply(lambda x: format_brl(x) if pd.notna(x) else "-")
            cm["r2_fmt"] = cm["r2"].apply(lambda x: "{:.3f}".format(float(x)) if pd.notna(x) else "-")
            st.dataframe(cm[["cluster", "name", "n", "trained", "mae_fmt", "r2_fmt"]], hide_index=True, use_container_width=True)

# -----------------------------
# TAB 9 ALERTS
# -----------------------------
with tabs[10]:
    st.subheader("Alerts and anomalies")
    if "is_anomaly" in df_filtered.columns:
        anom = df_filtered[df_filtered["is_anomaly"] == 1].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies", int(len(anom)))
        c2.metric("Anomaly pct", "{:.1f}%".format((len(anom) / max(1, len(df_filtered)) * 100)))
        c3.metric("Max anomaly price", format_brl(anom["Preco_Num"].max()) if len(anom) else format_brl(0))
        if len(anom):
            st.dataframe(anom[["FONTE", "PRODUTO", "Preco_Num", "CATEGORIA", "Logistica", "LINK"]],
                         hide_index=True, use_container_width=True)
        else:
            st.success("No anomalies in current filter.")
    else:
        st.info("No anomaly columns available.")

# -----------------------------
# TAB 10 SIMULATOR
# -----------------------------
with tabs[11]:
    st.subheader("Operational simulator (profit per hour)")
    st.caption("Parametric simulator for FDM decisions. Adjust costs and rank best use of machine time.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    custo_hora = col1.number_input("Cost/hour (BRL)", min_value=0.0, value=8.0, step=0.5)
    custo_grama = col2.number_input("Cost/gram (BRL)", min_value=0.0, value=0.12, step=0.01, format="%.2f")
    gramas_base = col3.number_input("Base grams (proxy)", min_value=10, value=60, step=5)
    taxa_falha = col4.number_input("Failure/refugo", min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.2f")
    taxa_market = col5.number_input("Marketplace fee", min_value=0.0, max_value=0.5, value=0.14, step=0.01, format="%.2f")
    embalagem = col6.number_input("Packaging (BRL)", min_value=0.0, value=4.0, step=0.5)

    sim_df = compute_profit(
        df_filtered,
        custo_hora=custo_hora,
        custo_grama=custo_grama,
        gramas_base=gramas_base,
        taxa_falha=taxa_falha,
        taxa_marketplace=taxa_market,
        embalagem=embalagem,
    )

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Avg profit (est)", format_brl(sim_df["Lucro_Estimado"].mean() if len(sim_df) else 0))
    cB.metric("Avg profit/hour", format_brl(sim_df["Lucro_por_Hora"].mean() if len(sim_df) else 0))
    cC.metric("Top profit/hour", format_brl(sim_df["Lucro_por_Hora"].max() if len(sim_df) else 0))
    cD.metric("Negative profit items", int((sim_df["Lucro_Estimado"] < 0).sum()) if len(sim_df) else 0)

    st.markdown("---")
    st.subheader("Top 30 by profit/hour")
    top = sim_df.sort_values("Lucro_por_Hora", ascending=False).head(30).copy()
    top["Price"] = top["Preco_Num"].apply(format_brl)
    top["Profit"] = top["Lucro_Estimado"].apply(format_brl)
    top["Profit/H"] = top["Lucro_por_Hora"].apply(format_brl)
    st.dataframe(top[["FONTE", "PRODUTO", "Price", "Profit", "Profit/H", "Horas_Estimadas", "Gramagem_Estimada", "LINK"]],
                 hide_index=True, use_container_width=True)

# -----------------------------
# TAB 11 RECOMMENDER
# -----------------------------
with tabs[12]:
    st.subheader("Recommender (what to list / produce)")
    st.caption("Combines: Gap finder + underpricing + flash + anomaly penalty.")

    base = df_filtered.copy()
    has_price = ("Delta_Preco" in base.columns) and base["Delta_Preco"].notna().any()

    cluster_score_map = {}
    if gap_df is not None and not gap_df.empty:
        cluster_score_map = dict(zip(gap_df["CLUSTER_MKT"].astype(int), gap_df["score_base"].astype(float)))

    base["cluster_score"] = base["CLUSTER_MKT"].astype(int).map(cluster_score_map).fillna(0.0) if "CLUSTER_MKT" in base.columns else 0.0
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

    rec = base.sort_values("score_rec", ascending=False).head(40).copy()
    rec["Price"] = rec["Preco_Num"].apply(format_brl)
    if has_price:
        rec["Expected"] = rec["Preco_Previsto"].apply(format_brl)
        rec["Delta"] = rec["Delta_Preco"].apply(format_brl)

    cols = ["score_rec", "CLUSTER_NOME", "FONTE", "PRODUTO", "Price", "Logistica", "LINK"]
    if has_price:
        cols += ["Expected", "Delta"]
    st.dataframe(rec[cols], hide_index=True, use_container_width=True)

# -----------------------------
# TAB 12 FORECAST
# -----------------------------
with tabs[13]:
    st.subheader("Forecast (requires date column in CSV)")
    st.caption("If your CSV has a date/time column, this becomes a real time series. Otherwise, it shows a note.")

    date_col = None
    for c in df_raw.columns:
        cu = str(c).upper()
        if any(k in cu for k in ["DATA", "DATE", "DIA", "HORA", "TIMESTAMP"]):
            date_col = c
            break

    if date_col is None:
        st.info("No date/time column found in current CSV. Add a column like Data/Hora to your Sheets to enable forecast.")
    else:
        try:
            tmp = df_raw.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if tmp.empty:
                st.warning("Date column exists but could not parse any values.")
            else:
                tmp["dia"] = tmp[date_col].dt.date
                ts = tmp.groupby("dia")["Preco_Num"].mean().reset_index()
                ts["dia"] = pd.to_datetime(ts["dia"])

                st.plotly_chart(px.line(ts, x="dia", y="Preco_Num", title="Daily avg ticket (observed)"), use_container_width=True)

                ts = ts.sort_values("dia")
                ts["mm7"] = ts["Preco_Num"].rolling(7, min_periods=3).mean()
                coef = np.polyfit(np.arange(len(ts)), ts["Preco_Num"].values, deg=1)
                ts["trend"] = coef[0] * np.arange(len(ts)) + coef[1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["Preco_Num"], mode="lines+markers", name="observed"))
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["mm7"], mode="lines", name="mm7"))
                fig.add_trace(go.Scatter(x=ts["dia"], y=ts["trend"], mode="lines", name="trend"))
                fig.update_layout(title="Ticket: observed vs moving avg vs trend")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Could not generate forecast: " + str(e))
