import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- CONFIGURAÇÃO DE ELITE ---
st.set_page_config(page_title="BCRUZ COMMAND | ORACLE 3D", layout="wide", page_icon="⚡")

# URL ATUALIZADA PELO USUÁRIO
URL_API = "https://script.google.com/macros/s/AKfycbx9ksJ2KMhPwyRaymUoYvAXR2Kvr_bcCUyZT-ICHNF0OwkgxVWm9HPqwQMo24LKz2gn/exec"
INVESTIMENTO_INICIAL = 4200.00

# Estilização Neon-Dark (CORREÇÃO: unsafe_allow_html)
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00FF41; }
    div[data-testid="stMetricValue"] { color: #00FF41; font-family: 'Courier New'; }
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 1px solid #0f0; }
    </style>
    """, unsafe_allow_html=True)

def clean_val(v):
    if not v: return 0.0
    res = re.findall(r"[-+]?\d*\.\d+|\d+", str(v).replace('.', '').replace(',', '.'))
    return float(res[0]) if res else 0.0

@st.cache_data(ttl=60) # Cache de 1 minuto para agilizar o celular
def load_and_analyze():
    try:
        res = requests.get(URL_API, timeout=30)
        df = pd.DataFrame(res.json())
        if df.empty: return df
        df['Preco_Num'] = df['Preço'].apply(clean_val)
        df['Vendas_Num'] = df['Vendas'].apply(clean_val)
        # ID Score: Prioriza alto preço e alto volume
        df['ID_Score'] = (df['Preco_Num'] * df['Vendas_Num']) / (df['Produto'].str.len() + 1)
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_and_analyze()

if not df.empty:
    st.title("⚡ BCRUZ COMMAND CENTER")
    
    tab_roi, tab_radar, tab_seo = st.tabs(["💰 MOTOR DE ROI", "📡 RADAR DE OPORTUNIDADE", "✍️ SEO ALQUIMISTA"])

    with tab_roi:
        st.subheader("Predição de Payback (Bambu A1)")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            lucro_total = st.number_input("Lucro Total Acumulado (R$)", value=0.0, step=50.0)
            meta_dia = st.number_input("Meta de Lucro Diário (R$)", value=50.0, step=10.0)
            faltam = INVESTIMENTO_INICIAL - lucro_total
            dias = int(faltam / meta_dia) if meta_dia > 0 else 0
            
            st.metric("Faltam para Quitar", f"R$ {faltam:.2f}")
            st.info(f"Faltam aproximadamente **{dias} dias** de operação.")
        with col_r2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = lucro_total,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, INVESTIMENTO_INICIAL]},
                         'bar': {'color': "#00FF41"},
                         'steps' : [{'range': [0, 2100], 'color': "#111"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab_radar:
        st.subheader("Clusters de Elite (K-Means)")
        X = df[['Preco_Num', 'Vendas_Num']]
        # Ajuste dinâmico de clusters baseado no volume de dados
        n_clusters = min(3, len(df))
        km = KMeans(n_clusters=n_clusters, n_init=10).fit(X)
        df['Segmento'] = km.labels_
        
        fig_scatter = px.scatter(df, x="Preco_Num", y="Vendas_Num", 
                                 color=df['Segmento'].astype(str), 
                                 size="ID_Score", hover_name="Produto", 
                                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Neon)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_seo:
        st.subheader("SEO Inteligente: Palavras que Vendem")
        top_vendas = df[df['Vendas_Num'] >= df['Vendas_Num'].median()]
        if not top_vendas.empty:
            text = " ".join(top_vendas['Produto']).lower()
            stopwords = ['com', 'para', 'kit', 'envio', '3d', 'promoção', 'shopee', 'elo7', 'de', 'do', 'da']
            wc = WordCloud(width=800, height=400, background_color="black", colormap="Greens", stopwords=stopwords).generate(text)
            fig_wc, ax = plt.subplots()
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.write("Dados insuficientes para gerar Nuvem de SEO.")

    st.markdown("---")
    st.write("### 📋 Base de Dados Consolidada")
    st.dataframe(df[['Produto', 'Preco_Num', 'Vendas_Num', 'Fonte', 'Link']], use_container_width=True)

else:
    st.warning("⚠️ Sistema em espera. Verifique se o robô enviou os dados para as abas da Planilha.")
