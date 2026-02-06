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

# --- CONFIGURAÇÃO BCRUZ COMMAND ---
st.set_page_config(page_title="BCRUZ | ORACLE 3D v7.0", layout="wide", page_icon="⚡")

# URL DO GOOGLE APPS SCRIPT
URL_API = "https://script.google.com/macros/s/AKfycbx9ksJ2KMhPwyRaymUoYvAXR2Kvr_bcCUyZT-ICHNF0OwkgxVWm9HPqwQMo24LKz2gn/exec"
INVESTIMENTO_A1 = 4200.00

# CSS customizado para manter a vibe Dark/Neon
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00FF41; }
    [data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Courier New', monospace; }
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 2px solid #00FF41; }
    .stDataFrame { border: 1px solid #00FF41; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

def clean_val(v):
    if not v: return 0.0
    res = re.findall(r"[-+]?\d*\.\d+|\d+", str(v).replace('.', '').replace(',', '.'))
    return float(res[0]) if res else 0.0

@st.cache_data(ttl=60)
def load_data():
    try:
        res = requests.get(URL_API, timeout=30)
        df = pd.DataFrame(res.json())
        if df.empty: return df
        df['Preco_N'] = df['Preço'].apply(clean_val)
        df['Vendas_N'] = df['Vendas'].apply(clean_val)
        df['Data_DT'] = pd.to_datetime(df['Data'], errors='coerce')
        # Score de Oportunidade BCRUZ
        df['Score'] = (df['Preco_N'] * df['Vendas_N']) / (df['Produto'].str.len() + 1)
        return df
    except: return pd.DataFrame()

df = load_data()

if not df.empty:
    st.title("🏹 BCRUZ INTELLIGENCE: DOMINAÇÃO SHOPEE & ELO7")
    
    tab_roi, tab_mercado, tab_seo, tab_tendencia = st.tabs(["💰 PAYBACK A1", "📡 RADAR DE ELITE", "✍️ SEO MASTER", "📈 TENDÊNCIAS"])

    with tab_roi:
        st.subheader("Calculadora de Quitação Acelerada")
        c1, c2, c3 = st.columns(3)
        lucro_ja = c1.number_input("Lucro Acumulado (R$)", value=0.0)
        meta_dia = c2.number_input("Meta Diária de Lucro (R$)", value=70.0)
        
        faltam = INVESTIMENTO_A1 - lucro_ja
        dias_restantes = int(faltam / meta_dia) if meta_dia > 0 else 0
        
        c3.metric("Status do Investimento", f"R$ {faltam:.2f}", delta=f"-{ (lucro_ja/INVESTIMENTO_A1)*100 :.1f}%")
        
        fig_roi = go.Figure(go.Indicator(
            mode = "gauge+number", value = lucro_ja,
            title = {'text': "Progresso da Bambu Lab A1"},
            gauge = {'axis': {'range': [None, INVESTIMENTO_A1]}, 'bar': {'color': "#00FF41"}}))
        st.plotly_chart(fig_roi, use_container_width=True)
        st.success(f"Faltam aproximadamente **{dias_restantes} dias** para a sua A1 se pagar totalmente.")

    with tab_mercado:
        st.subheader("Análise de Clusters (Onde o Caminho está Livre)")
        # ML: K-Means para separar o joio do trigo
        X = df[['Preco_N', 'Vendas_N']]
        n_clusters = min(3, len(df))
        km = KMeans(n_clusters=n_clusters, n_init=10).fit(X)
        df['Cluster'] = km.labels_
        df['Perfil'] = df['Cluster'].map({0: "Saturado/Barato", 1: "Oportunidade", 2: "Elite/Premium"})
        
        fig_scatter = px.scatter(df, x="Preco_N", y="Vendas_N", color="Perfil", size="Score",
                                 hover_name="Produto", template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Vivid) # CORRIGIDO AQUI
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_seo:
        st.subheader("SEO Alquimista: Títulos que Vendem")
        top_vendas = df[df['Vendas_N'] >= df['Vendas_N'].median()]
        if not top_vendas.empty:
            text = " ".join(top_vendas['Produto']).lower()
            sw = ['com', 'para', 'kit', 'envio', '3d', 'promoção', 'shopee', 'elo7', 'da', 'do', 'de']
            wc = WordCloud(width=800, height=350, background_color="black", colormap="Greens", stopwords=sw).generate(text)
            fig_wc, ax = plt.subplots()
            ax.imshow(wc); ax.axis("off")
            st.pyplot(fig_wc)
            
            # Sugestão de Título
            palavras_chave = Counter([w for w in re.findall(r'\w+', text) if len(w) > 3 and w not in sw]).most_common(5)
            sugestao = " ".join([p[0].upper() for p in palavras_chave])
            st.info(f"🚀 **Sugestão de Título Baseada em Dados:** {sugestao} - ALTA RESISTÊNCIA PREMIUM")

    with tab_tendencia:
        st.subheader("Análise de Saturação Temporal")
        df_tend = df.sort_values('Data_DT')
        if len(df_tend) > 1:
            fig_line = px.line(df_tend, x="Data_DT", y="Preco_N", color="Categoria", 
                               title="Variação de Preço no Tempo (Se cair, o mercado saturou!)",
                               template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Aguardando mais coletas do robô para gerar histórico de tendência.")

    st.markdown("---")
    st.write("### 📑 Comando Operacional (Tabela Completa)")
    st.dataframe(df[['Perfil', 'Fonte', 'Produto', 'Preco_N', 'Vendas_N', 'Link']], use_container_width=True)

else:
    st.warning("⚠️ O 'Oracle' está sem dados. Rode o robô no PC para alimentar a planilha.")
