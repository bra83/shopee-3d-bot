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
from collections import Counter # <-- CORREÇÃO: O IMPORT QUE FALTA

# --- CONFIGURAÇÃO BCRUZ COMMAND ---
st.set_page_config(page_title="BCRUZ | ORACLE 3D v8.0", layout="wide", page_icon="📈")

URL_API = "https://script.google.com/macros/s/AKfycbx9ksJ2KMhPwyRaymUoYvAXR2Kvr_bcCUyZT-ICHNF0OwkgxVWm9HPqwQMo24LKz2gn/exec"
INVESTIMENTO_A1 = 4200.00

# CSS Dark-Mode BCRUZ
st.markdown("""
    <style>
    .stApp { background-color: #000; color: #0f0; }
    [data-testid="stMetricValue"] { color: #0f0 !important; font-family: 'Courier New', monospace; }
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 2px solid #0f0; }
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
        # Score de Potencial: Onde o mercado paga mais por unidade
        df['Potencial'] = (df['Preco_N'] * 0.7) + (df['Vendas_N'] * 0.3)
        return df
    except: return pd.DataFrame()

df = load_data()

if not df.empty:
    st.title("🏹 BCRUZ INTELLIGENCE | SISTEMA DE ALTA MARGEM")
    
    tab_payback, tab_sniper, tab_seo = st.tabs(["💰 PAYBACK TRACKER", "🎯 RADAR SNIPER", "✍️ SEO & PALAVRAS"])

    with tab_payback:
        st.subheader("Cronômetro de Quitação: Bambu Lab A1")
        c1, c2, c3 = st.columns(3)
        ganho_total = c1.number_input("Total Ganho até Hoje (R$)", value=0.0)
        lucro_esperado = c2.number_input("Expectativa de Lucro/Dia (R$)", value=100.0)
        
        restante = INVESTIMENTO_A1 - ganho_total
        dias_payback = int(restante / lucro_esperado) if lucro_esperado > 0 else 0
        
        c3.metric("Falta para Zerar", f"R$ {restante:.2f}", delta=f"{ (ganho_total/INVESTIMENTO_A1)*100 :.1f}%")
        
        fig_roi = go.Figure(go.Indicator(
            mode = "gauge+number", value = ganho_total,
            title = {'text': "Progresso do Payback"},
            gauge = {'axis': {'range': [None, INVESTIMENTO_A1]}, 'bar': {'color': "#0f0"}}))
        st.plotly_chart(fig_roi, use_container_width=True)
        st.info(f"🚀 **Meta de Decisão:** Com lucro de R$ {lucro_esperado}/dia, sua máquina se paga em **{dias_payback} dias**.")

    with tab_sniper:
        st.subheader("Localizador de Oportunidades (Clusterização)")
        # ML K-Means para segmentar o mercado
        X = df[['Preco_N', 'Vendas_N']]
        n_clusters = min(3, len(df))
        km = KMeans(n_clusters=n_clusters, n_init=10).fit(X)
        df['Cluster'] = km.labels_
        df['Perfil'] = df['Cluster'].map({0: "Saturado (Fugir)", 1: "Potencial", 2: "Elite/Caminho Livre"})
        
        fig_scatter = px.scatter(df, x="Preco_N", y="Vendas_N", color="Perfil", size="Potencial",
                                 hover_name="Produto", template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.success("Analise os círculos da 'Elite': São produtos com alto preço e volume de vendas real.")

    with tab_seo:
        st.subheader("SEO Alquimista: Engenharia de Títulos")
        # Analisa o Top 30% dos anúncios com mais vendas
        top_vendas = df[df['Vendas_N'] >= df['Vendas_N'].quantile(0.7)]
        if not top_vendas.empty:
            text = " ".join(top_vendas['Produto']).lower()
            sw = ['para', 'com', 'kit', 'envio', '3d', 'promoção', 'shopee', 'elo7', 'de', 'do', 'da']
            
            col_wc, col_tips = st.columns(2)
            with col_wc:
                wc = WordCloud(width=800, height=400, background_color="black", colormap="Greens", stopwords=sw).generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig_wc)
            
            with col_tips:
                # Extração Automática de Títulos
                tokens = [w for w in re.findall(r'\w+', text) if len(w) > 3 and w not in sw]
                mais_comuns = Counter(tokens).most_common(5)
                st.write("### 💎 Top Palavras-Chave")
                for p, f in mais_comuns:
                    st.write(f"- **{p.upper()}**")
                
                sugestao = " ".join([k[0].upper() for k in mais_comuns])
                st.info(f"**Sugestão BCRUZ:** {sugestao} - ALTA QUALIDADE REFORÇADO")
        else:
            st.warning("Aguardando mais dados para gerar inteligência de SEO.")

    st.markdown("---")
    st.write("### 📑 Base de Dados Operacional")
    st.dataframe(df[['Perfil', 'Fonte', 'Produto', 'Preco_N', 'Vendas_N', 'Link']], use_container_width=True)

else:
    st.error("Sem dados detectados. Rode o robô para alimentar a planilha.")
