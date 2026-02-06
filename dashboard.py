import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- ENGINE DE INTELIGÊNCIA ---
st.set_page_config(page_title="BCRUZ COMMAND | ELITE 3D", layout="wide")

URL_API = "https://script.google.com/macros/s/AKfycbx9ksJ2KMhPwyRaymUoYvAXR2Kvr_bcCUyZT-ICHNF0OwkgxVWm9HPqwQMo24LKz2gn/exec"
META_INVESTIMENTO = 4200.00

def clean_val(v):
    if not v: return 0.0
    res = re.findall(r"[-+]?\d*\.\d+|\d+", str(v).replace('.', '').replace(',', '.'))
    return float(res[0]) if res else 0.0

@st.cache_data(ttl=300)
def get_data():
    try:
        res = requests.get(URL_API, timeout=30)
        df = pd.DataFrame(res.json())
        df['Preco_N'] = df['Preço'].apply(clean_val)
        df['Vendas_N'] = df['Vendas'].apply(clean_val)
        return df
    except: return pd.DataFrame()

# --- INTERFACE NEON ---
st.markdown("<style>.stApp { background-color: #000; color: #0f0; }</style>", unsafe_allow_stdio=True)
st.title("⚡ SISTEMA DE DOMINAÇÃO BCRUZ 3D")

df = get_data()

if not df.empty:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 ARBITRAGEM", "✍️ SEO & GAP", "⚙️ FILA INTELIGENTE", "📈 ROI PREDITIVO"])

    # --- TAB 1: ARBITRAGEM (SHOPEE VS ELO7) ---
    with tab1:
        st.subheader("Onde vender? (Diferencial de Margem)")
        # Compara onde o preço é mais alto para a mesma categoria
        fig_comp = px.box(df, x="Fonte", y="Preco_N", color="Fonte", points="all", template="plotly_dark")
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- TAB 2: SEO & GAP ---
    with tab2:
        st.subheader("Localizador de Vácuos de Mercado")
        fonte_sel = st.selectbox("Escolha a Fonte para analisar SEO", df['Fonte'].unique())
        texto = " ".join(df[df['Fonte']==fonte_sel]['Produto']).lower()
        wc = WordCloud(width=1000, height=400, background_color="black", colormap="Greens").generate(texto)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc); ax.axis("off")
        st.pyplot(fig_wc)

    # --- TAB 3: FILA DE IMPRESSÃO INTELIGENTE ---
    with tab3:
        st.subheader("Optimizador de Produção (A1 Fast-Track)")
        st.write("Insira os dados técnicos do Bambu Studio para comparar rentabilidade:")
        
        col1, col2, col3, col4 = st.columns(4)
        prod_nome = col1.text_input("Nome do Produto", "Suporte Elite")
        prod_tempo = col2.number_input("Tempo (Horas)", 1.0)
        prod_peso = col3.number_input("Peso (Gramas)", 50)
        prod_preco = col4.number_input("Preço Alvo (R$)", 80.0)

        custo_filamento = (prod_peso / 1000) * 95.0 # R$ 95/kg
        lucro_liquo = prod_preco - custo_filamento - (prod_preco * 0.18) # Taxas
        pph = lucro_liquo / prod_tempo

        st.metric("Lucro por Hora (PPH)", f"R$ {pph:.2f}")
        if pph > 25: st.success("✅ PRIORIDADE MÁXIMA: Alta eficiência de tempo.")

    # --- TAB 4: ROI PREDITIVO (MONTE CARLO) ---
    with tab4:
        st.subheader("Simulação de Quitação: Bambu Lab A1")
        vendas_dia = st.slider("Expectativa de Vendas Diárias", 1, 20, 3)
        margem_media = st.number_input("Margem Média por Peça (R$)", 40.0)
        
        # Simulação de 1000 cenários
        dias_sim = []
        for _ in range(1000):
            # Variação de 20% para cima ou para baixo nas vendas
            vendas_random = np.random.normal(vendas_dia, vendas_dia * 0.2)
            dias_sim.append(META_INVESTIMENTO / (max(1, vendas_random) * margem_media))
        
        previsao = int(np.mean(dias_sim))
        st.metric("Dias Estimados para Payback", f"{previsao} dias")
        fig_roi = px.histogram(dias_sim, nbins=50, title="Probabilidade de Quitação (Cenários)", template="plotly_dark")
        st.plotly_chart(fig_roi, use_container_width=True)

else:
    st.warning("⚠️ Transmissão interrompida. Verifique o Google Sheets.")
