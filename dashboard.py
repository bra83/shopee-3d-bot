import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, re, numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

st.set_page_config(page_title="BCRUZ | ORACLE 3D", layout="wide")
URL_API = "https://script.google.com/macros/s/AKfycbx9ksJ2KMhPwyRaymUoYvAXR2Kvr_bcCUyZT-ICHNF0OwkgxVWm9HPqwQMo24LKz2gn/exec"
INVESTIMENTO = 4200.0

st.markdown("<style>.stApp { background-color: #000; color: #0f0; }</style>", unsafe_allow_html=True)

def clean_val(v):
    if not v: return 0.0
    res = re.findall(r"[-+]?\d*\.\d+|\d+", str(v).replace('.','').replace(',','.'))
    return float(res[0]) if res else 0.0

@st.cache_data(ttl=60)
def load():
    try:
        r = requests.get(URL_API, timeout=30)
        df = pd.DataFrame(r.json())
        df['Preco_N'] = df['Preço'].apply(clean_val)
        df['Vendas_N'] = df['Vendas'].apply(clean_val)
        df['Score'] = (df['Preco_N'] * 0.7) + (df['Vendas_N'] * 0.3)
        return df
    except: return pd.DataFrame()

df = load()

if not df.empty:
    st.title("🏹 BCRUZ COMMAND: ESTRATÉGIA DE ALTA MARGEM")
    t1, t2, t3, t4 = st.tabs(["💰 PAYBACK", "📡 RADAR SNIPER", "✍️ SEO MASTER", "⚙️ FILA PPH"])

    with t1:
        c1, c2 = st.columns([1,2])
        ja_ganho = c1.number_input("Lucro Total (R$)", value=0.0)
        meta_dia = c1.number_input("Meta Diária (R$)", value=50.0)
        restante = INVESTIMENTO - ja_ganho
        c1.metric("Falta", f"R$ {restante:.2f}")
        c1.info(f"Payback em: {int(restante/meta_dia) if meta_dia > 0 else 0} dias")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=ja_ganho, gauge={'axis':{'range':[None,INVESTIMENTO]}, 'bar':{'color':"#0f0"}}))
        st.plotly_chart(fig_g)

    with t2:
        X = df[['Preco_N', 'Vendas_N']]
        km = KMeans(n_clusters=min(3, len(df)), n_init=10).fit(X)
        df['Perfil'] = km.labels_
        df['Status'] = df['Perfil'].map({0:"Saturado", 1:"Oportunidade", 2:"ELITE"})
        fig = px.scatter(df, x="Preco_N", y="Vendas_N", color="Status", size="Score", hover_name="Produto", template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        text = " ".join(df[df['Vendas_N']>0]['Produto']).lower()
        sw = ['para','com','kit','3d','shopee','elo7','envio']
        wc = WordCloud(width=800, height=400, background_color="black", colormap="Greens", stopwords=sw).generate(text)
        fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig_wc)
        # Sugestão SEO
        comuns = Counter([w for w in re.findall(r'\w+', text) if len(w)>3 and w not in sw]).most_common(5)
        st.success(f"Sugestão SEO: {' '.join([k[0].upper() for k in comuns])} - QUALIDADE PREMIUM")

    with t4:
        st.subheader("Simulador de Lucro por Hora ($PPH$)")
        col_p1, col_p2 = st.columns(2)
        tempo = col_p1.number_input("Tempo de Impressão (Horas)", value=2.0)
        venda = col_p1.number_input("Preço de Venda (R$)", value=80.0)
        peso = col_p1.number_input("Peso (Gramas)", value=100)
        lucro = venda - (venda*0.18) - ((peso/1000)*95) - (tempo*0.6)
        col_p2.metric("Lucro Líquido", f"R$ {lucro:.2f}")
        col_p2.metric("PPH (Renda/Hora)", f"R$ {lucro/tempo:.2f}")

    st.dataframe(df[['Status','Fonte','Produto','Preco_N','Vendas_N','Link']], use_container_width=True)
