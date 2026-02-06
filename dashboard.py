import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(
    page_title="BCRUZ AI Intelligence 3D",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS para deixar com cara de sistema executivo
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .stAlert {font-weight: bold;}
    h1, h2, h3 {color: #0e1117;}
</style>
""", unsafe_allow_html=True)

# --- 2. CARREGAMENTO E DATA SCIENCE ---
# ⚠️⚠️ COLOQUE SEU LINK CSV PUBLICO AQUI ⚠️⚠️
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?output=csv"

@st.cache_data(ttl=300)
def load_data():
    try:
        # Lê o CSV e trata colunas
        df = pd.read_csv(SHEET_CSV_URL, on_bad_lines='skip')
        
        # Limpeza de Colunas (Upper case para garantir)
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 1. Tratamento de Preço
        col_preco = next((c for c in df.columns if "PREÇO" in c or "PRICE" in c), None)
        if col_preco:
            df['Preco_Num'] = df[col_preco].astype(str).str.replace('R$', '', regex=False)
            df['Preco_Num'] = df['Preco_Num'].str.replace('.', '', regex=False).str.replace(',', '.')
            df['Preco_Num'] = pd.to_numeric(df['Preco_Num'], errors='coerce').fillna(0)
        else:
            df['Preco_Num'] = 0.0

        # 2. Tratamento de Prazo (Texto -> Número)
        col_prazo = next((c for c in df.columns if "PRAZO" in c or "FLASH" in c), None)
        if col_prazo:
            df['Prazo_Limpo'] = df[col_prazo].fillna("30 DIAS")
            def get_days(txt):
                txt = str(txt).upper()
                if "IMEDIATO" in txt or "PRONTA" in txt: return 1
                match = re.search(r'(\d+)', txt)
                return int(match.group(1)) if match else 15
            df['Dias_Producao'] = df['Prazo_Limpo'].apply(get_days)
        else:
            df['Dias_Producao'] = 15

        # 3. Engenharia de Recursos (Clusters)
        if len(df) > 5:
            # Agrupa produtos em 3 categorias baseadas em Preço e Prazo
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[['Preco_Num', 'Dias_Producao']])
            # Nomeia os clusters baseado na média de preço
            medias = df.groupby('Cluster')['Preco_Num'].mean().sort_values()
            mapa = {medias.index[0]: 'Econômico', medias.index[1]: 'Padrão', medias.index[2]: 'Premium'}
            df['Segmento'] = df['Cluster'].map(mapa)
        else:
            df['Segmento'] = 'Geral'

        return df
    except Exception as e:
        st.error(f"Erro no Data Science: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")
st.sidebar.markdown("---")

if not df.empty:
    # Filtros
    col_cat = next((c for c in df.columns if "CATEG" in c), "CATEGORIA")
    cats = st.sidebar.multiselect("Filtrar Nicho", df[col_cat].unique())
    if cats: df = df[df[col_cat].isin(cats)]
    
    st.sidebar.markdown("### Configurações da Bambu A1")
    custo_material = st.sidebar.number_input("Custo Material/Hora (R$)", value=10.0, step=1.0)
    margem_desejada = st.sidebar.slider("Margem de Lucro (%)", 10, 100, 30)

# --- 4. DASHBOARD PRINCIPAL ---
st.title("🚀 BCRUZ 3D: Sistema de Inteligência de Mercado")

if df.empty:
    st.warning("⚠️ Carregando dados... Se demorar, verifique o link CSV.")
else:
    # ABAS DO SISTEMA
    tab_kpi, tab_ai, tab_gen = st.tabs(["📊 Visão de Mercado", "🤖 Análise Avançada (IA)", "✨ Gerador de Anúncio Perfeito"])

    # --- ABA 1: VISÃO DE MERCADO ---
    with tab_kpi:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Concorrentes Mapeados", len(df))
        c2.metric("Preço Médio Mercado", f"R$ {df['Preco_Num'].mean():.2f}")
        c3.metric("Tempo Médio Entrega", f"{df['Dias_Producao'].mean():.1f} dias")
        
        # Oportunidade: Preço acima da média + Prazo Lento
        opp = len(df[(df['Preco_Num'] > df['Preco_Num'].mean()) & (df['Dias_Producao'] > 5)])
        c4.metric("🚨 Oportunidades Claras", opp, help="Produtos caros e lentos. Ataque aqui!")

        st.markdown("---")
        
        g1, g2 = st.columns(2)
        with g1:
            fig = px.scatter(df, x="Dias_Producao", y="Preco_Num", color="Segmento", size="Preco_Num",
                             hover_data=["PRODUTO"], title="Matriz de Valor: Tempo x Preço",
                             labels={"Dias_Producao": "Dias para Produzir", "Preco_Num": "Preço de Venda"})
            st.plotly_chart(fig, use_container_width=True)
        
        with g2:
            # Gráfico de Rosca de Prazos
            df['Tipo_Entrega'] = df['Dias_Producao'].apply(lambda x: "⚡ Imediato" if x <= 2 else ("📦 Normal" if x <= 10 else "🐢 Lento"))
            fig2 = px.pie(df, names="Tipo_Entrega", title="Market Share Logístico", hole=0.4, 
                          color_discrete_map={"⚡ Imediato": "#00FF00", "📦 Normal": "#FFA500", "🐢 Lento": "#FF0000"})
            st.plotly_chart(fig2, use_container_width=True)

    # --- ABA 2: INTELIGÊNCIA ARTIFICIAL ---
    with tab_ai:
        st.subheader("🧠 Regressão Linear & Tendências")
        st.info("A linha vermelha mostra se o cliente paga mais pela espera (personalização) ou se prefere rapidez.")
        
        # Regressão Linear
        X = df[['Dias_Producao']]
        y = df['Preco_Num']
        reg = LinearRegression().fit(X, y)
        df['Tendencia'] = reg.predict(X)
        
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=df['Dias_Producao'], y=df['Preco_Num'], mode='markers', name='Mercado'))
        fig_reg.add_trace(go.Scatter(x=df['Dias_Producao'], y=df['Tendencia'], mode='lines', name='Tendência IA', line=dict(color='red', width=3)))
        st.plotly_chart(fig_reg, use_container_width=True)
        
        # Nuvem de Palavras
        st.subheader("☁️ O que vende? (Nuvem de Palavras)")
        try:
            texto = " ".join(df['PRODUTO'].dropna().astype(str))
            # Stopwords simples em PT
            stops = ["de", "do", "da", "para", "com", "em", "um", "uma", "o", "a", "e", "kit", "3d", "pla", "impressão", "impresso"]
            wc = WordCloud(width=800, height=300, background_color='white', stopwords=stops).generate(texto)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        except:
            st.write("Dados de texto insuficientes.")

    # --- ABA 3: O GERADOR DE PRODUTOS DO PRESIDENTE (A JÓIA DA COROA) ---
    with tab_gen:
        st.markdown("### 🛠️ Criador de Anúncio Campeão")
        st.markdown("Digite o que você quer vender. A IA vai analisar os concorrentes e calcular o Preço e o Título ideal.")
        
        termo_busca = st.text_input("Qual produto você vai imprimir?", placeholder="Ex: Vaso Robert, Suporte Fone...")
        
        if termo_busca:
            # 1. Filtra a base para encontrar similares
            df_similar = df[df['PRODUTO'].str.contains(termo_busca, case=False, na=False)]
            
            if len(df_similar) > 0:
                st.success(f"🔎 Encontrei {len(df_similar)} concorrentes vendendo '{termo_busca}'. Analisando...")
                
                # --- ANÁLISE DE PALAVRAS-CHAVE (NLP) ---
                try:
                    # Vetorização para achar n-grams (palavras compostas) mais usadas
                    vec = CountVectorizer(ngram_range=(2, 2), stop_words=["de", "do", "para", "com", "em", "3d"])
                    X_vec = vec.fit_transform(df_similar['PRODUTO'])
                    sum_words = X_vec.sum(axis=0)
                    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                    top_keywords = [w[0].title() for w in words_freq[:3]] # Top 3 combinações
                except:
                    top_keywords = ["Decorativo", "Personalizado", "Exclusivo"]

                # --- CÁLCULO DE PREÇO ---
                preco_medio = df_similar['Preco_Num'].mean()
                preco_max = df_similar['Preco_Num'].quantile(0.90) # Top 10% mais caros
                preco_sugerido = preco_medio * 1.1 # 10% acima da média pois temos envio imediato
                
                # --- GERADOR DE TÍTULO ---
                palavras_poder = " ".join(top_keywords)
                titulo_gerado = f"{termo_busca.title()} {palavras_poder} - Pronta Entrega"
                
                # --- EXIBIÇÃO DO RESULTADO ---
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown("#### 🏆 Título Sugerido pela IA")
                    st.code(titulo_gerado, language="text")
                    st.caption("Baseado nas palavras mais frequentes dos concorrentes de sucesso.")
                    
                    st.markdown("#### 🏷️ Tags Recomendadas")
                    st.write(", ".join([f"#{k.replace(' ', '')}" for k in top_keywords] + ["#ProntaEntrega", "#3DPrinting"]))

                with col_res2:
                    st.markdown("#### 💲 Sugestão de Preço")
                    st.metric("Preço de Venda Ideal", f"R$ {preco_sugerido:.2f}", 
                              delta=f"R$ {preco_sugerido - custo_material:.2f} (Lucro Est.)")
                    
                    st.info(f"O mercado varia de R$ {df_similar['Preco_Num'].min():.2f} até R$ {preco_max:.2f}.")
                    
                # Mostra os concorrentes usados na base
                with st.expander(f"Ver os {len(df_similar)} concorrentes analisados"):
                    st.dataframe(df_similar[['PRODUTO', 'Preco_Num', 'Prazo_Limpo', 'LINK']])
                    
            else:
                st.warning(f"Não encontrei concorrentes diretos para '{termo_busca}' na base atual. Tente um termo mais genérico.")
