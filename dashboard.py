import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="BCRUZ AI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONEXÃO DE DADOS ---
# ⚠️ COLOQUE SEU LINK CSV AQUI (Arquivo > Compartilhar > Publicar na Web > CSV)
SHEET_CSV_URL = "https://script.google.com/macros/s/AKfycbzz1kNMVd7wDkem6Vrdb1v1sUyWyekpUWe8Dd-dI4VxgLqpFhJe9DmE6486apJ97dy6/exec"

@st.cache_data(ttl=60)
def carregar_dados():
    try:
        # Lê o CSV ignorando erros de linha
        df = pd.read_csv(SHEET_CSV_URL, on_bad_lines='skip')
        
        # --- LIMPEZA E PADRONIZAÇÃO (O SEGREDO DA IA) ---
        
        # 1. Preço (R$ 1.200,00 -> 1200.00)
        col_preco = 'PREÇO (R$)'
        if col_preco in df.columns:
            df['Preco_Num'] = df[col_preco].astype(str).str.replace('R$', '', regex=False)
            df['Preco_Num'] = df['Preco_Num'].str.replace('.', '', regex=False).str.replace(',', '.')
            df['Preco_Num'] = pd.to_numeric(df['Preco_Num'], errors='coerce').fillna(0)

        # 2. Prazo (Transformar texto em número para Regressão Linear)
        col_prazo = 'PRAZO DE PRODUÇÃO'
        if col_prazo in df.columns:
            df[col_prazo] = df[col_prazo].fillna("30 DIAS") # Assume pior caso se vazio
            
            def extrair_dias(texto):
                texto = str(texto).upper()
                if "IMEDIATO" in texto or "PRONTA" in texto:
                    return 1 # Envio imediato = 1 dia para fins matemáticos
                match = re.search(r'(\d+)', texto)
                if match:
                    return int(match.group(1))
                return 15 # Média padrão se não achar número

            df['Dias_Producao'] = df[col_prazo].apply(extrair_dias)
            
            # Cria a Flag de Ataque (Flash)
            df['Tipo_Logistica'] = df['Dias_Producao'].apply(
                lambda x: "⚡ FLASH (Até 2 dias)" if x <= 2 else ("🐢 LENTO (>7 dias)" if x > 7 else "📦 NORMAL")
            )

        # 3. Lucro Estimado (Simulação Presidencial)
        # Custo estimado de material + energia para peças médias (ajustável)
        CUSTO_MEDIO = 15.00 
        df['Lucro_Potencial'] = df['Preco_Num'] - CUSTO_MEDIO
        
        return df

    except Exception as e:
        st.error(f"Erro Crítico ao carregar dados: {e}")
        return pd.DataFrame()

# --- CARREGA DADOS ---
df = carregar_dados()

# --- SIDEBAR DE COMANDO ---
st.sidebar.title("🎛️ Centro de Comando")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)

if not df.empty:
    # Filtros
    categorias = st.sidebar.multiselect("Filtrar Categoria", df['CATEGORIA'].unique())
    if categorias:
        df = df[df['CATEGORIA'].isin(categorias)]
        
    preco_min, preco_max = st.sidebar.slider("Faixa de Preço (R$)", 
                                             float(df['Preco_Num'].min()), 
                                             float(df['Preco_Num'].max()), 
                                             (float(df['Preco_Num'].min()), float(df['Preco_Num'].max())))
    df = df[(df['Preco_Num'] >= preco_min) & (df['Preco_Num'] <= preco_max)]

# --- CORPO DO DASHBOARD ---
st.title("🚀 BCRUZ 3D: Market Intelligence v35.0")
st.markdown("### Análise Avançada para Dominação de Nicho com Bambu Lab A1")

if df.empty:
    st.warning("⚠️ Aguardando dados do Robô. Verifique se o script Python rodou.")
else:
    # ---------------------------------------------------------
    # ABA 1: VISÃO GERAL (KPIs)
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 KPIs & Mercado", "🤖 IA & Clustering", "📝 NLP & Tags", "🎯 Lista de Ataque"])
    
    with tab1:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Produtos Mapeados", len(df), delta="Total da Base")
        kpi2.metric("Preço Médio", f"R$ {df['Preco_Num'].mean():.2f}", delta="Ticket Médio")
        kpi3.metric("Tempo Médio Produção", f"{df['Dias_Producao'].mean():.1f} dias", delta="- Eficiência", delta_color="inverse")
        
        # Métrica de Oportunidade: Produtos caros e lentos
        oportunidades = df[(df['Dias_Producao'] >= 7) & (df['Preco_Num'] > df['Preco_Num'].mean())]
        kpi4.metric("💎 Minas de Ouro", len(oportunidades), help="Produtos caros (> média) e lentos (> 7 dias)")

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 Distribuição de Preços por Logística")
            fig_box = px.box(df, x="Tipo_Logistica", y="Preco_Num", color="Tipo_Logistica", 
                             title="Quem cobra mais? Rápido vs Lento",
                             color_discrete_map={"⚡ FLASH (Até 2 dias)": "#00FF41", "🐢 LENTO (>7 dias)": "#FF2B2B", "📦 NORMAL": "#FFA500"})
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col2:
            st.subheader("🧩 Share de Categorias")
            fig_pie = px.pie(df, names="CATEGORIA", values="Preco_Num", title="Faturamento Estimado por Nicho")
            st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------------------------------------------------
    # ABA 2: MACHINE LEARNING (K-MEANS & REGRESSÃO)
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 🧠 Inteligência Artificial Aplicada")
        
        col_ia1, col_ia2 = st.columns(2)
        
        with col_ia1:
            st.subheader("🔍 K-Means Clustering (Agrupamento)")
            st.info("A IA agrupou os produtos em 3 perfis automáticos.")
            
            # Prepara dados para o modelo
            X = df[['Preco_Num', 'Dias_Producao']].dropna()
            
            if len(X) > 5:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(X)
                
                # Mapeia nomes amigáveis para os clusters
                # (Lógica simplificada: ordena clusters pelo preço médio)
                cluster_means = df.groupby('Cluster')['Preco_Num'].mean().sort_values()
                cluster_map = {
                    cluster_means.index[0]: "Barato / Entrada",
                    cluster_means.index[1]: "Intermediário",
                    cluster_means.index[2]: "Premium / Luxo"
                }
                df['Segmento_IA'] = df['Cluster'].map(cluster_map)
                
                fig_cluster = px.scatter(
                    df, x="Dias_Producao", y="Preco_Num", color="Segmento_IA",
                    hover_data=['PRODUTO'], size="Preco_Num",
                    title="Segmentação Automática de Mercado",
                    labels={"Dias_Producao": "Dias para Produzir", "Preco_Num": "Preço (R$)"}
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Dados insuficientes para IA.")

        with col_ia2:
            st.subheader("📈 Regressão Linear: Tempo vale Dinheiro?")
            st.info("A linha vermelha mostra a tendência: O preço sobe quando o prazo aumenta?")
            
            if len(df) > 5:
                # Modelo Linear
                X_reg = df[['Dias_Producao']]
                y_reg = df['Preco_Num']
                reg = LinearRegression().fit(X_reg, y_reg)
                
                df['Tendencia_Preco'] = reg.predict(X_reg)
                
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=df['Dias_Producao'], y=df['Preco_Num'], mode='markers', name='Produtos'))
                fig_reg.add_trace(go.Scatter(x=df['Dias_Producao'], y=df['Tendencia_Preco'], mode='lines', name='Tendência (IA)', line=dict(color='red')))
                fig_reg.update_layout(title="Correlação: Prazo x Preço", xaxis_title="Dias de Produção", yaxis_title="Preço (R$)")
                
                st.plotly_chart(fig_reg, use_container_width=True)
                
                coef = reg.coef_[0]
                if coef > 0:
                    st.success(f"💡 Insight: Para cada dia a mais de produção, o preço sobe em média R$ {coef:.2f}.")
                else:
                    st.error(f"💡 Insight: O mercado não está pagando mais pela demora (Coef: {coef:.2f}). A Pronta Entrega vai destruir a concorrência.")

    # ---------------------------------------------------------
    # ABA 3: NLP & NUVEM DE PALAVRAS
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### 🗣️ O que o mercado está falando?")
        st.info("As palavras maiores são as mais usadas nos títulos dos concorrentes.")
        
        texto_completo = " ".join(df['PRODUTO'].astype(str))
        
        # Stopwords (palavras para ignorar)
        stopwords_pt = set(STOPWORDS)
        stopwords_pt.update(["de", "da", "do", "para", "com", "em", "um", "uma", "o", "a", "e", "kit", "personalizado", "3d", "pla", "impressão"])
        
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_pt, colormap='viridis').generate(texto_completo)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            st.markdown("#### 💡 Dica de SEO:")
            st.write("Use essas palavras-chave nos seus títulos para aparecer nas buscas onde seus concorrentes já estão.")
            
        except Exception as e:
            st.error("Poucos dados de texto para gerar nuvem.")

    # ---------------------------------------------------------
    # ABA 4: LISTA DE ATAQUE (SNIPER)
    # ---------------------------------------------------------
    with tab4:
        st.subheader("🎯 Tabela Tática de Oportunidades")
        st.markdown("Filtro automático: Produtos com **Prazo > 5 dias** e **Preço > R$ 50,00**.")
        
        # Filtro Sniper
        df_sniper = df[(df['Dias_Producao'] > 5) & (df['Preco_Num'] > 50)].sort_values(by="Dias_Producao", ascending=False)
        
        st.dataframe(
            df_sniper[['PRODUTO', 'PREÇO (R$)', 'PRAZO DE PRODUÇÃO', 'Lucro_Potencial', 'LINK']],
            column_config={
                "LINK": st.column_config.LinkColumn("Link Elo7"),
                "PREÇO (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                "Lucro_Potencial": st.column_config.ProgressColumn("Margem Estimada", format="R$ %.2f", min_value=0, max_value=200)
            },
            hide_index=True,
            use_container_width=True
        )
