import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from thefuzz import process, fuzz # Biblioteca nova para comparação
import re

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# ⚠️ SEU LINK CSV AQUI
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?output=csv"

@st.cache_data(ttl=60)
def carregar_dados():
    try:
        df = pd.read_csv(SHEET_CSV_URL, on_bad_lines='skip')
        
        # --- LIMPEZA ROBUSTA ---
        # 1. Normaliza colunas
        df.columns = [c.strip().upper() for c in df.columns]
        
        # 2. Identifica Preço
        col_preco = next((c for c in df.columns if "PREÇO" in c or "PRICE" in c), None)
        if col_preco:
            df['Preco_Num'] = df[col_preco].astype(str).str.replace('R$', '', regex=False)
            df['Preco_Num'] = df['Preco_Num'].str.replace('.', '', regex=False).str.replace(',', '.')
            df['Preco_Num'] = pd.to_numeric(df['Preco_Num'], errors='coerce').fillna(0)
        else:
            df['Preco_Num'] = 0.0

        # 3. Identifica Categoria e Fonte
        if 'CATEGORIA' not in df.columns: df['CATEGORIA'] = 'Geral'
        if 'FONTE' not in df.columns: df['FONTE'] = 'Elo7' # Default

        # 4. Tratamento Inteligente de Prazo
        col_prazo = next((c for c in df.columns if "PRAZO" in c), None)
        if col_prazo:
            df['Prazo_Original'] = df[col_prazo].fillna("30 DIAS")
            def extrair_dias(texto):
                texto = str(texto).upper()
                if "IMEDIATO" in texto or "PRONTA" in texto: return 1
                match = re.search(r'(\d+)', texto)
                return int(match.group(1)) if match else 15
            
            df['Dias_Producao'] = df['Prazo_Original'].apply(extrair_dias)
            df['Logistica'] = df['Dias_Producao'].apply(lambda x: "⚡ FLASH" if x <= 2 else ("🐢 LENTO" if x > 7 else "📦 NORMAL"))
        else:
            df['Dias_Producao'] = 15
            df['Logistica'] = "📦 NORMAL"

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

df = carregar_dados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")
st.sidebar.markdown("---")

if not df.empty:
    # Filtros Globais
    fontes = st.sidebar.multiselect("Fonte de Dados", df['FONTE'].unique(), default=df['FONTE'].unique())
    cats = st.sidebar.multiselect("Categorias", df['CATEGORIA'].unique())
    
    # Aplica Filtros
    df_filtered = df[df['FONTE'].isin(fontes)]
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ESTRUTURA DE ABAS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Executiva", 
        "⚔️ Comparador de Preços", 
        "🧠 IA & Insights", 
        "🧪 Laboratório (Crie seu Gráfico)",
        "📂 Base de Dados"
    ])

    # =========================================================
    # ABA 1: VISÃO EXECUTIVA (Resumo + Gráfico Categoria)
    # =========================================================
    with tab1:
        st.markdown("### 🏢 Panorama Geral do Mercado")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Produtos Monitorados", len(df_filtered))
        k2.metric("Preço Médio", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        k3.metric("Ticket Mais Alto", f"R$ {df_filtered['Preco_Num'].max():.2f}")
        k4.metric("Concorrentes Imediatos", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        
        st.markdown("---")
        
        # GRÁFICO PEDIDO: PREÇO POR CATEGORIA
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("💰 Média de Preço por Categoria")
            fig_cat = px.bar(df_filtered.groupby('CATEGORIA')['Preco_Num'].mean().reset_index().sort_values('Preco_Num'), 
                             x='Preco_Num', y='CATEGORIA', orientation='h', text_auto='.2f',
                             title="Onde está o dinheiro? (Ticket Médio)", color='Preco_Num')
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with col_g2:
            st.subheader("📦 Volume de Produtos por Categoria")
            fig_vol = px.pie(df_filtered, names='CATEGORIA', title="Saturação do Mercado")
            st.plotly_chart(fig_vol, use_container_width=True)

    # =========================================================
    # ABA 2: COMPARADOR DE PLATAFORMAS (Elo7 vs Outros)
    # =========================================================
    with tab2:
        st.header("⚔️ Arena de Batalha: Comparação de Preços")
        
        if len(df['FONTE'].unique()) < 2:
            st.warning("⚠️ Atenção: Você só tem dados de UMA FONTE (provavelmente Elo7). Para comparar, rode o robô na Shopee ou Mercado Livre também.")
            st.info("Mostrando análise de variação interna por enquanto.")
        
        # Lógica de Fuzzy Matching (Encontrar produtos iguais)
        st.subheader("🔍 Buscador de Produtos Similares")
        termo_busca = st.text_input("Digite o nome de um produto para comparar preços em todo o mercado:", placeholder="Ex: Vaso Robert")
        
        if termo_busca:
            # Filtra produtos que parecem com o que vc digitou
            matches = process.extract(termo_busca, df_filtered['PRODUTO'].unique(), limit=20, scorer=fuzz.token_set_ratio)
            produtos_similares = [x[0] for x in matches if x[1] > 60]
            
            df_comparacao = df_filtered[df_filtered['PRODUTO'].isin(produtos_similares)]
            
            if not df_comparacao.empty:
                # Gráfico de Dispersão Comparativa
                fig_comp = px.scatter(df_comparacao, x="FONTE", y="Preco_Num", color="FONTE", 
                                      size="Preco_Num", hover_data=["PRODUTO", "Logistica"],
                                      title=f"Dispersão de Preços: '{termo_busca}'")
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.dataframe(df_comparacao[['FONTE', 'PRODUTO', 'Preco_Num', 'Logistica', 'LINK']], hide_index=True)
            else:
                st.error("Nenhum produto similar encontrado.")

    # =========================================================
    # ABA 3: INTELIGÊNCIA ARTIFICIAL (Mantido da v35)
    # =========================================================
    with tab3:
        c_ia1, c_ia2 = st.columns(2)
        with c_ia1:
            st.subheader("🧠 Clusterização (Segmentos)")
            if len(df_filtered) > 10:
                X = df_filtered[['Preco_Num', 'Dias_Producao']]
                kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
                df_filtered['Cluster'] = kmeans.labels_
                fig_clus = px.scatter(df_filtered, x="Dias_Producao", y="Preco_Num", color=df_filtered['Cluster'].astype(str),
                                      title="Segmentos de Mercado (IA)")
                st.plotly_chart(fig_clus, use_container_width=True)
            else:
                st.warning("Dados insuficientes para IA.")
                
        with c_ia2:
            st.subheader("☁️ Nuvem de Oportunidades (SEO)")
            texto = " ".join(df_filtered['PRODUTO'].astype(str))
            sw = set(STOPWORDS)
            sw.update(["de", "para", "com", "em", "kit", "3d", "pla"])
            try:
                wc = WordCloud(width=800, height=400, background_color='white', stopwords=sw).generate(texto)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            except: st.write("Texto insuficiente.")

    # =========================================================
    # ABA 4: LABORATÓRIO (VOCÊ PEDE O GRÁFICO)
    # =========================================================
    with tab4:
        st.header("🧪 Laboratório de Análise Personalizada")
        st.markdown("Não achou o gráfico que queria? **Monte o seu agora.**")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            eixo_x = st.selectbox("Escolha o Eixo X (Horizontal)", df_filtered.columns, index=list(df_filtered.columns).index('CATEGORIA') if 'CATEGORIA' in df_filtered.columns else 0)
        
        with col_ctrl2:
            eixo_y = st.selectbox("Escolha o Eixo Y (Vertical/Valor)", [c for c in df_filtered.columns if df_filtered[c].dtype in ['float64', 'int64']], index=0)
            
        with col_ctrl3:
            tipo_grafico = st.selectbox("Tipo de Gráfico", ["Barras", "Linha", "Dispersão (Pontos)", "Pizza", "Histograma", "Boxplot"])
            
        st.markdown("---")
        
        # Gerador Mágico de Gráficos
        if tipo_grafico == "Barras":
            st.plotly_chart(px.bar(df_filtered, x=eixo_x, y=eixo_y, color="FONTE", barmode='group'), use_container_width=True)
        elif tipo_grafico == "Linha":
            st.plotly_chart(px.line(df_filtered.sort_values(eixo_x), x=eixo_x, y=eixo_y, color="FONTE"), use_container_width=True)
        elif tipo_grafico == "Dispersão (Pontos)":
            st.plotly_chart(px.scatter(df_filtered, x=eixo_x, y=eixo_y, color="FONTE", hover_data=['PRODUTO']), use_container_width=True)
        elif tipo_grafico == "Pizza":
            st.plotly_chart(px.pie(df_filtered, names=eixo_x, values=eixo_y), use_container_width=True)
        elif tipo_grafico == "Histograma":
            st.plotly_chart(px.histogram(df_filtered, x=eixo_x, color="FONTE"), use_container_width=True)
        elif tipo_grafico == "Boxplot":
            st.plotly_chart(px.box(df_filtered, x=eixo_x, y=eixo_y, color="FONTE"), use_container_width=True)

    # =========================================================
    # ABA 5: DADOS BRUTOS (PLANILHA)
    # =========================================================
    with tab5:
        st.header("📂 Dados da Planilha")
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Erro crítico: Não foi possível ler a planilha. Verifique o Link CSV.")
