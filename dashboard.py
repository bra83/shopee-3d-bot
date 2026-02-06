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

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# --- 2. LINKS DE CONEXÃO ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- FUNÇÃO DE LIMPEZA DE PREÇO ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0
    texto = str(valor).upper()
    # Remove tudo que não for número, vírgula ou ponto
    texto_limpo = re.sub(r'[^\d,.]', '', texto)
    try:
        if ',' in texto_limpo:
            # Padrão Brasileiro: 1.200,50 -> 1200.50
            texto_limpo = texto_limpo.replace('.', '').replace(',', '.')
        return float(texto_limpo)
    except:
        return 0.0

@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    logs = []
    
    fontes_config = [
        {"url": URL_ELO7, "nome": "Elo7"},
        {"url": URL_SHOPEE, "nome": "Shopee"}
    ]

    for f in fontes_config:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty:
                logs.append(f"⚠️ {f['nome']}: Planilha vazia.")
                continue

            # 1. Identificar PRODUTO
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), "PRODUTO")
            if col_prod in temp_df.columns:
                temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            else:
                temp_df['PRODUTO'] = "Sem Nome"

            # 2. Identificar PREÇO (Prioriza o seu novo nome "Preço (R$)")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE", "VALOR"])), None)
            if col_preco:
                temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else:
                temp_df['Preco_Num'] = 0.0

            # 3. Forçar FONTE correta
            temp_df['FONTE'] = f["nome"]

            # 4. CATEGORIA e LINK
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"

            # 5. PRAZO / DIAS
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)
            if col_prazo:
                temp_df['Prazo_Txt'] = temp_df[col_prazo].fillna("Normal")
                def get_days(t):
                    t = str(t).upper()
                    if "IMEDIATO" in t or "PRONTA" in t: return 1
                    m = re.search(r'(\d+)', t)
                    return int(m.group(1)) if m else 15
                temp_df['Dias_Producao'] = temp_df['Prazo_Txt'].apply(get_days)
            else:
                temp_df['Dias_Producao'] = 15
            
            temp_df['Logistica'] = temp_df['Dias_Producao'].apply(lambda x: "⚡ FLASH" if x <= 2 else "📦 NORMAL")

            # Finaliza colunas padrão
            df_final = temp_df[['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Logistica', 'Dias_Producao']].copy()
            dfs.append(df_final)
            logs.append(f"✅ {f['nome']}: {len(temp_df)} itens.")

        except Exception as e:
            logs.append(f"❌ {f['nome']}: Erro ({str(e)})")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), logs

# Executar carga
df, status_log = carregar_dados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")
for msg in status_log:
    if "✅" in msg: st.sidebar.success(msg)
    elif "⚠️" in msg: st.sidebar.warning(msg)
    else: st.sidebar.error(msg)

st.sidebar.markdown("---")

if not df.empty:
    # Filtros
    fontes_sel = st.sidebar.multiselect("Filtrar Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    df_filtered = df[df['FONTE'].isin(fontes_sel)]
    
    cats = st.sidebar.multiselect("Filtrar Categorias", df_filtered['CATEGORIA'].unique())
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS SUPREMAS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Geral", "⚔️ Comparador", "🧠 IA & Insights", "🧪 Laboratório", "📂 Dados Brutos"])

    # 1. VISÃO GERAL
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))
        c2.metric("Preço Médio", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        c3.metric("Fontes Ativas", len(df_filtered['FONTE'].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Dispersão de Preços por Canal")
            fig = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            st.subheader("Distribuição por Categoria")
            fig2 = px.pie(df_filtered, names='CATEGORIA', hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

    # 2. COMPARADOR
    with tab2:
        st.header("⚔️ Comparador Cross-Platform")
        termo = st.text_input("Buscar Produto (Ex: Vaso, Robert):", placeholder="Digite o nome...")
        if termo:
            nomes_unicos = df_filtered['PRODUTO'].unique()
            matches = process.extract(termo, nomes_unicos, limit=40, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 55]
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
            
            if not df_comp.empty:
                st.subheader(f"Resultados para: {termo}")
                fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", 
                                      hover_data=['PRODUTO'], title="Comparação de Preços Detectados")
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True, use_container_width=True)
            else:
                st.warning("Nenhum item similar encontrado.")

    # 3. IA & INSIGHTS
    with tab3:
        st.header("🧠 Inteligência Artificial Aplicada")
        col_ia1, col_ia2 = st.columns(2)
        with col_ia1:
            st.subheader("Agrupamento de Mercado (K-Means)")
            if len(df_filtered) > 10:
                X = df_filtered[['Preco_Num', 'Dias_Producao']].fillna(0)
                kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
                df_filtered['Cluster'] = kmeans.labels_.astype(str)
                fig_ia = px.scatter(df_filtered, x="Dias_Producao", y="Preco_Num", color="Cluster", 
                                    title="Segmentos: Preço x Prazo", hover_data=['PRODUTO'])
                st.plotly_chart(fig_ia, use_container_width=True)
        with col_ia2:
            st.subheader("Nuvem de Palavras (SEO)")
            texto = " ".join(df_filtered['PRODUTO'].astype(str))
            sw = set(STOPWORDS)
            sw.update(["de", "para", "com", "em", "kit", "3d", "pla", "impressão", "peca"])
            try:
                wc = WordCloud(width=600, height=400, background_color='black', stopwords=sw).generate(texto)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            except: st.write("Texto insuficiente.")

    # 4. LABORATÓRIO SELF-SERVICE
    with tab4:
        st.header("🧪 Laboratório de Gráficos")
        c1, c2, c3 = st.columns(3)
        cols_num = [c for c in df_filtered.columns if df_filtered[c].dtype in ['float64', 'int64']]
        with c1: ex = st.selectbox("Eixo X", df_filtered.columns, index=0)
        with c2: ey = st.selectbox("Eixo Y (Valor)", cols_num, index=0)
        with c3: tp = st.selectbox("Tipo de Gráfico", ["Barras", "Dispersão", "Boxplot", "Linha"])
        
        st.markdown("---")
        if tp == "Barras": st.plotly_chart(px.bar(df_filtered, x=ex, y=ey, color="FONTE", barmode="group"), use_container_width=True)
        elif tp == "Dispersão": st.plotly_chart(px.scatter(df_filtered, x=ex, y=ey, color="FONTE", hover_data=['PRODUTO']), use_container_width=True)
        elif tp == "Boxplot": st.plotly_chart(px.box(df_filtered, x=ex, y=ey, color="FONTE"), use_container_width=True)
        elif tp == "Linha": st.plotly_chart(px.line(df_filtered.sort_values(ex), x=ex, y=ey, color="FONTE"), use_container_width=True)

    # 5. DADOS BRUTOS
    with tab5:
        st.header("📂 Base de Dados Unificada")
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("⚠️ As planilhas não foram carregadas corretamente. Verifique os status na barra lateral.")
