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
    if pd.isna(valor) or str(valor).strip() == "": return 0.0
    texto = str(valor).upper()
    texto_limpo = re.sub(r'[^\d,.]', '', texto)
    try:
        if ',' in texto_limpo:
            texto_limpo = texto_limpo.replace('.', '').replace(',', '.')
        return float(texto_limpo)
    except: return 0.0

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
            # Lê o CSV
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty:
                logs.append(f"⚠️ {f['nome']}: Planilha vazia.")
                continue

            # 1. Identificar PRODUTO
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), "PRODUTO")
            if col_prod in temp_df.columns: temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            else: temp_df['PRODUTO'] = "Sem Nome"

            # 2. Identificar PREÇO
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE", "VALOR"])), None)
            if col_preco: temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else: temp_df['Preco_Num'] = 0.0

            # 3. Forçar FONTE
            temp_df['FONTE'] = f["nome"]

            # 4. CATEGORIA e LINK
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"

            # 5. PRAZO
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

            # Finaliza DataFrame
            cols_finais = ['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Logistica', 'Dias_Producao']
            # Cria colunas faltantes para evitar erro
            for c in cols_finais:
                if c not in temp_df.columns: temp_df[c] = 0 if c == 'Preco_Num' else "N/A"

            dfs.append(temp_df[cols_finais])
            logs.append(f"✅ {f['nome']}: {len(temp_df)} itens.")

        except Exception as e:
            logs.append(f"❌ {f['nome']}: Erro ({str(e)})")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), logs

# Executar Carga
df, status_log = carregar_dados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")

# Status Detalhado
st.sidebar.markdown("**Status da Carga:**")
for msg in status_log:
    if "✅" in msg: st.sidebar.success(msg)
    elif "⚠️" in msg: st.sidebar.warning(msg)
    else: st.sidebar.error(msg)
st.sidebar.markdown("---")

if not df.empty:
    # Filtros
    fontes_disp = df['FONTE'].unique()
    fontes_sel = st.sidebar.multiselect("Fontes Ativas", fontes_disp, default=fontes_disp)
    
    df_filtered = df[df['FONTE'].isin(fontes_sel)]
    
    cats = st.sidebar.multiselect("Filtrar Categoria", df_filtered['CATEGORIA'].unique())
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS ---
    tab1, tab2, tab3, tab4, tab5, tab_debug = st.tabs([
        "📊 Visão Geral", 
        "⚔️ Comparador", 
        "🧠 IA & Insights", 
        "🧪 Laboratório", 
        "📂 Dados",
        "🛠️ Raio-X"
    ])

    # 1. GERAL
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))
        c2.metric("Preço Médio", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        c3.metric("Fontes", len(df_filtered['FONTE'].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Quem é mais caro?")
            fig = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all")
            st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            st.subheader("Volume por Categoria")
            fig2 = px.pie(df_filtered, names='CATEGORIA')
            st.plotly_chart(fig2, use_container_width=True)

    # 2. COMPARADOR (CORRIGIDO)
    with tab2:
        st.header("⚔️ Comparador Cross-Platform")
        
        col_input, col_check = st.columns([3, 1])
        with col_input:
            termo = st.text_input("Buscar Produto (Filtro por Nome):", placeholder="Ex: Vaso, Robert, Suporte...")
        with col_check:
            st.write("") 
            st.write("") 
            mostrar_tudo = st.checkbox("Mostrar TUDO (Ignorar Nome)", value=False)
        
        df_comp = pd.DataFrame()

        if mostrar_tudo:
            # Mostra tudo sem filtrar (PROVA REAL)
            df_comp = df_filtered
            st.success(f"Mostrando todos os {len(df_comp)} produtos da base.")
        elif termo:
            # Filtro Fuzzy
            prods = df_filtered['PRODUTO'].unique()
            # Limiar baixado para 40 para pegar Shopee com nomes estranhos
            matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 40] 
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
        
        if not df_comp.empty:
            # Métricas Lado a Lado
            cols_metrics = st.columns(len(df_comp['FONTE'].unique()) + 1)
            for i, fonte in enumerate(df_comp['FONTE'].unique()):
                media = df_comp[df_comp['FONTE']==fonte]['Preco_Num'].mean()
                cols_metrics[i].metric(f"Média {fonte}", f"R$ {media:.2f}")

            fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", 
                                  hover_data=['PRODUTO'], title="Dispersão de Preços")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True, use_container_width=True)
        else:
            if not mostrar_tudo:
                st.info("Digite um termo acima ou marque 'Mostrar TUDO' para ver os dados.")

    # 3. IA
    with tab3:
        st.subheader("Segmentação (K-Means)")
        if len(df_filtered) > 10:
            X = df_filtered[['Preco_Num', 'Dias_Producao']].fillna(0)
            kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
            df_filtered['Cluster'] = kmeans.labels_
            st.plotly_chart(px.scatter(df_filtered, x="Dias_Producao", y="Preco_Num", color=df_filtered['Cluster'].astype(str)), use_container_width=True)
        
        st.subheader("Nuvem de Palavras")
        texto = " ".join(df_filtered['PRODUTO'].astype(str))
        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla"])
        try:
            wc = WordCloud(width=600, height=300, background_color='white', stopwords=sw).generate(texto)
            fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig_wc)
        except: st.write("Texto insuficiente.")

    # 4. LABORATÓRIO
    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1: cx = st.selectbox("Eixo X", df_filtered.columns)
        with c2: cy = st.selectbox("Eixo Y", [c for c in df_filtered.columns if df_filtered[c].dtype in ['float64', 'int64']])
        with c3: tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])
        if tp == "Barras": st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Dispersão": st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Boxplot": st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)

    # 5. DADOS
    with tab5: st.dataframe(df_filtered)

    # 6. RAIO-X (NOVO)
    with tab_debug:
        st.header("🛠️ Diagnóstico de Carga")
        st.write("Verifique abaixo se os dados da Shopee estão sendo carregados corretamente.")
        st.write(f"Linhas Totais: {len(df)}")
        st.write("Amostra dos dados carregados:")
        st.dataframe(df.head(100))

else:
    st.error("⚠️ Erro Crítico: Planilhas inacessíveis.")
