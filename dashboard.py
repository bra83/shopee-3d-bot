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

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# LINKS CSV (Já configurados com os seus links)
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- FUNÇÃO DE LIMPEZA DE PREÇO (BLINDADA) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "": return 0.0
    texto = str(valor).upper()
    # Remove tudo que não é numero, virgula ou ponto
    texto_limpo = re.sub(r'[^\d,.]', '', texto) 
    try:
        # Lógica Brasil: Se tem vírgula, ela é decimal
        if ',' in texto_limpo:
            texto_limpo = texto_limpo.replace('.', '').replace(',', '.')
        return float(texto_limpo)
    except: return 0.0

@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    status_msg = []
    
    fontes = [
        {"url": URL_ELO7, "nome": "Elo7"},
        {"url": URL_SHOPEE, "nome": "Shopee"}
    ]

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            # Normaliza colunas para MAIUSCULO
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty:
                status_msg.append(f"⚠️ {f['nome']}: Planilha vazia (Aguardando Robô).")
                continue

            # 1. Identifica PRODUTO
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), None)
            if col_prod: temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            else: temp_df['PRODUTO'] = "Sem Nome"

            # 2. Identifica PREÇO (Aqui está a blindagem para sua mudança)
            # Ele procura "PREÇO (R$)" primeiro, depois "PREÇO", depois "PRICE"
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE", "VALOR"])), None)
            
            if col_preco:
                temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else:
                temp_df['Preco_Num'] = 0.0

            # 3. Identifica FONTE
            col_fonte = next((c for c in temp_df.columns if "FONTE" in c or "SOURCE" in c), None)
            if col_fonte: temp_df['FONTE'] = temp_df[col_fonte].fillna(f["nome"])
            else: temp_df['FONTE'] = f["nome"]

            # 4. Categoria e Link
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"

            # 5. Prazo (Logística)
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

            # Finaliza DataFrame Limpo
            cols = ['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Logistica', 'Dias_Producao']
            for c in cols: 
                if c not in temp_df.columns: temp_df[c] = 0 if c == 'Preco_Num' else "N/A"
            
            dfs.append(temp_df[cols])
            status_msg.append(f"✅ {f['nome']}: {len(temp_df)} produtos carregados.")

        except Exception as e:
            status_msg.append(f"❌ {f['nome']}: Erro de Leitura ({str(e)})")

    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return final_df, status_msg

df, status_log = carregar_dados()

# --- DASHBOARD ---
st.sidebar.title("🎛️ Centro de Comando")
st.sidebar.markdown("**Status da Conexão:**")
for msg in status_log:
    if "✅" in msg: st.sidebar.success(msg)
    elif "⚠️" in msg: st.sidebar.warning(msg)
    else: st.sidebar.error(msg)
st.sidebar.markdown("---")

if not df.empty:
    fontes = st.sidebar.multiselect("Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    cats = st.sidebar.multiselect("Categorias", df['CATEGORIA'].unique())
    
    df_filtered = df[df['FONTE'].isin(fontes)]
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # ABAS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Geral", "⚔️ Comparador", "🧠 IA & Insights", "🧪 Laboratório", "📂 Dados"])

    # 1. GERAL
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))
        c2.metric("Preço Médio", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        c3.metric("Fontes", len(df_filtered['FONTE'].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1: st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all"), use_container_width=True)
        with col_g2: st.plotly_chart(px.pie(df_filtered, names='CATEGORIA'), use_container_width=True)

    # 2. COMPARADOR
    with tab2:
        st.header("⚔️ Comparador")
        termo = st.text_input("Buscar Similar:", placeholder="Ex: Vaso Robert")
        if termo:
            prods = df_filtered['PRODUTO'].unique()
            matches = process.extract(termo, prods, limit=30, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 55]
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
            if not df_comp.empty:
                st.plotly_chart(px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", hover_data=['PRODUTO']), use_container_width=True)
                st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True)
            else: st.warning("Nada encontrado.")

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
        sw.update(["de", "para", "com", "em", "kit", "3d", "pla", "impressão"])
        try:
            wc = WordCloud(width=600, height=300, background_color='white', stopwords=sw).generate(texto)
            fig_wc, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig_wc)
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

else:
    st.error("⚠️ Planilhas vazias.")
