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
from collections import Counter

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏢")

# --- 2. LINKS ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- LIMPEZA DE PREÇO (SÓ ISSO QUE EU MEXI PARA NÃO TRAVAR) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "": return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    texto = str(valor).upper().strip()
    texto = re.sub(r'[^\d,.]', '', texto)
    try:
        if ',' in texto: texto = texto.replace('.', '').replace(',', '.')
        return float(texto)
    except: return 0.0

@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    logs = []
    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty: continue

            # Mapeamento
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)

            temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            temp_df['FONTE'] = f["nome"]
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"
            
            if col_preco: temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else: temp_df['Preco_Num'] = 0.0

            # Lógica de Prazo
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
            
            cols = ['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Logistica', 'Dias_Producao']
            for c in cols: 
                if c not in temp_df.columns: temp_df[c] = ""
            
            dfs.append(temp_df[cols])

        except: pass

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df = carregar_dados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")

if not df.empty:
    fontes_sel = st.sidebar.multiselect("Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    df_filtered = df[df['FONTE'].isin(fontes_sel)]
    
    cats = st.sidebar.multiselect("Categorias", df_filtered['CATEGORIA'].unique())
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS (AQUI ESTÁ A ORDEM QUE VOCÊ QUER) ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visão Geral", 
        "⚔️ Comparador",  # <--- SEU CÓDIGO RESTAURADO AQUI
        "🧠 IA & Insights", 
        "🧪 Laboratório", 
        "💡 CRIADOR DE ANÚNCIOS", # <--- MANTIDO COMO PEDIU
        "📂 Dados"
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
        with col_g1: st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all"), use_container_width=True)
        with col_g2: st.plotly_chart(px.pie(df_filtered, names='CATEGORIA'), use_container_width=True)

    # ---------------------------------------------------------
    # 2. COMPARADOR (RESTAURADO DO SEU CÓDIGO ORIGINAL)
    # ---------------------------------------------------------
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
            df_comp = df_filtered
            st.success(f"Mostrando todos os {len(df_comp)} produtos da base.")
        elif termo:
            prods = df_filtered['PRODUTO'].unique()
            # SEU PARÂMETRO: Limit 50, Ratio, > 40
            matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 40] 
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
        
        if not df_comp.empty:
            # SEU LAYOUT DE MÉTRICAS
            cols_metrics = st.columns(len(df_comp['FONTE'].unique()) + 1)
            for i, fonte in enumerate(df_comp['FONTE'].unique()):
                media = df_comp[df_comp['FONTE']==fonte]['Preco_Num'].mean()
                cols_metrics[i].metric(f"Média {fonte}", f"R$ {media:.2f}")

            # SEU GRÁFICO
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
        sw.update(["de", "para", "3d", "pla", "com"])
        try:
            wc = WordCloud(width=600, height=300, background_color='white', stopwords=sw).generate(texto)
            fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig_wc)
        except: st.write("Texto insuficiente.")

    # 4. LABORATÓRIO
    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1: cx = st.selectbox("Eixo X", df_filtered.columns)
        with c2: cy = st.selectbox("Eixo Y", ['Preco_Num', 'Dias_Producao'])
        with c3: tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])
        if tp == "Barras": st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Dispersão": st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Boxplot": st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)

    # 5. CRIADOR DE ANÚNCIOS (A MÁGICA SALVA)
    with tab5:
        st.header("💡 IA: Gerador de Títulos Vencedores")
        keyword = st.text_input("Produto Foco (Ex: Suporte Fone):", "Vaso")
        if keyword:
            df_concorrentes = df[df['PRODUTO'].str.contains(keyword, case=False, na=False)]
            if not df_concorrentes.empty:
                texto_raw = " ".join(df_concorrentes['PRODUTO'].astype(str))
                palavras = [p for p in re.findall(r'\w+', texto_raw.lower()) if p not in sw and len(p) > 2]
                top_words = [x[0].title() for x in Counter(palavras).most_common(5)]
                st.success(f"Palavras-chave: {', '.join(top_words)}")
                st.code(f"{keyword.title()} 3D {' '.join(top_words[:2])} - Alta Qualidade", language="text")
                st.code(f"{top_words[0]} {keyword.title()} {' '.join(top_words[2:3])} - ENVIO IMEDIATO ⚡", language="text")
                st.info(f"Baseado na análise de {len(df_concorrentes)} produtos.")
            else:
                st.warning("Sem dados suficientes para esse termo.")

    # 6. DADOS
    with tab6: st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("⚠️ Planilhas vazias ou inacessíveis.")
