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

# --- FUNÇÃO DE LIMPEZA DE PREÇO (BLINDADA) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "": 
        return 0.0
    
    # Se já for número (float/int), retorna direto
    if isinstance(valor, (int, float)):
        return float(valor)
        
    texto = str(valor).upper()
    # Remove tudo que não é dígito, vírgula ou ponto
    texto_limpo = re.sub(r'[^\d,.]', '', texto)
    
    try:
        # Lógica BR: Se tem vírgula, é decimal (25,50 -> 25.50)
        # Se tem ponto e vírgula (1.200,50), remove ponto
        if ',' in texto_limpo:
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
            # Lê o CSV ignorando linhas ruins
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            
            # Normaliza colunas para maiúsculas e remove espaços
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty:
                logs.append(f"⚠️ {f['nome']}: Conectado, mas planilha vazia.")
                continue

            # --- MAPEAMENTO INTELIGENTE DE COLUNAS ---
            
            # 1. Identificar PRODUTO (Busca variações de nome)
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), "PRODUTO")
            if col_prod in temp_df.columns: 
                temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            else: 
                temp_df['PRODUTO'] = "Sem Nome"

            # 2. Identificar PREÇO
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE", "VALOR"])), None)
            if col_preco: 
                temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else: 
                temp_df['Preco_Num'] = 0.0

            # 3. Forçar FONTE
            temp_df['FONTE'] = f["nome"]

            # 4. CATEGORIA e LINK
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"

            # 5. PRAZO & LOGÍSTICA (Ajustado para ler 'X Dias' ou 'Imediato')
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c or "ENVIO" in c), None)
            
            if col_prazo:
                temp_df['Prazo_Txt'] = temp_df[col_prazo].fillna("Normal")
                
                def get_days(t):
                    t = str(t).upper()
                    # Regra de Ouro: Imediato/Pronta = 1 dia
                    if "IMEDIATO" in t or "PRONTA" in t: return 1
                    # Regra: Busca número (ex: "15 dias")
                    m = re.search(r'(\d+)', t)
                    return int(m.group(1)) if m else 15
                
                temp_df['Dias_Producao'] = temp_df['Prazo_Txt'].apply(get_days)
            else:
                # Se não tem coluna de prazo, assume 15 dias (pior cenário)
                temp_df['Dias_Producao'] = 15
            
            # Cria a flag visual
            temp_df['Logistica'] = temp_df['Dias_Producao'].apply(lambda x: "⚡ FLASH" if x <= 2 else "📦 NORMAL")

            # Finaliza DataFrame
            cols_finais = ['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Logistica', 'Dias_Producao']
            
            # Garante que todas as colunas existem
            for c in cols_finais:
                if c not in temp_df.columns: 
                    temp_df[c] = 0 if c == 'Preco_Num' else "N/A"

            # Adiciona apenas as colunas limpas
            dfs.append(temp_df[cols_finais])
            logs.append(f"✅ {f['nome']}: {len(temp_df)} itens carregados.")

        except Exception as e:
            logs.append(f"❌ {f['nome']}: Erro ({str(e)})")

    # Junta tudo num DataFrame único
    if dfs:
        return pd.concat(dfs, ignore_index=True), logs
    else:
        return pd.DataFrame(), logs

# --- EXECUÇÃO ---
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
    # Filtros Globais
    fontes_disp = df['FONTE'].unique()
    fontes_sel = st.sidebar.multiselect("Fontes Ativas", fontes_disp, default=fontes_disp)
    
    # Filtra por Fonte
    df_filtered = df[df['FONTE'].isin(fontes_sel)]
    
    # Filtra por Categoria
    if 'CATEGORIA' in df_filtered.columns:
        cats_disp = df_filtered['CATEGORIA'].unique()
        cats = st.sidebar.multiselect("Filtrar Categoria", cats_disp)
        if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS ESTRATÉGICAS ---
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
        c4.metric("Itens Flash (Pronta Entrega)", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Quem é mais caro?")
            fig = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all", title="Distribuição de Preços")
            st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            st.subheader("Volume por Categoria")
            fig2 = px.pie(df_filtered, names='CATEGORIA', title="Market Share")
            st.plotly_chart(fig2, use_container_width=True)

    # 2. COMPARADOR
    with tab2:
        st.header("⚔️ Comparador Cross-Platform")
        st.info("Veja como o mesmo produto é vendido no Elo7 vs Shopee.")
        
        col_input, col_check = st.columns([3, 1])
        with col_input:
            termo = st.text_input("Buscar Produto:", placeholder="Ex: Vaso Robert, Suporte Fone...")
        with col_check:
            st.write("") 
            st.write("") 
            mostrar_tudo = st.checkbox("Ver Tabela Completa", value=False)
        
        df_comp = pd.DataFrame()

        if mostrar_tudo:
            df_comp = df_filtered
        elif termo:
            # Lógica Fuzzy (Busca Aproximada)
            prods = df_filtered['PRODUTO'].unique()
            # Score > 40 para pegar variações (Shopee costuma ter títulos longos e sujos)
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
                                  hover_data=['PRODUTO'], title="Dispersão de Preços (Oportunidade)")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.dataframe(
                df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'Logistica', 'LINK']], 
                column_config={
                    "LINK": st.column_config.LinkColumn("Link"),
                    "Preco_Num": st.column_config.NumberColumn("Preço", format="R$ %.2f")
                },
                hide_index=True, 
                use_container_width=True
            )
        else:
            if not mostrar_tudo:
                st.warning("Digite um termo acima para comparar.")

    # 3. IA
    with tab3:
        st.subheader("🧠 Segmentação Automática (K-Means)")
        if len(df_filtered) > 10:
            X = df_filtered[['Preco_Num', 'Dias_Producao']].fillna(0)
            kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
            df_filtered['Cluster'] = kmeans.labels_
            
            fig_ia = px.scatter(
                df_filtered, x="Dias_Producao", y="Preco_Num", 
                color=df_filtered['Cluster'].astype(str),
                title="Agrupamento: Preço x Tempo de Produção",
                labels={"Dias_Producao": "Dias para Produzir", "Preco_Num": "Preço (R$)"}
            )
            st.plotly_chart(fig_ia, use_container_width=True)
        else:
            st.info("Dados insuficientes para IA (Mínimo 10 produtos).")
        
        st.subheader("☁️ Nuvem de Palavras (Termos Quentes)")
        texto = " ".join(df_filtered['PRODUTO'].astype(str))
        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla", "com", "em", "kit", "un", "peça"])
        
        try:
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=sw, colormap='viridis').generate(texto)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        except: st.write("Texto insuficiente para gerar nuvem.")

    # 4. LABORATÓRIO
    with tab4:
        st.subheader("🔬 Laboratório de Dados")
        c1, c2, c3 = st.columns(3)
        with c1: cx = st.selectbox("Eixo X", ['FONTE', 'CATEGORIA', 'Logistica', 'Dias_Producao'])
        with c2: cy = st.selectbox("Eixo Y", ['Preco_Num', 'Dias_Producao'])
        with c3: tp = st.selectbox("Tipo Gráfico", ["Boxplot", "Barras", "Dispersão"])
        
        if tp == "Barras": st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE", barmode='group'), use_container_width=True)
        elif tp == "Dispersão": st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Boxplot": st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)

    # 5. DADOS
    with tab5: 
        st.subheader("Base de Dados Completa")
        st.dataframe(df_filtered, use_container_width=True)

    # 6. RAIO-X
    with tab_debug:
        st.header("🛠️ Diagnóstico de Carga")
        st.write("Verifique se as colunas estão sendo lidas corretamente:")
        st.write(f"Total Linhas Brutas: {len(df)}")
        st.write("Colunas Detectadas:", list(df.columns))
        st.dataframe(df.head(50))

else:
    st.error("⚠️ Nenhuma planilha conectada ou dados vazios. Verifique os links no código.")
