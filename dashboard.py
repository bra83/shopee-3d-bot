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

# --- 2. LINKS DE CONEXÃO (Inseridos Automaticamente) ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

@st.cache_data(ttl=60)
def carregar_dados_unificados():
    dfs = []
    
    # Lista de Fontes para carregar
    fontes = [
        {"url": URL_ELO7, "nome_padrao": "Elo7"},
        {"url": URL_SHOPEE, "nome_padrao": "Shopee"}
    ]

    for f in fontes:
        try:
            # Lê o CSV ignorando linhas problemáticas
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            
            # --- PADRONIZAÇÃO DE COLUNAS (O Segredo da União) ---
            # Transforma tudo para maiúsculo e remove espaços (ex: " Preço " -> "PREÇO")
            temp_df.columns = [c.strip().upper() for c in temp_df.columns]
            
            # 1. Identifica ou Cria Coluna PRODUTO
            # Procura por 'PRODUTO', 'NOME', 'TITULO', 'ITEM'
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), "PRODUTO")
            if col_prod not in temp_df.columns: temp_df[col_prod] = "Sem Nome"
            temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})

            # 2. Identifica ou Cria Coluna PREÇO
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["PREÇO", "PRICE", "VALOR"])), None)
            if col_preco:
                # Limpa "R$" e converte "1.200,00" -> 1200.00
                temp_df['Preco_Num'] = temp_df[col_preco].astype(str).str.replace('R$', '', regex=False)
                temp_df['Preco_Num'] = temp_df['Preco_Num'].str.replace('.', '', regex=False).str.replace(',', '.')
                temp_df['Preco_Num'] = pd.to_numeric(temp_df['Preco_Num'], errors='coerce').fillna(0)
            else:
                temp_df['Preco_Num'] = 0.0

            # 3. Identifica ou Cria FONTE
            # Se a coluna 'FONTE' não existir na planilha, usamos o nome padrão (Elo7/Shopee)
            col_fonte = next((c for c in temp_df.columns if "FONTE" in c or "SOURCE" in c), None)
            if col_fonte:
                temp_df['FONTE_REAL'] = temp_df[col_fonte].fillna(f["nome_padrao"])
            else:
                temp_df['FONTE_REAL'] = f["nome_padrao"]

            # 4. Identifica CATEGORIA
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA_REAL'] = temp_df[col_cat] if col_cat else "Geral"

            # 5. Identifica LINK
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK_REAL'] = temp_df[col_link] if col_link else "#"

            # 6. Identifica PRAZO (Logística)
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)
            if col_prazo:
                temp_df['Prazo_Texto'] = temp_df[col_prazo].fillna("Normal")
                
                # Função inteligente para extrair dias
                def get_days(txt):
                    txt = str(txt).upper()
                    if "IMEDIATO" in txt or "PRONTA" in txt: return 1
                    # Procura números no texto (ex: "15 dias" -> 15)
                    m = re.search(r'(\d+)', txt)
                    return int(m.group(1)) if m else 15
                
                temp_df['Dias_Producao'] = temp_df['Prazo_Texto'].apply(get_days)
                
                # Classifica a oportunidade
                temp_df['Logistica'] = temp_df['Dias_Producao'].apply(
                    lambda x: "⚡ FLASH (Até 2 dias)" if x <= 2 else ("🐢 LENTO (>7 dias)" if x > 7 else "📦 NORMAL")
                )
            else:
                # Se não tiver prazo, assumimos o padrão do mercado
                temp_df['Dias_Producao'] = 15
                temp_df['Logistica'] = "📦 NORMAL"

            # Seleciona apenas as colunas limpas e padronizadas para o DataFrame Final
            df_clean = temp_df[['PRODUTO', 'Preco_Num', 'FONTE_REAL', 'CATEGORIA_REAL', 'LINK_REAL', 'Logistica', 'Dias_Producao']].copy()
            df_clean = df_clean.rename(columns={'FONTE_REAL': 'FONTE', 'CATEGORIA_REAL': 'CATEGORIA', 'LINK_REAL': 'LINK'})
            
            dfs.append(df_clean)
            
        except Exception as e:
            st.error(f"Erro ao ler {f['nome_padrao']}: {e}")

    # Junta tudo numa tabela só
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# Carrega os dados unificados
df = carregar_dados_unificados()

# --- SIDEBAR (FILTROS) ---
st.sidebar.title("🎛️ Centro de Comando")
st.sidebar.markdown("---")

if not df.empty:
    # Filtros Dinâmicos
    fontes = st.sidebar.multiselect("Fontes de Dados", df['FONTE'].unique(), default=df['FONTE'].unique())
    cats = st.sidebar.multiselect("Categorias", df['CATEGORIA'].unique())
    
    # Aplica os filtros
    df_filtered = df[df['FONTE'].isin(fontes)]
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS DO DASHBOARD ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral", 
        "⚔️ Comparador (Elo7 vs Shopee)", 
        "🧠 IA & Insights", 
        "🧪 Laboratório (Crie Gráficos)", 
        "📂 Base de Dados"
    ])

    # =========================================================
    # ABA 1: VISÃO GERAL (RESUMO EXECUTIVO)
    # =========================================================
    with tab1:
        st.markdown("### 🏢 Panorama Geral do Mercado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de Produtos", len(df_filtered))
        c2.metric("Preço Médio Global", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        c3.metric("Fontes Ativas", len(df_filtered['FONTE'].unique()))
        c4.metric("Concorrência Flash", len(df_filtered[df_filtered['Logistica'].str.contains("FLASH")]))
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1: 
            st.subheader("💰 Quem cobra mais caro? (Preço por Fonte)")
            # Gráfico de violino/boxplot mostra bem a distribuição de preços entre sites
            fig_fonte = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all",
                               title="Dispersão de Preço: Shopee vs Elo7")
            st.plotly_chart(fig_fonte, use_container_width=True)
            
        with col_g2: 
            st.subheader("📦 Onde estão os produtos? (Volume por Categoria)")
            fig_vol = px.pie(df_filtered, names='CATEGORIA', title="Distribuição do Mix de Produtos")
            st.plotly_chart(fig_vol, use_container_width=True)

    # =========================================================
    # ABA 2: COMPARADOR (CROSS-PLATFORM FUZZY MATCH)
    # =========================================================
    with tab2:
        st.header("⚔️ Arena de Batalha: Comparação Direta")
        st.info("Digite o nome de um produto (ex: 'Vaso') para encontrar itens similares em todas as plataformas.")
        
        termo = st.text_input("Buscar Produto:", placeholder="Ex: Vaso Robert, Suporte Fone, etc.")
        
        if termo:
            # Lógica de Similaridade (Fuzzy)
            prods = df_filtered['PRODUTO'].dropna().unique()
            matches = process.extract(termo, prods, limit=40, scorer=fuzz.token_set_ratio)
            # Aceita similaridade acima de 55% para pegar variações
            similares = [x[0] for x in matches if x[1] > 55]
            
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
            
            if not df_comp.empty:
                col_c1, col_c2 = st.columns([3, 1])
                
                with col_c1:
                    # Gráfico de Dispersão Comparativa
                    fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", 
                                          hover_data=["PRODUTO", "Logistica"], 
                                          title=f"Raio-X de Preços: '{termo}'")
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with col_c2:
                    st.markdown("### 🏆 Estatísticas")
                    st.metric("Média Shopee", f"R$ {df_comp[df_comp['FONTE']=='Shopee']['Preco_Num'].mean():.2f}")
                    st.metric("Média Elo7", f"R$ {df_comp[df_comp['FONTE']=='Elo7']['Preco_Num'].mean():.2f}")

                st.markdown("### 📋 Lista de Concorrentes Detectados")
                st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'Logistica', 'LINK']], hide_index=True, use_container_width=True)
            else:
                st.warning("Nenhum produto similar encontrado com esse nome.")

    # =========================================================
    # ABA 3: INTELIGÊNCIA ARTIFICIAL
    # =========================================================
    with tab3:
        st.header("🧠 Insights Automatizados")
        c_ia1, c_ia2 = st.columns(2)
        with c_ia1:
            st.subheader("Segmentação de Mercado (K-Means)")
            if len(df_filtered) > 10:
                X = df_filtered[['Preco_Num', 'Dias_Producao']].fillna(0)
                kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
                df_filtered['Cluster'] = kmeans.labels_
                fig_clus = px.scatter(df_filtered, x="Dias_Producao", y="Preco_Num", color=df_filtered['Cluster'].astype(str),
                                      title="Agrupamento Automático (Preço x Prazo)")
                st.plotly_chart(fig_clus, use_container_width=True)
            else:
                st.info("Dados insuficientes para IA.")
                
        with c_ia2:
            st.subheader("Termos Mais Usados (SEO)")
            texto = " ".join(df_filtered['PRODUTO'].astype(str))
            sw = set(STOPWORDS)
            sw.update(["de", "para", "com", "em", "kit", "3d", "pla", "impressão", "artesanal", "peca", "peça"])
            try:
                wc = WordCloud(width=800, height=400, background_color='white', stopwords=sw).generate(texto)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            except: st.write("Texto insuficiente.")

    # =========================================================
    # ABA 4: LABORATÓRIO (CRIE SEU GRÁFICO)
    # =========================================================
    with tab4:
        st.header("🧪 Laboratório de Dados")
        st.markdown("Monte gráficos personalizados cruzando qualquer informação.")
        
        c1, c2, c3 = st.columns(3)
        colunas_numericas = [c for c in df_filtered.columns if df_filtered[c].dtype in ['float64', 'int64', 'int32']]
        
        with c1: cx = st.selectbox("Eixo X (Horizontal)", df_filtered.columns, index=list(df_filtered.columns).index('CATEGORIA') if 'CATEGORIA' in df_filtered.columns else 0)
        with c2: cy = st.selectbox("Eixo Y (Vertical)", colunas_numericas, index=0 if colunas_numericas else 0)
        with c3: tipo = st.selectbox("Tipo de Gráfico", ["Barras", "Dispersão (Scatter)", "Boxplot (Caixa)", "Pizza", "Histograma"])
        
        st.markdown("---")
        
        if tipo == "Barras": st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE", barmode='group'), use_container_width=True)
        elif tipo == "Dispersão (Scatter)": st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE", hover_data=['PRODUTO']), use_container_width=True)
        elif tipo == "Boxplot (Caixa)": st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tipo == "Pizza": st.plotly_chart(px.pie(df_filtered, names=cx, values=cy), use_container_width=True)
        elif tipo == "Histograma": st.plotly_chart(px.histogram(df_filtered, x=cx, color="FONTE"), use_container_width=True)

    # =========================================================
    # ABA 5: DADOS BRUTOS
    # =========================================================
    with tab5:
        st.header("📂 Base de Dados Completa")
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Erro crítico: As planilhas parecem vazias ou os links estão inacessíveis.")
