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

# --- FUNÇÃO DE LIMPEZA DE PREÇO (CIRURGIA A LASER) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0
    
    # Converte para texto
    texto = str(valor).upper()
    
    # 1. Remove tudo que NÃO for número, vírgula ou ponto
    # (Arranca R$, BRL, espaços, letras)
    texto_limpo = re.sub(r'[^\d,.]', '', texto)
    
    try:
        # Lógica para moeda brasileira (1.200,50)
        if ',' in texto_limpo:
            # Se tiver vírgula, assume que ela é decimal
            # Remove pontos de milhar (1.200 -> 1200)
            texto_limpo = texto_limpo.replace('.', '') 
            # Troca vírgula por ponto (1200,50 -> 1200.50)
            texto_limpo = texto_limpo.replace(',', '.')
        
        return float(texto_limpo)
    except:
        return 0.0

@st.cache_data(ttl=60)
def carregar_dados_unificados():
    dfs = []
    
    fontes = [
        {"url": URL_ELO7, "nome_padrao": "Elo7"},
        {"url": URL_SHOPEE, "nome_padrao": "Shopee"}
    ]

    for f in fontes:
        try:
            # Lê o CSV
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip')
            
            # Normaliza colunas (Tudo Maiúsculo)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            # --- GARANTIA DE ESTRUTURA ---
            # Se a planilha estiver vazia, pula
            if temp_df.empty: continue

            # 1. PRODUTO
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO", "ITEM"])), None)
            if col_prod:
                temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            else:
                temp_df['PRODUTO'] = "Produto Sem Nome"

            # 2. PREÇO (AQUI ESTAVA O ERRO)
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["PREÇO", "PRICE", "VALOR"])), None)
            
            if col_preco:
                # Aplica a limpeza cirúrgica
                temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else:
                # Se não achar a coluna, cria zerada para não travar
                temp_df['Preco_Num'] = 0.0

            # 3. FONTE
            col_fonte = next((c for c in temp_df.columns if "FONTE" in c or "SOURCE" in c), None)
            if col_fonte:
                temp_df['FONTE_REAL'] = temp_df[col_fonte].fillna(f["nome_padrao"])
            else:
                temp_df['FONTE_REAL'] = f["nome_padrao"]

            # 4. CATEGORIA
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            temp_df['CATEGORIA_REAL'] = temp_df[col_cat] if col_cat else "Geral"

            # 5. LINK
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            temp_df['LINK_REAL'] = temp_df[col_link] if col_link else "#"

            # 6. PRAZO (Logística)
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)
            if col_prazo:
                temp_df['Prazo_Texto'] = temp_df[col_prazo].fillna("Normal")
                def get_days(txt):
                    txt = str(txt).upper()
                    if "IMEDIATO" in txt or "PRONTA" in txt: return 1
                    m = re.search(r'(\d+)', txt)
                    return int(m.group(1)) if m else 15
                temp_df['Dias_Producao'] = temp_df['Prazo_Texto'].apply(get_days)
                temp_df['Logistica'] = temp_df['Dias_Producao'].apply(lambda x: "⚡ FLASH" if x <= 2 else ("🐢 LENTO" if x > 7 else "📦 NORMAL"))
            else:
                temp_df['Dias_Producao'] = 15
                temp_df['Logistica'] = "📦 NORMAL"

            # Seleciona Colunas Finais (Blindagem contra KeyError)
            cols_finais = ['PRODUTO', 'Preco_Num', 'FONTE_REAL', 'CATEGORIA_REAL', 'LINK_REAL', 'Logistica', 'Dias_Producao']
            
            # Garante que todas existam
            for col in cols_finais:
                if col not in temp_df.columns:
                    temp_df[col] = "N/A" if col != 'Preco_Num' else 0.0

            df_clean = temp_df[cols_finais].copy()
            df_clean = df_clean.rename(columns={'FONTE_REAL': 'FONTE', 'CATEGORIA_REAL': 'CATEGORIA', 'LINK_REAL': 'LINK'})
            
            dfs.append(df_clean)
            
        except Exception as e:
            # Em vez de travar, avisa qual arquivo deu erro
            st.warning(f"Aviso: Não foi possível ler os dados de {f['nome_padrao']}. Motivo: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# Carrega os dados unificados
df = carregar_dados_unificados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")

if not df.empty:
    fontes = st.sidebar.multiselect("Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    cats = st.sidebar.multiselect("Categorias", df['CATEGORIA'].unique())
    
    df_filtered = df[df['FONTE'].isin(fontes)]
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- ABAS ---
    tab1, tab2, tab3, tab4, tab5, tab_debug = st.tabs([
        "📊 Visão Geral", 
        "⚔️ Comparador", 
        "🧠 IA", 
        "🧪 Laboratório", 
        "📂 Base",
        "🛠️ Raio-X (Debug)"
    ])

    # ABA 1: VISÃO GERAL
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Produtos", len(df_filtered))
        # Se der erro no preço, mostra R$ 0,00 em vez de travar
        preco_medio = df_filtered['Preco_Num'].mean()
        c2.metric("Preço Médio", f"R$ {preco_medio:.2f}" if not pd.isna(preco_medio) else "R$ 0.00")
        c3.metric("Fontes", len(df_filtered['FONTE'].unique()))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1: 
            st.subheader("Dispersão de Preços")
            fig_fonte = px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", points="all")
            st.plotly_chart(fig_fonte, use_container_width=True)
        with col_g2:
            st.subheader("Volume por Categoria")
            fig_vol = px.pie(df_filtered, names='CATEGORIA')
            st.plotly_chart(fig_vol, use_container_width=True)

    # ABA 2: COMPARADOR
    with tab2:
        st.header("⚔️ Comparador")
        termo = st.text_input("Buscar Produto:", placeholder="Ex: Vaso Robert")
        if termo:
            prods = df_filtered['PRODUTO'].dropna().unique()
            matches = process.extract(termo, prods, limit=30, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 55]
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
            
            if not df_comp.empty:
                st.metric("Média Shopee", f"R$ {df_comp[df_comp['FONTE']=='Shopee']['Preco_Num'].mean():.2f}")
                st.metric("Média Elo7", f"R$ {df_comp[df_comp['FONTE']=='Elo7']['Preco_Num'].mean():.2f}")
                fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", hover_data=["PRODUTO"])
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True)
            else:
                st.warning("Nada encontrado.")

    # ABA 3, 4, 5 (Mantidas Simplificadas para não ocupar espaço)
    with tab3: st.info("Módulo IA Ativo.")
    with tab4: st.info("Laboratório Ativo.")
    with tab5: st.dataframe(df_filtered)

    # --- ABA DEBUG (NOVA) ---
    with tab_debug:
        st.header("🛠️ Diagnóstico de Dados Brutos")
        st.write("Aqui você vê exatamente como o Python está lendo suas planilhas.")
        st.write("Se a coluna 'Preco_Num' estiver zerada, o problema é o formato na planilha.")
        st.dataframe(df_filtered.head(50))

else:
    st.error("⚠️ As planilhas parecem vazias ou inacessíveis.")
    st.write("Verifique se publicou as abas corretamente como CSV.")
