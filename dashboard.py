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

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="BCRUZ 3D Enterprise", layout="wide", page_icon="🏭")

# --- 2. LINKS DE DADOS (Elo7 + Shopee) ---
URL_ELO7 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=1574041650&single=true&output=csv"
URL_SHOPEE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRtLCFvhbktUToSC6XCCtsEk-Fats-FqW8Nv_fG9AG_8fWfu7pMIFq7Zo0m0oS37r0coiqQyn9ZWc0F/pub?gid=307441420&single=true&output=csv"

# --- FUNÇÃO DE LIMPEZA DE PREÇO (BLINDADA) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "": return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    
    texto = str(valor).upper().strip()
    # Remove R$, espaços e caracteres estranhos, mantém dígitos, vírgula e ponto
    texto = re.sub(r'[^\d,.]', '', texto)
    
    try:
        # Lógica Brasil: se tem vírgula, é decimal. (1.200,50 -> 1200.50)
        if ',' in texto:
            texto = texto.replace('.', '').replace(',', '.')
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
            
            if temp_df.empty:
                logs.append(f"⚠️ {f['nome']} vazia.")
                continue

            # Mapeamento de Colunas
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE", "VALOR"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), None)
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), None)
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)

            # Normalização
            temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            temp_df['FONTE'] = f["nome"]
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat else "Geral"
            temp_df['LINK'] = temp_df[col_link] if col_link else "#"
            
            # Preço
            if col_preco: temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else: temp_df['Preco_Num'] = 0.0

            # Prazo
            if col_prazo:
                temp_df['Prazo_Orig'] = temp_df[col_prazo].fillna("Normal")
                def get_days(t):
                    t = str(t).upper()
                    if "IMEDIATO" in t or "PRONTA" in t: return 1
                    m = re.search(r'(\d+)', t)
                    return int(m.group(1)) if m else 15
                temp_df['Dias_Producao'] = temp_df['Prazo_Orig'].apply(get_days)
            else:
                temp_df['Dias_Producao'] = 15
                temp_df['Prazo_Orig'] = "N/A"

            # Seleção Final
            cols = ['PRODUTO', 'Preco_Num', 'FONTE', 'CATEGORIA', 'LINK', 'Dias_Producao', 'Prazo_Orig']
            for c in cols: 
                if c not in temp_df.columns: temp_df[c] = ""
            
            dfs.append(temp_df[cols])
            logs.append(f"✅ {f['nome']} OK")

        except Exception as e:
            logs.append(f"❌ {f['nome']} Erro: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), logs

# --- EXECUÇÃO ---
df, logs = carregar_dados()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Painel de Controle")
st.sidebar.code("\n".join(logs)) # Log técnico discreto

if not df.empty:
    filtro_fonte = st.sidebar.multiselect("Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    df_filtered = df[df['FONTE'].isin(filtro_fonte)]
    
    cats = st.sidebar.multiselect("Categorias", df_filtered['CATEGORIA'].unique())
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- TABS (AQUI ESTÁ A ABA 5 RESTAURADA) ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Mercado", 
        "⚔️ Comparador", 
        "🧠 Clusters IA", 
        "☁️ Palavras-Chave",
        "💡 CRIADOR DE ANÚNCIOS", # <--- ELA VOLTOU
        "📂 Dados Brutos"
    ])

    # 1. MERCADO
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Produtos", len(df_filtered))
        c2.metric("Média de Preço", f"R$ {df_filtered['Preco_Num'].mean():.2f}")
        c3.metric("Pronta Entrega", len(df_filtered[df_filtered['Dias_Producao'] <= 2]))
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Preço por Fonte")
            st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE"), use_container_width=True)
        with col_g2:
            st.subheader("Categorias")
            st.plotly_chart(px.pie(df_filtered, names="CATEGORIA"), use_container_width=True)

    # 2. COMPARADOR
    with tab2:
        st.header("Comparador de Preços (Fuzzy Search)")
        termo = st.text_input("Digite o nome do produto para comparar (ex: Vaso Robert):")
        
        if termo:
            # Busca inteligente
            prods = df_filtered['PRODUTO'].unique()
            matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
            # Aceita similaridade > 50 (Shopee tem nomes sujos)
            similares = [x[0] for x in matches if x[1] > 50]
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
            
            if not df_comp.empty:
                st.plotly_chart(px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", hover_data=['PRODUTO']), use_container_width=True)
                st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True, use_container_width=True)
            else:
                st.warning("Nenhum produto similar encontrado.")

    # 3. CLUSTERS IA
    with tab3:
        st.subheader("Segmentação Automática (Preço x Prazo)")
        if len(df_filtered) > 10:
            X = df_filtered[['Preco_Num', 'Dias_Producao']].fillna(0)
            kmeans = KMeans(n_clusters=3, n_init=10).fit(X)
            df_filtered['Cluster'] = kmeans.labels_
            st.plotly_chart(px.scatter(df_filtered, x="Dias_Producao", y="Preco_Num", color=df_filtered['Cluster'].astype(str), title="Onde estão as oportunidades?"), use_container_width=True)
        else:
            st.info("Dados insuficientes para IA.")

    # 4. PALAVRAS
    with tab4:
        st.subheader("Termos mais usados nos títulos")
        texto = " ".join(df_filtered['PRODUTO'].astype(str))
        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla", "kit", "un", "com", "em", "o", "a"])
        try:
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=sw).generate(texto)
            fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)
        except: pass

    # 5. GERADOR DE TÍTULOS (A MÁGICA ESTÁ AQUI)
    with tab5:
        st.header("💡 IA: Gerador de Títulos Vencedores")
        st.markdown("Analisa o vocabulário dos concorrentes para criar títulos de alta conversão.")
        
        keyword = st.text_input("Produto Foco (Ex: Suporte Fone):", "Vaso")
        
        if keyword:
            # 1. Filtra concorrentes
            df_concorrentes = df[df['PRODUTO'].str.contains(keyword, case=False, na=False)]
            
            if not df_concorrentes.empty:
                # 2. Extrai palavras quentes
                texto_raw = " ".join(df_concorrentes['PRODUTO'].astype(str))
                palavras = [p for p in re.findall(r'\w+', texto_raw.lower()) if p not in sw and len(p) > 2]
                top_words = [x[0].title() for x in Counter(palavras).most_common(5)]
                
                st.success(f"Palavras-chave detectadas no nicho: {', '.join(top_words)}")
                
                # 3. Gera Títulos
                st.subheader("🏆 Sugestões de Títulos:")
                
                # Fórmula 1: SEO + Benefício
                t1 = f"{keyword.title()} 3D {' '.join(top_words[:2])} - Alta Qualidade"
                # Fórmula 2: Urgência (Flash)
                t2 = f"{top_words[0]} {keyword.title()} {' '.join(top_words[2:3])} - ENVIO IMEDIATO ⚡"
                # Fórmula 3: Kit/Promoção
                t3 = f"Kit {keyword.title()} Personalizado {' '.join(top_words[:3])}"
                
                st.code(t1, language="text")
                st.code(t2, language="text")
                st.code(t3, language="text")
                
                st.info(f"Baseado na análise de {len(df_concorrentes)} produtos concorrentes.")
            else:
                st.warning("Não encontrei dados suficientes para esse termo. Tente uma palavra mais genérica.")

    # 6. DADOS
    with tab6:
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Erro Crítico: Não foi possível carregar os dados. Verifique os links CSV no código.")
