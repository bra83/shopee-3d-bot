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

# --- 3. LIMPEZA DE PREÇO AGRESSIVA (ANTI-OUTLIER) ---
def limpar_preco(valor):
    if pd.isna(valor) or str(valor).strip() == "": return 0.0
    
    # Se já for número
    if isinstance(valor, (int, float)): 
        val = float(valor)
    else:
        # Tratamento de String (A selva dos formatos)
        texto = str(valor).upper().strip()
        # Limpa tudo que não é número, ponto ou vírgula
        texto = re.sub(r'[^\d,.]', '', texto)
        
        try:
            # LÓGICA DE DETECÇÃO DE FORMATO
            if ',' in texto: 
                # Formato BR (39,90 ou 1.200,00)
                texto = texto.replace('.', '') # Remove milhar (1.200 -> 1200)
                texto = texto.replace(',', '.') # Vírgula vira ponto (39,90 -> 39.90)
            elif texto.count('.') == 1:
                # Formato Misto (Pode ser 39.90 OU 1.200)
                partes = texto.split('.')
                if len(partes[1]) == 3: 
                    # Se tem 3 casas decimais (1.200), é milhar
                    texto = texto.replace('.', '')
                # Se tem 2 casas (39.90), deixa o ponto quieto
            
            val = float(texto)
        except: 
            return 0.0

    # --- GUILHOTINA DE ERROS ÓBVIOS ---
    # Ninguém vende um vaso de 3D print por 20 mil. Se for > 1500, ignora (deve ser a impressora).
    if val > 1500.0: 
        return 0.0 # Zera para ser filtrado depois
        
    return val

@st.cache_data(ttl=60)
def carregar_dados():
    dfs = []
    fontes = [{"url": URL_ELO7, "nome": "Elo7"}, {"url": URL_SHOPEE, "nome": "Shopee"}]

    for f in fontes:
        try:
            temp_df = pd.read_csv(f["url"], on_bad_lines='skip', dtype=str)
            temp_df.columns = [str(c).strip().upper() for c in temp_df.columns]
            
            if temp_df.empty: continue

            # Mapeamento
            col_prod = next((c for c in temp_df.columns if any(x in c for x in ["PRODUT", "NOME", "TITULO"])), "PRODUTO")
            col_preco = next((c for c in temp_df.columns if any(x in c for x in ["(R$)", "PREÇO", "PRICE"])), None)
            col_cat = next((c for c in temp_df.columns if "CATEG" in c), "Geral")
            col_link = next((c for c in temp_df.columns if "LINK" in c or "URL" in c), "#")
            col_prazo = next((c for c in temp_df.columns if "PRAZO" in c or "FLASH" in c), None)

            temp_df = temp_df.rename(columns={col_prod: 'PRODUTO'})
            temp_df['FONTE'] = f["nome"]
            temp_df['CATEGORIA'] = temp_df[col_cat] if col_cat in temp_df.columns else "Geral"
            temp_df['LINK'] = temp_df[col_link] if col_link in temp_df.columns else "#"
            
            if col_preco: 
                temp_df['Preco_Num'] = temp_df[col_preco].apply(limpar_preco)
            else: 
                temp_df['Preco_Num'] = 0.0

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
            
            # FILTRO LIMPO: Remove preços zerados ou inválidos (que a Guilhotina pegou)
            temp_df = temp_df[temp_df['Preco_Num'] > 0.1]
            
            dfs.append(temp_df[cols])

        except: pass

    # Concatena tudo
    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # --- FILTRO ESTATÍSTICO AUTOMÁTICO (NOVO!) ---
    # Remove os 2% mais caros (Outliers extremos que sobraram)
    if not final_df.empty:
        corte_superior = final_df['Preco_Num'].quantile(0.98) # Pega o teto de 98%
        final_df = final_df[final_df['Preco_Num'] <= corte_superior]
        
    return final_df

df = carregar_dados()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Centro de Comando")

if not df.empty:
    # FILTRO DE PREÇO MANUAL (Para você ajustar fino)
    st.sidebar.markdown("### 🔍 Filtro de Ticket")
    # Pega o preço máximo real que sobrou nos dados para configurar o slider
    max_val = float(df['Preco_Num'].max())
    # O padrão do slider agora é R$ 500,00 para garantir visualização limpa
    preco_max = st.sidebar.slider("Teto de Preço (R$)", 0.0, max_val, min(500.0, max_val))
    
    fontes_sel = st.sidebar.multiselect("Fontes", df['FONTE'].unique(), default=df['FONTE'].unique())
    
    df_filtered = df[
        (df['FONTE'].isin(fontes_sel)) & 
        (df['Preco_Num'] <= preco_max)
    ]
    
    cats = st.sidebar.multiselect("Categorias", df_filtered['CATEGORIA'].unique())
    if cats: df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(cats)]

    # --- LAYOUT ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visão Geral", "⚔️ Comparador", "🧠 IA & Insights", "🧪 Laboratório", "💡 Criador", "📂 Dados"
    ])

    # 1. GERAL
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Produtos", len(df_filtered))
        
        # Preço Médio Formatado BR
        media = df_filtered['Preco_Num'].mean()
        c2.metric("Ticket Médio", f"R$ {media:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        
        c3.metric("Fontes", len(df_filtered['FONTE'].unique()))
        c4.metric("Itens Flash", len(df_filtered[df_filtered['Logistica']=="⚡ FLASH"]))
        
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1: st.plotly_chart(px.box(df_filtered, x="FONTE", y="Preco_Num", color="FONTE", title="Distribuição de Preços (Limpa)"), use_container_width=True)
        with col_g2: st.plotly_chart(px.pie(df_filtered, names='CATEGORIA', title="Share de Categorias"), use_container_width=True)

    # 2. COMPARADOR
    with tab2:
        st.header("⚔️ Comparador de Preços")
        col_input, col_check = st.columns([3, 1])
        with col_input:
            termo = st.text_input("Filtrar Produto:", placeholder="Ex: Vaso Robert")
        with col_check:
            st.write("") 
            st.write("") 
            mostrar_tudo = st.checkbox("Ver Todos", value=False)
        
        df_comp = pd.DataFrame()

        if mostrar_tudo:
            df_comp = df_filtered
        elif termo:
            prods = df_filtered['PRODUTO'].unique()
            matches = process.extract(termo, prods, limit=50, scorer=fuzz.token_set_ratio)
            similares = [x[0] for x in matches if x[1] > 40] 
            df_comp = df_filtered[df_filtered['PRODUTO'].isin(similares)]
        
        if not df_comp.empty:
            cols_metrics = st.columns(len(df_comp['FONTE'].unique()) + 1)
            for i, fonte in enumerate(df_comp['FONTE'].unique()):
                media_local = df_comp[df_comp['FONTE']==fonte]['Preco_Num'].mean()
                fmt = f"R$ {media_local:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                cols_metrics[i].metric(f"Média {fonte}", fmt)

            fig_comp = px.scatter(df_comp, x="FONTE", y="Preco_Num", color="FONTE", size="Preco_Num", 
                                  hover_data=['PRODUTO'], title="Comparativo de Preços")
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(df_comp[['FONTE', 'PRODUTO', 'Preco_Num', 'LINK']], hide_index=True, use_container_width=True)
        else:
            if not mostrar_tudo: st.info("Busque um produto acima.")

    # 3. NUVENS
    with tab3:
        st.subheader("Nuvens de Inteligência")
        
        sw = set(STOPWORDS)
        sw.update(["de", "para", "3d", "pla", "com", "o", "a", "em", "do", "da", "kit", "un", "cm", "peças"])
        
        c_cloud1, c_cloud2 = st.columns(2)
        
        with c_cloud1:
            st.caption("☁️ MAIS FREQUENTES (O que todos vendem)")
            texto_geral = " ".join(df_filtered['PRODUTO'].astype(str))
            try:
                wc1 = WordCloud(width=400, height=300, background_color='white', stopwords=sw, colormap='Blues').generate(texto_geral)
                fig1, ax1 = plt.subplots(); ax1.imshow(wc1); ax1.axis("off"); st.pyplot(fig1)
            except: st.warning("Sem dados.")

        with c_cloud2:
            st.caption("💰 MAIOR VALOR AGREGADO (O que custa caro)")
            word_prices = {}
            for _, row in df_filtered.iterrows():
                # Normaliza para lower case para agrupar 'Vaso' e 'vaso'
                palavras = str(row['PRODUTO']).lower().split()
                for p in palavras:
                    if p not in sw and len(p) > 3:
                        if p not in word_prices: word_prices[p] = []
                        word_prices[p].append(row['Preco_Num'])
            
            if word_prices:
                avg_prices = {k: sum(v)/len(v) for k, v in word_prices.items() if len(v) > 1}
                if avg_prices:
                    wc2 = WordCloud(width=400, height=300, background_color='#222', colormap='Wistia', max_words=50).generate_from_frequencies(avg_prices)
                    fig2, ax2 = plt.subplots(); ax2.imshow(wc2); ax2.axis("off"); st.pyplot(fig2)
                else: st.warning("Dados insuficientes.")

    # 4. LAB
    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1: cx = st.selectbox("Eixo X", df_filtered.columns)
        with c2: cy = st.selectbox("Eixo Y", ['Preco_Num', 'Dias_Producao'])
        with c3: tp = st.selectbox("Tipo", ["Barras", "Dispersão", "Boxplot"])
        if tp == "Barras": st.plotly_chart(px.bar(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Dispersão": st.plotly_chart(px.scatter(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)
        elif tp == "Boxplot": st.plotly_chart(px.box(df_filtered, x=cx, y=cy, color="FONTE"), use_container_width=True)

    # 5. CRIADOR
    with tab5:
        st.header("Gerador de Títulos SEO")
        keyword = st.text_input("Produto:", "Vaso")
        if keyword:
            df_c = df[df['PRODUTO'].str.contains(keyword, case=False, na=False)]
            if not df_c.empty:
                txt = " ".join(df_c['PRODUTO'].astype(str))
                pals = [p for p in re.findall(r'\w+', txt.lower()) if p not in sw and len(p) > 2]
                top = [x[0].title() for x in Counter(pals).most_common(5)]
                st.success(f"Palavras-chave: {', '.join(top)}")
                st.code(f"{keyword.title()} 3D {' '.join(top[:2])} - Alta Qualidade")
            else: st.warning("Sem dados.")

    # 6. DADOS
    with tab6: st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("⚠️ Erro ao carregar dados. Verifique o Google Sheets.")
