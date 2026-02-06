import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ Intelligence 3D", layout="wide", page_icon="🖨️")

# COLE O LINK DE EXPORTAÇÃO CSV DA SUA PLANILHA AQUI
# (Arquivo > Compartilhar > Publicar na Web > CSV)
SHEET_CSV_URL = "https://script.google.com/macros/s/AKfycbzz1kNMVd7wDkem6Vrdb1v1sUyWyekpUWe8Dd-dI4VxgLqpFhJe9DmE6486apJ97dy6/exec"

@st.cache_data(ttl=60)
def carregar_dados():
    try:
        # Lê a planilha assumindo o cabeçalho novo
        df = pd.read_csv(SHEET_CSV_URL)
        
        # Limpeza de Preço (R$ 1.200,50 -> 1200.50)
        col_preco = 'PREÇO (R$)' # Nome exato que colocamos no Apps Script
        if col_preco in df.columns:
            df[col_preco] = df[col_preco].astype(str).str.replace('R$', '', regex=False)
            df[col_preco] = df[col_preco].str.replace('.', '', regex=False).str.replace(',', '.')
            df[col_preco] = pd.to_numeric(df[col_preco], errors='coerce').fillna(0)
        
        # Tratamento de Prazo (Coluna H)
        col_prazo = 'PRAZO DE PRODUÇÃO'
        if col_prazo in df.columns:
            df[col_prazo] = df[col_prazo].fillna("PADRÃO")
            # Cria flag de oportunidade
            df['OPORTUNIDADE'] = df[col_prazo].apply(
                lambda x: "🚨 ATAQUE (Lento)" if "DIAS" in str(x).upper() and int(re.search(r'\d+', str(x)).group(0) if re.search(r'\d+', str(x)) else 0) > 5 
                else ("⚡ CONCORRENTE (Rápido)" if "IMEDIATO" in str(x).upper() else "Normal")
            )
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# --- INTERFACE ---
st.title("🖨️ BCRUZ 3D - Central de Comando")
st.markdown("Análise de mercado para viabilidade da **Bambu Lab A1**.")

df = carregar_dados()

if not df.empty:
    # Sidebar de Filtros
    st.sidebar.header("Filtros Estratégicos")
    categorias = st.sidebar.multiselect("Categoria", df['CATEGORIA'].unique())
    filtro_prazo = st.sidebar.radio("Logística", ["Todos", "Apenas Lentos (>5 dias)", "Pronta Entrega"])

    # Aplica filtros
    df_filtered = df.copy()
    if categorias:
        df_filtered = df_filtered[df_filtered['CATEGORIA'].isin(categorias)]
    
    if filtro_prazo == "Apenas Lentos (>5 dias)":
        df_filtered = df_filtered[df_filtered['OPORTUNIDADE'] == "🚨 ATAQUE (Lento)"]
    elif filtro_prazo == "Pronta Entrega":
        df_filtered = df_filtered[df_filtered['OPORTUNIDADE'] == "⚡ CONCORRENTE (Rápido)"]

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Produtos Analisados", len(df_filtered))
    c2.metric("Preço Médio", f"R$ {df_filtered['PREÇO (R$)'].mean():.2f}")
    
    # Conta quantos são 'Lentos' (Oportunidade de Pronta Entrega)
    oportunidades = len(df_filtered[df_filtered['OPORTUNIDADE'] == "🚨 ATAQUE (Lento)"])
    c3.metric("Oportunidades de Ataque", oportunidades, help="Produtos que demoram mais de 5 dias para produzir")

    # Gráficos
    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("💰 Faixa de Preço por Prazo")
        fig_price = px.box(df_filtered, x="PRAZO DE PRODUÇÃO", y="PREÇO (R$)", points="all", color="OPORTUNIDADE")
        st.plotly_chart(fig_price, use_container_width=True)
        
    with col_g2:
        st.subheader("📊 Distribuição de Prazos")
        fig_pie = px.pie(df_filtered, names="PRAZO DE PRODUÇÃO", title="Market Share Logístico")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Tabela de Dados
    st.subheader("📋 Relatório de Inteligência")
    st.dataframe(
        df_filtered[['DATA CAPTURA', 'PRODUTO', 'PREÇO (R$)', 'PRAZO DE PRODUÇÃO', 'OPORTUNIDADE', 'LINK']],
        column_config={
            "LINK": st.column_config.LinkColumn("Link Elo7"),
            "PREÇO (R$)": st.column_config.NumberColumn(format="R$ %.2f")
        },
        hide_index=True
    )

else:
    st.warning("Aguardando dados do Robô...")
