import streamlit as st
import pandas as pd
import plotly.express as px
import re

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="BCRUZ 3D Admin", layout="wide", page_icon="🛡️")

# ⚠️ COLOQUE SEU LINK CSV AQUI
SHEET_CSV_URL = "https://script.google.com/macros/s/AKfycbw2MksECV2wTgCnJOxIQyZCKIz7pn1tyzjq1z7pQ6Bw2xdpw7SkfCUH81xHvwbjF6q8/exec"

@st.cache_data(ttl=60)
def carregar_dados():
    try:
        df = pd.read_csv(SHEET_CSV_URL)
        
        # --- 1. CORREÇÃO DE COLUNAS (BLINDAGEM) ---
        # Normaliza nomes das colunas para evitar erro se mudar "Preço" para "PREÇO"
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Procura qual coluna é o PREÇO
        col_preco = next((c for c in df.columns if "PREÇO" in c or "PRICE" in c), None)
        
        if col_preco:
            # Limpa R$ e converte
            df['Preco_Num'] = df[col_preco].astype(str).str.replace('R$', '', regex=False)
            df['Preco_Num'] = df['Preco_Num'].str.replace('.', '', regex=False).str.replace(',', '.')
            df['Preco_Num'] = pd.to_numeric(df['Preco_Num'], errors='coerce').fillna(0)
        else:
            df['Preco_Num'] = 0.0 # Valor padrão para não travar
            
        # Procura qual coluna é o PRAZO
        col_prazo = next((c for c in df.columns if "PRAZO" in c or "FLASH" in c), None)
        
        if col_prazo:
            df['Prazo_Limpo'] = df[col_prazo].fillna("PADRÃO")
        else:
            df['Prazo_Limpo'] = "NÃO DETECTADO"

        return df

    except Exception as e:
        st.error(f"Erro ao ler planilha: {e}")
        return pd.DataFrame()

df = carregar_dados()

if not df.empty:
    st.title("🛡️ BCRUZ Dashboard (Modo Seguro)")
    
    # Filtros
    st.sidebar.header("Filtros")
    
    # Verifica se a coluna CATEGORIA existe antes de filtrar
    col_cat = next((c for c in df.columns if "CATEG" in c), None)
    if col_cat:
        cats = st.sidebar.multiselect("Categoria", df[col_cat].unique())
        if cats: df = df[df[col_cat].isin(cats)]

    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Produtos", len(df))
    c2.metric("Média de Preço", f"R$ {df['Preco_Num'].mean():.2f}")
    
    # Conta quantos são 'IMEDIATO'
    flash_count = df['Prazo_Limpo'].astype(str).str.contains("IMEDIATO", case=False).sum()
    c3.metric("Pronta Entrega Detectados", flash_count)

    st.markdown("---")
    st.subheader("📋 Dados Carregados")
    st.dataframe(df, use_container_width=True)

else:
    st.warning("⚠️ Planilha vazia ou link CSV incorreto.")
