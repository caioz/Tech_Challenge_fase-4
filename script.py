import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Painel da B3",
    layout ="wide"
)

st.header("**PAINEL DE PREÇO E DIVIDENDO DA B3**")

ticker = st.text_input('Digite o ticket da ação', 'BBAS3')
empresa = yf.Ticker(f"{ticker}.SA")

tickerDF = empresa.history(
    start="2014-01-01",
    end="2025-12-31",
)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Empresa:** {empresa.info.get('longName', 'N/A')}")
with col2:
    st.write(f"**Setor:** {empresa.info.get('industry', 'N/A')}")
with col3:
    st.write(f"**Preço Atual:** {empresa.info.get('currentPrice', 0):.2f}")
    
st.line_chart(tickerDF['Close'])
st.bar_chart(tickerDF['Dividends'])