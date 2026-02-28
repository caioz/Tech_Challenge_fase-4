# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Previsor de Obesidade (Formulário -> Resultado)", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_model(path="model_pipeline.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def predict_with_model(model, input_df):
    """Retorna (pred_label, prob_df or None)"""
    if model is None:
        return None, None
    pred = model.predict(input_df)
    proba = None
    try:
        proba = model.predict_proba(input_df)
    except Exception:
        proba = None
    if proba is not None and hasattr(model, "classes_"):
        prob_df = pd.DataFrame(proba, columns=model.classes_)
    elif proba is not None:
        prob_df = pd.DataFrame(proba)
    else:
        prob_df = None
    return pred[0], prob_df

# ---------- Load model (optional) ----------
model = load_model("model_pipeline.joblib")
if model is None:
    st.sidebar.warning("Modelo não encontrado: previsões estarão desativadas. Coloque model_pipeline.joblib na pasta.")

# ---------- Initialize session state ----------
if "page" not in st.session_state:
    st.session_state.page = "form"   # "form" or "result"
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None

# ---------- Header ----------
st.markdown(
    "<h1 style='text-align:center;color:#0b6efd'>Previsor de Obesidade — Formulário</h1>",
    unsafe_allow_html=True
)
st.write("---")

# ---------- FORM PAGE ----------
def render_form():
    st.subheader("Preencha os dados do indivíduo")
    with st.form("form_predict", clear_on_submit=False):
        cols = st.columns(3)
        gender = cols[0].selectbox("Gênero", options=["Female","Male"])
        age = cols[0].number_input("Idade", min_value=10, max_value=100, value=30)
        height = cols[1].number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        weight = cols[1].number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
        family_history = cols[2].selectbox("Histórico familiar (obesidade)?", options=["yes","no"])
        FAVC = cols[2].selectbox("Consome alimentos calóricos?", options=["yes","no"])
        FCVC = st.selectbox("Frequência consumo vegetais (FCVC)", options=[1,2,3])
        NCP = st.selectbox("Número de refeições principais (NCP)", options=[1,2,3,4])
        CAEC = st.selectbox("Come entre refeições (CAEC)", options=["no","Sometimes","Frequently","Always"])
        SMOKE = st.selectbox("Fuma?", options=["yes","no"])
        CH2O = st.selectbox("Consumo de água (CH2O)", options=[1,2,3])
        SCC = st.selectbox("Monitora calorias (SCC)?", options=["yes","no"])
        FAF = st.selectbox("Atividade física (FAF)", options=[0,1,2,3])
        TUE = st.selectbox("Tempo com eletrônicos (TUE)", options=[0,1,2])
        CALC = st.selectbox("Consumo álcool (CALC)", options=["no","Sometimes","Frequently","Always"])
        MTRANS = st.selectbox("Transporte (MTRANS)", options=["Automobile","Public_Transport","Walking","Motorbike","Bike"])

        submitted = st.form_submit_button("Enviar e ver previsão")
        if submitted:
            # montar dataframe com mesmas colunas que o pipeline espera
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "family_history": family_history,
                "FAVC": FAVC,
                "FCVC": FCVC,
                "NCP": NCP,
                "CAEC": CAEC,
                "SMOKE": SMOKE,
                "CH2O": CH2O,
                "SCC": SCC,
                "FAF": FAF,
                "TUE": TUE,
                "CALC": CALC,
                "MTRANS": MTRANS
            }])
            # armazenar na sessão
            st.session_state.last_input = input_df
            # obter predição (se houver modelo)
            if model is not None:
                label, prob_df = predict_with_model(model, input_df)
                st.session_state.last_prediction = label
                st.session_state.last_probs = prob_df
            else:
                st.session_state.last_prediction = None
                st.session_state.last_probs = None
            st.session_state.page = "result"
            # força recarregar (para trocar de tela)
            st.rerun()

# ---------- RESULT PAGE ----------
def render_result():
    st.subheader("Resultado da Previsão")
    if st.session_state.last_input is None:
        st.info("Nenhuma previsão foi feita ainda. Volte ao formulário.")
        if st.button("Voltar ao formulário"):
            st.session_state.page = "form"
            st.rerun()
        return

    input_df = st.session_state.last_input.copy()
    st.write("Dados enviados:")
    st.table(input_df.T)

    # mostrar predição
    if st.session_state.last_prediction is None:
        st.warning("Modelo não disponível ou não carregado — sem predição.")
    else:
        st.success(f"Classe prevista: *{st.session_state.last_prediction}*")
        if st.session_state.last_probs is not None:
            st.write("Probabilidades por classe:")
            st.dataframe(st.session_state.last_probs.T)

    # Gráficos relacionados (exemplo simples)
    st.write("---")
    st.subheader("Gráficos relacionados à entrada")

    # Exemplo 1: IMC calculado daquela entrada
    w = float(input_df["Weight"].iloc[0])
    h = float(input_df["Height"].iloc[0])
    imc_val = round(w / (h**2), 2) if h > 0 else np.nan
    st.metric("IMC calculado", f"{imc_val:.2f}")

    # Exemplo 2: Gráfico de barras com probabilidades (se tiver)
    if st.session_state.last_probs is not None:
        prob_df = st.session_state.last_probs.T.reset_index()
        prob_df.columns = ["Classe", "Probabilidade"]
        fig = px.bar(prob_df, x="Classe", y="Probabilidade", text=prob_df["Probabilidade"].map("{:.2f}".format))
        st.plotly_chart(fig, use_container_width=True)

    # Exemplo 3: Simples gráfico ilustrativo (se você quiser, pode ajustar para base real)
    st.write("Exemplo: comparação do IMC da pessoa com médias hipotéticas por classe (ilustrativo).")
    # aqui você usaria médias reais calculadas no dataset; como exemplo, uso valores fictícios:
    example_means = {
        "Insufficient_Weight": 18,
        "Normal_Weight": 22,
        "Overweight_Level_I": 26,
        "Overweight_Level_II": 28,
        "Obesity_Type_I": 32,
        "Obesity_Type_II": 36,
        "Obesity_Type_III": 42
    }
    df_means = pd.DataFrame({
        "Classe": list(example_means.keys()),
        "IMC_medio": list(example_means.values())
    })
    df_means = df_means.sort_values("IMC_medio", ascending=True)
    fig2 = px.bar(df_means, x="IMC_medio", y="Classe", orientation="h", text="IMC_medio")
    # adicionar linha do IMC do usuário no gráfico:
    fig2.add_vline(x=imc_val, line_dash="dash", annotation_text="IMC pessoa", annotation_position="top right")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("---")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Voltar e editar formulário"):
            st.session_state.page = "form"
            st.rerun()
    with col2:
        if st.button("Nova previsão (limpar)"):
            st.session_state.last_input = None
            st.session_state.last_prediction = None
            st.session_state.last_probs = None
            st.session_state.page = "form"
            st.rerun()


# ---------- Main router ----------
if st.session_state.page == "form":
    render_form()
else:
    render_result()