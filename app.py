# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Previsor de Obesidade (Formulário -> Resultado)", layout="wide")

# ---------- Pipeline ----------
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

# ---------- Carregar modelo) ----------
model = load_model("model_pipeline.joblib")
if model is None:
    st.sidebar.warning("Modelo não encontrado: previsões estarão desativadas. Coloque model_pipeline.joblib na pasta.")

# ---------- inicializar Sesão ----------
if "page" not in st.session_state:
    st.session_state.page = "form"   
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None

# ---------- Cabeçalho ----------
st.markdown(
    "<h1 style='text-align:center;color:#0b6efd'>Previsor de Obesidade — Formulário</h1>",
    unsafe_allow_html=True
)
st.write("---")

# ---------- Dic ----------

gender_map = {
    "Feminino": "Female",
    "Masculino": "Male"
}

yes_no_map = {
    "Sim": "yes",
    "Não": "no"
}

caec_map = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always"
}

fcvc_map = {
    "Raramente": 1,
    "Às vezes": 2,
    "Sempre": 3
}

ch20_map = {
    "Raramente": 1,
    "Às vezes": 2,
    "Sempre": 3
}
calc_map = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always"
}

mtrans_map = {
    "Automóvel": "Automobile",
    "Transporte Público": "Public_Transport",
    "Caminhada": "Walking",
    "Motocicleta": "Motorbike",
    "Bicicleta": "Bike"
}

ncp_map ={
    "uma refeição": 1,
    "Duas": 2,
    "Três": 3,
    "Quatro ou mais": 4

}

faf_map = {
    "Nenhuma": 0,
    "uma a duas vezes por semana": 1,
    "Três a Quatro vezes por semana": 2,
    "Cinco ou mais vezes por semana": 3,
}

tue_map = {
    "0 a duas horas por dia": 0,
    "De Três a Cinco horas por dia": 1,
    "Mais de cinco horas por dia": 2
}

# ---------- PÁGINA ----------
def render_form():
    st.subheader("Preencha os dados do indivíduo")
    with st.form("form_predict", clear_on_submit=False):
        cols = st.columns(3)
        gender = cols[0].selectbox("Gênero", options=list(gender_map.keys()))
        age = cols[0].number_input("Idade", min_value=10, max_value=100, value=30)
        height = cols[1].number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        weight = cols[1].number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
        family_history = cols[2].selectbox("Histórico familiar (obesidade)?", options=list(yes_no_map.keys()))
        FAVC = cols[2].selectbox("Consome alimentos calóricos?", options=list(yes_no_map.keys()))
        FCVC = st.selectbox("Frequência consumo vegetais (FCVC)",options=list(fcvc_map.keys()))
        NCP = st.selectbox("Número de refeições principais (NCP)", options=list(ncp_map.keys()))
        CAEC = st.selectbox("Come entre refeições (CAEC)", options=list(caec_map.keys()))
        SMOKE = st.selectbox("Fuma?", options=list(yes_no_map.keys()))
        CH2O = st.selectbox("Consumo de água (CH2O)", options=list(ch20_map.keys()))
        SCC = st.selectbox("Monitora calorias (SCC)?", options=list(yes_no_map.keys()))
        FAF = st.selectbox("Atividade física (FAF)", options=list(faf_map.keys()))
        TUE = st.selectbox("Tempo com eletrônicos (TUE)", options=list(tue_map.keys()))
        CALC = st.selectbox("Consumo álcool (CALC)", options=list(calc_map.keys()))
        MTRANS = st.selectbox("Transporte (MTRANS)", options=list(mtrans_map.keys()))

        submitted = st.form_submit_button("Enviar e ver previsão")
        if submitted:
            # montar dataframe com mesmas colunas que o pipeline
            input_df = pd.DataFrame([{
            "Gender": gender_map[gender],
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": yes_no_map[family_history],
            "FAVC": yes_no_map[FAVC],
            "FCVC": fcvc_map[FCVC],
            "NCP": ncp_map[NCP],
            "CAEC": caec_map[CAEC],
            "SMOKE": yes_no_map[SMOKE],
            "CH2O": ch20_map[CH2O],
            "SCC": yes_no_map[SCC],
            "FAF": faf_map[FAF],
            "TUE": TUE,
            "CALC": calc_map[CALC],
            "MTRANS": mtrans_map[MTRANS]
            }])
            # armazenar na sessão
            st.session_state.last_input = input_df
            # obter predição 
            if model is not None:
                label, prob_df = predict_with_model(model, input_df)
                st.session_state.last_prediction = label
                st.session_state.last_probs = prob_df
            else:
                st.session_state.last_prediction = None
                st.session_state.last_probs = None
            st.session_state.page = "result"
            # força recarregar 
            st.rerun()

# ---------- Resultados da Página ----------
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

    # Gráficos relacionados
    st.write("---")
    st.subheader("Gráficos relacionados à entrada")

    # Exemplo 1: IMC calculado daquela entrada
    w = float(input_df["Weight"].iloc[0])
    h = float(input_df["Height"].iloc[0])
    imc_val = round(w / (h**2), 2) if h > 0 else np.nan
    st.metric("IMC calculado", f"{imc_val:.2f}")

    # Exemplo 2: Gráfico de barras com probabilidades
    if st.session_state.last_probs is not None:
        prob_df = st.session_state.last_probs.T.reset_index()
        prob_df.columns = ["Classe", "Probabilidade"]
        fig = px.bar(prob_df, x="Classe", y="Probabilidade", text=prob_df["Probabilidade"].map("{:.2f}".format))
        st.plotly_chart(fig, use_container_width=True)

    # Exemplo 3: Simples gráfico ilustrativo
    st.write("Exemplo: comparação do IMC da pessoa com médias hipotéticas por classe (ilustrativo).")
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
    # adicionar linha do IMC no gráfico:
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



if st.session_state.page == "form":
    render_form()
else:
    render_result()