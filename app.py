import streamlit as st
import os
from dotenv import load_dotenv

from utils.gemini_client import configurar_modelo, interpretar_exame
from utils.diabetes_model import carregar_modelo, prever
from utils.components import (
    renderizar_header,
    renderizar_pilares,
    renderizar_formulario_diabetes,
    renderizar_resposta,
    renderizar_placeholder,
    renderizar_resultado_modelo,
)


st.set_page_config(
    page_title="dIAbetes — IA para Diagnóstico de Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def carregar_css(caminho: str):
    with open(caminho, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css(os.path.join(os.path.dirname(__file__), "static", "style.css"))


load_dotenv()


def obter_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


# Treina o modelo na inicialização (cacheado)
if "modelo_treinado" not in st.session_state:
    with st.spinner("🔬 Treinando modelo de Machine Learning..."):
        st.session_state["modelo_treinado"] = carregar_modelo()

renderizar_header()
renderizar_pilares()

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    dados_paciente, analisar = renderizar_formulario_diabetes()

with col_result:
    st.markdown('<div class="card-title">💬 Resultado da Análise</div>',
                unsafe_allow_html=True)

    if analisar:
        api_key = obter_api_key()

        if not api_key:
            st.error(
                "⚠️ Chave de API não configurada. "
                "Defina a variável GEMINI_API_KEY no arquivo .env do projeto."
            )
        elif dados_paciente is None:
            st.warning(
                "📋 Por favor, preencha os dados clínicos para análise."
            )
        else:
            # Predição do modelo ML
            resultado_ml = prever(dados_paciente)
            renderizar_resultado_modelo(resultado_ml)

            # Interpretação via Gemini
            with st.spinner("🌸 Gerando interpretação personalizada..."):
                try:
                    client = configurar_modelo(api_key)
                    historico = st.session_state.get("historico", None)

                    dados_para_gemini = (
                        f"## Resultado do Modelo de Machine Learning\n"
                        f"Predição: {resultado_ml['label']}\n"
                        f"Score de decisão: {resultado_ml['score']:.4f}\n\n"
                        f"## Dados Clínicos da Paciente\n"
                        f"- Gestações: {dados_paciente['Pregnancies']}\n"
                        f"- Glicose: {dados_paciente['Glucose']} mg/dL\n"
                        f"- Pressão Arterial: {dados_paciente['BloodPressure']} mmHg\n"
                        f"- Espessura Dobra Cutânea: {dados_paciente['SkinThickness']} mm\n"
                        f"- Insulina: {dados_paciente['Insulin']} µU/mL\n"
                        f"- IMC (BMI): {dados_paciente['BMI']} kg/m²\n"
                        f"- Função Pedigree Diabetes: {dados_paciente['DiabetesPedigreeFunction']}\n"
                        f"- Idade: {dados_paciente['Age']} anos\n\n"
                        f"## Métricas do Modelo\n"
                        f"- Acurácia: {resultado_ml['metricas_modelo']['accuracy']:.2%}\n"
                        f"- Recall: {resultado_ml['metricas_modelo']['recall']:.2%}\n"
                        f"- F1-Score: {resultado_ml['metricas_modelo']['f1_score']:.2%}\n"
                    )

                    resposta, historico = interpretar_exame(
                        client, dados_para_gemini, "", historico
                    )
                    st.session_state["historico"] = historico
                    renderizar_resposta(resposta)

                except Exception as e:
                    erro_str = str(e).lower()
                    if "429" in erro_str or "resource_exhausted" in erro_str or "quota" in erro_str:
                        st.warning(
                            "⏳ Cota de uso da API temporariamente excedida. "
                            "Aguarde alguns segundos e tente novamente."
                        )
                    elif "401" in erro_str or "invalid" in erro_str or "api_key" in erro_str:
                        st.error(
                            "🔑 Chave de API inválida. "
                            "Verifique a variável GEMINI_API_KEY no arquivo .env."
                        )
                    else:
                        st.error(
                            "Ocorreu um erro ao processar sua solicitação. "
                            "Tente novamente em alguns instantes."
                        )
    else:
        renderizar_placeholder()
