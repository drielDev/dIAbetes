import streamlit as st


def renderizar_header():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🩺 dIAbetes</div>
        <div class="header-subtitle">
            Sistema inteligente de apoio ao diagnóstico de diabetes —
            combinando Machine Learning e IA Generativa para uma análise
            acolhedora, clara e orientativa.
        </div>
    </div>
    """, unsafe_allow_html=True)


def renderizar_pilares():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">🤖</div>
            <div class="pillar-title">Modelo de ML</div>
            <div class="pillar-text">
                SGDClassifier otimizado por Algoritmo Genético,
                treinado com foco em recall para minimizar falsos negativos.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">💜</div>
            <div class="pillar-title">Interpretação Humanizada</div>
            <div class="pillar-text">
                A IA Generativa (Gemini) traduz os resultados do modelo
                em linguagem acolhedora e de fácil entendimento.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="pillar-card">
            <div class="pillar-icon">🎯</div>
            <div class="pillar-title">Orientações Práticas</div>
            <div class="pillar-text">
                Próximos passos claros e recomendações compatíveis com a
                realidade da paciente, incluindo prevenção e acompanhamento.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


def renderizar_formulario_diabetes():
    st.markdown("""
    <div class="card-title">📋 Dados Clínicos da Paciente</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 Preencha os dados clínicos abaixo. Esses são os mesmos parâmetros
        utilizados pelo modelo de Machine Learning para avaliar o risco de
        diabetes.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        pregnancies = st.number_input(
            "🤰 Gestações",
            min_value=0, max_value=20, value=0, step=1,
            help="Número total de gestações"
        )
        glucose = st.number_input(
            "🩸 Glicose (mg/dL)",
            min_value=0.0, max_value=300.0, value=100.0, step=1.0,
            help="Concentração de glicose plasmática (teste de tolerância)"
        )
        blood_pressure = st.number_input(
            "💉 Pressão Arterial (mmHg)",
            min_value=0.0, max_value=200.0, value=72.0, step=1.0,
            help="Pressão arterial diastólica"
        )
        skin_thickness = st.number_input(
            "📏 Espessura Dobra Cutânea (mm)",
            min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            help="Espessura da dobra cutânea do tríceps"
        )

    with col_b:
        insulin = st.number_input(
            "💊 Insulina (µU/mL)",
            min_value=0.0, max_value=900.0, value=80.0, step=1.0,
            help="Insulina sérica de 2 horas"
        )
        bmi = st.number_input(
            "⚖️ IMC (kg/m²)",
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Índice de Massa Corporal"
        )
        dpf = st.number_input(
            "🧬 Função Pedigree Diabetes",
            min_value=0.0, max_value=3.0, value=0.5, step=0.01,
            help="Índice de histórico familiar de diabetes"
        )
        age = st.number_input(
            "🎂 Idade (anos)",
            min_value=1, max_value=120, value=30, step=1,
            help="Idade da paciente em anos"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    analisar = st.button("🔍  Analisar Risco de Diabetes", use_container_width=True)

    dados = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    return dados, analisar


def renderizar_resultado_modelo(resultado: dict):
    predicao = resultado["predicao"]
    label = resultado["label"]
    metricas = resultado["metricas_modelo"]

    if predicao == 1:
        cor = "#dc2626"
        icone = "⚠️"
        bg = "#fef2f2"
        borda = "#f87171"
    else:
        cor = "#16a34a"
        icone = "✅"
        bg = "#f0fdf4"
        borda = "#4ade80"

    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 5px solid {borda};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 1.3rem; font-weight: 700; color: {cor}; margin-bottom: 0.5rem;">
            {icone} Predição do Modelo: {label}
        </div>
        <div style="font-size: 0.9rem; color: #6b7280;">
            Acurácia: {metricas['accuracy']:.1%} · Recall: {metricas['recall']:.1%} · F1: {metricas['f1_score']:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)


def renderizar_resposta(resposta: str):
    st.markdown(resposta)

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Aviso importante:</strong> Esta análise é
        gerada por Inteligência Artificial e tem caráter
        <strong>exclusivamente informativo e acadêmico</strong>.
        Não substitui a avaliação, o diagnóstico ou o
        tratamento de um(a) profissional de saúde.
        Consulte sempre seu(sua) médico(a).
    </div>
    """, unsafe_allow_html=True)


def renderizar_placeholder():
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #9ca3af;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🩺</div>
        <div style="font-size: 1.2rem; font-weight: 500; color: #6b7280; margin-bottom: 0.5rem;">
            O resultado da análise aparecerá aqui
        </div>
        <div style="font-size: 0.95rem;">
            Preencha os dados clínicos ao lado e clique em
            <strong>"Analisar Risco de Diabetes"</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
