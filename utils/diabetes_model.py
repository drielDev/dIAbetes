import pandas as pd
from src.train import train_models

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

_model_cache = {}


def carregar_modelo():
    if _model_cache:
        return _model_cache

    info = train_models()

    _model_cache.update({
        "modelo": info["best_model"],
        "nome_modelo": info["best_model_name"],
        "metricas": info["test_metrics"],  # 🔥 agora usa TEST
        "scaler": info["scaler"],
    })

    return _model_cache


def prever(dados_paciente: dict) -> dict:
    cache = carregar_modelo()
    modelo = cache["modelo"]
    scaler = cache["scaler"]
    df = pd.DataFrame([dados_paciente])
    X = df[FEATURE_NAMES]
    X_scaled = scaler.transform(X)
    pred = int(modelo.predict(X_scaled)[0])

    if hasattr(modelo, "predict_proba"):
        prob = float(modelo.predict_proba(X)[0][1])
    else:
        prob = float(modelo.decision_function(X)[0])

    return {
        "predicao": pred,
        "label": "Positivo para Diabetes" if pred == 1 else "Negativo para Diabetes",
        "score": prob,
        "modelo_usado": cache["nome_modelo"],
        "metricas_modelo": cache["metricas"],
        "dados_entrada": dados_paciente,
    }


# Compatibilidade opcional com app antigo
def treinar_modelo():
    return carregar_modelo()