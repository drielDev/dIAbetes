from google import genai
from google.genai import types
from utils.prompts import SYSTEM_PROMPT

MODEL = "gemini-2.5-flash"


def configurar_modelo(api_key: str):
    """Configura e retorna o cliente Gemini."""
    client = genai.Client(api_key=api_key)
    return client


def interpretar_exame(client, dados_exame: str, contexto_paciente: str,
                      historico: list | None = None) -> tuple[str, list]:
    """Envia dados do exame e contexto para o Gemini e retorna a interpretação.
    Mantém histórico de conversa para permitir perguntas de acompanhamento.
    """
    if historico is None:
        historico = []

    prompt = _montar_prompt(dados_exame, contexto_paciente)
    historico.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    resposta = client.models.generate_content(
        model=MODEL,
        contents=historico,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=4096,
        ),
    )
    texto = resposta.text
    historico.append(types.Content(role="model", parts=[types.Part(text=texto)]))
    return texto, historico


def _montar_prompt(dados_exame: str, contexto_paciente: str) -> str:
    """Monta o prompt final combinando dados do modelo ML e contexto da paciente."""
    partes = []

    if contexto_paciente.strip():
        partes.append(
            "## Contexto Adicional da Paciente\n"
            f"{contexto_paciente}\n"
        )

    partes.append(
        f"{dados_exame}\n"
    )

    partes.append(
        "Com base no resultado do modelo de Machine Learning e nos dados "
        "clínicos apresentados, forneça uma interpretação completa seguindo "
        "as diretrizes do sistema: linguagem acolhedora, análise de cada "
        "variável clínica, pontos de atenção, insights práticos e "
        "recomendações de próximos passos."
    )

    return "\n".join(partes)
