# Imagem base do Python 3.14
FROM python:3.14

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia arquivo de dependências primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instalação de ferramentas de compilação necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Cópia do código do projeto para o container
COPY . .

# Criação de diretórios para armazenar logs e métricas
RUN mkdir -p /app/logs /app/metrics

# Volumes para persistir logs e métricas no host
VOLUME ["/app/logs", "/app/metrics"]

# Healthcheck para monitorar se o container está saudável
# Verifica a cada 30s se a CPU não está travada em 99%
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psutil; assert psutil.cpu_percent() < 99" || exit 1

# Metadados do container para documentação
LABEL maintainer="dIAbetes-ml-pipeline" \
      version="1.0" \
      description="ML pipeline with monitoring and auto-scaling support"

# Comando executado quando o container inicia
CMD ["python", "src/train.py"]
