FROM python:3.14

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD \
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb && \
    jupyter nbconvert --to html --execute notebooks/03_evaluation.ipynb && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_interpretability.ipynb && \
    jupyter nbconvert --to html --execute notebooks/04_interpretability.ipynb
