# Relatório Técnico — Diagnóstico de Diabetes com Machine Learning

## 1. Introdução

O diabetes é uma doença crônica que afeta milhões de pessoas em todo o mundo e está associada a diversas complicações quando não diagnosticada e tratada adequadamente. Diante desse cenário, sistemas inteligentes de apoio ao diagnóstico podem auxiliar profissionais de saúde na triagem inicial de pacientes, otimizando o tempo de análise clínica e reduzindo possíveis erros humanos.

Este projeto propõe o desenvolvimento de um modelo de Machine Learning aplicado a dados clínicos, com o objetivo de prever a presença de diabetes em pacientes, atuando como uma ferramenta de suporte à decisão médica.


## 2. Objetivo

O objetivo deste trabalho é construir um modelo de classificação capaz de prever se um paciente possui diabetes a partir de variáveis clínicas e demográficas. O modelo proposto não substitui o diagnóstico médico, atuando exclusivamente como um sistema de apoio à decisão clínica.


## 3. Dataset

O dataset utilizado neste projeto é o Diabetes Dataset, disponibilizado publicamente na plataforma Kaggle. A base de dados é composta por registros de pacientes do sexo feminino, contendo variáveis clínicas e demográficas relevantes para o diagnóstico de diabetes.

As principais variáveis presentes no conjunto de dados incluem número de gestações, nível de glicose, pressão arterial, espessura da dobra cutânea, nível de insulina, índice de massa corporal (BMI), histórico familiar de diabetes e idade. A variável alvo, denominada "Outcome", indica a presença (1) ou ausência (0) de diabetes.


## 4. Exploração de Dados

Inicialmente, foi realizada uma análise exploratória dos dados com o objetivo de compreender a estrutura do dataset, identificar padrões relevantes e detectar possíveis problemas de qualidade nos dados.

A análise da variável alvo indicou um leve desbalanceamento entre as classes, com maior número de pacientes não diagnosticados com diabetes em relação aos diagnosticados. Esse fator reforça a importância do uso de métricas além da acurácia na avaliação dos modelos.

A análise estatística descritiva revelou diferenças significativas entre pacientes com e sem diabetes, especialmente nas variáveis relacionadas aos níveis de glicose, índice de massa corporal (BMI) e idade. Pacientes diagnosticados com diabetes apresentaram, em média, valores mais elevados nessas variáveis.

Durante a exploração dos dados, foram identificados valores inconsistentes em variáveis clínicas relevantes, como glicose, pressão arterial, espessura da dobra cutânea, insulina e BMI, onde o valor zero não representa uma condição fisiológica válida. Esses valores indicam a necessidade de um pré-processamento adequado antes da etapa de modelagem.

A análise de correlação evidenciou que a variável glicose possui a maior correlação com o diagnóstico de diabetes, o que está de acordo com o conhecimento clínico sobre a doença. Outras variáveis, como BMI e idade, também apresentaram correlação moderada com a variável alvo.


## 5. Pré-processamento

Nesta etapa, serão aplicadas técnicas de limpeza e transformação dos dados, incluindo o tratamento de valores inconsistentes, normalização das variáveis numéricas e construção de um pipeline de pré-processamento para preparação dos dados para os modelos de Machine Learning.


## 6. Modelagem

Serão utilizados diferentes algoritmos de classificação para o diagnóstico de diabetes, com o objetivo de comparar seus desempenhos e selecionar o modelo mais adequado ao contexto do problema.


## 7. Avaliação

Os modelos serão avaliados utilizando métricas como acurácia, precisão, recall e F1-score, considerando a importância do recall no contexto médico, onde falsos negativos podem trazer riscos à saúde do paciente.


## 8. Interpretação do Modelo

Serão utilizadas técnicas de interpretabilidade para compreender a influência das variáveis no processo de decisão do modelo, garantindo maior transparência e confiabilidade dos resultados.


## 9. Conclusão

Ao final do projeto, será apresentada uma análise crítica dos resultados obtidos, discutindo a viabilidade do modelo como ferramenta de apoio ao diagnóstico, suas limitações e possibilidades de aprimoramento.


## 10. Considerações Éticas
Discussão sobre o uso responsável de IA na área da saúde.
