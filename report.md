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

A etapa de pré-processamento teve como objetivo preparar os dados para a modelagem, garantindo qualidade, consistência e reprodutibilidade.

Inicialmente, foram tratados valores inconsistentes identificados na análise exploratória, onde o valor zero não representa uma condição fisiológica válida em variáveis clínicas como glicose, pressão arterial, insulina e índice de massa corporal. Esses valores foram substituídos por valores ausentes e posteriormente imputados utilizando a mediana de cada variável.

Em seguida, os dados foram separados em conjuntos de treino, validação e teste, respeitando a proporção das classes por meio de amostragem estratificada. Essa abordagem evita viés na avaliação dos modelos.
Os dados foram divididos em 70% para treino, 15% para validação e 15% para teste.


Por fim, foi construído um pipeline de pré-processamento utilizando normalização via StandardScaler, assegurando que todas as variáveis numéricas estivessem na mesma escala, o que é fundamental para diversos algoritmos de Machine Learning.


## 6. Modelagem

Foram utilizados diferentes algoritmos de classificação para o diagnóstico de diabetes, com o objetivo de comparar seus desempenhos e selecionar o modelo mais adequado ao contexto do problema.

Além dos modelos básicos, foram aplicadas técnicas de otimização de hiperparâmetros utilizando Grid Search e Randomized Search. Essas abordagens permitiram identificar combinações mais adequadas de parâmetros, priorizando a métrica de recall, considerada crítica no contexto médico. Também foi avaliado o modelo SGDClassifier, ampliando a comparação entre diferentes abordagens lineares e baseadas em árvores.


## 7. Avaliação

A avaliação final dos modelos foi realizada utilizando o conjunto de teste, reservado exclusivamente para mensurar a capacidade de generalização do modelo selecionado. 
A escolha do modelo final considerou principalmente a métrica de recall, devido à relevância clínica de minimizar falsos negativos no diagnóstico de diabetes.

A matriz de confusão e o relatório de classificação evidenciaram que o modelo selecionado apresentou desempenho consistente, mantendo equilíbrio entre sensibilidade e precisão, o que reforça sua adequação como sistema de apoio à decisão médica.




## 8. Interpretação do Modelo

O modelo selecionado foi o SGDClassifier, um modelo linear treinado com otimização por gradiente estocástico. 
Para garantir a interpretabilidade, foram analisados os coeficientes do modelo e aplicadas técnicas de SHAP (SHapley Additive Explanations).

A análise dos coeficientes indicou que variáveis como nível de glicose, índice de massa corporal (BMI) e idade possuem maior impacto na decisão do modelo. 
A análise com SHAP corroborou esses resultados, fornecendo explicações globais e individuais das previsões.

Esses achados estão alinhados com o conhecimento clínico sobre o diabetes, reforçando a adequação do modelo como sistema de apoio à decisão médica.



## 9. Conclusão

Os resultados obtidos demonstram que modelos de Machine Learning podem ser utilizados como ferramentas eficazes de apoio à decisão no diagnóstico de diabetes. 
O modelo SGDClassifier apresentou o melhor desempenho em termos de recall, métrica priorizada devido à relevância clínica de minimizar falsos negativos.

A análise de interpretabilidade indicou que variáveis como nível de glicose, índice de massa corporal (BMI) e idade são os principais fatores que influenciam as previsões do modelo, em conformidade com o conhecimento clínico existente.

Apesar dos bons resultados, o modelo possui limitações, como a utilização de um dataset relativamente pequeno e restrito a um único perfil populacional.
Como trabalhos futuros, sugere-se a avaliação de outros algoritmos, técnicas de balanceamento de classes e a validação em bases de dados externas.



## 10. Considerações Éticas

Este projeto tem caráter exclusivamente acadêmico e experimental. O modelo desenvolvido não substitui a avaliação médica profissional, sendo utilizado apenas como ferramenta de apoio à decisão clínica. O uso responsável de sistemas de Inteligência Artificial na área da saúde é fundamental para garantir segurança, ética e confiabilidade nos diagnósticos.

