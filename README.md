# Predição de atraso de voos com Machine Learning

Projeto de Data Science para previsão de atraso de voos utilizando Python, XGBoost e Databricks.

## Objetivo do projeto

O objetivo deste projeto é prever se um voo terá **atraso superior a 15 minutos na saída**, a partir de variáveis operacionais como companhia aérea, aeroporto de origem e destino, horário previsto e distância do voo.

A proposta foi desenvolver um fluxo completo de análise, passando por:

* exploração e entendimento dos dados;
* identificação de padrões de atraso;
* criação de variável alvo para classificação;
* treinamento e comparação de modelos de Machine Learning;
* análise de métricas de negócio e ajuste de limiar de decisão;
* exportação dos resultados e leitura no Databricks para validação complementar.

---

## Problema de negócio

Atrasos em voos impactam a experiência do passageiro, a operação das companhias aéreas e a eficiência dos aeroportos.
Antecipar a probabilidade de um voo atrasar pode ajudar em decisões como:

* planejamento operacional;
* alocação de equipes e recursos;
* comunicação preventiva com passageiros;
* monitoramento de rotas e horários com maior risco de atraso.

Neste projeto, o foco foi construir um modelo capaz de classificar se um voo pertence ou não ao grupo de **atraso relevante** (mais de 15 minutos).

---

## Base de dados

Foi utilizada uma base amostral de voos contendo informações como:

* `month`
* `day_of_month`
* `day_of_week`
* `op_unique_carrier`
* `origin`
* `dest`
* `crs_dep_time`
* `crs_arr_time`
* `distance`
* `dep_delay`
* `arr_delay`

A variável alvo do projeto foi construída a partir de `dep_delay`.

---

## Definição da variável alvo

Foi criada a variável binária `atraso_15min`, definida da seguinte forma:

* **0** → voo com atraso de saída menor ou igual a 15 minutos
* **1** → voo com atraso de saída superior a 15 minutos

Exemplo de criação:

```python
df_modelo['atraso_15min'] = (df_modelo['dep_delay'] > 15).astype(int)
```

Na amostra utilizada para modelagem, a distribuição da variável alvo ficou aproximadamente em:

* **79,68%** dos voos sem atraso relevante
* **20,32%** dos voos com atraso superior a 15 minutos

---

## Etapas do projeto

## 1. Análise exploratória dos dados (EDA)

Antes da modelagem, foi realizada uma análise exploratória para entender o comportamento dos atrasos e identificar padrões relevantes.

As análises incluíram:

* distribuição geral dos atrasos de saída;
* comparação entre média e mediana dos atrasos;
* atraso médio por companhia aérea;
* atraso médio por aeroporto de origem;
* atraso médio por mês;
* atraso médio por hora prevista de saída;
* atraso médio por dia da semana.

### Principais insights exploratórios

* A distribuição de `dep_delay` apresentou **forte assimetria à direita**, com presença de outliers muito altos.
* Em vários agrupamentos, a **média foi puxada para cima por atrasos extremos**, enquanto a **mediana mostrou que boa parte dos voos saía no horário ou até adiantada**.
* Alguns aeroportos e companhias apresentaram médias elevadas, mas com baixo volume de voos, exigindo cuidado na interpretação.
* O comportamento por horário sugeriu que determinados períodos do dia concentram atrasos maiores.

---

## 2. Preparação dos dados

Para a modelagem, foi criado um dataset específico contendo as variáveis preditoras e a variável alvo.

### Variáveis utilizadas no modelo

* `month`
* `day_of_month`
* `day_of_week`
* `op_unique_carrier`
* `origin`
* `dest`
* `crs_dep_time`
* `crs_arr_time`
* `distance`

As colunas categóricas foram transformadas com **one-hot encoding** utilizando `pd.get_dummies()`.

Exemplo:

```python
X = pd.get_dummies(
    X,
    columns=['op_unique_carrier', 'origin', 'dest'],
    drop_first=True
)
```

Ao final do processo, a matriz de entrada ficou com **588 variáveis preditoras**.

---

## 3. Modelagem

O projeto foi estruturado como um problema de **classificação binária**.

Foi realizada a separação entre treino e teste com `train_test_split`, utilizando estratificação da variável alvo.

### Modelos testados

* **Regressão Logística**
* **Random Forest**
* **XGBoost**

---

## 4. Avaliação dos modelos

Como o problema possui desbalanceamento entre classes, a avaliação não foi feita apenas com acurácia.
Também foram analisadas métricas como:

* **precision**
* **recall**
* **f1-score**
* **matriz de confusão**
* **curva ROC**
* **AUC**

O principal foco foi a capacidade de identificar voos com atraso superior a 15 minutos (**classe 1**).

---

## Resultado do modelo XGBoost

O modelo de melhor leitura para o objetivo do projeto foi o **XGBoost**, especialmente quando analisado em conjunto com ajuste de limiar de decisão.

### Resultado com limiar 0.50

Matriz de confusão:

* **1021** voos corretamente classificados como sem atraso relevante
* **554** voos classificados como atraso, mas que na prática não atrasaram mais de 15 minutos
* **152** voos que atrasaram e não foram identificados pelo modelo
* **250** voos corretamente classificados como atraso superior a 15 minutos

### Métricas com limiar 0.50

Para a **classe 1 (voos com atraso > 15 min)**:

* **Precision:** 0.31
* **Recall:** 0.62
* **F1-score:** 0.41

### Interpretação

Esse resultado indica que o modelo conseguiu recuperar uma parte relevante dos voos com atraso, mas ainda com presença de falsos positivos.
Como o objetivo do problema pode priorizar sensibilidade à detecção de atraso, também foi feita análise de limiares alternativos.

---

## 5. Ajuste de limiar de decisão

Além do limiar padrão de 0.50, foram testados outros pontos de corte para a probabilidade prevista pelo XGBoost.

### Comparação de limiares

| limiar | recall classe 1 | precision classe 1 | acurácia | leitura                                  |
| ------ | --------------- | ------------------ | -------- | ---------------------------------------- |
| 0.50   | 0.62            | 0.31               | 0.64     | Melhor equilíbrio geral                  |
| 0.40   | 0.81            | 0.26               | 0.50     | Mais sensível para detectar atrasos      |
| 0.35   | 0.87            | 0.24               | 0.41     | Recall alto, mas muitos falsos positivos |

### Conclusão sobre os limiares

* **0.50** entregou o melhor equilíbrio geral entre desempenho e estabilidade.
* **0.40** aumentou a sensibilidade do modelo para detectar atrasos, mas elevou o número de falsos positivos.
* **0.35** ampliou ainda mais o recall da classe 1, porém com perda importante de precisão e acurácia.

---

## 6. Curva ROC e AUC

O desempenho do XGBoost também foi avaliado com curva ROC.

* **AUC ≈ 0.68**

Esse resultado indica que o modelo possui capacidade de separação **moderada** entre voos com atraso relevante e voos sem atraso relevante, superando o desempenho de um classificador aleatório.

---

## 7. Integração com Databricks

Após a modelagem no Colab, os resultados do XGBoost foram exportados em arquivos `.csv` e carregados no **Databricks** para validação complementar.

Foram levados para o Databricks:

* `resultado_modelo_xgboost.csv`
* `comparacao_limiares_xgboost.csv`

No ambiente Databricks, os arquivos foram lidos com Spark e utilizados para:

* reconstrução da matriz de confusão;
* leitura da tabela de limiares;
* validação dos resultados do modelo em ambiente de dados.

Exemplo de leitura no Databricks:

```python
df_limiares = spark.read.csv(
    "/Volumes/workspace/default/flight_project_files/comparacao_limiares_xgboost.csv",
    header=True,
    inferSchema=True
)
```

---

## Tecnologias utilizadas

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **XGBoost**
* **Google Colab**
* **Databricks**

---

## Estrutura do projeto

```bash
predicao-atraso-voos-machine-learning/
│
├─ README.md
├─ predicao_atraso_voos.ipynb
├─ requirements.txt
├─ comparacao_limiares_xgboost.csv
└─ resultado_modelo_xgboost.csv
```

---

## Aprendizados do projeto

Durante o desenvolvimento deste projeto, os principais aprendizados foram:

* análise exploratória com foco em interpretação de média vs mediana;
* construção de variável alvo para classificação;
* tratamento de variáveis categóricas com one-hot encoding;
* comparação entre modelos de classificação;
* leitura prática de matriz de confusão, precision, recall e f1-score;
* ajuste de limiar de decisão com foco no objetivo do problema;
* uso de Databricks para consumir e validar resultados gerados na modelagem.

---

## Próximos passos

Como evolução futura do projeto, alguns caminhos possíveis são:

* testar novos algoritmos e ajustes de hiperparâmetros;
* trabalhar com features adicionais operacionais e meteorológicas;
* avaliar técnicas de balanceamento de classes;
* criar dashboard de monitoramento de atrasos;
* transformar o fluxo em pipeline mais estruturado de dados e modelagem.

---

## Autor

**Pablo Henrique B. Silva**

Projeto desenvolvido como estudo prático de Data Science com foco em análise exploratória, classificação e interpretação de modelos aplicados ao contexto de atrasos de voos.
