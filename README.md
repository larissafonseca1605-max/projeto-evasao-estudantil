# Previsão de Evasão Estudantil com Aprendizado de Máquina

Projeto prático da disciplina **Fundamentos de Inteligência Artificial**, do curso de **Análise e Desenvolvimento de Sistemas — ULBRA Torres**.

**Acadêmica:** Larissa Martins
**Avaliação:** AP2 — Projeto Prático
**Tema:** Previsão de evasão estudantil

## Sobre o projeto

Este projeto utiliza técnicas de Aprendizado de Máquina para identificar estudantes com maior risco de evasão acadêmica.

O problema foi tratado como uma classificação binária:

- `1 — Evasão`: estudante classificado originalmente como `Dropout`;

- `0 — Não evasão`: estudante classificado originalmente como `Enrolled` ou `Graduate`.

Foram utilizados algoritmos clássicos do Scikit-learn:

- Árvore de decisão;

- Regressão logística;

- KNN — K-Nearest Neighbors.

A árvore de decisão foi utilizada como algoritmo principal, enquanto a regressão logística e o KNN foram usados para a comparação crítica dos resultados.

## Dataset

O dataset utilizado é o **Predict Students' Dropout and Academic Success**, disponibilizado pelo UCI Machine Learning Repository.

O conjunto possui:

- 4.424 estudantes;

- 36 atributos preditores;

- informações acadêmicas, demográficas e socioeconômicas;

- resultados acadêmicos dos dois primeiros semestres;

- três classes originais: `Dropout`, `Enrolled` e `Graduate`.

Fonte oficial:

[UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Resultados principais

A avaliação foi realizada com divisão estratificada de 80% para treinamento e 20% para teste, além de validação cruzada estratificada com cinco partes.

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Árvore de decisão | 0,851 | 0,742 | 0,820 | 0,779 | 0,897 |
| **Regressão logística** | **0,873** | 0,787 | **0,831** | **0,808** | **0,927** |
| KNN | 0,838 | **0,912** | 0,549 | 0,686 | 0,888 |

A regressão logística apresentou o melhor equilíbrio entre as métricas. A árvore de decisão apresentou bom recall e possui a vantagem de ser mais fácil de interpretar. O KNN obteve alta precisão, mas apresentou recall mais baixo e deixou de identificar muitos estudantes que realmente evadiram.

## Métricas utilizadas

- **Acurácia:** proporção de previsões corretas;

- **Precisão:** proporção de estudantes realmente evadidos entre os classificados como em risco;

- **Recall:** proporção de estudantes evadidos que foram identificados pelo modelo;

- **F1-score:** equilíbrio entre precisão e recall;

- **ROC AUC:** capacidade do modelo de separar as classes em diferentes limiares;

- **Matriz de confusão:** apresenta acertos e erros de classificação.

## Estrutura do projeto

```
projeto-evasao-estudantil/
│
├── analise_evasao.py
├── requirements.txt
├── README.md
│
├── data/
│   └── data.csv
│
└── resultados/
    ├── metricas.json
    └── figuras/
        ├── 01_distribuicao_classes.png
        ├── 02_matriz_confusao_arvore_decisao.png
        ├── 02_matriz_confusao_knn.png
        ├── 02_matriz_confusao_regressao_logistica.png
        ├── 03_comparacao_modelos.png
        ├── 04_curvas_roc.png
        ├── 05_arvore_decisao.png
        ├── 06_importancia_atributos.png
        └── 07_saida_codigo.png
```

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone URL_DO_SEU_REPOSITORIO
cd NOME_DO_REPOSITORIO
```

Substitua `URL_DO_SEU_REPOSITORIO` pelo endereço do seu repositório no GitHub.

### 2. Instalar as bibliotecas

```bash
pip install -r requirements.txt
```

### 3. Executar o código

```bash
python analise_evasao.py
```

O código lê o arquivo `data/data.csv`, treina os modelos e gera os resultados na pasta `resultados/`.

## Arquivos gerados

Após a execução, o programa gera:

- `resultados/metricas.json`: métricas completas dos modelos;

- matrizes de confusão;

- gráfico de distribuição das classes;

- gráfico de comparação dos modelos;

- curvas ROC;

- visualização da árvore de decisão;

- gráfico de importância dos atributos;

- imagem com a saída principal do código.

## Observações e limitações

Este projeto tem finalidade acadêmica. O modelo não deve ser utilizado para tomar decisões automáticas sobre estudantes.

Um estudante classificado como em risco não necessariamente abandonará o curso. O resultado deve servir apenas como apoio para que a instituição possa oferecer acompanhamento pedagógico, financeiro ou psicossocial.

O dataset foi produzido a partir de uma instituição específica. Portanto, os resultados podem não se repetir da mesma forma em outras instituições, cursos ou regiões.

Além disso, o modelo utiliza informações acadêmicas dos dois primeiros semestres. Uma versão ainda mais antecipada poderia utilizar somente informações disponíveis no momento da matrícula.

## Referências

- [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

- [Artigo sobre o dataset — Predicting Student Dropout and Academic Success](https://www.mdpi.com/2306-5729/7/11/146)

- [Documentação do Scikit-learn — Validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)

- [Documentação do Scikit-learn — Métricas de avaliação](https://scikit-learn.org/stable/modules/model_evaluation.html)
