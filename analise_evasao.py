"""
Projeto AP2 - Fundamentos de Inteligência Artificial
Previsão de evasão estudantil com dados do UCI Machine Learning Repository.

Execução:
    python3 analise_evasao.py

O script usa data/data.csv, gera resultados/metricas.json e figuras/*.png.
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "data.csv"
OUT = BASE / "resultados"
FIG = OUT / "figuras"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET_ORIGINAL = "Target"
TARGET = "Evasao"


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    if TARGET_ORIGINAL not in df.columns:
        raise ValueError(f"Coluna alvo não encontrada. Colunas: {list(df.columns)}")

    # Dropout = classe positiva (1); Enrolled e Graduate = não evasão (0).
    df[TARGET] = (df[TARGET_ORIGINAL].astype(str).str.strip().str.lower() == "dropout").astype(int)
    class_counts = df[TARGET].value_counts().sort_index()
    class_pct = (class_counts / len(df) * 100).round(2)

    # Remover o alvo original e manter somente variáveis preditoras.
    X = df.drop(columns=[TARGET_ORIGINAL, TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "arvore_decisao": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "regressao_logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {
        "projeto": "Previsão de evasão estudantil",
        "dataset": {
            "instancias": int(len(df)),
            "atributos_preditores": int(X.shape[1]),
            "arquivo": "UCI Predict Students' Dropout and Academic Success",
            "alvo_original": sorted(df[TARGET_ORIGINAL].unique().tolist()),
            "alvo_binario": {"1": "Dropout", "0": "Enrolled ou Graduate"},
            "distribuicao": {
                "nao_evasao_0": {"quantidade": int(class_counts.get(0, 0)), "percentual": float(class_pct.get(0, 0))},
                "evasao_1": {"quantidade": int(class_counts.get(1, 0)), "percentual": float(class_pct.get(1, 0))},
            },
            "divisao": {"treino": int(len(X_train)), "teste": int(len(X_test)), "proporcao_teste": 0.20},
        },
        "modelos": {},
    }

    # Distribuição das classes.
    plt.figure(figsize=(7, 4.5))
    labels = ["Não evasão", "Evasão"]
    values = [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))]
    bars = plt.bar(labels, values, color=["#2a9d8f", "#e76f51"])
    plt.title("Distribuição da variável-alvo")
    plt.ylabel("Quantidade de estudantes")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 25, str(value), ha="center")
    savefig("01_distribuicao_classes.png")

    predictions = {}
    probabilities = {}
    for name, model in models.items():
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        predictions[name] = pred
        probabilities[name] = proba

        report = classification_report(y_test, pred, target_names=["Não evasão", "Evasão"], output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred).tolist()
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_evasao": float(precision_score(y_test, pred, zero_division=0)),
            "recall_evasao": float(recall_score(y_test, pred, zero_division=0)),
            "f1_evasao": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        }
        cv_metrics = {}
        for metric in scoring:
            vals = cv_scores[f"test_{metric}"]
            cv_metrics[metric] = {"media": float(vals.mean()), "desvio_padrao": float(vals.std())}

        results["modelos"][name] = {
            "teste": test_metrics,
            "validacao_cruzada_5_folds": cv_metrics,
            "matriz_confusao": cm,
            "relatorio_classificacao": report,
        }

        if name != "baseline":
            disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["Não evasão", "Evasão"])
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"Matriz de confusão — {name.replace('_', ' ').title()}")
            savefig(f"02_matriz_confusao_{name}.png")

    # Comparação de métricas na validação cruzada.
    model_names = ["arvore_decisao", "regressao_logistica", "knn"]
    pretty = {"arvore_decisao": "Árvore", "regressao_logistica": "Regressão logística", "knn": "KNN"}
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.24
    plt.figure(figsize=(10, 5.5))
    for i, name in enumerate(model_names):
        vals = [results["modelos"][name]["validacao_cruzada_5_folds"][m]["media"] for m in metrics_to_plot]
        plt.bar(x + (i - 1) * width, vals, width, label=pretty[name])
    plt.xticks(x, ["Acurácia", "Precisão", "Recall", "F1", "ROC AUC"])
    plt.ylim(0, 1.05)
    plt.ylabel("Média na validação cruzada")
    plt.title("Comparação dos modelos — validação cruzada estratificada")
    plt.legend()
    savefig("03_comparacao_modelos.png")

    # Curvas ROC dos três modelos.
    plt.figure(figsize=(7, 5.5))
    for name in model_names:
        RocCurveDisplay.from_predictions(y_test, probabilities[name], name=pretty[name])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.title("Curvas ROC no conjunto de teste")
    savefig("04_curvas_roc.png")

    # Árvore para interpretação visual.
    tree = models["arvore_decisao"]
    plt.figure(figsize=(22, 10))
    plot_tree(tree, feature_names=feature_names, class_names=["Não evasão", "Evasão"], filled=True, max_depth=3, fontsize=7)
    plt.title("Árvore de decisão (visualização dos três primeiros níveis)")
    savefig("05_arvore_decisao.png")

    # Importância por permutação no conjunto de teste.
    perm = permutation_importance(models["arvore_decisao"], X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1)
    imp = pd.DataFrame({"atributo": feature_names, "importancia_media": perm.importances_mean, "desvio": perm.importances_std}).sort_values("importancia_media", ascending=False).head(12)
    results["importancia_permutacao_arvore"] = imp.to_dict(orient="records")
    plt.figure(figsize=(9, 6))
    plt.barh(imp["atributo"][::-1], imp["importancia_media"][::-1], xerr=imp["desvio"][::-1], color="#264653")
    plt.xlabel("Variação média do F1 ao permutar")
    plt.title("Atributos mais relevantes — importância por permutação")
    savefig("06_importancia_atributos.png")

    (OUT / "metricas.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Projeto executado com sucesso.")
    print(f"Dataset: {len(df)} linhas, {X.shape[1]} preditores")
    print(f"Classes: não evasão={values[0]}, evasão={values[1]}")
    for name in model_names:
        m = results["modelos"][name]["teste"]
        print(f"{pretty[name]} | teste: acurácia={m['accuracy']:.3f}, precisão={m['precision_evasao']:.3f}, recall={m['recall_evasao']:.3f}, F1={m['f1_evasao']:.3f}, ROC AUC={m['roc_auc']:.3f}")
    print(f"Resultados salvos em: {OUT}")


if __name__ == "__main__":
    main()
