
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import DATA_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "reference" / "reference.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

TARGET = "default.payment.next.month"
RANDOM_STATE = 666
TEST_SIZE = 0.20


def load_reference_data() -> pd.DataFrame:
    """Carrega e valida o conjunto de referência."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de referência não encontrado: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Não treinar se o contrato de dados falhar.
    df = DATA_SCHEMA.validate(df, lazy=True)

    if df.empty:
        raise ValueError("O conjunto de referência está vazio.")

    return df


def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features da variável-alvo."""

    if TARGET not in df.columns:
        raise ValueError(f"Coluna-alvo ausente: {TARGET}")

    # ID é identificador, não uma variável explicativa.
    X = df.drop(columns=[TARGET, "ID"])
    y = df[TARGET].astype(int)

    return X, y


def build_model() -> Pipeline:
    """Cria o pipeline de pré-processamento e classificação."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Calcula as métricas de classificação."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def main() -> None:
    """Treina, avalia e salva a baseline."""

    print("Carregando dados de referência...")
    df = load_reference_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Registros para treinamento: {len(X_train)}")
    print(f"Registros para teste: {len(X_test)}")
    print(f"Features utilizadas: {X.shape[1]}")

    model = build_model()

    print("\nTreinando modelo...")
    model.fit(X_train, y_train)

    print("\nAvaliando modelo...")
    metrics = evaluate_model(model, X_test, y_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    model_path = ARTIFACTS_DIR / "baseline_logistic_regression.joblib"
    metrics_path = ARTIFACTS_DIR / "baseline_metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print("\nMétricas:")
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ):
        print(f"{metric}: {metrics[metric]:.4f}")

    print("\nMatriz de confusão:")
    print(metrics["confusion_matrix"])

    print(f"\nModelo salvo em: {model_path}")
    print(f"Métricas salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
