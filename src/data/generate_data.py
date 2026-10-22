
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Diretórios do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PRODUCTION_DIR = PROJECT_ROOT / "data" / "production"
INVALID_DIR = PROJECT_ROOT / "data" / "invalid"

RANDOM_STATE = 42
TEST_SIZE = 0.30


def load_data() -> pd.DataFrame:
    """Carrega o dataset original e verifica sua estrutura básica."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Remove espaços acidentais nos nomes das colunas.
    df.columns = df.columns.str.strip()

    target = "default.payment.next.month"

    if target not in df.columns:
        raise ValueError(
            f"Coluna-alvo '{target}' não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError("O dataset original está vazio.")

    if "ID" in df.columns and df["ID"].duplicated().any():
        raise ValueError("O dataset original possui IDs duplicados.")

    print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    return df


def generate_reference_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os conjuntos de referência e produção."""

    target = "default.payment.next.month"

    reference, production = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target],
    )

    return reference.copy(), production.copy()


def generate_production_data(
    production: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simula alterações controladas na distribuição das features.

    LIMIT_BAL: redução de 25% nos limites de crédito.
    PAY_0: aumento de um nível em 30% dos registros.
    O alvo não é alterado artificialmente.
    """

    production = production.copy()

    # Alteração 1: redução do limite de crédito.
    production["LIMIT_BAL"] = (
        production["LIMIT_BAL"] * 0.75
    ).round().astype("int64")

    # Alteração 2: aumento de atrasos de pagamento.
    if "PAY_0" in production.columns:
        rng = __import__("numpy").random.default_rng(RANDOM_STATE)

        mask = rng.random(len(production)) < 0.30

        production.loc[mask, "PAY_0"] = (
            production.loc[mask, "PAY_0"] + 1
        ).clip(upper=8)

    return production


def generate_invalid_data(
    df: pd.DataFrame,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Gera um lote com erros intencionais para testar o contrato."""

    invalid = df.sample(
        n=min(n_samples, len(df)),
        random_state=RANDOM_STATE,
    ).copy()

    # Erro 1: limite de crédito negativo.
    invalid.loc[invalid.index[0], "LIMIT_BAL"] = -1000

    # Erro 2: idade ausente.
    invalid.loc[invalid.index[1], "AGE"] = float("nan")

    # Erro 3: ID duplicado, quando houver registros suficientes.
    if len(invalid) >= 3 and "ID" in invalid.columns:
        invalid.loc[invalid.index[2], "ID"] = invalid.iloc[0]["ID"]

    return invalid


def save_data(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    invalid: pd.DataFrame,
) -> None:
    """Salva os três conjuntos nos diretórios correspondentes."""

    for directory in (
        REFERENCE_DIR,
        PRODUCTION_DIR,
        INVALID_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    reference.to_csv(
        REFERENCE_DIR / "reference.csv",
        index=False,
    )

    production.to_csv(
        PRODUCTION_DIR / "production.csv",
        index=False,
    )

    invalid.to_csv(
        INVALID_DIR / "invalid_batch.csv",
        index=False,
    )

    print("\nArquivos gerados:")
    print(f"Referência: {REFERENCE_DIR / 'reference.csv'}")
    print(f"Produção:   {PRODUCTION_DIR / 'production.csv'}")
    print(f"Inválidos:  {INVALID_DIR / 'invalid_batch.csv'}")


def main() -> None:
    """Executa o pipeline de preparação dos dados."""

    df = load_data()

    reference, production_base = generate_reference_data(df)

    production = generate_production_data(production_base)

    invalid = generate_invalid_data(df)

    save_data(reference, production, invalid)

    print("\nResumo dos conjuntos:")
    print(f"Referência: {reference.shape}")
    print(f"Produção:   {production.shape}")
    print(f"Inválidos:  {invalid.shape}")

    print("\nDistribuição do alvo na referência:")
    print(reference["default.payment.next.month"].value_counts(normalize=True))

    print("\nDistribuição do alvo na produção:")
    print(production["default.payment.next.month"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
