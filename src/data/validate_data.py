
from pathlib import Path

import pandas as pd
import pandera.errors

from src.data.schema import DATA_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production" / "production.csv"
INVALID_PATH = PROJECT_ROOT / "data" / "invalid" / "invalid_batch.csv"


def validate_file(path: Path) -> bool:
    """Valida um arquivo CSV e informa se ele passou no contrato."""

    print(f"\nValidando: {path.name}")

    if not path.exists():
        print(f"ERRO: arquivo não encontrado: {path}")
        return False

    df = pd.read_csv(path)

    try:
        DATA_SCHEMA.validate(df, lazy=True)
        print("APROVADO: contrato de dados respeitado.")
        return True

    except pandera.errors.SchemaErrors as exc:
        print("REJEITADO: falha no contrato de dados.")
        print(exc.failure_cases.to_string(index=False))
        return False

    except pandera.errors.SchemaError as exc:
        print("REJEITADO: falha no contrato de dados.")
        print(exc)
        return False


def main() -> None:
    reference_valid = validate_file(REFERENCE_PATH)
    production_valid = validate_file(PRODUCTION_PATH)
    invalid_valid = validate_file(INVALID_PATH)

    print("\nResumo da validação:")
    print(f"Referência válida: {reference_valid}")
    print(f"Produção válida: {production_valid}")
    print(f"Lote inválido aceito: {invalid_valid}")

    if not reference_valid or not production_valid:
        raise SystemExit(
            "Pipeline interrompido: referência ou produção inválida."
        )

    if invalid_valid:
        raise SystemExit(
            "Teste falhou: o lote propositalmente inválido foi aceito."
        )

    print("\nTodos os testes de validação foram concluídos.")


if __name__ == "__main__":
    main()
