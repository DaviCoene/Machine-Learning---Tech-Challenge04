
import pandera.pandas as pa
from pandera import Check


DATA_SCHEMA = pa.DataFrameSchema(
    {
        "ID": pa.Column(
            int,
            nullable=False,
            unique=True,
            coerce=True,
        ),
        "LIMIT_BAL": pa.Column(
            int,
            Check.ge(0),
            nullable=False,
            coerce=True,
        ),
        "SEX": pa.Column(
            int,
            Check.isin([1, 2]),
            nullable=False,
            coerce=True,
        ),
        "EDUCATION": pa.Column(
            int,
            Check.isin([0, 1, 2, 3, 4, 5, 6]),
            nullable=False,
            coerce=True,
        ),
        "MARRIAGE": pa.Column(
            int,
            Check.isin([0, 1, 2, 3]),
            nullable=False,
            coerce=True,
        ),
        "AGE": pa.Column(
            int,
            [Check.ge(18), Check.le(100)],
            nullable=False,
            coerce=True,
        ),
        "PAY_0": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "PAY_2": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "PAY_3": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "PAY_4": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "PAY_5": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "PAY_6": pa.Column(
            int,
            [Check.ge(-2), Check.le(8)],
            nullable=False,
            coerce=True,
        ),
        "BILL_AMT1": pa.Column(float, nullable=False, coerce=True),
        "BILL_AMT2": pa.Column(float, nullable=False, coerce=True),
        "BILL_AMT3": pa.Column(float, nullable=False, coerce=True),
        "BILL_AMT4": pa.Column(float, nullable=False, coerce=True),
        "BILL_AMT5": pa.Column(float, nullable=False, coerce=True),
        "BILL_AMT6": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT1": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT2": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT3": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT4": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT5": pa.Column(float, nullable=False, coerce=True),
        "PAY_AMT6": pa.Column(float, nullable=False, coerce=True),
        "default.payment.next.month": pa.Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
    coerce=False,
)
