import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "cdss.db"


def get_conn():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing database: {DB_PATH}. Run ingest_drugbank.py first."
        )
    return sqlite3.connect(DB_PATH)


def table_exists(table_name):
    conn = get_conn()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    conn.close()
    return row is not None


def missing_tables():
    required = [
        "drug_dictionary",
        "ddi_rules",
        "side_effects",
        "diseases"
    ]
    return [t for t in required if not table_exists(t)]


def db_status():
    conn = get_conn()
    status = {}

    for table in [
        "drug_dictionary",
        "ddi_rules",
        "side_effects",
        "diseases"
    ]:
        try:
            status[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            status[table] = 0

    conn.close()
    return status


def search_drugs(query, limit=20):
    query = (query or "").strip()

    if len(query) < 2:
        return []

    conn = get_conn()

    sql = """
        SELECT
            drugbank_id,
            generic_name,
            display_name,
            half_life,
            indication,
            toxicity
        FROM drug_dictionary
        WHERE search_lower LIKE ?
        ORDER BY
            CASE
                WHEN lower(display_name) = lower(?)
                  OR lower(generic_name) = lower(?)
                THEN 0
                ELSE 1
            END,
            length(display_name)
        LIMIT ?
    """

    df = pd.read_sql_query(
        sql,
        conn,
        params=(
            f"%{query.lower()}%",
            query,
            query,
            limit
        )
    )

    conn.close()
    return df.to_dict("records")


def resolve_drug(name):
    rows = search_drugs(name, limit=1)
    return rows[0] if rows else None


def get_side_effects(drugbank_id, limit=100):
    conn = get_conn()

    df = pd.read_sql_query(
        """
        SELECT DISTINCT side_effect_name
        FROM side_effects
        WHERE drug_id = ?
        ORDER BY side_effect_name
        LIMIT ?
        """,
        conn,
        params=(drugbank_id, limit)
    )

    conn.close()
    return df["side_effect_name"].tolist()


def get_diseases(drugbank_id, limit=50):
    conn = get_conn()

    df = pd.read_sql_query(
        """
        SELECT DISTINCT disease_name
        FROM diseases
        WHERE drug_id = ?
        ORDER BY disease_name
        LIMIT ?
        """,
        conn,
        params=(drugbank_id, limit)
    )

    conn.close()
    return df["disease_name"].tolist()


def _drug_names(generic_name, display_name=None):
    names = {generic_name.lower()}

    if display_name:
        names.add(display_name.lower())

    return list(names)


def get_interactions_for_drug(generic_name, display_name=None, limit=100):
    names = _drug_names(generic_name, display_name)
    placeholders = ",".join(["?"] * len(names))

    conn = get_conn()

    sql = f"""
        SELECT
            CASE
                WHEN lower(drug1_name) IN ({placeholders})
                THEN drug2_name
                ELSE drug1_name
            END AS other_drug,
            severity,
            mechanism
        FROM ddi_rules
        WHERE lower(drug1_name) IN ({placeholders})
           OR lower(drug2_name) IN ({placeholders})
        LIMIT ?
    """

    params = names + names + names + [limit]

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    return df.to_dict("records")


def analyze_regimen(drugs):
    """
    drugs = list of dict:
    {
        "drugbank_id": ...,
        "generic_name": ...,
        "display_name": ...
    }
    """
    results = []

    if len(drugs) < 2:
        return results

    conn = get_conn()

    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a = drugs[i]
            b = drugs[j]

            names_a = _drug_names(
                a["generic_name"],
                a.get("display_name")
            )

            names_b = _drug_names(
                b["generic_name"],
                b.get("display_name")
            )

            ph_a = ",".join(["?"] * len(names_a))
            ph_b = ",".join(["?"] * len(names_b))

            sql = f"""
                SELECT
                    drug1_name,
                    drug2_name,
                    severity,
                    mechanism
                FROM ddi_rules
                WHERE
                    (
                        lower(drug1_name) IN ({ph_a})
                        AND lower(drug2_name) IN ({ph_b})
                    )
                    OR
                    (
                        lower(drug1_name) IN ({ph_b})
                        AND lower(drug2_name) IN ({ph_a})
                    )
                LIMIT 1
            """

            params = names_a + names_b + names_b + names_a

            df = pd.read_sql_query(sql, conn, params=params)

            if not df.empty:
                row = df.iloc[0].to_dict()
                row["drug_a"] = a.get("display_name") or a["generic_name"]
                row["drug_b"] = b.get("display_name") or b["generic_name"]
                results.append(row)

    conn.close()
    return results