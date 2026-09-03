import sqlite3
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "cdss.db"

SIDE_EFFECTS_CSV = BASE / "data" / "raw" / "drug_side_effect_with_names.csv"
DISEASES_CSV = BASE / "data" / "raw" / "drug_disease_with_names.csv"


def main():
    if not DB_PATH.exists():
        raise SystemExit(
            "Missing data/cdss.db. Run ingest_drugbank.py first."
        )

    if not SIDE_EFFECTS_CSV.exists():
        raise SystemExit(
            f"Missing file: {SIDE_EFFECTS_CSV}"
        )

    if not DISEASES_CSV.exists():
        raise SystemExit(
            f"Missing file: {DISEASES_CSV}"
        )

    conn = sqlite3.connect(DB_PATH)

    # --------------------------------------------------------
    # SIDE EFFECTS
    # --------------------------------------------------------
    print("Loading side effects...")
    se = pd.read_csv(
        SIDE_EFFECTS_CSV,
        usecols=["DRUG_ID", "SIDE_EFFECT_NAME"]
    )

    se.columns = ["drug_id", "side_effect_name"]
    se = se.dropna(subset=["drug_id", "side_effect_name"])
    se = se.drop_duplicates()

    se.to_sql(
        "side_effects",
        conn,
        if_exists="replace",
        index=False
    )

    print(f"Loaded {len(se):,} side effect records.")

    # --------------------------------------------------------
    # DISEASES
    # --------------------------------------------------------
    print("Loading diseases...")
    dz = pd.read_csv(
        DISEASES_CSV,
        usecols=["DRUG_ID", "DISEASE_NAME"]
    )

    dz.columns = ["drug_id", "disease_name"]
    dz = dz.dropna(subset=["drug_id", "disease_name"])
    dz = dz.drop_duplicates()

    dz.to_sql(
        "diseases",
        conn,
        if_exists="replace",
        index=False
    )

    print(f"Loaded {len(dz):,} disease records.")

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------
    print("Creating indexes...")
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_side_effects_drug "
        "ON side_effects(drug_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_diseases_drug "
        "ON diseases(drug_id)"
    )
    conn.commit()

    # --------------------------------------------------------
    # FINAL COUNTS
    # --------------------------------------------------------
    se_count = conn.execute(
        "SELECT COUNT(*) FROM side_effects"
    ).fetchone()[0]

    dz_count = conn.execute(
        "SELECT COUNT(*) FROM diseases"
    ).fetchone()[0]

    conn.close()

    print("\n✅ CSV ingestion complete.")
    print(f"side_effects records: {se_count:,}")
    print(f"diseases records:     {dz_count:,}")


if __name__ == "__main__":
    main()