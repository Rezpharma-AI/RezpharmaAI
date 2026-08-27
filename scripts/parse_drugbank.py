import sqlite3
from pathlib import Path
import xml.etree.ElementTree as ET

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "cdss.db"


# ============================================================
# FIND THE XML FILE
# ============================================================

def find_xml_file():
    preferred = [
        RAW_DIR / "full database.xml",
        RAW_DIR / "full_database.xml",
        RAW_DIR / "drugbank.xml",
        RAW_DIR / "drugbank_all_full_database.xml",
    ]

    for path in preferred:
        if path.exists():
            return path

    xml_files = list(RAW_DIR.glob("*.xml"))

    if xml_files:
        return max(xml_files, key=lambda f: f.stat().st_size)

    raise FileNotFoundError(
        "No XML file found in data/raw/. "
        "Make sure full database.xml is inside data/raw."
    )


# ============================================================
# HELPERS
# ============================================================

def local_tag(tag):
    """
    DrugBank XML uses namespaces.
    This removes the namespace from the tag name.
    """
    if "}" in tag:
        return tag.split("}")[-1]
    return tag


def infer_severity(description):
    if not description:
        return "MODERATE"

    d = description.lower()

    if "contraindicat" in d:
        return "CONTRAINDICATED"

    major_terms = [
        "fatal",
        "life-threatening",
        "torsades",
        "serotonin syndrome",
        "respiratory depression",
        "severe hypotension",
        "major bleeding",
        "rhabdomyolysis",
        "lactic acidosis",
        "toxicity",
        "bleeding",
        "arrhythmia",
        "hyperkalemia",
        "qt prolongation",
        "hypotension",
    ]

    if any(term in d for term in major_terms):
        return "MAJOR"

    return "MODERATE"


def infer_onset(description):
    if not description:
        return "Unknown"

    d = description.lower()

    if "rapid" in d or "immediate" in d:
        return "Rapid"

    if "delayed" in d or "week" in d or "chronic" in d:
        return "Delayed"

    return "Unknown"


def check_file_type(path: Path):
    with open(path, "rb") as f:
        head = f.read(100)

    if head.startswith(b"PK"):
        raise RuntimeError(
            "This file is actually a ZIP file, not XML. "
            "Extract it first and then run again."
        )

    if head.startswith(b"\x1f\x8b"):
        raise RuntimeError(
            "This file is GZIP compressed. "
            "Extract it first and then run again."
        )

    stripped = head.lstrip().lower()

    if stripped.startswith(b"<html") or stripped.startswith(b"<!doctype html"):
        raise RuntimeError(
            "This file is an HTML page, not XML. "
            "Download the real XML file again."
        )


# ============================================================
# DATABASE SETUP
# ============================================================

def ensure_database(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ddi_rules (
            drug1 TEXT,
            drug2 TEXT,
            severity TEXT,
            mechanism TEXT,
            management TEXT,
            onset TEXT
        )
    """)

    cur.execute("PRAGMA table_info(ddi_rules)")
    columns = [row[1] for row in cur.fetchall()]

    if "onset" not in columns:
        cur.execute("ALTER TABLE ddi_rules ADD COLUMN onset TEXT DEFAULT 'Unknown'")

    conn.commit()


# ============================================================
# MAIN PARSER
# ============================================================

def parse_and_inject():
    xml_path = find_xml_file()

    size_mb = xml_path.stat().st_size / (1024 * 1024)

    print(f"Using file: {xml_path}")
    print(f"File size: {size_mb:.1f} MB")

    check_file_type(xml_path)

    print("Parsing DrugBank XML.")
    print("For a 1.6 GB file, this may take 10-30 minutes.")
    print("Keep this window open.")

    conn = sqlite3.connect(DB_PATH)
    ensure_database(conn)

    cur = conn.cursor()

    # Remove old DrugBank rows before importing again
    cur.execute("DELETE FROM ddi_rules WHERE mechanism LIKE 'DrugBank:%'")

    batch = []
    inserted = 0
    batch_size = 2000

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        tag = local_tag(elem.tag)

        if tag != "drug":
            continue

        drug_name = None

        for child in elem:
            if local_tag(child.tag) == "name" and child.text:
                drug_name = child.text.strip()
                break

        if drug_name:
            for child in elem:
                if local_tag(child.tag) != "drug-interactions":
                    continue

                for interaction in child:
                    if local_tag(interaction.tag) != "drug-interaction":
                        continue

                    other_name = None
                    description = ""

                    for field in interaction:
                        field_tag = local_tag(field.tag)

                        if field_tag == "name" and field.text:
                            other_name = field.text.strip()

                        elif field_tag == "description" and field.text:
                            description = field.text.strip()

                    if other_name:
                        severity = infer_severity(description)
                        onset = infer_onset(description)

                        batch.append((
                            drug_name,
                            other_name,
                            severity,
                            f"DrugBank: {description or 'No description provided.'}",
                            "Review interaction and monitor patient.",
                            onset
                        ))

                        if len(batch) >= batch_size:
                            cur.executemany(
                                "INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?, ?)",
                                batch
                            )

                            inserted += len(batch)
                            batch = []
                            print(f"Parsed {inserted} interactions...")

        elem.clear()

    if batch:
        cur.executemany(
            "INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?, ?)",
            batch
        )
        inserted += len(batch)

    print("Creating search indexes...")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ddi_drug1_lower
        ON ddi_rules (lower(drug1))
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_ddi_drug2_lower
        ON ddi_rules (lower(drug2))
    """)

    conn.commit()

    total = cur.execute("SELECT COUNT(*) FROM ddi_rules").fetchone()[0]

    conn.close()

    print("")
    print("✅ DrugBank import complete.")
    print(f"Inserted DrugBank interaction rows: {inserted}")
    print(f"Total DDI rows in database: {total}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        parse_and_inject()
    except Exception as e:
        print("")
        print("❌ ERROR:")
        print(e)