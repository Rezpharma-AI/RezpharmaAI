import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
import time

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "cdss.db"
XML_PATH = BASE / "data" / "raw" / "full database.xml"


def local_name(tag):
    """
    Removes XML namespace if present.

    Example:
        {http://www.drugbank.ca}drug  ->  drug
        drug                          ->  drug
    """
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def ingest():
    if not XML_PATH.exists():
        print(f"Error: Could not find {XML_PATH}")

        raw = BASE / "data" / "raw"
        if raw.exists():
            print("Files in raw folder:")
            for f in raw.iterdir():
                print(" -", f.name)
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Faster import settings for large build
    cur.execute("PRAGMA journal_mode=MEMORY;")
    cur.execute("PRAGMA synchronous=OFF;")

    print("Setting up tables...")
    cur.execute("DROP TABLE IF EXISTS drug_dictionary")
    cur.execute("DROP TABLE IF EXISTS ddi_rules")

    cur.execute("""
        CREATE TABLE drug_dictionary (
            drugbank_id TEXT PRIMARY KEY,
            generic_name TEXT,
            display_name TEXT,
            search_lower TEXT,
            indication TEXT,
            half_life TEXT,
            toxicity TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE ddi_rules (
            drug1_id TEXT,
            drug2_id TEXT,
            drug1_name TEXT,
            drug2_name TEXT,
            mechanism TEXT,
            severity TEXT
        )
    """)

    print(f"Parsing {XML_PATH.name}...")
    print("This is a large file. It may take several minutes.")

    context = ET.iterparse(XML_PATH, events=("end",))

    drug_count = 0
    ddi_count = 0
    start_time = time.time()

    drug_rows = []
    ddi_rows = []

    seen_tags = set()

    for event, elem in context:
        tag = local_name(elem.tag)

        # Used for debugging if no drugs are found
        if len(seen_tags) < 200:
            seen_tags.add(tag)

        if tag != "drug":
            continue

        dbid = None
        name = None
        indication = None
        half_life = None
        toxicity = None
        interactions_elem = None

        # Scan direct children of this <drug> element
        for child in elem:
            child_tag = local_name(child.tag)

            if child_tag == "drugbank-id" and dbid is None:
                dbid = (child.text or "").strip()

            elif child_tag == "name" and name is None:
                name = (child.text or "").strip()

            elif child_tag == "indication" and indication is None:
                indication = (child.text or "").strip()

            elif child_tag == "half-life" and half_life is None:
                half_life = (child.text or "").strip()

            elif child_tag == "toxicity" and toxicity is None:
                toxicity = (child.text or "").strip()

            elif child_tag == "drug-interactions" and interactions_elem is None:
                interactions_elem = child

        if dbid and name:
            drug_rows.append(
                (
                    dbid,
                    name,
                    name,
                    name.lower(),
                    indication or "N/A",
                    half_life or "N/A",
                    toxicity or "N/A",
                )
            )
            drug_count += 1

            if interactions_elem is not None:
                for inter in interactions_elem:
                    if local_name(inter.tag) != "drug-interaction":
                        continue

                    inter_id = None
                    inter_name = None
                    desc = None

                    for inter_child in inter:
                        inter_child_tag = local_name(inter_child.tag)

                        if inter_child_tag == "drugbank-id" and inter_id is None:
                            inter_id = (inter_child.text or "").strip()

                        elif inter_child_tag == "name" and inter_name is None:
                            inter_name = (inter_child.text or "").strip()

                        elif inter_child_tag == "description" and desc is None:
                            desc = (inter_child.text or "").strip()

                    if inter_id and inter_name and desc:
                        severity = "Moderate"
                        desc_lower = desc.lower()

                        if any(
                            word in desc_lower
                            for word in [
                                "contraindicated",
                                "fatal",
                                "severe",
                                "torsades",
                                "neuroleptic malignant",
                                "serotonin syndrome",
                            ]
                        ):
                            severity = "Major"
                        elif any(
                            word in desc_lower
                            for word in [
                                "avoid",
                                "risk",
                                "decreased",
                                "increased",
                            ]
                        ):
                            severity = "Moderate"
                        else:
                            severity = "Minor"

                        ddi_rows.append(
                            (
                                dbid,
                                inter_id,
                                name,
                                inter_name,
                                desc,
                                severity,
                            )
                        )
                        ddi_count += 1

                        # Flush interaction rows periodically
                        if len(ddi_rows) >= 10000:
                            cur.executemany(
                                "INSERT INTO ddi_rules VALUES (?,?,?,?,?,?)",
                                ddi_rows,
                            )
                            ddi_rows = []

        # Clear processed drug element to save RAM
        elem.clear()

        # Flush drug rows periodically
        if len(drug_rows) >= 1000:
            cur.executemany(
                "INSERT OR IGNORE INTO drug_dictionary VALUES (?,?,?,?,?,?,?)",
                drug_rows,
            )
            drug_rows = []

        # Progress message
        if drug_count % 2000 == 0 and drug_count > 0:
            conn.commit()
            elapsed = time.time() - start_time
            print(
                f"Processed {drug_count} drugs, "
                f"{ddi_count} interactions... "
                f"elapsed={elapsed:.0f}s"
            )

    # Final flush
    if drug_rows:
        cur.executemany(
            "INSERT OR IGNORE INTO drug_dictionary VALUES (?,?,?,?,?,?,?)",
            drug_rows,
        )

    if ddi_rows:
        cur.executemany(
            "INSERT INTO ddi_rules VALUES (?,?,?,?,?,?)",
            ddi_rows,
        )

    conn.commit()

    if drug_count == 0:
        print("No <drug> elements were recognized.")
        print("Sample tags found:")
        for t in sorted(seen_tags)[:50]:
            print(" -", t)
    else:
        print(
            f"Done! Ingested {drug_count} drugs and {ddi_count} interactions "
            f"in {time.time() - start_time:.1f}s."
        )

    print("Creating indexes for fast UI performance...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_search ON drug_dictionary(search_lower)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ddi1 ON ddi_rules(drug1_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ddi2 ON ddi_rules(drug2_name)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    ingest()