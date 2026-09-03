import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = Path("data/raw/full database.xml")
DB_PATH = Path("data/cdss.db")

def local_tag(tag):
    return tag.split("}")[-1] if "}" in tag else tag

def parse_all_names():
    if not XML_PATH.exists():
        print("❌ full database.xml not found in data/raw/")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("🧹 Clearing old dictionary...")
    cur.execute("DROP TABLE IF EXISTS drug_dictionary")
    cur.execute("CREATE TABLE drug_dictionary (display_name TEXT, generic_name TEXT, search_lower TEXT)")
    
    print("📖 Parsing 1.6GB XML for ALL Generics and Brands...")
    batch = []
    count = 0
    
    for event, elem in ET.iterparse(XML_PATH, events=("end",)):
        if local_tag(elem.tag) == "drug":
            generic_name = None
            brands = []
            
            for child in elem:
                tag = local_tag(child.tag)
                if tag == "name" and child.text and not generic_name:
                    generic_name = child.text.strip()
                elif tag == "products":
                    for prod in child:
                        if local_tag(prod.tag) == "product":
                            for p_child in prod:
                                if local_tag(p_child.tag) == "name" and p_child.text:
                                    brands.append(p_child.text.strip())
                elif tag == "international-brands":
                    for brand in child:
                        if local_tag(brand.tag) == "international-brand":
                            for b_child in brand:
                                if local_tag(b_child.tag) == "name" and b_child.text:
                                    brands.append(b_child.text.strip())
            
            if generic_name:
                # Add the generic name itself
                batch.append((generic_name, generic_name, generic_name.lower()))
                # Add all mapped brands
                for b in set(brands):
                    batch.append((f"{b} ({generic_name})", generic_name, b.lower()))
                    
            if len(batch) >= 10000:
                cur.executemany("INSERT INTO drug_dictionary VALUES (?, ?, ?)", batch)
                count += len(batch)
                batch = []
                print(f"Processed {count} names...")
            elem.clear()
            
    if batch:
        cur.executemany("INSERT INTO drug_dictionary VALUES (?, ?, ?)", batch)
        count += len(batch)
        
    print("⚡ Creating high-speed search index...")
    cur.execute("CREATE INDEX idx_search ON drug_dictionary(search_lower)")
    conn.commit()
    conn.close()
    print(f"✅ DONE! Mapped {count} Generics and Brands into SQLite.")

if __name__ == "__main__":
    parse_all_names()