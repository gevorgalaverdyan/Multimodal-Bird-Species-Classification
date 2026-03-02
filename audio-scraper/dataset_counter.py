import csv
from pathlib import Path

DATA_DIR   = "./data"
OUTPUT_CSV = "./dataset_contents.csv"

data_path = Path(DATA_DIR)
rows = []

for species_dir in sorted(data_path.iterdir()):
    if not species_dir.is_dir():
        continue

    parts     = species_dir.name.split(".", 1)
    class_id  = int(parts[0])
    species   = parts[1].replace("_", " ") if len(parts) > 1 else species_dir.name
    mp3s      = sorted(species_dir.glob("*.mp3"))
    count     = len(mp3s)

    if count == 0:
        rows.append({
            "class_id":    class_id,
            "species":     species,
            "folder":      species_dir.name,
            "count":       0,
            "status":      "MISSING",
            "file":        "",
        })
    else:
        for mp3 in mp3s:
            rows.append({
                "class_id":    class_id,
                "species":     species,
                "folder":      species_dir.name,
                "count":       count,
                "status":      "OK" if count >= 5 else "LOW",
                "file":        mp3.name,
            })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class_id", "species", "folder", "count", "status", "file"])
    writer.writeheader()
    writer.writerows(rows)

# Print summary
total_species  = len(set(r["class_id"] for r in rows))
ok             = len(set(r["class_id"] for r in rows if r["status"] == "OK"))
low            = len(set(r["class_id"] for r in rows if r["status"] == "LOW"))
missing        = len(set(r["class_id"] for r in rows if r["status"] == "MISSING"))
total_files    = sum(1 for r in rows if r["file"])

print(f"Species total : {total_species}")
print(f"  OK  (≥5)   : {ok}")
print(f"  LOW (<5)   : {low}")
print(f"  MISSING    : {missing}")
print(f"Total MP3s   : {total_files}")
print(f"\nSaved to {OUTPUT_CSV}")