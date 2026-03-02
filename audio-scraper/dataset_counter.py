import csv
from pathlib import Path

DATA_DIR   = "./data"
OUTPUT_CSV = "./dataset_contents.csv"

data_path = Path(DATA_DIR)
rows = []

for species_dir in sorted(data_path.iterdir()):
    if not species_dir.is_dir():
        continue

    parts    = species_dir.name.split(".", 1)
    class_id = int(parts[0])
    species  = parts[1].replace("_", " ") if len(parts) > 1 else species_dir.name
    mp3s     = sorted(species_dir.glob("*.mp3"))
    count    = len(mp3s)
    status   = "OK" if count >= 5 else "LOW" if count > 0 else "MISSING"
    files    = ", ".join(mp3.name for mp3 in mp3s)

    if status == "MISSING":
        print(f"⚠️  MISSING: {species} (folder: {species_dir.name})")
    if status == "LOW":
        print(f"⚠️  LOW: {species} (folder: {species_dir.name}) has only {count} MP3s")

    rows.append({
        "class_id": class_id,
        "species":  species,
        "folder":   species_dir.name,
        "count":    count,
        "status":   status,
        "files":    files,
    })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class_id", "species", "folder", "count", "status", "files"])
    writer.writeheader()
    writer.writerows(rows)

# Summary
ok      = sum(1 for r in rows if r["status"] == "OK")
low     = sum(1 for r in rows if r["status"] == "LOW")
missing = sum(1 for r in rows if r["status"] == "MISSING")
total   = sum(r["count"] for r in rows)

print(f"Species total : {len(rows)}")
print(f"  OK  (≥5)   : {ok}")
print(f"  LOW (<5)   : {low}")
print(f"  MISSING    : {missing}")
print(f"Total MP3s   : {total}")
print(f"\nSaved to {OUTPUT_CSV}")