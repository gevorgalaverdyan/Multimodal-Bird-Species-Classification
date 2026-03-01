import os
import csv

rows = []
for folder in sorted(os.listdir("./data")):
    folder_path = os.path.join("./data", folder)
    if not os.path.isdir(folder_path):
        continue
    idx, species = folder.split(".", 1)
    for fname in os.listdir(folder_path):
        if fname.endswith(".mp3"):
            xc_id = fname.replace(".mp3", "").split("_")[-1]
            rows.append({
                "file_path": f"{folder}/{fname}",
                "class_id": int(idx),
                "species": species.replace("_", " "),
                "xc_id": xc_id,
                "xc_url": f"https://xeno-canto.org/{xc_id}"
            })

with open("./data/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_path", "class_id", "species", "xc_id", "xc_url"])
    writer.writeheader()
    writer.writerows(rows)