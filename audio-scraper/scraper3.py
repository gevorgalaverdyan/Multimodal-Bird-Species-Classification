import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
XENO_CANTO_API_KEY = os.getenv("XENO_CANTO_API_KEY", "").strip()
API_ENDPOINT = "https://xeno-canto.org/api/3/recordings"

RECORDINGS_PER_SPECIES = 5

if not XENO_CANTO_API_KEY:
    raise RuntimeError("Missing XENO_CANTO_API_KEY in .env")

# (folder_index, original_folder_name, xeno_canto_query_name)
MISSING_SPECIES = [
    (3,   "Sooty Albatross", "Sooty Albatross"),
    (34,  "Gray crowned Rosy Finch", "Grey-crowned Rosy Finch"),
    (74, "Florida Jay", "Florida scrub jay"),
]


def build_file_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        return "https:" + raw_url
    elif raw_url.startswith("http"):
        return raw_url
    else:
        return "https://" + raw_url


pbar = tqdm(total=len(MISSING_SPECIES))

for folder_idx, folder_slug, query_name in MISSING_SPECIES:
    folder_name = f"./data/{folder_idx:03d}.{folder_slug}"

    # Skip if already has mp3s
    if os.path.isdir(folder_name) and any(f.endswith(".mp3") for f in os.listdir(folder_name)):
        print(f"\nSkipping {folder_idx:03d} {folder_slug} (already downloaded)")
        pbar.update(1)
        continue

    print(f"\nProcessing: {folder_idx:03d} {folder_slug!r} → querying as {query_name!r}")

    # Try exact match first
    params = {
        "query": f'en:"={query_name}" grp:birds',
        "key": XENO_CANTO_API_KEY,
        "per_page": RECORDINGS_PER_SPECIES,
    }
    response = requests.get(API_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    recordings = data.get("recordings", [])

    # Fallback to wildcard
    if not recordings:
        print(f"  No exact match — retrying with wildcard")
        params["query"] = f'en:"{query_name}" grp:birds'
        response = requests.get(API_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        recordings = data.get("recordings", [])

    if not recordings:
        print(f"  No recordings found for '{query_name}' (tried exact and wildcard)")
        pbar.update(1)
        continue

    print(f"  Found {data.get('numRecordings', '?')} total — downloading up to {len(recordings)}")
    os.makedirs(folder_name, exist_ok=True)

    for recording in recordings:
        recording_id = recording["id"]
        out_mp3 = os.path.join(folder_name, f"{folder_slug}_{recording_id}.mp3")

        if os.path.exists(out_mp3):
            print(f"  Skipping {out_mp3} (already exists)")
            continue

        file_url = build_file_url(recording["file"])
        print(f"  Downloading XC{recording_id} → {out_mp3}")

        try:
            audio = requests.get(file_url, timeout=60, stream=True)
            audio.raise_for_status()
            with open(out_mp3, "wb") as f:
                for chunk in audio.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            print(f"  Warning: Failed to download XC{recording_id}: {e}")

    pbar.update(1)

pbar.close()
print("\nDone!")