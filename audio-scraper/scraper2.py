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
    (9,   "Brewer_Blackbird",              "Brewer's Blackbird"),
    (19,  "Gray_Catbird",                  "Grey Catbird"),
    (22,  "Chuck_will_Widow",              "Chuck-will's-widow"),
    (23,  "Brandt_Cormorant",              "Brandt's Cormorant"),
    (34,  "Gray_crowned_Rosy_Finch",       "Grey-crowned Rosy-Finch"),
    (50,  "Eared_Grebe",                   "Black-necked Grebe"),
    (61,  "Heermann_Gull",                 "Heermann's Gull"),
    (67,  "Anna_Hummingbird",              "Anna's Hummingbird"),
    (70,  "Green_Violetear",               "Mexican Violetear"),
    (74,  "Florida_Jay",                   "Florida Scrub-Jay"),
    (78,  "Gray_Kingbird",                 "Grey Kingbird"),
    (83,  "White_breasted_Kingfisher",     "White-throated Kingfisher"),
    (93,  "Clark_Nutcracker",              "Clark's Nutcracker"),
    (98,  "Scott_Oriole",                  "Scott's Oriole"),
    (103, "Sayornis",                      "Black Phoebe"),
    (104, "American_Pipit",               "Buff-bellied Pipit"),
    (105, "Whip_poor_Will",               "Eastern Whip-poor-will"),
    (107, "Common_Raven",                  "Northern Raven"),
    (110, "Geococcyx",                     "Greater Roadrunner"),
    (113, "Baird_Sparrow",                 "Baird's Sparrow"),
    (115, "Brewer_Sparrow",               "Brewer's Sparrow"),
    (122, "Harris_Sparrow",               "Harris's Sparrow"),
    (123, "Henslow_Sparrow",              "Henslow's Sparrow"),
    (124, "Le_Conte_Sparrow",             "LeConte's Sparrow"),
    (125, "Lincoln_Sparrow",              "Lincoln's Sparrow"),
    (126, "Nelson_Sharp_tailed_Sparrow",  "Nelson's Sparrow"),
    (134, "Cape_Glossy_Starling",         "Long-tailed Glossy Starling"),
    (135, "Bank_Swallow",                 "Sand Martin"),
    (141, "Artic_Tern",                   "Arctic Tern"),
    (146, "Forsters_Tern",               "Forster's Tern"),
    (159, "Black_and_white_Warbler",      "Black-and-white Warbler"),
    (178, "Swainson_Warbler",             "Swainson's Warbler"),
    (180, "Wilson_Warbler",              "Wilson's Warbler"),
    (193, "Bewick_Wren",                  "Bewick's Wren"),
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