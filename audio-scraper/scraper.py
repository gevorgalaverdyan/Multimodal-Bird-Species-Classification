import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
XENO_CANTO_API_KEY = os.getenv("XENO_CANTO_API_KEY", "").strip()
API_ENDPOINT = "https://xeno-canto.org/api/3/recordings"

input_file = "./clean.bird.classes.txt"
RECORDINGS_PER_SPECIES = 5  # how many recordings to download per species

if not XENO_CANTO_API_KEY:
    raise RuntimeError("Missing XENO_CANTO_API_KEY in .env")


def normalize_species_name(name: str) -> str:
    """
    Xeno-canto uses hyphens in compound English names (e.g. 'Black-footed Albatross').
    Our input file uses spaces instead. Replace spaces between lowercase words with hyphens.
    Strategy: hyphenate any space where the previous word starts uppercase and the
    current word starts lowercase — covers compound adjectives like:
      'Black footed' → 'Black-footed'
      'Red winged'   → 'Red-winged'
      'Yellow headed' → 'Yellow-headed'
    while leaving title-case boundaries (separate words) intact.
    """
    words = name.split()
    result = [words[0]]
    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]
        # Compound adjective: e.g. "Black" (upper) + "footed" (lower) → "Black-footed"
        if prev_word[0].isupper() and curr_word[0].islower():
            result[-1] = result[-1] + "-" + curr_word
        else:
            result.append(curr_word)
    return " ".join(result)


def build_file_url(raw_url: str) -> str:
    """
    The API `file` field may return either:
      - '//xeno-canto.org/...'   (protocol-relative, as shown in docs)
      - 'https://xeno-canto.org/...' (already absolute)
    Normalize both to a valid https:// URL.
    """
    if raw_url.startswith("//"):
        return "https:" + raw_url
    elif raw_url.startswith("http"):
        return raw_url
    else:
        return "https://" + raw_url


try:
    with open(input_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    pbar = tqdm(total=len(lines))

    for idx, specie in enumerate(lines):
        normalized = normalize_species_name(specie)
        print(f"\nProcessing: {(idx+1):03d} {specie!r} → querying as {normalized!r}")

        # First try exact match (faster, more precise)
        query = f'en:"={normalized}" grp:birds'
        params = {
            "query": query,
            "key": XENO_CANTO_API_KEY,
            "per_page": RECORDINGS_PER_SPECIES,
        }

        response = requests.get(API_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        recordings = data.get("recordings", [])

        # Fallback: wildcard search if exact match returns nothing
        if not recordings:
            print(f"  No exact match — retrying with wildcard for '{normalized}'")
            params["query"] = f'en:"{normalized}" grp:birds'
            response = requests.get(API_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            recordings = data.get("recordings", [])

        if not recordings:
            print(f"  No recordings found for '{specie}' (tried exact and wildcard)")
            pbar.update(1)
            continue

        print(f"  Found {data.get('numRecordings', '?')} total — downloading up to {len(recordings)}")

        specie_slug = specie.replace(" ", "_")
        folder_name = f"./data/{(idx+1):03d}.{specie_slug}"
        os.makedirs(folder_name, exist_ok=True)

        for recording in recordings:
            recording_id = recording["id"]
            out_mp3 = os.path.join(folder_name, f"{specie_slug}_{recording_id}.mp3")

            if os.path.exists(out_mp3):
                print(f"  Skipping {out_mp3} (already exists)")
                continue

            # FIX: normalize URL before requesting — avoids 'https:https://' double-prefix
            file_url = build_file_url(recording["file"])
            print(f"  Downloading XC{recording_id} from {file_url}")

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

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")