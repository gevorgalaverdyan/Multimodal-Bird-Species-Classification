from pathlib import Path

spec_root = Path("spectrograms")

LOW = 0
OK = 0
MISSING = 0

min = float("inf")
max = float("-inf")
mean = 0 
total = 0

for species_dir in sorted(spec_root.iterdir()):
    if species_dir.is_dir():
        count = len(list(species_dir.glob("*.png")))
        total += count
        if count == 0:        
            MISSING += 1
        elif count < 30:    
            LOW += 1
        else:
            OK += 1

        if count < min and count > 0:
            min = count
        if count > max:
            max = count
            
mean = total // (OK + LOW + MISSING) if (OK + LOW + MISSING) > 0 else 0

print(f"✓  {OK} OK  |  {LOW} LOW  |  {MISSING} MISSING")
print(f"Min: {min}, Max: {max}, Mean: {mean}")