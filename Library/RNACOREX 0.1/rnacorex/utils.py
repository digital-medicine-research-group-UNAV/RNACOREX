import os
import gdown
from pathlib import Path

def check_engines():

    """
    
    Check if all required engine files are present in the 'engines/' directory.

    Returns:
        bool: True if all required files are present, False otherwise.
        
    """

    base_dir = os.path.dirname(__file__)
    engines_dir = os.path.join(base_dir, "engines")

    required_files = ["DIANA_targets.txt", "Targetscan_targets.txt", "MTB_targets_25.csv", "TarBase_v9.tsv", "gencode.v47.basic.annotation.gtf"] 
    missing_files = []

    for filename in required_files:
        if not os.path.isfile(os.path.join(engines_dir, filename)):
            missing_files.append(filename)

    if missing_files:
        print("⚠️ The following engine files are missing from the 'engines/' directory:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ All required engines are present.")
    


def download():

    file_ids = {
        "DIANA_targets.txt": "10FjHjMYshpla2mK4WkiCDrbTw5Bz3PEo",
        "gencode.v47.basic.annotation.gtf": "1pdsg5ZnvMdQiJV4y1nrsoHzbP_mPBKCY",
        "MTB_targets_25.csv": "1rTr4gKPChCKiUFLTvYlDNLEItniEB21K",
        "TarBase_v9.tsv": "1ShqnwHImQraRLpPXmC6Z-uzHBj8cu777",
        "Targetscan_targets.txt": "1-mNLLyV9oz5kBJ0Py3wZONuiGVbhhx7A"
    }

    engines_dir = Path(__file__).parent / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in file_ids.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        destination = engines_dir / filename
        gdown.download(url, str(destination), quiet=False)
        print(f"✓ Saved {filename}")


if __name__ == "__main__":

    download()
