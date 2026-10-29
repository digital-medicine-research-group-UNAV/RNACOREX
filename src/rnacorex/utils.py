import os
import gdown
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINES_DIR = PROJECT_ROOT / "Engine"

FILE_IDS = {
    "interactions.parquet": "1Z6ee9ysko6P_WpK5LeVe0M6VDeT6ueVx",
}


def check_engines():

    """Check whether all required engine files are present."""

    missing_files = [
        filename
        for filename in FILE_IDS
        if not (ENGINES_DIR / filename).is_file()
    ]

    if missing_files:
        print("⚠️ Missing engine files:")
        for filename in missing_files:
            print(f"  - {filename}")
        return False

    return True


def download():

    """Download missing engine files."""

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, file_id in FILE_IDS.items():
        destination = ENGINES_DIR / filename

        # No descargar archivos que ya existen.
        if destination.is_file():
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        result = gdown.download(
            url,
            str(destination),
            quiet=False,
        )

        if result is None or not destination.is_file():
            # Evitar dejar un archivo incompleto.
            if destination.exists():
                destination.unlink()
            raise RuntimeError(
                f"Could not download {filename}. "
                "Check the Google Drive permissions and your connection."
            )

        print(f"✓ Saved default engine")

def ensure_engines():

    """Download missing engines automatically."""
    
    if not check_engines():
        download()

    if not check_engines():
        raise RuntimeError(
            "Some RNACOREX engine files are still missing."
        )