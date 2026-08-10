# config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR_2 = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

print(f"BASE_DIR: {BASE_DIR}")
print(f"BASE_DIR_2: {BASE_DIR_2}")
print(f"DATA_DIR: {DATA_DIR}")

CSV_PATH_TRAIN = (
    DATA_DIR
    / "DDSM"
    / "CSV"
    / "mass_case_description_train_set.csv"
)

DDSM_PATH = Path(
    "/Volumes/Samsung_T7/Not_Sync/"
    "CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm"
)

