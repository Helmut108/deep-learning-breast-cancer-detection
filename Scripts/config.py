# config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR_2 = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

csv_path_train = (
    DATA_DIR
    / "DDSM"
    / "CSV"
    / "mass_case_description_train_set.csv"
)

csv_path_test = (
    DATA_DIR
    / "DDSM"
    / "CSV"
    / "mass_case_description_test_set.csv"
)

checkpoint_path = (
    BASE_DIR / "best_model_50 epochs.pth"
)

ddsm_path = Path(
    "/Volumes/Samsung_T7/Not_Sync/"
    "CBIS-DDSM-All-doiJNLP-zzWs5zfZ/cbis_ddsm"
)

