import sys

import pandas as pd
from pathlib import Path

soyabeans_path = 'data/raw/soybean-large.data'

path = Path(__file__).parents[3] / soyabeans_path

try:
    data = pd.read_csv(path, header=None, na_values="?")
except FileNotFoundError:
    print(f"File {soyabeans_path} not found")
    sys.exit()

names = [
    "class", "date", "plant_stand", "precip", "temp", "hail",
    "crop_hist", "area_damaged", "severity", "seed_tmt", "germination",
    "plant_growth", "leaves", "leafspots_halo", "leafspots_marg", "leafspot_size",
    "leaf_shread", "leaf_malf", "leaf_mild", "stem", "lodging",
    "stem_cankers", "canker_lesion", "fruiting_bodies", "external_decay", "mycelium",
    "int_discolor", "sclerotia", "fruit_pods", "fruit_spots", "seed",
    "mold_growth", "seed_discolor", "seed_size", "shriveling", "roots",
]

ORDINAL = [
    "date",           # april -> october (0..6)
    "plant_stand",    # normal < lt-normal
    "precip",         # lt-norm < norm < gt-norm
    "temp",           # lt-norm < norm < gt-norm
    "crop_hist",      # diff-lst-year -> same-lst-sev-yrs
    "severity",       # minor < pot-severe < severe
    "germination",    # 90-100% -> 80-89% -> lt-80%
    "seed_size",      # norm < lt-norm
    "stem_cankers",   # absent < below-soil < above-soil < above-sec-nde (+ ?).
]

NOMINAL = [
    "hail", "area_damaged", "seed_tmt", "plant_growth", "leaves",
    "leafspots_halo", "leafspots_marg", "leaf_shread", "leaf_malf",
    "leaf_mild", "stem", "lodging", "canker_lesion",
    "fruiting_bodies", "external_decay", "mycelium", "int_discolor",
    "sclerotia", "fruit_pods", "fruit_spots", "seed", "mold_growth",
    "seed_discolor", "shriveling", "roots", "leafspot_size",
]

# -- Not the plant symptoms
ENVIRONMENTAL = [
    "date", "plant_stand", "precip", "temp", "hail",
    "crop_hist", "area_damaged", "severity", "seed_tmt", "germination",
]

BINARY = [
    "hail", "plant_stand", "plant_growth", "leaves", "leaf_shread", "leaf_malf",
    "stem", "lodging", "fruiting_bodies", "mycelium", "sclerotia", "seed",
    "mold_growth", "seed_discolor", "seed_size", "shriveling",
]

# Ordinal levels contaminated by an out-of-scale "dna" value.
# Kept in NOMINAL for now;
DNA_CONTAMINATED = ["leafspots_marg", "leafspot_size", "fruit_pods", "fruit_spots"]

data.columns = names

print(data.shape)
print(data['class'].nunique())
print(data.isna().sum().sum())
print(data['leaves'].isna().sum())
