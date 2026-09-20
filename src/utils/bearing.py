"""Bearing damage taxonomy (Block 6).

Builds the curated bearings.csv table: HOW each bearing was damaged.
This is the file the ML evaluation splits will reference later.
"""

from .parsers import condition_key
from .metadata import DATA_RAW

# Primary taxonomy by name prefix (from the Paderborn bearing scheme):
#   KAxx -> artificial damage  (EDM / drilling / engraving)
#   KIxx -> real damage        (accelerated lifetime test / fatigue)
#   KBxx -> combination        (artificial AND real damage together)
#   Kxxx -> healthy
def damage_type_for(bearing):
    if bearing.startswith("KA"):
        return "artificial"
    if bearing.startswith("KI"):
        return "real"
    if bearing.startswith("KB"):
        return "combination"
    return "healthy"


# Curated overrides for bearings with TWO damage sources (real + added artificial).
# Verify against the official bearing description sheet, then fill in:
#   "KIxx": "drilling applied on real fatigue pitting", ...
CURATED_NOTES = {
    # "KI01": "real (fatigue) + drilling",
}


def build_bearings_table(metadata, save=True):
    """Map every bearing present in the corpus to its damage type.

    Returns DataFrame: bearing, damage_type, n_recordings, source_condition(s).
    Optionally writes data/processed/bearings.csv (opt-in default on).
    """
    import pandas as pd

    rows = []
    for bearing, grp in metadata.groupby("bearing"):
        rows.append({
            "bearing": bearing,
            "damage_type": damage_type_for(bearing),
            "n_recordings": int(len(grp)),
            "conditions": sorted(set(grp.apply(condition_key, axis=1))),
            "note": CURATED_NOTES.get(bearing, ""),
        })
    table = pd.DataFrame(rows).sort_values("bearing").reset_index(drop=True)

    if save:
        out = DATA_RAW.parent / "processed" / "bearings.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print("saved:", out)
    return table