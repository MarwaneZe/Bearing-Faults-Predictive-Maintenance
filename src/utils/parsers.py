from pathlib import Path

HEALTHY_BEARINGS = {
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
}


def parse_filename(path):
    """Split a file name into its operating-condition fields.

    Example:  'N15_M07_F10_K001_3'  ->  speed=1500, torque=0.7,
    force=1000, bearing=K001, rep=3.
    """
    n_code, m_code, f_code, bearing, rep = Path(path).stem.split("_")

    return {
        "bearing": bearing,
        "speed_rpm": int(n_code[1:]) * 100,  # N09 -> 900
        "torque_Nm": int(m_code[1:]) / 10,  # M07 -> 0.7
        "force_N": int(f_code[1:]) * 100,  # F10 -> 1000
        "rep": int(rep),  # '11' -> 11
    }


def get_label(bearing_id):
    """Return 'healthy' or 'damaged' for a bearing id."""
    return "healthy" if bearing_id in HEALTHY_BEARINGS else "damaged"
