import os
from typing import Dict, List


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------
# Ngưỡng tương đồng
# ---------------------------
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "keepitreal/vietnamese-sbert",
).strip()

def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
MIN_SIM_STRICT = _get_float_env("MIN_SIM_STRICT", 0.48)
MIN_SIM_LOOSE = _get_float_env("MIN_SIM_LOOSE", 0.38)
REJECT_LOW_SIM_THRESHOLD = _get_float_env("REJECT_LOW_SIM_THRESHOLD", 0.3)

MATCH_BLEND_SEMANTIC = _get_float_env("MATCH_BLEND_SEMANTIC", 0.72)
MATCH_BLEND_LEXICAL = _get_float_env("MATCH_BLEND_LEXICAL", 0.28)

MATCH_MIN_SIM_WITH_CATEGORY = _get_float_env("MATCH_MIN_SIM_WITH_CATEGORY", 0.38)
MATCH_MIN_SIM_NO_CATEGORY = _get_float_env("MATCH_MIN_SIM_NO_CATEGORY", 0.45)
MATCH_RELEVANCE_FLOOR_GATED = _get_float_env("MATCH_RELEVANCE_FLOOR_GATED", 0.36)

CATEGORY_MISMATCH_REJECT_SIM = _get_float_env("CATEGORY_MISMATCH_REJECT_SIM", 0.65)
OSRM_TIMEOUT_SECONDS = _get_float_env("OSRM_TIMEOUT_SECONDS", 4)

# ---------------------------
# Nhãn danh mục + mẫu gốc
# ---------------------------
CATEGORY_LABELS: List[str] = ["food", "vehicle", "clothes", "education"]
CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "food": [
        "thuc pham do an gao mi tom sua nuoc mam rau cu qua",
        "food meal groceries rice noodles milk canned",
        "nhu yeu pham an uong",
    ],
    "vehicle": [
        "xe may xe dap xe lan phuong tien di lai",
        "vehicle motorbike bicycle wheelchair transport",
        "ho tro phuong tien di chuyen",
    ],
    "clothes": [
        "quan ao chan man ao am giay dep do mac",
        "clothes jacket blanket shoes apparel",
        "do sinh hoat chan goi man",
    ],
    "education": [
        "sach vo but tap hoc phi laptop may tinh do hoc tap",
        "education school notebook textbook tuition laptop",
        "ho tro hoc sinh sinh vien",
    ],
     "household": [
        "noi bep do gia dung do sinh hoat quat dien tu lanh may giat ban ghe giuong tu quan ao",
        "household furniture kitchen appliances home goods",
        "noi that dung cu gia dung",
    ],
    "medical": [
        "thuoc y te phau thuat vien phi kham benh sau khoe",
        "medical medicine health care hospital treatment",
        "ho tro y te suc khoe",
    ],
}

# ---------------------------
# OSRM (khoảng cách đường bộ, tùy chọn)
# ---------------------------
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
OSRM_PROFILE = os.getenv("OSRM_PROFILE", "driving").strip() or "driving"
OSRM_ENABLED = _env_bool("OSRM_ENABLED", "0")
OSRM_MAX_CALLS = int(os.getenv("OSRM_MAX_CALLS", "5"))
OSRM_ENRICH_IN_MATCHES = _env_bool("OSRM_ENRICH_IN_MATCHES", "0")
OSRM_ENRICH_TOP_N = int(os.getenv("OSRM_ENRICH_TOP_N", "5"))
OSRM_GLOBAL_CACHE_SIZE = int(os.getenv("OSRM_GLOBAL_CACHE_SIZE", "5000"))

# Gỡ lỗi / chẩn đoán
DEBUG_SEMANTIC_MATCH = _env_bool("DEBUG_SEMANTIC_MATCH", "0")

