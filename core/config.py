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

MIN_SIM_STRICT = float(os.getenv("MIN_SIM_STRICT", "0.48"))
MIN_SIM_LOOSE = float(os.getenv("MIN_SIM_LOOSE", "0.38"))

REJECT_LOW_SIM_THRESHOLD = float(os.getenv("REJECT_LOW_SIM_THRESHOLD", "0.3"))

# /matches: trộn lexical để bài ngắn (vd "tang gao") vẫn gần bài dài khẩn cấp
MATCH_BLEND_SEMANTIC = float(os.getenv("MATCH_BLEND_SEMANTIC", "0.72"))
MATCH_BLEND_LEXICAL = float(os.getenv("MATCH_BLEND_LEXICAL", "0.28"))
# Đã qua cổng danh mục → cho phép sàn thấp hơn
MATCH_MIN_SIM_WITH_CATEGORY = float(os.getenv("MATCH_MIN_SIM_WITH_CATEGORY", "0.38"))
MATCH_MIN_SIM_NO_CATEGORY = float(os.getenv("MATCH_MIN_SIM_NO_CATEGORY", "0.45"))
MATCH_RELEVANCE_FLOOR_GATED = float(os.getenv("MATCH_RELEVANCE_FLOOR_GATED", "0.36"))
CATEGORY_MISMATCH_REJECT_SIM = float(os.getenv("CATEGORY_MISMATCH_REJECT_SIM", "0.65"))

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
}

# ---------------------------
# OSRM (khoảng cách đường bộ, tùy chọn)
# ---------------------------
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
OSRM_PROFILE = os.getenv("OSRM_PROFILE", "driving").strip() or "driving"
OSRM_TIMEOUT_SECONDS = float(os.getenv("OSRM_TIMEOUT_SECONDS", "4"))
OSRM_ENABLED = _env_bool("OSRM_ENABLED", "0")
OSRM_MAX_CALLS = int(os.getenv("OSRM_MAX_CALLS", "5"))
OSRM_ENRICH_IN_MATCHES = _env_bool("OSRM_ENRICH_IN_MATCHES", "0")
OSRM_ENRICH_TOP_N = int(os.getenv("OSRM_ENRICH_TOP_N", "5"))
OSRM_GLOBAL_CACHE_SIZE = int(os.getenv("OSRM_GLOBAL_CACHE_SIZE", "5000"))

# Gỡ lỗi / chẩn đoán
DEBUG_SEMANTIC_MATCH = _env_bool("DEBUG_SEMANTIC_MATCH", "0")

