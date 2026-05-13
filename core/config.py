import os
from typing import Dict, List, Set


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------
# Ngưỡng tương đồng
# ---------------------------
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "keepitreal/vietnamese-sbert",
).strip()

MIN_SIM_STRICT = _get_float_env("MIN_SIM_STRICT", 0.48)
MIN_SIM_LOOSE = _get_float_env("MIN_SIM_LOOSE", 0.38)
# Ngưỡng riêng cho /related (mặc định = MIN_SIM_LOOSE; có thể tăng qua env để giảm nhiễu)
RELATED_MIN_SIM_LOOSE = _get_float_env("RELATED_MIN_SIM_LOOSE", MIN_SIM_LOOSE)
REJECT_LOW_SIM_THRESHOLD = _get_float_env("REJECT_LOW_SIM_THRESHOLD", 0.3)

MATCH_BLEND_SEMANTIC = _get_float_env("MATCH_BLEND_SEMANTIC", 0.30)
MATCH_BLEND_LEXICAL = _get_float_env("MATCH_BLEND_LEXICAL", 0.70)

MATCH_MIN_SIM_WITH_CATEGORY = _get_float_env("MATCH_MIN_SIM_WITH_CATEGORY", 0.38)
MATCH_MIN_SIM_NO_CATEGORY = _get_float_env("MATCH_MIN_SIM_NO_CATEGORY", 0.45)
MATCH_RELEVANCE_FLOOR_GATED = _get_float_env("MATCH_RELEVANCE_FLOOR_GATED", 0.36)

CATEGORY_MISMATCH_REJECT_SIM = _get_float_env("CATEGORY_MISMATCH_REJECT_SIM", 0.65)
OSRM_TIMEOUT_SECONDS = _get_float_env("OSRM_TIMEOUT_SECONDS", 4)

# Lọc “cùng household nhưng khác loại đồ” (vd: nồi cơm vs chỉ quạt)
HOUSEHOLD_FACET_STRICT = _env_bool("HOUSEHOLD_FACET_STRICT", "1")

# Category inference confidence thresholds
CATEGORY_INFERENCE_MIN_CONFIDENCE = _get_float_env("CATEGORY_INFERENCE_MIN_CONFIDENCE", 0.75)
CATEGORY_INFERENCE_GAP_THRESHOLD = _get_float_env("CATEGORY_INFERENCE_GAP_THRESHOLD", 0.10)

# ---------------------------
# Nhãn danh mục + mẫu gốc (CLEAN - NO OVERLAPPING)
# ---------------------------
CATEGORY_LABELS: List[str] = ["food", "vehicle", "clothes", "education", "household", "medical"]

CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    # ✅ FOOD: chỉ về ăn uống
    "food": [
        "gao com mi tom sua nuoc mam rau cu qua do an thuc pham",
        "rice noodles milk food groceries canned meat vegetables",
        "an com gao mi do an",
    ],
    
    # ✅ VEHICLE: chỉ về di chuyển
    "vehicle": [
        "xe may xe dap xe lan phuong tien di lai van chuyen",
        "motorcycle bicycle scooter wheelchair transport vehicle",
        "chay xe di xe",
    ],
    
    # ✅ CLOTHES: chỉ về mặc mũ dép - KHÔNG có nội thất!
    "clothes": [
        "quan ao giay dep chan man vay ao thun ao khoac do mac thoi trang",
        "clothes shoes pants jacket shirt apparel fashion",
        "ao em be quan be giay em be",
    ],
    
    # ✅ EDUCATION: chỉ về học hành - KHÔNG có nội thất!
    "education": [
        "sach vo but tap hoc phi laptop may tinh giao khoa do hoc tap",
        "book notebook textbook school tuition computer writing supplies",
        "hoc tap em be tap chi but chi",
    ],
    
    # ✅ HOUSEHOLD: chỉ về nội thất/bếp/gia dụng - KHÔNG có clothes!
    "household": [
        "noi bep bep gas tu lanh may giat quat dien ban ghe tu giuong do gia dung",
        "kitchen furniture fridge washing machine fan bed table chair",
        "tu ao noi niều do sinh hoat tu do",
    ],
    
    # ✅ MEDICAL: chỉ về y tế - KHÔNG overlap
    "medical": [
        "thuoc y te phau thuat benh vien kham benh sang khoe tiem chung",
        "medicine doctor hospital health care treatment vaccine",
        "thuoc men y te suc khoe",
    ],
}

# ---------------------------
# Explicit Keywords - HIGH PRIORITY (dùng trước semantic inference)
# ---------------------------
CATEGORY_EXPLICIT_KEYWORDS: Dict[str, Set[str]] = {
     "food": {
        "gao", "gạo", "com", "cơm", "mi", "mì", "banh", "bánh",
        "my tom", "mỳ tôm", "tom", "tôm", "sua", "sữa", "nuoc", "nước",
        "mam", "mắm", "do an", "đồ ăn", "thuc pham", "thực phẩm",
        "me", "mè", "tam", "tám", "ca", "cá", "thit", "thịt",
        "rau", "cu", "qua", "trung", "trứng",
        "sua dac", "sữa đặc", "sua tuoi", "sữa tươi", "sua hop", "sữa hộp",
        "banh mi", "bánh mì", "thieu an", "thiếu ăn", "doi", "đói",
    },
    
    "vehicle": {
        "xe may", "xe máy", "xe dap", "xe đạp", "xe lan", "xe lăn",
        "phuong tien", "phương tiện", "motor", "bike", "bicycle",
        "scooter", "xe go", "xe gỗ", "xe tay ga", "xe tho", "xe thở",
    },
    
    "clothes": {
        "quan ao", "quần áo", "giay", "giày", "dep", "dép",
        "ao thun", "áo thun", "ao khoac", "áo khoác", "quan jean",
        "vay", "váy", "chan", "chân", "man", "màn", "do mac", "đồ mặc",
        "ao am", "áo âm", "do gia", "đồ già", "thoi trang", "thời trang",
        "tay ao", "tay áo", "non", "nón", "khant", "khănt", "vo", "vớ",
        "balo", "ba lô",
    },
    
    "household": {
        "noi", "nồi", "bep", "bếp", "bep gas", "bếp gas", "tu lanh",
        "tủ lạnh", "may giat", "máy giặt", "quat", "quạt", "dien", "điện",
        "ban", "bàn", "ghe", "ghế", "tu", "tủ", "tu quan ao", "tủ quần áo",
        "giuong", "giường", "do gia dung", "đồ gia dụng",
        "do sinh hoat", "đồ sinh hoạt", "noi nieu", "nồi nêu", "chao", "chảo",
        "xoong", "xoong", "chom", "chóm", "may hut", "máy hút",
        "tu do", "tủ đồ", "tu ao", "tủ áo",
    },
    
    "education": {
        "sach", "sách", "vo", "vở", "but", "bút", "tap", "tập",
        "hoc", "học", "hoc phi", "học phí", "laptop", "may tinh", "máy tính",
        "giao khoa", "giáo khoa", "cap hoc sinh", "cấp học sinh",
        "em be", "em bé", "bang diem", "bảng điểm", "bang cap", "bằng cấp",
        "chung chi", "chứng chỉ",
    },
    
    "medical": {
        "thuoc", "thuốc", "y te", "y tế", "benh vien", "bệnh viện",
        "kham benh", "khám bệnh", "phau thuat", "phẫu thuật",
        "tiem chung", "tiêm chủng", "cam cum", "cảm cúm",
        "dau om", "đau ốm", "suc khoe", "sức khỏe", "bac si", "bác sĩ",
        "dung cu y te", "dụng cụ y tế", "bang phuc", "băng phục",
    },

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