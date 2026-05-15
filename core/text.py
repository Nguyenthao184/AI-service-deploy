from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from core import config as core_config
from core.similarity import semantic_similarity_single_target


def normalize_semantic_text(text: str) -> str:
    """
    Chuẩn hóa văn bản cho ghép ngữ nghĩa / luật:
    - chữ thường
    - bỏ dấu
    - chỉ giữ ký tự từ và khoảng trắng
    - gộp khoảng trắng
    """
    value = (text or "").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def semantic_level(score: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


# ✅ NEW: Explicit keyword check TRƯỚC semantic inference
def get_category_by_explicit_keywords(text: str) -> Optional[str]:
    """
    Kiểm tra từ khóa rõ ràng - nếu match → return category luôn.
    Độ ưu tiên cao hơn semantic inference vì keyword = signal rõ ràng.
    
    Examples:
    - "cần gạo" → food (match "gao")
    - "tặng giường" → household (match "giuong")
    - "cần sách" → education (match "sach")
    """
    text_norm = normalize_semantic_text(text)
    
    # Kiểm tra từng danh mục
    for category, keywords in core_config.CATEGORY_EXPLICIT_KEYWORDS.items():
        for keyword in keywords:
            # Word boundary: " keyword " để tránh false positive
            if f" {keyword} " in f" {text_norm} ":
                return category
    
    return None


# ✅ IMPROVED: Semantic inference với confidence gap check
# File: text.py

def infer_category_label(text: str) -> Tuple[Optional[str], float]:
    """
    Inference danh mục từ semantic similarity với gap checking.
    """
    normalized = normalize_semantic_text(text)
    if not normalized:
        return None, 0.0

    best_label: Optional[str] = None
    best_score = -1.0
    scores_by_label: Dict[str, float] = {}
    
    for label in core_config.CATEGORY_LABELS:
        prototypes = core_config.CATEGORY_PROTOTYPES.get(label, [])
        if not prototypes:
            continue
            
        sims = semantic_similarity_single_target(normalized, prototypes)
        label_score = float(np.max(sims)) if len(sims) > 0 else 0.0
        scores_by_label[label] = label_score
        
        if label_score > best_score:
            best_score = label_score
            best_label = label

    # ✅ Giảm confidence threshold từ 0.75 → 0.60
    if best_label is None or best_score < 0.60:
        return None, 0.0
    
    # ✅ Kiểm tra gap (confidence gap)
    sorted_scores = sorted(scores_by_label.values(), reverse=True)
    if len(sorted_scores) >= 2:
        gap = sorted_scores[0] - sorted_scores[1]
        # ✅ Giảm gap threshold từ 0.10 → 0.05 để strict hơn
        if gap < 0.05:
            return None, 0.0
    
    return best_label, max(0.0, min(1.0, best_score))


def normalize_category_label(raw: Optional[str], fallback_text: str) -> Tuple[Optional[str], float]:
    value = (raw or "").strip().lower()
    if value in core_config.CATEGORY_LABELS:
        return value, 1.0
    return infer_category_label(fallback_text)

def extract_facets(text: str) -> Set[str]:
    text = normalize_semantic_text(text)
    facets = set()

    if any(f" {k} " in f" {text} " for k in [
        "gao", "mi tom", "sua", "do an", "thuc pham"
    ]):
        facets.add("food_basic")
    if any(f" {k} " in f" {text} " for k in [
        "ban hoc",
        "ghe hoc sinh",
        "ghe hoc",
        "ban hoc sinh",
    ]):
        facets.add("edu_study")
    if any(f" {k} " in f" {text} " for k in [
        "cap hoc sinh",
        "cap sach",
        "cap",
        "balo",
        "ba lo",
    ]):
        facets.add("edu_bag")
    if any(f" {k} " in f" {text} " for k in [
    "rau", "cu", "rau cu"
    ]):
        facets.add("food_vegetable")
    if any(f" {k} " in f" {text} " for k in [
        "sua em be", "sua tre em", "bot an dam"
    ]):
        facets.add("food_baby")

    if any(f" {k} " in f" {text} " for k in [
        "sach", "vo", "giao khoa"
    ]):
        facets.add("edu_books")

    if any(f" {k} " in f" {text} " for k in [
        "but", "viet", "tap"
    ]):
        facets.add("edu_writing")

    if any(f" {k} " in f" {text} " for k in [
        "laptop", "may tinh", "tablet"
    ]):
        facets.add("edu_tech")

    if any(f" {k} " in f" {text} " for k in [
        "noi", "bep", "tu lanh", "lo vi song"
    ]):
        facets.add("house_kitchen")

    if any(f" {k} " in f" {text} " for k in [
        "giuong",
        "nem",
        "chan ga",
    ]):
        facets.add("house_sleep")
    if any(f" {k} " in f" {text} " for k in [
        "ke sach",
        "ke tivi",
    ]):
        facets.add("house_storage")
    if any(f" {k} " in f" {text} " for k in [
        "ban ghe",
        "ghe sofa",
        "sofa",
        "ban an",
        "ghe go",
        "bo ban ghe",
    ]):
        facets.add("house_table")

    if any(f" {k} " in f" {text} " for k in [
        "tu quan ao", "tu ao", "tu do"
    ]):
        facets.add("house_storage")

    if any(f" {k} " in f" {text} " for k in [
        "quan ao", "ao khoac", "giay", "dep"
    ]):
        facets.add("clothes_wear")

    return facets
def has_multi_intent_overlap(target_text: str, cand_text: str) -> bool:
    """
    Cầu ý định theo luật cho các tình huống quyên góp thường gặp.
    """
    tw = _text_for_wearable_clothes_intent(target_text)
    cw = _text_for_wearable_clothes_intent(cand_text)
    rules: List[Tuple[Set[str], Set[str], str]] = [
        ({"quan ao", "do mac", "ao thun", "ao khoac"}, {"quan ao", "do mac", "ao thun", "giay", "dep", "chan am"}, "wear"),
        ({"thuc pham", "do an", "gao"}, {"gao", "mi", "my tom", "do an", "thuc pham"}, "any"),
        ({"hoc phi", "sach vo", "hoc tap"}, {"sach", "vo", "tap", "laptop", "may tinh"}, "any"),
        ({"do gia dung", "noi com", "noi nieu", "bep gas"}, {"do sinh hoat", "noi nieu", "quat dien", "tu lanh", "may giat"}, "any"),
        ({"giuong", "tu quan ao", "ban ghe"}, {"noi that", "do gia dung", "do sinh hoat", "tu quan ao"}, "any"),
    ]
    for target_terms, cand_terms, mode in rules:
        t_src = tw if mode == "wear" else target_text
        c_src = cw if mode == "wear" else cand_text
        if any(f" {term} " in f" {t_src} " for term in target_terms) and any(f" {term} " in f" {c_src} " for term in cand_terms):
                       
            return True
    return False


def should_reject_food_mismatch(target_text: str, cand_text: str) -> bool:
    """
    Loại nếu target và candidate là food nhưng khác loại:
    - target "gạo" nhưng candidate "mì"
    - target "sữa" nhưng candidate "gạo"
    """
    food_keywords = {
        "gao": {"gao", "com"},
        "mi": {"mi", "mi tom", "my tom"},
        "sua": {"sua tuoi", "sua bot", "sua dac", "sua hop"},
    }
    
    target_hits = {k for k, kws in food_keywords.items() 
                   if any(f" {kw} " in f" {target_text} " for kw in kws)}
    cand_hits = {k for k, kws in food_keywords.items() 
                 if any(f" {kw} " in f" {cand_text} " for kw in kws)}

    if not target_hits or not cand_hits:
        return False
    
    return len(target_hits & cand_hits) == 0


def is_emergency_case(text: str) -> bool:
    return any(f" {k} " in f" {text} " for k in ["chay nha", "mat nha", "hoa hoan"])

    
def should_reject_education_mismatch(target_text: str, cand_text: str) -> bool:
    """
    Chỉ reject khi cả hai đều có keywords giáo dục nhưng thuộc nhóm KHÁC.
    """
    edu_groups = {
        "books": {"sach", "vo"},        
        "writing": {"but"},             
        "tech": {"laptop", "may tinh"}, 
        "general": {"hoc tap"},         
    }

    target_hits = {g for g, kw in edu_groups.items() 
                  if any(f" {k} " in f" {target_text} " for k in kw)}
    cand_hits = {g for g, kw in edu_groups.items() 
                 if any(f" {k} " in f" {cand_text} " for k in kw)}

    if not target_hits or not cand_hits:
        return False

    if target_hits & cand_hits:
        return False

    return True


def must_reject_by_rules(
    target_text: str,
    cand_text: str,
    match_sim: float,
    target_category: Optional[str],
    cand_category: Optional[str],
) -> bool:
    if not cand_text:
        return True

    target_words = set(target_text.split())
    cand_words = set(cand_text.split())
    overlap = len(target_words & cand_words)
    
    if match_sim < core_config.REJECT_LOW_SIM_THRESHOLD and overlap == 0:
        return True

    if (
        target_category is not None
        and cand_category is not None
        and target_category != cand_category
        and match_sim < core_config.CATEGORY_MISMATCH_REJECT_SIM
    ):
        return True

    return False


def is_relevant_enough(match_sim: float) -> bool:
    return match_sim >= core_config.MIN_SIM_LOOSE


def urgency_score(text: str) -> float:
    score = 0.0
    urgent_keywords = [
        "khan cap",
        "gap",
        "can ngay",
        "chay nha",
        "mat het",
        "khong con",
        "doi",
        "thieu an",
    ]
    for kw in urgent_keywords:
        if kw in text:
            score += 0.5
    return min(score, 1.0)


def relevance_penalty(match_sim: float) -> float:
    if match_sim < 0.45:
        return -1.5
    if match_sim < 0.5:
        return -0.5
    return 0.0


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(f" {k} " in f" {text} " for k in keywords)


def _fold_vn_d(text: str) -> str:
    """Chuẩn hóa chữ đ/Đ → d để khớp cụm mùa sau bước bỏ dấu."""
    return text.replace("\u0111", "d").replace("\u0110", "d")


def _has_clothes_context(text: str) -> bool:
    """Có tín hiệu quần áo (tránh chỉ dựa substring 'ao' quá ngắn)."""
    return _contains_any(
        text,
        [
            "quan ao",
            "do mac",
            "ao khoac",
            "quan jean",
            "ao am",
            "ao thun",
            "vay ",
            " chan ",
            " chan",
            " man ",
            " man",
            "giay ",
            " dep ",
            " dep",
        ],
    )


def _clothes_season_winter(text: str) -> bool:
    t = _fold_vn_d(text)
    return _contains_any(
        t,
        [
            "mua dong",
            "dong lanh",
            "ret muot",
            "lanh gia",
            "ao am",
            "chan am",
            "quan len",
            "non len",
            "khan len",
            "giu am",
            "ao long",
            "lot am",
        ],
    )


def _clothes_season_summer(text: str) -> bool:
    t = _fold_vn_d(text)
    return _contains_any(
        t,
        [
            "mua he",
            "mua ha",
            "quan dui",
            "quan short",
            "vay he",
            "ao thun mong",
            "non rong",
            "dep tong",
            "ong rong",
        ],
    )


def should_reject_clothes_season_mismatch(target_text: str, cand_text: str) -> bool:
    """
    Loại khi một bài nói rõ quần áo mùa đông và bài kia rõ mùa hè.
    """
    if not (_has_clothes_context(target_text) and _has_clothes_context(cand_text)):
        return False
    tw = _clothes_season_winter(target_text)
    ts = _clothes_season_summer(target_text)
    cw = _clothes_season_winter(cand_text)
    cs = _clothes_season_summer(cand_text)
    if tw and cs:
        return True
    if ts and cw:
        return True
    return False


def should_reject_wardrobe_clothes_mismatch(target_text: str, cand_text: str) -> bool:
    """
    Alias cho should_reject_clothes_season_mismatch().
    Loại khi quần áo không phù hợp mùa.
    """
    return should_reject_clothes_season_mismatch(target_text, cand_text)


# Cụm tủ / kệ — nếu không tách, chuỗi "tu quan ao" sẽ khớp nhầm cụm "quan ao" (quần áo).
_WARDROBE_STORAGE_PHRASES: Tuple[str, ...] = (
    "tu quan ao",
    "tu ao",
    "ke quan ao",
    "tu do",
)


def _text_for_wearable_clothes_intent(text: str) -> str:
    t = text
    for p in _WARDROBE_STORAGE_PHRASES:
        t = t.replace(p, " ")
    return re.sub(r"\s+", " ", t).strip()


def _has_wearable_clothes_context(text: str) -> bool:
    """Tín hiệu quần áo để mặc (sau khi đã bỏ cụm tủ)."""
    if not text:
        return False
    return _contains_any(
        text,
        [
            "quan ao",
            "do mac",
            "ao thun",
            "ao khoac",
            "ao len",
            "ao am",
            "quan jean",
            "quan tay",
            "quan dui",
            "quan short",
            "vay ",
            " vay",
            "giay ",
            " giay",
            "dep ",
            " dep",
            "chan am",
            "non len",
            "khan quang",
            "gang tay",
            "tui xach",
            "balo",
        ],
    )


def is_cross_domain_hard_reject(target_text: str, cand_text: str) -> bool:
    """
    Loại cứng các cặp lệch miền rõ ràng.
    """
    edu_keywords = ["hoc tap", "sach", "vo", "but", "hoc phi", "laptop", "may tinh"]
    vehicle_keywords = ["xe may", "xe dap", "xe lan", "phuong tien", "o to", "oto"]

    target_is_edu = _contains_any(target_text, edu_keywords)
    cand_is_edu = _contains_any(cand_text, edu_keywords)
    target_is_vehicle = _contains_any(target_text, vehicle_keywords)
    cand_is_vehicle = _contains_any(cand_text, vehicle_keywords)

    if target_is_edu and cand_is_vehicle and not cand_is_edu:
        return True
    if target_is_vehicle and cand_is_edu and not cand_is_vehicle:
        return True
    return False


def extract_intents(text: str) -> Set[str]:
    intents: Set[str] = set()

    # Ánh xạ ngữ cảnh khi khẩn cấp
    if any(f" {k} " in f" {text} " for k in ["chay nha", "mat nha", "mat het", "khong con nha", "hoa hoan"]):
        intents.update({"household", "clothes", "food"})

    clothes_text = _text_for_wearable_clothes_intent(text)

    groups: Dict[str, List[str]] = {
        "education": [
            "hoc tap", "sach", "vo", "but", "hoc phi", "laptop", "may tinh",
            "giao khoa", "cap hoc sinh", "ban hoc sinh","ban hoc tap", "ghe hoc sinh",
        ],
        "vehicle": ["xe may", "xe dap", "xe lan", "phuong tien", "o to", "oto", "xe tay ga"],
        "food": [
            "gao", "my tom", "mi tom", "thuc pham", "do an", "sua", "thieu an",
            "khong du an", "doi", "can thuc pham", "can gao", "can do an",
        ],
        "clothes": [
            "quan ao",
            "do mac",
            "ao thun",
            "ao khoac",
            "ao len",
            "ao am",
            "quan jean",
            "quan tay",
            "quan dui",
            "quan short",
            "quan lot",
            "jean",
            "giay",
            "dep",
            " vay",
            "vay ",
            "chan am",
            "man lot",
            "non len",
            "khan quang",
            "gang tay",
            "tui xach",
            "balo",
        ],
        "household": [
            "noi com",
            "noi nieu",
            "noi ap suat",
            "noi chao",
            "bep gas",
            "bep dien",
            "bep tu",
            "gia dung",
            "do gia dung",
            "do sinh hoat",
            "quat dien",
            "quat cay",
            "quat ban",
            "quat tran",
            "tu lanh",
            "may giat",
            "ban ghe",
            "giuong",
            "tu quan ao",
            "tu ao",
            "tu do",
            "ke sach",
            "rem cua",
            "nem",
            "chan ga",
        ],
        "medical": ["thuoc", "y te", "phau thuat", "vien phi", "kham benh"],
    }

    for intent, keywords in groups.items():
        haystack = clothes_text if intent == "clothes" else text
        for kw in keywords:
            if f" {kw} " in f" {haystack} ":
                intents.add(intent)
                break

    if re.search(r"(^|\s)quat(\s|$)", text):
        intents.add("household")

    return intents


# Facet gia dụng: cùng intent household nhưng khác nhóm (vd: nồi cơm vs chỉ quạt).
HOUSEHOLD_FACET_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "kitchen": (
        "noi com",
        "noi nieu",
        "am dien",
        "bep dien",
        "bep gas",
        "bep tu",
        "lo nuong",
        "xoong",
        "chao",
        "may xay",
        "may ep",
        "noi ap suat",
        "noi chao",
        "noi nau",
    ),
    "cooling": (
        "quat dien",
        "quat cay",
        "quat ban",
        "quat dung",
        "quat tran",
        "quat lung",
        " dieu hoa",
        " may lanh",
    ),
    "laundry": ("may giat", "may say tam", "may say quan ao"),
    "cold_storage": ("tu lanh", "tu dong"),
    "furniture": (
        "tu quan ao",
        "tu ao",
        "giuong",
        "nem",
        "chan ga",
        "rem cua",
        "ban ghe",
        "ghe sofa",
        "sofa",
        "ke sach",
        "ke tivi",
    ),
}


def extract_household_facets(text: str) -> Set[str]:
    facets: Set[str] = set()
    for facet, phrases in HOUSEHOLD_FACET_KEYWORDS.items():
        if any(f" {ph} " in f" {text} " for ph in phrases):
            facets.add(facet)
    if "cooling" not in facets and re.search(r"(^|\s)quat(\s|$)", text):
        facets.add("cooling")
    return facets


def _household_multi_item_or_bundle_language(text: str) -> bool:
    """Bài liệt kê nhiều đồ / combo → không khóa theo facet một-một."""
    if _contains_any(
        text,
        [
            "do gia dung",
            "do sinh hoat",
            "nhieu do",
            "cac loai",
            "combo",
            "goi ho tro",
            "hang tan phong",
        ],
    ):
        return True
    if re.search(r"\b(va|v[aà]|k[eè]m|c[uù]ng\s+v[oớ]i)\b", text):
        return True
    return False


def should_reject_household_facet_mismatch(target_text: str, cand_text: str) -> bool:
    """
    Loại khi cả hai đều gia dụng cụ thể nhưng khác nhóm (vd: chỉ nồi cơm vs chỉ quạt).
    Cho phép khi một bên nêu nhiều món (nồi cơm và quạt) hoặc ngôn ngữ gói chung.
    """
    if not core_config.HOUSEHOLD_FACET_STRICT:
        return False
    if is_emergency_case(target_text) or is_emergency_case(cand_text):
        return False
    if _household_multi_item_or_bundle_language(target_text) or _household_multi_item_or_bundle_language(cand_text):
        return False

    tf = extract_household_facets(target_text)
    cf = extract_household_facets(cand_text)
    if not tf or not cf:
        return False
    if len(tf) >= 2 or len(cf) >= 2:
        return False
    if tf & cf:
        return False
    return "household" in extract_intents(target_text) and "household" in extract_intents(cand_text)


def _has_vehicle_signal(text: str) -> bool:
    return _contains_any(
        text,
        ["xe may", "xe dap", "xe lan", "phuong tien", "o to", "oto", "xe go", "xe tay ga"],
    )


def _has_food_signal(text: str) -> bool:
    return _contains_any(
        text,
        [
            "gao",
            "com ",
            " com",
            "thuc pham",
            "do an",
            "mi tom",
            "my tom",
            "sua ",
            "thieu an",
            "doi ",
            "can gao",
            "can do an",
            "banh ",
            "rau ",
            "thit ",
        ],
    )


def _has_education_signal(text: str) -> bool:
    return _contains_any(
        text,
        [
            "hoc tap",
            "sach ",
            " sach",
            "vo ",
            " vo",
            "but ",
            " but",
            "hoc phi",
            "laptop",
            "may tinh",
            "giao khoa",
            "cap hoc sinh",
        ],
    )


def _has_medical_signal(text: str) -> bool:
    return _contains_any(
        text,
        ["thuoc", "y te", "benh vien", "kham benh", "phau thuat", "tiem chung"],
    )


def should_reject_vehicle_vs_static_goods_cross(target_text: str, cand_text: str) -> bool:
    """
    Xe / phương tiện không ghép related với gạo, sách, nồi cơm… nếu mỗi bên chỉ một miền hàng.
    """
    if is_emergency_case(target_text) or is_emergency_case(cand_text):
        return False

    vt = _has_vehicle_signal(target_text)
    vc = _has_vehicle_signal(cand_text)
    if vt and vc:
        return False

    def _static_goods_side(t: str) -> bool:
        return (
            _has_food_signal(t)
            or _has_education_signal(t)
            or _has_medical_signal(t)
            or bool(extract_household_facets(t))
            or _contains_any(
                t,
                [
                    "do gia dung",
                    "do sinh hoat",
                    "gia dung",
                    "noi com",
                    "quat dien",
                    "tu lanh",
                    "may giat",
                ],
            )
        )

    if vt and not vc and _static_goods_side(cand_text):
        return True
    if vc and not vt and _static_goods_side(target_text):
        return True
    return False
def should_reject_vehicle_furniture_cross(target_text: str, cand_text: str) -> bool:
    """
    Reject vehicle vs furniture mismatch
    VD:
    - xe đạp ↔ giường
    - xe máy ↔ sofa

    Không reject:
    - xe ↔ xe
    """

    vehicle_keywords = [
        "xe may", "xe dap", "xe lan", "phuong tien",
        "o to", "oto", "xe tay ga", "motor", "bike"
    ]

    furniture_keywords = [
        "giuong",
        "nem",
        "sofa",
        "ke sach",
        "ke tivi",
        "ban an",
        "tu ao",
        "tu do"
    ]

    target_is_vehicle = any(f" {k} " in f" {target_text} " for k in vehicle_keywords)
    cand_is_vehicle = any(f" {k} " in f" {cand_text} " for k in vehicle_keywords)
    
    target_is_furniture = any(f" {k} " in f" {target_text} " for k in furniture_keywords)
    cand_is_furniture = any(f" {k} " in f" {cand_text} " for k in furniture_keywords)

    if target_is_vehicle and cand_is_furniture and not cand_is_vehicle:
        return True

    if target_is_furniture and cand_is_vehicle and not cand_is_furniture:
        return True

    return False
def should_reject_education_food_cross(target_text: str, cand_text: str) -> bool:
    """
    Reject education vs food mismatch

    VD:
    - sách ↔ mì tôm
    - laptop ↔ sữa
    """

    edu_keywords = [
        "sach",
        "vo",
        "but",
        "tap",
        "hoc phi",
        "laptop",
        "may tinh",
        "giao khoa",
        "cap hoc sinh",
        "hoc tap"
    ]

    food_keywords = [
        "gao",
        "mi tom",
        "my tom",
        "thuc pham",
        "do an",
        "sua",
        "banh",
        "rau",
        "thit",
        "nuoc",
        "mam",
        "trung"
    ]

    target_is_edu = any(f" {k} " in f" {target_text} " for k in edu_keywords)
    cand_is_edu = any(f" {k} " in f" {cand_text} " for k in edu_keywords)

    target_is_food = any(f" {k} " in f" {target_text} " for k in food_keywords)
    cand_is_food = any(f" {k} " in f" {cand_text} " for k in food_keywords)

    if target_is_edu and cand_is_food and not cand_is_edu:
        return True

    if target_is_food and cand_is_edu and not cand_is_food:
        return True

    return False
def _has_study_area_furniture_signal(text: str) -> bool:
    """Bàn/ghế học tập — không nhầm với máy giặt / bếp."""
    return _contains_any(
        text,
        [
            "ban hoc",
            "ghe hoc sinh",
            "ghe hoc",
            "ban ghe hoc",
            "ban hoc sinh",
        ],
    )
def should_reject_household_vs_education_furniture(
    target_text: str,
    cand_text: str,
) -> bool:

    study_keywords = [
        "ban hoc",
        "ban hoc sinh",
        "ban hoc tap",
        "ghe hoc sinh",
    ]

    furniture_keywords = [
        "ban ghe",
        "bo ban ghe",
        "ghe sofa",
        "sofa",
        "ban an",
    ]

    target_is_study = any(f" {k} " in f" {target_text} " for k in study_keywords)
    cand_is_study = any(f" {k} " in f" {cand_text} " for k in study_keywords)

    target_is_furniture = any(f" {k} " in f" {target_text} " for k in furniture_keywords)
    cand_is_furniture = any(f" {k} " in f" {cand_text} " for k in furniture_keywords)

    if target_is_study and cand_is_furniture and not cand_is_study:
        return True

    if target_is_furniture and cand_is_study and not cand_is_furniture:
        return True

    return False

def _has_major_appliance_signal(text: str) -> bool:
    facets = extract_household_facets(text)
    if facets & {"kitchen", "laundry", "cold_storage"}:
        return True
    return _contains_any(
        text,
        ["may giat", "bep gas", "bep dien", "bep tu", "tu lanh", "noi com", "quat dien"],
    )


def should_reject_study_furniture_vs_major_appliance(target_text: str, cand_text: str) -> bool:
    """
    Bàn học / ghế học sinh không related máy giặt, bếp gas, tủ lạnh… (cùng DB household vẫn lọc được).
    """
    if is_emergency_case(target_text) or is_emergency_case(cand_text):
        return False
    st = _has_study_area_furniture_signal(target_text)
    sc = _has_study_area_furniture_signal(cand_text)
    at = _has_major_appliance_signal(target_text)
    ac = _has_major_appliance_signal(cand_text)
    if st and ac and not sc and not at:
        return True
    if sc and at and not st and not ac:
        return True
    return False


def should_reject_wearable_clothes_vs_storage_furniture(target_text: str, cand_text: str) -> bool:
    """
    Quần áo để mặc vs tủ / giường / nội thất chứa đồ.
    """
    wear_t = _has_wearable_clothes_context(_text_for_wearable_clothes_intent(target_text))
    wear_c = _has_wearable_clothes_context(_text_for_wearable_clothes_intent(cand_text))
    furn_t = _contains_any(
        target_text,
        ["tu quan ao", "tu ao", "ke quan ao", "giuong", "nem", "tu do", "ban ghe", "ke sach"],
    )
    furn_c = _contains_any(
        cand_text,
        ["tu quan ao", "tu ao", "ke quan ao", "giuong", "nem", "tu do", "ban ghe", "ke sach"],
    )

    if wear_t and furn_c and not wear_c and not furn_t:
        return True
    if wear_c and furn_t and not wear_t and not furn_c:
        return True
    return False


def should_reject_by_intent(target_text: str, cand_text: str) -> bool:

    target_intents = extract_intents(target_text)
    cand_intents = extract_intents(cand_text)

    if not target_intents or not cand_intents:
        return False

    overlap = target_intents & cand_intents

    if overlap:
        return False

    if "vehicle" in target_intents and "vehicle" not in cand_intents:
        return True

    if "vehicle" in cand_intents and "vehicle" not in target_intents:
        return True

    return True


def is_food_urgency_target(target_text: str) -> bool:
    food_need_keywords = [
        "khong du an", "thieu an", "doi", "can thuc pham",
        "can gao", "can do an"
    ]
    return _contains_any(target_text, food_need_keywords)


def should_reject_for_food_urgency(target_text: str, cand_text: str) -> bool:
    """
    Chế độ chặt cho yêu cầu thực phẩm khẩn.
    """
    if not is_food_urgency_target(target_text):
        return False
    cand_intents = extract_intents(cand_text)
    if "food" in cand_intents:
        return False
    if "household" in cand_intents:
        return False
    return True


def should_reject_vehicle_offer_when_vehicle_not_allowed(
    cand_text: str,
    allowed_categories: Set[str],
) -> bool:
    """
    Bài CHO chỉ về xe/phương tiện: loại nếu target không có nhãn vehicle.
    """
    if "vehicle" in allowed_categories:
        return False
    if not _contains_any(cand_text, ["xe may", "xe dap", "xe lan", "phuong tien", "o to", "oto"]):
        return False
    if _contains_any(
        cand_text,
        [
            "thuc pham", "gao", "do an", "mi tom", "my tom", "tang gao",
            "tang mi", "com ", " com", "banh mi", "quan ao", "ao quan",
            "tang ao", "ao am", "giay", "dep", "quan jean", "chan ",
            " man",
        ],
    ):
        return False
    return True
def should_reject_household_food_cross(target_text: str, cand_text: str) -> bool:
    """
    Reject household appliance vs food mismatch

    VD:
    - quạt điện ↔ rau củ
    - nồi cơm ↔ gạo
    """

    household_keywords = [
        "quat dien",
        "quat cay",
        "quat ban",
        "noi com",
        "noi dien",
        "bep gas",
        "bep dien",
        "tu lanh",
        "may giat",
        "do gia dung",
        "gia dung",
    ]

    food_keywords = [
        "gao",
        "rau",
        "rau cu",
        "thit",
        "ca",
        "mi tom",
        "my tom",
        "do an",
        "thuc pham",
        "sua",
    ]

    target_is_house = any(f" {k} " in f" {target_text} " for k in household_keywords)
    cand_is_house = any(f" {k} " in f" {cand_text} " for k in household_keywords)

    target_is_food = any(f" {k} " in f" {target_text} " for k in food_keywords)
    cand_is_food = any(f" {k} " in f" {cand_text} " for k in food_keywords)

    if target_is_house and cand_is_food and not cand_is_house:
        return True

    if target_is_food and cand_is_house and not target_is_house:
        return True

    return False

def should_reject_for_vehicle_target(target_text: str, cand_text: str) -> bool:
    
    target_intents = extract_intents(target_text)
    if "vehicle" not in target_intents:
        return False
    cand_intents = extract_intents(cand_text)
    return "vehicle" not in cand_intents