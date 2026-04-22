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


def has_multi_intent_overlap(target_text: str, cand_text: str) -> bool:
    """
    Cầu ý định theo luật cho các tình huống quyên góp thường gặp.
    """
    rules: List[Tuple[Set[str], Set[str]]] = [
        ({"quan ao", "quan", "ao", "do mac"}, {"ao", "quan", "mac", "chan", "am"}),
        ({"thuc pham", "do an", "gao"}, {"gao", "mi", "my tom", "do an", "thuc pham"}),
        ({"chay nha", "mat nha", "hoa hoan"}, {"chan", "noi", "bep", "do gia dung", "vat dung sinh hoat"}),
        ({"hoc phi", "sach vo", "hoc tap"}, {"sach", "vo", "tap", "laptop", "may tinh"}),
    ]
    for target_terms, cand_terms in rules:
        if any(term in target_text for term in target_terms) and any(term in cand_text for term in cand_terms):
            return True
    return False


def infer_category_label(text: str) -> Tuple[Optional[str], float]:
    normalized = normalize_semantic_text(text)
    if not normalized:
        return None, 0.0

    best_label: Optional[str] = None
    best_score = -1.0
    for label in core_config.CATEGORY_LABELS:
        prototypes = core_config.CATEGORY_PROTOTYPES.get(label, [])
        if not prototypes:
            continue
        sims = semantic_similarity_single_target(normalized, prototypes)
        label_score = float(np.max(sims)) if len(sims) > 0 else 0.0
        if label_score > best_score:
            best_score = label_score
            best_label = label

    if best_label is None:
        return None, 0.0
    return best_label, max(0.0, min(1.0, best_score))


def normalize_category_label(raw: Optional[str], fallback_text: str) -> Tuple[Optional[str], float]:
    value = (raw or "").strip().lower()
    if value in core_config.CATEGORY_LABELS:
        return value, 1.0
    return infer_category_label(fallback_text)


def must_reject_by_rules(
    target_text: str,
    cand_text: str,
    match_sim: float,
    target_category: Optional[str],
    cand_category: Optional[str],
) -> bool:
    # Loại bắt buộc: không có tín hiệu chữ + điểm ngữ nghĩa quá thấp.
    if not cand_text:
        return True

    target_words = set(target_text.split())
    cand_words = set(cand_text.split())
    overlap = len(target_words & cand_words)
    if match_sim < core_config.REJECT_LOW_SIM_THRESHOLD and overlap == 0:
        return True

    # Loại bắt buộc: danh mục lệch mạnh và điểm ngữ nghĩa không đủ cao.
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
    return any(k in text for k in keywords)


def _fold_vn_d(text: str) -> str:
    """Chuẩn hóa chữ đ/Đ → d để khớp cụm mùa sau bước bỏ dấu."""
    return text.replace("\u0111", "d").replace("\u0110", "d")


def _has_clothes_context(text: str) -> bool:
    """Có tín hiệu quần áo (tránh chỉ dựa substring 'ao' quá ngắn)."""
    if "clothes" in extract_intents(text):
        return True
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
    Loại khi một bài nói rõ quần áo mùa đông và bài kia rõ mùa hè (embedding thường vẫn rất giống).
    Chỉ kích hoạt khi cả hai bên đều có ngữ cảnh quần áo.
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


def is_cross_domain_hard_reject(target_text: str, cand_text: str) -> bool:
    """
    Loại cứng các cặp lệch miền rõ ràng.
    Nếu bài đích cần đồ học tập mà ứng viên chỉ có xe (hoặc ngược lại) thì loại.
    """
    edu_keywords = ["hoc tap", "sach", "vo", "but", "hoc phi", "laptop", "may tinh"]
    vehicle_keywords = ["xe may", "xe dap", "xe lan", "phuong tien"]

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

    # Ánh xạ ngữ cảnh khi khẩn cấp:
    # "cháy nhà / mất nhà" thường ngụ ý đồ gia dụng + quần áo + thực phẩm.
    if any(k in text for k in ["chay nha", "mat nha", "mat het", "khong con nha", "hoa hoan"]):
        intents.update({"household", "clothes", "food"})

    groups: Dict[str, List[str]] = {
        "education": ["hoc tap", "sach", "vo", "but", "hoc phi", "laptop", "may tinh", "giao khoa"],
        "vehicle": ["xe may", "xe dap", "xe lan", "phuong tien"],
        "food": [
            "gao",
            "my tom",
            "mi tom",
            "thuc pham",
            "do an",
            "sua",
            "thieu an",
            "khong du an",
            "doi",
            "can thuc pham",
            "can gao",
            "can do an",
        ],
        "clothes": ["quan ao", "ao", "quan", "quan jean", "jean", "ao khoac", "giay", "dep", "chan", "man", "do mac"],
        "household": ["noi", "bep", "noi com", "gia dung", "do sinh hoat"],
        "medical": ["thuoc", "y te", "phau thuat", "vien phi", "kham benh"],
    }
    for intent, keywords in groups.items():
        if any(k in text for k in keywords):
            intents.add(intent)
    return intents


def should_reject_by_intent(target_text: str, cand_text: str) -> bool:
    """
    Nếu bài đích có ý định rõ, ứng viên phải trùng ít nhất một ý định.
    Ngược lại loại để tránh nhiễu ngữ nghĩa (vd: học tập → gạo/nồi/quần áo).
    """
    target_intents = extract_intents(target_text)
    if not target_intents:
        return False
    cand_intents = extract_intents(cand_text)
    return len(target_intents & cand_intents) == 0


def is_food_urgency_target(target_text: str) -> bool:
    food_need_keywords = ["khong du an", "thieu an", "doi", "can thuc pham", "can gao", "can do an"]
    return _contains_any(target_text, food_need_keywords)


def should_reject_for_food_urgency(target_text: str, cand_text: str) -> bool:
    """
    Chế độ chặt cho yêu cầu thực phẩm khẩn:
    - giữ thực phẩm
    - cho phép gia dụng như hỗ trợ phụ
    - loại các loại khác (quần áo/xe/học tập/y tế)
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
    Tránh nhãn food/clothes nhầm trong DB vẫn làm giao khác rỗng.
    """
    if "vehicle" in allowed_categories:
        return False
    if not _contains_any(cand_text, ["xe may", "xe dap", "xe lan", "phuong tien"]):
        return False
    if _contains_any(
        cand_text,
        [
            "thuc pham",
            "gao",
            "do an",
            "mi tom",
            "my tom",
            "tang gao",
            "tang mi",
            "com ",
            " com",
            "banh mi",
            "quan ao",
            "ao quan",
            "tang ao",
            "ao am",
            "giay",
            "dep",
            "quan jean",
            "chan ",
            " man",
        ],
    ):
        return False
    return True


def should_reject_for_vehicle_target(target_text: str, cand_text: str) -> bool:
    """
    Cổng chặt khi bài đích cần phương tiện:
    khi rõ ràng xin xe/vận chuyển thì chỉ giữ ứng viên có ý định vehicle.
    """
    target_intents = extract_intents(target_text)
    if "vehicle" not in target_intents:
        return False
    cand_intents = extract_intents(cand_text)
    return "vehicle" not in cand_intents

