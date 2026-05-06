from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Set, Tuple, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

from core import config as core_config
from core import fraud as core_fraud
from core import geo as core_geo
from core import similarity as core_similarity
from core import text as core_text


app = FastAPI(title="Dịch vụ ghép nối AI SmartDonate")


class AiPost(BaseModel):
    id: int
    loai_bai: Literal["CHO", "NHAN"]
    tieu_de: str
    mo_ta: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    region: Optional[str] = None
    created_at: datetime
    danh_muc: Optional[str] = None
    danh_mucs: Optional[List[str]] = None


class MatchRequest(BaseModel):
    post_id: int
    # Cho phép 1 bài: khi không có ứng viên đối ứng trong DB, backend vẫn gọi được; endpoint trả [].
    posts: List[AiPost] = Field(min_length=1)
    # True if user has entered their address; used for location-based matching
    user_has_address: bool = True   
    # User interest categories (from liked posts). Used when user_has_address=False
    user_interests: Optional[List[Dict[str, Any]]] = None
    mode: Optional[str] = "matches" # matches: tìm kiếm theo vị trí, semantic: tìm kiếm theo nội dung


class MatchResponseItem(BaseModel):
    post_id: int
    score: float
    distance: Optional[float] = None
    match_percent: float
    reasons: List[str] = Field(default_factory=list)
    breakdown: Optional[Dict[str, float]] = None


class SemanticCandidateInput(BaseModel):
    id: int
    noi_dung: str
    category: Optional[str] = None
    location: Optional[str] = None


class SemanticMatchRequest(BaseModel):
    noi_dung: str
    candidates: List[SemanticCandidateInput] = Field(min_length=1)
    top_k: int = 5
    min_score: float = 0.2
    category: Optional[str] = None
    location: Optional[str] = None


class SemanticMatchResponseItem(BaseModel):
    id: int
    score: float
    level: Literal["HIGH", "MEDIUM", "LOW"]
    noi_dung: str


class FraudUserInput(BaseModel):
    user_id: int
    posts_per_day: float
    content_similarity: float
    donation_growth: float
    same_ip_accounts: float
    activity_score: float
    burst_activity: float = 0.0
    max_jump: float = 0.0
    variance: float = 0.0


class FraudCheckRequest(BaseModel):
    users: List[FraudUserInput] = Field(min_length=2)


class FraudCheckItem(BaseModel):
    user_id: int
    risk: Literal["HIGH", "LOW"]


class CampaignFraudInput(BaseModel):
    campaign_id: int
    campaigns_per_user: float
    donation_growth: float
    self_donation_ratio: float
    unique_donors: float
    donation_frequency: float


class CampaignFraudCheckRequest(BaseModel):
    campaigns: List[CampaignFraudInput] = Field(min_length=2)


class CampaignFraudCheckItem(BaseModel):
    campaign_id: int
    risk: Literal["HIGH", "LOW"]


"""
🔧 FIX #2: ai_service/main.py - Full patched matches() endpoint
Replace the entire @app.post("/matches") function with this
"""

@app.post("/matches", response_model=List[MatchResponseItem])
def matches(req: MatchRequest) -> List[MatchResponseItem]:
    target = next((p for p in req.posts if p.id == req.post_id), None)
    if target is None:
        raise HTTPException(status_code=400, detail="Không tìm thấy post_id trong danh sách posts.")

    others = [p for p in req.posts if p.id != req.post_id]
    if not others:
        return []

    target_text = core_text.normalize_semantic_text((target.tieu_de + " " + target.mo_ta).strip())
    
    # ✅ FIX: Build allowed_categories with infer fallback
    allowed_categories: Set[str] = set()
    if target.danh_muc:
        allowed_categories.add(target.danh_muc)
    if target.danh_mucs:
        allowed_categories.update(target.danh_mucs)

    if not allowed_categories:
        inferred, score = core_text.infer_category_label(target_text)
        if inferred and score > 0.7: 
            allowed_categories.add(inferred)

    target_intents = core_text.extract_intents(target_text)
    other_texts = [core_text.normalize_semantic_text((p.tieu_de + " " + p.mo_ta).strip()) for p in others]
    semantic_sims = core_similarity.semantic_similarity_scores(target_text, other_texts)
    lexical_sims = core_similarity.lexical_similarity_scores(target_text, other_texts)

    scored_rows_geo: List[Tuple[MatchResponseItem, float]] = []
    scored_rows_nogeo: List[Tuple[MatchResponseItem, float]] = []
    now = datetime.now(timezone.utc)
    target_urgency = core_text.urgency_score(target_text)

    w_sum = core_config.MATCH_BLEND_SEMANTIC + core_config.MATCH_BLEND_LEXICAL
    if w_sum <= 0:
        w_sum = 1.0
    w_sem = core_config.MATCH_BLEND_SEMANTIC / w_sum
    w_lex = core_config.MATCH_BLEND_LEXICAL / w_sum
    min_sim_cut = (
        core_config.MATCH_MIN_SIM_WITH_CATEGORY
        if allowed_categories
        else core_config.MATCH_MIN_SIM_NO_CATEGORY
    )
    rel_floor = (
        core_config.MATCH_RELEVANCE_FLOOR_GATED
        if allowed_categories
        else core_config.MIN_SIM_LOOSE
    )
    is_emergency = core_text.is_emergency_case(target_text)
    
    for idx, cand in enumerate(others):
        cand_text = other_texts[idx]
        semantic_sim = float(semantic_sims[idx])
        lexical_sim = float(lexical_sims[idx])
        match_sim = max(0.0, min(1.0, w_sem * semantic_sim + w_lex * lexical_sim))
        reasons: List[str] = []

        if core_config.DEBUG_SEMANTIC_MATCH:
            print("TARGET:", target_text)
            print("CAND:", cand_text)
            print("BLEND:", round(match_sim, 6), "SEM:", round(semantic_sim, 6), "LEX:", round(lexical_sim, 6))
        
        if core_text.should_reject_food_mismatch(target_text, cand_text):
            continue
        if "food" in target_intents and len(target_intents) == 1:
            cand_intents = core_text.extract_intents(cand_text)
            if "food" not in cand_intents:
                continue 
        if core_text.should_reject_education_mismatch(target_text, cand_text):
            continue
        if core_text.should_reject_wardrobe_clothes_mismatch(target_text, cand_text):
            continue
        if allowed_categories:
            cand_codes: Set[str] = set()
            if cand.danh_muc:
                cand_codes.add(cand.danh_muc)
            if cand.danh_mucs:
                cand_codes.update(cand.danh_mucs)
            
            # ✅ FIX #2: Infer candidate categories nếu không có
            if not cand_codes:
                inferred, _ = core_text.infer_category_label(cand_text)
                if inferred:
                    cand_codes.add(inferred)
            
            if "vehicle" not in allowed_categories:
                cand_codes.discard("vehicle")

            # ✅ FIX: Change logic - only skip if có category nhưng mismatch
            if cand_codes and cand_codes.isdisjoint(allowed_categories):
                continue
            reasons.append("category_gate")
            if core_text.should_reject_vehicle_offer_when_vehicle_not_allowed(
                cand_text, allowed_categories
            ):
                continue
        else:
            if not is_emergency:
                if core_text.should_reject_by_intent(target_text, cand_text):
                    continue
                reasons.append("intent_gate")

        if match_sim < min_sim_cut:
            continue
        if match_sim < rel_floor:
            continue

        similarity_score = match_sim * 7.0

        interest_score = 0.0
        if not req.user_has_address and req.user_interests:
            cand_categories = set()
            if cand.danh_muc:
                cand_categories.add(cand.danh_muc)
            if cand.danh_mucs:
                cand_categories.update(cand.danh_mucs)
            
            user_interest_codes = {interest['code'] for interest in req.user_interests}
            matching_interests = cand_categories & user_interest_codes
            
            if matching_interests:
                max_weight = max(
                    (interest['weight'] for interest in req.user_interests if interest['code'] in matching_interests),
                    default=0
                )
                interest_score = min(3.0, max_weight / 5.0)
                reasons.append("interest_match")
        
        distance_km: Optional[float] = None
        location_score = 0.0
        has_geo = not (
            target.lat is None
            or target.lng is None
            or cand.lat is None
            or cand.lng is None
        )
        if has_geo:
            distance_km = core_geo.haversine_km(
                target.lat,
                target.lng,
                cand.lat,
                cand.lng,
            )
            if req.mode == "matches":
                if distance_km > 20.0:
                    continue    
                location_score = core_geo.score_location_km(distance_km)
            elif req.mode == "related":
                location_score = 0.0
            reasons.append("geo_ok")
        else:
            reasons.append("geo_unknown")

        cand_created_at = cand.created_at
        if cand_created_at.tzinfo is None:
            cand_created_at = cand_created_at.replace(tzinfo=timezone.utc)
        else:
            cand_created_at = cand_created_at.astimezone(timezone.utc)
        delta_days = (now - cand_created_at).total_seconds() / 86400.0
        if delta_days < 0:
            delta_days = 0.0
        time_score = core_geo.score_time_days(delta_days)

        penalty = core_text.relevance_penalty(match_sim)
        final_score = similarity_score + location_score + interest_score + time_score + (target_urgency * 1.5) + penalty
        
        if not has_geo and req.user_has_address:
            final_score -= 1.25

        match_percent = min(100.0, (match_sim * 100.0))
        item = MatchResponseItem(
            post_id=cand.id,
            score=round(final_score, 6),
            distance=(round(distance_km, 6) if distance_km is not None else None),
            match_percent=round(match_percent, 2),
            reasons=reasons,
            breakdown={
                "semantic_sim": float(round(semantic_sim, 6)),
                "lexical_sim": float(round(lexical_sim, 6)),
                "match_sim": float(round(match_sim, 6)),
                "similarity_score": float(round(similarity_score, 6)),
                "location_score": float(round(location_score, 6)),
                "interest_score": float(round(interest_score, 6)),
                "time_score": float(round(time_score, 6)),
                "urgency": float(round(target_urgency, 6)),
                "penalty": float(round(penalty, 6)),
            },
        )
        bucket = scored_rows_geo if has_geo else scored_rows_nogeo
        bucket.append((item, match_sim))

    strict_rows_geo = [row for row in scored_rows_geo if row[1] >= core_config.MIN_SIM_STRICT]
    strict_rows_nogeo = [row for row in scored_rows_nogeo if row[1] >= core_config.MIN_SIM_STRICT]

    multi_need = len(allowed_categories) > 1
    intent_loose = bool(
        {"food", "clothes", "household", "education"} & target_intents
    )
    use_loose_fill = (len(strict_rows_geo) + len(strict_rows_nogeo) < 5) and (multi_need or intent_loose)

    loose_floor = core_config.MIN_SIM_LOOSE

    if not strict_rows_geo:
        scored_rows_geo.sort(key=lambda row: row[0].score, reverse=True)
        scored_rows_nogeo.sort(key=lambda row: row[0].score, reverse=True)
        merged = scored_rows_geo + scored_rows_nogeo
        return [row[0] for row in merged[:5]]

    if use_loose_fill:
        loose_geo = [
            row
            for row in scored_rows_geo
            if loose_floor <= row[1] < core_config.MIN_SIM_STRICT
        ]
        loose_nogeo = [
            row
            for row in scored_rows_nogeo
            if loose_floor <= row[1] < core_config.MIN_SIM_STRICT
        ]
        strict_ids = {row[0].post_id for row in strict_rows_geo}
        loose_geo = [row for row in loose_geo if row[0].post_id not in strict_ids]
        loose_nogeo = [row for row in loose_nogeo if row[0].post_id not in strict_ids]

        strict_rows_geo.sort(key=lambda row: row[0].score, reverse=True)
        loose_geo.sort(key=lambda row: row[0].score, reverse=True)
        loose_nogeo.sort(key=lambda row: row[0].score, reverse=True)
        merged = strict_rows_geo + loose_geo + loose_nogeo
        return [row[0] for row in merged[:5]]

    strict_rows_geo.sort(key=lambda row: row[0].score, reverse=True)
    return [row[0] for row in strict_rows_geo[:5]]


@app.post("/semantic-matches", response_model=List[SemanticMatchResponseItem])
def semantic_matches(req: SemanticMatchRequest) -> List[SemanticMatchResponseItem]:
    target_text = core_text.normalize_semantic_text(req.noi_dung or "")
    if not target_text:
        raise HTTPException(status_code=400, detail="noi_dung không được để trống.")

    candidate_texts = [core_text.normalize_semantic_text(c.noi_dung or "") for c in req.candidates]
    sims = core_similarity.semantic_similarity_single_target(target_text, candidate_texts)
    target_words = set(target_text.split())
    target_category = (req.category or "").strip().lower()
    target_location = (req.location or "").strip().lower()

    rows: List[SemanticMatchResponseItem] = []
    for idx, cand in enumerate(req.candidates):
        cand_text = candidate_texts[idx]
        score = float(sims[idx])

        cand_category = (cand.category or "").strip().lower()
        if target_category and cand_category and target_category == cand_category:
            score += 0.1

        cand_words = set(cand_text.split())
        if len(target_words & cand_words) >= 2:
            score += 0.05

        cand_location = (cand.location or "").strip().lower()
        if target_location and cand_location and target_location == cand_location:
            score += 0.05

        if core_text.has_multi_intent_overlap(target_text, cand_text):
            score += 0.05

        score = max(0.0, min(1.0, score))
        if score < req.min_score:
            continue

        rows.append(
            SemanticMatchResponseItem(
                id=cand.id,
                score=round(score, 6),
                level=core_text.semantic_level(score),
                noi_dung=cand.noi_dung,
            )
        )

    rows.sort(key=lambda x: x.score, reverse=True)
    top_k = max(1, min(50, int(req.top_k)))
    return rows[:top_k]


@app.post("/fraud-check", response_model=List[FraudCheckItem])
def fraud_check(req: FraudCheckRequest) -> List[FraudCheckItem]:
    """
    Phát hiện gian lận bằng IsolationForest với đặc trưng hành vi mở rộng.
    - predict = -1 → bất thường → HIGH
    - predict = 1  → bình thường → LOW
    """
    ds_users = req.users

    ma_tran_dac_trung = [core_fraud.build_fraud_features(nguoi_dung) for nguoi_dung in ds_users]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.2,
        random_state=42,
    )
    model.fit(ma_tran_dac_trung)

    du_doan = model.predict(ma_tran_dac_trung)

    ket_qua: List[FraudCheckItem] = []
    for chi_so, du_doan_muc in enumerate(du_doan):
        nguoi_dung = ds_users[chi_so]

        rule_high = (
            (
                nguoi_dung.posts_per_day > 5
                or nguoi_dung.burst_activity > 6
            )
            and nguoi_dung.same_ip_accounts > 5
            and (
                nguoi_dung.content_similarity >= 0.9
                or nguoi_dung.donation_growth >= 150
                or nguoi_dung.activity_score >= 15
                or nguoi_dung.max_jump >= 2500000
                or nguoi_dung.variance >= 0.15
            )
        )

        muc_rui_ro: Literal["HIGH", "LOW"] = "HIGH" if (int(du_doan_muc) == -1 or rule_high) else "LOW"
        ket_qua.append(
            FraudCheckItem(
                user_id=ds_users[chi_so].user_id,
                risk=muc_rui_ro,
            )
        )

    return ket_qua
@app.post("/related", response_model=List[MatchResponseItem])
def related(req: MatchRequest) -> List[MatchResponseItem]:
    """
    Related posts matching - SEMANTIC ONLY (no geo, no category gate)
    
    Same loai_bai (CHO↔CHO, NHAN↔NHAN)
    Match based on content similarity
    Return top matches by semantic score
    """
    target = next((p for p in req.posts if p.id == req.post_id), None)
    if target is None:
        raise HTTPException(status_code=400, detail="Không tìm thấy post_id trong danh sách posts.")
 
    others = [p for p in req.posts if p.id != req.post_id]
    if not others:
        return []
 
    target_text = core_text.normalize_semantic_text((target.tieu_de + " " + target.mo_ta).strip())
    target_intents = core_text.extract_intents(target_text)
    
    other_texts = [core_text.normalize_semantic_text((p.tieu_de + " " + p.mo_ta).strip()) for p in others]
    semantic_sims = core_similarity.semantic_similarity_scores(target_text, other_texts)
    lexical_sims = core_similarity.lexical_similarity_scores(target_text, other_texts)
 
    scored_rows: List[Tuple[MatchResponseItem, float]] = []
    now = datetime.now(timezone.utc)
    target_urgency = core_text.urgency_score(target_text)
 
    w_sum = core_config.MATCH_BLEND_SEMANTIC + core_config.MATCH_BLEND_LEXICAL
    if w_sum <= 0:
        w_sum = 1.0
    w_sem = core_config.MATCH_BLEND_SEMANTIC / w_sum
    w_lex = core_config.MATCH_BLEND_LEXICAL / w_sum
    
    min_sim_cut = core_config.MIN_SIM_LOOSE
    rel_floor = core_config.MIN_SIM_LOOSE
    
    for idx, cand in enumerate(others):
        cand_text = other_texts[idx]
        semantic_sim = float(semantic_sims[idx])
        lexical_sim = float(lexical_sims[idx])
        match_sim = max(0.0, min(1.0, w_sem * semantic_sim + w_lex * lexical_sim))
        reasons: List[str] = []
 
        if core_config.DEBUG_SEMANTIC_MATCH:
            print("TARGET:", target_text)
            print("CAND:", cand_text)
            print("BLEND:", round(match_sim, 6), "SEM:", round(semantic_sim, 6), "LEX:", round(lexical_sim, 6))

        if core_text.is_cross_domain_hard_reject(target_text, cand_text):
            continue
      
        if core_text.should_reject_wardrobe_clothes_mismatch(target_text, cand_text):
            continue

        if core_text.should_reject_education_mismatch(target_text, cand_text):
            continue

        if core_text.should_reject_by_intent(target_text, cand_text):
            continue
        if match_sim < min_sim_cut:
            continue
        if match_sim < rel_floor:
            continue
 
        similarity_score = match_sim * 7.0
 
        cand_created_at = cand.created_at
        if cand_created_at.tzinfo is None:
            cand_created_at = cand_created_at.replace(tzinfo=timezone.utc)
        else:
            cand_created_at = cand_created_at.astimezone(timezone.utc)
        delta_days = (now - cand_created_at).total_seconds() / 86400.0
        if delta_days < 0:
            delta_days = 0.0
        time_score = core_geo.score_time_days(delta_days)
 
        penalty = core_text.relevance_penalty(match_sim)
        
        final_score = similarity_score + time_score + (target_urgency * 0.5) + penalty
        
        reasons.append("semantic_match")
 
        match_percent = min(100.0, (match_sim * 100.0))
        item = MatchResponseItem(
            post_id=cand.id,
            score=round(final_score, 6),
            distance=None,  
            match_percent=round(match_percent, 2),
            reasons=reasons,
            breakdown={
                "semantic_sim": float(round(semantic_sim, 6)),
                "lexical_sim": float(round(lexical_sim, 6)),
                "match_sim": float(round(match_sim, 6)),
                "similarity_score": float(round(similarity_score, 6)),
                "location_score": 0.0,
                "interest_score": 0.0,
                "time_score": float(round(time_score, 6)),
                "urgency": float(round(target_urgency, 6)),
                "penalty": float(round(penalty, 6)),
            },
        )
        scored_rows.append((item, match_sim))
 
    scored_rows.sort(key=lambda row: row[0].score, reverse=True)
    
    filtered = [row for row in scored_rows if row[0].match_percent >= 50.0]
    
    if not filtered:
        filtered = scored_rows
    
    return [row[0] for row in filtered[:10]]

@app.post("/campaign-fraud-check", response_model=List[CampaignFraudCheckItem])
def campaign_fraud_check(req: CampaignFraudCheckRequest) -> List[CampaignFraudCheckItem]:
    """
    Phát hiện gian lận chiến dịch gây quỹ bằng IsolationForest.
    - predict = -1 → bất thường → HIGH
    - predict = 1  → bình thường → LOW
    """
    ds_campaign = req.campaigns

    ma_tran_dac_trung = [core_fraud.build_campaign_fraud_features(chien_dich) for chien_dich in ds_campaign]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.2,
        random_state=42,
    )
    model.fit(ma_tran_dac_trung)

    du_doan = model.predict(ma_tran_dac_trung)

    ket_qua: List[CampaignFraudCheckItem] = []
    for chi_so, du_doan_muc in enumerate(du_doan):
        chien_dich = ds_campaign[chi_so]

        rule_high = (
            chien_dich.campaigns_per_user >= 4
            and chien_dich.donation_growth >= 200
            and chien_dich.self_donation_ratio >= 0.6
            and chien_dich.unique_donors <= 3
            and chien_dich.donation_frequency >= 10
        )

        muc_rui_ro: Literal["HIGH", "LOW"] = "HIGH" if (int(du_doan_muc) == -1 or rule_high) else "LOW"
        ket_qua.append(
            CampaignFraudCheckItem(
                campaign_id=chien_dich.campaign_id,
                risk=muc_rui_ro,
            )
        )

    return ket_qua
