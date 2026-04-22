from __future__ import annotations

from typing import List, Any


def build_fraud_features(user: Any) -> List[float]:
    """
    Chuyển dữ liệu user thành vector 5 đặc trưng:
    [
        posts_per_day,
        content_similarity,
        donation_growth,
        same_ip_accounts,
        activity_score
    ]
    """
    return [
        float(user.posts_per_day),
        float(user.content_similarity),
        float(user.donation_growth),
        float(user.same_ip_accounts),
        float(user.activity_score),
    ]


def build_campaign_fraud_features(campaign: Any) -> List[float]:
    """
    Chuyển dữ liệu campaign thành vector 5 đặc trưng:
    [
        campaigns_per_user,
        donation_growth,
        self_donation_ratio,
        unique_donors,
        donation_frequency
    ]
    """
    return [
        float(campaign.campaigns_per_user),
        float(campaign.donation_growth),
        float(campaign.self_donation_ratio),
        float(campaign.unique_donors),
        float(campaign.donation_frequency),
    ]

