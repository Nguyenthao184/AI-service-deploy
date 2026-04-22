from __future__ import annotations

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Tính khoảng cách haversine (km) giữa hai tọa độ.
    """
    ban_kinh_trai_dat_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    do_lech_lat = lat2_rad - lat1_rad
    do_lech_lon = lon2_rad - lon1_rad

    gia_tri_a = np.sin(do_lech_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(do_lech_lon / 2.0) ** 2
    gia_tri_c = 2.0 * np.arctan2(np.sqrt(gia_tri_a), np.sqrt(1.0 - gia_tri_a))

    return float(ban_kinh_trai_dat_km * gia_tri_c)


def score_location_km(distance_km: float) -> float:
    # Giảm mượt theo khoảng cách: 3 * exp(-d/5)
    return float(3.0 * np.exp(-max(0.0, distance_km) / 5.0))


def score_time_days(delta_days: float) -> float:
    if delta_days < 1:
        return 2.0
    if delta_days < 3:
        return 1.0
    return 0.0

