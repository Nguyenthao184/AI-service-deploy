from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

from core import config


def get_osrm_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    cache: Dict[Tuple[float, float, float, float], float],
    call_count: List[int],
) -> Optional[float]:
    """
    Lấy khoảng cách đường đi từ OSRM (km theo đường bộ).
    Trả về None nếu gọi OSRM thất bại.
    """
    key = (round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6))
    if key in cache:
        return cache[key]

    if not config.OSRM_ENABLED:
        return None

    if call_count[0] >= config.OSRM_MAX_CALLS:
        return None

    try:
        call_count[0] += 1
        distance_km = _osrm_distance_km_cached(*key)
        if distance_km is None:
            return None
        cache[key] = distance_km
        return distance_km
    except Exception:
        return None


@lru_cache(maxsize=max(16, config.OSRM_GLOBAL_CACHE_SIZE))
def _osrm_distance_km_cached(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Bộ nhớ đệm toàn cục giữa các request (theo tiến trình).
    Mọi ngoại lệ trả về None (không thử lại).
    """
    if not config.OSRM_ENABLED:
        return None

    coordinates = f"{lon1},{lat1};{lon2},{lat2}"
    encoded_coordinates = quote(coordinates, safe=";,.-")
    url = f"{config.OSRM_BASE_URL}/route/v1/{config.OSRM_PROFILE}/{encoded_coordinates}?overview=false"

    try:
        req = Request(url, headers={"User-Agent": "smart-donate-37-ai/1.0"})
        with urlopen(req, timeout=config.OSRM_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if payload.get("code") != "Ok":
            return None

        routes = payload.get("routes") or []
        if not routes:
            return None

        distance_m = routes[0].get("distance")
        if distance_m is None:
            return None

        return float(distance_m) / 1000.0
    except Exception:
        return None
