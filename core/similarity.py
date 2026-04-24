
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import SEMANTIC_MODEL_NAME
import logging
logger = logging.getLogger(__name__)
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@lru_cache(maxsize=1)
def get_semantic_model() -> Optional["SentenceTransformer"]:
    """
    Tải mô hình ngữ nghĩa một lần mỗi tiến trình.
    Hỗ trợ (mặc định keepitreal/vietnamese-sbert; đổi qua SEMANTIC_MODEL_NAME):
    - keepitreal/vietnamese-sbert
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    """
    
    if SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        return model
    except Exception as e:
        logger.warning(f"Failed to load model {SEMANTIC_MODEL_NAME}: {e}")
        return None


def semantic_similarity_scores(target_text: str, other_texts: List[str]) -> np.ndarray:
    """
    Trả về điểm tương đồng cosine giữa `target_text` và từng chuỗi trong `other_texts`.
    """
    model = get_semantic_model()
   
    if model is None:
        logger.warning("Semantic model not loaded, fallback to TF-IDF")
        texts = [target_text] + other_texts
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=7000,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        target_vec = tfidf_matrix[0:1]
        cand_vecs = tfidf_matrix[1:]
        return cosine_similarity(target_vec, cand_vecs)[0]

    embeddings = model.encode(
        [target_text] + other_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    target_emb = embeddings[0]
    cand_embs = embeddings[1:]
    # normalize_embeddings=True → cosine xấp xỉ tích vô hướng
    return cand_embs @ target_emb


def lexical_similarity_scores(target_text: str, other_texts: List[str]) -> np.ndarray:
    texts = [target_text] + other_texts
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=7000,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    target_vec = tfidf_matrix[0:1]
    cand_vecs = tfidf_matrix[1:]
    return cosine_similarity(target_vec, cand_vecs)[0]


def semantic_similarity_single_target(target_text: str, other_texts: List[str]) -> np.ndarray:
    """
    Tương đồng ưu tiên ngữ nghĩa cho ghép văn bản chung.
    Dùng embedding mô hình làm tín hiệu chính, từ vựng làm dự phòng để ổn định.
    """
    semantic_sims = semantic_similarity_scores(target_text, other_texts)
    lexical_sims = lexical_similarity_scores(target_text, other_texts)
    return np.clip((0.9 * semantic_sims) + (0.1 * lexical_sims), 0.0, 1.0)

