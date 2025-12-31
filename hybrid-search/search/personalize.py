from dataclasses import dataclass
from typing import Dict, Set, Any

from .index import Index


@dataclass
class UserProfile:
    tags: Set[str]


def build_user_profile_from_tags(tags) -> UserProfile:
    # Accept list/set/tuple; normalize to lowercase strings
    clean = {str(t).strip().lower() for t in tags if str(t).strip()}
    return UserProfile(tags=clean)


def personalization_score(index: Index, doc_id: str, profile: UserProfile) -> float:
    """
    Tag-overlap personalization score in [0, 1]:
      p(d) = |U ∩ T_d| / |U|
    """
    if not profile.tags:
        return 0.0

    doc = index.docs_by_id.get(doc_id, {})
    doc_tags = {str(t).strip().lower() for t in doc.get("tags", []) if str(t).strip()}

    overlap = len(profile.tags & doc_tags)
    return overlap / max(1, len(profile.tags))


def personalization_explain(index: Index, doc_id: str, profile: UserProfile) -> Dict[str, Any]:
    """
    Small helper for UI/explanations.
    """
    doc = index.docs_by_id.get(doc_id, {})
    doc_tags = {str(t).strip().lower() for t in doc.get("tags", []) if str(t).strip()}
    matched = sorted(profile.tags & doc_tags)

    return {
        "matched_tags": matched,
        "doc_tags": sorted(doc_tags),
        "user_tags": sorted(profile.tags),
    }
