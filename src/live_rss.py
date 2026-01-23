# src/live_rss.py
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

import feedparser


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_id(title: str, link: str) -> str:
    h = hashlib.sha1((title.strip() + "|" + link.strip()).encode("utf-8")).hexdigest()[:10]
    return f"eh_rss_{h}"


def fetch_rss_events(rss_url: str, max_events: int = 6) -> List[Dict[str, Any]]:
    f = feedparser.parse(rss_url)
    entries = getattr(f, "entries", []) or []

    out: List[Dict[str, Any]] = []
    for e in entries[:max_events]:
        title = str(getattr(e, "title", "") or "").strip()
        link = str(getattr(e, "link", "") or "").strip()
        summary = str(getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
        published = str(getattr(e, "published", "") or getattr(e, "updated", "") or "").strip()

        event_id = _stable_id(title or "untitled", link or "nolink")

        # IMPORTANT: keep these shapes compatible with event_horizon_core:
        # - jurisdiction/category/trade/dates are dicts
        # - tags is list[str]
        # - evidence is LIST[dict]
        event: Dict[str, Any] = {
            "event_id": event_id,
            "created_at": _now_utc_iso(),
            "event_type": "POLICY_RSS",
            "lifecycle_stage": "Signal",
            "category": {"name": "RSS_INGEST"},
            "jurisdiction": {"scope": "GLOBAL"},
            "title": title or "(no title)",
            "summary": summary or "(no summary)",
            "dates": {"published": published},
            "trade": {},
            "evidence": [
                {
                    "source_type": "rss",
                    "feed_url": rss_url,
                    "url": link,
                    "publisher": "BBC" if "bbc" in rss_url else "RSS",
                    "published_at": published,
                    "title": title,
                }
            ],
            "tags": ["rss", "live"],
        }

        out.append(event)

    return out
