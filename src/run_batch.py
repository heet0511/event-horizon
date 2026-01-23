# src/run_batch.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src import event_horizon_core as core  # type: ignore
from src.live_rss import fetch_rss_events

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

FEED_PATH = OUT / "feed.json"
ALERTS_DIR = OUT / "alerts"
GRAPHS_DIR = OUT / "graphs"
RANKINGS_DIR = OUT / "rankings"


# ----------------------------
# IO helpers
# ----------------------------
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Loaders
# ----------------------------
def load_ontology() -> Dict[str, Any]:
    p = DATA / "trade_ontology.json"
    if not p.exists():
        raise FileNotFoundError("Missing data/trade_ontology.json")
    obj = read_json(p)
    if not isinstance(obj, dict):
        raise ValueError("data/trade_ontology.json must be a JSON object (dict)")
    return obj


def load_company_universe() -> Dict[str, Any]:
    """
    Accepts:
      - dict with keys like {"universe_id","version","companies":[...]}
      - OR a plain list of companies
    Returns a dict with at least {"companies":[...]}.
    """
    p = DATA / "company_universe.json"
    if not p.exists():
        raise FileNotFoundError("Missing data/company_universe.json")
    obj = read_json(p)

    if isinstance(obj, list):
        return {"universe_id": "local", "version": "1", "companies": obj}

    if isinstance(obj, dict):
        companies = obj.get("companies")
        if isinstance(companies, list):
            return obj
        raise ValueError("data/company_universe.json dict must contain a 'companies' JSON list")

    raise ValueError("data/company_universe.json must be a JSON dict or list")


def load_demo_events() -> List[Dict[str, Any]]:
    events_dir = DATA / "events"
    if not events_dir.exists():
        raise FileNotFoundError("Missing data/events/ directory")

    events: List[Dict[str, Any]] = []
    for p in sorted(events_dir.glob("*.json")):
        obj = read_json(p)
        if not isinstance(obj, dict):
            raise ValueError(f"{p} must be a JSON object (dict) event")
        events.append(obj)
    return events


# ----------------------------
# Normalizers / guards
# ----------------------------
def coerce_event_schema(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make RSS events safe for event_horizon_core by coercing known fields
    into dicts/lists in the shapes the core expects.
    """

    def ensure_dict(key: str, wrap_key: str = "value") -> None:
        v = event.get(key)
        if v is None:
            event[key] = {}
        elif isinstance(v, dict):
            return
        elif isinstance(v, str):
            event[key] = {wrap_key: v}
        else:
            event[key] = {wrap_key: str(v)}

    def ensure_list(key: str) -> None:
        v = event.get(key)
        if v is None:
            event[key] = []
        elif isinstance(v, list):
            return
        elif isinstance(v, str):
            event[key] = [v]
        else:
            event[key] = [str(v)]

    # Core often expects dicts for these
    ensure_dict("jurisdiction", wrap_key="scope")
    ensure_dict("category", wrap_key="name")
    ensure_dict("trade", wrap_key="raw")
    ensure_dict("dates", wrap_key="raw")

    # Core expects list[str]
    ensure_list("tags")

    # ✅ IMPORTANT: evidence must be LIST[dict] (core iterates it)
    ev = event.get("evidence")
    if ev is None:
        event["evidence"] = []
    elif isinstance(ev, list):
        fixed = []
        for item in ev:
            if isinstance(item, dict):
                fixed.append(item)
            elif isinstance(item, str):
                fixed.append({"raw": item})
            else:
                fixed.append({"raw": str(item)})
        event["evidence"] = fixed
    elif isinstance(ev, dict):
        event["evidence"] = [ev]
    elif isinstance(ev, str):
        event["evidence"] = [{"raw": ev}]
    else:
        event["evidence"] = [{"raw": str(ev)}]

    # Preferred canonical keys
    if "event_type" not in event and "type" in event and isinstance(event["type"], str):
        event["event_type"] = event["type"]

    if "lifecycle_stage" not in event and "stage" in event and isinstance(event["stage"], str):
        event["lifecycle_stage"] = event["stage"]

    return event


def normalize_stage(event: Dict[str, Any]) -> str:
    s = event.get("lifecycle_stage") or event.get("stage")
    if isinstance(s, str) and s.strip():
        return s.strip()
    return "UNKNOWN"


def normalize_type(event: Dict[str, Any]) -> str:
    t = event.get("event_type") or event.get("type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "UNKNOWN"


def render_alert(
    *,
    event_id: str,
    title: str,
    etype: str,
    stage: str,
    summary: str,
    top_beneficiaries: List[str],
    top_adverse: List[str],
    confidence: float,
) -> str:
    lines = []
    lines.append("=== EVENT HORIZON ALERT (v1) ===")
    lines.append(f"Event: {event_id}")
    lines.append(f"Title: {title}")
    lines.append(f"Type/Stage: {etype} / {stage}")
    lines.append("")
    lines.append("Summary:")
    lines.append(summary.strip() or "—")
    lines.append("")
    lines.append("Top likely beneficiaries:")
    if top_beneficiaries:
        for x in top_beneficiaries:
            lines.append(f"+ {x}")
    else:
        lines.append("—")
    lines.append("")
    lines.append("Top likely adversely impacted:")
    if top_adverse:
        for x in top_adverse:
            lines.append(f"- {x}")
    else:
        lines.append("—")
    lines.append("")
    lines.append(f"Confidence: {confidence:.3f}")
    lines.append("")
    lines.append("Interpretation: Causal, event-driven exposure ranking (demo logic) — not financial advice.")
    return "\n".join(lines)


# ----------------------------
# Per-event pipeline
# ----------------------------
def process_one_event(event: Dict[str, Any], ontology: Dict[str, Any], universe: Dict[str, Any]) -> Dict[str, Any]:
    event = coerce_event_schema(event)

    event_id = str(event.get("event_id") or "unknown_event").strip()
    title = str(event.get("title") or "Untitled").strip()
    summary = str(event.get("summary") or "").strip()

    etype = normalize_type(event)
    stage = normalize_stage(event)

    graph = core.build_impact_graph(event=event, ontology=ontology)
    ranked: List[Dict[str, Any]] = core.rank_exposures(ontology=ontology, event=event, universe=universe)

    conf = 0.0
    if ranked and isinstance(ranked[0], dict) and isinstance(ranked[0].get("confidence"), (int, float)):
        conf = float(ranked[0]["confidence"])

    beneficiaries: List[str] = []
    adverse: List[str] = []

    for r in ranked[:20]:
        if not isinstance(r, dict):
            continue
        ticker = str(r.get("ticker") or "").strip()
        name = str(r.get("company_name") or "").strip()
        direction = str(r.get("direction") or "").upper()
        label = f"{ticker} — {name}".strip(" —")

        if direction == "BENEFIT" and len(beneficiaries) < 3:
            beneficiaries.append(label)
        if direction == "HARM" and len(adverse) < 3:
            adverse.append(label)

        if len(beneficiaries) >= 3 and len(adverse) >= 3:
            break

    write_json(GRAPHS_DIR / f"{event_id}_graph.json", graph)
    write_json(RANKINGS_DIR / f"{event_id}_ranked.json", ranked)
    write_text(
        ALERTS_DIR / f"{event_id}.txt",
        render_alert(
            event_id=event_id,
            title=title,
            etype=etype,
            stage=stage,
            summary=summary,
            top_beneficiaries=beneficiaries,
            top_adverse=adverse,
            confidence=conf,
        ),
    )

    return {
        "event_id": event_id,
        "title": title,
        "type": etype,
        "stage": stage,
        "confidence": conf,
        "top_beneficiaries": beneficiaries,
        "top_adversely_impacted": adverse,
    }


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event Horizon batch pipeline")
    p.add_argument("--source", choices=["demo", "rss"], default="demo")
    p.add_argument("--rss-url", default="", help="RSS/Atom feed URL (required for --source rss)")
    p.add_argument("--max-events", type=int, default=6, help="Max RSS items to ingest")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    ontology = load_ontology()
    universe = load_company_universe()

    if args.source == "demo":
        events = load_demo_events()
    else:
        if not args.rss_url.strip():
            raise ValueError("For --source rss you must pass --rss-url '<feed_url>'")
        events = fetch_rss_events(args.rss_url.strip(), max_events=args.max_events)

    feed_rows: List[Dict[str, Any]] = []
    failed = 0

    for ev in events:
        try:
            row = process_one_event(ev, ontology=ontology, universe=universe)
            feed_rows.append(row)
        except Exception as e:
            failed += 1
            eid = str(ev.get("event_id") or "unknown") if isinstance(ev, dict) else "unknown"
            print(f"[ERROR] Event {eid} failed: {e}")

    write_json(FEED_PATH, feed_rows)

    print(f"Generated {len(feed_rows)} alerts.")
    print(f"- {FEED_PATH}")
    print("- outputs/alerts/*.txt")
    print("- outputs/graphs/*.json")
    print("- outputs/rankings/*.json")

    if failed:
        print(f"[WARN] {failed} event(s) failed. See errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
