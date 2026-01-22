import json
from pathlib import Path
from typing import Any, Dict, List

# Reuse functions from core by importing it
from src import event_horizon_core as core  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
OUTPUTS_DIR = ROOT / "outputs"

ONTOLOGY_PATH = DATA_DIR / "trade_ontology.json"
UNIVERSE_PATH = DATA_DIR / "company_universe.json"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    ontology = core.load_json(ONTOLOGY_PATH)
    universe = core.load_json(UNIVERSE_PATH)

    event_files = sorted(EVENTS_DIR.glob("*.json"))
    if not event_files:
        raise SystemExit(f"No events found in {EVENTS_DIR}")

    feed: List[Dict[str, Any]] = []

    for ev_path in event_files:
        event = core.load_json(ev_path)
        conf, components = core.compute_confidence(ontology, event)
        ranked = core.rank_exposures(ontology, event, universe)
        graph = core.build_impact_graph(event, ontology)
        alert_text = core.format_alert(event, ranked, conf, components)

        event_id = event.get("event_id", ev_path.stem)

        # Write per-event artifacts
        (OUTPUTS_DIR / "alerts").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "graphs").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "rankings").mkdir(parents=True, exist_ok=True)

        (OUTPUTS_DIR / "alerts" / f"{event_id}.txt").write_text(alert_text, encoding="utf-8")
        write_json(OUTPUTS_DIR / "graphs" / f"{event_id}_graph.json", graph)
        write_json(OUTPUTS_DIR / "rankings" / f"{event_id}_ranked.json", ranked)

        # Build feed summary row
        winners = [r for r in ranked if r["direction"] == "BENEFIT"][:3]
        losers = [r for r in ranked if r["direction"] == "HARM"][:3]

        feed.append({
            "event_id": event_id,
            "title": event.get("title"),
            "type": event.get("event_type"),
            "stage": event.get("lifecycle_stage"),
            "confidence": conf,
            "top_beneficiaries": [f"{w['ticker']} ({w['exposure_score']})" for w in winners],
            "top_adversely_impacted": [f"{l['ticker']} ({l['exposure_score']})" for l in losers]
        })

    write_json(OUTPUTS_DIR / "feed.json", feed)

    print(f"Generated {len(feed)} alerts.")
    print(f"- outputs/feed.json")
    print(f"- outputs/alerts/*.txt")
    print(f"- outputs/graphs/*.json")
    print(f"- outputs/rankings/*.json")


if __name__ == "__main__":
    main()

