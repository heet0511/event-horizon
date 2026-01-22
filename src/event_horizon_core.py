import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

ONTOLOGY_PATH = DATA_DIR / "trade_ontology.json"
UNIVERSE_PATH = DATA_DIR / "company_universe.json"
EVENT_PATH = DATA_DIR / "sample_event.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    return (s or "").lower().strip()


def keyword_hits(text: str, keywords: List[str]) -> int:
    t = normalize_text(text)
    hits = 0
    for kw in keywords:
        k = normalize_text(kw)
        if k and k in t:
            hits += 1
    return hits


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def stage_weight(ontology: Dict[str, Any], stage: str) -> float:
    weights = ontology.get("confidence_model", {}).get("stage_weights", {})
    return float(weights.get(stage, 0.50))


def compute_confidence(
    ontology: Dict[str, Any],
    event: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    v1 confidence = weighted blend of:
      - source_reliability: avg evidence reliability (0..1)
      - multi_source_corroboration: grows with number of independent sources (0..1)
      - text_clarity: heuristic based on presence of structured trade fields + keywords
      - stage_weight: from ontology stage weights (0..1)
      - historical_analog_strength: placeholder (0.55 in v1)
    """
    cm = ontology.get("confidence_model", {}).get("components", {})
    w_source = float(cm.get("source_reliability", {}).get("weight", 0.30))
    w_multi = float(cm.get("multi_source_corroboration", {}).get("weight", 0.20))
    w_clarity = float(cm.get("text_clarity", {}).get("weight", 0.15))
    w_stage = float(cm.get("stage_weight", {}).get("weight", 0.20))
    w_analog = float(cm.get("historical_analog_strength", {}).get("weight", 0.15))

    evidence = event.get("evidence", []) or []
    if evidence:
        rels = [float(e.get("reliability", 0.6)) for e in evidence]
        source_rel = clamp(sum(rels) / len(rels))
    else:
        source_rel = 0.55

    # corroboration: 1 source ~0.35, 2 ~0.55, 3 ~0.70, 4+ ~0.80+
    n_src = len(evidence)
    multi = clamp(1 - math.exp(-0.6 * n_src))  # 0, .45, .70, .83...
    # shift a bit so 1 source isn't too high
    multi = clamp(0.15 + 0.85 * multi)

    # clarity: do we have trade fields + product keywords?
    trade = event.get("trade", {}) or {}
    title = event.get("title", "") or ""
    summary = event.get("summary", "") or ""
    text = f"{title}\n{summary}\n{json.dumps(trade)}"
    clarity = 0.45
    if trade.get("measure_type"):
        clarity += 0.20
    products = trade.get("products", []) or []
    if products:
        clarity += 0.15
    if keyword_hits(text, ["license", "export", "entity list", "end-use", "ban", "control"]) > 0:
        clarity += 0.15
    clarity = clamp(clarity)

    stage = event.get("lifecycle_stage", "Draft")
    stg = clamp(stage_weight(ontology, stage))

    analog = 0.55  # v1 placeholder

    overall = (
        w_source * source_rel +
        w_multi * multi +
        w_clarity * clarity +
        w_stage * stg +
        w_analog * analog
    )
    overall = clamp(overall)

    components = {
        "source_reliability": round(source_rel, 3),
        "multi_source_corroboration": round(multi, 3),
        "text_clarity": round(clarity, 3),
        "stage_weight": round(stg, 3),
        "historical_analog_strength": round(analog, 3)
    }
    return round(overall, 3), components


def build_product_keyword_bank(ontology: Dict[str, Any]) -> List[str]:
    cats = ontology.get("trade_objects", {}).get("product_categories", []) or []
    bank: List[str] = []
    for c in cats:
        bank.extend(c.get("keywords", []) or [])
    # unique while preserving order
    seen = set()
    out = []
    for k in bank:
        kk = normalize_text(k)
        if kk and kk not in seen:
            seen.add(kk)
            out.append(k)
    return out


def compute_company_relevance(event: Dict[str, Any], company: Dict[str, Any], ontology: Dict[str, Any]) -> float:
    """
    v1 relevance heuristic:
      - keyword overlap between event text and company keywords
      - sector overlap between event-inferred sectors and company sectors
      - tag boosts for export-sensitive / equipment / EDA
    """
    title = event.get("title", "") or ""
    summary = event.get("summary", "") or ""
    trade = event.get("trade", {}) or {}
    products = trade.get("products", []) or []
    text = f"{title}\n{summary}\n{' '.join(products)}"

    c_keywords = company.get("keywords", []) or []
    hit_count = keyword_hits(text, c_keywords)

    # normalize hits to 0..1
    kw_score = clamp(hit_count / 4.0)  # 4+ hits ~= strong

    # infer sectors from product categories in ontology
    inferred_sectors: List[str] = []
    pcats = ontology.get("trade_objects", {}).get("product_categories", []) or []
    text2 = normalize_text(text)
    for pc in pcats:
        for kw in (pc.get("keywords", []) or []):
            if normalize_text(kw) in text2:
                inferred_sectors.extend(pc.get("sectors", []) or [])
                break

    company_sectors = set(company.get("sectors", []) or [])
    if inferred_sectors:
        overlap = len(company_sectors.intersection(set(inferred_sectors)))
        sector_score = clamp(overlap / 2.0)  # 2 overlaps ~= strong
    else:
        sector_score = 0.35  # unknown → mild baseline

    tag_boost = 0.0
    tags = set(company.get("tags", []) or [])
    if "export-sensitive" in tags:
        tag_boost += 0.10
    if "wafer-fab-equipment" in tags or "lithography" in tags:
        tag_boost += 0.08
    if "eda" in tags:
        tag_boost += 0.08

    relevance = clamp(0.55 * kw_score + 0.35 * sector_score + tag_boost)
    return relevance


def infer_direction(event: Dict[str, Any], company: Dict[str, Any]) -> str:
    """
    v1 direction heuristic:
      - Export controls/sanctions: exporters often harmed; domestic substitutes may benefit
      - Retaliation signals: uncertainty typically harms exporters; substitutes may benefit
      - Customs enforcement: shipping/audit friction harms exporters; keep others MIXED for credibility
    """
    etype = (event.get("event_type") or "").upper()
    tags = set(company.get("tags", []) or [])

    if etype in {"EXPORT_CONTROL", "SANCTIONS_UPDATE"}:
        if "domestic-substitute" in tags or "industrial-policy-beneficiary" in tags:
            return "BENEFIT"
        if "export-sensitive" in tags:
            return "HARM"
        if "supply-chain-central" in tags or "critical-tooling" in tags:
            return "MIXED"
        return "MIXED"

    if etype == "RETALIATION_SIGNAL":
        if "export-sensitive" in tags:
            return "HARM"
        if "domestic-substitute" in tags or "industrial-policy-beneficiary" in tags:
            return "BENEFIT"
        return "MIXED"

    if etype == "CUSTOMS_ENFORCEMENT":
        if "export-sensitive" in tags:
            return "HARM"
        return "MIXED"

    return "MIXED"


def compute_magnitude(company: Dict[str, Any]) -> float:
    """
    v1 magnitude: tag-based heuristic (0..1).
    """
    tags = set(company.get("tags", []) or [])
    mag = 0.55  # slightly higher baseline for demo clarity

    if "export-sensitive" in tags:
        mag += 0.20
    if "critical-tooling" in tags or "supply-chain-central" in tags:
        mag += 0.15
    if "data-center" in tags:
        mag += 0.10

    # NEW: policy beneficiary boost
    if "domestic-substitute" in tags:
        mag += 0.18
    if "industrial-policy-beneficiary" in tags:
        mag += 0.12

    return clamp(mag)


def compute_immediacy(event: Dict[str, Any]) -> float:
    """
    v1 immediacy: based on stage + presence of effective date.
    """
    stage = event.get("lifecycle_stage", "Draft")
    base = {
        "Signal": 0.45,
        "Draft": 0.50,
        "CredibleDraft": 0.62,
        "Adopted": 0.75,
        "Enforced": 0.85
    }.get(stage, 0.55)

    dates = event.get("dates", {}) or {}
    if dates.get("effective_date") or (event.get("trade", {}) or {}).get("effective_window", {}).get("start"):
        base += 0.10

    return clamp(base)


def rank_exposures(
    ontology: Dict[str, Any],
    event: Dict[str, Any],
    universe: Dict[str, Any]
) -> List[Dict[str, Any]]:
    overall_conf, _ = compute_confidence(ontology, event)
    immed = compute_immediacy(event)

    ranked = []
    for c in universe.get("companies", []) or []:
        relevance = compute_company_relevance(event, c, ontology)
        magnitude = compute_magnitude(c)
        direction = infer_direction(event, c)

        # v1 exposure score (0..100)
        raw = relevance * magnitude * immed * overall_conf
        score = int(round(100 * raw))

        drivers = []
        if relevance >= 0.6:
            drivers.append("High relevance to event keywords/sectors")
        if "export-sensitive" in (c.get("tags", []) or []):
            drivers.append("Export-sensitive profile")
        if "domestic-substitute" in (c.get("tags", []) or []):
            drivers.append("Potential domestic substitution beneficiary")
        if "industrial-policy-beneficiary" in (c.get("tags", []) or []):
            drivers.append("Industrial policy tailwind archetype")

        ranked.append({
            "ticker": c.get("ticker"),
            "company_name": c.get("name"),
            "direction": direction,
            "exposure_score": score,
            "confidence": overall_conf,
            "drivers": drivers[:3]
        })

    # Sort: strongest exposures first, then keep BENEFIT/HARM groups for display later
    ranked.sort(key=lambda x: x["exposure_score"], reverse=True)
    return ranked


def build_impact_graph(event: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a simple event->mechanism->sector graph from ontology template keywords.
    v1 uses a generic export-control chain.
    """
    etype = (event.get("event_type") or "").upper()
    title = event.get("title", "") or ""
    trade = event.get("trade", {}) or {}
    products = ", ".join(trade.get("products", []) or []) or "semiconductor-related items"

    if etype in {"EXPORT_CONTROL", "SANCTIONS_UPDATE"}:
        chain = [
            "Supply restriction / licensing friction",
            "Lead times increase + uncertainty",
            "Downstream production constraints",
            "Capex + sourcing shifts"
        ]
        mechanism_label = f"{etype}: licensing friction on {products}"
    else:
        chain = ["Policy change", "Cost/constraint shift", "Sector impact", "Second-order effects"]
        mechanism_label = f"{etype}: policy mechanism"

    nodes = [{"id": "E1", "type": "EVENT", "label": title or "Trade policy event"},
             {"id": "M1", "type": "MECHANISM", "label": mechanism_label}]
    edges = [{"from": "E1", "to": "M1", "relation": "causes"}]

    # Add chain nodes
    prev = "M1"
    for i, step in enumerate(chain, start=2):
        nid = f"N{i}"
        nodes.append({"id": nid, "type": "EFFECT", "label": step})
        edges.append({"from": prev, "to": nid, "relation": "leads_to"})
        prev = nid

    return {"graph_id": f"graph_{event.get('event_id', 'event')}", "nodes": nodes, "edges": edges}


def format_alert(event: Dict[str, Any], ranked: List[Dict[str, Any]], confidence: float, components: Dict[str, float]) -> str:
    title = event.get("title", "Event detected")
    etype = event.get("event_type", "UNKNOWN")
    stage = event.get("lifecycle_stage", "Draft")

    # separate winners/losers
    winners = [r for r in ranked if r["direction"] == "BENEFIT" and r["exposure_score"] >= 10][:5]
    losers = [r for r in ranked if r["direction"] == "HARM" and r["exposure_score"] >= 10][:5]

    # If no clear direction, fallback to top exposures
    if not winners and not losers:
        top = ranked[:5]
        winners = top[:3]
        losers = top[3:5]

    lines = []
    lines.append("=== EVENT HORIZON ALERT (v1) ===")
    lines.append(f"Title: {title}")
    lines.append(f"Type/Stage: {etype} / {stage}")
    lines.append("")
    lines.append("Top likely beneficiaries:")
    for w in winners:
        lines.append(f"  + {w['ticker']} — {w['company_name']} (score {w['exposure_score']})")
    lines.append("")
    lines.append("Top likely adversely impacted:")
    for l in losers:
        lines.append(f"  - {l['ticker']} — {l['company_name']} (score {l['exposure_score']})")
    lines.append("")
    lines.append(f"Confidence: {confidence} (breakdown: {components})")
    lines.append("")
    lines.append("Interpretation: This is a causal, event-driven exposure ranking. Not financial advice.")
    return "\n".join(lines)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ontology = load_json(ONTOLOGY_PATH)
    universe = load_json(UNIVERSE_PATH)
    event = load_json(EVENT_PATH)

    conf, components = compute_confidence(ontology, event)
    ranked = rank_exposures(ontology, event, universe)
    graph = build_impact_graph(event, ontology)

    # Write outputs
    (OUTPUTS_DIR / "ranked_impacts.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "impact_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    # Print alert to console
    alert_text = format_alert(event, ranked, conf, components)
    print(alert_text)


if __name__ == "__main__":
    main()

