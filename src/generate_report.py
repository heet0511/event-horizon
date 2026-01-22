import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ALERTS = OUT / "alerts"
FEED = OUT / "feed.json"

def main():
    feed = json.loads(FEED.read_text(encoding="utf-8"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md = []
    md.append("# Event Horizon — Demo Feed (v1)\n")
    md.append(f"_Generated: {ts}_\n")
    md.append("**Module:** Regulation & Policy → Trade / Export Controls (Semiconductors)\n")
    md.append("**What this is:** Explainable event-driven exposure ranking (not financial advice).\n")
    md.append("---\n")

    md.append("## How it works (v1)\n")
    md.append("- Ingest policy signals (demo events for now)\n")
    md.append("- Normalize into a standard event schema\n")
    md.append("- Generate an impact chain (causal steps)\n")
    md.append("- Rank company exposure using: relevance × magnitude × immediacy × confidence\n")
    md.append("- Output: beneficiaries / adversely impacted + confidence breakdown\n")
    md.append("\n---\n")

    md.append("## Feed Summary\n")
    for item in feed:
        md.append(f"### {item['event_id']}: {item['title']}")
        md.append(f"- Type/Stage: **{item['type']} / {item['stage']}**")
        md.append(f"- Confidence: **{item['confidence']}**")
        md.append(f"- Beneficiaries: {', '.join(item['top_beneficiaries']) or '—'}")
        md.append(f"- Adversely impacted: {', '.join(item['top_adversely_impacted']) or '—'}")
        md.append("")

    md.append("---\n")
    md.append("## Full Alerts\n")

    for item in feed:
        alert_path = ALERTS / f"{item['event_id']}.txt"
        md.append(f"### {item['event_id']}\n")
        if alert_path.exists():
            md.append("```")
            md.append(alert_path.read_text(encoding="utf-8").rstrip())
            md.append("```")
        else:
            md.append("_Missing alert file._")
        md.append("")

    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote outputs/REPORT.md")

if __name__ == "__main__":
    main()

