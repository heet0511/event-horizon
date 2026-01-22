import json
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
import graphviz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FEED_PATH = OUT / "feed.json"
REPORT_PATH = OUT / "REPORT.md"
ALERTS_DIR = OUT / "alerts"
GRAPHS_DIR = OUT / "graphs"
RANKINGS_DIR = OUT / "rankings"

st.set_page_config(page_title="Event Horizon (v1)", layout="wide")

st.title("Event Horizon (v1)")
st.caption(
    "Policy → Market Impact Intelligence (Demo). Explainable exposure ranking — not financial advice."
)


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def graph_to_dot(graph: dict) -> str:
    lines = [
        "digraph G {",
        'rankdir="LR";',
        "nodesep=0.35;",
        "ranksep=0.55;",
        'node [shape=box, style="rounded", fontsize=12, margin="0.18,0.10"];',
    ]

    for n in graph.get("nodes", []):
        # Wrap long labels across multiple lines for readability
        label = (n.get("label") or "").strip()
        label_wrapped = "\n".join(textwrap.wrap(label, width=28))

        # Cap to ~4 lines so nodes don't get too tall
        if label_wrapped.count("\n") > 3:
            parts = label_wrapped.split("\n")[:4]
            parts[-1] = (parts[-1][:25] + "...") if len(parts[-1]) > 25 else (parts[-1] + "…")
            label_wrapped = "\n".join(parts)

        label_wrapped = label_wrapped.replace('"', '\\"')
        lines.append(f'"{n["id"]}" [label="{label_wrapped}"];')

    for e in graph.get("edges", []):
        lines.append(f'"{e["from"]}" -> "{e["to"]}";')

    lines.append("}")
    return "\n".join(lines)


# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")

    if st.button("Run pipeline (batch + report)", type="primary"):
        st.write("Running…")
        code1, out1 = run_cmd(["python3", "-m", "src.run_batch"])
        code2, out2 = run_cmd(["python3", "-m", "src.generate_report"])
        st.text_area("Batch output", out1, height=140)
        st.text_area("Report output", out2, height=120)
        if code1 == 0 and code2 == 0:
            st.success("Pipeline complete.")
        else:
            st.error("Pipeline failed. Scroll output for details.")

        st.divider()

    st.caption("Outputs expected: feed.json + REPORT.md")


# ---- Main: Load feed ----
if not FEED_PATH.exists():
    st.warning(
        "No outputs/feed.json yet. Click **Run pipeline (batch + report)** in the sidebar."
    )
    st.stop()

feed = load_json(FEED_PATH)

# --- KPI cards ---
df_all = pd.DataFrame(feed)

c1, c2, c3 = st.columns(3)
c1.metric("Events", len(df_all))

avg_conf = float(df_all["confidence"].mean()) if not df_all.empty else 0.0
c2.metric("Avg confidence", f"{avg_conf:.3f}")

top_row = df_all.sort_values("confidence", ascending=False).head(1)
top_event = top_row["event_id"].iloc[0] if not top_row.empty else "—"
top_conf = float(top_row["confidence"].iloc[0]) if not top_row.empty else 0.0
c3.metric("Top confidence", f"{top_conf:.3f}", help=f"Event: {top_event}")

st.divider()

# Feed table
st.subheader("Feed")

# --- Filters ---
f1, f2, f3 = st.columns([1, 1, 2])
types = ["All"] + sorted(df_all["type"].dropna().unique().tolist())
stages = ["All"] + sorted(df_all["stage"].dropna().unique().tolist())

type_choice = f1.selectbox("Type", types, index=0)
stage_choice = f2.selectbox("Stage", stages, index=0)
query = f3.text_input("Search title", "")

df_show = df_all.copy()
if type_choice != "All":
    df_show = df_show[df_show["type"] == type_choice]
if stage_choice != "All":
    df_show = df_show[df_show["stage"] == stage_choice]
if query.strip():
    df_show = df_show[df_show["title"].str.contains(query.strip(), case=False, na=False)]

# Show table
if not df_show.empty:
    st.dataframe(
        df_show[["event_id", "type", "stage", "confidence", "title"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No events match your filters.")

# Pick event
event_ids = df_show["event_id"].tolist() if not df_show.empty else df_all["event_id"].tolist()
selected = st.selectbox("Select an event", event_ids, index=0)

# ---- Event detail ----
st.subheader(f"Event Detail: {selected}")

col1, col2 = st.columns([1, 1])

# Alert text
alert_path = ALERTS_DIR / f"{selected}.txt"
with col1:
    st.markdown("### Alert")
    if alert_path.exists():
        st.code(alert_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Alert file missing. Run the pipeline.")

# Rankings table
ranking_path = RANKINGS_DIR / f"{selected}_ranked.json"
with col2:
    st.markdown("### Ranked Impacts")
    if ranking_path.exists():
        ranked = load_json(ranking_path)
        rdf = pd.DataFrame(ranked)
        if not rdf.empty:
            st.dataframe(
                rdf[
                    ["ticker", "company_name", "direction", "exposure_score", "confidence"]
                ].head(15),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Ranking file missing. Run the pipeline.")

# Impact Graph (visual)
graph_path = GRAPHS_DIR / f"{selected}_graph.json"
st.markdown("### Impact Graph")
st.caption("EVENT → MECHANISM → SUPPLY → EFFECT → COMPANY EXPOSURE")

if graph_path.exists():
    g = load_json(graph_path)
    dot = graph_to_dot(g)
    st.graphviz_chart(dot)
    with st.expander("Show graph JSON"):
        st.json(g)
else:
    st.info("Graph file missing. Run the pipeline.")

st.divider()

# Report (render)
st.subheader("Compiled Report")
if REPORT_PATH.exists():
    report_md = REPORT_PATH.read_text(encoding="utf-8")
    st.markdown(report_md)
    st.download_button(
        "Download REPORT.md",
        data=report_md.encode("utf-8"),
        file_name="REPORT.md",
        mime="text/markdown",
    )
else:
    st.info("Report missing. Run the pipeline.")
