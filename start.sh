#!/usr/bin/env bash
set -e

# (Optional) generate outputs on boot so the site isn't empty
python3 -m src.run_batch --source rss --rss-url "https://feeds.bbci.co.uk/news/business/rss.xml" --max-events 8 || true
python3 -m src.generate_report || true

# Start Streamlit on Render's assigned port
streamlit run src/app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0

