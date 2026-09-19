# Procfile — for Render / Heroku / Railway free-tier hosting.
# $PORT is injected automatically by the hosting platform.
# Streamlit is configured headless (no browser auto-open) for server environments.
web: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
