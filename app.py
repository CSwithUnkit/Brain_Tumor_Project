# Proxy app for Hugging Face Spaces deployment
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "dashboard/app.py"]
    sys.exit(stcli.main())
