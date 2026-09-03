# RezpharmaCDSS

> Multi-Module Polypharmacy CDSS

## Stats
- Drugs: 4,566
- DDIs: 2,855,310
- ADRs: 5,701

## Run
- API: uvicorn api_server:app --reload
- UI: streamlit run app_v2.py
- CLI: python cdss_cli.py analyze --drugs "Warfarin, Aspirin"
