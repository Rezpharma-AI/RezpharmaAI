import os

def w(f, t):
    os.makedirs(os.path.dirname(f) or '.', exist_ok=True)
    open(f, 'w', encoding='utf-8').write(t)
    print(f"Created {f}")

w("README.md", "# RezpharmaCDSS\n\n> Multi-Module Polypharmacy CDSS\n\n## Stats\n- Drugs: 4,566\n- DDIs: 2,855,310\n- ADRs: 5,701\n\n## Run\n- API: uvicorn api_server:app --reload\n- UI: streamlit run app_v2.py\n- CLI: python cdss_cli.py analyze --drugs \"Warfarin, Aspirin\"\n")

w("docs/API.md", "# API Reference\n\nBase URL: http://127.0.0.1:8000\n\n## Endpoints\n- GET /\n- POST /api/v1/m1/analyze\n- GET /api/v1/advisor/recommendations\n\nSwagger UI: http://127.0.0.1:8000/docs\n")

w("docs/MODULES.md", "# Modules\n\n## M1\nfrom src.m1_ddi_adr.rule_engine import DDIRuleEngine\nengine = DDIRuleEngine()\nhits = engine.check_interactions(['Warfarin', 'Aspirin'])\n\n## M3\nfrom src.m3_analysis_lab.kinetic_gfr import KineticGFR\nkgfr = KineticGFR()\npost = kgfr.estimate([(0,0.9), (12,1.5)], 1.1)\n")

w("docs/ROADMAP.md", "# Roadmap\n\n- [x] Core Modules\n- [x] 2.85M DDI Knowledge Base\n- [x] FastAPI Microservices\n- [x] Event-Driven Redis Bus\n- [ ] Docker Deployment\n")

cli = "import sys, argparse\nsys.path.insert(0, '.')\n\ndef cmd_analyze(args):\n    from src.m1_ddi_adr.rule_engine import DDIRuleEngine\n    drugs = [d.strip() for d in args.drugs.split(',')]\n    engine = DDIRuleEngine()\n    hits = engine.check_interactions(drugs)\n    print(f'Analyzing {len(drugs)} drugs...')\n    for h in hits[:5]:\n        print(f'  [{h[\"severity\"].upper()}] {h[\"perpetrator\"]} + {h[\"victim\"]}')\n\ndef cmd_gfr(args):\n    from src.m3_analysis_lab.kinetic_gfr import KineticGFR\n    vals = [float(x) for x in args.creatinine.split(',')]\n    series = [(i*12, v) for i, v in enumerate(vals)]\n    kgfr = KineticGFR()\n    post = kgfr.estimate(series, 1.1)\n    print(f'Kinetic GFR: {post.mean():.1f} mL/min')\n\ndef main():\n    p = argparse.ArgumentParser()\n    sub = p.add_subparsers(dest='cmd')\n    p1 = sub.add_parser('analyze'); p1.add_argument('--drugs', required=True); p1.set_defaults(func=cmd_analyze)\n    p2 = sub.add_parser('gfr'); p2.add_argument('--creatinine', required=True); p2.set_defaults(func=cmd_gfr)\n    args = p.parse_args()\n    if args.cmd: args.func(args)\n    else: p.print_help()\n\nif __name__ == '__main__':\n    main()\n"
w("cdss_cli.py", cli)

print("Done! Documentation and CLI created successfully.")