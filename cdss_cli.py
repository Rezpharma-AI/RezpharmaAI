import sys, argparse
sys.path.insert(0, '.')

def cmd_analyze(args):
    from src.m1_ddi_adr.rule_engine import DDIRuleEngine
    drugs = [d.strip() for d in args.drugs.split(',')]
    engine = DDIRuleEngine()
    hits = engine.check_interactions(drugs)
    print(f'Analyzing {len(drugs)} drugs...')
    for h in hits[:5]:
        print(f'  [{h["severity"].upper()}] {h["perpetrator"]} + {h["victim"]}')

def cmd_gfr(args):
    from src.m3_analysis_lab.kinetic_gfr import KineticGFR
    vals = [float(x) for x in args.creatinine.split(',')]
    series = [(i*12, v) for i, v in enumerate(vals)]
    kgfr = KineticGFR()
    post = kgfr.estimate(series, 1.1)
    print(f'Kinetic GFR: {post.mean():.1f} mL/min')

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    p1 = sub.add_parser('analyze'); p1.add_argument('--drugs', required=True); p1.set_defaults(func=cmd_analyze)
    p2 = sub.add_parser('gfr'); p2.add_argument('--creatinine', required=True); p2.set_defaults(func=cmd_gfr)
    args = p.parse_args()
    if args.cmd: args.func(args)
    else: p.print_help()

if __name__ == '__main__':
    main()
