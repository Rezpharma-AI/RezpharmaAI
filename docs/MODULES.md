# Modules

## M1
from src.m1_ddi_adr.rule_engine import DDIRuleEngine
engine = DDIRuleEngine()
hits = engine.check_interactions(['Warfarin', 'Aspirin'])

## M3
from src.m3_analysis_lab.kinetic_gfr import KineticGFR
kgfr = KineticGFR()
post = kgfr.estimate([(0,0.9), (12,1.5)], 1.1)
