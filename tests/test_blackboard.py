import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.blackboard.blackboard import Blackboard, ProvenanceError
from src.cdss_core.distributions import NormalPosterior
from src.cdss_core.fusion import precision_weighted_fusion
from src.cdss_core.log_lr import LogLikelihoodRatio

def test_duplicate_provenance_raises():
    bb = Blackboard()
    bb.update_latent('GFR', NormalPosterior(45, 5), 'lab_001', 'M3')
    with pytest.raises(ProvenanceError):
        bb.update_latent('GFR', NormalPosterior(46, 4), 'lab_001', 'M3')

def test_fusion_order_invariant():
    a = NormalPosterior(45, 5)
    b = NormalPosterior(50, 3)
    f1 = precision_weighted_fusion([a, b])
    f2 = precision_weighted_fusion([b, a])
    assert abs(f1.mu - f2.mu) < 1e-10

def test_fusion_reduces_uncertainty():
    a = NormalPosterior(45, 5)
    b = NormalPosterior(50, 3)
    fused = precision_weighted_fusion([a, b])
    assert fused.sigma < a.sigma and fused.sigma < b.sigma

def test_harm_lr_accumulation():
    bb = Blackboard()
    bb.add_harm_evidence('QT', LogLikelihoodRatio(1.5, 'm1_qt', 'M1'))
    bb.add_harm_evidence('QT', LogLikelihoodRatio(0.8, 'm3_k', 'M3'))
    assert abs(bb.get_fused_log_lr('QT') - 2.3) < 1e-10
