import os
from pathlib import Path

# Fix test_blackboard.py
Path("tests/test_blackboard.py").write_text(
    "import pytest, sys, os\n"
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n"
    "from src.blackboard.blackboard import Blackboard, ProvenanceError\n"
    "from src.cdss_core.distributions import NormalPosterior\n"
    "from src.cdss_core.fusion import precision_weighted_fusion\n"
    "from src.cdss_core.log_lr import LogLikelihoodRatio\n"
    "\n"
    "def test_duplicate_provenance_raises():\n"
    "    bb = Blackboard()\n"
    "    bb.update_latent('GFR', NormalPosterior(45, 5), 'lab_001', 'M3')\n"
    "    with pytest.raises(ProvenanceError):\n"
    "        bb.update_latent('GFR', NormalPosterior(46, 4), 'lab_001', 'M3')\n"
    "\n"
    "def test_fusion_order_invariant():\n"
    "    a = NormalPosterior(45, 5)\n"
    "    b = NormalPosterior(50, 3)\n"
    "    f1 = precision_weighted_fusion([a, b])\n"
    "    f2 = precision_weighted_fusion([b, a])\n"
    "    assert abs(f1.mu - f2.mu) < 1e-10\n"
    "\n"
    "def test_fusion_reduces_uncertainty():\n"
    "    a = NormalPosterior(45, 5)\n"
    "    b = NormalPosterior(50, 3)\n"
    "    fused = precision_weighted_fusion([a, b])\n"
    "    assert fused.sigma < a.sigma and fused.sigma < b.sigma\n"
    "\n"
    "def test_harm_lr_accumulation():\n"
    "    bb = Blackboard()\n"
    "    bb.add_harm_evidence('QT', LogLikelihoodRatio(1.5, 'm1_qt', 'M1'))\n"
    "    bb.add_harm_evidence('QT', LogLikelihoodRatio(0.8, 'm3_k', 'M3'))\n"
    "    assert abs(bb.get_fused_log_lr('QT') - 2.3) < 1e-10\n",
    encoding="utf-8"
)
print("Fixed test_blackboard.py")

# Fix test_advisor.py
Path("tests/test_advisor.py").write_text(
    "import pytest, sys, os\n"
    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))\n"
    "from src.advisor.utility import ExpectedUtilityCalculator\n"
    "from src.advisor.knapsack import AlertBudgetSelector\n"
    "from src.advisor.cusum import CUSUMMonitor\n"
    "\n"
    "def test_expected_utility():\n"
    "    calc = ExpectedUtilityCalculator()\n"
    "    calc.add_action('hold', u_harm=-10, u_noharm=5)\n"
    "    result = calc.select_best_action(p_harm=0.8)\n"
    "    assert abs(result['expected_utility'] - (0.8 * -10 + 0.2 * 5)) < 1e-9\n"
    "\n"
    "def test_knapsack_budget():\n"
    "    sel = AlertBudgetSelector(budget=2)\n"
    "    alerts = [{'net_benefit': i, 'attention_cost': 1} for i in range(5)]\n"
    "    assert len(sel.select_alerts(alerts)) <= 2\n"
    "\n"
    "def test_cusum_alarm():\n"
    "    mon = CUSUMMonitor(threshold=5.0, slack=0.5)\n"
    "    for _ in range(20):\n"
    "        mon.update(1.0)\n"
    "    assert mon.alarm_count > 0\n",
    encoding="utf-8"
)
print("Fixed test_advisor.py")
print("Done! Now run: python -m pytest tests/ -v")