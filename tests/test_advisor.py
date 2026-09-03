import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.advisor.utility import ExpectedUtilityCalculator
from src.advisor.knapsack import AlertBudgetSelector
from src.advisor.cusum import CUSUMMonitor

def test_expected_utility():
    calc = ExpectedUtilityCalculator()
    calc.add_action('hold', u_harm=-10, u_noharm=5)
    result = calc.select_best_action(p_harm=0.8)
    assert abs(result['expected_utility'] - (0.8 * -10 + 0.2 * 5)) < 1e-9

def test_knapsack_budget():
    sel = AlertBudgetSelector(budget=2)
    alerts = [{'net_benefit': i, 'attention_cost': 1} for i in range(5)]
    assert len(sel.select_alerts(alerts)) <= 2

def test_cusum_alarm():
    mon = CUSUMMonitor(threshold=5.0, slack=0.5)
    for _ in range(20):
        mon.update(1.0)
    assert mon.alarm_count > 0
