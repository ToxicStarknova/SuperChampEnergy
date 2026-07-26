import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class SimulationParams:
    capacity: float = 30.0
    usable_pct: float = 100.0
    charge_rate: float = 10.0
    grid_efficiency: float = 95.0
    solar_efficiency: float = 85.0
    min_soc: float = 10.0
    max_soc: float = 100.0
    mic: float = 18.0
    mec: float = 6.0
    region: str = "rural"

    @property
    def usable_capacity_kwh(self) -> float:
        return self.capacity * (self.usable_pct / 100.0)

    @property
    def min_soc_kwh(self) -> float:
        return self.usable_capacity_kwh * (self.min_soc / 100.0)

    @property
    def max_soc_kwh(self) -> float:
        return self.usable_capacity_kwh * (self.max_soc / 100.0)

    @property
    def grid_rte_decimal(self) -> float:
        return self.grid_efficiency / 100.0

    @property
    def solar_charge_efficiency(self) -> float:
        grid_sqrt = np.sqrt(self.grid_rte_decimal)
        return (self.solar_efficiency / 100.0) / max(0.01, grid_sqrt)


@dataclass
class FinancialROIParams:
    battery_capex: float = 5500.0
    inverter_capex: float = 1500.0
    grant_amount: float = 2100.0  # e.g., SEAI grant in Ireland
    electricity_inflation_pct: float = 3.0
    discount_rate_pct: float = 4.0
    annual_degradation_pct: float = 2.0
    horizon_years: int = 10

    @property
    def net_investment(self) -> float:
        return max(0.0, (self.battery_capex + self.inverter_capex) - self.grant_amount)


@dataclass
class DualTariffParams:
    exit_fee_per_switch: float = 50.0  # Early contract termination fee per switch
    num_switches_per_year: int = 2     # Spring & Autumn switches
    winter_months: List[int] = field(default_factory=lambda: [11, 12, 1, 2, 3])  # Nov - Mar (5 months)
    summer_months: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 9, 10]) # Apr - Oct (7 months)

    @property
    def total_annual_fees(self) -> float:
        return self.exit_fee_per_switch * self.num_switches_per_year


@dataclass
class DualTariffResult:
    winter_supplier: str
    winter_tariff: str
    winter_strategy: str
    winter_cost: float
    summer_supplier: str
    summer_tariff: str
    summer_strategy: str
    summer_cost: float
    total_exit_fees: float
    net_annual_bill: float
    extra_savings_vs_single_best: float


class FinancialROICalculator:
    """Calculates financial metrics: Simple Payback, 10-Year Cumulative Savings, ROI %, and NPV."""
    
    @staticmethod
    def calculate_roi(annual_savings_year1: float, params: FinancialROIParams) -> Dict[str, Any]:
        net_inv = params.net_investment
        if net_inv <= 0 or annual_savings_year1 <= 0:
            return {
                'net_investment': net_inv,
                'payback_years': 0.0,
                'ten_year_net_savings': annual_savings_year1 * params.horizon_years,
                'roi_percent': 0.0,
                'npv': 0.0,
                'yearly_cash_flows': [annual_savings_year1] * params.horizon_years
            }

        simple_payback = net_inv / annual_savings_year1 if annual_savings_year1 > 0 else 99.0
        
        yearly_cash_flows = []
        cumulative_savings = 0.0
        npv = -net_inv
        
        for yr in range(1, params.horizon_years + 1):
            degradation_factor = (1.0 - (params.annual_degradation_pct / 100.0)) ** (yr - 1)
            inflation_factor = (1.0 + (params.electricity_inflation_pct / 100.0)) ** (yr - 1)
            discount_factor = (1.0 + (params.discount_rate_pct / 100.0)) ** yr
            
            savings_yr = annual_savings_year1 * degradation_factor * inflation_factor
            yearly_cash_flows.append(savings_yr)
            cumulative_savings += savings_yr
            npv += savings_yr / discount_factor

        ten_year_net = cumulative_savings - net_inv
        roi_pct = (ten_year_net / net_inv) * 100.0 if net_inv > 0 else 0.0

        return {
            'net_investment': round(net_inv, 2),
            'payback_years': round(simple_payback, 1),
            'ten_year_savings': round(cumulative_savings, 2),
            'ten_year_net_savings': round(ten_year_net, 2),
            'roi_percent': round(roi_pct, 1),
            'npv': round(npv, 2),
            'yearly_cash_flows': [round(cf, 2) for cf in yearly_cash_flows]
        }


@dataclass
class HardwareExpansionParams:
    """Parameters for CapEx inputs and baseline hardware for sensitivity matrix evaluation."""
    battery_cost_per_kwh: float = 300.0
    pv_cost_per_kwp: float = 900.0
    baseline_pv_kwp: float = 6.1


@dataclass
class HardwareScenarioResult:
    """Stores simulation & financial evaluation results for a single hardware expansion scenario."""
    battery_addition_kwh: float
    pv_scale_factor: float
    pv_addition_kwp: float
    total_battery_capacity_kwh: float
    winning_supplier: str
    winning_tariff: str
    winning_strategy: str
    annual_bill: float
    incremental_savings: float
    expansion_capex: float
    simple_payback_years: float
    ten_year_npv: float = 0.0
    ten_year_roi_percent: float = 0.0
    is_sweet_spot: bool = False
    is_baseline: bool = False

