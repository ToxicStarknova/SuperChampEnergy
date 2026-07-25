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
