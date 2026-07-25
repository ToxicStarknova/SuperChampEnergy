import re
import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Dict, Any, List

from core.models import SimulationParams

@njit
def _calc_cost_with_overage(imports: np.ndarray, prices: np.ndarray, is_ev_window: np.ndarray, 
                            ev_overage_rate: float, months: np.ndarray, has_overage_penalty: bool,
                            ev_cap_kwh: float = 1000.0) -> Tuple[np.ndarray, bool]:
    """Computes import electricity costs including EV bimonthly promotional cap overage penalties."""
    n = len(imports)
    cost = np.zeros(n)
    ev_bimonthly_usage = np.zeros(6)
    
    for i in range(n):
        grid_import = imports[i]
        
        if is_ev_window[i]:
            bimonthly_idx = int((months[i] - 1) / 2)
            if bimonthly_idx > 5: bimonthly_idx = 5
            
            current_usage = ev_bimonthly_usage[bimonthly_idx]
            ev_bimonthly_usage[bimonthly_idx] += grid_import
            
            if has_overage_penalty:
                if current_usage >= ev_cap_kwh:
                    cost[i] = grid_import * ev_overage_rate
                elif ev_bimonthly_usage[bimonthly_idx] > ev_cap_kwh:
                    under_amount = ev_cap_kwh - current_usage
                    over_amount = ev_bimonthly_usage[bimonthly_idx] - ev_cap_kwh
                    cost[i] = (under_amount * prices[i]) + (over_amount * ev_overage_rate)
                else:
                    cost[i] = grid_import * prices[i]
            else:
                cost[i] = grid_import * prices[i]
        else:
            cost[i] = grid_import * prices[i]
            
    limit_exceeded = False
    if has_overage_penalty:
        for val in ev_bimonthly_usage:
            if val > ev_cap_kwh:
                limit_exceeded = True
                break
            
    return cost, limit_exceeded


@njit
def _fast_simulate(consumptions: np.ndarray, generations: np.ndarray, hours: np.ndarray, months: np.ndarray,
                   force_charge_mask: np.ndarray, pre_charge_mask: np.ndarray, is_arbitrage_profitable_mask: np.ndarray,
                   usable_cap_kwh: float, min_soc_kwh: float, max_soc_kwh: float,
                   grid_rte: float, solar_charge_efficiency: float, grid_efficiency_sqrt: float,
                   charge_rate_limit: float, mic: float, mec: float, strategy_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast JIT compiled battery dispatch simulation kernel.
    Strategy IDs:
      0: self-consumption
      1: import-minimiser
      2: export-maximiser
      3: balanced-export-maximiser
      4: import-minimiser-summer-pass
    """
    n = len(consumptions)
    grid_imports, grid_exports, soc_track = np.zeros(n), np.zeros(n), np.zeros(n)
    battery_soc = min_soc_kwh
    half_hour_charge_limit = charge_rate_limit * 0.5

    for i in range(n):
        hour, month = hours[i], months[i]
        is_heating_season = (month in [11, 12, 1, 2])
        is_summer = not is_heating_season
        home_demand, solar_gen = consumptions[i], generations[i]
        
        is_force_charge_hour = force_charge_mask[i] if strategy_id >= 1 else False
        is_pre_charge_hour = True if (strategy_id in [2, 3] and pre_charge_mask[i] and not is_force_charge_hour) else False
        is_arbitrage_profitable = is_arbitrage_profitable_mask[i]
            
        self_consumption = min(home_demand, solar_gen)
        remaining_demand, excess_solar = home_demand - self_consumption, solar_gen - self_consumption
        grid_import, grid_export = remaining_demand, 0.0
        
        # Track total battery discharge energy in this 30-min window (capped by inverter half_hour_charge_limit)
        interval_discharge_kwh = 0.0

        # Step 1: Discharging battery for home consumption (when not force-charging)
        if not is_force_charge_hour:
            available_energy = max(0.0, battery_soc - min_soc_kwh)
            allowed_discharge = max(0.0, half_hour_charge_limit - interval_discharge_kwh)
            discharge_for_home = min(remaining_demand, available_energy * grid_efficiency_sqrt, allowed_discharge)
            
            if discharge_for_home > 0.001:
                battery_soc -= discharge_for_home / grid_efficiency_sqrt
                grid_import = remaining_demand - discharge_for_home
                interval_discharge_kwh += discharge_for_home
                
        # Step 2: Excess Solar Routing
        if excess_solar > 0:
            if strategy_id in [2, 3] and is_pre_charge_hour and is_arbitrage_profitable:
                grid_export += excess_solar
            elif strategy_id == 4 and is_summer:
                # FIX: Strategy 4 ("Summer Pass") bypasses solar charging during summer months regardless of arbitrage margin
                grid_export += excess_solar
            else:
                space_in_battery = max(0.0, max_soc_kwh - battery_soc)
                charge_from_solar = min(excess_solar, space_in_battery / solar_charge_efficiency, half_hour_charge_limit)
                if charge_from_solar > 0.001:
                    battery_soc += charge_from_solar * solar_charge_efficiency
                    grid_export += (excess_solar - charge_from_solar)
                else:
                    grid_export += excess_solar
                    
        # Step 3: Strategy Specific Grid Interventions
        if strategy_id >= 1: 
            # Pre-charge Grid Export / Arbitrage Dump
            if is_pre_charge_hour and is_arbitrage_profitable:
                if not (strategy_id == 3 and is_heating_season):
                    available_energy = max(0.0, battery_soc - min_soc_kwh)
                    # FIX: Clamped by remaining inverter discharge headroom in this interval
                    allowed_discharge = max(0.0, half_hour_charge_limit - interval_discharge_kwh)
                    energy_to_discharge = min(available_energy * grid_efficiency_sqrt, allowed_discharge)
                    max_export_allowed = max(0.0, mec * 0.5 - grid_export)
                    energy_to_discharge = min(energy_to_discharge, max_export_allowed)
                    
                    if energy_to_discharge > 0.001:
                        battery_soc -= (energy_to_discharge / grid_efficiency_sqrt)
                        grid_export += energy_to_discharge
                        interval_discharge_kwh += energy_to_discharge
                        
            # Force Charging from Grid
            if is_force_charge_hour:
                space_in_battery = max(0.0, max_soc_kwh - battery_soc)
                charge_power = min(charge_rate_limit, mic)
                energy_to_charge = min(max(0.0, charge_power * 0.5), space_in_battery / grid_efficiency_sqrt)
                max_allowed_home_import = max(0.0, mic * 0.5 - energy_to_charge)
                grid_import = min(grid_import, max_allowed_home_import)
                
                if energy_to_charge > 0.001:
                    battery_soc += energy_to_charge * grid_efficiency_sqrt
                    grid_import += energy_to_charge
                    
        # Apply MEC Grid Export Cap
        if grid_export / 0.5 > mec:
            grid_export = mec * 0.5
            
        grid_imports[i], grid_exports[i] = grid_import, grid_export
        soc_track[i] = (battery_soc / usable_cap_kwh) * 100.0 if usable_cap_kwh > 0 else 0.0
        
    return grid_imports, grid_exports, soc_track


def run_simulation(df_hdf: pd.DataFrame, import_prices: pd.Series, export_price: float, 
                   strategy: str, force_charge_hours: List[bool], params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Runs battery simulation for fixed rate tariffs."""
    usable_cap_kwh = params.usable_capacity_kwh
    min_soc_kwh, max_soc_kwh = params.min_soc_kwh, params.max_soc_kwh
    grid_rte = params.grid_rte_decimal
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    solar_charge_efficiency = params.solar_charge_efficiency
    
    strategy_map = {
        'self-consumption': 0, 
        'import-minimiser': 1, 
        'export-maximiser': 2, 
        'balanced-export-maximiser': 3, 
        'import-minimiser-summer-pass': 4
    }
    
    hour_array = df_hdf.index.hour.values
    force_charge_mask = np.array(force_charge_hours, dtype=np.bool_)[hour_array]
    
    pre_charge_hours = np.zeros(24, dtype=np.bool_)
    if strategy in ['export-maximiser', 'balanced-export-maximiser']:
        for h in range(24):
            if force_charge_hours[h]:
                for offset in range(1, 5): 
                    pre_charge_hours[(h - offset) % 24] = True
    pre_charge_mask = pre_charge_hours[hour_array]
    
    cheapest_import_rate = np.min(import_prices.values[force_charge_mask]) if np.any(force_charge_mask) else 99.0
    arb_margin_c_kwh = ((export_price * grid_rte) - cheapest_import_rate) * 100.0
    is_arbitrage_profitable = arb_margin_c_kwh > 0
    is_arbitrage_profitable_mask = np.full(len(df_hdf), is_arbitrage_profitable, dtype=np.bool_)
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        df_hdf['consumption'].values, df_hdf['generation'].values, hour_array, df_hdf.index.month.values, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, params.charge_rate, params.mic, params.mec, strategy_map.get(strategy, 0)
    )
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh


def run_dynamic_simulation(df_hdf: pd.DataFrame, import_prices: pd.Series, export_price: Any, 
                           strategy: str, params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Runs battery simulation for dynamic wholesale tariffs with adaptive force-charging duration."""
    usable_cap_kwh = params.usable_capacity_kwh
    min_soc_kwh, max_soc_kwh = params.min_soc_kwh, params.max_soc_kwh
    grid_rte = params.grid_rte_decimal
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    solar_charge_efficiency = params.solar_charge_efficiency
    
    strategy_map = {
        'self-consumption': 0, 
        'import-minimiser': 1, 
        'export-maximiser': 2, 
        'balanced-export-maximiser': 3, 
        'import-minimiser-summer-pass': 4
    }
    
    # Adaptive force-charging calculation: compute required 30-min intervals based on battery size & charge rate
    hours_to_charge = usable_cap_kwh / max(0.5, params.charge_rate)
    req_intervals = max(4, min(16, int(round(hours_to_charge * 2.0))))
    
    df_temp = pd.DataFrame({'price': import_prices.values, 'date': df_hdf.index.date})
    df_temp['rank'] = df_temp.groupby('date')['price'].rank(method='first')
    force_charge_mask = (df_temp['rank'] <= req_intervals).values
    
    pre_charge_mask = np.zeros(len(force_charge_mask), dtype=np.bool_)
    if strategy in ['export-maximiser', 'balanced-export-maximiser']:
        is_dynamic_fit = isinstance(export_price, np.ndarray) and np.ptp(export_price) > 0.001
        if is_dynamic_fit:
            df_temp['export_price'] = export_price
            df_temp['exp_rank'] = df_temp.groupby('date')['export_price'].rank(method='first', ascending=False)
            pre_charge_mask = (df_temp['exp_rank'] <= req_intervals).values & (export_price > 0)
        else:
            for i in range(len(force_charge_mask)):
                if force_charge_mask[i]:
                    start_idx = max(0, i - 8)
                    for j in range(start_idx, i):
                        if not force_charge_mask[j]: pre_charge_mask[j] = True
                        
    min_daily_price = df_temp.groupby('date')['price'].transform('min').values
    is_arbitrage_profitable_mask = ((export_price * grid_rte) > min_daily_price) & (export_price > 0)
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        df_hdf['consumption'].values, df_hdf['generation'].values, df_hdf.index.hour.values, df_hdf.index.month.values, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, params.charge_rate, params.mic, params.mec, strategy_map.get(strategy, 0)
    )
    
    cheapest_import_rate = np.mean(import_prices.values[force_charge_mask]) if np.any(force_charge_mask) else 99.0
    if isinstance(export_price, np.ndarray):
        pos_exports = export_price[export_price > 0]
        avg_export = np.mean(pos_exports) if len(pos_exports) > 0 else 0.0
        arb_margin_c_kwh = ((avg_export * grid_rte) - cheapest_import_rate) * 100.0
    else:
        arb_margin_c_kwh = ((export_price * grid_rte) - cheapest_import_rate) * 100.0
    
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh
