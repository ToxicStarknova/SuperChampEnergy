import re
import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Dict, Any, List, Optional

from core.models import (SimulationParams, DualTariffParams, DualTariffResult, 
                         HardwareExpansionParams, HardwareScenarioResult, FinancialROIParams, FinancialROICalculator)

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
      5: import-only-no-pv
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
            elif strategy_id == 5:
                # import-only-no-pv completely bypasses solar charging year-round
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


STRATEGY_MAP = {
    'self-consumption': 0, 
    'import-minimiser': 1, 
    'export-maximiser': 2, 
    'balanced-export-maximiser': 3, 
    'import-minimiser-summer-pass': 4,
    'import-only-no-pv': 5
}


@njit(fastmath=True)
def _run_simulation_from_arrays(consumption: np.ndarray, generation: np.ndarray, 
                                hour_array: np.ndarray, month_array: np.ndarray,
                                import_prices: np.ndarray, export_price: float, 
                                force_charge_hours_24: np.ndarray, strategy_id: int, 
                                usable_cap_kwh: float, min_soc_kwh: float, max_soc_kwh: float,
                                grid_rte: float, solar_charge_efficiency: float,
                                charge_rate: float, mic: float, mec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pure Numba JIT array-based kernel for fixed rate battery simulation."""
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    
    force_charge_mask = force_charge_hours_24[hour_array]
    
    pre_charge_hours = np.zeros(24, dtype=np.bool_)
    if strategy_id == 2 or strategy_id == 3:
        for h in range(24):
            if force_charge_hours_24[h]:
                for offset in range(1, 5): 
                    pre_charge_hours[(h - offset) % 24] = True
    pre_charge_mask = pre_charge_hours[hour_array]
    
    cheapest_import_rate = 99.0
    has_fc = False
    for i in range(len(import_prices)):
        if force_charge_mask[i]:
            if not has_fc or import_prices[i] < cheapest_import_rate:
                cheapest_import_rate = import_prices[i]
                has_fc = True
                
    arb_margin_c_kwh = ((export_price * grid_rte) - cheapest_import_rate) * 100.0
    is_arbitrage_profitable = arb_margin_c_kwh > 0
    is_arbitrage_profitable_mask = np.full(len(consumption), is_arbitrage_profitable, dtype=np.bool_)
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        consumption, generation, hour_array, month_array, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, charge_rate, mic, mec, strategy_id
    )
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh


def run_simulation(df_hdf: pd.DataFrame, import_prices: pd.Series, export_price: float, 
                   strategy: str, force_charge_hours: List[bool], params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Runs battery simulation for fixed rate tariffs."""
    strategy_id = STRATEGY_MAP.get(strategy, 0)
    fc_hours_arr = np.array(force_charge_hours, dtype=np.bool_)
    imp_prices_arr = import_prices.values if isinstance(import_prices, pd.Series) else import_prices
    
    return _run_simulation_from_arrays(
        df_hdf['consumption'].values.astype(np.float64), 
        df_hdf['generation'].values.astype(np.float64),
        df_hdf.index.hour.values.astype(np.int64), 
        df_hdf.index.month.values.astype(np.int64),
        imp_prices_arr.astype(np.float64), 
        float(export_price), 
        fc_hours_arr, 
        int(strategy_id),
        float(params.usable_capacity_kwh), 
        float(params.min_soc_kwh), 
        float(params.max_soc_kwh),
        float(params.grid_rte_decimal), 
        float(params.solar_charge_efficiency), 
        float(params.charge_rate), 
        float(params.mic), 
        float(params.mec)
    )


@njit(fastmath=True)
def _calc_dynamic_masks(import_prices: np.ndarray, export_prices: np.ndarray, 
                        day_ids: np.ndarray, req_intervals: int, strategy_id: int, grid_rte: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pure Numba helper for dynamic tariff mask generation."""
    n = len(import_prices)
    force_charge_mask = np.zeros(n, dtype=np.bool_)
    pre_charge_mask = np.zeros(n, dtype=np.bool_)
    is_arbitrage_profitable_mask = np.zeros(n, dtype=np.bool_)
    
    num_days = day_ids[-1] + 1 if n > 0 else 0
    
    start_idx = 0
    for day in range(num_days):
        end_idx = start_idx
        while end_idx < n and day_ids[end_idx] == day:
            end_idx += 1
            
        day_len = end_idx - start_idx
        if day_len > 0:
            day_prices = import_prices[start_idx:end_idx]
            sorted_indices = np.argsort(day_prices)
            for k in range(min(req_intervals, day_len)):
                idx_in_day = sorted_indices[k]
                force_charge_mask[start_idx + idx_in_day] = True
                
            min_p = day_prices[sorted_indices[0]]
            for i_d in range(start_idx, end_idx):
                if (export_prices[i_d] * grid_rte > min_p) and (export_prices[i_d] > 0):
                    is_arbitrage_profitable_mask[i_d] = True
                    
            if strategy_id == 2 or strategy_id == 3:
                max_exp = export_prices[start_idx]
                min_exp = export_prices[start_idx]
                for i_d in range(start_idx + 1, end_idx):
                    if export_prices[i_d] > max_exp: max_exp = export_prices[i_d]
                    if export_prices[i_d] < min_exp: min_exp = export_prices[i_d]
                
                is_dynamic_fit = (max_exp - min_exp) > 0.001
                if is_dynamic_fit:
                    day_exp_prices = export_prices[start_idx:end_idx]
                    sorted_exp_indices = np.argsort(-day_exp_prices)
                    for k in range(min(req_intervals, day_len)):
                        idx_in_day = sorted_exp_indices[k]
                        global_idx = start_idx + idx_in_day
                        if export_prices[global_idx] > 0:
                            pre_charge_mask[global_idx] = True
                else:
                    for i_d in range(start_idx, end_idx):
                        if force_charge_mask[i_d]:
                            p_start = max(start_idx, i_d - 8)
                            for j_d in range(p_start, i_d):
                                if not force_charge_mask[j_d]:
                                    pre_charge_mask[j_d] = True

        start_idx = end_idx

    cheapest_sum = 0.0
    cheapest_count = 0
    for i in range(n):
        if force_charge_mask[i]:
            cheapest_sum += import_prices[i]
            cheapest_count += 1
    cheapest_import_rate = cheapest_sum / cheapest_count if cheapest_count > 0 else 99.0
    
    pos_exp_sum = 0.0
    pos_exp_count = 0
    for i in range(n):
        if export_prices[i] > 0:
            pos_exp_sum += export_prices[i]
            pos_exp_count += 1
    avg_export = pos_exp_sum / pos_exp_count if pos_exp_count > 0 else 0.0
    arb_margin_c_kwh = ((avg_export * grid_rte) - cheapest_import_rate) * 100.0

    return force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, arb_margin_c_kwh


@njit(fastmath=True)
def _run_dynamic_simulation_from_arrays(consumption: np.ndarray, generation: np.ndarray,
                                        hour_array: np.ndarray, month_array: np.ndarray, day_ids: np.ndarray,
                                        import_prices: np.ndarray, export_prices: np.ndarray, 
                                        strategy_id: int, usable_cap_kwh: float, min_soc_kwh: float, max_soc_kwh: float,
                                        grid_rte: float, solar_charge_efficiency: float,
                                        charge_rate: float, mic: float, mec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pure Numba JIT array-based kernel for dynamic wholesale battery simulation."""
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    
    hours_to_charge = usable_cap_kwh / max(0.5, charge_rate)
    req_intervals = max(4, min(16, int(round(hours_to_charge * 2.0))))
    
    force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, arb_margin_c_kwh = _calc_dynamic_masks(
        import_prices, export_prices, day_ids, req_intervals, strategy_id, grid_rte
    )
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        consumption, generation, hour_array, month_array, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, charge_rate, mic, mec, strategy_id
    )
    
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh


def run_dynamic_simulation(df_hdf: pd.DataFrame, import_prices: pd.Series, export_price: Any, 
                           strategy: str, params: SimulationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Runs battery simulation for dynamic wholesale tariffs with adaptive force-charging duration."""
    strategy_id = STRATEGY_MAP.get(strategy, 0)
    imp_prices_arr = import_prices.values if isinstance(import_prices, pd.Series) else import_prices
    
    if isinstance(export_price, (int, float)):
        exp_prices_arr = np.full(len(df_hdf), float(export_price), dtype=np.float64)
    elif isinstance(export_price, pd.Series):
        exp_prices_arr = export_price.values.astype(np.float64)
    else:
        exp_prices_arr = np.asarray(export_price, dtype=np.float64)
        
    dates_array = df_hdf.index.date
    _, day_ids = np.unique(dates_array, return_inverse=True)
    
    return _run_dynamic_simulation_from_arrays(
        df_hdf['consumption'].values.astype(np.float64),
        df_hdf['generation'].values.astype(np.float64),
        df_hdf.index.hour.values.astype(np.int64),
        df_hdf.index.month.values.astype(np.int64),
        day_ids.astype(np.int64),
        imp_prices_arr.astype(np.float64),
        exp_prices_arr,
        int(strategy_id),
        float(params.usable_capacity_kwh),
        float(params.min_soc_kwh),
        float(params.max_soc_kwh),
        float(params.grid_rte_decimal),
        float(params.solar_charge_efficiency),
        float(params.charge_rate),
        float(params.mic),
        float(params.mec)
    )


def evaluate_dual_tariffs(df_res: pd.DataFrame, df_hdf: pd.DataFrame, 
                          dual_params: DualTariffParams) -> List[DualTariffResult]:
    """
    Evaluates seasonal switching between a Winter Tariff and a Summer Tariff,
    deducting early contract cancellation fees to find optimal dual-tariff combinations.
    """
    if df_res.empty or df_hdf.empty:
        return []

    # Exclude baseline rows
    df_active = df_res[df_res['Strategy'] != 'baseline-no-battery'].copy()
    if df_active.empty:
        return []

    months_array = df_hdf.index.month.values
    winter_mask = np.isin(months_array, dual_params.winter_months)
    summer_mask = np.isin(months_array, dual_params.summer_months)
    
    num_winter_months = len(dual_params.winter_months)
    num_summer_months = len(dual_params.summer_months)
    
    single_best_bill = df_active['Bill'].min()

    # Find best strategy for each unique tariff in winter vs summer
    unique_tariffs = df_active[['Supplier', 'Tariff', '_id', 'Fixed', 'Bonus']].drop_duplicates().to_dict('records')
    
    seasonal_tariff_profiles = []
    for t_info in unique_tariffs:
        tid = t_info['_id']
        sub_rows = df_active[df_active['_id'] == tid]
        
        # Calculate seasonal cost for each strategy under this tariff
        winter_best_cost = 999999.0
        winter_best_strat = ""
        summer_best_cost = 999999.0
        summer_best_strat = ""
        
        for _, row in sub_rows.iterrows():
            strat = row['Strategy']
            # Reconstruct net monthly sums for winter vs summer
            # Import cost - Export revenue during winter
            # We approximate winter/summer net from row data proportional to energy sums
            imp_cost = row['Import']
            exp_rev = row['Export']
            
            # Weighted seasonal split
            winter_ratio = np.sum(winter_mask) / len(winter_mask)
            summer_ratio = np.sum(summer_mask) / len(summer_mask)
            
            c_winter = (imp_cost - exp_rev) * winter_ratio + (row['Fixed'] * (num_winter_months / 12.0))
            c_summer = (imp_cost - exp_rev) * summer_ratio + (row['Fixed'] * (num_summer_months / 12.0))
            
            if c_winter < winter_best_cost:
                winter_best_cost = c_winter
                winter_best_strat = strat
                
            if c_summer < summer_best_cost:
                summer_best_cost = c_summer
                summer_best_strat = strat
                
        seasonal_tariff_profiles.append({
            'Supplier': t_info['Supplier'],
            'Tariff': t_info['Tariff'],
            '_id': tid,
            'Fixed': t_info['Fixed'],
            'Bonus': t_info['Bonus'],
            'winter_cost': winter_best_cost,
            'winter_strat': winter_best_strat,
            'summer_cost': summer_best_cost,
            'summer_strat': summer_best_strat
        })

    dual_results = []
    total_exit_fees = dual_params.total_annual_fees
    
    for w in seasonal_tariff_profiles:
        for s in seasonal_tariff_profiles:
            # Dual-tariff annual bill = Winter Cost + Summer Cost + Exit Fees - Cash Bonuses
            net_bill = w['winter_cost'] + s['summer_cost'] + total_exit_fees - w['Bonus'] - s['Bonus']
            extra_savings = single_best_bill - net_bill
            
            dual_results.append(DualTariffResult(
                winter_supplier=w['Supplier'],
                winter_tariff=w['Tariff'],
                winter_strategy=str(w['winter_strat']).replace('-', ' ').title(),
                winter_cost=round(w['winter_cost'], 2),
                summer_supplier=s['Supplier'],
                summer_tariff=s['Tariff'],
                summer_strategy=str(s['summer_strat']).replace('-', ' ').title(),
                summer_cost=round(s['summer_cost'], 2),
                total_exit_fees=round(total_exit_fees, 2),
                net_annual_bill=round(net_bill, 2),
                extra_savings_vs_single_best=round(extra_savings, 2)
            ))
            
    # Sort by lowest net annual bill
    dual_results.sort(key=lambda x: x.net_annual_bill)
    return dual_results


@njit(fastmath=True)
def _calc_ideal_daily_adaptive(costs_matrix: np.ndarray, exp_rev_matrix: np.ndarray,
                               imports_matrix: np.ndarray, exports_matrix: np.ndarray,
                               soc_matrix: np.ndarray, day_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes ideal daily adaptive strategy selection using pure Numba JIT.
    costs_matrix: shape (5, N)
    exp_rev_matrix: shape (5, N)
    imports_matrix: shape (5, N)
    exports_matrix: shape (5, N)
    soc_matrix: shape (5, N)
    """
    n = costs_matrix.shape[1]
    num_days = day_ids[-1] + 1 if n > 0 else 0
    
    ideal_imp = np.zeros(n, dtype=np.float64)
    ideal_exp = np.zeros(n, dtype=np.float64)
    ideal_soc = np.zeros(n, dtype=np.float64)
    ideal_costs = np.zeros(n, dtype=np.float64)
    ideal_exp_rev = np.zeros(n, dtype=np.float64)
    win_counts = np.zeros(6, dtype=np.int64)
    daily_winners = np.zeros(num_days, dtype=np.int64)
    
    start_idx = 0
    for day in range(num_days):
        end_idx = start_idx
        while end_idx < n and day_ids[end_idx] == day:
            end_idx += 1
            
        day_len = end_idx - start_idx
        if day_len > 0:
            best_strat = 0
            best_day_cost = 1e9
            for s in range(6):
                net_c = 0.0
                for i in range(start_idx, end_idx):
                    net_c += costs_matrix[s, i] - exp_rev_matrix[s, i]
                if net_c < best_day_cost:
                    best_day_cost = net_c
                    best_strat = s
                    
            win_counts[best_strat] += 1
            daily_winners[day] = best_strat
            for i in range(start_idx, end_idx):
                ideal_imp[i] = imports_matrix[best_strat, i]
                ideal_exp[i] = exports_matrix[best_strat, i]
                ideal_soc[i] = soc_matrix[best_strat, i]
                ideal_costs[i] = costs_matrix[best_strat, i]
                ideal_exp_rev[i] = exp_rev_matrix[best_strat, i]
                
        start_idx = end_idx
        
    return ideal_imp, ideal_exp, ideal_soc, ideal_costs, ideal_exp_rev, win_counts, daily_winners


def warmup_engine():
    """Trigger Numba JIT compilation ahead of time using dummy 1D/2D arrays."""
    n = 48
    dummy_float = np.zeros(n, dtype=np.float64)
    dummy_int = np.zeros(n, dtype=np.int64)
    dummy_bool = np.zeros(n, dtype=np.bool_)
    dummy_bool_24 = np.zeros(24, dtype=np.bool_)
    dummy_matrix = np.zeros((6, n), dtype=np.float64)

    # Pre-compile all Numba @njit kernels ahead of time
    _ = _calc_cost_with_overage(dummy_float, dummy_float, dummy_bool, 0.0, dummy_int, False)
    _ = _fast_simulate(dummy_float, dummy_float, dummy_int, dummy_int, dummy_bool, dummy_bool, dummy_bool, 10.0, 1.0, 9.0, 0.9, 0.95, np.sqrt(0.9), 3.0, 5.0, 5.0, 0)
    _ = _run_simulation_from_arrays(dummy_float, dummy_float, dummy_int, dummy_int, dummy_float, 0.18, dummy_bool_24, 0, 10.0, 1.0, 9.0, 0.9, 0.95, 3.0, 5.0, 5.0)
    _ = _calc_dynamic_masks(dummy_float, dummy_float, dummy_int, 4, 0, 0.9)
    _ = _run_dynamic_simulation_from_arrays(dummy_float, dummy_float, dummy_int, dummy_int, dummy_int, dummy_float, dummy_float, 0, 10.0, 1.0, 9.0, 0.9, 0.95, 3.0, 5.0, 5.0)
    _ = _calc_ideal_daily_adaptive(dummy_matrix, dummy_matrix, dummy_matrix, dummy_matrix, dummy_matrix, dummy_int)


from core.parsers import get_half_hourly_rates_for_row


def run_hardware_expansion_matrix(df_hdf: pd.DataFrame, valid_tariffs: pd.DataFrame, 
                                   dynamic_suppliers: List[Dict[str, Any]], dam_prices_c_kwh: np.ndarray,
                                   base_params: SimulationParams, expansion_params: HardwareExpansionParams) -> List[HardwareScenarioResult]:
    """
    Evaluates 30 hardware expansion scenarios (5 Battery Additions x 6 PV Scale Factors).
    Returns a list of HardwareScenarioResult objects.
    """
    n_samples = len(df_hdf)
    orig_imports = df_hdf['consumption'].values.astype(np.float64)
    orig_exports = df_hdf['generation'].values.astype(np.float64)
    hours_array = df_hdf.index.hour.values.astype(np.int64)
    months_array = df_hdf.index.month.values.astype(np.int64)
    dates_array = df_hdf.index.date
    
    _, day_ids = np.unique(dates_array, return_inverse=True)
    day_ids = day_ids.astype(np.int64)
    
    unique_dates = np.unique(dates_array)
    num_days = len(unique_dates)
    scaling_factor = 365.0 / num_days if num_days > 0 else 1.0
    
    base_cap = float(base_params.usable_capacity_kwh)
    min_soc_kwh = float(base_params.min_soc_kwh)
    grid_rte = float(base_params.grid_rte_decimal)
    solar_charge_eff = float(base_params.solar_charge_efficiency)
    charge_rate = float(base_params.charge_rate)
    mic = float(base_params.mic)
    mec = float(base_params.mec)

    battery_additions = [0.0, 5.0, 10.0, 15.0, 20.0]
    pv_scale_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    all_strategies = ['self-consumption', 'import-minimiser', 'export-maximiser', 'balanced-export-maximiser', 'import-minimiser-summer-pass', 'import-only-no-pv']
    
    # Pre-parse tariff rates & static arrays once across all scenarios
    fixed_tariff_data = []
    for _, row in valid_tariffs.iterrows():
        try: fit_rate = float(row['Fit unit']) / 100.0 if not pd.isna(row.get('Fit unit')) else 0.18
        except (ValueError, TypeError): fit_rate = 0.18
            
        import_prices_arr, is_ev_window, ev_overage_rate, has_overage_penalty, _, _ = get_half_hourly_rates_for_row(row, df_hdf.index)
        
        first_day_prices = import_prices_arr[:48]
        min_p = np.min(first_day_prices)
        fc_24 = np.zeros(24, dtype=np.bool_)
        for h in range(24):
            if first_day_prices[h*2] <= min_p + 0.001 or first_day_prices[h*2+1] <= min_p + 0.001:
                fc_24[h] = True
                
        try: fixed_charges = float(row['Standing charge']) + float(row.get('PSO Levy', 0))
        except (ValueError, TypeError): fixed_charges = 300.0
            
        cash_bonus = float(row.get('Cash bonus', 0.0)) if not pd.isna(row.get('Cash bonus')) else 0.0
        
        fixed_tariff_data.append({
            'supplier': row['Supplier'], 'tariff': row['Tariff name'],
            'fit_rate': fit_rate, 'import_prices_arr': import_prices_arr,
            'is_ev_window': is_ev_window, 'ev_overage_rate': ev_overage_rate,
            'has_overage_penalty': has_overage_penalty, 'fc_24': fc_24,
            'fixed_charges': fixed_charges, 'cash_bonus': cash_bonus
        })
        
    dynamic_tariff_data = []
    for dyn in dynamic_suppliers:
        if isinstance(dyn['Fit unit'], str) and dyn['Fit unit'].upper() == "DAM":
            fit_rate_arr = dam_prices_c_kwh.copy() / 100.0
            fit_payment_time = dyn.get('FIT Payment time', '')
            if fit_payment_time:
                match = re.search(r'fit_(\d+)_(\d+)', fit_payment_time)
                if match:
                    start_h, end_h = int(match.group(1)), int(match.group(2))
                    fit_window = (hours_array >= start_h) & (hours_array < end_h)
                    fit_rate_arr = np.where(fit_window, fit_rate_arr, 0.0)
        else:
            try: fit_rate_arr = np.full(n_samples, float(dyn['Fit unit']) / 100.0, dtype=np.float64)
            except ValueError: fit_rate_arr = np.full(n_samples, 0.18, dtype=np.float64)
            
        fixed_charges = dyn['Standing charge'] * 1.09  
        cash_bonus = dyn.get('Cash bonus', 0.0)
        
        prices = dam_prices_c_kwh.copy()
        is_night_mask = (hours_array >= 23) | (hours_array < 8)
        is_peak_mask = (hours_array >= 17) & (hours_array < 19)
        is_day_mask = ~(is_night_mask | is_peak_mask)
        prices[is_night_mask] += dyn['Night']
        prices[is_day_mask] += dyn['Day']
        prices[is_peak_mask] += dyn['Peak']
        import_prices_arr = (prices / 100.0) * 1.09
        
        dynamic_tariff_data.append({
            'supplier': dyn['Supplier'], 'tariff': dyn['Tariff name'],
            'fit_rate_arr': fit_rate_arr, 'import_prices_arr': import_prices_arr,
            'fixed_charges': fixed_charges, 'cash_bonus': cash_bonus
        })
        
    costs_matrix = np.zeros((6, n_samples), dtype=np.float64)
    exp_rev_matrix = np.zeros((6, n_samples), dtype=np.float64)
    imports_matrix = np.zeros((6, n_samples), dtype=np.float64)
    exports_matrix = np.zeros((6, n_samples), dtype=np.float64)
    soc_matrix = np.zeros((6, n_samples), dtype=np.float64)

    raw_results = []
    baseline_bill = 0.0

    for batt_add in battery_additions:
        usable_cap = base_cap + batt_add
        max_soc = usable_cap # 100% SoC upper bound
        
        for pv_scale in pv_scale_factors:
            scaled_exports = orig_exports * pv_scale
            
            best_scenario_bill = 1e9
            best_supplier = ""
            best_tariff = ""
            best_strategy = ""
            
            # Evaluate Fixed Tariffs
            for ft in fixed_tariff_data:
                for strat_idx in range(6):
                    imp, exp, soc, _ = _run_simulation_from_arrays(
                        orig_imports, scaled_exports, hours_array, months_array,
                        ft['import_prices_arr'], ft['fit_rate'], ft['fc_24'], strat_idx,
                        usable_cap, min_soc_kwh, max_soc, grid_rte, solar_charge_eff,
                        charge_rate, mic, mec
                    )
                    costs, _ = _calc_cost_with_overage(
                        imp, ft['import_prices_arr'], ft['is_ev_window'], ft['ev_overage_rate'], months_array, ft['has_overage_penalty']
                    )
                    exp_rev = exp * ft['fit_rate']
                    costs_matrix[strat_idx, :] = costs
                    exp_rev_matrix[strat_idx, :] = exp_rev
                    imports_matrix[strat_idx, :] = imp
                    exports_matrix[strat_idx, :] = exp
                    soc_matrix[strat_idx, :] = soc
                    
                    bill = (np.sum(costs) - np.sum(exp_rev)) * scaling_factor + ft['fixed_charges'] - ft['cash_bonus']
                    if bill < best_scenario_bill:
                        best_scenario_bill = bill
                        best_supplier = ft['supplier']
                        best_tariff = ft['tariff']
                        best_strategy = all_strategies[strat_idx]
                        
                # Ideal Daily Adaptive
                _, _, _, ideal_costs, ideal_exp_rev, _, _ = _calc_ideal_daily_adaptive(
                    costs_matrix, exp_rev_matrix, imports_matrix, exports_matrix, soc_matrix, day_ids
                )
                ideal_bill = (np.sum(ideal_costs) - np.sum(ideal_exp_rev)) * scaling_factor + ft['fixed_charges'] - ft['cash_bonus']
                if ideal_bill < best_scenario_bill:
                    best_scenario_bill = ideal_bill
                    best_supplier = ft['supplier']
                    best_tariff = ft['tariff']
                    best_strategy = "Ideal Daily Adaptive"

            # Evaluate Dynamic Tariffs
            for dt in dynamic_tariff_data:
                for strat_idx in range(6):
                    imp, exp, soc, _ = _run_dynamic_simulation_from_arrays(
                        orig_imports, scaled_exports, hours_array, months_array, day_ids,
                        dt['import_prices_arr'], dt['fit_rate_arr'], strat_idx,
                        usable_cap, min_soc_kwh, max_soc, grid_rte, solar_charge_eff,
                        charge_rate, mic, mec
                    )
                    costs, _ = _calc_cost_with_overage(
                        imp, dt['import_prices_arr'], np.zeros(n_samples, dtype=np.bool_), 0.0, months_array, False
                    )
                    exp_rev = exp * dt['fit_rate_arr']
                    costs_matrix[strat_idx, :] = costs
                    exp_rev_matrix[strat_idx, :] = exp_rev
                    imports_matrix[strat_idx, :] = imp
                    exports_matrix[strat_idx, :] = exp
                    soc_matrix[strat_idx, :] = soc
                    
                    bill = (np.sum(costs) - np.sum(exp_rev)) * scaling_factor + dt['fixed_charges'] - dt['cash_bonus']
                    if bill < best_scenario_bill:
                        best_scenario_bill = bill
                        best_supplier = dt['supplier']
                        best_tariff = dt['tariff']
                        best_strategy = all_strategies[strat_idx]
                        
                # Ideal Daily Adaptive
                _, _, _, ideal_costs, ideal_exp_rev, _, _ = _calc_ideal_daily_adaptive(
                    costs_matrix, exp_rev_matrix, imports_matrix, exports_matrix, soc_matrix, day_ids
                )
                ideal_bill = (np.sum(ideal_costs) - np.sum(ideal_exp_rev)) * scaling_factor + dt['fixed_charges'] - dt['cash_bonus']
                if ideal_bill < best_scenario_bill:
                    best_scenario_bill = ideal_bill
                    best_supplier = dt['supplier']
                    best_tariff = dt['tariff']
                    best_strategy = "Ideal Daily Adaptive"

            is_base = (batt_add == 0.0 and pv_scale == 1.0)
            if is_base:
                baseline_bill = best_scenario_bill

            raw_results.append({
                'batt_add': batt_add,
                'pv_scale': pv_scale,
                'total_cap': usable_cap,
                'supplier': best_supplier,
                'tariff': best_tariff,
                'strategy': best_strategy,
                'bill': best_scenario_bill,
                'is_base': is_base
            })

    # Financial post-processing for all 30 scenarios
    final_results = []
    best_payback = 999.0
    sweet_spot_idx = -1

    for idx, r in enumerate(raw_results):
        batt_add = r['batt_add']
        pv_scale = r['pv_scale']
        pv_add_kwp = (pv_scale - 1.0) * expansion_params.baseline_pv_kwp
        
        capex = (batt_add * expansion_params.battery_cost_per_kwh) + (pv_add_kwp * expansion_params.pv_cost_per_kwp)
        savings = max(0.0, baseline_bill - r['bill'])
        
        if r['is_base']:
            payback = 0.0
            npv_val = 0.0
            roi_pct_val = 0.0
        elif savings > 0.01:
            payback = capex / savings
            roi_res = FinancialROICalculator.calculate_roi(
                savings,
                FinancialROIParams(
                    battery_capex=capex,
                    inverter_capex=0.0,
                    grant_amount=0.0,
                    horizon_years=10,
                    electricity_inflation_pct=3.0,
                    annual_degradation_pct=2.0,
                    discount_rate_pct=5.0
                )
            )
            npv_val = roi_res['npv']
            roi_pct_val = roi_res['roi_percent']
        else:
            payback = 999.0
            npv_val = -capex
            roi_pct_val = -100.0
            
        if not r['is_base'] and savings > 0 and payback < best_payback:
            best_payback = payback
            sweet_spot_idx = idx

        final_results.append(HardwareScenarioResult(
            battery_addition_kwh=batt_add,
            pv_scale_factor=pv_scale,
            pv_addition_kwp=round(pv_add_kwp, 2),
            total_battery_capacity_kwh=r['total_cap'],
            winning_supplier=r['supplier'],
            winning_tariff=r['tariff'],
            winning_strategy=r['strategy'].replace('-', ' ').title(),
            annual_bill=round(r['bill'], 2),
            incremental_savings=round(savings, 2),
            expansion_capex=round(capex, 2),
            simple_payback_years=round(payback, 1) if payback < 900 else 99.0,
            ten_year_npv=round(npv_val, 2),
            ten_year_roi_percent=round(roi_pct_val, 1),
            is_sweet_spot=False,
            is_baseline=r['is_base']
        ))

    if sweet_spot_idx >= 0:
        final_results[sweet_spot_idx].is_sweet_spot = True

    return final_results


# Automatically pre-compile Numba kernels on module load
warmup_engine()


