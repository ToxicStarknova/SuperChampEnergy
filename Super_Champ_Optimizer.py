import os
import re
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from datetime import datetime
from numba import njit

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]

# =====================================================================
# TOOLTIP HELPER CLASS
# =====================================================================
class ToolTip:
    """Creates a hover tooltip window for a specific widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)  # Ensure it renders on top layer
        
        # Fixed: Standard tk.Label uses padx/pady instead of padding
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#fef08a", foreground="#1e293b",
                         relief=tk.SOLID, borderwidth=1,
                         font=("Helvetica", 9, "normal"), padx=6, pady=6, wraplength=320)
        label.pack()

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


# =====================================================================
# 1. CORE PARSING & FAST SIMULATION ENGINE (NUMBA)
# =====================================================================

def parse_hdf(file_path):
    header_idx = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            lower_line = line.lower()
            if 'read date' in lower_line and 'read type' in lower_line:
                header_idx = i
                break
                
    df_raw = pd.read_csv(file_path, skiprows=header_idx, engine='c', skipinitialspace=True)
    df_raw.columns = [c.strip().replace('"', '') for c in df_raw.columns]
    
    mprn_col = next((c for c in df_raw.columns if 'mprn' in c.lower()), None)
    meter_col = next((c for c in df_raw.columns if 'serial' in c.lower()), None)
    mprn_val = str(df_raw[mprn_col].dropna().iloc[0]) if mprn_col and not df_raw[mprn_col].empty else "00000000000"
    meter_val = str(df_raw[meter_col].dropna().iloc[0]) if meter_col and not df_raw[meter_col].empty else "00000000"

    date_col = next(c for c in df_raw.columns if 'read date' in c.lower())
    type_col = next(c for c in df_raw.columns if 'read type' in c.lower())
    val_col = next((c for c in df_raw.columns if 'read val' in c.lower()), None)
    
    df_raw['timestamp'] = pd.to_datetime(df_raw[date_col].astype(str).str.replace('"', ''), format='mixed', dayfirst=True)
    df_raw['timestamp'] = df_raw['timestamp'] - pd.Timedelta(minutes=30)
    df_raw[val_col] = pd.to_numeric(df_raw[val_col], errors='coerce')
    
    df_raw['ReadType_Clean'] = df_raw[type_col].astype(str).str.lower()
    df_raw.loc[df_raw['ReadType_Clean'].str.contains('import'), 'Type'] = 'consumption'
    df_raw.loc[df_raw['ReadType_Clean'].str.contains('export'), 'Type'] = 'generation'
    
    df_raw = df_raw.dropna(subset=['Type', 'timestamp'])
    df = df_raw.pivot_table(index='timestamp', columns='Type', values=val_col, aggfunc='sum').fillna(0.0)
    
    if 'consumption' not in df.columns: df['consumption'] = 0.0
    if 'generation' not in df.columns: df['generation'] = 0.0
        
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df, mprn_val, meter_val

def filter_last_12_full_months(df):
    if df.empty: return df
    latest_ts = df.index[-1]
    end_date = datetime(latest_ts.year, latest_ts.month, 1)
    start_date = datetime(end_date.year - 1, end_date.month, 1)
    return df[(df.index >= start_date) & (df.index < end_date)]

def get_half_hourly_rates_for_row(row, date_range):
    plan_type = row['Plan type']
    day_rate = float(row['Day unit']) / 100.0
    peak_rate = float(row['Peak unit']) / 100.0 if not pd.isna(row.get('Peak unit')) else day_rate
    night_rate = float(row['Night unit']) / 100.0 if not pd.isna(row.get('Night unit')) else day_rate
    ev_rate = float(row['Ev unit']) / 100.0 if not pd.isna(row.get('Ev unit')) else None
    
    raw_ev_overage = row.get('Ev overage unit')
    if pd.notna(raw_ev_overage) and str(raw_ev_overage).strip() != "":
        ev_overage_rate = float(raw_ev_overage) / 100.0
        has_overage_penalty = True
    else:
        ev_overage_rate = ev_rate if ev_rate is not None else day_rate
        has_overage_penalty = False
    
    extra_tags = []
    if not pd.isna(row.get('Extra')):
        try: extra_tags = json.loads(row['Extra'].replace("''", '"'))
        except: pass

    ev_hours = set()
    if ev_rate is not None:
        for tag in extra_tags:
            if tag.startswith("ev_"):
                parts = tag.split("_")
                start_h, end_h = int(parts[1]), int(parts[2])
                if start_h < end_h: ev_hours.update(range(start_h, end_h))
                else:
                    ev_hours.update(range(start_h, 24))
                    ev_hours.update(range(0, end_h))

    prices = []
    is_ev_window = []
    
    for dt in date_range:
        hour, is_weekend = dt.hour, dt.weekday() >= 5
        in_ev = False
        
        if plan_type == '24h': 
            prices.append(day_rate)
        elif plan_type == 'day/night':
            prices.append(night_rate if hour >= 23 or hour < 8 else day_rate)
        elif plan_type == 'smart':
            if hour in ev_hours and ev_rate is not None: 
                prices.append(ev_rate)
                in_ev = True
            elif hour >= 23 or hour < 8: 
                prices.append(night_rate)
            elif 17 <= hour < 19:
                prices.append(day_rate if is_weekend and "no_peak_weekend" in extra_tags else peak_rate)
            else: 
                prices.append(day_rate)
        else: 
            prices.append(day_rate)
            
        is_ev_window.append(in_ev)
            
    return pd.Series(prices, index=date_range), np.array(is_ev_window, dtype=np.bool_), ev_overage_rate, has_overage_penalty

def prepare_dam(hdf_idx, dam_file):
    df_dam = pd.read_csv(dam_file, low_memory=False)
    df_dam.columns = df_dam.columns.str.strip().str.replace('"', '')
    
    if 'auction' in df_dam.columns:
        df_dam = df_dam[df_dam['auction'].astype(str).str.strip() == 'DAM']
    
    dt_series = pd.to_datetime(df_dam['timestamp'].astype(str).str.replace('"', ''), format='mixed')
    if dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_convert('Europe/Dublin').dt.tz_localize(None)
        
    df_dam['datetime'] = dt_series
    df_dam = df_dam.set_index('datetime').sort_index()
    df_dam = df_dam[~df_dam.index.duplicated(keep='first')]
    
    dam_resampled = df_dam[['price_eur']].reindex(hdf_idx, method='ffill').bfill()
    return dam_resampled['price_eur'].values / 10.0  # Convert Eur/MWh to c/kWh

def parse_dynamic_suppliers(file_path, region):
    df_fixed = pd.read_csv(file_path)
    df_fixed.columns = df_fixed.columns.str.strip().str.replace('"', '')
    suppliers = []
    for _, row in df_fixed.iterrows():
        supplier_name = row['c/KWh, ex. VAT']
        if pd.isna(supplier_name) or str(supplier_name).strip() == "":
            continue
            
        sc_str = str(row['SC € p.a. ex. VAT Urban (Rural)'])
        urban_sc, rural_sc = 0.0, 0.0
        match = re.search(r'([\d\.]+)\s*\(([\d\.]+)\)', sc_str)
        if match:
            urban_sc, rural_sc = float(match.group(1)), float(match.group(2))
        else:
            try: urban_sc = rural_sc = float(sc_str)
            except: pass
            
        suppliers.append({
            'Supplier': supplier_name,
            'Tariff name': 'Dynamic Wholesale',
            'Plan type': 'dynamic',
            'Night': float(row['Night']),
            'Day': float(row['Day']),
            'Peak': float(row['Peak']),
            'Fit unit': float(row['FIT']),
            'Standing charge': rural_sc if region.lower() == 'rural' else urban_sc,
            'PSO Levy': 0.0,
            'Cash bonus': 0.0
        })
    return suppliers

@njit
def _calc_cost_with_overage(imports, prices, is_ev_window, ev_overage_rate, months, has_overage_penalty):
    n = len(imports)
    cost = np.zeros(n)
    ev_bimonthly_usage = np.zeros(6)
    
    for i in range(n):
        grid_import = imports[i]
        
        if is_ev_window[i]:
            bimonthly_idx = int((months[i] - 1) / 2)
            current_usage = ev_bimonthly_usage[bimonthly_idx]
            ev_bimonthly_usage[bimonthly_idx] += grid_import
            
            # Fixed: Only apply overcharge rate boundaries if an active penalty policy threshold exists
            if has_overage_penalty:
                if current_usage >= 1000.0:
                    cost[i] = grid_import * ev_overage_rate
                elif ev_bimonthly_usage[bimonthly_idx] > 1000.0:
                    under_amount = 1000.0 - current_usage
                    over_amount = ev_bimonthly_usage[bimonthly_idx] - 1000.0
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
            if val > 1000.0:
                limit_exceeded = True
                break
            
    return cost, limit_exceeded

@njit
def _fast_simulate(consumptions, generations, hours, months,
                   force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask,
                   usable_cap_kwh, min_soc_kwh, max_soc_kwh,
                   grid_rte, solar_charge_efficiency, grid_efficiency_sqrt,
                   charge_rate_limit, mic, mec, strategy_id):
    
    n = len(consumptions)
    grid_imports, grid_exports, soc_track = np.zeros(n), np.zeros(n), np.zeros(n)
    battery_soc = min_soc_kwh

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
        
        if not is_force_charge_hour:
            available_energy = max(0.0, battery_soc - min_soc_kwh)
            discharge_for_home = min(remaining_demand, available_energy * grid_efficiency_sqrt, charge_rate_limit * 0.5)
            if discharge_for_home > 0.001:
                battery_soc -= discharge_for_home / grid_efficiency_sqrt
                grid_import = remaining_demand - discharge_for_home
                
        if excess_solar > 0:
            if strategy_id in [2, 3] and is_pre_charge_hour and is_arbitrage_profitable:
                grid_export += excess_solar
            elif strategy_id == 4 and is_summer and is_arbitrage_profitable:
                grid_export += excess_solar
            else:
                space_in_battery = max(0.0, max_soc_kwh - battery_soc)
                charge_from_solar = min(excess_solar, space_in_battery / solar_charge_efficiency, charge_rate_limit * 0.5)
                if charge_from_solar > 0.001:
                    battery_soc += charge_from_solar * solar_charge_efficiency
                    grid_export += (excess_solar - charge_from_solar)
                else:
                    grid_export += excess_solar
                    
        if strategy_id >= 1: 
            if is_pre_charge_hour and is_arbitrage_profitable:
                if not (strategy_id == 3 and is_heating_season):
                    available_energy = max(0.0, battery_soc - min_soc_kwh)
                    energy_to_discharge = min(available_energy * grid_efficiency_sqrt, charge_rate_limit * 0.5)
                    # Limit grid discharge by MEC export headroom
                    max_export_allowed = max(0.0, mec * 0.5 - grid_export)
                    energy_to_discharge = min(energy_to_discharge, max_export_allowed)
                    if energy_to_discharge > 0.001:
                        battery_soc -= (energy_to_discharge / grid_efficiency_sqrt)
                        grid_export += energy_to_discharge
                        
            if is_force_charge_hour:
                space_in_battery = max(0.0, max_soc_kwh - battery_soc)
                home_import_power = grid_import / 0.5
                available_grid_power = max(0.0, mic - home_import_power)
                charge_power = min(charge_rate_limit, available_grid_power)
                energy_to_charge = min(max(0.0, charge_power * 0.5), space_in_battery / grid_efficiency_sqrt)
                
                if energy_to_charge > 0.001:
                    battery_soc += energy_to_charge * grid_efficiency_sqrt
                    grid_import += energy_to_charge
                    
        if grid_export / 0.5 > mec: grid_export = mec * 0.5
            
        grid_imports[i], grid_exports[i] = grid_import, grid_export
        soc_track[i] = (battery_soc / usable_cap_kwh) * 100.0 if usable_cap_kwh > 0 else 0.0
        
    return grid_imports, grid_exports, soc_track

def run_simulation(df_hdf, import_prices, export_price, strategy, force_charge_hours, params):
    usable_cap_kwh = params['capacity'] * (params['usable_pct'] / 100.0)
    min_soc_kwh, max_soc_kwh = usable_cap_kwh * (params['min_soc'] / 100.0), usable_cap_kwh * (params['max_soc'] / 100.0)
    grid_rte = params['grid_efficiency'] / 100.0
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    solar_charge_efficiency = (params['solar_efficiency'] / 100.0) / max(0.01, grid_efficiency_sqrt)
    
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
                for offset in range(1, 5): pre_charge_hours[(h - offset) % 24] = True
    pre_charge_mask = pre_charge_hours[hour_array]
    
    cheapest_import_rate = np.min(import_prices.values[force_charge_mask]) if np.any(force_charge_mask) else 99.0
    arb_margin_c_kwh = ((export_price * grid_rte) - cheapest_import_rate) * 100.0
    is_arbitrage_profitable = arb_margin_c_kwh > 0
    is_arbitrage_profitable_mask = np.full(len(df_hdf), is_arbitrage_profitable, dtype=np.bool_)
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        df_hdf['consumption'].values, df_hdf['generation'].values, hour_array, df_hdf.index.month.values, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, params['charge_rate'], params['mic'], params['mec'], strategy_map.get(strategy, 0)
    )
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh

def run_dynamic_simulation(df_hdf, import_prices, export_price, strategy, params):
    usable_cap_kwh = params['capacity'] * (params['usable_pct'] / 100.0)
    min_soc_kwh, max_soc_kwh = usable_cap_kwh * (params['min_soc'] / 100.0), usable_cap_kwh * (params['max_soc'] / 100.0)
    grid_rte = params['grid_efficiency'] / 100.0
    grid_efficiency_sqrt = np.sqrt(grid_rte)
    solar_charge_efficiency = (params['solar_efficiency'] / 100.0) / max(0.01, grid_efficiency_sqrt)
    
    strategy_map = {
        'self-consumption': 0, 
        'import-minimiser': 1, 
        'export-maximiser': 2, 
        'balanced-export-maximiser': 3, 
        'import-minimiser-summer-pass': 4
    }
    
    df_temp = pd.DataFrame({'price': import_prices.values, 'date': df_hdf.index.date})
    df_temp['rank'] = df_temp.groupby('date')['price'].rank(method='first')
    force_charge_mask = (df_temp['rank'] <= 6).values
    
    pre_charge_mask = np.zeros(len(force_charge_mask), dtype=np.bool_)
    if strategy in ['export-maximiser', 'balanced-export-maximiser']:
        for i in range(len(force_charge_mask)):
            if force_charge_mask[i]:
                start_idx = max(0, i - 8)
                for j in range(start_idx, i):
                    if not force_charge_mask[j]: pre_charge_mask[j] = True
                        
    min_daily_price = df_temp.groupby('date')['price'].transform('min').values
    is_arbitrage_profitable_mask = (export_price * grid_rte) > min_daily_price
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        df_hdf['consumption'].values, df_hdf['generation'].values, df_hdf.index.hour.values, df_hdf.index.month.values, 
        force_charge_mask, pre_charge_mask, is_arbitrage_profitable_mask, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, params['charge_rate'], params['mic'], params['mec'], strategy_map.get(strategy, 0)
    )
    
    # Calculate average dynamic arbitrage return
    cheapest_import_rate = np.mean(import_prices.values[force_charge_mask]) if np.any(force_charge_mask) else 99.0
    arb_margin_c_kwh = ((export_price * grid_rte) - cheapest_import_rate) * 100.0
    
    return grid_imports, grid_exports, soc_track, arb_margin_c_kwh

# =====================================================================
# 2. TKINTER GUI APPLICATION
# =====================================================================

class HomeBatteryCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Home Battery & Tariff Optimization Tool - V2.1")
        self.root.geometry("1720x920")
        self.root.minsize(1400, 780)
        
        self.hdf_path, self.tariff_path = tk.StringVar(), tk.StringVar()
        self.dam_path, self.dynamic_adders_path = tk.StringVar(), tk.StringVar()
        self.leaderboard_data = None
        self.df_hdf = None 
        self.unique_dates = []
        self.current_date_idx = 0
        self.detailed_results = {}
        
        self.mprn = "00000000000"
        self.meter_serial = "00000000"
        self.custom_tariffs = []
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#4f46e5")
        self.style.configure("Sub.TLabel", font=("Helvetica", 10, "italic"))
        self.style.configure("Action.TButton", font=("Helvetica", 10, "bold"), background="#4f46e5", foreground="white")
        self.style.configure("Secondary.TButton", font=("Helvetica", 9))
        
        self.setup_ui()
        
    def treeview_sort_column(self, tv, col, reverse):
        l = []
        for k in tv.get_children(''):
            val = tv.set(k, col)
            l.append((val, k))
            
        def clean_val(val):
            val_clean = str(val).replace('€', '').replace('c/kWh', '').replace('%', '').replace(',', '').strip()
            try:
                return (1, float(val_clean))
            except ValueError:
                return (0, val_clean.lower())
                
        l.sort(key=lambda t: clean_val(t[0]), reverse=reverse)
        
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)
            
        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(header_frame, text="Home Battery & Tariff Optimizer", style="Header.TLabel").pack(side=tk.LEFT)
        
        workspace = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        workspace.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL ---
        left_panel = ttk.Frame(workspace, padding=(0, 0, 15, 0))
        workspace.add(left_panel, weight=1)
        
        # 1. Files
        files_frame = ttk.LabelFrame(left_panel, text=" 1. Input Source Files ", padding=10)
        files_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Warning Disclaimer: HDF profile baseline requirement
        lbl_warn = ttk.Label(files_frame, text="* WARNING: Works best with baseline (pre-battery) HDF files.\n  Existing battery storage/arbitraging will distort results.",
                             foreground="#ef4444", font=("Helvetica", 8, "italic"), justify=tk.LEFT)
        lbl_warn.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        
        lbl_hdf = ttk.Label(files_frame, text="ESB HDF:", font=("Helvetica", 9, "underline"), cursor="hand2")
        lbl_hdf.grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.hdf_path, width=25).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_hdf).grid(row=1, column=2, pady=2)
        ToolTip(lbl_hdf, "Smart meter readings in calculated 30-min kWh intervals from ESB Networks. Works best with un-metered/pre-battery baseline profiles.")
        
        lbl_tariff = ttk.Label(files_frame, text="Tariff DB:", font=("Helvetica", 9, "underline"), cursor="hand2")
        lbl_tariff.grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.tariff_path, width=25).grid(row=2, column=1, padx=2, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_tariff).grid(row=2, column=2, pady=2)
        ToolTip(lbl_tariff, "Tariff spreadsheet database. Rates can be downloaded from www.energypal.ie under the smartplans table (download button is at the bottom of the table).")

        lbl_dam = ttk.Label(files_frame, text="DAM Prices (Opt):", font=("Helvetica", 9, "underline"), cursor="hand2")
        lbl_dam.grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.dam_path, width=25).grid(row=3, column=1, padx=2, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_dam).grid(row=3, column=2, pady=2)
        ToolTip(lbl_dam, "Day-Ahead Market wholesale prices. Download reports from semopx.com (e.g. May 2026). Copy the first 3 columns of 'Auction_to' sheet as CSV.")

        lbl_dyn = ttk.Label(files_frame, text="Dyn Adders (Opt):", font=("Helvetica", 9, "underline"), cursor="hand2")
        lbl_dyn.grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.dynamic_adders_path, width=25).grid(row=4, column=1, padx=2, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_dyn).grid(row=4, column=2, pady=2)
        ToolTip(lbl_dyn, "Supplier standing charges and unit cost adjustments for dynamic wholesale tariffs (loaded from CSV).")
        
        ttk.Button(files_frame, text="+ Create Custom Tariff", style="Secondary.TButton", command=self.open_custom_tariff_dialog).grid(row=5, column=0, columnspan=3, pady=(5, 0), sticky=tk.EW)
        
        # 2. Hardware Config
        params_frame = ttk.LabelFrame(left_panel, text=" 2. Battery & Grid Hardware Configuration ", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="Capacity (kWh):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.entry_capacity = ttk.Entry(params_frame, width=8); self.entry_capacity.insert(0, "30.0"); self.entry_capacity.grid(row=0, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Usable Depth (%):").grid(row=0, column=2, sticky=tk.W, pady=4)
        self.entry_usable_pct = ttk.Entry(params_frame, width=8); self.entry_usable_pct.insert(0, "100"); self.entry_usable_pct.grid(row=0, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Chg Rate (kW):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.entry_charge_rate = ttk.Entry(params_frame, width=8); self.entry_charge_rate.insert(0, "15.0"); self.entry_charge_rate.grid(row=1, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Region:").grid(row=1, column=2, sticky=tk.W, pady=4)
        self.combo_region = ttk.Combobox(params_frame, values=["urban", "rural"], width=6, state="readonly"); self.combo_region.set("rural"); self.combo_region.grid(row=1, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Min SoC (%):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.entry_minsoc = ttk.Entry(params_frame, width=8); self.entry_minsoc.insert(0, "10"); self.entry_minsoc.grid(row=2, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Max SoC (%):").grid(row=2, column=2, sticky=tk.W, pady=4)
        self.entry_maxsoc = ttk.Entry(params_frame, width=8); self.entry_maxsoc.insert(0, "100"); self.entry_maxsoc.grid(row=2, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Import (MIC):").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.entry_mic = ttk.Entry(params_frame, width=8); self.entry_mic.insert(0, "18"); self.entry_mic.grid(row=3, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Export (MEC):").grid(row=3, column=2, sticky=tk.W, pady=4)
        self.entry_mec = ttk.Entry(params_frame, width=8); self.entry_mec.insert(0, "6"); self.entry_mec.grid(row=3, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Grid RTE (%):").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.entry_grid_eff = ttk.Entry(params_frame, width=8); self.entry_grid_eff.insert(0, "90"); self.entry_grid_eff.grid(row=4, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Solar RTE (%):").grid(row=4, column=2, sticky=tk.W, pady=4)
        self.entry_solar_eff = ttk.Entry(params_frame, width=8); self.entry_solar_eff.insert(0, "85"); self.entry_solar_eff.grid(row=4, column=3, sticky=tk.W, pady=4, padx=5)

        # 3. Strategy Explanations (Updated: Clean display summary row layout with fixed hover tooltips)
        explainer_frame = ttk.LabelFrame(left_panel, text=" 3. Charging Strategies (Hover for deep details) ", padding=10)
        explainer_frame.pack(fill=tk.X, pady=(0, 10))
        
        strategies_info = [
            ("• Self-Consumption", "Uses solar first; never charges from grid.", 
             "Prioritizes storing excess solar production locally. The battery will never charge using grid power, acting strictly as a solar sponge. Good baseline comparison strategy."),
            ("• Import-Minimiser", "Force-charges from grid during cheapest hours.", 
             "Force-charges the home battery system up to max capacity during the lowest cost daily tariff window to reliably cover your daytime load profiles."),
            ("• Export-Maximiser", "Dumps battery to grid before cheap hours.", 
             "Forces a proactive battery energy dump directly to the grid in the 4 hours immediately prior to the cheap grid window starting, clearing maximum volume space to collect cheap night power."),
            ("• Balanced-Export", "Arbitrages in summer; preserves power in winter.", 
             "Runs aggressive arbitrage/grid-dump protocols during spring and summer months, but bypasses the pre-charge grid dump during winter season (Nov-Feb) to ensure home heating security."),
            ("• Import-Minimiser (Summer Pass)", "Bypasses battery charging in summer.", 
             "Prevents solar generation from charging the battery between March and October to bypass AC/DC conversion rounds. Feeds solar to the grid at 100% efficiency instead.")
        ]
        
        for label_text, brief_text, tip_text in strategies_info:
            frame_row = ttk.Frame(explainer_frame)
            frame_row.pack(anchor=tk.W, pady=2, fill=tk.X)
            
            lbl_title = ttk.Label(frame_row, text=label_text, foreground="#4f46e5", font=("Helvetica", 9, "underline", "bold"), cursor="hand2")
            lbl_title.pack(side=tk.LEFT)
            
            lbl_brief = ttk.Label(frame_row, text=f" - {brief_text}", font=("Helvetica", 9, "normal"), foreground="#475569")
            lbl_brief.pack(side=tk.LEFT)
            
            # Bind tooltip triggers to both UI text objects
            ToolTip(lbl_title, tip_text)
            ToolTip(lbl_brief, tip_text)

        # 4. Telemetry Output (Scrollable Console Box Frame)
        self.stats_frame = ttk.LabelFrame(left_panel, text=" Engine Telemetry Console ", padding=5)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        console_container = ttk.Frame(self.stats_frame)
        console_container.pack(fill=tk.BOTH, expand=True)
        
        self.txt_stats = tk.Text(console_container, height=6, bg="#f8fafc", 
                                 font=("Consolas", 9), wrap=tk.WORD, bd=1, relief=tk.SOLID)
        self.txt_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt_stats.insert(tk.END, "Waiting for optimization sweep execution...")
        self.txt_stats.config(state=tk.DISABLED)
        
        console_scrollbar = ttk.Scrollbar(console_container, orient=tk.VERTICAL, command=self.txt_stats.yview)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_stats.config(yscrollcommand=console_scrollbar.set)

        ttk.Button(left_panel, text="Run Optimization Sweep", style="Action.TButton", command=self.run_sweep).pack(fill=tk.X, ipady=5)

        # --- RIGHT TABS PANEL ---
        self.right_notebook = ttk.Notebook(workspace)
        workspace.add(self.right_notebook, weight=3)

        # Tab 1: Rankings
        tab_rankings = ttk.Frame(self.right_notebook, padding=10)
        self.right_notebook.add(tab_rankings, text="  Leaderboard Rankings  ")

        table_title_frame = ttk.Frame(tab_rankings)
        table_title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(table_title_frame, text="Tariff Leadership Rankings", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self.lbl_savings = ttk.Label(table_title_frame, text="", font=("Helvetica", 11, "bold"), foreground="#059669")
        self.lbl_savings.pack(side=tk.LEFT, padx=(30, 0))
        
        btn_export_csv = ttk.Button(table_title_frame, text="⬇ Export Table to CSV", command=self.export_leaderboard)
        btn_export_csv.pack(side=tk.RIGHT, padx=10)

        table_frame = ttk.Frame(tab_rankings)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("rank", "supplier", "tariff", "strategy", "arbitrage", "imp_kwh", "exp_kwh", "import", "export", "june", "dec", "fixed", "bill")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="none")
        
        self.tree.heading("rank", text="#")
        self.tree.heading("supplier", text="Supplier")
        self.tree.heading("tariff", text="Tariff Name")
        self.tree.heading("strategy", text="Winning Strategy")
        self.tree.heading("arbitrage", text="Arb. Return")
        self.tree.heading("imp_kwh", text="Imp (kWh)")
        self.tree.heading("exp_kwh", text="Exp (kWh)")
        self.tree.heading("import", text="Import Cost")
        self.tree.heading("export", text="Export FIT")
        self.tree.heading("june", text="June (€)")
        self.tree.heading("dec", text="Dec (€)")
        self.tree.heading("fixed", text="Fixed (€)")
        self.tree.heading("bill", text="Annual Bill (€)")

        self.tree.column("rank", width=30, anchor=tk.CENTER)
        self.tree.column("supplier", width=100, anchor=tk.W)
        self.tree.column("tariff", width=190, anchor=tk.W)
        self.tree.column("strategy", width=150, anchor=tk.CENTER)
        self.tree.column("arbitrage", width=100, anchor=tk.CENTER)
        self.tree.column("imp_kwh", width=75, anchor=tk.E)
        self.tree.column("exp_kwh", width=75, anchor=tk.E)
        self.tree.column("import", width=80, anchor=tk.E)
        self.tree.column("export", width=80, anchor=tk.E)
        self.tree.column("june", width=65, anchor=tk.E)
        self.tree.column("dec", width=65, anchor=tk.E)
        self.tree.column("fixed", width=65, anchor=tk.E)
        self.tree.column("bill", width=100, anchor=tk.E)

        self.tree.tag_configure('best_baseline', background='#ffedd5', foreground='#b45309')

        # Bind sorting to columns
        for col in cols:
            self.tree.heading(col, text=self.tree.heading(col, 'text'), 
                              command=lambda _col=col: self.treeview_sort_column(self.tree, _col, False))

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: HDF Profile
        self.tab_visualizer = ttk.Frame(self.right_notebook, padding=10)
        self.right_notebook.add(self.tab_visualizer, text="  HDF Base Profile  ")
        
        hdf_ctrl_frame = ttk.Frame(self.tab_visualizer)
        hdf_ctrl_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(hdf_ctrl_frame, text="View Average Daily Profile for: ").pack(side=tk.LEFT)
        self.hdf_month_combo = ttk.Combobox(hdf_ctrl_frame, values=["All Year"] + MONTH_NAMES, state="readonly", width=15)
        self.hdf_month_combo.set("All Year"); self.hdf_month_combo.pack(side=tk.LEFT)
        self.hdf_month_combo.bind("<<ComboboxSelected>>", self.update_hdf_graph)

        self.graph_container = ttk.Frame(self.tab_visualizer)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        if HAS_MATPLOTLIB:
            self.fig_hdf = Figure(figsize=(6, 4), dpi=100); self.ax_hdf = self.fig_hdf.add_subplot(111)
            self.canvas_hdf = FigureCanvasTkAgg(self.fig_hdf, master=self.graph_container)
            self.canvas_hdf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tabs 3, 4, 5, 6: Daily Viewers
        self.top_tabs = []
        for i in range(1, 4):
            frame = ttk.Frame(self.right_notebook, padding=10)
            self.right_notebook.add(frame, text=f"  Top {i}  ")
            self.setup_daily_tab(frame, str(i))
            
        frame_dyn = ttk.Frame(self.right_notebook, padding=10)
        self.right_notebook.add(frame_dyn, text="  Top Dynamic  ")
        self.setup_daily_tab(frame_dyn, "Dynamic")

    def setup_daily_tab(self, parent, tab_id):
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_frame = ttk.Frame(nav_frame)
        title_frame.pack(fill=tk.X)
        
        lbl_info = ttk.Label(title_frame, text=f"Run sweep to populate Rank {tab_id}", font=("Helvetica", 11, "bold"), foreground="#4f46e5")
        lbl_info.pack(side=tk.LEFT, pady=(0, 10))
        
        idx = len(self.top_tabs)
        btn_export_hdf = ttk.Button(title_frame, text="⬇ Export Simulated HDF", command=lambda local_idx=idx: self.export_simulated_hdf(local_idx))
        btn_export_hdf.pack(side=tk.RIGHT, pady=(0, 10))
        
        ctrl_subframe = ttk.Frame(nav_frame)
        ctrl_subframe.pack(fill=tk.X)
        
        ttk.Button(ctrl_subframe, text="< Prev Day", command=lambda: self.change_day(-1)).pack(side=tk.LEFT)
        lbl_date = ttk.Label(ctrl_subframe, text="[Date]", font=("Helvetica", 10, "bold"))
        lbl_date.pack(side=tk.LEFT, padx=15)
        
        ttk.Label(ctrl_subframe, text="Jump to Month:").pack(side=tk.LEFT, padx=(20, 5))
        combo_month = ttk.Combobox(ctrl_subframe, values=MONTH_NAMES, state="readonly", width=12)
        combo_month.pack(side=tk.LEFT)
        combo_month.bind("<<ComboboxSelected>>", lambda e: self.jump_to_month(combo_month.get()))
        ttk.Button(ctrl_subframe, text="Next Day >", command=lambda: self.change_day(1)).pack(side=tk.RIGHT)
        
        graph_frame = ttk.Frame(parent)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        if HAS_MATPLOTLIB:
            fig = Figure(figsize=(8, 5), dpi=100); ax1 = fig.add_subplot(111); ax2 = ax1.twinx()
            canvas = FigureCanvasTkAgg(fig, master=graph_frame); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.top_tabs.append({'frame': parent, 'lbl_info': lbl_info, 'lbl_date': lbl_date, 'combo_month': combo_month, 
                                  'fig': fig, 'ax1': ax1, 'ax2': ax2, 'canvas': canvas, 'internal_id': None, 'strategy': None})

    def update_console(self, text_string, color_hex="#334155"):
        self.txt_stats.config(state=tk.NORMAL)
        self.txt_stats.delete("1.0", tk.END)
        self.txt_stats.insert(tk.END, text_string)
        self.txt_stats.tag_add("color_tag", "1.0", tk.END)
        self.txt_stats.tag_config("color_tag", foreground=color_hex)
        self.txt_stats.config(state=tk.DISABLED)
        self.root.update()

    # ----------------- LOGIC & UPDATES -----------------

    def browse_hdf(self):
        f = filedialog.askopenfilename(filetypes=[("HDF CSV", "*.csv")]); 
        if f: self.hdf_path.set(f)
    def browse_tariff(self):
        f = filedialog.askopenfilename(filetypes=[("Tariff DB", "*.csv")]); 
        if f: self.tariff_path.set(f)
    def browse_dam(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if f: self.dam_path.set(f)
    def browse_dyn(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if f: self.dynamic_adders_path.set(f)

    def export_leaderboard(self):
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            messagebox.showwarning("Warning", "No simulation data available to export. Please run a sweep first.")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Results Table")
        if filepath:
            try:
                export_df = self.leaderboard_data.copy()
                export_df = export_df.drop(columns=['_id', 'is_dynamic']) 
                export_df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Leaderboard exported successfully to:\n{os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data:\n{str(e)}")

    def export_simulated_hdf(self, local_idx):
        if self.df_hdf is None or not self.top_tabs:
            messagebox.showwarning("Warning", "No simulation data available. Please run a sweep first.")
            return
            
        tab_ui = self.top_tabs[local_idx]
        tid = tab_ui['internal_id']
        strategy = tab_ui['strategy']
        
        if not tid or not strategy:
            messagebox.showwarning("Warning", "No results mapped to this tab yet. Run a sweep first.")
            return
            
        sim_data = self.detailed_results.get(tid, {}).get(strategy)
        if sim_data is None:
            messagebox.showerror("Error", "Simulated data not found for this strategy.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title=f"Save Simulated HDF - {strategy.replace('-', ' ').title()}",
            initialfile=f"simulated_hdf_{strategy}.csv"
        )
        if not filepath:
            return
            
        try:
            end_times = self.df_hdf.index + pd.Timedelta(minutes=30)
            formatted_times = end_times.strftime('%d-%m-%Y %H:%M')
            
            import_vals = sim_data['import']
            export_vals = sim_data['export']
            
            mprn_col = str(self.mprn) if self.mprn else "12345678912"
            meter_col = str(self.meter_serial) if self.meter_serial else "SIMULATED_METER"
            
            rows = []
            for t, imp, exp in zip(formatted_times, import_vals, export_vals):
                rows.append([mprn_col, meter_col, f"{imp:.4f}", "Active Import Interval Value", t])
                rows.append([mprn_col, meter_col, f"{exp:.4f}", "Active Export Interval Value", t])
                
            df_export = pd.DataFrame(rows, columns=['MPRN', 'Meter Serial Number', 'Read Value', 'Read Type', 'Read Date and End Time'])
            
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(f"MPRN,Meter Serial Number,Read Value,Read Type,Read Date and End Time\n")
                df_export.to_csv(f, index=False, header=False)
                
            messagebox.showinfo("Success", f"Simulated HDF exported successfully to:\n{os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export simulated HDF:\n{str(e)}")
 
    def open_custom_tariff_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Custom Tariff"); dlg.geometry("480x480"); dlg.grab_set() 
        frame = ttk.Frame(dlg, padding=15); frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Supplier Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ent_sup = ttk.Entry(frame, width=20); ent_sup.insert(0, "Custom Energy"); ent_sup.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Tariff Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ent_name = ttk.Entry(frame, width=20); ent_name.insert(0, "My Custom Plan"); ent_name.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Plan Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        combo_type = ttk.Combobox(frame, values=["smart", "day/night", "24h"], state="readonly", width=17)
        combo_type.set("smart"); combo_type.grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Standing Charge (€/yr):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ent_sc = ttk.Entry(frame, width=10); ent_sc.insert(0, "300"); ent_sc.grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Day Unit (c/kWh):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ent_day = ttk.Entry(frame, width=10); ent_day.insert(0, "35.0"); ent_day.grid(row=4, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Night Unit (c/kWh):").grid(row=5, column=0, sticky=tk.W, pady=5)
        ent_night = ttk.Entry(frame, width=10); ent_night.insert(0, "20.0"); ent_night.grid(row=5, column=1, sticky=tk.W, pady=5)
        ttk.Label(frame, text="Peak Unit (c/kWh):").grid(row=6, column=0, sticky=tk.W, pady=5)
        ent_peak = ttk.Entry(frame, width=10); ent_peak.insert(0, "45.0"); ent_peak.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(frame, text="EV/Boost Unit (c/kWh):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ent_ev = ttk.Entry(frame, width=10); ent_ev.insert(0, "10.0"); ent_ev.grid(row=7, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(frame, text="EV Overage (c/kWh):").grid(row=7, column=2, sticky=tk.W, pady=5, padx=5)
        ent_ev_overage = ttk.Entry(frame, width=8); ent_ev_overage.insert(0, "35.0"); ent_ev_overage.grid(row=7, column=3, sticky=tk.W, pady=5)

        ttk.Label(frame, text="Export FIT (c/kWh):").grid(row=8, column=0, sticky=tk.W, pady=5)
        ent_fit = ttk.Entry(frame, width=10); ent_fit.insert(0, "18.0"); ent_fit.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        ev_frame = ttk.Frame(frame); ev_frame.grid(row=9, column=0, columnspan=4, sticky=tk.W, pady=10)
        ttk.Label(ev_frame, text="EV Start Hour (0-23):").pack(side=tk.LEFT)
        ent_ev_start = ttk.Entry(ev_frame, width=4); ent_ev_start.insert(0, "2"); ent_ev_start.pack(side=tk.LEFT, padx=5)
        ttk.Label(ev_frame, text="End Hour:").pack(side=tk.LEFT)
        ent_ev_end = ttk.Entry(ev_frame, width=4); ent_ev_end.insert(0, "5"); ent_ev_end.pack(side=tk.LEFT, padx=5)

        def save_tariff():
            try:
                new_tariff = {
                    'Supplier': ent_sup.get().strip(), 'Tariff name': ent_name.get().strip() + " (Custom)", 'Plan type': combo_type.get(),
                    'Standing charge': float(ent_sc.get() or 0.0), 'PSO Levy': 0.0, 'Cash bonus': 0.0, 'Day unit': float(ent_day.get() or 0.0),
                    'Night unit': float(ent_night.get() or 0.0), 'Peak unit': float(ent_peak.get() or 0.0), 'Ev unit': float(ent_ev.get() or 0.0),
                    'Ev overage unit': float(ent_ev_overage.get() or 0.0), 'Fit unit': float(ent_fit.get() or 0.0), 
                    'Supply Region': self.combo_region.get().strip().lower(), 
                    'Extra': f'["ev_{int(ent_ev_start.get())}_{int(ent_ev_end.get())}"]' if ent_ev.get() else ""
                }
                self.custom_tariffs.append(new_tariff)
                messagebox.showinfo("Success", f"Added Custom Tariff: {new_tariff['Tariff name']}\nIncluded in next sweep.")
                dlg.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please ensure all rates and hours are valid numbers.", parent=dlg)
        ttk.Button(frame, text="Save & Add to Database", command=save_tariff).grid(row=10, column=0, columnspan=4, pady=15, sticky=tk.EW)

    def run_sweep(self):
        if not self.hdf_path.get() or (not self.tariff_path.get() and not self.custom_tariffs):
            messagebox.showerror("Error", "Please select an HDF file and a Tariff DB (or Custom Tariff).")
            return

        try:
            params = {
                'capacity': float(self.entry_capacity.get()), 'usable_pct': float(self.entry_usable_pct.get()),
                'charge_rate': float(self.entry_charge_rate.get()), 'grid_efficiency': float(self.entry_grid_eff.get()),
                'solar_efficiency': float(self.entry_solar_eff.get()), 'min_soc': float(self.entry_minsoc.get()),
                'max_soc': float(self.entry_maxsoc.get()), 'mic': float(self.entry_mic.get()),
                'mec': float(self.entry_mec.get()), 'region': self.combo_region.get().strip().lower()
            }
        except ValueError:
            messagebox.showerror("Error", "Check numeric parameters."); return

        self.update_console("Parsing Input Data & Pre-compiling Engine Tracks...", "#f59e0b")

        try:
            start_time = time.time()
            raw_hdf, mprn_val, meter_val = parse_hdf(self.hdf_path.get().strip())
            self.df_hdf = filter_last_12_full_months(raw_hdf)
            self.mprn, self.meter_serial = mprn_val, meter_val
            
            if self.df_hdf.empty: raise ValueError("No valid data left after filtering.")
            
            self.unique_dates = np.unique(self.df_hdf.index.date); self.current_date_idx = 0
            self.detailed_results.clear(); self.update_hdf_graph() 

            df_tariffs = pd.read_csv(self.tariff_path.get().strip()) if self.tariff_path.get() else pd.DataFrame()
            df_tariffs.columns = df_tariffs.columns.str.strip() if not df_tariffs.empty else []

            if self.custom_tariffs:
                # Update the region of custom tariffs to match current parameters
                for t in self.custom_tariffs:
                    t['Supply Region'] = params['region']
                df_tariffs = pd.concat([df_tariffs, pd.DataFrame(self.custom_tariffs)], ignore_index=True)

            valid_tariffs = df_tariffs[(df_tariffs['Supply Region'].str.lower() == params['region']) & (df_tariffs['Plan type'].str.lower() != 'gas')]

            dam_prices_c_kwh, dynamic_suppliers = None, []
            if self.dam_path.get() and self.dynamic_adders_path.get():
                try:
                    dam_prices_c_kwh = prepare_dam(self.df_hdf.index, self.dam_path.get().strip())
                    dynamic_suppliers = parse_dynamic_suppliers(self.dynamic_adders_path.get().strip(), params['region'])
                except Exception as e:
                    messagebox.showwarning("Dynamic Pricing Skipped", f"Could not load dynamic files. Skipping dynamic analysis.\n{e}")

            results = []; int_id = 0
            total_rows = len(self.df_hdf)
            num_tariffs = len(valid_tariffs) + len(dynamic_suppliers)
            
            orig_imports = self.df_hdf['consumption'].values
            orig_exports = self.df_hdf['generation'].values
            months_array = self.df_hdf.index.month.values
            
            mask_june = (months_array == 6)
            mask_dec = (months_array == 12)

            num_days = len(self.unique_dates)
            scaling_factor = 365.0 / num_days if num_days > 0 else 1.0
            is_short_duration = num_days < 330
            exceeded_plans = []

            all_strategies = ['self-consumption', 'import-minimiser', 'export-maximiser', 'balanced-export-maximiser', 'import-minimiser-summer-pass']

            # 1. Standard Sweep
            for _, row in valid_tariffs.iterrows():
                fit_rate = float(row['Fit unit']) / 100.0 if not pd.isna(row.get('Fit unit')) else 0.18
                import_prices, is_ev_window, ev_overage_rate, has_overage_penalty = get_half_hourly_rates_for_row(row, self.df_hdf.index)
                first_day_prices = import_prices.iloc[:48]
                hourly_prices = first_day_prices.groupby(first_day_prices.index.hour).first()
                force_charge_hours = [hourly_prices.get(h, 99.0) <= hourly_prices.min() + 0.001 for h in range(24)]
                
                tid = f"T_{int_id}"; int_id += 1
                self.detailed_results[tid] = {'meta': row.to_dict()}
                fixed_charges = float(row['Standing charge']) + float(row.get('PSO Levy', 0)) - (float(row.get('Cash bonus', 0)) if not pd.isna(row.get('Cash bonus')) else 0.0)
                monthly_fixed = fixed_charges / 12.0

                baseline_import_costs, base_limit_exceeded = _calc_cost_with_overage(orig_imports, import_prices.values, is_ev_window, ev_overage_rate, months_array, has_overage_penalty)
                annual_imp_base = np.sum(baseline_import_costs)
                annual_exp_base = np.sum(orig_exports * fit_rate)
                net_bill_base = (annual_imp_base - annual_exp_base) * scaling_factor + fixed_charges
                
                base_imp_kwh = np.sum(orig_imports)
                base_exp_kwh = np.sum(orig_exports)
                
                base_june = np.sum(baseline_import_costs[mask_june]) - np.sum(orig_exports[mask_june] * fit_rate) + monthly_fixed
                base_dec = np.sum(baseline_import_costs[mask_dec]) - np.sum(orig_exports[mask_dec] * fit_rate) + monthly_fixed
                
                self.detailed_results[tid]['baseline-no-battery'] = {'import': orig_imports, 'export': orig_exports, 'soc': np.zeros(len(orig_imports))}
                results.append({
                    'Supplier': row['Supplier'], 'Tariff': row['Tariff name'], 'Strategy': 'baseline-no-battery', 
                    'Arbitrage': "N/A", 'Imp_kWh': base_imp_kwh, 'Exp_kWh': base_exp_kwh,
                    'Import': annual_imp_base, 'Export': annual_exp_base, 
                    'June': base_june, 'Dec': base_dec, 'Fixed': fixed_charges,
                    'Bill': net_bill_base, '_id': tid, 'is_dynamic': False
                })

                for strategy in all_strategies:
                    imports, exports, soc, is_arb = run_simulation(self.df_hdf, import_prices, fit_rate, strategy, force_charge_hours, params)
                    self.detailed_results[tid][strategy] = {'import': imports, 'export': exports, 'soc': soc}
                    
                    strategy_import_costs, strat_limit_exceeded = _calc_cost_with_overage(imports, import_prices.values, is_ev_window, ev_overage_rate, months_array, has_overage_penalty)
                    if strat_limit_exceeded:
                        exceeded_plans.append(f"{row['Supplier']} {row['Tariff name']} ({strategy})")
                        
                    annual_imp_cost = np.sum(strategy_import_costs)
                    annual_exp_rev = np.sum(exports * fit_rate)
                    net_bill = (annual_imp_cost - annual_exp_rev) * scaling_factor + fixed_charges
                    
                    strat_imp_kwh = np.sum(imports)
                    strat_exp_kwh = np.sum(exports)
                    
                    strat_june = np.sum(strategy_import_costs[mask_june]) - np.sum(exports[mask_june] * fit_rate) + monthly_fixed
                    strat_dec = np.sum(strategy_import_costs[mask_dec]) - np.sum(exports[mask_dec] * fit_rate) + monthly_fixed
                    
                    arb_display = "N/A"
                    if strategy not in ['baseline-no-battery', 'self-consumption'] and is_arb is not None:
                        if is_arb > 0:
                            arb_display = f"{is_arb:.2f} c/kWh"
                        else:
                            arb_display = "N/A"
                            
                    results.append({
                        'Supplier': row['Supplier'], 'Tariff': row['Tariff name'], 'Strategy': strategy, 'Arbitrage': arb_display, 
                        'Imp_kWh': strat_imp_kwh, 'Exp_kWh': strat_exp_kwh,
                        'Import': annual_imp_cost, 'Export': annual_exp_rev, 
                        'June': strat_june, 'Dec': strat_dec, 'Fixed': fixed_charges,
                        'Bill': net_bill, '_id': tid, 'is_dynamic': False
                    })

            # 2. Dynamic Sweep
            for dyn in dynamic_suppliers:
                fit_rate = dyn['Fit unit'] / 100.0
                fixed_charges = dyn['Standing charge'] * 1.09  # Add 9% VAT
                monthly_fixed = fixed_charges / 12.0
                
                prices = dam_prices_c_kwh.copy()
                hour = self.df_hdf.index.hour
                is_night = (hour >= 23) | (hour < 8); is_peak = (hour >= 17) & (hour < 19); is_day = ~(is_night | is_peak)
                prices[is_night] += dyn['Night']; prices[is_day] += dyn['Day']; prices[is_peak] += dyn['Peak']
                import_prices = pd.Series(prices / 100.0, index=self.df_hdf.index)
                import_prices = import_prices * 1.09  
                
                dyn_is_ev_window = np.zeros(len(self.df_hdf), dtype=np.bool_)
                dyn_ev_overage_rate = 0.0
                
                tid = f"T_{int_id}"; int_id += 1
                self.detailed_results[tid] = {'meta': dyn}
                
                baseline_import_costs, base_limit_exceeded = _calc_cost_with_overage(orig_imports, import_prices.values, dyn_is_ev_window, dyn_ev_overage_rate, months_array, False)
                annual_imp_base = np.sum(baseline_import_costs)
                annual_exp_base = np.sum(orig_exports * fit_rate)
                net_bill_base = (annual_imp_base - annual_exp_base) * scaling_factor + fixed_charges
                
                base_imp_kwh = np.sum(orig_imports)
                base_exp_kwh = np.sum(orig_exports)
                
                base_june = np.sum(baseline_import_costs[mask_june]) - np.sum(orig_exports[mask_june] * fit_rate) + monthly_fixed
                base_dec = np.sum(baseline_import_costs[mask_dec]) - np.sum(orig_exports[mask_dec] * fit_rate) + monthly_fixed
                
                self.detailed_results[tid]['baseline-no-battery'] = {'import': orig_imports, 'export': orig_exports, 'soc': np.zeros(len(orig_imports))}
                results.append({
                    'Supplier': dyn['Supplier'], 'Tariff': dyn['Tariff name'], 'Strategy': 'baseline-no-battery', 
                    'Arbitrage': "N/A", 'Imp_kWh': base_imp_kwh, 'Exp_kWh': base_exp_kwh,
                    'Import': annual_imp_base, 'Export': annual_exp_base, 
                    'June': base_june, 'Dec': base_dec, 'Fixed': fixed_charges,
                    'Bill': net_bill_base, '_id': tid, 'is_dynamic': True
                })

                for strategy in all_strategies:
                    imports, exports, soc, is_arb = run_dynamic_simulation(self.df_hdf, import_prices, fit_rate, strategy, params)
                    self.detailed_results[tid][strategy] = {'import': imports, 'export': exports, 'soc': soc}
                    
                    strategy_import_costs, strat_limit_exceeded = _calc_cost_with_overage(imports, import_prices.values, dyn_is_ev_window, dyn_ev_overage_rate, months_array, False)
                    if strat_limit_exceeded:
                        exceeded_plans.append(f"{dyn['Supplier']} {dyn['Tariff name']} ({strategy})")
                        
                    annual_imp_cost = np.sum(strategy_import_costs)
                    annual_exp_rev = np.sum(exports * fit_rate)
                    net_bill = (annual_imp_cost - annual_exp_rev) * scaling_factor + fixed_charges
                    
                    strat_imp_kwh = np.sum(imports)
                    strat_exp_kwh = np.sum(exports)
                    
                    strat_june = np.sum(strategy_import_costs[mask_june]) - np.sum(exports[mask_june] * fit_rate) + monthly_fixed
                    strat_dec = np.sum(strategy_import_costs[mask_dec]) - np.sum(exports[mask_dec] * fit_rate) + monthly_fixed
                    
                    arb_display = "N/A"
                    if strategy not in ['baseline-no-battery', 'self-consumption'] and is_arb is not None:
                        if is_arb > 0:
                            arb_display = f"{is_arb:.2f} c/kWh"
                        else:
                            arb_display = "N/A"
                            
                    results.append({
                        'Supplier': dyn['Supplier'], 'Tariff': dyn['Tariff name'], 'Strategy': strategy, 'Arbitrage': arb_display, 
                        'Imp_kWh': strat_imp_kwh, 'Exp_kWh': strat_exp_kwh,
                        'Import': annual_imp_cost, 'Export': annual_exp_rev, 
                        'June': strat_june, 'Dec': strat_dec, 'Fixed': fixed_charges,
                        'Bill': net_bill, '_id': tid, 'is_dynamic': True
                    })

            calc_time = time.time() - start_time
            df_res = pd.DataFrame(results)
            
            total_sims = num_tariffs * 6
            total_steps = total_rows * total_sims
            mem_usage_kb = df_res.memory_usage(deep=True).sum() / 1024.0 if not df_res.empty else 0.0
            
            telemetry = (
                f"[✓] Data Points: {total_rows:,} ({num_days} days)\n"
                f"[✓] Tariffs Evaluated: {num_tariffs}\n"
                f"[✓] Total Simulations: {total_sims:,} runs\n"
                f"⚡ Iterations Computed: {total_steps:,} steps\n"
                f"⏱️ CPU Exec Time: {calc_time:.4f} seconds\n"
                f"📊 Data Frame Memory: {mem_usage_kb:.1f} KB"
            )
            if is_short_duration:
                telemetry += f"\n⚠️ Short Data Warning: {num_days} days scaled by {scaling_factor:.2f}x to simulate annualized totals."
            
            ev_exceeded_names = sorted(list(set([p.split(' (')[0] for p in exceeded_plans])))
            if ev_exceeded_names:
                telemetry += f"\n⚠️ EV Policy Cap Exceeded: {', '.join(ev_exceeded_names[:2])} (Calculated via 1,000 kWh bi-monthly overage rule thresholds)."
                    
            self.update_console(telemetry, "#10b981")

            baseline_mask = df_res['Strategy'] == 'baseline-no-battery'
            
            best_base_bill = df_res[baseline_mask]['Bill'].min() if not df_res[baseline_mask].empty else 0
            best_opt_bill = df_res[~baseline_mask]['Bill'].min() if not df_res[~baseline_mask].empty else 0
            self.lbl_savings.config(text=f"💰 System Optimization Savings: €{best_base_bill - best_opt_bill:,.2f} / yr  |  Cheapest Unoptimized Plan: €{best_base_bill:,.2f}")

            best_baseline_row = df_res[baseline_mask].loc[df_res[baseline_mask]['Bill'].idxmin()] if not df_res[baseline_mask].empty else pd.DataFrame()
            self.leaderboard_data = pd.concat([df_res[~baseline_mask].copy(), pd.DataFrame([best_baseline_row])]).sort_values(by='Bill').reset_index(drop=True)
            
            for item in self.tree.get_children(): self.tree.delete(item)
            for idx, row in self.leaderboard_data.iterrows():
                tags = ('best_baseline',) if row['Strategy'] == 'baseline-no-battery' else ()
                self.tree.insert("", "end", values=(
                    idx + 1, row['Supplier'], row['Tariff'], row['Strategy'].replace('-', ' ').title(), row['Arbitrage'], 
                    f"{row['Imp_kWh']:,.0f}", f"{row['Exp_kWh']:,.0f}", 
                    f"€ {row['Import']:,.2f}", f"€ {row['Export']:,.2f}",
                    f"€ {row['June']:,.2f}", f"€ {row['Dec']:,.2f}", f"€ {row['Fixed']:,.2f}",
                    f"€ {row['Bill']:,.2f}"
                ), tags=tags)

            if HAS_MATPLOTLIB:
                top_3 = df_res[~baseline_mask].sort_values(by='Bill').head(3).reset_index(drop=True)
                for i, (_, row) in enumerate(top_3.iterrows()):
                    if i >= len(self.top_tabs): break
                    tab_ui = self.top_tabs[i]; tab_ui['internal_id'] = row['_id']; tab_ui['strategy'] = row['Strategy']
                    self.right_notebook.tab(tab_ui['frame'], text=f"  #{i+1}: {row['Supplier']}  ")
                    tab_ui['lbl_info'].config(text=f"{i+1}. {row['Supplier']} - {row['Tariff']}\nWinning Strategy: {row['Strategy'].replace('-', ' ').title()}")
                
                df_dynamic = df_res[(df_res.get('is_dynamic', False) == True) & (~baseline_mask)]
                if not df_dynamic.empty:
                    self.right_notebook.tab(self.top_tabs[3]['frame'], state='normal')
                    best_dyn = df_dynamic.sort_values(by='Bill').iloc[0]
                    tab_ui = self.top_tabs[3]; tab_ui['internal_id'] = best_dyn['_id']; tab_ui['strategy'] = best_dyn['Strategy']
                    self.right_notebook.tab(tab_ui['frame'], text=f"  Dyn: {best_dyn['Supplier']}  ")
                    tab_ui['lbl_info'].config(text=f"Top Dynamic: {best_dyn['Supplier']} - {best_dyn['Tariff']}\nWinning Strategy: {best_dyn['Strategy'].replace('-', ' ').title()}")
                else:
                    self.right_notebook.tab(self.top_tabs[3]['frame'], state='hidden')
                    
                self.update_daily_charts()
            messagebox.showinfo("Success", "Sweep complete! Check the visualizer tabs.")

        except Exception as e: 
            messagebox.showerror("Error", str(e))
            self.update_console("Simulation Failure occurred during analysis loop execution.", "#ef4444")

    def update_hdf_graph(self, event=None):
        if not HAS_MATPLOTLIB or self.df_hdf is None: return
        month_sel = self.hdf_month_combo.get(); df_target = self.df_hdf
        if month_sel != "All Year":
            df_target = self.df_hdf[self.df_hdf.index.month == (MONTH_NAMES.index(month_sel) + 1)]
            if df_target.empty: return

        hourly_avg = df_target.groupby(df_target.index.hour).mean() * 2.0
        self.ax_hdf.clear()
        self.ax_hdf.plot(hourly_avg.index, hourly_avg['consumption'], label="Avg Grid Import (kW)", color="#4f46e5", linewidth=2.5)
        self.ax_hdf.plot(hourly_avg.index, hourly_avg['generation'], label="Avg Grid Export (kW)", color="#10b981", linewidth=2.5)
        self.ax_hdf.set_title(f"Average Load Profile: {month_sel}", fontsize=11, fontweight="bold")
        self.ax_hdf.set_xlabel("Hour"); self.ax_hdf.set_ylabel("Power (kW)")
        self.ax_hdf.set_xticks(range(0, 24, 2)); self.ax_hdf.grid(True, linestyle="--", alpha=0.5); self.ax_hdf.legend()
        self.fig_hdf.tight_layout(); self.canvas_hdf.draw()

    def change_day(self, delta):
        if not len(self.unique_dates): return
        self.current_date_idx = (self.current_date_idx + delta) % len(self.unique_dates)
        self.update_daily_charts()

    def jump_to_month(self, month_name):
        if not len(self.unique_dates): return
        m_idx = MONTH_NAMES.index(month_name) + 1
        for i, dt in enumerate(self.unique_dates):
            if dt.month == m_idx:
                self.current_date_idx = i; self.update_daily_charts()
                return

    def update_daily_charts(self):
        if not HAS_MATPLOTLIB or self.df_hdf is None: return
        target_date = self.unique_dates[self.current_date_idx]
        mask = (self.df_hdf.index.date == target_date)
        hours = self.df_hdf.index[mask].hour + self.df_hdf.index[mask].minute / 60.0
        
        orig_imp = self.df_hdf['consumption'].values[mask] * 2.0
        orig_exp = self.df_hdf['generation'].values[mask] * 2.0

        for tab_ui in self.top_tabs:
            if not tab_ui['internal_id']: continue
            tab_ui['lbl_date'].config(text=target_date.strftime("%A, %d %b %Y"))
            tab_ui['combo_month'].set(target_date.strftime("%B"))
            tab_ui['ax1'].clear(); tab_ui['ax2'].clear()
            
            sim_data = self.detailed_results[tab_ui['internal_id']][tab_ui['strategy']]
            tab_ui['ax1'].plot(hours, orig_imp, color="gray", linestyle="--", alpha=0.6, label="Orig. House Load")
            tab_ui['ax1'].plot(hours, orig_exp, color="lightgreen", linestyle="--", alpha=0.6, label="Orig. Solar Export")
            tab_ui['ax1'].plot(hours, sim_data['import'][mask] * 2.0, color="#ef4444", linewidth=2, label="Rev. Grid Import")
            tab_ui['ax1'].plot(hours, sim_data['export'][mask] * 2.0, color="#10b981", linewidth=2, label="Rev. Grid Export")
            
            tab_ui['ax2'].fill_between(hours, 0, sim_data['soc'][mask], color="#f59e0b", alpha=0.15)
            tab_ui['ax2'].plot(hours, sim_data['soc'][mask], color="#f59e0b", linewidth=1.5, label="Battery SoC (%)")
            
            tab_ui['ax1'].set_ylabel("Power (kW)"); tab_ui['ax2'].set_ylabel("SoC (%)"); tab_ui['ax2'].set_ylim(0, 105)
            tab_ui['ax1'].set_xticks(range(0, 25, 2)); tab_ui['ax1'].grid(True, linestyle=":", alpha=0.7)
            
            l1, lab1 = tab_ui['ax1'].get_legend_handles_labels(); l2, lab2 = tab_ui['ax2'].get_legend_handles_labels()
            tab_ui['ax1'].legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
            tab_ui['fig'].tight_layout(); tab_ui['canvas'].draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HomeBatteryCalculatorApp(root)
    root.mainloop()