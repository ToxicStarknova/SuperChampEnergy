import os
import re
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    
    # Extract Original MPRN and Meter Serial for Synthetic Generation Later
    mprn_col = next((c for c in df_raw.columns if 'mprn' in c.lower()), None)
    meter_col = next((c for c in df_raw.columns if 'serial' in c.lower()), None)
    mprn_val = str(df_raw[mprn_col].dropna().iloc[0]) if mprn_col and not df_raw[mprn_col].empty else "00000000000"
    meter_val = str(df_raw[meter_col].dropna().iloc[0]) if meter_col and not df_raw[meter_col].empty else "00000000"

    date_col = next(c for c in df_raw.columns if 'read date' in c.lower())
    type_col = next(c for c in df_raw.columns if 'read type' in c.lower())
    val_col = next((c for c in df_raw.columns if 'read val' in c.lower()), None)
    
    # Fast Pandas Vectorized Datetime parsing
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
        
    return df.sort_index(), mprn_val, meter_val

def filter_last_12_full_months(df):
    if df.empty: return df
    latest_ts = df.index[-1]
    end_date = datetime(latest_ts.year, latest_ts.month, 1)
    start_date = datetime(end_date.year - 1, end_date.month, 1)
    return df[(df.index >= start_date) & (df.index < end_date)]

def get_half_hourly_rates_for_row(row, date_range):
    plan_type = row['Plan type']
    day_rate = float(row['Day unit']) / 100.0
    peak_rate = float(row['Peak unit']) / 100.0 if not pd.isna(row['Peak unit']) else day_rate
    night_rate = float(row['Night unit']) / 100.0 if not pd.isna(row['Night unit']) else day_rate
    ev_rate = float(row['Ev unit']) / 100.0 if not pd.isna(row['Ev unit']) else None
    
    extra_tags = []
    if not pd.isna(row['Extra']):
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
    for dt in date_range:
        hour, is_weekend = dt.hour, dt.weekday() >= 5
        if plan_type == '24h': prices.append(day_rate)
        elif plan_type == 'day/night':
            prices.append(night_rate if hour >= 23 or hour < 8 else day_rate)
        elif plan_type == 'smart':
            if hour in ev_hours and ev_rate is not None: prices.append(ev_rate)
            elif hour >= 23 or hour < 8: prices.append(night_rate)
            elif 17 <= hour < 19:
                prices.append(day_rate if is_weekend and "no_peak_weekend" in extra_tags else peak_rate)
            else: prices.append(day_rate)
        else: prices.append(day_rate)
            
    return pd.Series(prices, index=date_range)

@njit
def _fast_simulate(consumptions, generations, hours, months,
                   force_charge_array, is_arbitrage_profitable,
                   usable_cap_kwh, min_soc_kwh, max_soc_kwh,
                   grid_rte, solar_charge_efficiency, grid_efficiency_sqrt,
                   charge_rate_limit, mic, mec, strategy_id):
    
    n = len(consumptions)
    grid_imports, grid_exports, soc_track = np.zeros(n), np.zeros(n), np.zeros(n)
    battery_soc = min_soc_kwh
    
    pre_charge_hours = np.zeros(24, dtype=np.bool_)
    if strategy_id == 2 or strategy_id == 3:  
        for h in range(24):
            if force_charge_array[h]:
                for offset in range(1, 5): pre_charge_hours[(h - offset) % 24] = True

    for i in range(n):
        hour, month = hours[i], months[i]
        is_heating_season = (month in [11, 12, 1, 2])
        is_summer = not is_heating_season
        home_demand, solar_gen = consumptions[i], generations[i]
        
        is_force_charge_hour = force_charge_array[hour] if strategy_id >= 1 else False
        is_pre_charge_hour = True if (strategy_id in [2, 3] and pre_charge_hours[hour] and not is_force_charge_hour) else False
            
        self_consumption = min(home_demand, solar_gen)
        remaining_demand, excess_solar = home_demand - self_consumption, solar_gen - self_consumption
        grid_import, grid_export = remaining_demand, 0.0
        
        if not is_force_charge_hour:
            available_energy = max(0.0, battery_soc - min_soc_kwh)
            discharge_for_home = min(remaining_demand, available_energy * grid_efficiency_sqrt, charge_rate_limit * 0.5)
            if discharge_for_home > 0.001:
                battery_soc -= discharge_for_home / grid_efficiency_sqrt
                grid_import = remaining_demand - discharge_for_home
                
        # Handle Excess Solar
        if excess_solar > 0:
            if strategy_id in [2, 3] and is_pre_charge_hour and is_arbitrage_profitable:
                grid_export += excess_solar
            elif strategy_id == 4 and is_summer:
                # Strategy 4: Summer Solar Pass-Through
                grid_export += excess_solar
            else:
                space_in_battery = max(0.0, max_soc_kwh - battery_soc)
                charge_from_solar = min(excess_solar, space_in_battery / solar_charge_efficiency, charge_rate_limit * 0.5)
                if charge_from_solar > 0.001:
                    battery_soc += charge_from_solar * solar_charge_efficiency
                    grid_export += (excess_solar - charge_from_solar)
                else:
                    grid_export += excess_solar
                    
        # Handle Force Charging / Arbitrage Dump
        if strategy_id >= 1: 
            if is_pre_charge_hour and is_arbitrage_profitable:
                if not (strategy_id == 3 and is_heating_season):
                    available_energy = max(0.0, battery_soc - min_soc_kwh)
                    energy_to_discharge = min(available_energy * grid_efficiency_sqrt, charge_rate_limit * 0.5)
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
    
    force_charge_array = np.array(force_charge_hours, dtype=np.bool_)
    is_force_charge_interval = force_charge_array[import_prices.index.hour]
    cheapest_import_rate = np.min(import_prices.values[is_force_charge_interval]) if np.any(is_force_charge_interval) else 99.0
    is_arbitrage_profitable = (export_price * grid_rte) > cheapest_import_rate
    
    grid_imports, grid_exports, soc_track = _fast_simulate(
        df_hdf['consumption'].values, df_hdf['generation'].values, df_hdf.index.hour.values, df_hdf.index.month.values, 
        force_charge_array, is_arbitrage_profitable, usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, 
        solar_charge_efficiency, grid_efficiency_sqrt, params['charge_rate'], params['mic'], params['mec'], strategy_map.get(strategy, 0)
    )
    return grid_imports, grid_exports, soc_track, is_arbitrage_profitable

# =====================================================================
# 2. TKINTER GUI APPLICATION
# =====================================================================

class HomeBatteryCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Home Battery & Tariff Optimization Tool")
        self.root.geometry("1550x850")
        self.root.minsize(1300, 750)
        
        self.hdf_path, self.tariff_path = tk.StringVar(), tk.StringVar()
        self.leaderboard_data = None
        self.df_hdf = None 
        self.unique_dates = []
        self.current_date_idx = 0
        self.detailed_results = {}
        
        # Meta storage for HDF Export
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
        ttk.Label(files_frame, text="ESB HDF:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.hdf_path, width=25).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_hdf).grid(row=0, column=2, pady=2)
        
        ttk.Label(files_frame, text="Tariff DB:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(files_frame, textvariable=self.tariff_path, width=25).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(files_frame, text="Browse", command=self.browse_tariff).grid(row=1, column=2, pady=2)
        
        ttk.Button(files_frame, text="+ Create Custom Tariff", style="Secondary.TButton", command=self.open_custom_tariff_dialog).grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)
        
        # 2. Hardware Config (Explicit Grid Alignment)
        params_frame = ttk.LabelFrame(left_panel, text=" 2. Battery & Grid Hardware Configuration ", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="Capacity (kWh):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.entry_capacity = ttk.Entry(params_frame, width=8)
        self.entry_capacity.insert(0, "30.0")
        self.entry_capacity.grid(row=0, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Usable Depth (%):").grid(row=0, column=2, sticky=tk.W, pady=4)
        self.entry_usable_pct = ttk.Entry(params_frame, width=8)
        self.entry_usable_pct.insert(0, "100")
        self.entry_usable_pct.grid(row=0, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Chg Rate (kW):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.entry_charge_rate = ttk.Entry(params_frame, width=8)
        self.entry_charge_rate.insert(0, "18.0")
        self.entry_charge_rate.grid(row=1, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Region:").grid(row=1, column=2, sticky=tk.W, pady=4)
        self.combo_region = ttk.Combobox(params_frame, values=["urban", "rural"], width=6, state="readonly")
        self.combo_region.set("rural")
        self.combo_region.grid(row=1, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Min SoC (%):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.entry_minsoc = ttk.Entry(params_frame, width=8)
        self.entry_minsoc.insert(0, "10")
        self.entry_minsoc.grid(row=2, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Max SoC (%):").grid(row=2, column=2, sticky=tk.W, pady=4)
        self.entry_maxsoc = ttk.Entry(params_frame, width=8)
        self.entry_maxsoc.insert(0, "100")
        self.entry_maxsoc.grid(row=2, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Import (MIC):").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.entry_mic = ttk.Entry(params_frame, width=8)
        self.entry_mic.insert(0, "18")
        self.entry_mic.grid(row=3, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Export (MEC):").grid(row=3, column=2, sticky=tk.W, pady=4)
        self.entry_mec = ttk.Entry(params_frame, width=8)
        self.entry_mec.insert(0, "6")
        self.entry_mec.grid(row=3, column=3, sticky=tk.W, pady=4, padx=5)

        ttk.Label(params_frame, text="Grid RTE (%):").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.entry_grid_eff = ttk.Entry(params_frame, width=8)
        self.entry_grid_eff.insert(0, "90")
        self.entry_grid_eff.grid(row=4, column=1, sticky=tk.W, pady=4, padx=5)
        
        ttk.Label(params_frame, text="Solar RTE (%):").grid(row=4, column=2, sticky=tk.W, pady=4)
        self.entry_solar_eff = ttk.Entry(params_frame, width=8)
        self.entry_solar_eff.insert(0, "85")
        self.entry_solar_eff.grid(row=4, column=3, sticky=tk.W, pady=4, padx=5)

        # 3. Strategy Explanations
        explainer_frame = ttk.LabelFrame(left_panel, text=" 3. Strategy Explanations ", padding=10)
        explainer_frame.pack(fill=tk.X, pady=(0, 10))
        
        explainer_text = (
            "• Self-Consumption: Prioritizes storing excess solar for home use. Never force-charges from the grid.\n\n"
            "• Import-Minimiser: Force-charges the battery during cheap night rates to cover daytime load.\n\n"
            "• Import-Minimiser (Summer Pass): Acts as Import-Minimiser, but prevents solar from charging the battery from Mar-Oct to bypass AC conversion losses.\n\n"
            "• Balanced-Export: Dumps battery to the grid during peak hours to maximize Export profit (Arbitrage)."
        )
        ttk.Label(explainer_frame, text=explainer_text, wraplength=350, justify=tk.LEFT).pack(fill=tk.BOTH)

        # 4. Telemetry Output
        self.stats_frame = ttk.LabelFrame(left_panel, text=" Engine Telemetry ", padding=10)
        self.stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_stats = ttk.Label(
            self.stats_frame, 
            text="Waiting for simulation...\n\n\n\n", 
            foreground="#6b7280", 
            font=("Consolas", 9), 
            justify=tk.LEFT
        )
        self.lbl_stats.pack(fill=tk.BOTH)

        ttk.Button(left_panel, text="Run Optimization Sweep", style="Action.TButton", command=self.run_sweep).pack(fill=tk.X, ipady=5)

        # --- RIGHT TABS PANEL ---
        self.right_notebook = ttk.Notebook(workspace)
        workspace.add(self.right_notebook, weight=3)

        # Tab 1: Rankings
        tab_rankings = ttk.Frame(self.right_notebook, padding=10)
        self.right_notebook.add(tab_rankings, text="  Leaderboard Rankings  ")

        table_frame = ttk.Frame(tab_rankings)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("rank", "supplier", "tariff", "strategy", "arbitrage", "import", "export", "bill")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="none")
        self.tree.heading("rank", text="#")
        self.tree.heading("supplier", text="Supplier")
        self.tree.heading("tariff", text="Tariff Name")
        self.tree.heading("strategy", text="Winning Strategy")
        self.tree.heading("arbitrage", text="Arb. Viable?")
        self.tree.heading("import", text="Import Cost (€)")
        self.tree.heading("export", text="Export FIT (€)")
        self.tree.heading("bill", text="Annual Bill (€)")

        self.tree.column("rank", width=40, anchor=tk.CENTER)
        self.tree.column("supplier", width=110, anchor=tk.W)
        self.tree.column("tariff", width=200, anchor=tk.W)
        self.tree.column("strategy", width=190, anchor=tk.CENTER)
        self.tree.column("arbitrage", width=80, anchor=tk.CENTER)
        self.tree.column("import", width=100, anchor=tk.E)
        self.tree.column("export", width=100, anchor=tk.E)
        self.tree.column("bill", width=110, anchor=tk.E)

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
        self.hdf_month_combo.set("All Year")
        self.hdf_month_combo.pack(side=tk.LEFT)
        self.hdf_month_combo.bind("<<ComboboxSelected>>", self.update_hdf_graph)

        self.graph_container = ttk.Frame(self.tab_visualizer)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        if HAS_MATPLOTLIB:
            self.fig_hdf = Figure(figsize=(6, 4), dpi=100)
            self.ax_hdf = self.fig_hdf.add_subplot(111)
            self.canvas_hdf = FigureCanvasTkAgg(self.fig_hdf, master=self.graph_container)
            self.canvas_hdf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tabs 3, 4, 5: Top 3 Daily Viewers
        self.top_tabs = []
        for i in range(1, 4):
            frame = ttk.Frame(self.right_notebook, padding=10)
            self.right_notebook.add(frame, text=f"  Top {i}  ")
            self.setup_daily_tab(frame, i)

    def setup_daily_tab(self, parent, rank_idx):
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_frame = ttk.Frame(nav_frame)
        title_frame.pack(fill=tk.X)
        
        lbl_info = ttk.Label(title_frame, text=f"Run sweep to populate Rank {rank_idx}", font=("Helvetica", 11, "bold"), foreground="#4f46e5")
        lbl_info.pack(side=tk.LEFT, pady=(0, 10))
        
        # Format Export Button
        btn_export_hdf = ttk.Button(title_frame, text="⬇ Export Simulated HDF", command=lambda: self.export_simulated_hdf(rank_idx - 1))
        btn_export_hdf.pack(side=tk.RIGHT, pady=(0, 10))
        
        ctrl_subframe = ttk.Frame(nav_frame)
        ctrl_subframe.pack(fill=tk.X)
        
        ttk.Button(ctrl_subframe, text="< Prev Day", command=lambda: self.change_day(-1)).pack(side=tk.LEFT)
        
        self.lbl_date_vars = self.lbl_date_vars if hasattr(self, 'lbl_date_vars') else []
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
            fig = Figure(figsize=(8, 5), dpi=100)
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.top_tabs.append({'frame': parent, 'lbl_info': lbl_info, 'lbl_date': lbl_date, 
                                  'combo_month': combo_month, 'fig': fig, 'ax1': ax1, 'ax2': ax2, 'canvas': canvas,
                                  'internal_id': None, 'strategy': None})

    # ----------------- LOGIC & UPDATES -----------------

    def browse_hdf(self):
        f = filedialog.askopenfilename(filetypes=[("HDF CSV", "*.csv")])
        if f: self.hdf_path.set(f)

    def browse_tariff(self):
        f = filedialog.askopenfilename(filetypes=[("Tariff DB", "*.csv")])
        if f: self.tariff_path.set(f)

    def open_custom_tariff_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Custom Tariff")
        dlg.geometry("450x450")
        dlg.grab_set() 
        
        frame = ttk.Frame(dlg, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
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
        
        ttk.Label(frame, text="Export FIT (c/kWh):").grid(row=8, column=0, sticky=tk.W, pady=5)
        ent_fit = ttk.Entry(frame, width=10); ent_fit.insert(0, "18.0"); ent_fit.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        ev_frame = ttk.Frame(frame)
        ev_frame.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=10)
        ttk.Label(ev_frame, text="EV Start Hour (0-23):").pack(side=tk.LEFT)
        ent_ev_start = ttk.Entry(ev_frame, width=4); ent_ev_start.insert(0, "2"); ent_ev_start.pack(side=tk.LEFT, padx=5)
        ttk.Label(ev_frame, text="End Hour:").pack(side=tk.LEFT)
        ent_ev_end = ttk.Entry(ev_frame, width=4); ent_ev_end.insert(0, "5"); ent_ev_end.pack(side=tk.LEFT, padx=5)

        def save_tariff():
            try:
                new_tariff = {
                    'Supplier': ent_sup.get().strip(),
                    'Tariff name': ent_name.get().strip() + " (Custom)",
                    'Plan type': combo_type.get(),
                    'Standing charge': float(ent_sc.get() or 0.0),
                    'PSO Levy': 0.0,
                    'Cash bonus': 0.0,
                    'Day unit': float(ent_day.get() or 0.0),
                    'Night unit': float(ent_night.get() or 0.0),
                    'Peak unit': float(ent_peak.get() or 0.0),
                    'Ev unit': float(ent_ev.get() or 0.0),
                    'Fit unit': float(ent_fit.get() or 0.0),
                    'Supply Region': self.combo_region.get().strip().lower(), 
                    'Extra': f'["ev_{int(ent_ev_start.get())}_{int(ent_ev_end.get())}"]' if ent_ev.get() else ""
                }
                self.custom_tariffs.append(new_tariff)
                messagebox.showinfo("Success", f"Added Custom Tariff: {new_tariff['Tariff name']}\nIt will be included in the next sweep.")
                dlg.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please ensure all rates and hours are valid numbers.", parent=dlg)

        ttk.Button(frame, text="Save & Add to Database", command=save_tariff).grid(row=10, column=0, columnspan=2, pady=15, sticky=tk.EW)

    def export_simulated_hdf(self, tab_idx):
        tab_ui = self.top_tabs[tab_idx]
        internal_id = tab_ui['internal_id']
        strategy = tab_ui['strategy']

        if not internal_id or self.df_hdf is None:
            messagebox.showerror("Error", "No simulation data available to export. Run a sweep first.")
            return

        supplier_name = self.detailed_results[internal_id].get('meta', {}).get('Supplier', 'Supplier')
        tariff_name = self.detailed_results[internal_id].get('meta', {}).get('Tariff name', 'Tariff').replace("/", "-")
        default_filename = f"Simulated_HDF_{supplier_name}_{tariff_name}.csv"

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv")]
        )

        if not filename:
            return

        try:
            sim_data = self.detailed_results[internal_id][strategy]
            
            # Realign timestamps to match the original ESB "end of period" format and use exact date string format
            times = self.df_hdf.index + pd.Timedelta(minutes=30)
            dt_strs = times.strftime('%d/%m/%Y %H:%M')

            # Create standard ESB formatted Import DataFrame
            df_imp = pd.DataFrame({
                'timestamp': times,
                'MPRN': self.mprn,
                'Meter Serial Number': self.meter_serial,
                'Read Value': sim_data['import'].round(3),
                'Read Type': 'Active Import Interval (kWh)',
                'Read Date and End Time': dt_strs
            })

            # Create standard ESB formatted Export DataFrame
            df_exp = pd.DataFrame({
                'timestamp': times,
                'MPRN': self.mprn,
                'Meter Serial Number': self.meter_serial,
                'Read Value': sim_data['export'].round(3),
                'Read Type': 'Active Export Interval (kWh)',
                'Read Date and End Time': dt_strs
            })

            # Combine and sort strictly matching the ESB format: Reverse chronological, Import strictly before Export
            df_out = pd.concat([df_imp, df_exp]).sort_values(
                by=['timestamp', 'Read Type'], 
                ascending=[False, False]
            ).drop(columns=['timestamp'])
            
            # Set exact strict column order
            final_cols = ['MPRN', 'Meter Serial Number', 'Read Value', 'Read Type', 'Read Date and End Time']
            df_out = df_out[final_cols]
            
            df_out.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Successfully exported Synthetic HDF file to:\n{filename}\n\nYou can now upload this to price comparison websites or Excel.")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to generate HDF: {str(e)}")

    def run_sweep(self):
        if not self.hdf_path.get() or (not self.tariff_path.get() and not self.custom_tariffs):
            messagebox.showerror("Error", "Please select an HDF file and either a Tariff DB or add a Custom Tariff.")
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
            messagebox.showerror("Error", "Check numeric parameters.")
            return

        self.lbl_stats.config(text="Parsing HDF Data...\n\n\n\n", foreground="#f59e0b")
        self.root.update()

        try:
            start_time = time.time()

            raw_hdf, mprn_val, meter_val = parse_hdf(self.hdf_path.get().strip())
            self.df_hdf = filter_last_12_full_months(raw_hdf)
            self.mprn = mprn_val
            self.meter_serial = meter_val
            
            if self.df_hdf.empty: raise ValueError("No valid data left after filtering.")
            
            self.unique_dates = np.unique(self.df_hdf.index.date)
            self.current_date_idx = 0
            self.detailed_results.clear()
            self.update_hdf_graph() 

            if self.tariff_path.get():
                df_tariffs = pd.read_csv(self.tariff_path.get().strip())
                df_tariffs.columns = df_tariffs.columns.str.strip()
            else:
                df_tariffs = pd.DataFrame()

            if self.custom_tariffs:
                df_custom = pd.DataFrame(self.custom_tariffs)
                df_tariffs = pd.concat([df_tariffs, df_custom], ignore_index=True)

            valid_tariffs = df_tariffs[(df_tariffs['Supply Region'].str.lower() == params['region']) & (df_tariffs['Plan type'].str.lower() != 'gas')]
            
            if valid_tariffs.empty:
                raise ValueError("No valid tariffs found for the selected region.")

            results = []
            int_id = 0
            
            total_rows = len(self.df_hdf)
            num_tariffs = len(valid_tariffs)

            for _, row in valid_tariffs.iterrows():
                fit_rate = float(row['Fit unit']) / 100.0 if not pd.isna(row['Fit unit']) else 0.18
                import_prices = get_half_hourly_rates_for_row(row, self.df_hdf.index)
                force_charge_hours = [p <= import_prices.iloc[:48:2].min() + 0.001 for p in import_prices.iloc[:48:2]]
                
                tid = f"T_{int_id}"; int_id += 1
                self.detailed_results[tid] = {'meta': row.to_dict()}

                for strategy in ['self-consumption', 'import-minimiser', 'import-minimiser-summer-pass', 'balanced-export-maximiser']:
                    imports, exports, soc, is_arb = run_simulation(self.df_hdf, import_prices, fit_rate, strategy, force_charge_hours, params)
                    self.detailed_results[tid][strategy] = {'import': imports, 'export': exports, 'soc': soc}
                    
                    annual_import_cost = np.sum(imports * import_prices.values)
                    annual_export_credit = np.sum(exports * fit_rate)
                    net_bill = annual_import_cost - annual_export_credit
                    fixed = float(row['Standing charge']) + float(row['PSO Levy']) - (float(row['Cash bonus']) if not pd.isna(row['Cash bonus']) else 0.0)

                    results.append({
                        'Supplier': row['Supplier'], 'Tariff': row['Tariff name'], 'Strategy': strategy, 
                        'Arbitrage': "Yes" if is_arb else "No", 
                        'Import': annual_import_cost, 'Export': annual_export_credit,
                        'Bill': net_bill + fixed, '_id': tid
                    })

            calc_time = time.time() - start_time
            total_calcs = total_rows * num_tariffs * 4
            ops_estimate = total_calcs * 18
            
            telemetry = (
                f"[✓] Data Points: {total_rows:,} (Half-hourly)\n"
                f"[✓] Tariffs Evaluated: {num_tariffs}\n"
                f"[✓] Total Simulations: {num_tariffs * 4:,}\n"
                f"⚡ C-Compiled OPs: ~{ops_estimate:,}\n"
                f"⏱️ Exec Time: {calc_time:.2f} seconds"
            )
            self.lbl_stats.config(text=telemetry, foreground="#10b981")

            self.leaderboard_data = pd.DataFrame(results).sort_values(by='Bill').reset_index(drop=True)
            
            for item in self.tree.get_children(): self.tree.delete(item)
            for idx, row in self.leaderboard_data.iterrows():
                self.tree.insert("", "end", values=(
                    idx + 1, row['Supplier'], row['Tariff'], 
                    row['Strategy'].replace('-', ' ').title(), row['Arbitrage'], 
                    f"€ {row['Import']:,.2f}", f"€ {row['Export']:,.2f}", f"€ {row['Bill']:,.2f}"
                ))

            top_3 = self.leaderboard_data.head(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                tab_ui = self.top_tabs[i]
                tab_ui['internal_id'] = row['_id']
                tab_ui['strategy'] = row['Strategy']
                
                title = f"{i+1}. {row['Supplier']} - {row['Tariff']}"
                strat_text = row['Strategy'].replace('-', ' ').title()
                arb_note = "" if row['Arbitrage'] == "Yes" else " | ⚠️ Arbitrage Not Economical"
                
                self.right_notebook.tab(tab_ui['frame'], text=f"  #{i+1}: {row['Supplier']}  ")
                tab_ui['lbl_info'].config(text=f"{title}\nWinning Strategy: {strat_text}{arb_note}")
                
            self.update_daily_charts()
            messagebox.showinfo("Success", "Sweep complete! Check the Top 3 tabs for daily breakdown.")

        except Exception as e: 
            messagebox.showerror("Error", str(e))
            self.lbl_stats.config(text="Simulation Failed.", foreground="#ef4444")

    def update_hdf_graph(self, event=None):
        if not HAS_MATPLOTLIB or self.df_hdf is None: return
        month_sel = self.hdf_month_combo.get()
        df_target = self.df_hdf
        
        if month_sel != "All Year":
            m_idx = MONTH_NAMES.index(month_sel) + 1
            df_target = self.df_hdf[self.df_hdf.index.month == m_idx]
            if df_target.empty: return

        hourly_avg = df_target.groupby(df_target.index.hour).mean() * 2.0
        self.ax_hdf.clear()
        self.ax_hdf.plot(hourly_avg.index, hourly_avg['consumption'], label="Avg Grid Import (kW)", color="#4f46e5", linewidth=2.5)
        self.ax_hdf.plot(hourly_avg.index, hourly_avg['generation'], label="Avg Grid Export (kW)", color="#10b981", linewidth=2.5)
        self.ax_hdf.set_title(f"Average Load Profile: {month_sel}", fontsize=11, fontweight="bold")
        self.ax_hdf.set_xlabel("Hour"), self.ax_hdf.set_ylabel("Power (kW)")
        self.ax_hdf.set_xticks(range(0, 24, 2)), self.ax_hdf.grid(True, linestyle="--", alpha=0.5), self.ax_hdf.legend()
        self.fig_hdf.tight_layout(), self.canvas_hdf.draw()

    def change_day(self, delta):
        if not len(self.unique_dates): return
        self.current_date_idx = (self.current_date_idx + delta) % len(self.unique_dates)
        self.update_daily_charts()

    def jump_to_month(self, month_name):
        if not len(self.unique_dates): return
        m_idx = MONTH_NAMES.index(month_name) + 1
        for i, dt in enumerate(self.unique_dates):
            if dt.month == m_idx:
                self.current_date_idx = i
                self.update_daily_charts()
                return

    def update_daily_charts(self):
        if not HAS_MATPLOTLIB or self.df_hdf is None: return
        
        target_date = self.unique_dates[self.current_date_idx]
        date_str = target_date.strftime("%A, %d %b %Y")
        month_name = target_date.strftime("%B")
        mask = (self.df_hdf.index.date == target_date)
        hours = self.df_hdf.index[mask].hour + self.df_hdf.index[mask].minute / 60.0
        
        orig_imp = self.df_hdf['consumption'].values[mask] * 2.0
        orig_exp = self.df_hdf['generation'].values[mask] * 2.0

        for tab_ui in self.top_tabs:
            if not tab_ui['internal_id']: continue
            
            tab_ui['lbl_date'].config(text=date_str)
            tab_ui['combo_month'].set(month_name)
            tab_ui['ax1'].clear(); tab_ui['ax2'].clear()
            
            sim_data = self.detailed_results[tab_ui['internal_id']][tab_ui['strategy']]
            rev_imp = sim_data['import'][mask] * 2.0
            rev_exp = sim_data['export'][mask] * 2.0
            soc = sim_data['soc'][mask]
            
            tab_ui['ax1'].plot(hours, orig_imp, color="gray", linestyle="--", alpha=0.6, label="Orig. House Load")
            tab_ui['ax1'].plot(hours, orig_exp, color="lightgreen", linestyle="--", alpha=0.6, label="Orig. Solar Export")
            tab_ui['ax1'].plot(hours, rev_imp, color="#ef4444", linewidth=2, label="Rev. Grid Import")
            tab_ui['ax1'].plot(hours, rev_exp, color="#10b981", linewidth=2, label="Rev. Grid Export")
            
            tab_ui['ax2'].fill_between(hours, 0, soc, color="#f59e0b", alpha=0.15)
            tab_ui['ax2'].plot(hours, soc, color="#f59e0b", linewidth=1.5, label="Battery SoC (%)")
            
            tab_ui['ax1'].set_ylabel("Power (kW)"), tab_ui['ax2'].set_ylabel("SoC (%)"), tab_ui['ax2'].set_ylim(0, 105)
            tab_ui['ax1'].set_xticks(range(0, 25, 2)), tab_ui['ax1'].grid(True, linestyle=":", alpha=0.7)
            
            l1, lab1 = tab_ui['ax1'].get_legend_handles_labels()
            l2, lab2 = tab_ui['ax2'].get_legend_handles_labels()
            tab_ui['ax1'].legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
            tab_ui['fig'].tight_layout(); tab_ui['canvas'].draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = HomeBatteryCalculatorApp(root)
    root.mainloop()