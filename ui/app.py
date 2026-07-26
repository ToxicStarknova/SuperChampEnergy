import os
import re
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import customtkinter as ctk
from typing import Dict, Any, List, Optional

from core.models import SimulationParams, FinancialROIParams, FinancialROICalculator, DualTariffParams, DualTariffResult
from core.parsers import (parse_hdf, filter_last_12_full_months, normalize_tariff_dataframe, 
                          get_half_hourly_rates_for_row, prepare_dam, parse_dynamic_suppliers, MONTH_NAMES)
from core.engine import (run_simulation, run_dynamic_simulation, _calc_cost_with_overage, evaluate_dual_tariffs,
                         _run_simulation_from_arrays, _run_dynamic_simulation_from_arrays, _calc_ideal_daily_adaptive)
from core.report_generator import generate_html_report
from ui.components import ToolTip, CustomTariffDialog, FinancialROIDialog
from ui.charts import ChartManager, HAS_MATPLOTLIB
from ui.dual_tariff_dialog import DualTariffDialog

if HAS_MATPLOTLIB:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HomeBatteryCalculatorApp:
    """Main Application Window for Home Battery & Tariff Optimization Tool."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Home Battery & Tariff Optimization Tool - V2.5 Professional")
        self.root.geometry("1720x940")
        self.root.minsize(1400, 820)
        
        self.hdf_path = tk.StringVar()
        self.tariff_path = tk.StringVar()
        self.dam_path = tk.StringVar()
        self.dynamic_adders_path = tk.StringVar()
        
        self.leaderboard_data = None
        self.df_hdf = None 
        self.unique_dates = []
        self.current_date_idx = 0
        self.detailed_results = {}
        
        self.mprn = "00000000000"
        self.meter_serial = "00000000"
        self.custom_tariffs = []
        
        self.roi_params = {
            'battery_capex': 5500.0,
            'inverter_capex': 1500.0,
            'grant_amount': 2100.0,
            'electricity_inflation_pct': 3.0,
            'annual_degradation_pct': 2.0
        }
        self.dual_params = DualTariffParams()
        
        self.status_queue = queue.Queue()
        self.setup_ui()
        self.check_queue_loop()

    def setup_ui(self):
        # Base Container
        main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header Area
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        ctk.CTkLabel(header_frame, text="Home Battery & Tariff Optimizer", 
                     font=("Segoe UI", 22, "bold"), text_color=("#4f46e5", "#818cf8")).pack(side=tk.LEFT)
        
        ctk.CTkButton(header_frame, text="⚙️ Financial ROI Setup", width=140, fg_color="transparent", border_width=1,
                      text_color=("#4f46e5", "#818cf8"), command=self.open_financial_roi_dialog).pack(side=tk.RIGHT, padx=4)

        ctk.CTkButton(header_frame, text="🔀 Dual-Tariff Analysis", width=150, fg_color="transparent", border_width=1,
                      text_color=("#0284c7", "#38bdf8"), command=self.open_dual_tariff_dialog).pack(side=tk.RIGHT, padx=4)
        
        ctk.CTkButton(header_frame, text="📄 Export HTML Report", width=150, fg_color="#10b981", hover_color="#059669",
                      command=self.export_html_report).pack(side=tk.RIGHT, padx=4)

        # Workspace PanedWindow
        workspace = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        workspace.pack(fill=tk.BOTH, expand=True)

        # --- LEFT CONTROLS PANEL ---
        left_panel = ctk.CTkFrame(workspace, fg_color="transparent")
        workspace.add(left_panel, weight=1)
        
        # 1. Source Files Group
        files_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        files_frame.pack(fill=tk.X, pady=(0, 12), padx=(0, 10))
        
        ctk.CTkLabel(files_frame, text="1. Input Source Files", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(10, 2))
        lbl_warn = ctk.CTkLabel(files_frame, text="* Best with baseline un-metered profiles. Existing storage alters logs.",
                                font=("Segoe UI", 10, "italic"), text_color=("#ef4444", "#f87171"))
        lbl_warn.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=12, pady=(0, 8))
        
        lbl_hdf = ctk.CTkLabel(files_frame, text="ESB HDF:", font=("Segoe UI", 11, "underline"))
        lbl_hdf.grid(row=2, column=0, sticky=tk.W, padx=12, pady=4)
        ctk.CTkEntry(files_frame, textvariable=self.hdf_path, width=170).grid(row=2, column=1, padx=4, pady=4)
        ctk.CTkButton(files_frame, text="Browse", width=75, command=self.browse_hdf).grid(row=2, column=2, padx=12, pady=4)
        ToolTip(lbl_hdf, "Smart meter readings in 30-min kWh intervals from ESB Networks.")
        
        lbl_tariff = ctk.CTkLabel(files_frame, text="Tariff DB:", font=("Segoe UI", 11, "underline"))
        lbl_tariff.grid(row=3, column=0, sticky=tk.W, padx=12, pady=4)
        ctk.CTkEntry(files_frame, textvariable=self.tariff_path, width=170).grid(row=3, column=1, padx=4, pady=4)
        ctk.CTkButton(files_frame, text="Browse", width=75, command=self.browse_tariff).grid(row=3, column=2, padx=12, pady=4)
        ToolTip(lbl_tariff, "Tariff spreadsheet database matching energypal.ie smartplans tables.")

        lbl_dam = ctk.CTkLabel(files_frame, text="DAM Price:", font=("Segoe UI", 11, "underline"))
        lbl_dam.grid(row=4, column=0, sticky=tk.W, padx=12, pady=4)
        ctk.CTkEntry(files_frame, textvariable=self.dam_path, width=170).grid(row=4, column=1, padx=4, pady=4)
        ctk.CTkButton(files_frame, text="Browse", width=75, command=self.browse_dam).grid(row=4, column=2, padx=12, pady=4)
        ToolTip(lbl_dam, "Day-Ahead Market wholesale prices from semopx.com sheet structural logs.")

        lbl_dyn = ctk.CTkLabel(files_frame, text="Dyn Adder:", font=("Segoe UI", 11, "underline"))
        lbl_dyn.grid(row=5, column=0, sticky=tk.W, padx=12, pady=4)
        ctk.CTkEntry(files_frame, textvariable=self.dynamic_adders_path, width=170).grid(row=5, column=1, padx=4, pady=4)
        ctk.CTkButton(files_frame, text="Browse", width=75, command=self.browse_dyn).grid(row=5, column=2, padx=12, pady=4)
        ToolTip(lbl_dyn, "Supplier standing charges and adjustments for dynamic wholesale tracks.")
        
        ctk.CTkButton(files_frame, text="+ Create Custom Tariff", fg_color="transparent", border_width=1,
                      text_color=("#3b82f6", "#60a5fa"), command=self.open_custom_tariff_dialog).grid(row=6, column=0, columnspan=3, pady=(8, 12), padx=12, sticky=tk.EW)
        
        # 2. Hardware Config Group
        params_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        params_frame.pack(fill=tk.X, pady=(0, 12), padx=(0, 10))
        
        ctk.CTkLabel(params_frame, text="2. Hardware & Grid Settings", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=12, pady=(10, 6))
        
        ctk.CTkLabel(params_frame, text="Capacity (kWh):").grid(row=1, column=0, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_capacity = ctk.CTkEntry(params_frame, width=65); self.entry_capacity.insert(0, "30.0"); self.entry_capacity.grid(row=1, column=1, sticky=tk.W, pady=4)
        
        ctk.CTkLabel(params_frame, text="Usable Depth (%):").grid(row=1, column=2, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_usable_pct = ctk.CTkEntry(params_frame, width=65); self.entry_usable_pct.insert(0, "100"); self.entry_usable_pct.grid(row=1, column=3, sticky=tk.W, pady=4)

        ctk.CTkLabel(params_frame, text="Chg Rate (kW):").grid(row=2, column=0, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_charge_rate = ctk.CTkEntry(params_frame, width=65); self.entry_charge_rate.insert(0, "10.0"); self.entry_charge_rate.grid(row=2, column=1, sticky=tk.W, pady=4)
        
        ctk.CTkLabel(params_frame, text="Region:").grid(row=2, column=2, sticky=tk.W, padx=(12, 4), pady=4)
        self.combo_region = ctk.CTkComboBox(params_frame, values=["urban", "rural"], width=85, state="readonly"); self.combo_region.set("rural"); self.combo_region.grid(row=2, column=3, sticky=tk.W, pady=4)

        ctk.CTkLabel(params_frame, text="Min SoC (%):").grid(row=3, column=0, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_minsoc = ctk.CTkEntry(params_frame, width=65); self.entry_minsoc.insert(0, "10"); self.entry_minsoc.grid(row=3, column=1, sticky=tk.W, pady=4)
        
        ctk.CTkLabel(params_frame, text="Max SoC (%):").grid(row=3, column=2, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_maxsoc = ctk.CTkEntry(params_frame, width=65); self.entry_maxsoc.insert(0, "100"); self.entry_maxsoc.grid(row=3, column=3, sticky=tk.W, pady=4)

        ctk.CTkLabel(params_frame, text="Import (MIC):").grid(row=4, column=0, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_mic = ctk.CTkEntry(params_frame, width=65); self.entry_mic.insert(0, "18"); self.entry_mic.grid(row=4, column=1, sticky=tk.W, pady=4)
        
        ctk.CTkLabel(params_frame, text="Export (MEC):").grid(row=4, column=2, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_mec = ctk.CTkEntry(params_frame, width=65); self.entry_mec.insert(0, "6"); self.entry_mec.grid(row=4, column=3, sticky=tk.W, pady=4)

        ctk.CTkLabel(params_frame, text="Grid RTE (%):").grid(row=5, column=0, sticky=tk.W, padx=(12, 4), pady=4)
        self.entry_grid_eff = ctk.CTkEntry(params_frame, width=65); self.entry_grid_eff.insert(0, "95"); self.entry_grid_eff.grid(row=5, column=1, sticky=tk.W, pady=4)
        
        ctk.CTkLabel(params_frame, text="Solar RTE (%):").grid(row=5, column=2, sticky=tk.W, padx=(12, 4), pady=(4, 12))
        self.entry_solar_eff = ctk.CTkEntry(params_frame, width=65); self.entry_solar_eff.insert(0, "85"); self.entry_solar_eff.grid(row=5, column=3, sticky=tk.W, pady=(4, 12))

        # 3. Strategy Documentation Box
        explainer_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        explainer_frame.pack(fill=tk.X, pady=(0, 12), padx=(0, 10))
        
        ctk.CTkLabel(explainer_frame, text="3. Charging Strategy Profiles", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 4))
        strategies_info = [
            ("• Self-Consumption", "Uses solar first; never charges from grid.", "Prioritizes storing excess solar production locally. The battery acts strictly as a solar sponge."),
            ("• Import-Minimiser", "Force-charges from grid during cheapest hours.", "Force-charges the battery system up to max capacity during the lowest cost daily tariff window."),
            ("• Export-Maximiser", "Dumps battery to grid before cheap hours.", "Forces a proactive battery energy dump directly to the grid in the 4 hours prior to the cheap window starting."),
            ("• Balanced-Export", "Arbitrages in summer; preserves winter power.", "Runs arbitrage dump protocols during spring/summer, but preserves winter heating security bounds."),
            ("• Import-Min (Pass)", "Bypasses battery charging in summer cycle.", "Prevents solar generation from charging battery between March and October to bypass structural round efficiency losses."),
            ("• Ideal Daily Adaptive", "Oracle EMS: Picks optimal strategy each day.", "Evaluates every strategy on each individual day to model theoretical maximum smart EMS savings benchmark.")
        ]
        
        for label_text, brief_text, tip_text in strategies_info:
            frame_row = ctk.CTkFrame(explainer_frame, fg_color="transparent")
            frame_row.pack(anchor=tk.W, pady=2, fill=tk.X, padx=12)
            lbl_title = ctk.CTkLabel(frame_row, text=label_text, text_color=("#4f46e5", "#818cf8"), font=("Segoe UI", 11, "bold", "underline"))
            lbl_title.pack(side=tk.LEFT)
            lbl_brief = ctk.CTkLabel(frame_row, text=f" - {brief_text}", font=("Segoe UI", 11), text_color=("#475569", "#94a3b8"))
            lbl_brief.pack(side=tk.LEFT)
            ToolTip(lbl_title, tip_text); ToolTip(lbl_brief, tip_text)

        # Run Action Button
        self.btn_run = ctk.CTkButton(left_panel, text="Run Optimization Sweep", font=("Segoe UI", 14, "bold"), 
                                     fg_color="#4f46e5", hover_color="#4338ca", command=self.start_sweep_thread)
        self.btn_run.pack(side=tk.BOTTOM, fill=tk.X, ipady=6, padx=(0, 10), pady=(10, 0))

        # 4. Engine Telemetry Box
        self.stats_frame = ctk.CTkFrame(left_panel, corner_radius=10)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0), padx=(0, 10))
        ctk.CTkLabel(self.stats_frame, text="Engine Telemetry Console", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 2))
        
        self.txt_stats = ctk.CTkTextbox(self.stats_frame, font=("Consolas", 11), fg_color=("#f8fafc", "#0f172a"), border_width=1)
        self.txt_stats.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.txt_stats.insert(tk.END, "Waiting for optimization sweep execution...")
        self.txt_stats.configure(state="disabled")

        # --- RIGHT ANALYSIS NOTEBOOK ---
        self.right_notebook = ttk.Notebook(workspace)
        workspace.add(self.right_notebook, weight=3)

        # Tab 1: Leaderboard
        tab_rankings = ctk.CTkFrame(self.right_notebook, fg_color="transparent")
        self.right_notebook.add(tab_rankings, text="  Leaderboard Rankings  ")

        # Metric KPI Cards Row
        kpi_container = ctk.CTkFrame(tab_rankings, fg_color="transparent")
        kpi_container.pack(fill=ctk.X, pady=(0, 15))

        # Card 1: Annual Savings
        card_savings = ctk.CTkFrame(kpi_container, corner_radius=12, fg_color=("#e6f4ea", "#14291e"))
        card_savings.pack(side=ctk.LEFT, expand=True, fill=ctk.BOTH, padx=4, ipady=4)
        ctk.CTkLabel(card_savings, text="OPTIMAL ANNUAL SAVINGS", font=("Segoe UI", 10, "bold"), text_color=("#137333", "#81c995")).pack(pady=(8, 2))
        self.lbl_kpi_savings = ctk.CTkLabel(card_savings, text="€0.00", font=("Segoe UI", 22, "bold"), text_color=("#137333", "#a8dab5"))
        self.lbl_kpi_savings.pack(pady=(0, 4))
        self.lbl_sub_savings = ctk.CTkLabel(card_savings, text="vs. unoptimized baseline tariff", font=("Segoe UI", 10, "italic"), text_color=("#5f6368", "#9aa0a6"))
        self.lbl_sub_savings.pack(pady=(0, 8))

        # Card 2: Top Strategy
        card_strategy = ctk.CTkFrame(kpi_container, corner_radius=12, fg_color=("#e8f0fe", "#1a233a"))
        card_strategy.pack(side=ctk.LEFT, expand=True, fill=ctk.BOTH, padx=4, ipady=4)
        ctk.CTkLabel(card_strategy, text="WINNING STRATEGY", font=("Segoe UI", 10, "bold"), text_color=("#1a73e8", "#8ab4f8")).pack(pady=(8, 2))
        self.lbl_kpi_strategy = ctk.CTkLabel(card_strategy, text="N/A", font=("Segoe UI", 18, "bold"), text_color=("#1a73e8", "#adc6ff"))
        self.lbl_kpi_strategy.pack(pady=(2, 4))
        self.lbl_sub_strategy = ctk.CTkLabel(card_strategy, text="Max efficiency operational mode", font=("Segoe UI", 10, "italic"), text_color=("#5f6368", "#9aa0a6"))
        self.lbl_sub_strategy.pack(pady=(0, 8))

        # Card 3: Payback Years
        card_payback = ctk.CTkFrame(kpi_container, corner_radius=12, fg_color=("#fef7e0", "#2d2417"))
        card_payback.pack(side=ctk.LEFT, expand=True, fill=ctk.BOTH, padx=4, ipady=4)
        ctk.CTkLabel(card_payback, text="ESTIMATED PAYBACK", font=("Segoe UI", 10, "bold"), text_color=("#b06000", "#fdd663")).pack(pady=(8, 2))
        self.lbl_kpi_payback = ctk.CTkLabel(card_payback, text="N/A", font=("Segoe UI", 20, "bold"), text_color=("#b06000", "#ffe082"))
        self.lbl_kpi_payback.pack(pady=(2, 4))
        self.lbl_sub_payback = ctk.CTkLabel(card_payback, text="Simple CAPEX payback horizon", font=("Segoe UI", 10, "italic"), text_color=("#5f6368", "#9aa0a6"), wraplength=180)
        self.lbl_sub_payback.pack(pady=(0, 8))

        # Card 4: Grid Ceilings
        card_limits = ctk.CTkFrame(kpi_container, corner_radius=12, fg_color=("#f3e8ff", "#2a1b3d"))
        card_limits.pack(side=ctk.LEFT, expand=True, fill=ctk.BOTH, padx=4, ipady=4)
        ctk.CTkLabel(card_limits, text="METER CEILING STATUS", font=("Segoe UI", 10, "bold"), text_color=("#7e22ce", "#c084fc")).pack(pady=(8, 2))
        self.lbl_kpi_limits = ctk.CTkLabel(card_limits, text="Nominal", font=("Segoe UI", 18, "bold"), text_color=("#7e22ce", "#d8b4fe"))
        self.lbl_kpi_limits.pack(pady=(2, 4))
        self.lbl_sub_limits = ctk.CTkLabel(card_limits, text="MIC / MEC headroom limits", font=("Segoe UI", 10, "italic"), text_color=("#5f6368", "#9aa0a6"), wraplength=180)
        self.lbl_sub_limits.pack(pady=(0, 8))

        # Data Control Toolbar
        table_title_frame = ctk.CTkFrame(tab_rankings, fg_color="transparent")
        table_title_frame.pack(fill=tk.X, pady=(0, 8))
        ctk.CTkLabel(table_title_frame, text="Tariff Structural Leadership Rankings Table", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(table_title_frame, text="⬇ Export Table to CSV", width=150, fg_color="transparent", border_width=1,
                      text_color=("#475569", "#cbd5e1"), command=self.export_leaderboard).pack(side=tk.RIGHT, padx=6)

        table_frame = ctk.CTkFrame(tab_rankings, corner_radius=8)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ("rank", "supplier", "tariff", "strategy", "arbitrage", "imp_kwh", "exp_kwh", "import", "export", "june", "dec", "fixed", "bonus", "bill")
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
        self.tree.heading("bonus", text="Bonus (€)")
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
        self.tree.column("bonus", width=65, anchor=tk.E)
        self.tree.column("bill", width=100, anchor=tk.E)

        for col in cols:
            self.tree.heading(col, text=self.tree.heading(col, 'text'), 
                              command=lambda _col=col: self.treeview_sort_column(self.tree, _col, False))

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: HDF Base Profile Viewer
        self.tab_visualizer = ctk.CTkFrame(self.right_notebook, fg_color="transparent")
        self.right_notebook.add(self.tab_visualizer, text="  HDF Base Profile  ")
        
        hdf_ctrl_frame = ctk.CTkFrame(self.tab_visualizer, fg_color="transparent")
        hdf_ctrl_frame.pack(fill=tk.X, pady=(5, 10))
        ctk.CTkLabel(hdf_ctrl_frame, text="View Average Daily Profile for: ", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=6)
        self.hdf_month_combo = ctk.CTkComboBox(hdf_ctrl_frame, values=["All Year"] + MONTH_NAMES, state="readonly", width=130, command=self.update_hdf_graph)
        self.hdf_month_combo.set("All Year"); self.hdf_month_combo.pack(side=tk.LEFT)

        self.graph_container = ctk.CTkFrame(self.tab_visualizer)
        self.graph_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        if HAS_MATPLOTLIB:
            self.fig_hdf = Figure(figsize=(6, 4), dpi=100); self.ax_hdf = self.fig_hdf.add_subplot(111)
            self.canvas_hdf = FigureCanvasTkAgg(self.fig_hdf, master=self.graph_container)
            self.canvas_hdf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Simulated Performance Tracking Subviews
        self.top_tabs = []
        for i in range(1, 4):
            frame = ctk.CTkFrame(self.right_notebook, fg_color="transparent")
            self.right_notebook.add(frame, text=f"  Top {i}  ")
            self.setup_daily_tab(frame, str(i))
            
        frame_dyn = ctk.CTkFrame(self.right_notebook, fg_color="transparent")
        self.right_notebook.add(frame_dyn, text="  Top Dynamic  ")
        self.setup_daily_tab(frame_dyn, "Dynamic")

        self.apply_theme_styling()

    def apply_theme_styling(self):
        style = ttk.Style()
        style.theme_use('clam')
        mode = ctk.get_appearance_mode()
        
        if mode == "Dark":
            style.configure("Treeview", background="#1d1d1d", foreground="#ffffff", fieldbackground="#1d1d1d", rowheight=28)
            style.configure("Treeview.Heading", background="#2b2b2b", foreground="#ffffff", bordercolor="#1d1d1d", font=("Segoe UI", 9, "bold"))
            style.map("Treeview.Heading", background=[('active', '#3b82f6')], foreground=[('active', '#ffffff')])
            self.tree.tag_configure('best_baseline', background='#451a03', foreground='#f97316')
            style.configure("Vertical.TScrollbar", background="#3f3f46", troughcolor="#18181b", bordercolor="#18181b", arrowcolor="#ffffff", gripcount=0)
            style.configure("TNotebook", background="#1d1d1d", borderwidth=0)
            style.configure("TNotebook.Tab", background="#2b2b2b", foreground="#ffffff", bordercolor="#1d1d1d", lightcolor="#2b2b2b", darkcolor="#2b2b2b", padding=[14, 6], font=("Segoe UI", 9, "bold"))
            style.map("TNotebook.Tab", background=[("selected", "#3b82f6")], foreground=[("selected", "#ffffff")])
            style.configure("Panedwindow", background="#1d1d1d")
        else:
            style.configure("Treeview", background="#ffffff", foreground="#000000", fieldbackground="#ffffff", rowheight=28)
            style.configure("Treeview.Heading", background="#f1f5f9", foreground="#0f172a", bordercolor="#cbd5e1", font=("Segoe UI", 9, "bold"))
            style.map("Treeview.Heading", background=[('active', '#3b82f6')], foreground=[('active', '#ffffff')])
            self.tree.tag_configure('best_baseline', background='#ffedd5', foreground='#b45309')
            style.configure("Vertical.TScrollbar", background="#cbd5e1", troughcolor="#f1f5f9", bordercolor="#cbd5e1", arrowcolor="#000000", gripcount=0)
            style.configure("TNotebook", background="#ebebeb", borderwidth=0)
            style.configure("TNotebook.Tab", background="#dbdbdb", foreground="#000000", bordercolor="#ebebeb", lightcolor="#dbdbdb", darkcolor="#dbdbdb", padding=[14, 6], font=("Segoe UI", 9, "bold"))
            style.map("TNotebook.Tab", background=[("selected", "#3b82f6")], foreground=[("selected", "#ffffff")])
            style.configure("Panedwindow", background="#ebebeb")

    def setup_daily_tab(self, parent, tab_id):
        nav_frame = ctk.CTkFrame(parent, fg_color="transparent")
        nav_frame.pack(fill=tk.X, pady=(5, 5))
        
        title_frame = ctk.CTkFrame(nav_frame, fg_color="transparent")
        title_frame.pack(fill=tk.X)
        
        lbl_info = ctk.CTkLabel(title_frame, text=f"Run sweep to populate Rank {tab_id}", font=("Segoe UI", 12, "bold"), text_color="#4f46e5")
        lbl_info.pack(side=tk.LEFT, pady=(0, 10), padx=6)
        
        idx = len(self.top_tabs)
        btn_export_hdf = ctk.CTkButton(title_frame, text="Export Simulated HDF", width=140, fg_color="transparent", border_width=1, command=lambda local_idx=idx: self.export_simulated_hdf(local_idx))
        btn_export_hdf.pack(side=tk.RIGHT, pady=(0, 10), padx=6)
        
        ctrl_subframe = ctk.CTkFrame(nav_frame, fg_color="transparent")
        ctrl_subframe.pack(fill=tk.X, padx=6)
        
        ctk.CTkButton(ctrl_subframe, text="< Prev Day", width=90, command=lambda: self.change_day(-1)).pack(side=tk.LEFT)
        lbl_date = ctk.CTkLabel(ctrl_subframe, text="[Date]", font=("Segoe UI", 12, "bold"))
        lbl_date.pack(side=tk.LEFT, padx=15)
        
        ctk.CTkLabel(ctrl_subframe, text="Jump to Month:").pack(side=tk.LEFT, padx=(20, 5))
        combo_month = ctk.CTkComboBox(ctrl_subframe, values=MONTH_NAMES, state="readonly", width=110, command=lambda val: self.jump_to_month(val))
        combo_month.pack(side=tk.LEFT)
        ctk.CTkButton(ctrl_subframe, text="Next Day >", width=90, command=lambda: self.change_day(1)).pack(side=tk.RIGHT)
        
        graph_frame = ctk.CTkFrame(parent)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        if HAS_MATPLOTLIB:
            fig = Figure(figsize=(8, 5), dpi=100); ax1 = fig.add_subplot(111); ax2 = ax1.twinx()
            canvas = FigureCanvasTkAgg(fig, master=graph_frame); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.top_tabs.append({'frame': parent, 'lbl_info': lbl_info, 'lbl_date': lbl_date, 'combo_month': combo_month, 
                                  'fig': fig, 'ax1': ax1, 'ax2': ax2, 'canvas': canvas, 'internal_id': None, 'strategy': None})

    def check_queue_loop(self):
        """Polls thread queue for status updates and UI updates."""
        try:
            while True:
                msg_type, payload = self.status_queue.get_nowait()
                if msg_type == "CONSOLE":
                    text, color = payload
                    self.update_console(text, color)
                elif msg_type == "SWEEP_COMPLETE":
                    self.btn_run.configure(state="normal", text="Run Optimization Sweep")
                    self.on_sweep_finished(payload)
                elif msg_type == "SWEEP_ERROR":
                    self.btn_run.configure(state="normal", text="Run Optimization Sweep")
                    messagebox.showerror("Error", str(payload))
                    self.update_console("Simulation Failure occurred during execution loop tracking.", "#ef4444")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue_loop)

    def update_console(self, text_string, color_hex="#334155"):
        self.txt_stats.configure(state="normal")
        self.txt_stats.delete("1.0", tk.END)
        self.txt_stats.insert(tk.END, text_string)
        self.txt_stats.configure(state="disabled", text_color=color_hex)

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

        # Re-sync leaderboard_data with treeview visual order
        if self.leaderboard_data is not None and not self.leaderboard_data.empty:
            sorted_indices = []
            for k in tv.get_children(''):
                row_vals = tv.item(k)['values']
                supp, t_name, strat = row_vals[1], row_vals[2], row_vals[3].lower().replace(' ', '-')
                match = self.leaderboard_data[(self.leaderboard_data['Supplier'] == supp) & (self.leaderboard_data['Tariff'] == t_name)]
                if not match.empty:
                    sorted_indices.append(match.index[0])
            if len(sorted_indices) == len(self.leaderboard_data):
                self.leaderboard_data = self.leaderboard_data.reindex(sorted_indices).reset_index(drop=True)
                self.update_subtabs_from_leaderboard()

    def browse_hdf(self):
        f = filedialog.askopenfilename(filetypes=[("HDF CSV", "*.csv")])
        if f: self.hdf_path.set(f)
    def browse_tariff(self):
        f = filedialog.askopenfilename(filetypes=[("Tariff DB", "*.csv")])
        if f: self.tariff_path.set(f)
    def browse_dam(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if f: self.dam_path.set(f)
    def browse_dyn(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if f: self.dynamic_adders_path.set(f)

    def open_custom_tariff_dialog(self):
        region = self.combo_region.get().strip().lower()
        CustomTariffDialog(self.root, region, self.custom_tariffs.append)

    def open_financial_roi_dialog(self):
        def update_params(new_dict):
            self.roi_params.update(new_dict)
            if self.leaderboard_data is not None and not self.leaderboard_data.empty:
                self.refresh_kpi_cards()
        FinancialROIDialog(self.root, self.roi_params, update_params)

    def open_dual_tariff_dialog(self):
        if self.leaderboard_data is None or self.leaderboard_data.empty or self.df_hdf is None:
            messagebox.showwarning("Warning", "No simulation results available. Please run an optimization sweep first.")
            return

        def recalculate(new_dual_params):
            self.dual_params = new_dual_params
            return evaluate_dual_tariffs(self.leaderboard_data, self.df_hdf, self.dual_params)

        dual_results = evaluate_dual_tariffs(self.leaderboard_data, self.df_hdf, self.dual_params)
        DualTariffDialog(self.root, dual_results, self.dual_params, recalculate)

    def start_sweep_thread(self):
        if not self.hdf_path.get() or (not self.tariff_path.get() and not self.custom_tariffs):
            messagebox.showerror("Error", "Please select an HDF file and a Tariff DB (or Custom Tariff).")
            return
        try:
            params = SimulationParams(
                capacity=float(self.entry_capacity.get()),
                usable_pct=float(self.entry_usable_pct.get()),
                charge_rate=float(self.entry_charge_rate.get()),
                grid_efficiency=float(self.entry_grid_eff.get()),
                solar_efficiency=float(self.entry_solar_eff.get()),
                min_soc=float(self.entry_minsoc.get()),
                max_soc=float(self.entry_maxsoc.get()),
                mic=float(self.entry_mic.get()),
                mec=float(self.entry_mec.get()),
                region=self.combo_region.get().strip().lower()
            )
        except ValueError:
            messagebox.showerror("Error", "Check numeric parameters.")
            return

        self.btn_run.configure(state="disabled", text="Running Optimization Sweep...")
        self.update_console("Parsing Input Data & Pre-compiling Engine Tracks...", "#f59e0b")
        
        thread = threading.Thread(target=self._run_sweep_worker, args=(params,), daemon=True)
        thread.start()

    def _run_sweep_worker(self, params: SimulationParams):
        try:
            start_time = time.time()
            raw_hdf, mprn_val, meter_val = parse_hdf(self.hdf_path.get().strip())
            df_hdf = filter_last_12_full_months(raw_hdf)
            
            if df_hdf.empty: raise ValueError("No valid data left after filtering.")
            
            df_tariffs = pd.read_csv(self.tariff_path.get().strip()) if self.tariff_path.get() else pd.DataFrame()
            df_tariffs.columns = df_tariffs.columns.str.strip() if not df_tariffs.empty else []
            df_tariffs = normalize_tariff_dataframe(df_tariffs)

            if self.custom_tariffs:
                for t in self.custom_tariffs: t['Supply Region'] = params.region
                df_tariffs = pd.concat([df_tariffs, pd.DataFrame(self.custom_tariffs)], ignore_index=True)

            valid_tariffs = df_tariffs[(df_tariffs['Supply Region'].str.lower() == params.region) & (df_tariffs['Plan type'].str.lower() != 'gas') & (df_tariffs['Plan type'].str.lower() != 'dynamic')]

            dam_prices_c_kwh, dynamic_suppliers = None, []
            if self.dam_path.get() and self.dynamic_adders_path.get():
                try:
                    dam_prices_c_kwh = prepare_dam(df_hdf.index, self.dam_path.get().strip())
                    dynamic_suppliers = parse_dynamic_suppliers(self.dynamic_adders_path.get().strip(), params.region)
                except Exception as e:
                    self.status_queue.put(("CONSOLE", (f"Dynamic Pricing Skipped: {e}", "#f59e0b")))

            results = []; int_id = 0
            detailed_results = {}
            total_rows = len(df_hdf)
            num_tariffs = len(valid_tariffs) + len(dynamic_suppliers)
            if num_tariffs == 0:
                raise ValueError("No tariffs found matching the selected region and criteria.")
            
            n_samples = len(df_hdf)
            orig_imports = df_hdf['consumption'].values.astype(np.float64)
            orig_exports = df_hdf['generation'].values.astype(np.float64)
            hours_array = df_hdf.index.hour.values.astype(np.int64)
            months_array = df_hdf.index.month.values.astype(np.int64)
            dates_array = df_hdf.index.date
            
            _, day_ids = np.unique(dates_array, return_inverse=True)
            day_ids = day_ids.astype(np.int64)
            
            usable_cap_kwh = float(params.usable_capacity_kwh)
            min_soc_kwh = float(params.min_soc_kwh)
            max_soc_kwh = float(params.max_soc_kwh)
            grid_rte = float(params.grid_rte_decimal)
            solar_charge_eff = float(params.solar_charge_efficiency)
            charge_rate = float(params.charge_rate)
            mic = float(params.mic)
            mec = float(params.mec)

            mask_june, mask_dec = (months_array == 6), (months_array == 12)
            unique_dates = np.unique(dates_array)
            num_days = len(unique_dates)
            scaling_factor = 365.0 / num_days if num_days > 0 else 1.0
            is_short_duration = num_days < 330
            exceeded_plans = []
            all_strategies = ['self-consumption', 'import-minimiser', 'export-maximiser', 'balanced-export-maximiser', 'import-minimiser-summer-pass']
            
            days_june = max(1, np.sum(mask_june) // 48)
            days_dec = max(1, np.sum(mask_dec) // 48)

            # Pre-allocate 2D buffers (5 x N) for strategy outputs per tariff
            costs_matrix = np.zeros((5, n_samples), dtype=np.float64)
            exp_rev_matrix = np.zeros((5, n_samples), dtype=np.float64)
            imports_matrix = np.zeros((5, n_samples), dtype=np.float64)
            exports_matrix = np.zeros((5, n_samples), dtype=np.float64)
            soc_matrix = np.zeros((5, n_samples), dtype=np.float64)

            # 1. Standard Fixed Sweep Track
            for _, row in valid_tariffs.iterrows():
                try:
                    fit_rate = float(row['Fit unit']) / 100.0 if not pd.isna(row.get('Fit unit')) else 0.18
                    has_missing_fit = False
                except (ValueError, TypeError):
                    fit_rate = 0.18
                    has_missing_fit = True
                    
                import_prices_arr, is_ev_window, ev_overage_rate, has_overage_penalty, has_unknown_type, has_missing_rates = get_half_hourly_rates_for_row(row, df_hdf.index)
                if has_missing_fit: has_missing_rates = True
                    
                tariff_label = row['Tariff name']
                if has_unknown_type: tariff_label += " [Unknown Plan Type]"
                elif has_missing_rates: tariff_label += " [Missing Rates]"
                    
                first_day_prices = import_prices_arr[:48]
                min_p = np.min(first_day_prices)
                force_charge_hours_24 = np.zeros(24, dtype=np.bool_)
                for h in range(24):
                    if first_day_prices[h*2] <= min_p + 0.001 or first_day_prices[h*2+1] <= min_p + 0.001:
                        force_charge_hours_24[h] = True
                
                tid = f"T_{int_id}"; int_id += 1
                detailed_results[tid] = {'meta': row.to_dict()}
                
                try: fixed_charges = float(row['Standing charge']) + float(row.get('PSO Levy', 0))
                except (ValueError, TypeError): fixed_charges = 300.0; has_missing_rates = True
                    
                cash_bonus = float(row.get('Cash bonus', 0.0)) if not pd.isna(row.get('Cash bonus')) else 0.0
                monthly_fixed = fixed_charges / 12.0

                baseline_import_costs, base_limit_exceeded = _calc_cost_with_overage(orig_imports, import_prices_arr, is_ev_window, ev_overage_rate, months_array, has_overage_penalty)
                annual_imp_base = np.sum(baseline_import_costs)
                annual_exp_base = np.sum(orig_exports * fit_rate)
                net_bill_base = (annual_imp_base - annual_exp_base) * scaling_factor + fixed_charges - cash_bonus
                
                base_imp_kwh, base_exp_kwh = np.sum(orig_imports), np.sum(orig_exports)
                base_june = (np.sum(baseline_import_costs[mask_june]) - np.sum(orig_exports[mask_june] * fit_rate)) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                base_dec = (np.sum(baseline_import_costs[mask_dec]) - np.sum(orig_exports[mask_dec] * fit_rate)) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0
                
                detailed_results[tid]['baseline-no-battery'] = {'import': orig_imports, 'export': orig_exports, 'soc': np.zeros(n_samples)}
                results.append({
                    'Supplier': row['Supplier'], 'Tariff': tariff_label, 'Strategy': 'baseline-no-battery', 
                    'Arbitrage': "Check Data" if (has_unknown_type or has_missing_rates) else "N/A", 'Imp_kWh': base_imp_kwh, 'Exp_kWh': base_exp_kwh,
                    'Import': annual_imp_base, 'Export': annual_exp_base, 
                    'June': base_june, 'Dec': base_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                    'Bill': net_bill_base, '_id': tid, 'is_dynamic': False
                })

                for strat_idx in range(5):
                    strategy = all_strategies[strat_idx]
                    imports, exports, soc, is_arb = _run_simulation_from_arrays(
                        orig_imports, orig_exports, hours_array, months_array,
                        import_prices_arr, fit_rate, force_charge_hours_24, strat_idx,
                        usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, solar_charge_eff,
                        charge_rate, mic, mec
                    )
                    detailed_results[tid][strategy] = {'import': imports, 'export': exports, 'soc': soc}
                    
                    strategy_import_costs, strat_limit_exceeded = _calc_cost_with_overage(imports, import_prices_arr, is_ev_window, ev_overage_rate, months_array, has_overage_penalty)
                    if strat_limit_exceeded:
                        exceeded_plans.append(f"{row['Supplier']} {row['Tariff name']} ({strategy})")
                        
                    exp_rev = exports * fit_rate
                    costs_matrix[strat_idx, :] = strategy_import_costs
                    exp_rev_matrix[strat_idx, :] = exp_rev
                    imports_matrix[strat_idx, :] = imports
                    exports_matrix[strat_idx, :] = exports
                    soc_matrix[strat_idx, :] = soc
                    
                    annual_imp_cost = np.sum(strategy_import_costs)
                    annual_exp_rev = np.sum(exp_rev)
                    net_bill = (annual_imp_cost - annual_exp_rev) * scaling_factor + fixed_charges - cash_bonus
                    
                    strat_imp_kwh, strat_exp_kwh = np.sum(imports), np.sum(exports)
                    strat_june = (np.sum(strategy_import_costs[mask_june]) - np.sum(exp_rev[mask_june])) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                    strat_dec = (np.sum(strategy_import_costs[mask_dec]) - np.sum(exp_rev[mask_dec])) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0
                    
                    arb_display = f"{is_arb:.2f} c/kWh" if (strategy not in ['baseline-no-battery', 'self-consumption'] and is_arb is not None and is_arb > 0) else "N/A"
                            
                    results.append({
                        'Supplier': row['Supplier'], 'Tariff': tariff_label, 'Strategy': strategy, 'Arbitrage': "Check Data" if (has_unknown_type or has_missing_rates) else arb_display, 
                        'Imp_kWh': strat_imp_kwh, 'Exp_kWh': strat_exp_kwh,
                        'Import': annual_imp_cost, 'Export': annual_exp_rev, 
                        'June': strat_june, 'Dec': strat_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                        'Bill': net_bill, '_id': tid, 'is_dynamic': False
                    })

                # --- IDEAL DAILY ADAPTIVE STRATEGY (PURE NUMBA JIT) ---
                ideal_imp, ideal_exp, ideal_soc, ideal_costs, ideal_exp_rev, win_counts = _calc_ideal_daily_adaptive(
                    costs_matrix, exp_rev_matrix, imports_matrix, exports_matrix, soc_matrix, day_ids
                )
                strategy_win_counts = {all_strategies[i]: int(win_counts[i]) for i in range(5)}
                detailed_results[tid]['ideal-daily-adaptive'] = {'import': ideal_imp, 'export': ideal_exp, 'soc': ideal_soc, 'win_counts': strategy_win_counts}
                
                annual_ideal_imp_cost = np.sum(ideal_costs)
                annual_ideal_exp_rev = np.sum(ideal_exp_rev)
                net_bill_ideal = (annual_ideal_imp_cost - annual_ideal_exp_rev) * scaling_factor + fixed_charges - cash_bonus
                
                ideal_june = (np.sum(ideal_costs[mask_june]) - np.sum(ideal_exp_rev[mask_june])) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                ideal_dec = (np.sum(ideal_costs[mask_dec]) - np.sum(ideal_exp_rev[mask_dec])) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0

                results.append({
                    'Supplier': row['Supplier'], 'Tariff': tariff_label, 'Strategy': 'ideal-daily-adaptive', 
                    'Arbitrage': "Adaptive", 'Imp_kWh': np.sum(ideal_imp), 'Exp_kWh': np.sum(ideal_exp),
                    'Import': annual_ideal_imp_cost, 'Export': annual_ideal_exp_rev, 
                    'June': ideal_june, 'Dec': ideal_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                    'Bill': net_bill_ideal, '_id': tid, 'is_dynamic': False
                })

            # 2. Dynamic Tariff Sweep Track
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
                monthly_fixed = fixed_charges / 12.0
                
                prices = dam_prices_c_kwh.copy()
                is_night_mask = (hours_array >= 23) | (hours_array < 8)
                is_peak_mask = (hours_array >= 17) & (hours_array < 19)
                is_day_mask = ~(is_night_mask | is_peak_mask)
                prices[is_night_mask] += dyn['Night']
                prices[is_day_mask] += dyn['Day']
                prices[is_peak_mask] += dyn['Peak']
                import_prices_arr = (prices / 100.0) * 1.09
                
                dyn_is_ev_window = np.zeros(n_samples, dtype=np.bool_)
                dyn_ev_overage_rate = 0.0
                tid = f"T_{int_id}"; int_id += 1
                detailed_results[tid] = {'meta': dyn}
                
                baseline_import_costs, base_limit_exceeded = _calc_cost_with_overage(orig_imports, import_prices_arr, dyn_is_ev_window, dyn_ev_overage_rate, months_array, False)
                annual_imp_base = np.sum(baseline_import_costs)
                annual_exp_base = np.sum(orig_exports * fit_rate_arr)
                net_bill_base = (annual_imp_base - annual_exp_base) * scaling_factor + fixed_charges - cash_bonus
                
                base_imp_kwh, base_exp_kwh = np.sum(orig_imports), np.sum(orig_exports)
                base_june = (np.sum(baseline_import_costs[mask_june]) - np.sum(orig_exports[mask_june] * fit_rate_arr[mask_june])) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                base_dec = (np.sum(baseline_import_costs[mask_dec]) - np.sum(orig_exports[mask_dec] * fit_rate_arr[mask_dec])) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0
                
                detailed_results[tid]['baseline-no-battery'] = {'import': orig_imports, 'export': orig_exports, 'soc': np.zeros(n_samples)}
                results.append({
                    'Supplier': dyn['Supplier'], 'Tariff': dyn['Tariff name'], 'Strategy': 'baseline-no-battery', 
                    'Arbitrage': "N/A", 'Imp_kWh': base_imp_kwh, 'Exp_kWh': base_exp_kwh,
                    'Import': annual_imp_base, 'Export': annual_exp_base, 
                    'June': base_june, 'Dec': base_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                    'Bill': net_bill_base, '_id': tid, 'is_dynamic': True
                })

                for strat_idx in range(5):
                    strategy = all_strategies[strat_idx]
                    imports, exports, soc, is_arb = _run_dynamic_simulation_from_arrays(
                        orig_imports, orig_exports, hours_array, months_array, day_ids,
                        import_prices_arr, fit_rate_arr, strat_idx,
                        usable_cap_kwh, min_soc_kwh, max_soc_kwh, grid_rte, solar_charge_eff,
                        charge_rate, mic, mec
                    )
                    detailed_results[tid][strategy] = {'import': imports, 'export': exports, 'soc': soc}
                    
                    strategy_import_costs, strat_limit_exceeded = _calc_cost_with_overage(imports, import_prices_arr, dyn_is_ev_window, dyn_ev_overage_rate, months_array, False)
                    
                    exp_rev = exports * fit_rate_arr
                    costs_matrix[strat_idx, :] = strategy_import_costs
                    exp_rev_matrix[strat_idx, :] = exp_rev
                    imports_matrix[strat_idx, :] = imports
                    exports_matrix[strat_idx, :] = exports
                    soc_matrix[strat_idx, :] = soc

                    annual_imp_cost = np.sum(strategy_import_costs)
                    annual_exp_rev = np.sum(exp_rev)
                    net_bill = (annual_imp_cost - annual_exp_rev) * scaling_factor + fixed_charges - cash_bonus
                    
                    strat_imp_kwh, strat_exp_kwh = np.sum(imports), np.sum(exports)
                    strat_june = (np.sum(strategy_import_costs[mask_june]) - np.sum(exp_rev[mask_june])) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                    strat_dec = (np.sum(strategy_import_costs[mask_dec]) - np.sum(exp_rev[mask_dec])) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0
                    
                    arb_display = f"{is_arb:.2f} c/kWh" if (strategy not in ['baseline-no-battery', 'self-consumption'] and is_arb is not None and is_arb > 0) else "N/A"
                            
                    results.append({
                        'Supplier': dyn['Supplier'], 'Tariff': dyn['Tariff name'], 'Strategy': strategy, 'Arbitrage': arb_display, 
                        'Imp_kWh': strat_imp_kwh, 'Exp_kWh': strat_exp_kwh,
                        'Import': annual_imp_cost, 'Export': annual_exp_rev, 
                        'June': strat_june, 'Dec': strat_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                        'Bill': net_bill, '_id': tid, 'is_dynamic': True
                    })

                # Ideal Daily Adaptive for Dynamic (Pure Numba JIT)
                ideal_imp, ideal_exp, ideal_soc, ideal_costs, ideal_exp_rev, win_counts = _calc_ideal_daily_adaptive(
                    costs_matrix, exp_rev_matrix, imports_matrix, exports_matrix, soc_matrix, day_ids
                )
                strategy_win_counts = {all_strategies[i]: int(win_counts[i]) for i in range(5)}
                detailed_results[tid]['ideal-daily-adaptive'] = {'import': ideal_imp, 'export': ideal_exp, 'soc': ideal_soc, 'win_counts': strategy_win_counts}
                
                annual_ideal_imp_cost = np.sum(ideal_costs)
                annual_ideal_exp_rev = np.sum(ideal_exp_rev)
                net_bill_ideal = (annual_ideal_imp_cost - annual_ideal_exp_rev) * scaling_factor + fixed_charges - cash_bonus
                
                ideal_june = (np.sum(ideal_costs[mask_june]) - np.sum(ideal_exp_rev[mask_june])) * (30.0 / days_june) + monthly_fixed if days_june > 0 else 0
                ideal_dec = (np.sum(ideal_costs[mask_dec]) - np.sum(ideal_exp_rev[mask_dec])) * (31.0 / days_dec) + monthly_fixed if days_dec > 0 else 0

                results.append({
                    'Supplier': dyn['Supplier'], 'Tariff': dyn['Tariff name'], 'Strategy': 'ideal-daily-adaptive', 
                    'Arbitrage': "Adaptive", 'Imp_kWh': np.sum(ideal_imp), 'Exp_kWh': np.sum(ideal_exp),
                    'Import': annual_ideal_imp_cost, 'Export': annual_ideal_exp_rev, 
                    'June': ideal_june, 'Dec': ideal_dec, 'Fixed': fixed_charges, 'Bonus': cash_bonus,
                    'Bill': net_bill_ideal, '_id': tid, 'is_dynamic': True
                })

            calc_time = time.time() - start_time
            df_res = pd.DataFrame(results)
            
            payload = {
                'df_res': df_res,
                'df_hdf': df_hdf,
                'mprn': mprn_val,
                'meter_serial': meter_val,
                'detailed_results': detailed_results,
                'unique_dates': unique_dates,
                'num_days': num_days,
                'scaling_factor': scaling_factor,
                'total_rows': total_rows,
                'num_tariffs': num_tariffs,
                'calc_time': calc_time,
                'exceeded_plans': exceeded_plans,
                'is_short_duration': is_short_duration,
                'params': params
            }
            self.status_queue.put(("SWEEP_COMPLETE", payload))
        except Exception as e:
            self.status_queue.put(("SWEEP_ERROR", e))

    def on_sweep_finished(self, payload: Dict[str, Any]):
        df_res = payload['df_res']
        self.df_hdf = payload['df_hdf']
        self.mprn = payload['mprn']
        self.meter_serial = payload['meter_serial']
        self.detailed_results = payload['detailed_results']
        self.unique_dates = payload['unique_dates']
        self.current_date_idx = 0
        
        num_days = payload['num_days']
        total_rows = payload['total_rows']
        num_tariffs = payload['num_tariffs']
        calc_time = payload['calc_time']
        exceeded_plans = payload['exceeded_plans']
        is_short_duration = payload['is_short_duration']
        scaling_factor = payload['scaling_factor']

        mem_usage_kb = df_res.memory_usage(deep=True).sum() / 1024.0
        total_sims = num_tariffs * 7
        total_steps = total_rows * total_sims
        
        telemetry = (
            f"[✓] Data Points: {total_rows:,} ({num_days} days)\n"
            f"[✓] Tariffs Evaluated: {num_tariffs}\n"
            f"[✓] Total Simulations: {total_sims:,} runs (inc. Ideal Adaptive)\n"
            f"⚡ Iterations Computed: {total_steps:,} steps\n"
            f"⏱️ CPU Exec Time: {calc_time:.4f} seconds\n"
            f"📊 Data Frame Memory: {mem_usage_kb:.1f} KB"
        )
        if is_short_duration:
            telemetry += f"\n⚠️ Short Data Warning: Data scaled by {scaling_factor:.2f}x for annual calculations."
        
        ev_exceeded_names = sorted(list(set([p.split(' (')[0] for p in exceeded_plans])))
        if ev_exceeded_names:
            telemetry += f"\n⚠️ EV Policy Cap Exceeded: {', '.join(ev_exceeded_names[:2])} (Breached 1k kWh bi-monthly threshold rules)."
                
        self.update_console(telemetry, "#10b981")

        baseline_mask = df_res['Strategy'] == 'baseline-no-battery'
        best_baseline_row = df_res[baseline_mask].loc[df_res[baseline_mask]['Bill'].idxmin()] if not df_res[baseline_mask].empty else pd.DataFrame()
        self.leaderboard_data = pd.concat([df_res[~baseline_mask].copy(), pd.DataFrame([best_baseline_row])]).sort_values(by='Bill').reset_index(drop=True)
        
        self.refresh_kpi_cards(ev_exceeded_names)
        self.populate_treeview()
        self.update_subtabs_from_leaderboard()
        self.update_hdf_graph()
        
        messagebox.showinfo("Success", "Sweep complete! Leaderboard populated.")

    def refresh_kpi_cards(self, ev_exceeded_names=None):
        if self.leaderboard_data is None or self.leaderboard_data.empty: return
        
        df_res = self.leaderboard_data
        baseline_mask = df_res['Strategy'] == 'baseline-no-battery'
        best_base_bill = df_res[baseline_mask]['Bill'].min() if not df_res[baseline_mask].empty else 0
        best_opt_bill = df_res[~baseline_mask]['Bill'].min() if not df_res[~baseline_mask].empty else 0
        
        total_savings = max(0.0, best_base_bill - best_opt_bill)
        winning_row = df_res.iloc[0]
        winning_strategy_name = str(winning_row['Strategy']).replace('-', ' ').title()
        
        self.lbl_kpi_savings.configure(text=f"€{total_savings:,.2f} / yr")
        self.lbl_sub_savings.configure(text=f"Cheapest Baseline: €{best_base_bill:,.2f}")
        
        self.lbl_kpi_strategy.configure(text=winning_strategy_name)
        self.lbl_sub_strategy.configure(text=f"Supplier: {winning_row['Supplier']}")
        
        # Financial ROI Calculation
        roi_p = FinancialROIParams(
            battery_capex=self.roi_params.get('battery_capex', 5500.0),
            inverter_capex=self.roi_params.get('inverter_capex', 1500.0),
            grant_amount=self.roi_params.get('grant_amount', 2100.0),
            electricity_inflation_pct=self.roi_params.get('electricity_inflation_pct', 3.0),
            annual_degradation_pct=self.roi_params.get('annual_degradation_pct', 2.0)
        )
        roi_results = FinancialROICalculator.calculate_roi(total_savings, roi_p)
        
        payback_str = f"{roi_results['payback_years']} Yrs" if roi_results['payback_years'] < 90 else "N/A"
        self.lbl_kpi_payback.configure(text=payback_str)
        self.lbl_sub_payback.configure(text=f"10-Yr ROI: {roi_results['roi_percent']}% (NPV: €{roi_results['npv']:,.0f})")
        
        if ev_exceeded_names:
            self.lbl_kpi_limits.configure(text="Cap Exceeded", text_color=("#d93025", "#f28b82"))
            exceeded_list = ", ".join(ev_exceeded_names[:2])
            self.lbl_sub_limits.configure(text=f"Breached: {exceeded_list}")
        else:
            self.lbl_kpi_limits.configure(text="Nominal", text_color=("#7e22ce", "#d8b4fe"))
            self.lbl_sub_limits.configure(text="Within physical MIC / MEC profiles")

    def populate_treeview(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        if self.leaderboard_data is None: return
        
        for idx, row in self.leaderboard_data.iterrows():
            tags = ('best_baseline',) if row['Strategy'] == 'baseline-no-battery' else ()
            self.tree.insert("", "end", values=(
                idx + 1, row['Supplier'], row['Tariff'], str(row['Strategy']).replace('-', ' ').title(), row['Arbitrage'], 
                f"{row['Imp_kWh']:,.0f}", f"{row['Exp_kWh']:,.0f}", 
                f"€ {row['Import']:,.2f}", f"€ {row['Export']:,.2f}",
                f"€ {row['June']:,.2f}", f"€ {row['Dec']:,.2f}", f"€ {row['Fixed']:,.2f}",
                f"€ {row['Bonus']:,.2f}", f"€ {row['Bill']:,.2f}"
            ), tags=tags)

    def update_subtabs_from_leaderboard(self):
        if not HAS_MATPLOTLIB or self.leaderboard_data is None: return
        
        df_res = self.leaderboard_data
        baseline_mask = df_res['Strategy'] == 'baseline-no-battery'
        top_3 = df_res[~baseline_mask].head(3).reset_index(drop=True)
        
        for i, (_, row) in enumerate(top_3.iterrows()):
            if i >= len(self.top_tabs): break
            tab_ui = self.top_tabs[i]
            tab_ui['internal_id'] = row['_id']
            tab_ui['strategy'] = row['Strategy']
            self.right_notebook.tab(tab_ui['frame'], text=f"  #{i+1}: {row['Supplier']}  ")
            tab_ui['lbl_info'].configure(text=f"{i+1}. {row['Supplier']} - {row['Tariff']}\nWinning Strategy: {str(row['Strategy']).replace('-', ' ').title()}")
        
        df_dynamic = df_res[(df_res.get('is_dynamic', False) == True) & (~baseline_mask)]
        if not df_dynamic.empty:
            self.right_notebook.tab(self.top_tabs[3]['frame'], state='normal')
            best_dyn = df_dynamic.iloc[0]
            tab_ui = self.top_tabs[3]
            tab_ui['internal_id'] = best_dyn['_id']
            tab_ui['strategy'] = best_dyn['Strategy']
            self.right_notebook.tab(tab_ui['frame'], text=f"  Dyn: {best_dyn['Supplier']}  ")
            tab_ui['lbl_info'].configure(text=f"Top Dynamic: {best_dyn['Supplier']} - {best_dyn['Tariff']}\nWinning Strategy: {str(best_dyn['Strategy']).replace('-', ' ').title()}")
        else:
            self.right_notebook.tab(self.top_tabs[3]['frame'], state='hidden')
            
        self.update_daily_charts()

    def update_hdf_graph(self, event=None):
        if not HAS_MATPLOTLIB or self.df_hdf is None: return
        month_sel = self.hdf_month_combo.get()
        ChartManager.render_hdf_profile(self.fig_hdf, self.ax_hdf, self.canvas_hdf, self.df_hdf, month_sel)

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
        if not HAS_MATPLOTLIB or self.df_hdf is None or not len(self.unique_dates): return
        target_date = self.unique_dates[self.current_date_idx]
        
        for tab_ui in self.top_tabs:
            if not tab_ui['internal_id']: continue
            tab_ui['lbl_date'].configure(text=target_date.strftime("%A, %d %b %Y"))
            tab_ui['combo_month'].set(target_date.strftime("%B"))
            
            sim_data = self.detailed_results.get(tab_ui['internal_id'], {}).get(tab_ui['strategy'])
            if sim_data:
                ChartManager.render_daily_chart(tab_ui['fig'], tab_ui['ax1'], tab_ui['ax2'], tab_ui['canvas'], self.df_hdf, sim_data, target_date)

    def export_leaderboard(self):
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            messagebox.showwarning("Warning", "No simulation data available to export.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Results Table")
        if filepath:
            try:
                export_df = self.leaderboard_data.copy().drop(columns=['_id', 'is_dynamic'], errors='ignore')
                export_df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Leaderboard exported successfully to:\n{os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data:\n{str(e)}")

    def export_simulated_hdf(self, local_idx):
        if self.df_hdf is None or not self.top_tabs:
            messagebox.showwarning("Warning", "No simulation data available.")
            return
        tab_ui = self.top_tabs[local_idx]
        tid = tab_ui['internal_id']; strategy = tab_ui['strategy']
        if not tid or not strategy:
            messagebox.showwarning("Warning", "No results mapped to this tab yet.")
            return
        sim_data = self.detailed_results.get(tid, {}).get(strategy)
        if sim_data is None:
            messagebox.showerror("Error", "Simulated data not found for this strategy.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
            title=f"Save Simulated HDF - {strategy.replace('-', ' ').title()}",
            initialfile=f"simulated_hdf_{strategy}.csv"
        )
        if not filepath: return
        try:
            end_times = self.df_hdf.index + pd.Timedelta(minutes=30)
            formatted_times = end_times.strftime('%d/%m/%Y %H:%M')
            import_vals, export_vals = sim_data['import'], sim_data['export']
            mprn_col = str(self.mprn) if self.mprn else "12345678912"
            meter_col = str(self.meter_serial) if self.meter_serial else "SIMULATED_METER"
            rows = []
            for t, imp, exp in reversed(list(zip(formatted_times, import_vals, export_vals))):
                rows.append([mprn_col, meter_col, f"{imp:.4f}", "Active Import Interval (kWh)", t])
                rows.append([mprn_col, meter_col, f"{exp:.4f}", "Active Export Interval (kWh)", t])
            df_export = pd.DataFrame(rows, columns=['MPRN', 'Meter Serial Number', 'Read Value', 'Read Type', 'Read Date and End Time'])
            df_export.to_csv(filepath, index=False, encoding='utf-8')
            messagebox.showinfo("Success", f"Simulated HDF exported successfully to:\n{os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export simulated HDF:\n{str(e)}")

    def export_html_report(self):
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            messagebox.showwarning("Warning", "No simulation results available. Please run a sweep first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html", filetypes=[("HTML Files", "*.html")],
            title="Save Optimization Audit Report", initialfile="battery_optimization_report.html"
        )
        if not filepath: return
        
        try:
            baseline_mask = self.leaderboard_data['Strategy'] == 'baseline-no-battery'
            best_base_bill = self.leaderboard_data[baseline_mask]['Bill'].min() if not self.leaderboard_data[baseline_mask].empty else 0
            best_opt_bill = self.leaderboard_data[~baseline_mask]['Bill'].min() if not self.leaderboard_data[~baseline_mask].empty else 0
            annual_savings = max(0.0, best_base_bill - best_opt_bill)
            
            roi_p = FinancialROIParams(
                battery_capex=self.roi_params.get('battery_capex', 5500.0),
                inverter_capex=self.roi_params.get('inverter_capex', 1500.0),
                grant_amount=self.roi_params.get('grant_amount', 2100.0),
                electricity_inflation_pct=self.roi_params.get('electricity_inflation_pct', 3.0),
                annual_degradation_pct=self.roi_params.get('annual_degradation_pct', 2.0)
            )
            roi_results = FinancialROICalculator.calculate_roi(annual_savings, roi_p)
            roi_results['annual_savings'] = annual_savings
            roi_results['capex'] = self.roi_params.get('battery_capex', 5500.0) + self.roi_params.get('inverter_capex', 1500.0)
            roi_results['grant'] = self.roi_params.get('grant_amount', 2100.0)

            params_dict = {
                'capacity': self.entry_capacity.get(),
                'usable_pct': self.entry_usable_pct.get(),
                'charge_rate': self.entry_charge_rate.get(),
                'min_soc': self.entry_minsoc.get(),
                'max_soc': self.entry_maxsoc.get(),
                'grid_efficiency': self.entry_grid_eff.get(),
                'solar_efficiency': self.entry_solar_eff.get(),
                'region': self.combo_region.get(),
                'mic': self.entry_mic.get(),
                'mec': self.entry_mec.get()
            }
            
            winning_row = self.leaderboard_data.iloc[0]
            
            dual_results = evaluate_dual_tariffs(self.leaderboard_data, self.df_hdf, self.dual_params)
            
            generate_html_report(self.leaderboard_data, winning_row, params_dict, roi_results, dual_results, self.mprn, self.meter_serial, filepath)
            messagebox.showinfo("Success", f"HTML Audit Report generated successfully:\n{os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate HTML report:\n{str(e)}")
