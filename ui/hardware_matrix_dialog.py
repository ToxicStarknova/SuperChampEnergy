import customtkinter as ctk
import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Any, Optional

from core.models import SimulationParams, HardwareExpansionParams, HardwareScenarioResult
from core.engine import run_hardware_expansion_matrix
from ui.components import ToolTip


class HardwareMatrixDialog(ctk.CTkToplevel):
    """
    Modal window presenting the 5x6 Hardware Sensitivity & Expansion Matrix.
    Models combinations of Battery Additions (+0 to +20 kWh) and PV Scalings (100% to 200%).
    """

    def __init__(self, parent, df_hdf: pd.DataFrame, valid_tariffs: pd.DataFrame,
                 dynamic_suppliers: List[Dict[str, Any]], dam_prices_c_kwh: np.ndarray,
                 base_params: SimulationParams):
        super().__init__(parent)

        self.title("⚡ Hardware Sensitivity & Expansion Matrix")
        self.geometry("1180x820")
        self.minsize(1000, 700)

        # Store data references
        self.df_hdf = df_hdf
        self.valid_tariffs = valid_tariffs
        self.dynamic_suppliers = dynamic_suppliers
        self.dam_prices_c_kwh = dam_prices_c_kwh
        self.base_params = base_params

        self.expansion_params = HardwareExpansionParams()
        self.results: List[HardwareScenarioResult] = []

        # Make modal window focused and top
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._recalculate()

    def _build_ui(self):
        # Header Container
        header_frame = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=10)
        header_frame.pack(fill="x", padx=15, pady=10)

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="⚡ Hardware Sensitivity & Expansion Matrix",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#f8fafc"
        )
        title_lbl.pack(side="left", padx=15, pady=12)

        subtitle_lbl = ctk.CTkLabel(
            header_frame,
            text="Evaluate CapEx & Payback for 30 Hardware Scenarios (5 Battery x 6 PV)",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        subtitle_lbl.pack(side="left", padx=5, pady=12)

        # CapEx Inputs Bar
        inputs_frame = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=8)
        inputs_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Battery Cost Entry
        ctk.CTkLabel(inputs_frame, text="Battery (€/kWh):", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=(15, 5), pady=10)
        self.batt_cost_entry = ctk.CTkEntry(inputs_frame, width=90, placeholder_text="300")
        self.batt_cost_entry.insert(0, str(int(self.expansion_params.battery_cost_per_kwh)))
        self.batt_cost_entry.pack(side="left", padx=(0, 15), pady=10)

        # PV Cost Entry
        ctk.CTkLabel(inputs_frame, text="PV Expansion (€/kWp):", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=(10, 5), pady=10)
        self.pv_cost_entry = ctk.CTkEntry(inputs_frame, width=90, placeholder_text="900")
        self.pv_cost_entry.insert(0, str(int(self.expansion_params.pv_cost_per_kwp)))
        self.pv_cost_entry.pack(side="left", padx=(0, 15), pady=10)

        # Baseline PV Entry
        ctk.CTkLabel(inputs_frame, text="Baseline PV (kWp):", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=(10, 5), pady=10)
        self.pv_kwp_entry = ctk.CTkEntry(inputs_frame, width=80, placeholder_text="6.1")
        self.pv_kwp_entry.insert(0, str(self.expansion_params.baseline_pv_kwp))
        self.pv_kwp_entry.pack(side="left", padx=(0, 15), pady=10)

        # Recalculate Button
        recalc_btn = ctk.CTkButton(
            inputs_frame,
            text="🔄 Recalculate Matrix",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3b82f6", hover_color="#2563eb",
            command=self._recalculate
        )
        recalc_btn.pack(side="right", padx=15, pady=10)

        # Summary KPI Cards Frame
        self.kpi_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.kpi_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Scrollable Matrix Frame
        self.matrix_container = ctk.CTkScrollableFrame(self, fg_color="#0f172a", corner_radius=10)
        self.matrix_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _recalculate(self):
        try:
            batt_cost = float(self.batt_cost_entry.get())
            pv_cost = float(self.pv_cost_entry.get())
            baseline_pv = float(self.pv_kwp_entry.get())
        except ValueError:
            batt_cost, pv_cost, baseline_pv = 300.0, 900.0, 6.1

        self.expansion_params.battery_cost_per_kwh = batt_cost
        self.expansion_params.pv_cost_per_kwp = pv_cost
        self.expansion_params.baseline_pv_kwp = baseline_pv

        # Run 30-scenario simulation engine
        self.results = run_hardware_expansion_matrix(
            self.df_hdf, self.valid_tariffs, self.dynamic_suppliers,
            self.dam_prices_c_kwh, self.base_params, self.expansion_params
        )

        self._render_kpis()
        self._render_matrix()

    def _render_kpis(self):
        for widget in self.kpi_frame.winfo_children():
            widget.destroy()

        sweet_spot = next((r for r in self.results if r.is_sweet_spot), None)
        baseline = next((r for r in self.results if r.is_baseline), None)

        if not sweet_spot or not baseline:
            return

        # KPI 1: Sweet Spot Config
        kpi1 = ctk.CTkFrame(self.kpi_frame, fg_color="#065f46", corner_radius=8)
        kpi1.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkLabel(kpi1, text="🏆 SWEET SPOT CONFIG", font=ctk.CTkFont(size=10, weight="bold"), text_color="#6ee7b7").pack(anchor="w", padx=12, pady=(8, 0))
        cfg_str = f"+{int(sweet_spot.battery_addition_kwh)} kWh Batt | +{int(round((sweet_spot.pv_scale_factor-1)*100))}% PV (+{sweet_spot.pv_addition_kwp} kWp)"
        ctk.CTkLabel(kpi1, text=cfg_str, font=ctk.CTkFont(size=13, weight="bold"), text_color="#ffffff").pack(anchor="w", padx=12, pady=(0, 8))

        # KPI 2: CapEx Required
        kpi2 = ctk.CTkFrame(self.kpi_frame, fg_color="#1e293b", corner_radius=8)
        kpi2.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(kpi2, text="💶 EXPANSION CAPEX", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(kpi2, text=f"€{sweet_spot.expansion_capex:,.0f}", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38bdf8").pack(anchor="w", padx=12, pady=(0, 8))

        # KPI 3: Extra Annual Savings
        kpi3 = ctk.CTkFrame(self.kpi_frame, fg_color="#1e293b", corner_radius=8)
        kpi3.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(kpi3, text="📈 INCREMENTAL SAVINGS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(kpi3, text=f"€{sweet_spot.incremental_savings:,.2f} / yr", font=ctk.CTkFont(size=14, weight="bold"), text_color="#4ade80").pack(anchor="w", padx=12, pady=(0, 8))

        # KPI 4: Simple Payback
        kpi4 = ctk.CTkFrame(self.kpi_frame, fg_color="#1e293b", corner_radius=8)
        kpi4.pack(side="left", fill="x", expand=True, padx=(5, 0))
        ctk.CTkLabel(kpi4, text="⏱️ SIMPLE PAYBACK", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(kpi4, text=f"{sweet_spot.simple_payback_years:.1f} Years", font=ctk.CTkFont(size=14, weight="bold"), text_color="#facc15").pack(anchor="w", padx=12, pady=(0, 8))

    def _render_matrix(self):
        for widget in self.matrix_container.winfo_children():
            widget.destroy()

        battery_additions = [0.0, 5.0, 10.0, 15.0, 20.0]
        pv_scale_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        # Top Header Row (Battery Additions)
        lbl_corner = ctk.CTkLabel(
            self.matrix_container, text="PV Scaling \\ Battery",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#94a3b8", width=140
        )
        lbl_corner.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")

        for c_idx, batt_add in enumerate(battery_additions):
            col_txt = f"Baseline (+0 kWh)" if batt_add == 0 else f"+{int(batt_add)} kWh Battery"
            col_lbl = ctk.CTkLabel(
                self.matrix_container, text=col_txt,
                font=ctk.CTkFont(size=12, weight="bold"), text_color="#f8fafc", fg_color="#1e293b", corner_radius=6, height=35
            )
            col_lbl.grid(row=0, column=c_idx + 1, padx=6, pady=6, sticky="nsew")

        # Result Rows (PV Scalings)
        res_map = {(r.battery_addition_kwh, r.pv_scale_factor): r for r in self.results}

        for r_idx, pv_scale in enumerate(pv_scale_factors):
            pv_pct = int(round(pv_scale * 100))
            pv_add = (pv_scale - 1.0) * self.expansion_params.baseline_pv_kwp
            row_txt = f"{pv_pct}% PV Baseline" if pv_scale == 1.0 else f"{pv_pct}% PV (+{pv_add:.2f} kWp)"

            row_lbl = ctk.CTkLabel(
                self.matrix_container, text=row_txt,
                font=ctk.CTkFont(size=12, weight="bold"), text_color="#f8fafc", fg_color="#1e293b", corner_radius=6, width=140
            )
            row_lbl.grid(row=r_idx + 1, column=0, padx=6, pady=6, sticky="nsew")

            for c_idx, batt_add in enumerate(battery_additions):
                res = res_map.get((batt_add, pv_scale))
                if not res:
                    continue

                cell_color = "#1e293b"
                border_color = "#334155"
                border_width = 1

                if res.is_sweet_spot:
                    cell_color = "#064e3b"
                    border_color = "#10b981"
                    border_width = 2
                elif res.is_baseline:
                    cell_color = "#334155"
                    border_color = "#64748b"

                cell_frame = ctk.CTkFrame(
                    self.matrix_container,
                    fg_color=cell_color,
                    border_color=border_color,
                    border_width=border_width,
                    corner_radius=8
                )
                cell_frame.grid(row=r_idx + 1, column=c_idx + 1, padx=6, pady=6, sticky="nsew")

                # Badge Header
                header_txt = "BASELINE" if res.is_baseline else ("🏆 SWEET SPOT" if res.is_sweet_spot else f"CapEx: €{res.expansion_capex:,.0f}")
                header_clr = "#94a3b8" if res.is_baseline else ("#34d399" if res.is_sweet_spot else "#38bdf8")
                ctk.CTkLabel(cell_frame, text=header_txt, font=ctk.CTkFont(size=9, weight="bold"), text_color=header_clr).pack(anchor="w", padx=8, pady=(6, 0))

                # Bill Value
                ctk.CTkLabel(cell_frame, text=f"Bill: €{res.annual_bill:,.0f}/yr", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff").pack(anchor="w", padx=8, pady=(2, 0))

                # Delta Savings & Payback
                if res.is_baseline:
                    sub_txt = "Baseline Reference"
                    sub_clr = "#94a3b8"
                elif res.incremental_savings > 0:
                    sub_txt = f"+€{res.incremental_savings:,.0f}/yr | {res.simple_payback_years:.1f} yrs"
                    sub_clr = "#4ade80"
                else:
                    sub_txt = "No Extra Savings"
                    sub_clr = "#f87171"

                ctk.CTkLabel(cell_frame, text=sub_txt, font=ctk.CTkFont(size=10, weight="bold"), text_color=sub_clr).pack(anchor="w", padx=8, pady=(0, 6))

                # Detailed Tooltip
                tip_text = (
                    f"Configuration: +{int(res.battery_addition_kwh)} kWh Battery | {int(round(res.pv_scale_factor*100))}% PV\n"
                    f"Total Battery: {res.total_battery_capacity_kwh:.1f} kWh | PV Addition: +{res.pv_addition_kwp:.2f} kWp\n"
                    f"Winning Tariff: {res.winning_supplier} {res.winning_tariff}\n"
                    f"Winning Strategy: {res.winning_strategy}\n"
                    f"Annual Bill: €{res.annual_bill:,.2f}\n"
                    f"Incremental Savings vs Baseline: €{res.incremental_savings:,.2f} / year\n"
                    f"Expansion CapEx: €{res.expansion_capex:,.2f}\n"
                    f"Simple Payback: {res.simple_payback_years:.1f} Years"
                )
                ToolTip(cell_frame, tip_text)
