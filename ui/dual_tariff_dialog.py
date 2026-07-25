import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from typing import List, Dict, Any, Callable

from core.models import DualTariffParams, DualTariffResult

class DualTariffDialog:
    """Dialog window for displaying Seasonal Dual-Tariff Early Exit Switcher analysis."""
    def __init__(self, parent, dual_results: List[DualTariffResult], 
                 current_params: DualTariffParams, on_recalculate_callback: Callable[[DualTariffParams], List[DualTariffResult]]):
        self.parent = parent
        self.dual_results = dual_results
        self.current_params = current_params
        self.on_recalculate_callback = on_recalculate_callback
        
        self.dlg = ctk.CTkToplevel(self.parent)
        self.dlg.title("🔀 Seasonal Dual-Tariff Contract Switcher Analysis")
        self.dlg.geometry("1100x650")
        self.dlg.grab_set()
        
        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        # Header Controls Frame
        header = ctk.CTkFrame(self.dlg, corner_radius=10)
        header.pack(fill=tk.X, padx=15, pady=15)
        
        ctk.CTkLabel(header, text="Seasonal Contract Switcher Analysis", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=12, pady=(10, 4))
        ctk.CTkLabel(header, text="Evaluates switching between a Winter Tariff (cheap import) and Summer Tariff (high FIT export), accounting for contract cancellation fees.",
                     font=("Segoe UI", 10, "italic"), text_color=("#475569", "#94a3b8")).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=12, pady=(0, 10))

        ctk.CTkLabel(header, text="Exit Fee per Switch (€):").grid(row=2, column=0, sticky=tk.W, padx=(12, 4), pady=6)
        self.ent_fee = ctk.CTkEntry(header, width=80)
        self.ent_fee.insert(0, str(self.current_params.exit_fee_per_switch))
        self.ent_fee.grid(row=2, column=1, sticky=tk.W, pady=6)

        ctk.CTkButton(header, text="Re-evaluate Dual-Tariffs", fg_color="#4f46e5", hover_color="#4338ca", command=self._recalculate).grid(row=2, column=2, padx=15, pady=6)

        # Table Container
        table_frame = ctk.CTkFrame(self.dlg, corner_radius=8)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        cols = ("rank", "winter", "w_strat", "summer", "s_strat", "fees", "bill", "savings")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="none")
        
        self.tree.heading("rank", text="#")
        self.tree.heading("winter", text="Winter Plan (Nov-Mar)")
        self.tree.heading("w_strat", text="Winter Strategy")
        self.tree.heading("summer", text="Summer Plan (Apr-Oct)")
        self.tree.heading("s_strat", text="Summer Strategy")
        self.tree.heading("fees", text="Exit Fees")
        self.tree.heading("bill", text="Net Annual Bill")
        self.tree.heading("savings", text="Net Extra Profit")

        self.tree.column("rank", width=35, anchor=tk.CENTER)
        self.tree.column("winter", width=220, anchor=tk.W)
        self.tree.column("w_strat", width=140, anchor=tk.CENTER)
        self.tree.column("summer", width=220, anchor=tk.W)
        self.tree.column("s_strat", width=140, anchor=tk.CENTER)
        self.tree.column("fees", width=80, anchor=tk.E)
        self.tree.column("bill", width=110, anchor=tk.E)
        self.tree.column("savings", width=110, anchor=tk.E)

        self.tree.tag_configure('profitable', background='#e6f4ea', foreground='#137333')

        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _populate_table(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        if not self.dual_results: return
        
        for idx, res in enumerate(self.dual_results[:30]):
            tags = ('profitable',) if res.extra_savings_vs_single_best > 0 else ()
            w_name = f"{res.winter_supplier} - {res.winter_tariff}"
            s_name = f"{res.summer_supplier} - {res.summer_tariff}"
            
            savings_str = f"+ €{res.extra_savings_vs_single_best:,.2f}" if res.extra_savings_vs_single_best > 0 else f"€{res.extra_savings_vs_single_best:,.2f}"
            
            self.tree.insert("", "end", values=(
                idx + 1, w_name, res.winter_strategy, s_name, res.summer_strategy,
                f"€{res.total_exit_fees:,.2f}", f"€{res.net_annual_bill:,.2f}", savings_str
            ), tags=tags)

    def _recalculate(self):
        try:
            fee = float(self.ent_fee.get())
            self.current_params.exit_fee_per_switch = fee
            self.dual_results = self.on_recalculate_callback(self.current_params)
            self._populate_table()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid numeric exit fee.", parent=self.dlg)
