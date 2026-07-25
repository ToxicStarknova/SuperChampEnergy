import os
import json
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from typing import Callable, Dict, Any

class ToolTip:
    """Creates a sleek, flat-styled hover tooltip window for modern UIs."""
    def __init__(self, widget, text: str):
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
        tw.attributes("-topmost", True)
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#1e293b", foreground="#f8fafc",
                         relief=tk.FLAT, borderwidth=0,
                         font=("Segoe UI" if os.name == "nt" else "Helvetica", 9, "normal"), 
                         padx=10, pady=8, wraplength=340)
        label.pack()

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class CustomTariffDialog:
    """Dialog for creating user-defined custom electricity tariffs."""
    def __init__(self, parent, current_region: str, on_save_callback: Callable[[Dict[str, Any]], None]):
        self.parent = parent
        self.current_region = current_region
        self.on_save_callback = on_save_callback
        
        self.dlg = ctk.CTkToplevel(self.parent)
        self.dlg.title("Add Custom Tariff")
        self.dlg.geometry("540x520")
        self.dlg.grab_set()
        
        self._build_ui()
        
    def _build_ui(self):
        frame = ctk.CTkFrame(self.dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(frame, text="Supplier Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ent_sup = ctk.CTkEntry(frame, width=150); self.ent_sup.insert(0, "Custom Energy"); self.ent_sup.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Tariff Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ent_name = ctk.CTkEntry(frame, width=150); self.ent_name.insert(0, "My Custom Plan"); self.ent_name.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Plan Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.combo_type = ctk.CTkComboBox(frame, values=["smart", "day/night", "24h"], state="readonly", width=140)
        self.combo_type.set("smart"); self.combo_type.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Standing Charge (€/yr):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.ent_sc = ctk.CTkEntry(frame, width=100); self.ent_sc.insert(0, "300"); self.ent_sc.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Day Unit (c/kWh):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.ent_day = ctk.CTkEntry(frame, width=100); self.ent_day.insert(0, "35.0"); self.ent_day.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Night Unit (c/kWh):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.ent_night = ctk.CTkEntry(frame, width=100); self.ent_night.insert(0, "20.0"); self.ent_night.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Peak Unit (c/kWh):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.ent_peak = ctk.CTkEntry(frame, width=100); self.ent_peak.insert(0, "45.0"); self.ent_peak.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="EV/Boost Unit (c/kWh):").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.ent_ev = ctk.CTkEntry(frame, width=100); self.ent_ev.insert(0, "10.0"); self.ent_ev.grid(row=7, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="EV Overage (c/kWh):").grid(row=7, column=2, sticky=tk.W, pady=5, padx=5)
        self.ent_ev_overage = ctk.CTkEntry(frame, width=80); self.ent_ev_overage.insert(0, "35.0"); self.ent_ev_overage.grid(row=7, column=3, sticky=tk.W, pady=5)

        ctk.CTkLabel(frame, text="Export FIT (c/kWh):").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.ent_fit = ctk.CTkEntry(frame, width=100); self.ent_fit.insert(0, "18.0"); self.ent_fit.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        ctk.CTkLabel(frame, text="Cash Bonus (€):").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.ent_bonus = ctk.CTkEntry(frame, width=100); self.ent_bonus.insert(0, "0.0"); self.ent_bonus.grid(row=9, column=1, sticky=tk.W, pady=5)
        
        ev_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ev_frame.grid(row=10, column=0, columnspan=4, sticky=tk.W, pady=12)
        ctk.CTkLabel(ev_frame, text="EV Start Hour (0-23):").pack(side=tk.LEFT)
        self.ent_ev_start = ctk.CTkEntry(ev_frame, width=45); self.ent_ev_start.insert(0, "2"); self.ent_ev_start.pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(ev_frame, text="End Hour:").pack(side=tk.LEFT)
        self.ent_ev_end = ctk.CTkEntry(ev_frame, width=45); self.ent_ev_end.insert(0, "5"); self.ent_ev_end.pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(frame, text="Save & Add to Database", fg_color="#4f46e5", hover_color="#4338ca", command=self._save_tariff).grid(row=11, column=0, columnspan=4, pady=15, sticky=tk.EW)

    def _save_tariff(self):
        try:
            new_tariff = {
                'Supplier': self.ent_sup.get().strip(), 
                'Tariff name': self.ent_name.get().strip() + " (Custom)", 
                'Plan type': self.combo_type.get(),
                'Standing charge': float(self.ent_sc.get() or 0.0), 
                'PSO Levy': 0.0, 
                'Cash bonus': float(self.ent_bonus.get() or 0.0), 
                'Day unit': float(self.ent_day.get() or 0.0),
                'Night unit': float(self.ent_night.get() or 0.0), 
                'Peak unit': float(self.ent_peak.get() or 0.0), 
                'Ev unit': float(self.ent_ev.get() or 0.0),
                'Ev overage unit': float(self.ent_ev_overage.get() or 0.0), 
                'Fit unit': float(self.ent_fit.get() or 0.0), 
                'Supply Region': self.current_region, 
                'Extra': f'["ev_{int(self.ent_ev_start.get())}_{int(self.ent_ev_end.get())}"]' if self.ent_ev.get() else ""
            }
            self.on_save_callback(new_tariff)
            messagebox.showinfo("Success", f"Added Custom Tariff: {new_tariff['Tariff name']}\nIncluded in next sweep.")
            self.dlg.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please ensure all rates and hours are valid numbers.", parent=self.dlg)


class FinancialROIDialog:
    """Dialog for configuring battery CAPEX, grants, and viewing 10-year ROI cash flows."""
    def __init__(self, parent, roi_params: Dict[str, Any], on_update_callback: Callable[[Dict[str, Any]], None]):
        self.parent = parent
        self.roi_params = roi_params
        self.on_update_callback = on_update_callback
        
        self.dlg = ctk.CTkToplevel(self.parent)
        self.dlg.title("Financial ROI & Payback Configuration")
        self.dlg.geometry("480x420")
        self.dlg.grab_set()
        
        self._build_ui()

    def _build_ui(self):
        frame = ctk.CTkFrame(self.dlg, corner_radius=0, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(frame, text="Financial ROI & CAPEX Settings", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))

        ctk.CTkLabel(frame, text="Battery Cost (€):").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.ent_battery = ctk.CTkEntry(frame, width=120); self.ent_battery.insert(0, str(self.roi_params.get('battery_capex', 5500.0))); self.ent_battery.grid(row=1, column=1, sticky=tk.W, pady=6)
        
        ctk.CTkLabel(frame, text="Inverter Cost (€):").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.ent_inverter = ctk.CTkEntry(frame, width=120); self.ent_inverter.insert(0, str(self.roi_params.get('inverter_capex', 1500.0))); self.ent_inverter.grid(row=2, column=1, sticky=tk.W, pady=6)
        
        ctk.CTkLabel(frame, text="SEAI / Gov Grant (€):").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.ent_grant = ctk.CTkEntry(frame, width=120); self.ent_grant.insert(0, str(self.roi_params.get('grant_amount', 2100.0))); self.ent_grant.grid(row=3, column=1, sticky=tk.W, pady=6)

        ctk.CTkLabel(frame, text="Electricity Inflation (%/yr):").grid(row=4, column=0, sticky=tk.W, pady=6)
        self.ent_inflation = ctk.CTkEntry(frame, width=120); self.ent_inflation.insert(0, str(self.roi_params.get('electricity_inflation_pct', 3.0))); self.ent_inflation.grid(row=4, column=1, sticky=tk.W, pady=6)

        ctk.CTkLabel(frame, text="Battery Degradation (%/yr):").grid(row=5, column=0, sticky=tk.W, pady=6)
        self.ent_degradation = ctk.CTkEntry(frame, width=120); self.ent_degradation.insert(0, str(self.roi_params.get('annual_degradation_pct', 2.0))); self.ent_degradation.grid(row=5, column=1, sticky=tk.W, pady=6)

        ctk.CTkButton(frame, text="Update Financial Model", fg_color="#4f46e5", hover_color="#4338ca", command=self._save_roi).grid(row=6, column=0, columnspan=2, pady=20, sticky=tk.EW)

    def _save_roi(self):
        try:
            updated = {
                'battery_capex': float(self.ent_battery.get()),
                'inverter_capex': float(self.ent_inverter.get()),
                'grant_amount': float(self.ent_grant.get()),
                'electricity_inflation_pct': float(self.ent_inflation.get()),
                'annual_degradation_pct': float(self.ent_degradation.get())
            }
            self.on_update_callback(updated)
            messagebox.showinfo("Success", "Financial ROI parameters updated.", parent=self.dlg)
            self.dlg.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric parameters.", parent=self.dlg)
