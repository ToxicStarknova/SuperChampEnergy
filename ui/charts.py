import customtkinter as ctk
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

HAS_MATPLOTLIB = False
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


class ChartManager:
    """Manages Matplotlib charts and dark/light theme applications."""
    
    @staticmethod
    def get_theme_colors():
        mode = ctk.get_appearance_mode()
        if mode == "Dark":
            return {
                'fig_bg': "#2b2b2b",
                'ax_bg': "#1e1e1e",
                'text_color': "#ffffff",
                'grid_color': "#3a3a3a"
            }
        else:
            return {
                'fig_bg': "#ebebeb",
                'ax_bg': "#ffffff",
                'text_color': "#000000",
                'grid_color': "#cbd5e1"
            }

    @staticmethod
    def apply_theme_to_axis(ax, colors):
        ax.set_facecolor(colors['ax_bg'])
        ax.xaxis.label.set_color(colors['text_color'])
        ax.yaxis.label.set_color(colors['text_color'])
        ax.tick_params(colors=colors['text_color'])
        for spine in ax.spines.values():
            spine.set_color(colors['grid_color'])

    @classmethod
    def render_hdf_profile(cls, fig, ax, canvas, df_hdf: pd.DataFrame, month_sel: str):
        if not HAS_MATPLOTLIB or df_hdf is None or df_hdf.empty:
            return
            
        df_target = df_hdf
        if month_sel != "All Year":
            if month_sel in MONTH_NAMES:
                m_num = MONTH_NAMES.index(month_sel) + 1
                df_target = df_hdf[df_hdf.index.month == m_num]
                if df_target.empty: return

        hourly_avg = df_target.groupby(df_target.index.hour).mean() * 2.0
        ax.clear()
        
        colors = cls.get_theme_colors()
        fig.patch.set_facecolor(colors['fig_bg'])
        cls.apply_theme_to_axis(ax, colors)
        
        ax.title.set_color(colors['text_color'])
        ax.plot(hourly_avg.index, hourly_avg['consumption'], label="Avg Grid Import (kW)", color="#4f46e5", linewidth=2.5)
        ax.plot(hourly_avg.index, hourly_avg['generation'], label="Avg Grid Export (kW)", color="#10b981", linewidth=2.5)
        ax.set_title(f"Average Load Profile: {month_sel}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour"); ax.set_ylabel("Power (kW)")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.5, color=colors['grid_color'])
        ax.legend(facecolor=colors['ax_bg'], edgecolor=colors['grid_color'], labelcolor=colors['text_color'])
        fig.tight_layout()
        canvas.draw()

    @classmethod
    def render_daily_chart(cls, fig, ax1, ax2, canvas, df_hdf: pd.DataFrame, sim_data: Dict[str, Any], target_date):
        if not HAS_MATPLOTLIB or df_hdf is None:
            return
            
        mask = (df_hdf.index.date == target_date)
        if not np.any(mask): return
        
        hours = df_hdf.index[mask].hour + df_hdf.index[mask].minute / 60.0
        orig_imp = df_hdf['consumption'].values[mask] * 2.0
        orig_exp = df_hdf['generation'].values[mask] * 2.0

        ax1.clear(); ax2.clear()
        colors = cls.get_theme_colors()
        
        fig.patch.set_facecolor(colors['fig_bg'])
        cls.apply_theme_to_axis(ax1, colors)
        cls.apply_theme_to_axis(ax2, colors)
        
        ax1.plot(hours, orig_imp, color="gray", linestyle="--", alpha=0.6, label="Orig. House Load")
        ax1.plot(hours, orig_exp, color="lightgreen", linestyle="--", alpha=0.6, label="Orig. Solar Export")
        ax1.plot(hours, sim_data['import'][mask] * 2.0, color="#ef4444", linewidth=2, label="Rev. Grid Import")
        ax1.plot(hours, sim_data['export'][mask] * 2.0, color="#10b981", linewidth=2, label="Rev. Grid Export")
        
        ax2.fill_between(hours, 0, sim_data['soc'][mask], color="#f59e0b", alpha=0.15)
        ax2.plot(hours, sim_data['soc'][mask], color="#f59e0b", linewidth=1.5, label="Battery SoC (%)")
        
        ax1.set_ylabel("Power (kW)"); ax2.set_ylabel("SoC (%)"); ax2.set_ylim(0, 105)
        ax1.set_xticks(range(0, 25, 2))
        ax1.grid(True, linestyle=":", alpha=0.7, color=colors['grid_color'])
        
        l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8, facecolor=colors['ax_bg'], edgecolor=colors['grid_color'], labelcolor=colors['text_color'])
        fig.tight_layout()
        canvas.draw()

    @classmethod
    def render_adaptive_heatmap(cls, fig, ax1, ax2, canvas, daily_winners, all_strategies, unique_dates):
        if not HAS_MATPLOTLIB or daily_winners is None or not len(unique_dates):
            return
            
        ax1.clear(); ax2.clear()
        colors = cls.get_theme_colors()
        fig.patch.set_facecolor(colors['fig_bg'])
        cls.apply_theme_to_axis(ax1, colors)
        
        # Hide the second axis since we don't need it
        ax2.set_visible(False)
        
        # Plot a scatter of the winning strategies
        days = np.arange(len(daily_winners))
        
        # Map colors for each strategy
        strat_colors = ["#475569", "#ef4444", "#eab308", "#f97316", "#3b82f6", "#10b981"]
        
        for i, strat in enumerate(all_strategies):
            mask = (daily_winners == i)
            if np.any(mask):
                ax1.scatter(days[mask], daily_winners[mask], color=strat_colors[i % len(strat_colors)], label=strat.replace('-', ' ').title(), s=20)
                
        # Format axes
        ax1.set_yticks(range(len(all_strategies)))
        ax1.set_yticklabels([s.replace('-', ' ').title() for s in all_strategies], fontsize=8)
        ax1.set_title("Annual Distribution of Winning Strategies (Ideal Adaptive)", fontsize=11, fontweight="bold", color=colors['text_color'])
        
        # Add month boundaries as vertical lines and setup X-axis labels
        if len(unique_dates) == len(days):
            months = np.array([d.month for d in unique_dates])
            month_starts = np.where(months[:-1] != months[1:])[0] + 1
            
            # Calculate midpoints for month labels
            all_starts = np.insert(month_starts, 0, 0)
            midpoints = []
            labels = []
            for i in range(len(all_starts)):
                start = all_starts[i]
                end = all_starts[i+1] if i+1 < len(all_starts) else len(days)
                midpoints.append((start + end) / 2)
                labels.append(MONTH_NAMES[months[start]-1][:3])
                
            for ms in month_starts:
                ax1.axvline(x=ms, color=colors['grid_color'], linestyle=":", alpha=0.5)
                
            ax1.set_xticks(midpoints)
            ax1.set_xticklabels(labels, fontsize=9)
            ax1.set_xlabel("Month of Year")
        else:
            ax1.set_xlabel("Day of Year")
                
        ax1.grid(True, axis='y', linestyle="--", alpha=0.3, color=colors['grid_color'])
        ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15), ncol=3, fontsize=8, facecolor=colors['ax_bg'], edgecolor=colors['grid_color'], labelcolor=colors['text_color'])
        fig.tight_layout()
        canvas.draw()
