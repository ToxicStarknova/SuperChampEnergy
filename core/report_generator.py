import os
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

from core.models import DualTariffResult

def generate_html_report(leaderboard_df: pd.DataFrame, winning_row: pd.Series, 
                         params_dict: Dict[str, Any], roi_dict: Dict[str, Any], 
                         dual_results: Optional[List[DualTariffResult]] = None,
                         mprn: str = "00000000000", meter_serial: str = "00000000", 
                         output_filepath: str = "battery_optimization_report.html") -> str:
    """Generates an HTML Audit Summary Report for saving or printing to PDF."""
    
    top_5_df = leaderboard_df.head(5)
    top_5_rows_html = ""
    for idx, row in top_5_df.iterrows():
        top_5_rows_html += f"""
        <tr>
            <td style="text-align: center; font-weight: bold;">#{idx + 1}</td>
            <td>{row['Supplier']}</td>
            <td>{row['Tariff']}</td>
            <td>{str(row['Strategy']).replace('-', ' ').title()}</td>
            <td style="text-align: right;">{row['Imp_kWh']:,.0f} kWh</td>
            <td style="text-align: right;">{row['Exp_kWh']:,.0f} kWh</td>
            <td style="text-align: right; font-weight: bold; color: #1e293b;">€{row['Bill']:,.2f}</td>
        </tr>
        """

    dual_rows_html = ""
    if dual_results:
        for idx, res in enumerate(dual_results[:5]):
            profit_style = "color: #16a34a; font-weight: bold;" if res.extra_savings_vs_single_best > 0 else "color: #64748b;"
            profit_str = f"+ €{res.extra_savings_vs_single_best:,.2f}" if res.extra_savings_vs_single_best > 0 else f"€{res.extra_savings_vs_single_best:,.2f}"
            dual_rows_html += f"""
            <tr>
                <td style="text-align: center; font-weight: bold;">#{idx + 1}</td>
                <td><strong>{res.winter_supplier}</strong> ({res.winter_tariff})</td>
                <td>{res.winter_strategy}</td>
                <td><strong>{res.summer_supplier}</strong> ({res.summer_tariff})</td>
                <td>{res.summer_strategy}</td>
                <td style="text-align: right;">€{res.total_exit_fees:,.2f}</td>
                <td style="text-align: right; font-weight: bold;">€{res.net_annual_bill:,.2f}</td>
                <td style="text-align: right; {profit_style}">{profit_str}</td>
            </tr>
            """
    else:
        dual_rows_html = "<tr><td colspan='8' style='text-align: center; color: #94a3b8;'>No seasonal dual-tariff combinations evaluated.</td></tr>"

    winning_supplier_name = winning_row.get('Supplier', 'Unknown') if hasattr(winning_row, 'get') else str(winning_row['Supplier'])
    winning_tariff_name = winning_row.get('Tariff', winning_row.get('Tariff name', 'Standard Tariff')) if hasattr(winning_row, 'get') else 'Standard Tariff'
    winning_strategy_name = str(winning_row.get('Strategy', 'Self Consumption') if hasattr(winning_row, 'get') else winning_row['Strategy']).replace('-', ' ').title()
    
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Home Battery & Tariff Optimization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 25px;
            color: #1e293b; background-color: #f8fafc;
        }}
        .header {{
            background: linear-gradient(135deg, #4f46e5, #3b82f6);
            color: white; padding: 25px; border-radius: 12px;
            margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .grid {{
            display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 20px; margin-bottom: 25px;
        }}
        .card {{
            background: white; padding: 20px; border-radius: 10px;
            border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        .card h3 {{ margin-top: 0; color: #334155; font-size: 16px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }}
        .kpi-box {{
            background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
            padding: 15px; text-align: center; margin-bottom: 20px;
        }}
        .kpi-title {{ font-size: 12px; color: #166534; font-weight: bold; text-transform: uppercase; }}
        .kpi-val {{ font-size: 32px; color: #15803d; font-weight: bold; margin: 4px 0; }}
        table {{
            width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px;
        }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #f1f5f9; color: #475569; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        .footer {{
            text-align: center; font-size: 12px; color: #94a3b8; margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ SuperChampEnergy Optimization Report</h1>
        <p>Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} | MPRN: {mprn} | Meter: {meter_serial}</p>
    </div>

    <div class="kpi-box">
        <div class="kpi-title">Optimal Annual Savings Opportunity</div>
        <div class="kpi-val">€{roi_dict.get('annual_savings', 0.0):,.2f} / yr</div>
        <div style="font-size: 13px; color: #166534;">Winning Tariff: <strong>{winning_supplier_name} - {winning_tariff_name}</strong> ({winning_strategy_name})</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>⚙️ Hardware & Grid Parameters</h3>
            <table>
                <tr><td>Battery Capacity:</td><td><strong>{params_dict.get('capacity', 30.0)} kWh</strong> ({params_dict.get('usable_pct', 100)}% Usable)</td></tr>
                <tr><td>Inverter Charge Rate:</td><td><strong>{params_dict.get('charge_rate', 10.0)} kW</strong></td></tr>
                <tr><td>State of Charge Limits:</td><td><strong>{params_dict.get('min_soc', 10)}% - {params_dict.get('max_soc', 100)}%</strong></td></tr>
                <tr><td>Grid / Solar Efficiency:</td><td><strong>{params_dict.get('grid_efficiency', 95)}% / {params_dict.get('solar_efficiency', 85)}%</strong></td></tr>
                <tr><td>Supply Region & Limits:</td><td><strong>{str(params_dict.get('region', 'rural')).title()}</strong> (MIC: {params_dict.get('mic', 18)}kW / MEC: {params_dict.get('mec', 6)}kW)</td></tr>
            </table>
        </div>

        <div class="card">
            <h3>📊 Financial Payback & ROI (10-Year)</h3>
            <table>
                <tr><td>Equipment & Installation CAPEX:</td><td>€{roi_dict.get('capex', 7000.0):,.2f}</td></tr>
                <tr><td>Government / SEAI Grant:</td><td>- €{roi_dict.get('grant', 2100.0):,.2f}</td></tr>
                <tr><td>Net Investment:</td><td><strong>€{roi_dict.get('net_investment', 4900.0):,.2f}</strong></td></tr>
                <tr><td>Simple Payback Period:</td><td><strong>{roi_dict.get('payback_years', 0.0)} Years</strong></td></tr>
                <tr><td>10-Year Cumulative Savings:</td><td><strong>€{roi_dict.get('ten_year_savings', 0.0):,.2f}</strong></td></tr>
                <tr><td>10-Year Net Return (ROI):</td><td><strong style="color: #16a34a;">{roi_dict.get('roi_percent', 0.0)}%</strong> (NPV: €{roi_dict.get('npv', 0.0):,.2f})</td></tr>
            </table>
        </div>
    </div>

    <div class="card">
        <h3>🏆 Top Single-Tariff Leaderboard</h3>
        <table>
            <thead>
                <tr>
                    <th style="text-align: center;">Rank</th>
                    <th>Supplier</th>
                    <th>Tariff Name</th>
                    <th>Strategy</th>
                    <th style="text-align: right;">Import kWh</th>
                    <th style="text-align: right;">Export kWh</th>
                    <th style="text-align: right;">Est. Annual Bill</th>
                </tr>
            </thead>
            <tbody>
                {top_5_rows_html}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h3>🔀 Seasonal Dual-Tariff Early Exit Analysis</h3>
        <p style="font-size: 12px; color: #64748b; margin-top: 0;">Evaluates switching between a Winter Tariff (Nov-Mar) and Summer Tariff (Apr-Oct), deducting 2 × €50 early contract exit fees.</p>
        <table>
            <thead>
                <tr>
                    <th style="text-align: center;">Rank</th>
                    <th>Winter Plan (Nov-Mar)</th>
                    <th>Winter Strategy</th>
                    <th>Summer Plan (Apr-Oct)</th>
                    <th>Summer Strategy</th>
                    <th style="text-align: right;">Exit Fees</th>
                    <th style="text-align: right;">Net Annual Bill</th>
                    <th style="text-align: right;">Net Extra Profit</th>
                </tr>
            </thead>
            <tbody>
                {dual_rows_html}
            </tbody>
        </table>
    </div>

    <div class="footer">
        Generated by SuperChampEnergy Home Battery & Tariff Optimization Tool V2.5. Reports are for estimation purposes.
    </div>
</body>
</html>
"""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(report_html)
        
    return output_filepath
