# Home Battery & Tariff Optimizer (V2.5 Professional)

A high-performance Python desktop application built using **CustomTkinter**, **Pandas**, **NumPy**, **Numba**, and **Matplotlib** to model home battery performance, evaluate dispatch strategies, and compare electricity tariffs.

The application parses ESB Networks HDF (Harmonised Data Files) containing 30-minute smart meter interval readings. It simulates battery performance across standard fixed tariffs, custom tariffs, and dynamic wholesale energy plans (Day-Ahead Market), displaying a ranked leaderboard of estimated annual bills, financial payback metrics, and interactive charts.

> [!IMPORTANT]
> **Baseline Load Profile Requirement:** This simulation works best with baseline (pre-battery) load profiles. If your HDF file already contains battery storage or energy arbitraging usage, this will show up as solar export on the file and distort the simulation outputs.

---

## Key Features

* ⚡ **Ultra-Fast Numba JIT Engine (~125ms Execution):** Simulates 370+ annual battery charge/discharge runs across 30-minute interval readings in **~125 milliseconds (~0.12s)** using pure C-speed Numba `@njit(fastmath=True)` array kernels.
* 🔥 **Module-Level Startup Warm-Up:** Pre-compiles Numba LLVM routines on module import to completely eliminate first-run JIT latency during user optimization sweeps.
* 🔄 **Non-Blocking Multithreaded GUI:** Runs optimization sweeps in a background thread to keep the CustomTkinter user interface smooth and responsive with real-time console telemetry updates.
* 🧠 **Oracle EMS Ideal Daily Adaptive Strategy:** Automatically selects the optimal daily strategy across the entire year using pure Numba vector array logic.
* 💰 **Financial Payback & ROI Calculator:** Models equipment CAPEX, government/SEAI grants, electricity price inflation, and 10-year battery capacity degradation curves to project Net Present Value (NPV), Simple Payback Period (years), and 10-Year ROI %.
* 📄 **Interactive HTML Audit Report Export:** Generates formatted HTML audit summary reports complete with hardware parameters, winning tariff breakdowns, 10-year cash flow tables, and leaderboard rankings.
* 📊 **Dynamic Wholesale (DAM) Tariff Modeling:** Integrates Day-Ahead Market (DAM) wholesale hourly pricing and supplier adders, applying 9% Irish VAT to dynamic import tariffs.
* 🚘 **EV Tariff Cap Modeling:** Models bi-monthly promotional caps (e.g. 1,000 kWh limit on cheap EV night rates) and alerts users if thresholds are exceeded.
* 📥 **Simulated HDF Export:** Exports simulated battery import/export profiles back into the ESB HDF format for compatibility with external comparison platforms like EnergyPal.ie.
* 📈 **Data Scaling & Partial Month Handling:** Automatically scales datasets covering fewer than 365 days for accurate annual bill projections.

---

## Operational Charging Strategies

1. **Self-Consumption:** Prioritizes storing excess solar production locally. The battery acts strictly as a solar sponge without force-charging from the grid.
2. **Import-Minimiser:** Force-charges the battery up to max capacity during the lowest cost daily tariff window to cover daytime loads.
3. **Export-Maximiser:** Forces a battery energy dump directly to the grid prior to cheap windows starting to clear space for low-cost power.
4. **Balanced Export Maximiser:** Runs arbitrage dump protocols during spring/summer while preserving winter heating security bounds.
5. **Import-Minimiser (Summer Pass):** Prevents solar generation from charging the battery between March and October to bypass structural round-trip AC/DC conversion losses.
6. **Ideal Daily Adaptive:** Oracle EMS strategy that dynamically selects the single best-performing strategy for each individual day of the year.

---

## Performance & Optimization

| Benchmark Stage | Execution Time (371 Strategy Runs) | Speedup |
| :--- | :--- | :--- |
| **Multiprocessing IPC Overhead** | 13.70 seconds | 0.55x |
| **Single-Threaded Non-JIT Loop** | ~7.50 seconds | 1.0x |
| **Zero-Overhead Numba JIT Engine** | **0.1258 seconds (125 ms)** 🚀 | **~60x Faster** |

---

## Repository & Project Architecture

The application uses a clean, decoupled module structure:

```text
SuperChampEnergy/
├── core/
│   ├── engine.py           # Numba JIT simulation kernels & dispatch algorithms
│   ├── models.py           # SimulationParams, FinancialROIParams, and ROICalculator
│   ├── parsers.py          # ESB HDF, Tariff DB, DAM, and Dynamic Supplier parsers
│   └── report_generator.py # Interactive HTML audit report exporter
├── ui/
│   ├── app.py              # Main CustomTkinter Application & Threading manager
│   ├── components.py       # Tooltips, Custom Tariff Dialog, and Financial ROI Dialog
│   └── charts.py           # Matplotlib figures & dark/light theme manager
├── HDF_calckWh_SAMPLE_23-06-2025.csv             # Sample smart meter interval readings
├── energypal tarriffs 03072026.csv             # Sample tariff database
├── Dynamic tarrif supplier fixed costs_260626.csv # Sample dynamic supplier costs
├── DAM prices MAy 2026.csv                        # Sample Day-Ahead Market prices
└── Super_Champ_Optimizer.py                       # Main application launcher entrypoint
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ToxicStarknova/SuperChampEnergy.git
   cd SuperChampEnergy
   ```

2. **Install required dependencies:**
   Python 3.8+ is required. Install the necessary libraries:
   ```bash
   pip install pandas numpy numba matplotlib customtkinter
   ```

3. **Run the application:**
   ```bash
   python Super_Champ_Optimizer.py
   ```

---

## How to Use

1. **Select Source Files:**
   - **ESB HDF:** Select your ESB HDF CSV file (or use `HDF_calckWh_SAMPLE_23-06-2025.csv`).
   - **Tariff DB:** Select a tariff spreadsheet database (such as `energypal tarriffs 03072026.csv`), or click **+ Create Custom Tariff** to add manual rates.
   - **DAM & Dynamic Adders (Optional):** Load Day-Ahead Market prices (`DAM prices MAy 2026.csv`) and Dynamic Supplier costs (`Dynamic tarrif supplier fixed costs_260626.csv`).
2. **Hardware & Grid Settings:** Enter your battery capacity (kWh), inverter charge rate (kW), SoC bounds (%), round-trip efficiency (%), and MIC/MEC grid limits.
3. **Financial ROI Setup (Optional):** Click **⚙️ Financial ROI Setup** in the top header to enter equipment CAPEX, SEAI grant amounts, and electricity inflation expectations.
4. **Run Optimization:** Click **Run Optimization Sweep** to compute and display results in real time.
5. **Export & Report:** Click **📄 Export HTML Report** to save a complete audit summary report, or **⬇ Export Table to CSV** to save leaderboard results.

---

## Disclaimer

This tool is designed for estimation and comparison purposes. Actual household battery performance and utility bills may vary based on weather variations, consumption habits, and utility pricing updates.
