# Home Battery & Tariff Optimizer

A high-performance Python desktop application (built with Tkinter, Pandas, and Numba JIT) designed to help homeowners with solar PV systems identify the most cost-effective electricity tariffs and optimize their home battery setups. 

By parsing official ESB Networks HDF (Harmonised Data Files) containing 30-minute smart meter interval reads, the tool simulates battery performance across multiple standard fixed tariffs, custom tariffs, and dynamic wholesale energy plans to rank them in an interactive leaderboard.

---

## Key Features

* **Blazing-Fast Simulation Engine:** Uses Numba's JIT compilation (`@njit`) to simulate an entire year of half-hourly battery charging/discharging states, constraints, and tariff calculations in under 0.05 seconds per tariff.
* **Comprehensive Tariff Sweep:** Compares fixed tariffs, custom-defined plans, and dynamic wholesale plans side-by-side.
* **Dynamic Wholesale (DAM) Modeling:** Incorporates Day-Ahead Market (DAM) wholesale pricing and supplier adders. Automatically applies a **9% VAT rate** to dynamic tariffs for a fair, direct comparison with consumer-facing fixed tariffs.
* **Smart EV Cap & Overage Rules:** Accurately models bimonthly promo caps (e.g. Energia's 1000 kWh limit on EV rates) and flags/warns users in the telemetry panel if they will exceed their promo cap.
* **Simulated HDF Export:** Exports simulated battery imports and exports back into an ESB-compatible HDF format. This allows users to upload their simulated file to tariff comparison websites (like [EnergyPal.ie](https://www.energypal.ie)) for a complete audit including standing charges, PSO levies, and cash bonuses.
* **Matplotlib Interactive Visualization:**
  * **HDF Base Profile:** View average load and generation profile by month or year.
  * **Interactive Daily Viewer:** Alternates color-coded lines representing original house load, solar export, simulated grid import, simulated grid export, and battery State of Charge (SoC %).
* **Smart Data Scaling:** Automatically detects if the uploaded HDF covers less than a full year (330 days) and scales the simulated net electricity costs to a 365-day equivalent for an accurate annual bill projection.
* **100% Offline & Private:** All file parsing, JIT compilation, and simulations run entirely locally on your machine.

---

## Simulation Strategies

1. **Self-Consumption:** Prioritizes storing excess solar power. Never force-charges from the grid.
2. **Import-Minimiser:** Force-charges the battery during the cheapest daily window to cover daytime loads.
3. **Import-Minimiser (Summer Pass):** Force-charges during cheap windows in winter, but checks if arbitrage is profitable before bypassing solar charging from March to October (avoiding AC-DC-AC conversion losses).
4. **Balanced Export Maximiser:** Force-charges the battery during cheap windows and exports battery power to the grid during peak tariff hours to maximize feed-in-tariff (FIT) profits. Preserves battery for household heating loads during winter (Nov-Feb).

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ToxicStarknova/SuperChampEnergy.git
   cd SuperChampEnergy
   ```

2. **Install Python dependencies:**
   Make sure you have Python 3.8+ installed. Install the required numerical and plotting libraries:
   ```bash
   pip install pandas numpy numba matplotlib
   ```

3. **Run the application:**
   ```bash
   python Super_Champ_V14.py
   ```

---

## How to Use

1. **Download HDF:** Log in to your ESB Networks portal and download your smart meter data. Choose **"30-minute readings in calculated kWh"** as a CSV file.
2. **Launch & Load:**
   * Select your **ESB HDF** file.
   * Load the standard fixed **Tariff DB** (CSV).
   * *(Optional)* Load **DAM Prices** (wholesale) and **Dynamic Fixed Adders** (CSV) to evaluate wholesale energy plans.
3. **Configure Hardware:** Set your battery capacity (kWh), inverter charge rate (kW), minimum/maximum State of Charge (SoC %), and round-trip efficiency (RTE %).
4. **Run Optimization:** Click **Run Optimization Sweep**. The tool compiles the engine telemetry and ranks all plans by net annual bill.
5. **HDF Export:** Navigate to the daily visualizer tabs and click **Export Simulated HDF** to download your custom, battery-simulated smart meter CSV.

---

## Disclaimer

This tool is designed for estimation and design-comparison purposes. Actual household battery performance and utility bills may vary based on weather variations, changes in personal energy consumption habits, and utility pricing updates.
