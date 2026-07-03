# Home Battery & Tariff Optimizer

A Python desktop application built using Tkinter, Pandas, and Numba to model home battery performance and compare electricity tariffs.

The application parses ESB Networks HDF (Harmonised Data Files) containing 30-minute smart meter interval readings. It simulates battery performance across standard fixed tariffs, custom tariffs, and dynamic wholesale energy plans, and displays a ranked leaderboard of estimated annual bills.

> [!IMPORTANT]
> **Baseline Load Profile Requirement:** This simulation works best with baseline (pre-battery) load profiles. If your HDF file already contains battery storage or energy arbitraging usage, this will show up as solar export on the file and distort the simulation outputs.

---

## Features

* **Battery Dispatch Modeling:** Simulates battery charging/discharging states, constraints, and efficiency losses across an entire year of 30-minute intervals.
* **Tariff Comparison:** Compares fixed, custom, and dynamic wholesale tariffs side-by-side.
* **Dynamic Wholesale (DAM) Modeling:** Models Day-Ahead Market (DAM) wholesale pricing and supplier adders, applying a 9% VAT rate to dynamic tariffs for comparison.
* **EV Tariff Cap Modeling:** Models bi-monthly promotional caps (e.g. 1000 kWh limit on cheap EV rates) and displays if thresholds are exceeded.
* **Simulated HDF Export:** Exports the simulated battery import/export profiles back into the ESB HDF format for compatibility with external tariff comparison tools like EnergyPal.ie.
* **Data Scaling:** Automatically scales datasets covering fewer than 365 days to show annualized bill projections.
* **Local Processing:** Runs locally on your machine without external network requirements (except optionally loading files).

---

## Simulation Strategies

1. **Self-Consumption:** Storing excess solar power without force-charging from the grid.
2. **Import-Minimiser:** Force-charging the battery during cheap tariff windows to cover daytime loads.
3. **Export-Maximiser:** Discharging the battery to the grid before cheap rate windows start to clear space for low-cost power.
4. **Balanced Export Maximiser:** Force-charges during cheap windows and exports battery power to the grid during peak tariff hours to maximize feed-in-tariff (FIT) profits, while preserving battery for winter heating.
5. **Import-Minimiser (Summer Pass):** Prevents solar from charging the battery between March and October to bypass round-trip AC/DC conversion losses.

---

## Sample Data Files Included

The repository contains anonymized sample data files to test the optimizer:
* **`HDF_calckWh_SAMPLE_23-06-2025.csv`**: Anonymized 30-minute interval smart meter reading data.
* **`energypal tarriffs 03072026.csv`**: A sample tariff spreadsheet database downloaded from EnergyPal.ie containing standard smart plans.
* **`Dynamic tarrif supplier fixed costs_260626.csv`**: Fixed supply cost parameters for dynamic plans.
* **`DAM prices MAy 2026.csv`**: Sample Day-Ahead Market pricing. To update this for future months, download the monthly market reports from [SEMOpx Reports](https://www.semopx.com/news/monthly-market-report-may-2026), copy the first 3 columns of the "Auction_to" sheet of the report Excel document, and save as a CSV.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ToxicStarknova/SuperChampEnergy.git
   cd SuperChampEnergy
   ```

2. **Install dependencies:**
   Python 3.8+ is required. Install the necessary libraries:
   ```bash
   pip install pandas numpy numba matplotlib
   ```

3. **Run the application:**
   ```bash
   python Super_Champ_V20.py
   ```

---

## How to Use

1. **HDF File:** Select your ESB HDF file (or use the provided `HDF_calckWh_SAMPLE_23-06-2025.csv`).
2. **Tariff DB:** Select a tariff database CSV file (such as the provided `energypal tarriffs 03072026.csv`), or use the **Create Custom Tariff** button to enter plan rates manually.
3. **DAM & Dynamic Adders (Optional):** Load the Day-Ahead Market prices and Dynamic Supplier fixed costs files to enable dynamic tariff comparison.
4. **Hardware Configuration:** Enter your battery capacity, charge rate limits, depth of discharge bounds, and round-trip efficiency percentages.
5. **Run Sweep:** Click **Run Optimization Sweep** to compute and display results.

---

## Disclaimer

This tool is designed for estimation and comparison purposes. Actual household battery performance and utility bills may vary based on weather variations, consumption habits, and utility pricing updates.
