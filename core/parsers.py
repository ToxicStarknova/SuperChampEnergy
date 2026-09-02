import re
import json
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]

def parse_hdf(file_path: str) -> Tuple[pd.DataFrame, str, str]:
    """Parses ESB Networks Harmonised Data Files (HDF) with 30-minute interval readings."""
    header_idx = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            lower_line = line.lower()
            if 'read date' in lower_line and 'read type' in lower_line:
                header_idx = i
                break
                
    df_raw = pd.read_csv(file_path, skiprows=header_idx, engine='c', skipinitialspace=True)
    df_raw.columns = [c.strip().replace('"', '') for c in df_raw.columns]
    
    mprn_col = next((c for c in df_raw.columns if 'mprn' in c.lower()), None)
    meter_col = next((c for c in df_raw.columns if 'serial' in c.lower()), None)
    mprn_val = "00000000000"
    if mprn_col and not df_raw[mprn_col].dropna().empty:
        mprn_val = str(df_raw[mprn_col].dropna().iloc[0])
    meter_val = "00000000"
    if meter_col and not df_raw[meter_col].dropna().empty:
        meter_val = str(df_raw[meter_col].dropna().iloc[0])

    date_col = next((c for c in df_raw.columns if 'read date' in c.lower()), None)
    type_col = next((c for c in df_raw.columns if 'read type' in c.lower()), None)
    val_col = next((c for c in df_raw.columns if 'read val' in c.lower()), None)
    
    if not date_col or not type_col or not val_col:
        raise ValueError("Invalid HDF file structure: Required columns 'Read Date', 'Read Type', or 'Read Value' missing.")

    cleaned_dates = df_raw[date_col].astype(str).str.replace('"', '').str.strip()
    try:
        df_raw['timestamp'] = pd.to_datetime(cleaned_dates, format='%d/%m/%Y %H:%M')
    except Exception:
        df_raw['timestamp'] = pd.to_datetime(cleaned_dates, format='mixed', dayfirst=True)

    df_raw['timestamp'] = df_raw['timestamp'] - pd.Timedelta(minutes=30)
    df_raw[val_col] = pd.to_numeric(df_raw[val_col], errors='coerce')
    
    df_raw['ReadType_Clean'] = df_raw[type_col].astype(str).str.lower()
    df_raw.loc[df_raw['ReadType_Clean'].str.contains('import'), 'Type'] = 'consumption'
    df_raw.loc[df_raw['ReadType_Clean'].str.contains('export'), 'Type'] = 'generation'
    
    df_raw = df_raw.dropna(subset=['Type', 'timestamp'])
    df = df_raw.pivot_table(index='timestamp', columns='Type', values=val_col, aggfunc='sum').fillna(0.0)
    
    if 'consumption' not in df.columns: df['consumption'] = 0.0
    if 'generation' not in df.columns: df['generation'] = 0.0
        
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df, mprn_val, meter_val


def filter_last_12_full_months(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataset to the last 12 full calendar months."""
    if df.empty: return df
    latest_ts = df.index[-1]
    end_date = datetime(latest_ts.year, latest_ts.month, 1)
    start_date = datetime(end_date.year - 1, end_date.month, 1)
    filtered_df = df[(df.index >= start_date) & (df.index < end_date)]
    return filtered_df if not filtered_df.empty else df


def normalize_tariff_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names in tariff spreadsheet databases."""
    if df.empty:
        return df
    
    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(' ', '').replace('_', '')
        if c_clean == 'supplier': col_map[col] = 'Supplier'
        elif c_clean in ['planname', 'tariffname']: col_map[col] = 'Tariff name'
        elif c_clean == 'plantype': col_map[col] = 'Plan type'
        elif c_clean == 'supplyregion': col_map[col] = 'Supply Region'
        elif c_clean == 'standingcharge': col_map[col] = 'Standing charge'
        elif c_clean == 'psolevy': col_map[col] = 'PSO Levy'
        elif c_clean == 'cashbonus': col_map[col] = 'Cash bonus'
        elif c_clean == 'dayunit': col_map[col] = 'Day unit'
        elif c_clean == 'nightunit': col_map[col] = 'Night unit'
        elif c_clean == 'peakunit': col_map[col] = 'Peak unit'
        elif c_clean == 'evunit': col_map[col] = 'Ev unit'
        elif c_clean == 'evoverageunit': col_map[col] = 'Ev overage unit'
        elif c_clean == 'fitunit': col_map[col] = 'Fit unit'
        elif c_clean == 'extra': col_map[col] = 'Extra'
        
    df = df.rename(columns=col_map)
    
    if 'Extra' not in df.columns or df['Extra'].isna().all() or (df['Extra'].astype(str).str.strip() == '').all():
        df['Extra'] = ""
        sch_col = next((c for c in df.columns if 'schedule' in str(c).lower()), None)
        if sch_col:
            for idx, row in df.iterrows():
                sch_id = str(row[sch_col]).strip().lower()
                extra_list = []
                ev_match = re.search(r'ev-(\d+)-(\d+)', sch_id)
                if ev_match:
                    extra_list.append(f"ev_{ev_match.group(1)}_{ev_match.group(2)}")
                if 'no-peak-weekend' in sch_id or 'no_peak_weekend' in sch_id:
                    extra_list.append("no_peak_weekend")
                if extra_list:
                    df.at[idx, 'Extra'] = json.dumps(extra_list)
                    
    return df


def get_half_hourly_rates_for_row(row: pd.Series, date_range: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, bool]:
    """Generates half-hourly price series and EV window mask for a single tariff row."""
    plan_type = str(row['Plan type']).strip().lower()
    
    has_missing_rates = pd.isna(row.get('Day unit'))
    if plan_type == 'day/night':
        has_missing_rates = has_missing_rates or pd.isna(row.get('Night unit'))
    elif plan_type == 'smart':
        has_missing_rates = has_missing_rates or pd.isna(row.get('Night unit')) or pd.isna(row.get('Peak unit'))
        
    has_unknown_type = plan_type not in ['24h', 'day/night', 'smart']
    
    try:
        day_rate = float(row['Day unit']) / 100.0 if not pd.isna(row.get('Day unit')) else 0.0
    except (ValueError, TypeError):
        day_rate = 0.0
        has_missing_rates = True
        
    try:
        peak_rate = float(row['Peak unit']) / 100.0 if not pd.isna(row.get('Peak unit')) else day_rate
    except (ValueError, TypeError):
        peak_rate = day_rate
        has_missing_rates = True
        
    try:
        night_rate = float(row['Night unit']) / 100.0 if not pd.isna(row.get('Night unit')) else day_rate
    except (ValueError, TypeError):
        night_rate = day_rate
        has_missing_rates = True
        
    try:
        ev_rate = float(row['Ev unit']) / 100.0 if not pd.isna(row.get('Ev unit')) else None
    except (ValueError, TypeError):
        ev_rate = None
        has_missing_rates = True
        
    raw_ev_overage = row.get('Ev overage unit')
    try:
        if pd.notna(raw_ev_overage) and str(raw_ev_overage).strip() != "":
            ev_overage_rate = float(raw_ev_overage) / 100.0
            has_overage_penalty = True
        else:
            ev_overage_rate = ev_rate if ev_rate is not None else day_rate
            has_overage_penalty = False
    except (ValueError, TypeError):
        ev_overage_rate = day_rate
        has_overage_penalty = False
        has_missing_rates = True
    
    extra_tags = []
    if not pd.isna(row.get('Extra')):
        try:
            extra_val = str(row['Extra']).replace("''", '"')
            extra_tags = json.loads(extra_val)
        except Exception:
            extra_tags = []

    ev_hours = set()
    if ev_rate is not None:
        for tag in extra_tags:
            if isinstance(tag, str) and tag.startswith("ev_"):
                parts = tag.split("_")
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    start_h, end_h = int(parts[1]), int(parts[2])
                    if start_h < end_h:
                        ev_hours.update(range(start_h, end_h))
                    else:
                        ev_hours.update(range(start_h, 24))
                        ev_hours.update(range(0, end_h))

    hour_array = date_range.hour.values
    is_weekend_array = (date_range.weekday.values >= 5)
    is_night_mask = (hour_array >= 23) | (hour_array < 8)
    is_peak_mask = (hour_array >= 17) & (hour_array < 19)

    prices_arr = np.full(len(date_range), day_rate, dtype=np.float64)
    is_ev_window = np.zeros(len(date_range), dtype=np.bool_)
    
    if plan_type == 'day/night':
        prices_arr[is_night_mask] = night_rate
    elif plan_type == 'smart':
        prices_arr[is_night_mask] = night_rate
        if "no_peak_weekend" in extra_tags:
            peak_cond = is_peak_mask & (~is_weekend_array)
        else:
            peak_cond = is_peak_mask
        prices_arr[peak_cond] = peak_rate
        
        if ev_rate is not None and len(ev_hours) > 0:
            ev_mask = np.zeros(len(date_range), dtype=np.bool_)
            for h in ev_hours:
                ev_mask |= (hour_array == h)
            prices_arr[ev_mask] = ev_rate
            is_ev_window = ev_mask
            
    if len(prices_arr) == 0:
        prices_arr = np.full(len(date_range), day_rate, dtype=np.float64)
        has_unknown_type = True
        
    return prices_arr, is_ev_window, ev_overage_rate, has_overage_penalty, has_unknown_type, has_missing_rates


def prepare_dam(hdf_idx: pd.DatetimeIndex, dam_file: str) -> np.ndarray:
    """Parses SEMOpx Day-Ahead Market wholesale prices and resamples to HDF index in c/kWh."""
    df_dam = pd.read_csv(dam_file, low_memory=False)
    df_dam.columns = df_dam.columns.str.strip().str.replace('"', '')
    
    if 'auction' in df_dam.columns:
        df_dam = df_dam[df_dam['auction'].astype(str).str.strip() == 'DAM']
    
    dt_series = pd.to_datetime(df_dam['timestamp'].astype(str).str.replace('"', ''), format='mixed')
    if dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_convert('Europe/Dublin').dt.tz_localize(None)
        
    df_dam['datetime'] = dt_series
    df_dam = df_dam.set_index('datetime').sort_index()
    df_dam = df_dam[~df_dam.index.duplicated(keep='first')]
    
    dam_resampled = df_dam[['price_eur']].reindex(hdf_idx, method='ffill').bfill()
    return dam_resampled['price_eur'].values / 10.0  # Eur/MWh to c/kWh


def parse_dynamic_suppliers(file_path: str, region: str) -> List[Dict[str, Any]]:
    """Parses dynamic supplier fixed cost parameters and standing charges."""
    df_fixed = pd.read_csv(file_path)
    df_fixed.columns = df_fixed.columns.str.strip().str.replace('"', '')
    suppliers = []
    
    for _, row in df_fixed.iterrows():
        supplier_name = row['c/KWh, ex. VAT']
        if pd.isna(supplier_name) or str(supplier_name).strip() == "":
            continue
            
        sc_str = str(row['SC € p.a. ex. VAT Urban (Rural)']).replace('€', '').replace(',', '').strip()
        urban_sc, rural_sc = 0.0, 0.0
        match = re.search(r'([\d\.]+)\s*\(([\d\.]+)\)', sc_str)
        if match:
            urban_sc, rural_sc = float(match.group(1)), float(match.group(2))
        else:
            try:
                urban_sc = rural_sc = float(re.findall(r'[\d\.]+', sc_str)[0])
            except Exception:
                urban_sc, rural_sc = 300.0, 350.0
            
        fit_val = row['FIT']
        try:
            fit_unit = float(fit_val)
        except ValueError:
            fit_unit = str(fit_val).strip()
            
        suppliers.append({
            'Supplier': supplier_name,
            'Tariff name': 'Dynamic Wholesale',
            'Plan type': 'dynamic',
            'Night': float(row['Night']),
            'Day': float(row['Day']),
            'Peak': float(row['Peak']),
            'Fit unit': fit_unit,
            'FIT Payment time': str(row.get('FIT Payment time', '')).strip(),
            'Standing charge': rural_sc if region.lower() == 'rural' else urban_sc,
            'PSO Levy': 0.0,
            'Cash bonus': 0.0
        })
    return suppliers
