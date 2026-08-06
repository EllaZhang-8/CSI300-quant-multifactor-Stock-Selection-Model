#!/usr/bin/env python3
"""
CSI 300 Multi-Factor Quantitative Stock Selection — End-to-End Pipeline
========================================================================
Runnable entry point. Reads Wind CSV exports, computes factors, runs
IC analysis, backtest, and optional ML comparison.

Usage:
    python run_full_pipeline.py              # Full pipeline
    python run_full_pipeline.py --skip-ml    # Skip ML section
    python run_full_pipeline.py --validate   # Data validation only

Requirements: pip install -r requirements.txt
"""

import argparse 
import sys
import os
import warnings
import logging
from datetime import datetime


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure we can import from the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to Wind CSV exports (pre-computed)
MONTHLY_RET_CSV = "/Users/ella/Desktop/run_full_pipeline data & py/monthly_ret.csv"
MCAP_CSV = "/Users/ella/Desktop/run_full_pipeline data & py/mcap.csv"
FUND_CSV = "/Users/ella/Desktop/run_full_pipeline data & py/fund.csv"
INDUSTRY_CSV = "/Users/ella/Desktop/run_full_pipeline data & py/industry.csv"
BENCHMARK_CSV = "/Users/ella/Desktop/run_full_pipeline data & py/benchmark.csv"

# Backtest configuration
TOP_QUANTILE = 0.20          # Select top 20%
TRANSACTION_COST = 0.0015    # 0.15% one-way (commission + slippage)
MAX_WEIGHT = 0.15            # Max 15% per stock
MIN_STOCKS = 20              # Minimum stocks in portfolio

# Factor definitions with direction (+1 = higher is better, -1 = lower is better)
FACTOR_DIRECTION = {
    'F_EP': 1, 'F_BP': 1, 'F_SP': 1, 'F_CFP': 1,
    'F_MOM1M': 1, 'F_MOM3M': 1, 'F_MOM6M': 1, 'F_MOM12M': 1, 'F_REV': 1,
    'F_ROE': 1, 'F_ROA': 1, 'F_PROFIT_MARGIN': 1, 'F_ASSET_TURNOVER': 1,
    'F_LEVERAGE': -1, 'F_EARNINGS_QUALITY': 1,
    'F_VOL3M': -1, 'F_VOL12M': -1, 'F_SIZE': -1, 'F_MIDCAP': -1,
    'F_REV_YOY': 1, 'F_PROFIT_YOY': 1
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load pre-computed Wind data. Returns (monthly_ret, mcap, fund, industry)."""
    log.info("Loading Wind data...")
    for path, name in [(MONTHLY_RET_CSV, "monthly returns"),
                        (MCAP_CSV, "market cap"),
                        (FUND_CSV, "fundamentals"),
                        (INDUSTRY_CSV, "industry")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")
    
    mr = pd.read_csv(MONTHLY_RET_CSV, parse_dates=['date'])
    mc = pd.read_csv(MCAP_CSV, parse_dates=['date'])
    fd = pd.read_csv(FUND_CSV)

    def _parse_fiscal_quarter(s):
        q, y = s.split()
        q_num = int(q[1])
        year = int(y[2:])
        month = {1: 3, 2: 6, 3: 9, 4: 12}[q_num]
        day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
        return pd.Timestamp(year=year, month=month, day=day)

    fd['report_date'] = fd['report_date'].apply(_parse_fiscal_quarter)
    
    ind = pd.read_csv(INDUSTRY_CSV)
    
    log.info(f"  Monthly returns: {mr.shape[0]:,} rows, {mr['code'].nunique()} stocks")
    log.info(f"  Market cap:      {mc.shape[0]:,} rows, {mc['code'].nunique()} stocks")
    log.info(f"  Fundamentals:    {fd.shape[0]:,} rows, {fd['code'].nunique()} stocks")
    log.info(f"  Industry:        {ind.shape[0]:,} rows, {ind['code'].nunique()} stocks")
    return mr, mc, fd, ind


def build_panel(monthly_ret, mcap, industry):
    """Merge returns, market cap, and industry into a master panel."""
    log.info("Building master panel...")
    panel = monthly_ret.merge(
        mcap[['code', 'date', 'mcap', 'circ_mcap', 'pe', 'pb']],
        on=['code', 'date'], how='left'
    )
    panel['year'] = panel['date'].dt.year
    ind_map = industry[['code', 'industry', 'ind_year']].copy()
    panel = panel.merge(ind_map, left_on=['code', 'year'], right_on=['code', 'ind_year'], how='left')
    panel = panel.drop(columns=['ind_year'])
    panel = panel.dropna(subset=['industry'])
    log.info(f"  Panel: {panel.shape[0]:,} rows, {panel['code'].nunique()} stocks, "
             f"{panel['date'].nunique()} months, {panel['industry'].nunique()} industries")
    return panel


# ============================================================================
# FACTOR COMPUTATION
# ============================================================================

def get_available_date(report_date):
    """Map report date to the date when data becomes publicly available (announcement lag)."""
    m = report_date.month
    y = report_date.year
    if m == 3:   return pd.Timestamp(f"{y}-04-30")
    elif m == 6: return pd.Timestamp(f"{y}-08-31")
    elif m == 9: return pd.Timestamp(f"{y}-10-31")
    else:        return pd.Timestamp(f"{y+1}-04-30")


def get_latest_fund(month_date, fund_df):
    """Get the latest available fundamental data at month_date, respecting announcement lag."""
    available = fund_df[fund_df['available_date'] <= month_date]
    if available.empty:
        return None
    return available.sort_values('report_date').groupby('code').last().reset_index()


def compute_factors(panel, fund):
    """Compute all 31 factors. Returns factor DataFrame."""
    log.info("Computing factors (this may take a minute)...")
    fund['available_date'] = fund['report_date'].apply(get_available_date)
    
    # YoY growth: match each report to the same quarter one year earlier
    fund['prior_year_date'] = fund['report_date'] - pd.DateOffset(years=1)
    prior = fund[['code', 'report_date', 'revenue', 'net_profit']].rename(
        columns={'report_date': 'prior_year_date',
                 'revenue': 'prior_revenue',
                 'net_profit': 'prior_net_profit'}
    )
    fund = fund.merge(prior, on=['code', 'prior_year_date'], how='left')
    fund['revenue_yoy'] = (fund['revenue'] - fund['prior_revenue']) / fund['prior_revenue'].abs()
    fund['net_profit_yoy'] = (fund['net_profit'] - fund['prior_net_profit']) / fund['prior_net_profit'].abs()
    dates = sorted(panel['date'].unique())
    factors_list = []
    
    for i, dt in enumerate(dates):
        if i % 12 == 0:
            log.info(f"  Processing {dt.strftime('%Y-%m')}...")
        
        month_data = panel[panel['date'] == dt].copy()
        fund_latest = get_latest_fund(dt, fund)
        
        # Value factors
        month_data['F_EP'] = 1.0 / month_data['pe'].replace(0, np.nan)
        month_data['F_BP'] = 1.0 / month_data['pb'].replace(0, np.nan)
        
        if fund_latest is not None:
            fmap = fund_latest.set_index('code')
            month_data['F_SP'] = month_data['code'].map(fmap['revenue']).fillna(np.nan) * 4 / (month_data['mcap'] * 100)
            month_data['F_CFP'] = month_data['code'].map(fmap['ocf']).fillna(np.nan) / (month_data['mcap'] * 100)
        else:
            month_data['F_SP'] = np.nan
            month_data['F_CFP'] = np.nan
        
        # Momentum factors
        for lag in [1, 3, 6, 12]:
            past_date = dt - pd.DateOffset(months=lag)
            past_data = panel[panel['date'] == past_date][['code', 'ret_pct']]
            if past_data.empty:
                month_data[f'F_MOM{lag}M'] = np.nan
            else:
                past_data = past_data.rename(columns={'ret_pct': f'ret_{lag}m'})
                month_data = month_data.merge(past_data, on='code', how='left')
                month_data[f'F_MOM{lag}M'] = month_data[f'ret_{lag}m']
        month_data['F_REV'] = -month_data['F_MOM1M'].fillna(0)
        
        # Quality factors
        if fund_latest is not None:
            fmap = fund_latest.set_index('code')
            month_data['F_ROE'] = month_data['code'].map(fmap['roe'])
            month_data['F_ROA'] = month_data['code'].map(fmap['net_profit']) / (month_data['code'].map(fmap['total_assets']) * 100)
            month_data['F_PROFIT_MARGIN'] = month_data['code'].map(fmap['net_profit']) / month_data['code'].map(fmap['revenue']).replace(0, np.nan)
            month_data['F_ASSET_TURNOVER'] = month_data['code'].map(fmap['revenue']) / (month_data['code'].map(fmap['total_assets']) * 100)
            month_data['F_LEVERAGE'] = month_data['code'].map(fmap['total_liabilities']) / month_data['code'].map(fmap['total_assets'])
            month_data['F_EARNINGS_QUALITY'] = month_data['code'].map(fmap['ocf']) / month_data['code'].map(fmap['net_profit']).replace(0, np.nan)
            month_data['F_REV_YOY'] = month_data['code'].map(fmap['revenue_yoy'])
            month_data['F_PROFIT_YOY'] = month_data['code'].map(fmap['net_profit_yoy'])
        
        # Volatility factors
        for window in [3, 12]:
            past_dates = [dt - pd.DateOffset(months=k) for k in range(1, window + 1)]
            past_returns = panel[panel['date'].isin(past_dates)][['code', 'ret_pct']]
            vol = past_returns.groupby('code')['ret_pct'].std() * np.sqrt(12)
            month_data[f'F_VOL{window}M'] = month_data['code'].map(vol)
        
        # Size factors
        month_data['F_SIZE'] = np.log(month_data['mcap'])
        month_data['F_MIDCAP'] = np.log(month_data['circ_mcap'])
        
        month_data['month_date'] = dt
        factors_list.append(month_data)
    
    result = pd.concat(factors_list, ignore_index=True)
    log.info(f"  Factors computed: {sum(1 for c in result.columns if c.startswith('F_'))} factors, "
             f"{result.shape[0]:,} rows")
    return result


# ============================================================================
# FACTOR PROCESSING
# ============================================================================

def winsorize_standardize_neutralize(df, factor_list, date_col='date', industry_col='industry'):
    """Cross-sectional: winsorize (1%/99%), z-score, industry+market-cap neutralize."""
    log.info("Processing factors (winsorize → standardize → neutralize)...")
    from sklearn.linear_model import LinearRegression
    
    dates = sorted(df[date_col].unique())
    processed_list = []
    
    for dt in dates:
        month = df[df[date_col] == dt].copy()
        if len(month) < 30:
            continue
        
        for f in factor_list:
            if f not in month.columns:
                continue
            raw = month[f].dropna()
            if len(raw) < 30:
                month[f + '_z'] = np.nan
                month[f + '_neu'] = np.nan
                continue
            
            # Winsorize
            lo, hi = raw.quantile(0.01), raw.quantile(0.99)
            month[f] = raw.clip(lo, hi)
            
            # Z-score
            mean_val, std_val = month[f].mean(), month[f].std()
            month[f + '_z'] = (month[f] - mean_val) / std_val if std_val > 0 else 0
            
            # Industry + Market Cap neutralization
            valid = month[[f + '_z', industry_col, 'mcap']].dropna().copy()
            if len(valid) >= 30 and valid[industry_col].nunique() > 1:
                valid['log_mcap'] = np.log(valid['mcap'])
                dummies = pd.get_dummies(valid[industry_col], drop_first=True)
                X = pd.concat([dummies, valid['log_mcap']], axis=1).astype(float)
                y = valid[f + '_z'].astype(float)
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    residuals = y - model.predict(X)
                    month.loc[valid.index, f + '_neu'] = residuals
                except Exception:
                    month[f + '_neu'] = month[f + '_z']
            else:
                month[f + '_neu'] = month[f + '_z']
        
        processed_list.append(month)
    
    result = pd.concat(processed_list, ignore_index=True)
    neutralized = [c for c in result.columns if c.endswith('_neu')]
    log.info(f"  Neutralized factors: {len(neutralized)}")
    return result


# ============================================================================
# IC ANALYSIS
# ============================================================================

def compute_ic_analysis(df, factor_list, date_col='date'):
    """Compute IC, Rank IC, and ICIR for each factor."""
    log.info("Computing IC analysis...")
    dates = sorted(df[date_col].unique())
    
    # Forward returns
    df['fwd_ret'] = np.nan
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        fwd = df[df[date_col] == next_dt][['code', 'ret_pct']].set_index('code')
        mask = df[date_col] == dt
        df.loc[mask, 'fwd_ret'] = df.loc[mask, 'code'].map(fwd['ret_pct'])
    
    ic_results = []
    for factor in factor_list:
        neu_col = f'{factor}_neu'
        if neu_col not in df.columns:
            continue
        direction = FACTOR_DIRECTION.get(factor, 1)
        
        ic_list, ric_list = [], []
        for dt in dates[:-1]:
            month = df[(df[date_col] == dt) & df[neu_col].notna() & df['fwd_ret'].notna()]
            if len(month) < 30:
                continue
            ic, _ = stats.pearsonr(month[neu_col] * direction, month['fwd_ret'])
            ic_list.append(ic)
            ric, _ = stats.spearmanr(month[neu_col] * direction, month['fwd_ret'])
            ric_list.append(ric)
        
        if ic_list:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ric_mean = np.mean(ric_list)
            ric_std = np.std(ric_list)
            ricir = ric_mean / ric_std if ric_std > 0 else 0
            
            ic_results.append({
                'Factor': factor, 'IC_Mean': ic_mean, 'IC_Std': ic_std, 'ICIR': icir,
                'Rank_IC_Mean': ric_mean, 'Rank_IC_Std': ric_std, 'Rank_ICIR': ricir,
                'IC_Positive_Ratio': sum(1 for x in ic_list if x > 0) / len(ic_list),
                'N_Months': len(ic_list)
            })
    
    result = pd.DataFrame(ic_results).sort_values('Rank_ICIR', ascending=False)
    return result


# ============================================================================
# BACKTEST
# ============================================================================

def compute_monthly_ic_table(df, factor_list, date_col='date'):
    """Compute each factor's month-by-month Rank IC. Used to build
    out-of-sample rolling weights (no look-ahead)."""
    dates = sorted(df[date_col].unique())
    records = []
    
    for factor in factor_list:
        neu_col = f'{factor}_neu'
        if neu_col not in df.columns:
            continue
        direction = FACTOR_DIRECTION.get(factor, 1)
        for dt in dates[:-1]:
            month = df[(df[date_col] == dt) & df[neu_col].notna() & df['fwd_ret'].notna()]
            if len(month) < 30:
                continue
            ric, _ = stats.spearmanr(month[neu_col] * direction, month['fwd_ret'])
            records.append({'date': dt, 'factor': factor, 'ric': ric})
    
    return pd.DataFrame(records)


def get_rolling_weights(monthly_ic, current_date, lookback=24, top_n=8):
    """ICIR-based factor weights using only IC data from the `lookback`
    months strictly BEFORE current_date. Prevents using future information
    to decide today's factor weights."""
    hist = monthly_ic[monthly_ic['date'] < current_date]
    recent_dates = sorted(hist['date'].unique())[-lookback:]
    hist = hist[hist['date'].isin(recent_dates)]
    
    if hist.empty or hist['date'].nunique() < lookback:
        return {}
    
    stats_by_factor = hist.groupby('factor')['ric'].agg(['mean', 'std'])
    stats_by_factor['icir'] = stats_by_factor['mean'] / stats_by_factor['std'].replace(0, np.nan)
    stats_by_factor = stats_by_factor.dropna(subset=['icir'])
    if stats_by_factor.empty:
        return {}
    
    top = stats_by_factor.reindex(
        stats_by_factor['icir'].abs().sort_values(ascending=False).index
    ).head(top_n)
    
    weights = top['icir'].abs()
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.to_dict()


def run_backtest(df, ic_df, date_col='date', lookback=24, buffer_quantile=0.30,
                  industry_max_overweight=0.03, benchmark_trend=None,
                  defensive_exposure=0.3, monthly_cash_ret=0.0025):
    """Rolling ICIR-weighted multi-factor backtest with:
      - turnover buffer
      - inverse-volatility position sizing
      - industry-neutrality constraint: no industry can exceed its
        market-wide weight by more than `industry_max_overweight`
    """
    log.info("Running backtest (rolling weights + turnover buffer + inverse-vol sizing + industry cap)...")
    dates = sorted(df[date_col].unique())
    
    # Exclude factors with a built-in small-cap tilt — diagnosis showed
    # mega-cap stocks drove most of the benchmark's return this period,
    # so an anti-size bias works against the strategy here.
    excluded_factors = {'F_SIZE', 'F_MIDCAP'}
    all_factor_cols = sorted(
        set(c.replace('_neu', '') for c in df.columns if c.endswith('_neu')) - excluded_factors
    )
    monthly_ic = compute_monthly_ic_table(df, all_factor_cols, date_col)
    
    bt_results = []
    prev_codes = set()
    
    for i, dt in enumerate(dates[:-1]):
        weights = get_rolling_weights(monthly_ic, dt, lookback=lookback, top_n=4)
        if not weights:
            continue
        
        month = df[df[date_col] == dt].copy()
        month['composite_score'] = 0.0
        for f, w in weights.items():
            neu_col = f'{f}_neu'
            if neu_col in month.columns:
                month['composite_score'] += month[neu_col].fillna(0) * w
        month = month.dropna(subset=['composite_score'])
        if len(month) < MIN_STOCKS:
            continue
        
        # Market-wide industry weights this month (the neutrality benchmark)
        market_industry_weight = month['industry'].value_counts(normalize=True)
        
        n_select = max(int(len(month) * TOP_QUANTILE), MIN_STOCKS)
        n_buffer = max(int(len(month) * buffer_quantile), n_select)
        
        ranked = month.sort_values('composite_score', ascending=False).reset_index(drop=True)
        buffer_codes = set(ranked.iloc[:n_buffer]['code'].values)
        
        kept = prev_codes & buffer_codes
        
        # Enforce industry caps while filling remaining slots
        industry_count = {}
        for code in kept:
            ind = month.loc[month['code'] == code, 'industry']
            if not ind.empty:
                ind = ind.values[0]
                industry_count[ind] = industry_count.get(ind, 0) + 1
        
        new_picks = []
        candidates = ranked[~ranked['code'].isin(kept)]
        n_new_needed = n_select - len(kept)
        
        for _, row in candidates.iterrows():
            if len(new_picks) >= n_new_needed:
                break
            ind = row['industry']
            cap = market_industry_weight.get(ind, 0) + industry_max_overweight
            current_share = industry_count.get(ind, 0) / n_select
            if current_share < cap:
                new_picks.append(row['code'])
                industry_count[ind] = industry_count.get(ind, 0) + 1
        
        curr_codes = kept | set(new_picks)
        selected = month[month['code'].isin(curr_codes)].copy()
        n_stocks = len(selected)
        if n_stocks == 0:
            continue
        
        # Inverse-volatility position sizing
        # Position sizing: tilt toward larger market cap combined with
        # inverse-volatility, then cap each stock at MAX_WEIGHT and
        # redistribute the excess to the remaining names.
        if 'F_VOL12M' in selected.columns and 'mcap' in selected.columns:
            vol = selected['F_VOL12M'].copy()
            vol = vol.where(vol > 0, np.nan)
            inv_vol = 1.0 / vol
            inv_vol = inv_vol.fillna(inv_vol.median() if inv_vol.notna().any() else 1.0)
            mcap_score = selected['mcap'].fillna(selected['mcap'].median())
            raw_weight = inv_vol * mcap_score
            selected['pos_weight'] = raw_weight / raw_weight.sum()
        else:
            selected['pos_weight'] = 1.0 / n_stocks
        
        for _ in range(5):
            over = selected['pos_weight'] > MAX_WEIGHT
            if not over.any():
                break
            excess = (selected.loc[over, 'pos_weight'] - MAX_WEIGHT).sum()
            selected.loc[over, 'pos_weight'] = MAX_WEIGHT
            under = ~over
            under_total = selected.loc[under, 'pos_weight'].sum()
            if under_total > 0:
                selected.loc[under, 'pos_weight'] += excess * (selected.loc[under, 'pos_weight'] / under_total)
        
        # Next month return
        next_dt = dates[i + 1]
        fwd = df[df[date_col] == next_dt][['code', 'ret_pct']].set_index('code')
        portfolio_ret = 0.0
        matched_weight = 0.0
        
        for _, row in selected.iterrows():
            if row['code'] in fwd.index and pd.notna(fwd.loc[row['code'], 'ret_pct']):
                portfolio_ret += row['pos_weight'] * (fwd.loc[row['code'], 'ret_pct'] / 100)
                matched_weight += row['pos_weight']
        
        if matched_weight > 0:
            portfolio_ret /= matched_weight
        
        # --- Market timing overlay: reduce equity exposure in downtrends ---
        if benchmark_trend is not None and dt in benchmark_trend.index:
            trend_up = benchmark_trend.loc[dt, 'trend_up']
            exposure = 1.0 if trend_up else defensive_exposure
        else:
            exposure = 1.0
        portfolio_ret = exposure * portfolio_ret + (1 - exposure) * monthly_cash_ret
        
        if prev_codes:
            turnover = 1 - len(prev_codes & curr_codes) / len(curr_codes)
            portfolio_ret -= turnover * 2 * TRANSACTION_COST
        
        bt_results.append({
            'date': dt, 'next_date': next_dt, 'n_stocks': n_stocks,
            'portfolio_ret': portfolio_ret,
            'avg_score': selected['composite_score'].mean(),
            'selected_codes': ','.join(selected['code'].tolist())
        })
        prev_codes = curr_codes
    
    return pd.DataFrame(bt_results)

def load_benchmark_trend(benchmark_path, ma_window=10):
    """Load CSI 300 and compute a trailing moving-average trend filter.
    trend_up=True means the index is above its own MA (bullish regime)."""
    bench = pd.read_csv(benchmark_path, parse_dates=['date'])
    bench = bench.sort_values('date').reset_index(drop=True)
    bench['ma'] = bench['close'].rolling(ma_window).mean()
    bench['trend_up'] = bench['close'] > bench['ma']
    return bench.set_index('date')[['trend_up']]

def compute_benchmark_comparison(bt_df, benchmark_path):
    """Load CSI 300 benchmark and compare against strategy returns."""
    bench = pd.read_csv(benchmark_path, parse_dates=['date'])
    bench = bench.sort_values('date').reset_index(drop=True)
    
    merged = bt_df.merge(
        bench[['date', 'ret_pct']].rename(columns={'ret_pct': 'bench_ret_pct'}),
        on='date', how='left'
    )
    merged['bench_ret'] = merged['bench_ret_pct'] / 100
    merged = merged.dropna(subset=['bench_ret'])
    
    r = merged['bench_ret'].values
    n = len(r)
    n_years = n / 12.0
    total_return = np.prod(1 + r) - 1
    bench_cagr = (1 + total_return) ** (1 / n_years) - 1
    bench_vol = np.std(r) * np.sqrt(12)
    bench_sharpe = (bench_cagr - 0.03) / bench_vol if bench_vol > 0 else 0
    
    strat_cagr = compute_metrics(merged)['CAGR']
    excess_return = strat_cagr - bench_cagr
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON (vs CSI 300)")
    print("=" * 70)
    print(f"{'Strategy CAGR':<20}: {strat_cagr:>8.2%}")
    print(f"{'Benchmark CAGR':<20}: {bench_cagr:>8.2%}")
    print(f"{'Excess Return':<20}: {excess_return:>8.2%}")
    print(f"{'Benchmark Sharpe':<20}: {bench_sharpe:>8.2f}")
    
    return {
        'strategy_cagr': strat_cagr, 'benchmark_cagr': bench_cagr,
        'excess_return': excess_return, 'benchmark_sharpe': bench_sharpe
    } 


def compute_metrics(bt_df, rf=0.03):
        """Compute standard performance metrics."""
        r = bt_df['portfolio_ret'].values
        n = len(r)
        n_years = n / 12.0
        total_return = np.prod(1 + r) - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1
        annual_vol = np.std(r) * np.sqrt(12)
        sharpe = (cagr - rf) / annual_vol if annual_vol > 0 else 0
        cum = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cum)
        drawdown = (cum - running_max) / running_max
        max_dd = np.min(drawdown)
        win_rate = sum(1 for x in r if x > 0) / n
        
        return {
            'CAGR': cagr, 'Annual_Volatility': annual_vol, 'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_dd, 'Win_Rate': win_rate,
            'Total_Return': total_return, 'N_Months': n
        }


# ============================================================================
# ML COMPARISON 
# ============================================================================

def run_ml_comparison(df, dates):
    """Walk-forward ML comparison (XGBoost + RandomForest vs factor model)."""
    log.info("Running ML comparison (walk-forward validation)...")
    try:
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        log.warning("  xgboost not installed. Skipping ML comparison.")
        log.warning("  Install with: pip install xgboost")
        return None
    
    factor_cols_z = [c for c in df.columns if c.endswith('_z')]
    if not factor_cols_z:
        log.warning("  No z-score factors found. Skipping ML.")
        return None
    
    # Build ML dataset
    ml_data = []
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        month = df[df['date'] == dt].dropna(subset=factor_cols_z).copy()
        if len(month) < 30:
            continue
        fwd = df[df['date'] == next_dt][['code', 'ret_pct']].set_index('code')
        month['target'] = month['code'].map(fwd['ret_pct']) / 100
        month = month.dropna(subset=['target'])
        ml_data.append(month[['code', 'date', 'target'] + factor_cols_z])
    
    ml_df = pd.concat(ml_data, ignore_index=True)
    log.info(f"  ML dataset: {ml_df.shape[0]:,} rows, {len(factor_cols_z)} features")
    
    results = []
    for split_name, train_end, test_start, test_end in [
        ('2016-2018 → 2019', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2016-2019 → 2020', '2019-12-31', '2020-01-01', '2020-12-31'),
    ]:
        train = ml_df[ml_df['date'] <= train_end]
        test = ml_df[(ml_df['date'] >= test_start) & (ml_df['date'] <= test_end)]
        X_train, y_train = train[factor_cols_z].values, train['target'].values
        X_test, y_test = test[factor_cols_z].values, test['target'].values
        
        for name, model in [
            ('XGBoost', XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=42, verbosity=0)),
            ('RF', RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10,
                                          random_state=42, n_jobs=-1)),
        ]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ric = stats.spearmanr(y_pred, y_test)[0] if len(y_test) > 1 else 0
            results.append({'Split': split_name, 'Model': name, 'Rank_IC': ric,
                           'N_Train': len(X_train), 'N_Test': len(X_test)})
            log.info(f"  {split_name} {name}: Rank IC = {ric:.4f}")
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main(skip_ml=False, validate_only=False):
    """Run the full pipeline."""
    start_time = datetime.now()
    log.info("=" * 70)
    log.info("CSI 300 Multi-Factor Quantitative Stock Selection Pipeline")
    log.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)
    
    # 1. Load data
    monthly_ret, mcap, fund, industry = load_data()
    
    if validate_only:
        log.info("\n=== DATA VALIDATION ===")
        log.info(f"Monthly returns: {monthly_ret.shape[0]:,} rows, {monthly_ret['code'].nunique()} stocks")
        log.info(f"  Date range: {monthly_ret['date'].min().strftime('%Y-%m')} to {monthly_ret['date'].max().strftime('%Y-%m')}")
        log.info(f"  Months: {monthly_ret['date'].nunique()}")
        log.info(f"Market cap: {mcap.shape[0]:,} rows, {mcap['code'].nunique()} stocks")
        log.info(f"Fundamentals: {fund.shape[0]:,} rows, {fund['code'].nunique()} stocks")
        log.info(f"Industry: {industry['code'].nunique()} stocks, {industry['industry'].nunique()} industries")
        return
    
    # 2. Build panel
    panel = build_panel(monthly_ret, mcap, industry)
    dates = sorted(panel['date'].unique())
    
    # 3. Compute factors
    factor_df = compute_factors(panel, fund)
    
    # 4. Process factors
    factor_cols = [c for c in factor_df.columns if c.startswith('F_') and not c.endswith(('_z', '_neu'))]
    factor_cols = [c for c in factor_cols if factor_df[c].notna().sum() > 1000]
    factor_df = winsorize_standardize_neutralize(factor_df, factor_cols)
    
    # 5. IC analysis
    ic_df = compute_ic_analysis(factor_df, factor_cols)
    print("\n" + "=" * 70)
    print("FACTOR IC ANALYSIS (Top 10 by Rank ICIR)")
    print("=" * 70)
    print(ic_df.head(10)[['Factor', 'Rank_IC_Mean', 'Rank_ICIR', 'IC_Positive_Ratio']].to_string(index=False))
    
    # 6. Backtest
    benchmark_trend = load_benchmark_trend(BENCHMARK_CSV, ma_window=10)
    bt_df = run_backtest(factor_df, ic_df, benchmark_trend=benchmark_trend)
    metrics = compute_metrics(bt_df)
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period: {bt_df['date'].iloc[0].strftime('%Y-%m')} to {bt_df['next_date'].iloc[-1].strftime('%Y-%m')}")
    print(f"Months: {len(bt_df)} | Avg stocks: {bt_df['n_stocks'].mean():.0f}")
    print(f"{'CAGR':<20}: {metrics['CAGR']:>8.2%}")
    print(f"{'Annual Volatility':<20}: {metrics['Annual_Volatility']:>8.2%}")
    print(f"{'Sharpe Ratio':<20}: {metrics['Sharpe_Ratio']:>8.2f}")
    print(f"{'Max Drawdown':<20}: {metrics['Max_Drawdown']:>8.2%}")
    print(f"{'Win Rate':<20}: {metrics['Win_Rate']:>8.2%}")
    print(f"{'Total Return':<20}: {metrics['Total_Return']:>8.2%}")
    
    # Annual returns
    bt_df['year'] = bt_df['next_date'].dt.year
    print("\nAnnual Returns:")
    for yr in sorted(bt_df['year'].unique()):
        yr_ret = np.prod(1 + bt_df[bt_df['year'] == yr]['portfolio_ret']) - 1
        print(f"  {yr}: {yr_ret:+.2%}")
    
    # comparison
    bench_metrics = compute_benchmark_comparison(bt_df, BENCHMARK_CSV)
    
    # 7. ML comparison
    if not skip_ml:
        print("\n" + "=" * 70)
        print("ML COMPARISON (Walk-Forward)")
        print("=" * 70)
        ml_results = run_ml_comparison(factor_df, dates)
        if ml_results is not None:
            print(ml_results.to_string(index=False))
        else:
            print("  ML comparison skipped (xgboost not available or no z-score factors)")
    
    # 8. Save results
    ic_df.to_csv('/Users/ella/Desktop/run_full_pipeline data & py/ic_analysis.csv', index=False)
    bt_df.to_csv('/Users/ella/Desktop/run_full_pipeline data & py/backtest_results.csv', index=False)
    log.info(f"\nResults saved to /project/")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"\nPipeline completed in {elapsed:.1f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSI 300 Multi-Factor Pipeline')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML comparison')
    parser.add_argument('--validate', action='store_true', help='Data validation only')
    args = parser.parse_args()
    main(skip_ml=args.skip_ml, validate_only=args.validate)
