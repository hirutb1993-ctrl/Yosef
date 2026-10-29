import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def simulate_btc_prices(days=60, initial_price=60000.0, mu=0.10, sigma=0.60, seed=42):
    """
    Simulates daily Bitcoin prices using Geometric Brownian Motion (GBM).

    Parameters:
    - days: Number of days to simulate (default 60)
    - initial_price: Starting BTC price in USD (default 60,000)
    - mu: Annualized drift rate (default 0.10 or 10%)
    - sigma: Annualized volatility (default 0.60 or 60%)
    - seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / 365.0  # Daily time step

    # Generate daily returns using GBM formula
    # S(t+dt) = S(t) * exp((mu - 0.5 * sigma^2)*dt + sigma * sqrt(dt) * Z)
    daily_returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        days - 1
    )

    prices = [initial_price]
    for r in daily_returns:
        prices.append(prices[-1] * np.exp(r))

    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })

    return df

def calculate_moving_averages(df, short_window=7, long_window=30):
    """
    Calculates short-term and long-term Moving Averages using pandas.
    """
    df = df.copy()
    df['MA7'] = df['Price'].rolling(window=short_window).mean()
    df['MA30'] = df['Price'].rolling(window=long_window).mean()
    return df

def run_golden_cross_strategy(df, initial_cash=10000.0):
    """
    Simulates trading based on Golden Cross (MA7 > MA30) and Death Cross (MA7 < MA30).

    Returns:
    - ledger_df: DataFrame containing daily state, trade signals, and portfolio values.
    - performance_summary: Dictionary with key performance metrics.
    """
    df = calculate_moving_averages(df)

    cash = initial_cash
    btc_holding = 0.0

    ledger = []
    trades = []

    prev_ma7 = None
    prev_ma30 = None

    for idx, row in df.iterrows():
        day = idx + 1
        date = row['Date'].strftime('%Y-%m-%d')
        price = row['Price']
        ma7 = row['MA7']
        ma30 = row['MA30']

        signal = 'HOLD'
        trade_note = ''

        # We can only evaluate signals when both MAs are available
        if pd.notna(ma7) and pd.notna(ma30):
            if prev_ma7 is not None and prev_ma30 is not None:
                # Golden Cross: MA7 crosses above MA30
                if prev_ma7 <= prev_ma30 and ma7 > ma30:
                    if cash > 0:
                        btc_bought = cash / price
                        btc_holding += btc_bought
                        trade_note = f"BUY {btc_bought:.4f} BTC @ ${price:,.2f}"
                        cash = 0.0
                        signal = 'BUY (Golden Cross)'
                        trades.append({'Day': day, 'Date': date, 'Type': 'BUY', 'Price': price, 'BTC': btc_bought})
                # Death Cross: MA7 crosses below MA30
                elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                    if btc_holding > 0:
                        cash_received = btc_holding * price
                        trade_note = f"SELL {btc_holding:.4f} BTC @ ${price:,.2f}"
                        cash += cash_received
                        btc_holding = 0.0
                        signal = 'SELL (Death Cross)'
                        trades.append({'Day': day, 'Date': date, 'Type': 'SELL', 'Price': price, 'Cash': cash_received})

            prev_ma7 = ma7
            prev_ma30 = ma30

        total_value = cash + (btc_holding * price)

        ledger.append({
            'Day': day,
            'Date': date,
            'Price': price,
            'MA7': ma7 if pd.notna(ma7) else None,
            'MA30': ma30 if pd.notna(ma30) else None,
            'Signal': signal,
            'Trade': trade_note if trade_note else '-',
            'Cash ($)': cash,
            'BTC Holding': btc_holding,
            'Total Value ($)': total_value
        })

    ledger_df = pd.DataFrame(ledger)

    start_price = df.iloc[0]['Price']
    end_price = df.iloc[-1]['Price']
    final_value = ledger_df.iloc[-1]['Total Value ($)']

    strategy_return_pct = ((final_value - initial_cash) / initial_cash) * 100.0
    buy_hold_value = (initial_cash / start_price) * end_price
    buy_hold_return_pct = ((buy_hold_value - initial_cash) / initial_cash) * 100.0

    performance_summary = {
        'Initial Investment ($)': initial_cash,
        'Final Cash Balance ($)': cash,
        'Final BTC Holdings': btc_holding,
        'Final Portfolio Value ($)': final_value,
        'Strategy Return (%)': strategy_return_pct,
        'Buy & Hold Return (%)': buy_hold_return_pct,
        'Total Trades': len(trades),
        'Start BTC Price ($)': start_price,
        'End BTC Price ($)': end_price
    }

    return ledger_df, performance_summary, trades

def print_daily_ledger_and_summary(ledger_df, summary):
    """
    Prints a cleanly formatted daily ledger and portfolio performance summary.
    """
    print("=" * 115)
    print("                              BITCOIN 60-DAY GOLDEN CROSS TRADING LEDGER")
    print("=" * 115)
    header = f"{'Day':<5} {'Date':<11} {'BTC Price ($)':<14} {'MA7 ($)':<12} {'MA30 ($)':<12} {'Signal':<20} {'Cash ($)':<12} {'BTC':<10} {'Total Value ($)':<14}"
    print(header)
    print("-" * 115)

    for _, row in ledger_df.iterrows():
        ma7_str = f"${row['MA7']:,.2f}" if row['MA7'] is not None else "N/A"
        ma30_str = f"${row['MA30']:,.2f}" if row['MA30'] is not None else "N/A"

        print(f"{row['Day']:<5} {row['Date']:<11} ${row['Price']:<13,.2f} {ma7_str:<12} {ma30_str:<12} "
              f"{row['Signal']:<20} ${row['Cash ($)']:<11,.2f} {row['BTC Holding']:<10.4f} ${row['Total Value ($)']:<13,.2f}")

    print("=" * 115)
    print("                                   FINAL PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 115)
    print(f"Initial Cash Balance    : ${summary['Initial Investment ($)']:,.2f}")
    print(f"Starting BTC Price      : ${summary['Start BTC Price ($)']:,.2f}")
    print(f"Ending BTC Price        : ${summary['End BTC Price ($)']:,.2f}")
    print(f"Final Cash Balance      : ${summary['Final Cash Balance ($)']:,.2f}")
    print(f"Final BTC Holding       : {summary['Final BTC Holdings']:.4f} BTC")
    print(f"Final Portfolio Value   : ${summary['Final Portfolio Value ($)']:,.2f}")
    print(f"Strategy Total Return   : {summary['Strategy Return (%)']:+.2f}%")
    print(f"Buy & Hold Return       : {summary['Buy & Hold Return (%)']:+.2f}%")
    print(f"Total Trades Executed   : {summary['Total Trades']}")
    print("=" * 115)

def main():
    # 1. Simulate 60 days of BTC price data
    df = simulate_btc_prices(days=60, initial_price=60000.0, mu=0.15, sigma=0.65, seed=1)

    # 2. Run Golden Cross Strategy
    ledger_df, summary, trades = run_golden_cross_strategy(df, initial_cash=10000.0)

    # 3. Output results
    print_daily_ledger_and_summary(ledger_df, summary)

if __name__ == "__main__":
    main()
