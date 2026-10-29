import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from bitcoin_golden_cross import (
    simulate_btc_prices,
    calculate_moving_averages,
    run_golden_cross_strategy
)

class TestBitcoinGoldenCross(unittest.TestCase):

    def test_simulate_btc_prices_shape_and_seed(self):
        days = 60
        df1 = simulate_btc_prices(days=days, seed=42)
        df2 = simulate_btc_prices(days=days, seed=42)

        self.assertEqual(len(df1), days)
        self.assertIn('Date', df1.columns)
        self.assertIn('Price', df1.columns)
        self.assertTrue((df1['Price'] > 0).all())
        pd.testing.assert_frame_equal(df1, df2)

    def test_calculate_moving_averages(self):
        prices = list(range(1, 61))
        dates = [datetime(2025, 1, 1) + pd.Timedelta(days=i) for i in range(60)]
        df = pd.DataFrame({'Date': dates, 'Price': prices})

        df_ma = calculate_moving_averages(df, short_window=7, long_window=30)

        self.assertIn('MA7', df_ma.columns)
        self.assertIn('MA30', df_ma.columns)

        # Days 1 to 6 should be NaN for MA7
        self.assertTrue(df_ma['MA7'].iloc[:6].isna().all())
        # Day 7 MA7 should be average of 1..7 = 4.0
        self.assertAlmostEqual(df_ma['MA7'].iloc[6], 4.0)

        # Days 1 to 29 should be NaN for MA30
        self.assertTrue(df_ma['MA30'].iloc[:29].isna().all())
        # Day 30 MA30 should be average of 1..30 = 15.5
        self.assertAlmostEqual(df_ma['MA30'].iloc[29], 15.5)

    def test_golden_cross_signals(self):
        # Create synthetic price series to trigger Golden Cross and Death Cross
        # Days 1..30: prices high then dropping so MA7 < MA30
        # Days 31..45: prices jump up so MA7 crosses above MA30 (BUY)
        # Days 46..60: prices crash so MA7 crosses below MA30 (SELL)
        prices = [100.0] * 30 + [200.0] * 15 + [50.0] * 15
        dates = [datetime(2025, 1, 1) + pd.Timedelta(days=i) for i in range(60)]
        df = pd.DataFrame({'Date': dates, 'Price': prices})

        ledger_df, summary, trades = run_golden_cross_strategy(df, initial_cash=10000.0)

        self.assertGreater(len(trades), 0)
        types = [t['Type'] for t in trades]
        self.assertIn('BUY', types)
        self.assertIn('SELL', types)

        # Verify initial cash vs final value tracking logic
        self.assertEqual(summary['Initial Investment ($)'], 10000.0)
        self.assertIn('Final Portfolio Value ($)', summary)

if __name__ == '__main__':
    unittest.main()
