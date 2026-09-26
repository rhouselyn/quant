import unittest
import numpy as np
import pandas as pd
from quant_mvp.long_only import long_only_backtest, performance


class LongOnlyTests(unittest.TestCase):
    def test_returns_costs_and_turnover(self):
        p=pd.DataFrame([['1','A',2,np.log(1.1)],['1','B',1,0],
                        ['2','A',1,0],['2','B',2,np.log(1.2)]],
                       columns=['date','symbol','score','realized_return'])
        c,m=long_only_backtest(p,1,10)
        self.assertAlmostEqual(c.long_net_return.iloc[0],.999*1.1-1)
        self.assertAlmostEqual(c.long_net_return.iloc[1],.998*1.2-1)
        self.assertEqual(c.long_turnover.tolist(),[1.,2.])
        self.assertAlmostEqual(m['net']['final_equity'],.999*1.1*.998*1.2)

    def test_missing_return_does_not_replace_selected_stock(self):
        p=pd.DataFrame([['1','A',2,np.nan],['1','B',1,np.log(2)]],
                       columns=['date','symbol','score','realized_return'])
        c,m=long_only_backtest(p,1,0)
        self.assertEqual(c.long_net_return.iloc[0],0)
        self.assertEqual(m['missing_held_returns'],1)

    def test_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(performance([-.1])['max_drawdown'],-.1)

if __name__=='__main__':
    unittest.main()
