import unittest
import numpy as np
import pandas as pd
from quant_mvp.reliability import direction_metrics, walk_forward_calibration, rank_reliability


class ReliabilityTests(unittest.TestCase):
    def test_always_up_has_half_balanced_accuracy(self):
        frame = pd.DataFrame({'score': np.ones(12), 'up_probability': np.ones(12)*.9,
                              'realized_return': [.1]*8+[-.1]*2+[0, np.nan]})
        result = direction_metrics(frame)
        self.assertEqual(result['n'], 10)
        self.assertEqual(result['accuracy'], .8)
        self.assertEqual(result['balanced_accuracy'], .5)

    def test_future_labels_do_not_change_calibrated_predictions(self):
        frame = pd.DataFrame([dict(date=f'{i:02}', symbol=s, score=j+i*.01,
                                   realized_return=.1 if (i+j)%3 else -.1)
                              for i in range(12) for j,s in enumerate('ABCD')])
        original = walk_forward_calibration(frame, warmup=4)
        altered = frame.copy()
        altered.loc[altered.date >= '09', 'realized_return'] *= -1
        changed = walk_forward_calibration(altered, warmup=4)
        np.testing.assert_allclose(original[original.date <= '09'].up_probability,
                                   changed[changed.date <= '09'].up_probability)

    def test_stable_stock_beats_one_half_lucky_stock(self):
        rows = []
        for i in range(40):
            up = i%2 == 0
            for symbol in ['stable', 'lucky']:
                correct = i%5 != 0 if symbol == 'stable' else i < 20
                rows.append(dict(date=f'{i:02}',symbol=symbol,score=float(up),
                                 realized_return=.1 if up else -.1,
                                 up_probability=.9 if up == correct else .1))
        ranked = rank_reliability(pd.DataFrame(rows), count=1)
        self.assertEqual(ranked[ranked.selected].symbol.tolist(), ['stable'])


if __name__ == '__main__':
    unittest.main()
