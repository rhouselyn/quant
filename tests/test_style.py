import unittest
import numpy as np
import pandas as pd
from quant_mvp.style import style_panel, design_matrix, neutralize


class StyleTests(unittest.TestCase):
    def test_removes_known_component_and_handles_collinearity(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        design = np.column_stack([np.ones(200), x, x])
        score = 3*x + rng.normal(size=200)
        residual, r2 = neutralize(score, design)
        np.testing.assert_allclose(design.T @ (residual-residual.mean()), 0., atol=1e-10)
        self.assertGreater(r2, .8)
        np.testing.assert_allclose(neutralize(score, design, 0.)[0], score)

    def test_future_prices_cannot_change_past_styles(self):
        rng = np.random.default_rng(2)
        dates = pd.date_range('2020-01-03', periods=90, freq='W-FRI')
        panel = pd.DataFrame([dict(date=d, symbol=s, close=p, amount=1e8)
            for s in ['A', 'B', 'C']
            for d, p in zip(dates, np.exp(np.cumsum(rng.normal(0, .02, 90))))])
        original = style_panel(panel)
        panel.loc[panel.date > dates[65], ['close', 'amount']] *= 10
        changed = style_panel(panel)
        pd.testing.assert_frame_equal(original[original.date <= dates[65]],
                                      changed[changed.date <= dates[65]])

    def test_missing_styles_preserve_candidates(self):
        frame = pd.DataFrame({'factor': [np.nan, 2., 3.], 'industry': ['A', 'B', 'B']})
        design, _ = design_matrix(frame, columns=['factor'], industry=True)
        self.assertEqual(len(design), 3)
        self.assertTrue(np.isfinite(design).all())


if __name__ == '__main__':
    unittest.main()
