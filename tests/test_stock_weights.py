import unittest
import torch
from quant_mvp.model import extreme_score_loss, rank_ic_loss, _soft_rank


class StockWeightTests(unittest.TestCase):
    def test_score_weight_changes_relative_gradients_and_ignores_middle(self):
        score = torch.zeros(3, requires_grad=True)
        loss = extreme_score_loss(score, torch.tensor([1., -1., 0.]),
                                  sample_weight=torch.tensor([3., 1., 50.]))
        loss.backward()
        torch.testing.assert_close(score.grad, torch.tensor([-.375, .125, 0.]))

    def test_unit_weights_preserve_loss_and_gradients(self):
        score = torch.tensor([[.1, -.2, .4, .3]], requires_grad=True)
        ret = torch.tensor([[.03, .02, -.01, .04]])
        a = rank_ic_loss(score, ret)
        ga, = torch.autograd.grad(a, score)
        b = rank_ic_loss(score, ret, sample_weight=torch.ones(4))
        gb, = torch.autograd.grad(b, score)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(ga, gb)

    def test_weighted_rank_correlation_and_scale_invariance(self):
        s = torch.tensor([[.1, -.2, .4, .3, 99.]], requires_grad=True)
        r = torch.tensor([[.03, .02, -.01, .04, float('nan')]])
        w = torch.tensor([3., 1., 3., 1., 1000.])
        a = rank_ic_loss(s, r, sample_weight=w)
        torch.testing.assert_close(a, rank_ic_loss(s, r, sample_weight=w*7))
        sr, rr = _soft_rank(s[:, :4]), _soft_rank(r[:, :4])
        weights = w[:4] / w[:4].sum()
        x = sr - (sr*weights).sum()
        y = rr - (rr*weights).sum()
        expected = 1-(x*y*weights).sum()/((x*x*weights).sum()*(y*y*weights).sum()).sqrt()
        torch.testing.assert_close(a, expected)
        a.backward()
        self.assertEqual(s.grad[0, -1].item(), 0.)
        self.assertTrue(torch.isfinite(s.grad).all())


if __name__ == '__main__':
    unittest.main()
