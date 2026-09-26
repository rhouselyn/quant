import unittest
import torch
from quant_mvp.model import output_decorrelation_loss


class OutputDecorrTests(unittest.TestCase):
    def test_identical_four_heads_have_twelve_offdiagonal_units(self):
        x=torch.arange(6,dtype=torch.float32)[:,None].repeat(1,4)
        torch.testing.assert_close(output_decorrelation_loss(x),torch.tensor(12.))

    def test_uncorrelated_heads_and_weekly_shift_invariance(self):
        x=torch.tensor([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
        torch.testing.assert_close(output_decorrelation_loss(x),torch.tensor(0.))
        batch=torch.stack([x,x+torch.tensor([100.,100.])])
        torch.testing.assert_close(output_decorrelation_loss(batch),torch.tensor(0.))

    def test_masked_outlier_does_not_change_loss_or_gradients(self):
        torch.manual_seed(3)
        x=torch.randn(2,10,4,requires_grad=True)
        reference=output_decorrelation_loss(x)
        extra=torch.cat([x,torch.full((2,1,4),1e5)],dim=1)
        mask=torch.ones(2,11,dtype=torch.bool);mask[:,-1]=False
        result=output_decorrelation_loss(extra,mask)
        torch.testing.assert_close(result,reference)
        grad,=torch.autograd.grad(result,x)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)

    def test_constant_or_single_heads_remain_finite(self):
        for shape in [(4,1),(2,4,4)]:
            x=torch.ones(shape,requires_grad=True)
            loss=output_decorrelation_loss(x)
            loss.backward()
            self.assertEqual(loss.item(),0)
            self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':
    unittest.main()
