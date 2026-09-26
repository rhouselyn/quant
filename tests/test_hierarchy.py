import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from quant_mvp.hierarchy import (HierarchicalRanker, industry_exposure_baseline, industry_exposure_loss,
                                masked_infonce, weekly_stickiness_loss)
from quant_mvp.hierarchy_data import align_observed_features, industry_excess, market_features, arrays, MARKET_FEATURES
from quant_mvp.data import FEATURES


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.cfg=dict(n_features=24,d_model=16,embedding_dim=16,n_layers=1,dropout=0.,
            attention_heads=4,global_layers=2,encoder_chunk_size=6,market_features=6)
        self.ids=torch.tensor([0,0,0,1,1,1])
        self.model=HierarchicalRanker(self.cfg,self.ids).eval()

    def test_stock_permutation_equivariance(self):
        h=torch.randn(2,6,16);market=torch.randn(2,6);valid=torch.ones(2,6,dtype=torch.bool)
        _,expected,_=self.model.relate(h,market,valid)
        order=torch.tensor([3,1,5,0,4,2])
        _,actual,_=self.model.relate(h[:,order],market,valid[:,order],self.ids[order])
        torch.testing.assert_close(actual,expected[:,order],atol=1e-6,rtol=1e-5)

    def test_final_local_step_transmits_market_and_other_industry_gradients(self):
        h=torch.randn(1,6,16,requires_grad=True);market=torch.randn(1,6,requires_grad=True)
        _,scores,_=self.model.relate(h,market,torch.ones(1,6,dtype=torch.bool))
        scores[0,0].backward()
        self.assertGreater(h.grad[0,3:].abs().sum().item(),1e-10)
        self.assertGreater(market.grad.abs().sum().item(),1e-10)
        self.assertGreater(self.model.global_layers[-1].attn.in_proj_weight.grad.abs().sum().item(),1e-10)

    def test_missing_stock_and_empty_industry_do_not_contaminate_scores(self):
        h=torch.randn(1,6,16);market=torch.randn(1,6)
        valid=torch.tensor([[True,True,False,False,False,False]])
        _,expected,_=self.model.relate(h,market,valid)
        h[:,2:]=1e8
        _,actual,_=self.model.relate(h,market,valid)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[:,:2],expected[:,:2])

    def test_masked_contrastive_sample_has_no_gradient(self):
        q=torch.randn(1,6,16,requires_grad=True);k=torch.randn(1,6,16)
        valid=torch.tensor([[True,True,True,False,False,False]])
        loss=masked_infonce(q,k,valid)
        expected=masked_infonce(q[:,:3],k[:,:3],valid[:,:3])
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertEqual(q.grad[:,3:].abs().sum().item(),0.)

    def test_other_dates_cannot_change_current_cross_section(self):
        h=torch.randn(2,6,16);market=torch.randn(2,6);valid=torch.ones(2,6,dtype=torch.bool)
        _,expected,_=self.model.relate(h,market,valid)
        h[1]*=100;market[1]*=-100
        _,actual,_=self.model.relate(h,market,valid)
        torch.testing.assert_close(actual[0],expected[0])

    def test_calendar_gap_does_not_destroy_rolling_history_or_use_future(self):
        dates=pd.date_range('2020-01-03',periods=130,freq='W-FRI')
        close=np.exp(np.arange(130)*.002+np.sin(np.arange(130))*.01)
        p=pd.DataFrame(dict(date=dates,symbol='A',open=close,high=close*1.02,low=close*.98,
                            close=close,volume=np.arange(130)+100.,amount=close*100))
        p=p[p.date!=dates[90]].copy()
        a=align_observed_features(p,dates)
        self.assertTrue(a.loc[dates[90]:,'vol60'].notna().all())
        pd.testing.assert_series_equal(a.loc[dates[89]],a.loc[dates[90]],check_names=False)
        before=market_features(p)
        p.loc[p.date>dates[100],['open','high','low','close','amount']]*=10
        b=align_observed_features(p,dates)
        pd.testing.assert_frame_equal(a.loc[:dates[100]],b.loc[:dates[100]])
        pd.testing.assert_frame_equal(before.loc[:dates[100]],market_features(p).loc[:dates[100]])

    def test_industry_head_exists_only_when_weighted(self):
        self.assertIsNone(self.model.industry_head)
        h=torch.randn(2,6,16);market=torch.randn(2,6);valid=torch.ones(2,6,dtype=torch.bool)
        _,_,absent=self.model.relate(h,market,valid)
        self.assertIsNone(absent)
        model=HierarchicalRanker(dict(self.cfg,industry_head_weight=.5),self.ids).eval()
        self.assertEqual(model.industry_head.in_features,self.cfg['embedding_dim'])
        self.assertEqual(model.industry_head.out_features,1)
        _,_,industry=model.relate(h,market,valid)
        self.assertEqual(tuple(industry.shape),(2,int(self.ids.max())+1))
        # Industry tokens are ordered by industry id, so relabelling stocks cannot move them.
        h=torch.randn(1,6,16);market=torch.randn(1,6);valid=torch.ones(1,6,dtype=torch.bool)
        _,_,base=model.relate(h,market,valid)
        order=torch.tensor([3,4,5,0,1,2])
        _,_,shuffled=model.relate(h[:,order],market,valid[:,order],self.ids[order])
        torch.testing.assert_close(shuffled,base,atol=1e-5,rtol=1e-5)

    def test_industry_loss_reaches_every_relation_gate(self):
        model=HierarchicalRanker(dict(self.cfg,industry_head_weight=.5),self.ids).eval()
        model.train()
        h=torch.randn(2,6,16);market=torch.randn(2,6);valid=torch.ones(2,6,dtype=torch.bool)
        _,_,industry=model.relate(h,market,valid)
        industry.pow(2).mean().backward()
        for layer in list(model.local_layers)+list(model.global_layers):
            for name in ('attn_gate','ffn_gate'):
                self.assertGreater(getattr(layer,name).grad.abs().sum().item(),0.,name)

    def test_industry_excess_labels_center_on_the_universe_and_mask_thin_groups(self):
        y=np.array([[.06,-.02,np.nan,.12,.04,-.04],[.01,.01,.01,.01,.01,.01]],dtype='float32')
        valid_y=np.isfinite(y)
        available=np.ones_like(y,dtype=bool); available[1,0]=False
        ids=np.array([0,0,0,1,1,1])
        excess,mask=industry_excess(y,valid_y,available,ids,min_stocks=3)
        self.assertTrue(np.isfinite(excess[0]).all())
        np.testing.assert_allclose(excess[0],[-.012,.008],atol=1e-6)
        # A dropped stock removes its industry from the week instead of biasing it.
        np.testing.assert_allclose(excess[1],[0.,0.],atol=1e-6)
        self.assertTrue((mask==np.array([[False,True],[False,True]])).all())

    def test_industry_exposure_loss_spans_zero_to_one(self):
        ids=torch.tensor([[0,0,0,1,1,1]]).repeat(3,1)
        neutral=torch.tensor([[-1.,0.,1.,-1.,0.,1.],[-2.,0.,2.,-1.,1.,0.],[1.,1.,-2.,0.,-1.,1.]])
        step=torch.tensor([[0.,0.,0.,2.,2.,2.],[-1.,-1.,-1.,1.,1.,1.],[3.,3.,3.,-3.,-3.,-3.]])
        valid=torch.ones_like(neutral,dtype=torch.bool)
        self.assertTrue(torch.allclose(industry_exposure_loss(neutral,ids,valid),torch.zeros(()),atol=1e-6))
        self.assertTrue(torch.allclose(industry_exposure_loss(step,ids,valid),torch.ones(()),atol=1e-4))

    def test_industry_exposure_loss_matches_the_logged_numpy_twin(self):
        from quant_mvp.hierarchy_engine import score_eta2_by_industry
        ids=np.repeat(np.arange(29),20)
        available=np.ones((4,len(ids)),dtype=bool);available[:, ::37]=False
        rng=np.random.default_rng(3)
        dense=rng.normal(size=(4,len(ids)))+ids[None,:]*0.3
        dense[~available]=np.nan
        mask=torch.from_numpy(available)
        scores=torch.tensor(np.nan_to_num(dense),dtype=torch.float32)
        got=industry_exposure_loss(scores,torch.tensor(ids)[None].expand(4,-1),mask)
        eta2,neutral=score_eta2_by_industry(dense,available,ids)
        self.assertAlmostEqual(float(got),eta2,places=5)
        self.assertLess(neutral,eta2)

    def test_industry_exposure_loss_ignores_names_that_are_not_tradable(self):
        ids=torch.tensor([[0,0,0,1,1,1]])
        base=torch.tensor([[1.,0.,-1.,0.,1.,-1.]])
        valid=torch.tensor([[True,True,False,True,True,False]])
        padded=torch.tensor([[1.,0.,99.,0.,1.,-99.]])
        self.assertTrue(torch.allclose(industry_exposure_loss(base,ids,valid),
                                       industry_exposure_loss(padded,ids,valid),atol=1e-7))

    def test_industry_exposure_gradient_flattens_industry_means(self):
        ids=torch.tensor([[0,0,0,1,1,1]])
        scores=torch.tensor([[2.,1.6,1.2,-2.,-1.6,-1.2]],requires_grad=True)
        valid=torch.ones_like(scores,dtype=torch.bool)
        before=industry_exposure_loss(scores,ids,valid)
        grad,=torch.autograd.grad(before,scores)
        moved=scores.detach()-0.5*grad
        self.assertLess(float(industry_exposure_loss(moved,ids,valid)),float(before))
        self.assertGreater(float(before),0.9)

    def test_industry_exposure_baseline_is_the_permutation_null(self):
        from quant_mvp.hierarchy_engine import score_eta2_by_industry
        for counts in ([20]*29,[3,3,4],[1,1,580]):
            ids=np.repeat(np.arange(len(counts)),counts)
            dense=np.random.default_rng(7).normal(size=(20000,len(ids)))
            eta2,neutral=score_eta2_by_industry(dense,np.ones_like(dense,dtype=bool),ids)
            self.assertAlmostEqual(eta2,neutral,delta=.01)
            self.assertAlmostEqual(neutral,industry_exposure_baseline(counts),places=6)
        self.assertIsNone(industry_exposure_baseline([6]))
        self.assertAlmostEqual(industry_exposure_baseline([1,1,1]),1.)

    def test_evaluation_returns_are_not_clipped_to_training_target_bounds(self):
        from quant_mvp.hierarchy_engine import predict
        dates=pd.date_range('2020-01-03',periods=4,freq='W-FRI')
        symbols=[f'S{i:03}' for i in range(580)]
        panel=pd.MultiIndex.from_product([dates,symbols],names=['date','symbol']).to_frame(index=False)
        for feature in FEATURES:panel[feature]=np.linspace(1.,2.,len(panel))
        panel['observed']=True;panel['target']=.4
        panel.loc[(panel.date==dates[2]) & (panel.symbol==symbols[1]),'target']=-.6
        panel.loc[(panel.date==dates[2]) & (panel.symbol==symbols[2]),'target']=np.nan
        market=pd.DataFrame({'date':dates,**{f:np.arange(4,dtype=float) for f in MARKET_FEATURES}})
        universe=pd.DataFrame({'symbol':symbols,'industry_id':np.repeat(np.arange(29),20)})
        cfg=dict(cache_path='unused',market_path='market',universe_path='universe',lookback=2,
                 train_end=2,min_history_weeks=2)
        with patch('quant_mvp.hierarchy_data.pd.read_parquet',return_value=panel), \
             patch('quant_mvp.hierarchy_data.pd.read_csv',side_effect=[market,universe]):
            data=arrays(cfg)
        class ConstantModel:
            def eval(self):return self
            def __call__(self,windows,market,available):
                return None,None,torch.zeros(windows.shape[:2]),None
        prediction=predict(ConstantModel(),data,cfg,2,3,torch.device('cpu')).set_index('symbol')
        self.assertAlmostEqual(prediction.loc[symbols[0],'realized_return'],.4,places=6)
        self.assertAlmostEqual(prediction.loc[symbols[1],'realized_return'],-.6,places=6)
        self.assertTrue(np.isnan(prediction.loc[symbols[2],'realized_return']))


class StickinessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_identical_or_uniformly_shifted_weeks_cost_nothing(self):
        scores=torch.randn(2,6)
        all_valid=torch.ones_like(scores,dtype=torch.bool)
        self.assertAlmostEqual(float(weekly_stickiness_loss(scores,scores,all_valid,all_valid)),0.,places=6)
        shifted=torch.cat([scores[:1]+5.,scores[1:]-3.])
        self.assertAlmostEqual(float(weekly_stickiness_loss(shifted,scores,all_valid,all_valid)),0.,places=5)

    def test_support_changes_are_priced_and_uncovered_names_are_ignored(self):
        scores=torch.tensor([[1.,0.,-1.,2.,-2.,0.]])
        closed=torch.tensor([[True,True,False,True,True,True]])
        open_names=torch.tensor([[True,True,True,True,True,True]])
        padded=torch.tensor([[1.,0.,99.,2.,-2.,0.]])
        self.assertAlmostEqual(float(weekly_stickiness_loss(padded,scores,closed,closed)),0.,places=6)
        self.assertGreater(float(weekly_stickiness_loss(scores,scores,open_names,closed)),0.)

    def test_gradient_step_toward_last_week_lowers_the_penalty(self):
        past=torch.tensor([[2.,1.,0.,-1.,-2.,.5]])
        valid=torch.ones_like(past,dtype=torch.bool)
        moved=(past+torch.tensor([.3,-.2,.1,.4,-.5,.2])).requires_grad_(True)
        loss=weekly_stickiness_loss(moved,past,valid,valid)
        loss.backward()
        self.assertGreater(float(loss),0.)
        step=moved-.05*moved.grad/moved.grad.abs().max()
        self.assertLess(float(weekly_stickiness_loss(step,past,valid,valid)),float(loss))


    def test_capped_projection_is_the_top_n_book(self):
        from quant_mvp.hierarchy import top_k_weight_vector
        score=torch.arange(10., -1., -1.).reshape(1, -1)
        weights=top_k_weight_vector(score, 1. / 4)[0]
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertLessEqual(float(weights.max()), 1. / 4 + 1e-5)
        self.assertTrue(bool((weights[:4] > .249).all()))
        self.assertLess(float(weights[4:].abs().max()), 1e-5)

    def test_capped_measure_only_charges_changes_that_would_actually_trade(self):
        base=torch.tensor([[3.,2.,1.,0.,-1.,-2.]])
        valid=torch.ones(1, 6, dtype=torch.bool)
        # Rank 1 getting stronger moves no weight, so the book measure must not bill it.
        reinforced=(base+torch.tensor([[1.,0.,0.,0.,0.,0.]])).requires_grad_(True)
        self.assertAlmostEqual(float(weekly_stickiness_loss(reinforced, base, valid, valid, capacity=.25)), 0., places=6)
        self.assertGreater(float(weekly_stickiness_loss(reinforced, base, valid, valid, temperature_mult=.5)), 0.)
        # Name 4 dropping below name 5 replaces a holding: one slot of mass moves.
        swapped=torch.tensor([[3.,2.,1.,-1.,0.,-2.]])
        flow=weekly_stickiness_loss(swapped, base, valid, valid, capacity=.25)
        self.assertAlmostEqual(float(flow), .5, places=4)


    def test_dead_zone_bills_only_the_flow_beyond_the_band(self):
        base = torch.tensor([[3., 2., 1., 0., -1., -2.]])
        swapped = torch.tensor([[3., 2., 1., -1., 0., -2.]])
        valid = torch.ones(1, 6, dtype=torch.bool)
        flow = weekly_stickiness_loss(swapped, base, valid, valid, capacity=.25)
        self.assertAlmostEqual(float(flow), .5, places=4)
        # A quarter of the book turning over is inside a zone that tolerates 30%.
        self.assertAlmostEqual(float(weekly_stickiness_loss(swapped, base, valid, valid, capacity=.25, dead_zone=.3)), 0., places=6)
        self.assertAlmostEqual(float(weekly_stickiness_loss(swapped, base, valid, valid, capacity=.25, dead_zone=.2)), .1, places=4)
        free = weekly_stickiness_loss(reinforced := torch.tensor([[4., 2., 1., 0., -1., -2.]]), base, valid, valid, capacity=.25, dead_zone=.25)
        self.assertAlmostEqual(float(free), 0., places=6)


if __name__=='__main__':unittest.main()
