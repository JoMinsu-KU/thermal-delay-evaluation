"""Synthetic scheduler and integrity tests; no research inference or plant."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scheduler import simulate
from latency_replay import ReplaySampler
from validation_io import write_json, seal, verify_seal, completed_case, read_json
from plan import design
from mc_statistics import summarize, wilson, rank_summary
from latency_replay import uniform_index
from plan import METHODS


class Plant:
    def __init__(self, seconds=.6):
        self.entry={};self.duration_s=seconds;self.end_ns=round(seconds*1e9);self.now=0
        self.u=np.full(4,.2);self.fallback=False;self.applications=[]
    def advance_to(self,t):
        assert self.now<=t<=self.end_ns;self.now=t
    def apply(self,u,f):
        self.u=u.copy();self.fallback=f;self.applications.append((self.now,u.copy(),f))
    def raw(self):return self
    def snapshot_ns(self,t):return self.u.copy()
    def metrics(self):return {'completed':self.now==self.end_ns}


class Engine:
    def reset(self):self.count=0
    def decide(self,obs):
        previous=obs();self.count+=1
        return {'command':(previous+np.array([.04,0,0,0])).tolist(),'fallback_reason':None,
                'tokens':3,'filter_intervention':False}


class Delays:
    def __init__(self, delays):self.delays=delays
    def reset(self):pass
    def choose(self,d,i):return self.delays[i%len(self.delays)],i,0


def trace_rows():
    trials=[]
    for repeat in range(3):
        trials.append([{'request':f'{repeat}-{i}','path':'normal','fallback_reason':None,'tokens':3,
                        'e2e_ms':repeat*100000+i,'error':None} for i in range(6000)])
    return [r for group in zip(*trials) for r in group]


class ValidationTests(unittest.TestCase):
    def test_zero_policy_matches_unmodified_frozen_scheduler(self):
        import legacy_scheduler_reference as old
        rng=np.random.default_rng(7)
        for _ in range(40):
            delays=rng.choice([0.,10.,99.999999,100.,100.000001,150.,250.,1000.],size=8).tolist()
            def choose(trace,decision,i,strict):return delays[i%len(delays)],i,0
            old.choose_latency=choose
            a=old.simulate(Plant(1.2),Engine(),'delayed',[{}])
            b=simulate(Plant(1.2),Engine(),'delayed',Delays(delays))
            self.assertEqual(a,b)
    def test_missing_baselines_keep_gate_closed(self):
        from run_suite import evaluate_gate
        from validation_io import read_json
        with tempfile.TemporaryDirectory() as td:
            r=read_json(Path(__file__).with_name('reference.json'))
            write_json(Path(td)/'protocol.json',{'design':design(r)})
            gate=evaluate_gate(Path(td),'baseline')
            self.assertFalse(gate['passed']);self.assertEqual(len(gate['checks']),240)
    def test_exact_deadline_is_not_expired(self):
        m,t=simulate(Plant(),Engine(),'delayed',Delays([100]))
        self.assertEqual(m['missed_deadlines'],0);self.assertEqual(m['busy_skipped_ticks'],0)
        self.assertEqual(t[1]['time_ns'],100000000)
    def test_zero_hold_only_change_expired_command(self):
        results=[]
        for policy in ('zero','hold'):
            m,t=simulate(Plant(),Engine(),'delayed',Delays([150,10]),fallback_policy=policy);results.append((m,t))
        self.assertEqual(results[0][0]['missed_deadlines'],results[1][0]['missed_deadlines'])
        for m,t in results:self.assertGreater(m['busy_skipped_ticks'],0)
        a=next(x for x in results[0][1] if x['event']=='deadline_fallback')
        b=next(x for x in results[1][1] if x['event']=='deadline_fallback')
        self.assertEqual(a['command'],[0.]*4);self.assertEqual(b['command'],[.2]*4)
        self.assertEqual(a['time_ns'],b['time_ns'])
    def test_expired_result_never_applied_later(self):
        m,t=simulate(Plant(),Engine(),'delayed',Delays([250]))
        self.assertFalse(any(r['event']=='apply' for r in t));self.assertEqual(m['busy_skipped_ticks'],4)
    def test_hold_preserves_non_deadline_fallback(self):
        class Fallback(Engine):
            def decide(self,obs):return {'command':[0.]*4,'fallback_reason':'no_predicted_feasible_candidate','tokens':0}
        p=Plant(.1);m,t=simulate(p,Fallback(),'delayed',Delays([10]),fallback_policy='hold')
        self.assertEqual(p.u.tolist(),[0.]*4);self.assertTrue(p.fallback)
    def test_hold_does_not_add_noop_events(self):
        p=Plant(.6);m,t=simulate(p,Engine(),'delayed',Delays([150]),fallback_policy='hold')
        self.assertEqual(len(p.applications),1)
        self.assertTrue(all(x['no_change'] for x in t if x['event']=='deadline_fallback' and x['request']>0))
    def test_legacy_chooser_is_unchanged(self):
        calls=[]
        def choose(t,d,i,strict):calls.append((i,strict));return 12.,'source',0
        s=ReplaySampler([{}],'legacy',choose)
        self.assertEqual(s.choose({},5),(12.,'source',0));self.assertEqual(calls,[(5,True)])
    def test_seals_reject_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write_json(p/'metrics.json',{'cost':1});seal(p,{'case':'test','protocol_sha256':'abc'})
            verify_seal(p);write_json(p/'metrics.json',{'cost':2})
            with self.assertRaises(ValueError):verify_seal(p)
    def test_resume_checks_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cases/c/attempt_1';p.mkdir(parents=True);write_json(p/'metrics.json',{})
            seal(p,{'case':'c','protocol_sha256':'a'})
            self.assertEqual(completed_case(td,'c','a'),p)
            with self.assertRaises(ValueError):completed_case(td,'c','b')
    def test_fixed_case_counts(self):
        r=read_json(Path(__file__).with_name('reference.json'));d=design(r)
        self.assertEqual(sum(len(j['cases']) for j in d['jobs']),2544)
        keys=[c['key'] for j in d['jobs'] for c in j['cases']];self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual({j['phase']:sum(len(x['cases']) for x in d['jobs'] if x['phase']==j['phase']) for j in d['jobs']}, {'baseline':240,'monte_carlo':2304})


def empirical_fixture():
    trials=[]
    for rep in range(3):
        rows=[]
        for path,reason,n in [('normal',None,10000),('missing_observation','cold_start_or_missing_observation',1000),('no_predicted_feasible_candidate','no_predicted_feasible_candidate',1000)]:
            for i in range(n):
                rows.append(dict(request=len(rows),method='fv_mpc',repeat=rep,path=path,fallback_reason=reason,tokens=0,e2e_ms=float(rep*12000+len(rows)),error=None))
        trials.append(rows)
    return [r for group in zip(*trials) for r in group]


class MonteCarloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=empirical_fixture()
    def sampler(self,episode='ep0',replicate=0,rows=None):
        return ReplaySampler(self.rows if rows is None else rows,f'mc{replicate:04d}',None,seed=20261013,method='fv_mpc',episode=episode,replicate=replicate)
    def test_complete_analysis_rejects_missing_final_integrity(self):
        from analyze_results import final_input_integrity,final_state
        with tempfile.TemporaryDirectory() as td:
            gate=final_input_integrity(td)
            self.assertFalse(gate['available']);self.assertFalse(gate['passed'])
            self.assertEqual(final_state(528,528,[],2,2,gate),'INCOMPLETE_OR_FAILED')
    def test_complete_analysis_rejects_failed_or_invalid_final_integrity(self):
        from analyze_results import final_input_integrity,final_state
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'INPUT_INTEGRITY.json'
            receipts=[{'passed':False,'checked_files':1245,'failures':[{'file':'weights.pt','error':'SHA256 mismatch'}]},
                      {'passed':True,'checked_files':1245,'failures':[{'file':'weights.pt','error':'SHA256 mismatch'}]},
                      {'passed':True,'checked_files':0,'failures':[]}]
            for receipt in receipts:
                write_json(p,receipt);gate=final_input_integrity(td)
                self.assertFalse(gate['passed'])
                self.assertEqual(final_state(528,528,[],2,2,gate),'INCOMPLETE_OR_FAILED')
            p.write_text('{truncated',encoding='utf-8')
            self.assertEqual(final_state(528,528,[],2,2,final_input_integrity(td)),'INCOMPLETE_OR_FAILED')
    def test_complete_analysis_requires_integrity_and_all_other_gates(self):
        from analyze_results import final_input_integrity,final_state
        with tempfile.TemporaryDirectory() as td:
            write_json(Path(td)/'INPUT_INTEGRITY.json',{'passed':True,'checked_files':1245,'failures':[]})
            gate=final_input_integrity(td)
            self.assertTrue(gate['passed'])
            self.assertEqual(final_state(528,528,[],2,2,gate),'COMPLETE_VERIFIED')
            self.assertEqual(final_state(527,528,[],2,2,gate),'INCOMPLETE_OR_FAILED')
            self.assertEqual(final_state(528,528,[],1,2,gate),'INCOMPLETE_OR_FAILED')
            self.assertEqual(final_state(528,528,[{'case':'bad'}],2,2,gate),'INCOMPLETE_OR_FAILED')
    def test_reset_and_recreation_are_deterministic(self):
        a=self.sampler();b=self.sampler();d=dict(fallback_reason=None,tokens=0)
        x=[a.choose(d,i) for i in range(20)];self.assertEqual(x,[b.choose(d,i) for i in range(20)])
        a.reset();self.assertEqual(x,[a.choose(d,i) for i in range(20)]);self.assertEqual(len(a.records),20)
    def test_independent_episode_and_replicate_domains(self):
        d=dict(fallback_reason=None,tokens=0)
        values=[[s.choose(d,i) for i in range(30)] for s in (self.sampler(),self.sampler('ep1'),self.sampler(replicate=1))]
        self.assertEqual(len(set(tuple(x) for x in values)),3)
    def test_fv_potential_plans_coupled(self):
        a,b=self.sampler(),self.sampler()
        self.assertEqual(a.preflight(),b.preflight())
        self.assertEqual(len(a.preflight()['pool_families']),3)
    def test_exact_fallback_and_nearest_token(self):
        s=self.sampler();s.choose(dict(fallback_reason='cold_start_or_missing_observation',tokens=5),0)
        self.assertEqual(s.records[-1]['path'],'missing_observation');self.assertEqual(s.records[-1]['token_gap'],5)
        with self.assertRaises(ValueError):s.choose(dict(fallback_reason='absent',tokens=0),1)
    def test_pool_coverage_all_repeats_and_full_range(self):
        s=self.sampler();d=dict(fallback_reason=None,tokens=0)
        for i in range(600):s.choose(d,i)
        self.assertEqual({r['source_repeat'] for r in s.records},{0,1,2})
        self.assertGreater(max(r['pool_index'] for r in s.records),29000)
        self.assertLess(min(r['pool_index'] for r in s.records),1000)
    def test_source_row_hash_and_uniform_counter(self):
        from validation_io import object_hash
        s=self.sampler();d=dict(fallback_reason=None,tokens=0);s.choose(d,3)
        r=s.records[-1];self.assertEqual(r['source_row_sha256'],object_hash(self.rows[r['source_interleaved_row']]))
        self.assertTrue(all(0<=uniform_index('seed',i,7)<7 for i in range(100)))
    def test_changed_path_counts_stop(self):
        with self.assertRaises(ValueError):self.sampler(rows=self.rows[:-1]).preflight()
    def test_counter_limit(self):
        with self.assertRaises(ValueError):self.sampler().choose(dict(fallback_reason=None,tokens=0),600)
    def test_error_rows_excluded(self):
        rows=copy.deepcopy(self.rows[:5]);rows[0]['error']='timing failure'
        s=self.sampler(rows=rows)
        for i in range(20):s.choose(dict(fallback_reason=None,tokens=0),i)
        self.assertTrue(all(r['source_interleaved_row']!=0 for r in s.records))
    def test_wilson_known_values(self):
        self.assertAlmostEqual(wilson(8,16)['wilson95_low'],.2799956,places=6)
        self.assertEqual(wilson(0,16)['wilson95_low'],0.)
        self.assertAlmostEqual(wilson(16,16)['wilson95_high'],1.)
    def test_seed_and_replicate_count_freeze(self):
        r=read_json(Path(__file__).with_name('reference.json'))
        a=design(r);b=design(r,64)
        self.assertEqual(b['total_runs'],9456);self.assertNotEqual(a,b)
        self.assertNotEqual(a,design(r,16,1))
        with self.assertRaises(ValueError):design(r,1)
    def test_resume_rejects_changed_replicate_protocol(self):
        from run_suite import prepare_output
        reference=read_json(Path(__file__).with_name('reference.json'))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'project';root.mkdir();out=root/'runs/new_diagnostic'
            minimal={'D_run':'runs/original','sources':{}}
            env={'checked_at_utc':'synthetic','identity':'fixed'}
            prepare_output(root,out,minimal,False,env,design(reference,2))
            prepare_output(root,out,minimal,True,env,design(reference,2))
            with self.assertRaises(ValueError):prepare_output(root,out,minimal,True,env,design(reference,3))
    def test_tied_nearest_tokens_pool(self):
        rows=[dict(request=i,path='normal',fallback_reason=None,tokens=t,e2e_ms=float(i),error=None) for i,t in enumerate((2,4,6))]
        s=self.sampler(rows=rows)
        for i in range(20):s.choose(dict(fallback_reason=None,tokens=3),i)
        self.assertEqual({r['tokens'] for r in s.records},{2,4})
        self.assertTrue(all(r['token_gap']==1 for r in s.records))
    def test_full_rankings_probability_and_counts_fixture(self):
        eps=[f'e{i}' for i in range(24)];records={}
        base={'pi':3.,'fv_mpc':1.,'pino_mpc':2.,'shared_physics':1.8,'small_physics':1.9}
        for m in METHODS:
            for e in eps:records[('legacy','zero','zero',m,e)]={'episode_cost':base[m]}
        for rep in range(4):
            for m in METHODS:
                val=4. if m=='fv_mpc' and rep%2==0 else base[m]
                for e in eps:records[(f'mc{rep:04d}','zero','delayed',m,e)]={'episode_cost':val}
            for e in eps:records[(f'mc{rep:04d}','hold','delayed','fv_mpc',e)]={'episode_cost':1.1}
        out=summarize(records,eps,4)
        self.assertEqual(out['complete_replicates'],4);self.assertEqual(len(out['rankings']),8)
        self.assertEqual(len(out['cells']),24)
        p=next(x for x in out['probabilities'] if x['fv_fallback']=='zero' and x['event']=='FV_strictly_highest')
        self.assertEqual(p['successes'],2);self.assertEqual(p['proportion'],.5)
        del records[('mc0003','zero','delayed','pi','e0')]
        self.assertEqual(summarize(records,eps,4)['complete_replicates'],3)
    def test_ties_are_explicit_and_not_lexical_wins(self):
        out=rank_summary({m:1. for m in METHODS})
        self.assertEqual(set(out['ranks'].values()),{1});self.assertEqual(len(out['ties']),10)
    def test_replay_row_audit_rejects_tamper(self):
        import gzip,json
        from analyze_results import replay_audit
        from validation_io import write_json
        s=self.sampler();plan=s.preflight();trace=[]
        for i in range(3):
            d=dict(fallback_reason=None,tokens=0);delay,ref,gap=s.choose(d,i)
            trace.append(dict(event='request',request=i,trace_request=ref,delay_ms=delay,trace_token_gap=gap,**d))
        c=dict(mode='delayed',sequence='mc0000',method='fv_mpc',episode='ep0',replicate=0)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write_json(p/'draw_plan.json',plan)
            def save():
                with gzip.open(p/'replay_rows.json.gz','wt',encoding='utf-8') as f:json.dump(s.records,f)
            save();self.assertIsNotNone(replay_audit(p,c,trace,self.rows,20261013))
            s.records[0]['source_interleaved_row']+=1;save()
            with self.assertRaises(ValueError):replay_audit(p,c,trace,self.rows,20261013)
    def test_analyzer_sealed_synthetic_episode_end_to_end(self):
        import gzip,json,csv
        from analyze_results import analyze
        from validation_io import object_hash,sha256
        reference=read_json(Path(__file__).with_name('reference.json'));d=design(reference,2)
        job=next(j for j in d['jobs'] if j['phase']=='monte_carlo' and j['method']=='fv_mpc')
        case=job['cases'][0];rows=copy.deepcopy(self.rows)
        for r in rows:r['e2e_ms']=0.
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);write_json(out/'protocol.json',{'design':d})
            pool=out/'timing_pools/fv_mpc';pool.mkdir(parents=True)
            with gzip.open(pool/'rows.json.gz','wt',encoding='utf-8') as f:json.dump(rows,f)
            seal(pool,{'method':'fv_mpc','pool_object_sha256':object_hash(rows)})
            folder=out/'cases'/case['key']/'attempt_fixture';folder.mkdir(parents=True)
            condition={**{k:v for k,v in job.items() if k!='cases'},**case};write_json(folder/'condition.json',condition)
            s=ReplaySampler(rows,'mc0000',None,seed=d['seed'],method='fv_mpc',episode=case['episode'],replicate=0)
            write_json(folder/'draw_plan.json',s.preflight());trace=[]
            for i in range(600):
                now=i*100000000;decision=dict(fallback_reason=None,tokens=0);delay,ref,gap=s.choose(decision,i)
                trace.append(dict(event='request',request=i,snapshot_ns=now,ready_ns=now,compute_ready_ns=now,deadline_ns=now+100000000,expired=False,delay_ms=delay,trace_request=ref,trace_token_gap=gap,**decision))
                trace.append(dict(event='apply',request=i,time_ns=now,command=[0.]*4,fallback=False,no_change=i>0))
            with gzip.open(folder/'trace.jsonl.gz','wt',encoding='utf-8') as f:
                for r in trace:f.write(json.dumps(r)+'\n')
            with gzip.open(folder/'replay_rows.json.gz','wt',encoding='utf-8') as f:json.dump(s.records,f)
            with gzip.open(folder/'truth.csv.gz','wt',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['time_ns','goal_c','max_temperature_c']);w.writeheader()
                for i in range(601):w.writerow(dict(time_ns=i*100000000,goal_c=0.,max_temperature_c=1.))
            np.save(folder/'final_T64.npy',np.ones((64,64)))
            dt=np.full(6000,10000000,dtype=np.int32)
            np.savez_compressed(folder/'quadrature.npz',end_ns=np.cumsum(dt,dtype=np.int64),dt_ns=dt,mse_k2=np.ones(6000),command_mse=np.zeros(6000),fallback=np.zeros(6000,dtype=bool),peak_c=np.ones(6000))
            metrics=dict(duration_s=60.,completed=True,episode_cost=.00015,tracking_rmse_k=1.,max_temperature_c=1.,violation_s=0.,action_effort=0.,fallback_s=0.,action_delta_mean_square=0.,requests=600,missed_deadlines=0,busy_skipped_ticks=0,filter_interventions=0)
            write_json(folder/'metrics.json',metrics);seal(folder,{'case':case['key'],'protocol_sha256':sha256(out/'protocol.json')})
            report=analyze(out)
            self.assertEqual(report['completed_verified_cases'],1);self.assertTrue(report['independent_metric_audit_passed'])
            self.assertEqual(report['state'],'INCOMPLETE_OR_FAILED');self.assertEqual(report['expected_cases'],528)
            self.assertFalse(report['final_input_integrity_passed'])

if __name__=='__main__':unittest.main()


