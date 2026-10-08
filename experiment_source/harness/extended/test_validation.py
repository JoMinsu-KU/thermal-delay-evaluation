"""Synthetic software tests; never import the research simulator or a model."""
import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from step_plant import make_plant_class
from replay import replay_commands
from baseline_check import baseline_compare
from validation_io import (completed_case,read_json,read_latency,object_hash,seal,sha256,write_json,bundle)
from plan import all_jobs,design
from run_suite import evaluate_gate,prepare_output
from analyze_results import analyze,audit_case,paired,command_integrals
from worker import save_result


class SyntheticPlant:
    def __init__(self,base,entry,duration):
        self.now=self.tick=0;self.end_ns=round(duration*1e9);self.duration_s=duration
        self.field=np.ones((2,2),dtype=np.float64);self.p=np.zeros(4);self.u=np.zeros(4);self.goal=0.
        self.nominal=np.array([1.,0.,0.,0.,0.,1.,1.,1.,1.]);self.condition=None;self.mismatch_value=None
        self.error_integral=self.effort_integral=self.violation_s=self.fallback_s=self.delta_sum=0.
        self.peak=self.overshoot=1.;self.initial_violation=self.previous_violation=self.new_violation=False
        self.recovery_s=None;self.fallback=True;self.emissions=[];self.thermal_times=[]
        self.emit();self.emit()

    def emit(self):
        self.emissions.append(self.now)
        if self.tick%20==0:self.thermal_times.append(self.now)

    def apply(self,command,fallback):
        self.delta_sum+=float(np.mean((command-self.u)**2));self.u=command.copy();self.fallback=fallback

    def metrics(self):
        tracking=self.error_integral/self.duration_s;effort=self.effort_integral/self.duration_s
        delta=self.delta_sum/round(self.duration_s*10);terminal=float(np.mean(((self.field-self.goal)/100)**2))
        return {'episode_cost':tracking/10000+.5*terminal+.002*effort+.02*delta,
                'completed':self.now==self.end_ns,'duration_s':self.duration_s,
                'tracking_rmse_k':math.sqrt(tracking),'action_effort':effort,'action_delta_mean_square':delta,
                'max_temperature_c':self.peak,'fallback_s':self.fallback_s,'violation_s':self.violation_s}


def euler(field,p,u,theta,duration,max_dt=.01):
    count=max(1,math.ceil(duration/max_dt));value=field.copy()
    for _ in range(count):value-=duration/count*value
    return value,p.copy()


def plant(step=10.,duration=.6,truth_stride=1,field_stride=0):
    return make_plant_class(SyntheticPlant,euler)(None,None,duration,max_step_ms=step,
                                               truth_stride=truth_stride,field_stride=field_stride)


def applications():
    return [{'event':'apply','time_ns':0,'command':[0.]*4,'fallback':True,'no_change':True},
            {'event':'apply','time_ns':3_000_000,'command':[.04]*4,'fallback':False,'no_change':False},
            {'event':'apply','time_ns':13_000_000,'command':[.04]*4,'fallback':False,'no_change':True},
            {'event':'deadline_fallback','time_ns':138_000_000,'command':[0.]*4,'fallback':True,'no_change':False}]


class StepTests(unittest.TestCase):
    def test_refinement_preserves_sensor_and_thermal_clocks(self):
        errors=[]
        for h in (10.,5.,2.5,1.25):
            p=plant(h,truth_stride=10)
            for t in (3_000_000,13_000_000,38_000_000,600_000_000):p.advance_to(t)
            self.assertEqual(sorted(set(p.emissions)),list(range(0,600_000_001,10_000_000)))
            self.assertEqual(sorted(set(p.thermal_times)),[0,200_000_000,400_000_000,600_000_000])
            self.assertEqual([r['time_ns'] for r in p.truth_rows],list(range(0,600_000_001,100_000_000)))
            self.assertLessEqual(p.max_actual_dt_s,h/1000+1e-15)
            self.assertEqual(sum(r[1] for r in p.quadrature_rows),600_000_000)
            errors.append(abs(p.field.mean()-math.exp(-.6)))
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))

    def test_legacy_10ms_state_and_cost_quadrature(self):
        p=plant(duration=.2);p.advance_to(3_000_000);p.advance_to(20_000_000)
        first=1-.003;second=first*(1-.007);third=second*(1-.01)
        self.assertAlmostEqual(p.field.mean(),third,places=15)
        self.assertAlmostEqual(p.error_integral,first**2*.003+second**2*.007+third**2*.01,places=15)
        self.assertEqual(p.fv_steps,3)

    def test_fixed_noop_does_not_add_numerical_boundary(self):
        p=plant();q=plant();trace=applications()
        replay_commands(p,trace)
        q.advance_to(3_000_000);q.apply(np.full(4,.04),False)
        q.advance_to(138_000_000);q.apply(np.zeros(4),True);q.advance_to(q.end_ns)
        np.testing.assert_array_equal(p.field,q.field)
        self.assertEqual(p.quadrature_rows,q.quadrature_rows)
        self.assertNotIn(13_000_000,[r[0] for r in p.quadrature_rows])
        self.assertEqual(p.metrics()['episode_cost'],q.metrics()['episode_cost'])

    def test_fixed_rejects_inconsistent_noop_and_time(self):
        a=applications();a[1]=dict(a[1],no_change=True)
        with self.assertRaises(ValueError):replay_commands(plant(),a)
        a=applications();a[-1]=dict(a[-1],time_ns=1)
        with self.assertRaises(ValueError):replay_commands(plant(),a)

    def test_full_field_logging_does_not_change_sensor_grid(self):
        p=plant(1.25,2.,truth_stride=1,field_stride=100);p.advance_to(p.end_ns)
        self.assertEqual(p.field_times,[0,1_000_000_000,2_000_000_000])
        self.assertEqual(len(p.truth_rows),201)

    def test_independent_cost_audit_and_corruption(self):
        p=plant();tr=applications();m=replay_commands(p,tr)
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp);save_result(f,p,m,tr,True)
            self.assertTrue(audit_case(f,m,tr,p.truth_rows)['passed'])
            wrong=dict(m,episode_cost=m['episode_cost']+.01)
            self.assertFalse(audit_case(f,wrong,tr,p.truth_rows)['passed'])

    def test_invalid_step_and_clock(self):
        with self.assertRaises(ValueError):plant(1.)
        p=plant();p.advance_to(1)
        for t in (0,600_000_001):
            with self.assertRaises(ValueError):p.advance_to(t)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reference=read_json(Path(__file__).parent/'reference.json')

    def test_frozen_732_case_design(self):
        jobs=all_jobs(self.reference);cases=[c['key'] for j in jobs for c in j['cases']]
        self.assertEqual(len(jobs),34);self.assertEqual(len(cases),732);self.assertEqual(len(set(cases)),732)
        for phase,n in (('fixed',12),('baseline',240),('refine',480)):
            self.assertEqual(sum(len(j['cases']) for j in jobs if j['phase']==phase),n)
        for j in jobs:
            if j['source']=='D':self.assertEqual(len(j['cases']),24)
        self.assertTrue(all(e['split']=='test' and e['stratum']=='normal' and e['ood_group'] is None for e in self.reference['D_entries']))

    def test_gate_requires_all_240_and_rejects_one_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'protocol.json',{'design':design(self.reference)})
            self.assertFalse(evaluate_gate(root,'baseline')['passed']);ph=sha256(root/'protocol.json')
            baseline=[c for j in all_jobs(self.reference) if j['phase']=='baseline' for c in j['cases']]
            for i,c in enumerate(baseline):
                f=root/'cases'/c['key']/'attempt_1';write_json(f/'baseline_comparison.json',{'passed':i!=239})
                seal(f,{'case':c['key'],'protocol_sha256':ph})
            self.assertFalse(evaluate_gate(root,'baseline')['passed'])
            # A sealed failed result remains a failed result on resume.
            self.assertIsNotNone(completed_case(root,baseline[-1]['key'],ph))

    def test_baseline_ignores_wall_time_but_checks_actions_and_cost(self):
        old={'event':'request','choice':0,'scores':[1.,2.],'e2e_ms':20.,'command':[0.]*4}
        new=dict(old,scores=[1.001,2.001],e2e_ms=25.)
        a={'episode_cost':.1,'wall_seconds':5.};b={'episode_cost':.1,'wall_seconds':999.}
        self.assertTrue(baseline_compare(b,[new],a,[old])['passed'])
        self.assertFalse(baseline_compare(dict(b,episode_cost=.2),[new],a,[old])['passed'])
        self.assertFalse(baseline_compare(b,[dict(new,choice=1)],a,[old])['passed'])

    def test_resume_rejects_corruption_protocol_and_duplicate_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'cases/x/attempt_1';write_json(f/'metrics.json',{'cost':1.})
            seal(f,{'case':'x','protocol_sha256':'abc'})
            self.assertEqual(completed_case(root,'x','abc'),f)
            with self.assertRaises(ValueError):completed_case(root,'x','wrong')
            write_json(f/'metrics.json',{'cost':2.})
            with self.assertRaises(ValueError):completed_case(root,'x','abc')

    def test_incomplete_attempt_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);f=r/'cases/x/attempt_1/status.json';write_json(f,{'state':'FAILED'})
            self.assertIsNone(completed_case(r,'x','a'));self.assertTrue(f.exists())

    def test_generic_latency_loader_readonly_and_interleaved(self):
        for method in ('pi','fv_mpc','shared_physics'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg={'budget':{'latency_repeats':2}}
                for name in ('15_freeze/frozen.json','16_calibration/calibration.json','18_latency/rings/both/ring.json'):write_json(root/name,{})
                for repeat in range(2):
                    identity={'frozen':sha256(root/'15_freeze/frozen.json'),'calibration':sha256(root/'16_calibration/calibration.json'),
                              'ring':sha256(root/'18_latency/rings/both/ring.json'),'method':method,'repeat':repeat,'budget':cfg['budget']}
                    folder=root/'18_latency/trials'/f'{method}_{repeat}';folder.mkdir(parents=True)
                    rows=[]
                    for i,p in enumerate(('normal','missing_observation','no_predicted_feasible_candidate')):
                        row={'path':p,'request':repeat*3+i,'error':None}
                        rows.append({'identity':object_hash(identity),'index':i,'row':row,'sha256':object_hash(row)})
                    (folder/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                    seal(folder,identity)
                before={p:(sha256(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
                self.assertEqual([x['request'] for x in read_latency(root,cfg,method)],[0,3,1,4,2,5])
                self.assertEqual(before,{p:(sha256(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})

    def test_paired_bootstrap_matches_constant_effect_and_ties(self):
        idx=np.random.default_rng(1).integers(0,24,(1000,24));a=np.ones(24)
        r=paired(a,a*.9,idx)
        self.assertAlmostEqual(r['mean_B_minus_A'],-.1);self.assertAlmostEqual(r['relative_ci_low'],-10.)
        self.assertEqual(r['B_wins_tol_1e-10'],24)
        r=paired(a,a,idx);self.assertEqual(r['B_ties_tol_1e-06'],24)

    def test_partial_report_never_claims_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'protocol.json',{'design':design(self.reference)})
            result=analyze(root)
            self.assertEqual(len(result['missing_cases']),732);self.assertEqual(result['status'],'PARTIAL_OR_UNVERIFIED')
            self.assertNotIn('primary_direction_changes_observed',result)

    def test_end_to_end_partial_paired_report_from_sealed_toy_records(self):
        # Fabricated constant-temperature arrays test report wiring only.
        # They are never mixed with archived research records.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol={'design':design(self.reference)};write_json(root/'protocol.json',protocol)
            ph=sha256(root/'protocol.json')
            selected=[j for j in protocol['design']['jobs'] if j['h_ms']==10. and j.get('mode')=='zero'
                      and j.get('method') in ('shared_physics','small_physics')]
            for job in selected:
                temp=1. if job['method']=='shared_physics' else 2.
                toy=SimpleNamespace(field=np.full((2,2),temp),
                    quadrature_rows=[(t,10_000_000,temp**2,0.,True,temp) for t in range(10_000_000,60_000_000_001,10_000_000)],
                    truth_rows=[{'time_ns':t,'goal_c':0.,'max_temperature_c':temp,'mean_temperature_c':temp}
                                for t in range(0,60_000_000_001,100_000_000)])
                m={'episode_cost':temp**2*.00015,'duration_s':60.,'completed':True,'tracking_rmse_k':temp,
                   'action_effort':0.,'action_delta_mean_square':0.,'fallback_s':60.,'max_temperature_c':temp,
                   'violation_s':0.,'new_violation':False,'requests':600,'missed_deadlines':0,
                   'busy_skipped_ticks':0,'inference_failures':0,'filter_interventions':0}
                trace=[{'event':'request','snapshot_ns':t,'expired':False} for t in range(0,60_000_000_000,100_000_000)]
                for case in job['cases']:
                    f=root/'cases'/case['key']/'attempt_1';f.mkdir(parents=True)
                    write_json(f/'condition.json',{**{k:v for k,v in job.items() if k!='cases'},**case})
                    save_result(f,toy,m,trace,False);seal(f,{'case':case['key'],'protocol_sha256':ph})
            result=analyze(root)
            self.assertEqual(result['completed_audited_cases'],48)
            self.assertTrue(result['independent_metric_audit_passed'])
            self.assertEqual(result['status'],'PARTIAL_OR_UNVERIFIED')
            row=next(r for r in result['D_contrasts'] if r['contrast_B_minus_A']=='zero: Shared - small_physics')
            self.assertEqual(row['n'],24);self.assertEqual(row['B_wins_tol_1e-10'],24)
            self.assertAlmostEqual(row['mean_B_minus_A'],-.00045)

    def test_bundle_includes_failed_attempt(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'output';write_json(r/'cases/x/attempt_1/status.json',{'state':'FAILED'})
            with zipfile.ZipFile(bundle(r)) as z:self.assertEqual(len(z.namelist()),1)


if __name__=='__main__':unittest.main()
