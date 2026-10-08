"""Separate original L2 acceptance from the added-value hypothesis for residual JEV."""
from itertools import combinations
import numpy as np
from ..common import read_json,write_json,atomic_text,require
from .report import aggregate
from .evaluation import paired

LABELS={'pi':'PI','fv_mpc':'FV-MPC (teacher reference)','pino_mpc':'PINO-MPC',
        'shared_physics':'PINO + JEV correction','small_physics':'PINO + small correction'}


def compare(left,right,cfg):
    require(left and len({r['episode'] for r in left})==len(left),'Duplicate/empty episodes')
    require({r['episode'] for r in left}=={r['episode'] for r in right} and len(right)==len(left),'Unpaired comparison')
    diff=paired(left,right,cfg);a=aggregate(left);b=aggregate(right)
    safe=(a['new_violation']<=b['new_violation'] and a['initial_violation']<=b['initial_violation'] and
          a['violation_s']<=b['violation_s']+1e-8 and a['inference_failures']<=b['inference_failures'])
    upper=diff['ci95'][1] if diff['ci95'] else float('inf');margin=cfg['l2']['noninferiority_fraction']*b['cost_mean']
    return {**diff,'reference_cost_mean':b['cost_mean'],'policy_cost_mean':a['cost_mean'],
            'relative_cost_change':a['cost_mean']/b['cost_mean']-1 if b['cost_mean']>0 else None,
            'margin_absolute':margin,'violation_failure_nonworsening':safe,
            'noninferior':bool(upper<=margin and safe),'superior':bool(upper<0 and safe)}


def value_gates(comparisons,alpha,minimum=.05):
    normal=comparisons['delayed']['ID-normal'];overall=comparisons['delayed']['ID']
    return {
        'R0_nonzero_JEV_correction':alpha>0,
        'R1_normal_noninferior_to_PI_and_PINO':all(normal[m]['noninferior'] for m in ['pi','pino_mpc']),
        'R2_normal_cost_improvement_vs_PINO':bool(normal['pino_mpc']['superior'] and normal['pino_mpc']['relative_cost_change']<=-minimum),
        'R3_JEV_improves_on_small_model':bool(normal['small_physics']['superior']),
        'R4_overall_noninferior_to_small_model':bool(overall['small_physics']['noninferior'])}


def final_report(run,cfg,output):
    methods=cfg['l2']['methods'];loops={};comparisons={};all_pairs={}
    for mode,stage in [('zero','19_closed_loop'),('delayed','20_delayed_loop')]:
        rows={m:read_json(run/stage/(m+'.json')) for m in methods}
        groups=sorted({r['cohort'] for r in rows['shared_physics']})+['ID-normal']
        if any(r['cohort']=='ID' and r['stratum']=='stress' for r in rows['shared_physics']):groups+=['ID-stress']
        loops[mode]={};comparisons[mode]={};all_pairs[mode]={}
        for group in groups:
            selected={m:[r for r in rr if r['cohort']==group or (group.startswith('ID-') and r['cohort']=='ID' and r['stratum']==group[3:])] for m,rr in rows.items()}
            loops[mode][group]={m:aggregate(z) for m,z in selected.items()}
            comparisons[mode][group]={m:compare(selected['shared_physics'],selected[m],cfg) for m in methods if m!='shared_physics'}
            all_pairs[mode][group]={a+' minus '+b:compare(selected[a],selected[b],cfg) for a,b in combinations(methods,2)}
    latency={m:[read_json(run/'18_latency/trials'/f'{m}_{i}'/'summary.json') for i in range(cfg['budget']['latency_repeats'])] for m in methods}
    speed=all(v['e2e_ms']['p99'] is not None and v['e2e_ms']['p99']<=100 and v['failures']==0
              for repeat in latency['shared_physics'] for v in repeat['by_path'].values())
    train=read_json(run/'08_shared_physics/training.json');small=read_json(run/'09_small_physics/training.json')
    matched=train['batch_sequence']==small['batch_sequence'] and train['training_data_hash']==small['training_data_hash']
    require(matched,'Cannot interpret an unmatched residual experiment')
    operator=read_json(run/'21_ablation/operator_validation.json');p=operator['pino_balance'];t=cfg['acceptance']['operator']
    g3=(p['horizon_rmse_k']<=t['horizon_rmse_k'] and p['candidate_delta_rmse_k']<=t['candidate_delta_rmse_k'] and
        p['peak_error_k']<=t['peak_abs_error_k'] and p['order_accuracy'] is not None and p['order_accuracy']>=t['significant_order_accuracy'])
    resources=[read_json(f) for f in run.glob('*/guard.json') if f.parent.name!='22_report']
    resource_ok=all(r.get('limit_bytes') and r['peak_used_bytes']<=r['limit_bytes'] and not any(r.get('memory_events_delta',{}).get(k,0) for k in ['oom','oom_kill']) for r in resources)
    resume=read_json(run/'14_reload/resume_probe.json')
    original={'G0_environment':True,'G1_data':True,'G2_observer':True,'G3_operator_new_validation':bool(g3),
              'G4_shared_policy_learning':train['quality_pass'],
              'G5_ID_delayed_vs_PI_and_PINO_MPC':all(comparisons['delayed']['ID'][m]['noninferior'] for m in ['pi','pino_mpc']),
              'G6_shared_path_p99_100ms':speed,'G7_core_resources_and_optimizer_resume':bool(cfg['profile']=='core' and resource_ok and resume['passed'])}
    value=value_gates(comparisons,train['selection']['selected_alpha'],cfg['l2']['residual_learning']['minimum_improvement_fraction'])
    gates={**original,**value};passed=all(gates.values())
    status='SMOKE_ONLY' if cfg['profile']=='smoke' else 'QUALITY_PASS' if passed else 'QUALITY_NOT_ESTABLISHED'
    result={'profile':cfg['profile'],'evaluation_completed':True,'gates':gates,'original_gates':original,'added_value_gates':value,
            'quality_pass':passed,'scientific_status':status,'loops':loops,'paired_comparisons':comparisons,'all_pair_comparisons':all_pairs,
            'latency':latency,'paired_training_data':matched,'resume_probe':resume,'operator_validation':operator,
            'selected_alpha':{m:read_json(run/s/'selection.json')['selected_alpha'] for m,s in [('shared_physics','08_shared_physics'),('small_physics','09_small_physics')]},
            'resources':{'observed_within_quota':resource_ok,'peak_cgroup_gib':max(r['peak_used_bytes'] for r in resources)/2**30,
                         'peak_rss_gib':max(r['peak_process_tree_rss_bytes'] for r in resources)/2**30,'core_required_for_G7':True},
            'probability_and_regret_metrics':{m:read_json(run/'17_evaluation'/(m+'.summary.json')) for m in methods},
            'real_equipment_validated':False,'test_used_for_model_selection':False,
            'limitations':['One training seed; paired episode bootstrap does not measure training-seed uncertainty.',
                          'Both learners use identical 2s FV hold-action labels and mixed PI/PINO train-state data; no long-horizon teacher.',
                          'Identical optimizer steps and batches do not imply identical parameter count or training FLOPs.',
                          'Synthetic validation delays select alpha; final timing uses measured full-path traces.',
                          'Zero alpha reproduces PINO ranking but retains model overhead; it is not JEV added value.',
                          'OOD and initial-hot stress are reported separately; no deployment or hard realtime guarantee.',
                          'Normal-stratum 5% improvement and superiority over the small correction are extra promotion conditions; original G3-G6 are unchanged.']}
    write_json(output/'validation_summary.json',result)
    lines=['# L2 PINO 비용 보정 비교 결과','',f'Profile: {cfg["profile"]}; status: {status}',
           '실행 완료, 기존 기준 통과, JEV 추가 가치를 구분한다. 비용은 낮을수록 좋다.',
           f'동일 학습 데이터·배치 순서 확인: {matched}; 선택 보정 강도: {result["selected_alpha"]}',
           '', '| 판정 | 통과 |','|---|---|',*[f'| {k} | {v} |' for k,v in gates.items()]]
    for mode,groups in loops.items():
        for group,block in groups.items():
            lines+=['',f'## {mode} / {group}','','| 방식 | n | 비용 | Tracking K | 최대 overshoot K | 초기/신규 위반 | Miss / 요청 |','|---|---:|---:|---:|---:|---|---|']
            for m,v in block.items():
                lines.append(f'| {LABELS[m]} | {v["episodes"]} | {v["cost_mean"]:.8g} | {v["tracking_rmse_mean_k"]:.5g} | {v["maximum_overshoot_k"]:.5g} | {v["initial_violation"]}/{v["new_violation"]} | {v["missed_deadlines"]}/{v["requests"]} |')
            lines+=['','JEV 보정 − 기준: episode paired bootstrap 95% CI; 음수가 유리하다.']
            for ref,c in comparisons[mode][group].items():
                lines.append(f'- {LABELS[ref]}: 차이 {c["difference"]:.7g}, CI {c["ci95"]}, 비용 변화 {c["relative_cost_change"]:.2%}, 비열등={c["noninferior"]}, 우월={c["superior"]}.')
    lines+=['','## 실제 전체 경로 지연','','| 방식 | 반복 | 경로 | 요청 | p50 ms | p95 ms | p99 ms | 오류 | 시간 초과 |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for method,repeats in latency.items():
        for r in repeats:
            for path,v in r['by_path'].items():
                q=v['e2e_ms'];lines.append(f'| {LABELS[method]} | {r["repeat"]} | {path} | {v["requests"]} | {q["p50"]} | {q["p95"]} | {q["p99"]} | {v["failures"]} | {v["deadline_misses"]} |')
    lines+=['','## 해석 범위','',*[f'- {s}' for s in result['limitations']],
            '', '정답 FV-MPC가 PI보다 유리한지도 위 표와 validation_summary.json의 all_pair_comparisons에서 확인한다.']
    atomic_text(output/'report.md','\n'.join(lines)+'\n');return result
