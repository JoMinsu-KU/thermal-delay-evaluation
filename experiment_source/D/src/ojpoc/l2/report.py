from collections import defaultdict
import numpy as np
from ..common import read_json,write_json,atomic_text
from .contracts import ABLATIONS
from .evaluation import paired,regret_pairs


def aggregate(rows):
    return {'episodes':len(rows),'cost_mean':float(np.mean([r['episode_cost'] for r in rows])),
            'tracking_rmse_mean_k':float(np.mean([r['tracking_rmse_k'] for r in rows])),
            'maximum_overshoot_k':max(r['overshoot_k'] for r in rows),
            **{key:sum(r[key] for r in rows) for key in ['initial_violation','new_violation','violation_s','fallback_s','requests','missed_deadlines','busy_skipped_ticks','inference_failures','filter_interventions']},
            'maximum_temperature_c':max(r['max_temperature_c'] for r in rows),
            'action_effort_mean':float(np.mean([r['action_effort'] for r in rows])),
            'action_delta_mean_square':float(np.mean([r['action_delta_mean_square'] for r in rows]))}


def final_report(run,cfg,output):
    loops={};comparisons={};methods=cfg['l2']['methods']+ABLATIONS
    for mode,stage in [('zero','19_closed_loop'),('delayed','20_delayed_loop')]:
        rows={m:read_json(run/stage/(m+'.json')) for m in methods};groups=sorted({r['cohort'] for r in rows['shared_physics']})
        groups+=['ID-'+s for s in ['normal','stress'] if any(r['cohort']=='ID' and r['stratum']==s for r in rows['shared_physics'])]
        loops[mode]={};comparisons[mode]={}
        for group in groups:
            chosen={m:[r for r in rr if r['cohort']==group or (group.startswith('ID-') and r['cohort']=='ID' and r['stratum']==group[3:])] for m,rr in rows.items()}
            loops[mode][group]={m:aggregate(rr) for m,rr in chosen.items() if rr}
            comparisons[mode][group]={}
            for reference in ['pi','fv_mpc','pino_mpc','shared_no','small_physics','native_physics','shared_sensor','shared_thermal',*ABLATIONS]:
                left=chosen['shared_physics'];right=chosen[reference]
                ids={r['episode'] for r in right};left=[r for r in left if r['episode'] in ids]
                if not left or not right:continue
                diff=paired(left,right,cfg);a=aggregate(left);b=aggregate(right)
                safe=(a['new_violation']<=b['new_violation'] and a['initial_violation']<=b['initial_violation'] and
                      a['violation_s']<=b['violation_s']+1e-8 and a['inference_failures']<=b['inference_failures'])
                allowance=cfg['l2']['noninferiority_fraction']*b['cost_mean']
                comparisons[mode][group][reference]={**diff,'reference_cost_mean':b['cost_mean'],'margin_absolute':allowance,
                                                    'violation_failure_nonworsening':safe,'noninferior':bool(diff['ci95'] and diff['ci95'][1]<=allowance and safe)}
    latency={m:[read_json(run/'18_latency/trials'/f'{m}_{i}'/'summary.json') for i in range(cfg['budget']['latency_repeats'])] for m in methods}
    speed=all(v['e2e_ms']['p99'] is not None and v['e2e_ms']['p99']<=100 and v['failures']==0
              for repeat in latency['shared_physics'] for v in repeat['by_path'].values())
    train=read_json(run/'08_shared_physics/training.json')
    gate5=all(comparisons['delayed']['ID'][m]['noninferior'] for m in ['pi','pino_mpc'])
    operator=read_json(run/'21_ablation/operator_validation.json')
    pin=operator['pino_balance'];threshold=cfg['acceptance']['operator']
    gate3=(pin['horizon_rmse_k']<=threshold['horizon_rmse_k'] and pin['candidate_delta_rmse_k']<=threshold['candidate_delta_rmse_k']
           and pin['peak_error_k']<=threshold['peak_abs_error_k'] and pin['order_accuracy'] is not None and pin['order_accuracy']>=threshold['significant_order_accuracy'])
    resources=[read_json(p) for p in run.glob('*/guard.json') if p.parent.name!='22_report']
    resource_ok=all(r.get('limit_bytes') and r['peak_used_bytes']<=r['limit_bytes'] and not any(r.get('memory_events_delta',{}).get(k,0) for k in ['oom','oom_kill']) for r in resources)
    resume=read_json(run/'14_reload/resume_probe.json')
    gates={'G0_environment':True,'G1_data':True,'G2_observer':True,'G3_operator_new_validation':gate3,
           'G4_shared_policy_learning':train['quality_pass'],'G5_ID_delayed_vs_PI_and_PINO_MPC':gate5,
           'G6_shared_path_p99_100ms':speed,'G7_core_resources_and_optimizer_resume':bool(cfg['profile']=='core' and resource_ok and resume['passed'])}
    result={'profile':cfg['profile'],'evaluation_completed':True,'gates':gates,'quality_pass':all(gates.values()),
            'loops':loops,'paired_comparisons':comparisons,'policy_regret_pairs':regret_pairs(run,cfg),
            'operator_validation':operator,'resources':{'observed_within_quota':resource_ok,'peak_cgroup_gib':max(r['peak_used_bytes'] for r in resources)/2**30,'peak_rss_gib':max(r['peak_process_tree_rss_bytes'] for r in resources)/2**30,'core_required_for_G7':True},'resume_probe':resume,'calibration':read_json(run/'16_calibration/calibration.json'),
            'latency':latency,'real_equipment_validated':False,
            'scope':'L2 pilot/core shared multimodal evaluation; optional L3 KD/RGB/9B excluded',
            'OOD_is_reported_separately':True,'test_used_for_model_selection':False,'probability_and_regret_metrics':{m:read_json(run/'17_evaluation'/(m+'.summary.json')) for m in methods},
            'limitations':['one training seed','latency uses repeated finite input rings and virtual-time replay',
                           'operator checkpoints transferred from L1 pilot, not newly trained at L2 core budget',
                           'initial high-temperature stress is separate from new violations',
                           'physics zero/shuffle/operator swaps are inference diagnostics, not retrained controls',
                           'native-original and diagnostic ablations use preregistered loop subsets',
                           'cross-modality regret uses each modality-consistent FV teacher; compare cross-modality closed-loop cost on the common plant',
                           'passing sampled p99 and calibrated margins is not a safety/real-time guarantee']}
    if cfg['profile']=='smoke':result['scientific_status']='SMOKE_ONLY'
    else:result['scientific_status']='QUALITY_PASS' if result['quality_pass'] else 'QUALITY_NOT_ESTABLISHED'
    write_json(output/'validation_summary.json',result)
    lines=['# L2 실험 결과','',f'Profile: {cfg["profile"]}; scientific status: {result["scientific_status"]}',
           '실행 완료와 효과/실시간 기준 통과를 분리한다. 비용은 낮을수록 좋다.','',
           '| Gate | Pass |','|---|---|',*[f'| {k} | {v} |' for k,v in gates.items()]]
    for mode in loops:
        for group,block in loops[mode].items():
            lines+=['',f'## {mode} / {group}','',
                    '| Method | Episodes | Cost | Tracking K | Overshoot K | Initial/new violation | Fallback s | Missed/request |',
                    '|---|---:|---:|---:|---:|---|---:|---|']
            for method,m in block.items():
                lines.append(f'| {method} | {m["episodes"]} | {m["cost_mean"]:.7g} | {m["tracking_rmse_mean_k"]:.5g} | {m["maximum_overshoot_k"]:.5g} | {m["initial_violation"]}/{m["new_violation"]} | {m["fallback_s"]:.5g} | {m["missed_deadlines"]}/{m["requests"]} |')
            lines+=['','Shared Physics − reference (episode bootstrap 95% CI):','']
            for ref,c in comparisons[mode][group].items():lines.append(f'- {ref}: {c["difference"]}, CI={c["ci95"]}, paired episodes={c["episodes"]}, noninferior={c["noninferior"]}.')
    lines+=['','## 경로별 지연','', '| Method | Repeat | Path | n | p50 ms | p95 ms | p99 ms | Misses | Errors |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for method,repeats in latency.items():
        for row in repeats:
            for path,m in row['by_path'].items():
                q=m['e2e_ms'];lines.append(f'| {method} | {row["repeat"]} | {path} | {m["requests"]} | {q["p50"]} | {q["p95"]} | {q["p99"]} | {m["deadline_misses"]} | {m["failures"]} |')
    lines+=['','## 해석 범위','',*[f'- {s}' for s in result['limitations']],
            '', '모달리티 제거군은 제거된 값/mask/age와 observer 입력을 함께 제거했다. PLC actuator feedback은 세 조건 모두 유지한다.',
            '모든 모델은 같은 train/validation episode와 anchor 예산을 사용한다. native/shared/small의 optimizer step·노출 횟수는 명세 예산에 따라 다르며 각 training.json에 기록한다.',
            '고정된 G3 기준 미달은 quality 실패로 남긴다. 기존 L1 G3를 통과한 동결 모델로 진단 평가를 완주한 것이 신규 validation 기준을 통과했다는 뜻은 아니다.',
            'Calibration probability와 physics-margin episode는 겹치지 않는다. 최종 test/OOD는 선택·calibration 동결 후에만 성능을 평가한다.',
            '원시 regret/확률 calibration 지표는 17_evaluation, 물리 제거 실험은 21_ablation, 자원/실행 시간은 각 단계 guard.json과 summary.json을 참조한다.']
    atomic_text(output/'report.md','\n'.join(lines)+'\n');return result


def operator_validation(run,cfg,output):
    import torch
    from ..common import setup_torch
    from ..data import EpisodeCache
    from ..physics import branch
    from ..evaluation import tensor_inputs,operator_metrics
    from .cache import Rows,operator
    from .contracts import entries
    setup_torch(cfg);rows=Rows(run/'05_cache/both/validation');raw=EpisodeCache(run/'02_generate/data',entries(run,['validation']))
    result={}
    try:
        for variant in ['fno_data','pino_soft','pino_balance']:
            op=operator(run,variant);metrics=[]
            for i,(code,step) in enumerate(rows.arrays['index']):
                if not rows.arrays['usable'][i]:continue
                obs=raw.get(code).snapshot(step);field=rows.arrays['field'][i];commands=rows.arrays['commands'][i]
                reference=branch(field,obs.actuator,commands,obs.passport)
                inputs=tensor_inputs(field,obs,commands,'cuda')
                with torch.no_grad():
                    predicted,unprojected=op(*inputs,return_raw=True)
                    m=operator_metrics(predicted,unprojected,reference,*inputs,obs,{k:v[i] for k,v in rows.arrays.items()},1e-6)
                metrics.append(m)
            result[variant]={
                'horizon_rmse_k':float(np.sqrt(np.mean([m['horizon_mse'] for m in metrics]))),
                'candidate_delta_rmse_k':float(np.sqrt(np.mean([m['candidate_delta_mse'] for m in metrics]))),
                'field_rmse_by_horizon_k':np.sqrt(np.mean([m['field_mse_by_horizon'] for m in metrics],0)).tolist(),
                'peak_error_k':max(m['peak_error_k'] for m in metrics),
                'order_accuracy':sum(m['order_correct'] for m in metrics)/sum(m['order_pairs'] for m in metrics) if sum(m['order_pairs'] for m in metrics) else None,
                **{k:float(np.mean([m[k] for m in metrics])) for k in ['cost_mae','local_pde_mae_w_m2','raw_global_balance_mae_w_m2','projected_global_balance_mae_w_m2']},
                'validation_rows':len(metrics),'scope':'fixed observer-state FV branches; transferred L1 operator; no test selection'}
            write_json(output/(variant+'_rows.json'),metrics);del op
        write_json(output/'operator_validation.json',result);return result
    finally:rows.close();raw.close()
