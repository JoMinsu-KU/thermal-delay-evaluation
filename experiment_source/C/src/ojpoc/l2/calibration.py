import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from ..common import read_json, write_json, require, digest, setup_torch
from ..data import RawEpisode
from ..physics import branch, candidates
from .contracts import entries, anchors, assert_frozen
from .data import view
from .cache import operator, prediction, make_split
from .observer import Observer


def fit_temperature(records):
    """Fit only on the probability-calibration partition and deploy-time allowed candidates."""
    usable=[r for r in records if np.sum(r['target'])>0 and sum(r['allowed'])>0]
    if not usable:return {'temperature':1.,'fit_rows':0,'status':'NO_FITTABLE_ROWS'}
    def nll(log_t):
        total=0.
        for row in usable:
            allowed=np.array(row['allowed'],bool);target=np.array(row['target'])[allowed]
            if target.sum()==0:continue
            target=target/target.sum();scores=np.array(row['scores'])[allowed]/np.exp(log_t)
            total-=float(np.dot(target,scores-logsumexp(scores)))
        return total/len(usable)
    fit=minimize_scalar(nll,bounds=(-12,8),method='bounded')
    require(fit.success and np.isfinite(fit.fun),'Temperature fitting failed')
    return {'temperature':float(np.exp(fit.x)),'fit_rows':len(usable),'status':'FIT',
            'nll_before':nll(0.),'nll_after':nll(fit.x),'fit_split':'calibration_probability',
            'label_temperature_is_separate':True,'calibration_fit_metrics_are_not_test_metrics':True}


def margins(run,cfg,output):
    setup_torch(cfg);assert_frozen(run)
    base=run/'02_generate/data';selected=[e for e in entries(run,['calibration']) if e['calibration_role']=='physics_margin']
    result={};op=operator(run)
    for mode in ['both','sensor','thermal']:
        observer=Observer(run,mode);errors=[];episode_values={}
        for entry in selected:
            raw=RawEpisode(base/'raw'/entry['episode_id'])
            true_fields=np.load(base/'truth'/entry['episode_id']/'anchor_T64.npy',mmap_mode='r')
            true_p=np.load(base/'truth'/entry['episode_id']/'p_actual.npy',mmap_mode='r')
            values=[]
            for i,step in enumerate(anchors(cfg)):
                obs=view(raw.snapshot(step),mode)
                if not obs.usable:continue
                commands,valid=candidates(obs.previous)
                predicted,_=prediction(op,observer(obs),obs,commands)
                reference=branch(true_fields[i],true_p[step],commands,obs.passport)
                residual=reference.max((1,2,3))-predicted.cpu().numpy().max((1,2,3))
                values.append(float(max(0.,residual[valid].max())))
            raw.close();true_fields._mmap.close();true_p._mmap.close()
            require(bool(values),'No usable physics-calibration anchors')
            episode_values[entry['episode_id']]=max(values);errors.append(max(values))
        margin=float(np.quantile(errors,cfg['l2']['physics_margin_quantile'],method='higher'))
        result[mode]={'margin_k':margin,'episodes':len(errors),'episode_maximum_underprediction_k':episode_values,
                      'quantile':cfg['l2']['physics_margin_quantile'],'source':'true FV64 candidate futures minus observer+PINO peak',
                      'application':'independent safety gate only; scorer features 13/14 remain zero',
                      'coverage_guaranteed':False}
    write_json(output/'margins.json',result)
    for mode in ['both','sensor','thermal']:
        make_split(run,cfg,output/'cache'/mode/'calibration',mode,'calibration',op)
    return result


def calibrate_method(root,run,cfg,output,method):
    from .engine import Engine
    from .cache import Rows
    from .contracts import modality
    setup_torch(cfg);assert_frozen(run);mode=modality(method)
    rows=Rows(run/'16_calibration/cache'/mode/'calibration')
    selected={e['code']:e for e in entries(run,['calibration']) if e['calibration_role']=='probability'}
    engine=Engine(root,run,method,calibrated=False)
    engine.margin=read_json(run/'16_calibration/margins.json')[mode]['margin_k']
    records=[];raw=None;old=None;excluded=0
    try:
        for i,(code,step) in enumerate(rows.arrays['index']):
            code=int(code);step=int(step)
            if code not in selected:continue
            if code!=old:
                if raw:raw.close()
                raw=RawEpisode(run/'02_generate/data/raw'/selected[code]['episode_id']);old=code
            decision=engine.decide(lambda:raw.snapshot(step))
            target=np.array(rows.arrays['target'][i]);allowed=np.array(decision['predicted_safe_mask'])
            if decision['scores'] is None or target[allowed].sum()==0:excluded+=1;continue
            records.append({'scores':decision['scores'],'allowed':allowed.tolist(),'target':target.tolist()})
        result=fit_temperature(records);result['excluded_no_fittable_target_or_fallback']=excluded
        write_json(output/(method+'.json'),result);return result
    finally:
        if raw:raw.close()
        rows.close();engine.close()


def finalize(run,cfg,output):
    result={'margins':read_json(output/'margins.json'),
            'temperatures':{m:read_json(output/(m+'.json')) for m in cfg['l2']['methods']},
            'disjoint_calibration_roles':True,'test_accessed':False,
            'model_freeze_hash':digest(run/'15_freeze/frozen.json')}
    write_json(output/'calibration.json',result);return result
