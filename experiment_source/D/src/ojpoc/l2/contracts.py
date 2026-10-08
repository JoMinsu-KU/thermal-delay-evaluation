from pathlib import Path
import copy
import numpy as np
import yaml
from ..common import read_json, require, digest

BASE_METHODS = ['pi','fv_mpc','fno_mpc','pino_mpc','native_original','native_no','native_physics',
                'shared_no','shared_physics','small_physics','shared_sensor','shared_thermal']
TRAIN_METHODS = ['shared_no','shared_physics','small_physics','shared_sensor','shared_thermal','native_no','native_physics']
ABLATIONS = ['physics_zero','physics_shuffle','physics_fno','physics_soft']


def residual_enabled(cfg):
    return cfg['l2'].get('residual_learning', {}).get('enabled', False)


def ablations(cfg):
    return cfg['l2'].get('ablation_methods', ABLATIONS)


def train_methods(cfg):
    return cfg['l2'].get('train_methods', TRAIN_METHODS)


def configuration(root, profile=None, config_file=None):
    path=(root/(config_file or 'configs/l2.yaml')).resolve()
    require(path.parent==root.resolve()/'configs' and path.name.startswith('l2') and path.suffix=='.yaml','Use a project L2 config')
    settings=yaml.safe_load(path.read_text())
    profile=profile or settings['default_profile']
    require(settings['schema_version']=='thermal-l2-v1' and profile in settings['profiles'],'Invalid L2 profile')
    residual=settings.get('residual_learning',{}).get('enabled',False)
    if residual:
        require(settings['protocol_revision']=='20260928-paired-cost-residual-v1','Unknown residual protocol')
        require(settings['methods']==['pi','fv_mpc','pino_mpc','shared_physics','small_physics'] and
                settings['train_methods']==['shared_physics','small_physics'] and settings['ablation_methods']==[],
                'Preserve paired residual roster')
        require(not settings.get('rollout_learning',{}).get('enabled',False),'Residual arms must share collection data')
        r=settings['residual_learning'];rb=r['profiles'][profile]
        require(r['alphas']==[0.,.5,1.] and r['bound']==4. and r['minimum_improvement_fraction']==.05,'Preserve residual hypothesis')
        require(r['selection_delays']==['zero','38ms'] and r['label_mode']=='fv_hold_2s','Unknown residual selection/teacher')
        require(0<rb['train_episodes']<=settings['profiles'][profile]['counts']['train'] and rb['anchors']>0,'Invalid common collection budget')
        require(settings['physics_feature_version']=='candidate_relative_v3' and settings['inference_backend']=='cuda_graph','Residual execution contract')
        require(settings['noninferiority_fraction']==.05 and settings['require_quality_default'],'Residual requires original quality gates')
    else:
        require(settings['methods']==BASE_METHODS and settings['train_methods']==TRAIN_METHODS,'Locked model roster changed')
        require(settings['ablation_methods']==ABLATIONS,'Locked ablation roster changed')
    require(settings['deadline_ms']==100 and settings['bootstrap_resamples']>=2000,'Preserve acceptance criteria')
    require(settings.get('physics_feature_version','legacy') in ['legacy','energy_scaled_v2','candidate_relative_v3'],'Unknown physics representation')
    require(settings.get('inference_backend','eager') in ['eager','cuda_graph'],'Unknown inference backend')
    require(settings.get('shared_microbatch',1) in [1,2,4,8,16],'Unsupported shared microbatch')
    poc=yaml.safe_load((root/'configs/poc.yaml').read_text())
    require(poc['runtime']['actuation_enabled'] is False and poc['runtime']['mode']=='simulation_only','Simulation only')
    b=settings['profiles'][profile]
    rollout=settings.get('rollout_learning',{})
    if rollout.get('enabled',False):
        rb=rollout['profiles'][profile]
        require(settings.get('physics_feature_version')=='candidate_relative_v3' and settings.get('inference_backend')=='cuda_graph','Rollout requires revised shared inference')
        require(rb['rounds']>0 and rb['steps']>0 and 0<rb['train_episodes']<=b['counts']['train'],'Invalid rollout budget')
        require(0<rb['anchors']<=int(b['duration_s']*10)-21,'Invalid rollout anchors')
        require(rb['delays']==['zero','13ms','38ms','50ms','jitter'],'Preserve preregistered development delays')
    require(all(v>0 for k,v in b.items() if k!='counts'),'Positive experiment budgets required')
    require(b['counts']['calibration']%2==0 and b['counts']['ood']%4==0,'Balanced calibration/OOD groups required')
    require(b['duration_s']>2 and b['anchors']<=int(b['duration_s']*10)-20,'Insufficient future horizon')
    if profile!='smoke':
        require(b['duration_s']==60 and b['latency_requests']>=10000 and b['latency_repeats']>=3,'Full latency/episode budgets required')
    return {'schema':'thermal-l2-resolved-v1','seed':settings['seed'],'profile':profile,'poc':poc,
            'config_source':str(path.relative_to(root.resolve())),
            'acceptance':yaml.safe_load((root/'configs/acceptance.yaml').read_text()),
            'l2':{k:copy.deepcopy(v) for k,v in settings.items() if k!='profiles'},'budget':copy.deepcopy(b)}


def source_hashes(root):
    paths=[root/'run_l2.sh',root/'configs/l2.yaml',root/'configs/poc.yaml',root/'configs/acceptance.yaml',
           root/'scripts/activate_jev.sh',root/'scripts/env.sh',root/'locks/openjev.commit',root/'locks/requirements.lock.txt',
           root/'stages/L0/guard.py',root/'stages/L0/common.py']
    paths+=sorted((root/'src/ojpoc').rglob('*.py'))+sorted((root/'stages/L2').rglob('*.py'))+sorted((root/'tests/l2').glob('*.py'))
    paths+=sorted((root/'configs').glob('l2_*.yaml'))+sorted(root.glob('run_l2_*.sh'))
    return {str(p.relative_to(root)):digest(p) for p in paths}


def modality(method):
    return 'sensor' if method=='shared_sensor' else 'thermal' if method=='shared_thermal' else 'both'


def model_kind(method):
    if method in ABLATIONS: return 'shared_physics'
    return method


def has_physics(method):
    return method in ['shared_physics','small_physics','shared_sensor','shared_thermal','native_physics',*ABLATIONS]


def plan(cfg):
    entries=[]; counts=cfg['budget']['counts']
    for split in ['train','validation','calibration','test','ood']:
        for i in range(counts[split]):
            code=cfg['l2'].get('episode_code_offset',10000)+len(entries); group=['parameters','observation','delay','mismatch'][i%4] if split=='ood' else None
            within=i//4
            condition=None
            if group=='parameters': condition=['capacity_low','capacity_high','conductivity_low','conductivity_high'][within%4]
            if group=='observation': condition=['probe_noise','thermal_noise','sensor_burst','thermal_occlusion'][within%4]
            if group=='delay': condition=f'compute_{[0,50,100,200][within%4]}_apply_{20 if within%2 else 0}'
            if group=='mismatch': condition='heater_gain' if within%2==0 else 'surface_loss'
            role=('probability' if i<counts[split]//2 else 'physics_margin') if split=='calibration' else None
            entries.append({'code':code,'episode_id':f'l2_{code:06d}','group_id':f'l2_plant_recipe_{code:06d}',
                            'split':split,'sim_seed':cfg['seed']*10000+code,'noise_seed':cfg['seed']*10000+code+10**8,
                            'render_seed':cfg['seed']*10000+code+2*10**8,'ood_group':group,'ood_condition':condition,
                            'calibration_role':role,'stratum':'stress' if i%4==3 and split!='ood' else 'normal'})
    return entries


def anchors(cfg):
    values=np.linspace(1,int(cfg['budget']['duration_s']*10)-20,cfg['budget']['anchors'],dtype=int)
    require(len(set(values))==len(values),'Repeated anchors')
    return values


def entries(run, splits=None):
    rows=read_json(run/'02_generate/data/manifests/split.json')['episodes']
    return [r for r in rows if splits is None or r['split'] in splits]


def assert_frozen(run):
    freeze=read_json(run/'15_freeze/frozen.json')
    for name,expected in freeze['files'].items():
        require(digest(run/name)==expected,f'Frozen model/calibration changed: {name}')
    return freeze
