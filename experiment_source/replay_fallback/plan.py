"""Frozen crossed sensitivity design; no selection based on new outcomes."""
D_RUN = 'runs/L2/l2_auto_20260928T043858Z_2f3e14_pilot'
METHODS = ('pi', 'fv_mpc', 'pino_mpc', 'shared_physics', 'small_physics')
LABELS = dict(zip(METHODS, ('PI', 'FV-MPC', 'PINO-MPC', 'Shared-Residual', 'Small-Residual')))
STAGES = {'zero': '19_closed_loop', 'delayed': '20_delayed_loop'}
SEQUENCES = tuple(f'r{r}_b{b}' for r in range(3) for b in range(2))
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_RESAMPLES = 20000


def all_jobs(reference):
    jobs = []
    def add(phase, method, mode, sequence, fallback):
        name = f'{phase}_{method}_{mode}_{sequence}_{fallback}'
        jobs.append({'id': name, 'phase': phase, 'source': 'D', 'h_ms': 10.,
                     'method': method, 'mode': mode, 'sequence': sequence, 'fallback_policy': fallback,
                     'cases': [{'key': name + '_' + e['episode_id'], 'episode': e['episode_id']}
                               for e in reference['D_entries']]})
    for method in METHODS:
        for mode in ('zero', 'delayed'):
            add('baseline', method, mode, 'legacy', 'zero')
    for method in METHODS:
        for sequence in SEQUENCES:
            add('replay', method, 'delayed', sequence, 'zero')
    for sequence in ('legacy', *SEQUENCES):
        add('fallback', 'fv_mpc', 'delayed', sequence, 'hold')
    return jobs


def design(reference):
    return {
        'schema': 'jev-replay-fallback-validation-v1', 'created_date': '2026-10-08',
        'actuation_enabled': False, 'training': False, 'model_or_alpha_selection': False,
        'duration_s': 60., 'plant_max_step_ms': 10., 'jobs': all_jobs(reference),
        'normal_episodes': [e['episode_id'] for e in reference['D_entries']],
        'baseline_runs': 240, 'new_sequence_zero_fallback_runs': 720, 'hold_fallback_runs': 168,
        'total_runs': 1128, 'sequence_ids': list(SEQUENCES),
        'sequence_design': {
            'measurement_repeats': [0, 1, 2], 'normal_pool_starts': [0, 5000],
            'fallback_pool_starts': [0, 400], 'window_requests': 600,
            'rule': 'Restore each repeat from the original interleaved loader; preserve within-repeat order in each path/token-matched pool. Select start + global request_id, without wraparound.',
            'matching': 'Exact fallback reason and nearest valid token count, as in the original chooser. Timing-error records are excluded as in the original.',
            'scope': 'Six fixed conditional replay settings, not six independent hardware trials. Three original measurement repeats each supply two blocks; fallback blocks may overlap.',
        },
        'hold_intervention': 'At deadline expiration hold the currently applied vector and mark fallback=True. Preserve busy-until-compute-ready, skipped ticks, discarded late result, and all other zero fallback causes.',
        'unchanged': ['frozen weights and selected alpha', 'observations and noise', 'candidate prediction solver',
                      'calibration and gate', '100 ms request clock and deadline', '100 Hz sensor emission',
                      '10 ms/event quadrature', 'no-op application semantics', '64x64 FP64 plant'],
        'statistics': {'unit': 'paired episode', 'n': 24, 'resamples': BOOTSTRAP_RESAMPLES,
                       'seed': BOOTSTRAP_SEED, 'interval': 'conditional episode percentile 95%',
                       'sequence_uncertainty': 'Report each block and descriptive spread across six blocks; do not count 144 trajectories as independent episodes or infer a hardware-population CI.'},
        'reproduction_gate': 'All 240 legacy metrics and semantic events must reproduce before either new phase.',
        'scientific_stop_rule': 'Finish the fixed design regardless of effect direction. No retraining, reselection, added blocks, or positive-effect threshold.',
        'archive_10ms_D_wall_seconds': reference['archive_10ms_D_wall_seconds'],
    }
