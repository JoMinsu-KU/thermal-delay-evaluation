"""Frozen independent empirical latency resampling; no outcome-based selection."""
D_RUN = 'runs/L2/l2_auto_20260928T043858Z_2f3e14_pilot'
METHODS = ('pi', 'fv_mpc', 'pino_mpc', 'shared_physics', 'small_physics')
LABELS = dict(zip(METHODS, ('PI', 'FV-MPC', 'PINO-MPC', 'Shared-Residual', 'Small-Residual')))
STAGES = {'zero': '19_closed_loop', 'delayed': '20_delayed_loop'}
DEFAULT_REPLICATES = 16
DEFAULT_SEED = 20261013


def design(reference, replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED):
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError('At least two frozen replicates are required')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    episodes = [e['episode_id'] for e in reference['D_entries']]
    if len(episodes) != 24 or len(set(episodes)) != 24:
        raise ValueError('Expected original 24 distinct nominal held-out episodes')
    jobs = []
    def add(phase, method, mode, sequence, policy, replicate=None):
        name = f'{phase}_{method}_{mode}_{sequence}_{policy}'
        jobs.append(dict(id=name, phase=phase, source='D', h_ms=10., method=method,
                         mode=mode, sequence=sequence, fallback_policy=policy, replicate=replicate,
                         cases=[dict(key=name+'_'+e, episode=e) for e in episodes]))
    for method in METHODS:
        for mode in ('zero', 'delayed'):
            add('baseline', method, mode, 'legacy', 'zero')
    for rep in range(replicates):
        for method in METHODS:
            add('monte_carlo', method, 'delayed', f'mc{rep:04d}', 'zero', rep)
        add('monte_carlo', 'fv_mpc', 'delayed', f'mc{rep:04d}', 'hold', rep)
    return {
        'schema': 'jev-empirical-iid-latency-validation-v1', 'created_date': '2026-10-13',
        'actuation_enabled': False, 'training': False, 'model_or_alpha_selection': False,
        'duration_s': 60., 'plant_max_step_ms': 10., 'normal_episodes': episodes,
        'replicates': replicates, 'seed': seed, 'baseline_runs': 240,
        'monte_carlo_runs': replicates*144, 'total_runs': 240+replicates*144, 'jobs': jobs,
        'sampling': {
            'type': 'iid_with_replacement', 'pool': 'All three archived measurement repeats, with original interleaved loader order; 30000 normal-path rows plus 3000 rows for each fallback path per method before error exclusion.',
            'matching': 'Exact fallback reason, then minimum absolute token gap, including ties; error rows excluded. Never replace a missing fallback pool by normal timing.',
            'rng': 'SHA256 counter draws with rejection to remove modulo bias; key = protocol seed, method, episode, replicate, matched-pool membership hash, global request index.',
            'independence': 'Episode/replicate/method domains are distinct. FV zero and hold share the same domain and potential draw plan.',
            'pairing_limit': 'Actual selected rows and clocks remain equal only while feedback visits the same timing path/token pool; divergence is reported, never hidden by mismatched latency substitution.',
            'correlation': 'IID sampling destroys temporal correlation and pools three existing trials; it does not generate fresh hardware measurements or identify state-dependent timing.',
        },
        'hold_intervention': 'At expiration hold current applied command; preserve busy-until-completion, skipped activation, expired-result rejection, and all non-deadline fallback semantics.',
        'statistics': {
            'ranking_unit': 'One replicate is a complete set of 24 fixed episodes for all five methods plus paired FV-hold.',
            'probabilities': 'Empirical conditional Monte Carlo proportions across independent replicate draw plans; Wilson 95% intervals quantify simulation sampling precision only.',
            'distributions': 'Replicate means and changes relative to the fixed zero-delay baseline; median, 2.5/97.5 percentiles, min/max. Percentile spread is not a confidence interval.',
            'ties': 'Costs within 1e-10 are tied; strict reversals exclude ties.',
            'scope': 'Conditional on fixed models, fixed 24 episodes, observed timing pools and IID sampler. No inference over new hardware, training seeds, physical plants, or episode population.',
            'precision': '16 is a pilot budget. Choose at least 64 before launch for tighter Monte Carlo probability precision; neither budget resolves empirical-pool or hardware uncertainty.',
        },
        'reproduction_gate': 'All 240 original zero-delay/interleaved-replay cases must reproduce before Monte Carlo execution.',
        'scientific_stop_rule': 'Complete frozen replicate count regardless of signs or rankings; no early stopping or post-outcome enlargement in the same protocol.',
        'unchanged': ['frozen checkpoints and alpha', 'same data/observation noise', 'candidate/gate/solver', '100 ms decisions', '10 ms plant cap', '10 ms/event quadrature'],
        'archive_10ms_D_wall_seconds': reference['archive_10ms_D_wall_seconds'],
    }
