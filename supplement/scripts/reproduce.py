"""Recompute paper summaries from released records; CPU analysis only (MIT)."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['pi', 'fv_mpc', 'pino_mpc', 'shared_physics', 'small_physics']
LABELS = ['PI', 'FV-MPC', 'PINO-MPC', 'Shared-Residual', 'Small-Residual']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def csvread(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(out, name, rows):
    with (out / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def contrast(a, b, idx):
    d = b - a
    ci = np.quantile(d[idx].mean(1), [.025, .975])
    rc = np.quantile((b[idx].mean(1) / a[idx].mean(1) - 1) * 100, [.025, .975])
    return {'mean_A': float(a.mean()), 'mean_B': float(b.mean()), 'mean_difference': float(d.mean()),
            'difference_ci_low': float(ci[0]), 'difference_ci_high': float(ci[1]),
            'ratio_of_means_pct': float((b.mean() / a.mean() - 1) * 100),
            'ratio_ci_low': float(rc[0]), 'ratio_ci_high': float(rc[1]),
            'B_wins': int((d < -1e-10).sum()), 'ties': int((abs(d) <= 1e-10).sum()),
            'B_losses': int((d > 1e-10).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs')
    ap.add_argument('--skip-plots', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if out == ROOT / 'data' or (ROOT / 'data') in out.parents:
        raise ValueError('Output must not overwrite released data.')
    out.mkdir(parents=True, exist_ok=True)
    protocol = read(ROOT / 'data/protocol/extended_protocol.json')['design']
    episodes = protocol['normal_episodes']
    assert len(episodes) == len(set(episodes)) == 24
    idx = np.random.default_rng(20261007).integers(0, 24, (20000, 24))
    primary = {}
    for mode in ('zero', 'delayed'):
        for method in METHODS:
            rs = read(ROOT / f'data/primary/{mode}_{method}.json')
            by = {r['episode']: r for r in rs if r['cohort'] == 'ID' and r['stratum'] == 'normal'}
            assert set(by) == set(episodes)
            primary[mode, method] = np.array([by[e]['episode_cost'] for e in episodes])
    table6, paired = [], []
    for method, label in zip(METHODS, LABELS):
        a, b = primary['zero', method], primary['delayed', method]
        table6.append({'method': label, **contrast(a, b, idx)})
        paired += [{'comparison': method + ': replay-zero', 'episode': e, 'difference': float(d)}
                   for e, d in zip(episodes, b - a)]
    save(out, 'table6_primary_costs.csv', table6)
    table7 = []
    for mode in ('zero', 'delayed'):
        for other in ('small_physics', 'pino_mpc'):
            a, b = primary[mode, other], primary[mode, 'shared_physics']
            table7.append({'mode': mode, 'comparison': 'Shared-' + other, **contrast(a, b, idx)})
            paired += [{'comparison': ('replay' if mode == 'delayed' else mode) + ': shared-' + other,
                        'episode': e, 'difference': float(d)} for e, d in zip(episodes, b - a)]
    save(out, 'table7_paired_comparisons.csv', table7)
    save(out, 'paired_episode_differences.csv', paired)
    table5 = [r for r in csvread(ROOT / 'data/primary/latency_quantiles.csv')]
    # The raw audit recomputes latency quantiles; this file preserves pooled and repeat rows.
    save(out, 'table5_latency_all_scopes.csv', table5)

    metrics = csvread(ROOT / 'data/numerical/extended_metrics.csv')
    assert len(metrics) == 732
    mr = {(float(r['h_ms']), r['method'], r['mode'], r['episode']): r for r in metrics if r['source'] == 'D'}
    assert len(mr) == 720
    def values(h, m, mode):
        return np.array([float(mr[h, m, mode, e]['episode_cost']) for e in episodes])
    all_contrasts, table8, rankings = [], [], []
    for h in (10., 5., 2.5):
        for m in METHODS:
            row = {'h_ms': h, 'comparison': m + ' delay', **contrast(values(h, m, 'zero'), values(h, m, 'delayed'), idx)}
            all_contrasts.append(row)
            if m == 'fv_mpc':
                table8.append(row)
        for mode in ('zero', 'delayed'):
            for other in ('small_physics', 'pino_mpc'):
                row = {'h_ms': h, 'comparison': f'Shared-{other} {mode}',
                       **contrast(values(h, other, mode), values(h, 'shared_physics', mode), idx)}
                all_contrasts.append(row)
                if other == 'small_physics':
                    table8.append(row)
            rankings.append({'h_ms': h, 'mode': mode, 'methods': sorted(METHODS, key=lambda m: values(h, m, mode).mean())})
    expected = csvread(ROOT / 'data/numerical/contrasts.csv')
    for row in all_contrasts:
        old = next(r for r in expected if float(r['h_ms']) == row['h_ms'] and r['comparison'] == row['comparison'])
        for k in ('mean_difference', 'ratio_of_means_pct', 'difference_ci_low', 'difference_ci_high', 'ratio_ci_low', 'ratio_ci_high'):
            assert np.isclose(row[k], float(old[k]), atol=1e-12, rtol=1e-10), (k, row)
    for method in METHODS:
        for mode in ('zero', 'delayed'):
            assert np.allclose(values(10., method, mode), primary[mode, method], atol=1e-10, rtol=1e-8)
    save(out, 'table8_numerical_sensitivity.csv', table8)
    save(out, 'all_numerical_contrasts.csv', all_contrasts)

    # Development comparisons are recomputed within their original validation population.
    development = {'A': [], 'B': []}
    for step in (250, 1000):
        rows = read(ROOT / f'data/development/A/loops_{step}.json')
        training = read(ROOT / f'data/development/A/training_{step}.json')
        development['A'].append({'steps': step, 'episodes': len(rows),
            'mean_cost': float(np.mean([r['episode_cost'] for r in rows])),
            'normal_mean_cost': float(np.mean([r['episode_cost'] for r in rows if r['stratum'] == 'normal'])),
            'candidate_regret': training['selected']['mean_regret']})
    for path in sorted((ROOT / 'data/development/B').glob('*/selection.json')):
        s = read(path)
        rs = []
        for c in s['candidates']:
            rows = [read(f) for f in sorted((path.parent / f"03_selection/candidate_{c['round']:02d}").glob('*_l2_*.json'))]
            assert len(rows) == 40 and len({r['episode'] for r in rows}) == 8
            rs.append({'round': c['round'], 'regret': c['offline_validation']['mean_regret'],
                       'normal_cost': float(np.mean([r['episode_cost'] for r in rows if r['stratum'] == 'normal']))})
        low = min(rs, key=lambda r: r['regret'])
        chosen = next(r for r in rs if r['round'] == s['selected_round'])
        development['B'].append({'method': path.parent.name, 'minimum_regret_round': low['round'],
            'selected_round': chosen['round'], 'same_validation_normal_cost_reduction_pct': 100*(1-chosen['normal_cost']/low['normal_cost']),
            'selected_regret_change_pct': 100*(chosen['regret']/low['regret']-1)})
    dump(out / 'development_comparisons.json', development)
    adverse = next(r for r in csvread(ROOT / 'data/numerical/episode_sensitivity.csv')
                   if r['method'] == 'shared_physics' and r['mode'] == 'delayed' and r['episode'] == 'l2_240094'
                   and float(r['coarse_ms']) == 5 and float(r['fine_ms']) == 2.5)
    assert np.isclose(float(adverse['relative_cost_pct']), 3.8920167, atol=1e-7)
    result = {'passed': True, 'primary_independent_normal_episodes': 24,
        'numerical_executions': 741, 'extended_cases': 732, 'prior_C_cases': 9,
        'conditional_bootstrap_resamples': 20000, 'bootstrap_seed': 20261007,
        'verified_numerical_contrasts': len(all_contrasts), 'baseline_episode_costs_match': True,
        'rankings': rankings, 'adverse_individual_episode_relative_pct': float(adverse['relative_cost_pct']),
        'simulation_training_or_inference_executed': False,
        'raw_audit_required_for': ['Latency journal reaggregation', 'Quadrature cost reconstruction', 'Field-difference reconstruction']}
    dump(out / 'reproduction_checks.json', result)
    if not args.skip_plots:
        from plot_results import render
        render(ROOT, out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
