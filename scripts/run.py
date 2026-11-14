#!/usr/bin/env python3
"""Run the experiment selected in configs/local.json without command-line options."""
import json
from pathlib import Path
import sys

import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/local.json'


def object_keys(value, allowed, location):
    if not isinstance(value, dict):
        raise ValueError(f'{location} must be a JSON object')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown key in {location}: {', '.join(sorted(unknown))}")


def configuration_arguments(config, root=ROOT):
    """Resolve local paths before handing a validated argument list to the runner."""
    object_keys(config, {'dataset', 'stage', 'variant', 'profile', 'paths', 'build',
                         'search', 'evaluation'}, 'local configuration')
    argv = []
    choices = {
        'dataset': ('sift1m', 'gist1m', 'arxiv2m', 'paper', 'demo'),
        'stage': ('build', 'groundtruth', 'search', 'all'),
        'variant': tuple(run_benchmark.VARIANTS),
    }
    defaults = {'dataset': 'sift1m', 'stage': 'search', 'variant': 'full'}
    for key, allowed in choices.items():
        value = config.get(key, defaults[key])
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{key} must be one of: {', '.join(allowed)}")
        argv.extend(['--' + key, value])

    def path(value, location):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{location} must be a nonempty path string')
        resolved = Path(value).expanduser()
        return str((resolved if resolved.is_absolute() else root / resolved).resolve())

    argv.extend(['--config', path(config.get('profile', 'configs/paper.json'), 'profile')])
    paths = config.get('paths', {})
    path_options = {'build': ('build-dir', 'build'), 'data': ('data-root', 'dataset'),
                    'index': ('index-root', 'artifacts'), 'output': ('output-dir', 'results')}
    object_keys(paths, path_options, 'paths')
    for key, (option, default) in path_options.items():
        argv.extend(['--' + option, path(paths.get(key, default), 'paths.' + key)])

    sections = {
        'build': {'threads': ('build-threads', 1)},
        'search': {'topk': ('topk', 1), 'threads': ('threads', 1),
                   'ef_match': ('ef-match', 1), 'visit_budget': ('visit-budget', 1),
                   'expand_budget': ('expand-budget', 1), 'bridge_budget': ('bridge-budget', 0)},
        'evaluation': {'warmup': ('warmup', 0), 'repeats': ('repeats', 1),
                       'max_queries': ('max-queries', 1)},
    }
    for section, options in sections.items():
        values = config.get(section, {})
        allowed = set(options)
        if section == 'search':
            allowed.update(('force_plan', 'force_driver', 'calibrate_costs'))
        object_keys(values, allowed, section)
        for key, (option, minimum) in options.items():
            if key not in values:
                continue
            value = values[key]
            if section == 'evaluation' and key == 'max_queries' and value is None:
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f'{section}.{key} must be an integer >= {minimum}')
            argv.extend(['--' + option, str(value)])
    search = config.get('search', {})
    if 'force_plan' in search:
        value = search['force_plan']
        if not isinstance(value, str) or value not in ('auto', 'exact_meta', 'graph_bridge'):
            raise ValueError('search.force_plan must be auto, exact_meta, or graph_bridge')
        argv.extend(['--force-plan', value])
    driver = search.get('force_driver')
    if driver is not None:
        if not isinstance(driver, str) or not driver.strip():
            raise ValueError('search.force_driver must be a nonempty string or null')
        argv.extend(['--force-driver', driver])
    calibrate = search.get('calibrate_costs', False)
    if type(calibrate) is not bool:
        raise ValueError('search.calibrate_costs must be true or false')
    if calibrate:
        argv.append('--calibrate-costs')
    return argv


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError(f'This launcher takes no arguments. Edit {CONFIG} instead.')
        config = json.loads(CONFIG.read_text(encoding='utf8'))
        argv = configuration_arguments(config)
        print(f"Configuration: {CONFIG}", flush=True)
        print('Run: dataset={}, stage={}, variant={}'.format(
            config.get('dataset', 'sift1m'), config.get('stage', 'search'),
            config.get('variant', 'full')), flush=True)
        run_benchmark.main(argv)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'PAIRS: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
