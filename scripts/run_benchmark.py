#!/usr/bin/env python3
"""Run PAIRS build, ground-truth, and search stages on a prepared dataset."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'full': {},
    'numeric-only': {'driver-policy': 'numeric-only'},
    'smallest-domain': {'driver-policy': 'smallest-domain'},
    'no-bridge': {'max-bridge-budget': 0},
    'match-width-k': {},
}


def arguments(values):
    result = []
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                result.append('--' + key)
        else:
            result.extend(['--' + key, str(value)])
    return result


def parse_metrics(output):
    metrics = {}
    for key, value in re.findall(r'(\w+|Recall@\d+)=([0-9]+(?:\.[0-9]+)?)', output):
        if key in {'queries', 'QPS', 'avg_latency_ms', 'P50_ms', 'P95_ms', 'P99_ms',
                   'avg_distances', 'avg_predicate_checks', 'avg_visited',
                   'build_seconds', 'save_seconds', 'storage_gb', 'total_seconds'} or key.startswith('Recall@'):
            metrics[key] = float(value)
    return metrics


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def execute(command, log, records):
    started = time.perf_counter()
    completed = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
    log.write_text(completed.stdout, encoding='utf8')
    # Process time includes index loading and output I/O. Search QPS comes from
    # the native program's timer around the query batch.
    records.append({'argv': [str(x) for x in command], 'exit_code': completed.returncode,
                    'process_seconds': time.perf_counter() - started, 'log': log.name})
    if completed.returncode:
        raise RuntimeError(f'Command failed ({completed.returncode}), see {log}\n{completed.stdout[-3000:]}')
    return parse_metrics(completed.stdout)


def executable(build_dir, name):
    filename = name + ('.exe' if os.name == 'nt' else '')
    for folder in (build_dir, build_dir / 'bin', build_dir / 'Release', build_dir / 'Debug'):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f'Build the project first. Cannot find {filename} in {build_dir}')


def preflight(stage, data, spec, prefix, search, max_queries, build_dir):
    required = []
    programs = []
    if stage in ('build', 'all'):
        required.extend(data / spec[key] for key in ('base', 'attributes_file'))
        if spec.get('mapping'):
            required.append(data / spec['mapping'])
        programs.append('build_pairs_index')
    if stage in ('groundtruth', 'search', 'all'):
        required.extend(data / spec[key] for key in ('queries', 'filters'))
        if stage != 'all':
            required.extend(Path(str(prefix) + suffix)
                            for suffix in ('.meta', '.vectors.bin', '.attributes.bin'))
    if stage in ('groundtruth', 'all'):
        programs.append('generate_pairs_groundtruth')
    if stage in ('search', 'all'):
        programs.append('search_pairs_index')
    missing = [str(path) for path in required if not path.is_file()]
    if stage == 'search' and not (data / spec['groundtruth']).is_file():
        cached = prefix.parent / f"groundtruth_k{search['topk']}_n{max_queries or 'all'}.bin"
        if not (cached.is_file() and cached.with_suffix('.json').is_file()):
            missing.append(f"{data / spec['groundtruth']} (or generate truth with stage=groundtruth)")
    if missing:
        raise FileNotFoundError('Required data or index files are missing:\n  ' + '\n  '.join(missing))
    return {name: executable(build_dir, name) for name in programs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['sift1m', 'gist1m', 'arxiv2m', 'paper', 'demo'])
    parser.add_argument('--stage', choices=['build', 'groundtruth', 'search', 'all'], default='search')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper.json')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--index-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--force-plan', choices=['auto', 'exact_meta', 'graph_bridge'])
    parser.add_argument('--force-driver')
    for name in ['topk', 'ef-match', 'visit-budget', 'expand-budget', 'bridge-budget', 'threads',
                 'build-threads', 'repeats', 'warmup', 'max-queries']:
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--calibrate-costs', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf8'))
    spec = config['datasets'][args.dataset]
    build, search = dict(config['build']), dict(config['search'])
    for key in ['topk', 'ef-match', 'visit-budget', 'expand-budget', 'threads']:
        value = getattr(args, key.replace('-', '_'))
        if value is not None:
            if value <= 0:
                parser.error('--' + key + ' must be positive')
            search[key] = value
    if args.bridge_budget is not None:
        if args.bridge_budget < 0:
            parser.error('--bridge-budget must be nonnegative')
        search['max-bridge-budget'] = args.bridge_budget
    if args.build_threads is not None:
        if args.build_threads <= 0:
            parser.error('--build-threads must be positive')
        build['build-threads'] = args.build_threads
    search.update(VARIANTS[args.variant])
    if args.variant == 'match-width-k':
        search['ef-match'] = search['topk']
    if args.force_plan:
        search['force-plan'] = args.force_plan
    if args.force_driver:
        search['force-driver'] = args.force_driver
    if args.calibrate_costs:
        search['calibrate-costs'] = True
    repeats = config['evaluation']['repeats'] if args.repeats is None else args.repeats
    # Keep the requested value in the configuration. The search program clips it
    # to the number of queries, which may be smaller after --max-queries.
    warmup = config['evaluation']['warmup'] if args.warmup is None else args.warmup
    if repeats <= 0 or warmup < 0 or (args.max_queries is not None and args.max_queries <= 0):
        parser.error('repeats/max-queries must be positive and warmup nonnegative')
    data = args.data_root.resolve() / spec.get('directory', args.dataset)
    prefix = args.index_root.resolve() / args.dataset / 'index'
    programs = preflight(args.stage, data, spec, prefix, search, args.max_queries, args.build_dir.resolve())
    signature = hashlib.sha256(json.dumps({'search': search, 'warmup': warmup, 'repeats': repeats, 'max_queries': args.max_queries}, sort_keys=True).encode()).hexdigest()[:12]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir.resolve() / args.dataset / args.variant / signature / run_id
    output.mkdir(parents=True, exist_ok=False)
    records, rounds = [], []
    manifest = {'dataset': args.dataset, 'variant': args.variant, 'build': build, 'search': search,
                'warmup': warmup, 'repeats': repeats, 'max_queries': args.max_queries,
                'dataset_specification': spec, 'data_directory': str(data),
                'commands': records, 'rounds': rounds,
                'complete': False}

    def exe(name):
        return programs[name]

    try:
        if args.stage in ('build', 'all'):
            options = {'data': data / spec['base'], 'base-attrs': data / spec['attributes_file'],
                       'index-prefix': prefix, **build}
            if spec.get('mapping'):
                options['tag-mapping'] = data / spec['mapping']
            manifest['index_build'] = execute([exe('build_pairs_index'), *arguments(options)], output / 'build.log', records)
            manifest['index_components_bytes'] = {p.name: p.stat().st_size for p in sorted(prefix.parent.glob(prefix.name + '.*')) if p.is_file()}
            manifest['index_bytes'] = sum(manifest['index_components_bytes'].values())
        query = {'index-prefix': prefix, 'queries': data / spec['queries'], 'filters': data / spec['filters'],
                 'topk': search['topk'], 'threads': search['threads']}
        if args.max_queries:
            query['max-queries'] = args.max_queries
        generated_truth = prefix.parent / f"groundtruth_k{search['topk']}_n{args.max_queries or 'all'}.bin"
        truth_metadata = generated_truth.with_suffix('.json')
        identity = None
        if args.stage in ('groundtruth', 'search', 'all'):
            # The metadata binds the index components through their checksums.
            # Hash full query/filter files even for a prefix, so replacing the
            # workload invalidates cached truth rather than silently reusing it.
            identity = {'topk': search['topk'], 'max_queries': args.max_queries,
                        'index_meta_sha256': checksum(Path(str(prefix) + '.meta')),
                        'queries_sha256': checksum(data / spec['queries']),
                        'filters_sha256': checksum(data / spec['filters'])}
            manifest['input_identity'] = identity
        truth = data / spec['groundtruth']
        truth_format = 'ivecs'
        if args.stage in ('groundtruth', 'all'):
            truth = generated_truth
            truth_format = 'rcfs_bin'
            execute([exe('generate_pairs_groundtruth'), *arguments({**query, 'output': truth})], output / 'groundtruth.log', records)
            truth_metadata.write_text(json.dumps({'identity': identity, 'sha256': checksum(truth)}, indent=2) + '\n', encoding='utf8')
        if args.stage in ('search', 'all'):
            if not truth.is_file() and generated_truth.is_file() and truth_metadata.is_file():
                cached = json.loads(truth_metadata.read_text(encoding='utf8'))
                # Identity checks catch changed inputs. The separate truth hash
                # catches a truncated or replaced result file.
                if cached.get('identity') != identity or cached.get('sha256') != checksum(generated_truth):
                    raise ValueError('Generated ground truth is stale or corrupt. Run --stage groundtruth again.')
                truth, truth_format = generated_truth, 'rcfs_bin'
            if not truth.is_file():
                raise FileNotFoundError(f'Missing ground truth: {truth}. Download it or run --stage groundtruth.')
            # Supplied .ivecs truth has no sidecar binding. Its checksum records
            # which file was used, not whether it matches the prepared workload.
            manifest['groundtruth'] = {'path': str(truth), 'format': truth_format, 'sha256': checksum(truth)}
            for repeat in range(repeats):
                options = {**query, **search, 'warmup': warmup, 'gt': truth, 'gt-format': truth_format,
                           'output': output / f'results_{repeat}.bin', 'stats': output / f'clauses_{repeat}.csv'}
                metrics = execute([exe('search_pairs_index'), *arguments(options)], output / f'search_{repeat}.log', records)
                if 'QPS' not in metrics or f"Recall@{search['topk']}" not in metrics:
                    raise RuntimeError('Missing search metrics or ground-truth k is smaller than requested k')
                manifest['effective_warmup_queries'] = min(warmup, int(metrics['queries']))
                with (output / f'clauses_{repeat}.csv').open(newline='', encoding='utf8') as stream:
                    rows = list(csv.DictReader(stream))
                metrics['planning_ms_sum'] = sum(float(row['planning_ms']) for row in rows)
                metrics['execution_ms_sum'] = sum(float(row['execution_ms']) for row in rows)
                metrics['edge_visits'] = sum(int(row['edge_visits']) for row in rows)
                rounds.append(metrics)
            # Each percentile is already a within-run statistic. Taking its
            # median here does not pool individual query latencies across runs.
            manifest['median'] = {key: statistics.median(r[key] for r in rounds) for key in rounds[0]}
            manifest['aggregation'] = 'Median of complete runs. Latency percentiles are per-run percentiles, not pooled.'
        manifest['complete'] = True
    finally:
        (output / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print(output / 'summary.json')
    if rounds:
        print(json.dumps(manifest['median'], indent=2))


if __name__ == '__main__':
    main()
