"""One entry point. Checks never modify the frozen evidence in this repository."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / 'pipeline'


def execute(args, cwd):
    print('RUN', *args, flush=True)
    env = os.environ.copy()
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run([sys.executable, *args], cwd=cwd, env=env, check=True)


def checksums():
    manifest = ROOT / 'SHA256SUMS'
    if not manifest.exists():
        raise FileNotFoundError('SHA256SUMS is required; use a complete release.')
    count = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f'Invalid manifest entry: {name}')
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        count += 1
    print(f'PASS: {count} file checksums', flush=True)


def check():
    checksums()
    execute(['-m', 'unittest', 'discover', '-s', 'research/v4', '-p', 'test_*.py', '-v'], ROOT)
    execute(['research/v4/verify.py'], ROOT)
    execute(['-m', 'unittest', 'discover', '-s', 'research/v5', '-p', 'test_*.py', '-v'], ROOT)
    execute(['research/v5/verify.py'], ROOT)
    execute(['-m', 'unittest', 'discover', '-s', 'research/v6', '-p', 'test_*.py', '-v'], ROOT)
    execute(['research/v6/verify.py'], ROOT)
    with tempfile.TemporaryDirectory(prefix='sberindex-check-') as scratch:
        work = Path(scratch) / 'pipeline'
        shutil.copytree(PIPELINE, work, ignore=shutil.ignore_patterns('__pycache__'))
        for args in [
            ['-m', 'unittest', 'discover', '-p', 'test_*.py', '-v'],
            ['validate_final.py'], ['validate_upgrade.py'], ['analyze_v3.py'],
        ]:
            execute(args, work)
    print('PASS: frozen results and source signatures verified; no training performed.')


def reproduce(output):
    """Recompute in a NEW directory; never overwrite included evidence."""
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('Choose a non-existent output directory outside this repository.')
    output.mkdir(parents=True, exist_ok=False)
    for item in PIPELINE.iterdir():
        if item.is_file() and item.suffix in {'.py', '.json', '.md', '.cjs'}:
            shutil.copy2(item, output / item.name)
    for name in ['data', 'results']:
        shutil.copytree(PIPELINE / name, output / name)
    for name in ['final_results', 'upgrade_results', 'v3_results', 'figures']:
        (output / name).mkdir()
    commands = [
        ['experiment.py', '--phase', 'forecast'],
        ['experiment.py', '--phase', 'detector'], ['analysis_final.py'],
        ['release_candidate.py', '--phase', 'all'], ['confirmation_analysis.py'],
        ['validate_final.py'], ['upgrade_v2.py', '--phase', 'forecast'],
        ['upgrade_v2.py', '--phase', 'detector'], ['analyze_upgrade.py'],
        ['validate_upgrade.py'], ['upgrade_v3.py', '--phase', 'forecast'],
        ['upgrade_v3.py', '--phase', 'detector'], ['analyze_v3.py'],
        ['build_local_cases.py'],
    ]
    for args in commands:
        execute(args, output)
    print(f'Recomputed outputs: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check', help='Verify saved results without training or network')
    r = sub.add_parser('reproduce', help='Full CPU computation; may take hours and download weights')
    r.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('v5', help='Recompute V5 from frozen V4 evidence, without network')
    v.add_argument('--output', type=Path, required=True)
    v.add_argument('--config', type=Path, help='V5 JSON hyperparameters; defaults to research/v5/config.json')
    sub.add_parser('report', help='Render REPORT.md to REPORT.pdf')
    args = parser.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'reproduce':
        reproduce(args.output)
    elif args.command == 'v5':
        command=['research/v5/run.py', '--output', str(args.output.resolve())]
        if args.config: command.extend(['--config',str(args.config.resolve())])
        execute(command, ROOT)
    else:
        execute(['tools/build_report.py'], ROOT)


if __name__ == '__main__':
    main()
