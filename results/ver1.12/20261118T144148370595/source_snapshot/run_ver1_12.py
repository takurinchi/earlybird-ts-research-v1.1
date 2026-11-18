"""Run the unchanged paired design with 80% channel retention, then archive it."""
import json
import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from utils.config import load_config
from cifar_compaction_experiment import run_compaction_suite

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / '.cache' / 'ver1_12_status.json'


def status(stage, **details):
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text(json.dumps(dict(stage=stage, updated_at=datetime.now().isoformat(),
                                     **details), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-baseline')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config('configs/ts_compaction_ver1_12.yaml')
    old = load_config('configs/ts_compaction_masktrack.yaml')
    expected = load_config('configs/ts_compaction_masktrack.yaml')
    expected['pruning']['selected_ratio'] = 0.2
    expected['experiment']['output_root'] = 'outputs/ver1.12'
    if config != expected:
        raise ValueError('Unexpected configuration difference from Ver.1.1')
    # Preserve source and partial results from interrupted attempts.
    attempt_id = datetime.now().strftime('%Y%m%dT%H%M%S%f')
    source = ROOT / 'outputs' / 'ver1.12' / 'source_snapshots' / attempt_id
    source.mkdir(parents=True, exist_ok=False)
    for folder in ('configs', 'models', 'training', 'pruning', 'evaluation',
                   'earlybird', 'utils', 'analysis', 'docs'):
        shutil.copytree(ROOT / folder, source / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    (source / 'data').mkdir()
    for path in (ROOT / 'data').glob('*.py'):
        shutil.copy2(path, source / 'data' / path.name)
    shutil.copy2(ROOT / 'requirements.txt', source / 'requirements.txt')
    provenance = dict(version='1.12', retained_channel_fraction_target=0.8,
                      resume_baseline=args.resume_baseline,
                      pruning_ratio=0.2, previous_pruning_ratio=old['pruning']['selected_ratio'],
                      config_changes=['pruning.selected_ratio', 'experiment.output_root'],
                      note='Same detection rule; t_m is detected again. Global threshold ties may reduce retention below 80%.')
    (source / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    status('training', pid=os.getpid(), output_root=config['experiment']['output_root'],
           source_snapshot=str(source))
    result = run_compaction_suite(config, resume_baseline=args.resume_baseline)
    complete = (len(result['conditions']) == 16 and
                all(s.get('status') == 'completed' for s in result['conditions']))
    repo = ROOT.parent / 'earlybird-ts-research-v1.1-results'
    relative = Path('results') / 'ver1.12' / result['run_id']
    destination = repo / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(result['bundle'], destination)
    shutil.copytree(source, destination / 'source_snapshot')
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    with (destination / 'README.md').open('a', encoding='utf-8') as f:
        f.write('\n\nVersion: 1.12. Target retained BN channels: 80% (pruning ratio 0.2).\n'
                'Training and measurement settings inherit Ver.1.1. See source_snapshot and provenance.json.\n')
    status('results_saved', all_16_completed=complete, bundle=result['bundle'], repository_folder=str(destination))
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    def git(*args):
        subprocess.run(['git', '-C', str(repo), *args], check=True, env=env)
    git('add', '--', relative.as_posix())
    # Commit only this experiment, preserving any unrelated staged changes.
    git('-c', 'user.name=Codex', '-c', 'user.email=codex@localhost',
        'commit', '-m', f'Record Ver.1.12 retain-80-percent experiment {result["run_id"]}',
        '--', relative.as_posix())
    status('committed', all_16_completed=complete, repository_folder=str(destination))
    try:
        git('push', 'origin', 'HEAD')
    except subprocess.CalledProcessError:
        status('committed_push_failed', all_16_completed=complete, repository_folder=str(destination))
        raise
    status('pushed', all_16_completed=complete, repository_folder=str(destination))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        previous = json.loads(STATUS.read_text(encoding='utf-8-sig')) if STATUS.exists() else {}
        if previous.get('stage') != 'committed_push_failed':
            status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   pid=os.getpid(), error=repr(exc), previous_status=previous)
        raise
