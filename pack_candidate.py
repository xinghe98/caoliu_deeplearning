"""Pack GPU training outputs into a platform candidate ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REQUIRED = ('best_model.pth', 'evaluation_report.json')
OPTIONAL = (
    'training_history.json',
    'training_history.png',
    'validation_predictions.csv',
    'validation_error_cases.csv',
    'external_test_predictions.csv',
    'external_test_error_cases.csv',
    'split_manifest.csv',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pack_candidate(source_dir: Path, output_path: Path, run_id: str | None = None) -> Path:
    source_dir = source_dir.resolve()
    for name in REQUIRED:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(f'缺少必需文件: {name}')
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or output_path.stem.replace('candidate_', '')
    checksums: dict[str, str] = {}
    existing_manifest = source_dir / 'model_manifest.json'
    manifest = {}
    if existing_manifest.is_file():
        loaded = json.loads(existing_manifest.read_text(encoding='utf-8'))
        if not isinstance(loaded, dict):
            raise ValueError('model_manifest.json 必须是 JSON 对象')
        manifest.update(loaded)
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED + OPTIONAL:
            path = source_dir / name
            if not path.is_file():
                continue
            archive.write(path, name)
            checksums[name] = sha256_file(path)
        manifest.update({
            'version': run_id,
            'run_id': run_id,
        })
        report = json.loads((source_dir / 'evaluation_report.json').read_text(encoding='utf-8'))
        if isinstance(report, dict):
            manifest['data_manifest_hash'] = report.get('data_manifest_hash', '')
            metrics = report.get('validation') or {}
            if isinstance(metrics, dict):
                manifest['decision_threshold'] = metrics.get('threshold')
                manifest['temperature'] = metrics.get('temperature')
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        archive.writestr('model_manifest.json', manifest_bytes)
        checksums['model_manifest.json'] = hashlib.sha256(manifest_bytes).hexdigest()
        archive.writestr('SHA256SUMS.json', json.dumps(checksums, indent=2, ensure_ascii=False))
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='打包候选模型 ZIP')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args(argv)
    path = pack_candidate(args.source_dir, args.output, args.run_id)
    print(f'已生成候选包: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
