"""Safe import and validation for model packages returned by a GPU server."""

import hashlib
import json
import shutil
import stat
import uuid
import zipfile
from pathlib import Path

import torch
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from .config import get_settings
from .models import ModelVersion


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, target: Path, max_files: int, max_uncompressed: int) -> None:
    members = archive.infolist()
    if len(members) > max_files:
        raise HTTPException(status_code=422, detail='候选包文件数量超过限制')
    if len({member.filename for member in members}) != len(members):
        raise HTTPException(status_code=422, detail='候选包包含重复文件名')
    total = 0
    for member in members:
        destination = (target / member.filename).resolve()
        if not member.filename or Path(member.filename).is_absolute() or not destination.is_relative_to(target.resolve()):
            raise HTTPException(status_code=422, detail='候选包包含非法路径')
        if stat.S_ISLNK(member.external_attr >> 16):
            raise HTTPException(status_code=422, detail='候选包不能包含符号链接')
        total += member.file_size
        if total > max_uncompressed:
            raise HTTPException(status_code=422, detail='候选包解压后体积超过限制')
    archive.extractall(target)


def _verify_sha256sums(target: Path) -> None:
    sums_path = target / 'SHA256SUMS.json'
    if not sums_path.is_file():
        return
    try:
        expected = json.loads(sums_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=422, detail='SHA256SUMS.json 无法解析') from exc
    for relative, digest in expected.items():
        path = target / relative
        if not path.is_file():
            raise HTTPException(status_code=422, detail=f'SHA256SUMS 缺少文件: {relative}')
        if sha256_file(path) != digest:
            raise HTTPException(status_code=422, detail=f'SHA256 不匹配: {relative}')


async def import_candidate(session: Session, upload: UploadFile, version: str | None) -> ModelVersion:
    if not upload.filename or not upload.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=422, detail='候选模型必须使用 ZIP 包上传')
    settings = get_settings()
    root = settings.platform_data_dir / 'candidates'
    root.mkdir(parents=True, exist_ok=True)
    import_id = str(uuid.uuid4())
    archive_path = root / f'{import_id}.zip'
    max_bytes = settings.candidate_max_upload_mb * 1024 * 1024
    written = 0
    try:
        with archive_path.open('wb') as output:
            while block := await upload.read(1024 * 1024):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail='候选包超过上传大小限制')
                output.write(block)
        target = root / import_id
        target.mkdir()
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    imported = False
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, target, settings.candidate_max_files, settings.candidate_max_uncompressed_mb * 1024 * 1024)
        _verify_sha256sums(target)
        checkpoint = target / 'best_model.pth'
        report_path = target / 'evaluation_report.json'
        if not checkpoint.is_file() or not report_path.is_file():
            raise HTTPException(status_code=422, detail='候选包必须包含 best_model.pth 和 evaluation_report.json')
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint_data, dict) or 'model_state_dict' not in checkpoint_data:
            raise HTTPException(status_code=422, detail='checkpoint 缺少 model_state_dict')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        manifest_path = target / 'model_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        if not isinstance(manifest, dict):
            raise HTTPException(status_code=422, detail='model_manifest.json 必须是对象')
        resolved_version = version or manifest.get('version') or f'candidate-{import_id[:8]}'
        if session.query(ModelVersion).filter_by(version=resolved_version).first():
            raise HTTPException(status_code=409, detail='模型版本已存在')
        metrics = report.get('external_test') or report.get('validation') or report
        candidate = ModelVersion(
            version=resolved_version, status='candidate', checkpoint_path=str(checkpoint),
            decision_threshold=float(checkpoint_data.get('decision_threshold', manifest.get('decision_threshold', 0.5))),
            temperature=float(checkpoint_data.get('temperature', manifest.get('temperature', 1.0))),
            metrics=metrics if isinstance(metrics, dict) else {'value': metrics},
            data_manifest_hash=str(checkpoint_data.get('data_manifest_hash', manifest.get('data_manifest_hash', ''))),
        )
        session.add(candidate)
        session.commit()
        imported = True
        return candidate
    except zipfile.BadZipFile as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail='候选包不是有效 ZIP') from exc
    except HTTPException:
        session.rollback()
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=f'候选包校验失败: {exc}') from exc
    finally:
        archive_path.unlink(missing_ok=True)
        if not imported:
            shutil.rmtree(target, ignore_errors=True)
