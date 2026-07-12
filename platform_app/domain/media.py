import hashlib
from pathlib import Path

from PIL import Image

ALLOWED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MIN_IMAGE_SIDE = 224


def is_under_roots(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f'图片不存在: {path}')
    if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f'不支持的图片扩展名: {path.suffix}')
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f'图片为空文件: {path}')
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
            raise ValueError(f'图片尺寸过小 ({width}x{height}): {path}')
        ratio = max(width, height) / max(1, min(width, height))
        if ratio > 20:
            raise ValueError(f'图片长宽比异常: {path}')
        mime = Image.MIME.get(image.format or '', 'image/jpeg')
    return {
        'width': width,
        'height': height,
        'file_size': size,
        'sha256': file_sha256(path),
        'mime_type': mime,
    }
