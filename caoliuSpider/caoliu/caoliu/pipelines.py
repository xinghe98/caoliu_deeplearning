from itemadapter import ItemAdapter
from scrapy.pipelines.images import ImagesPipeline
from scrapy import Request
from scrapy.exceptions import DropItem
import csv
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from PIL import Image
from twisted.internet.threads import deferToThread

from .content_keys import canonical_key, extract_info_hash_from_magnet, normalize_info_hash


class CaoliuIndexPipeline:
    """Assign stable content keys and maintain a local CSV audit log."""

    def __init__(self, download_dir):
        self.download_dir = download_dir
        self.csv_file = None
        self.csv_writer = None

    @classmethod
    def from_crawler(cls, crawler):
        download_dir = crawler.settings.get('CAOLIU_DOWNLOAD_DIR', './downloads')
        return cls(download_dir)

    def open_spider(self, spider):
        os.makedirs(self.download_dir, exist_ok=True)
        csv_path = os.path.join(self.download_dir, 'index.csv')
        file_exists = os.path.exists(csv_path)
        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8-sig')
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow([
                'content_key', 'video_id', 'title', 'download_link', 'source_url', 'info_hash',
            ])

    def process_item(self, item, spider):
        source_url = item.get('source_url') or item.get('url') or ''
        magnet = item.get('magnet_uri') or item.get('download_link') or ''
        info_hash = normalize_info_hash(item.get('info_hash') or '') or extract_info_hash_from_magnet(magnet)
        try:
            content_key = item.get('content_key') or canonical_key(source_url, info_hash, magnet)
        except ValueError as exc:
            raise DropItem(str(exc)) from exc
        item['content_key'] = content_key
        item['info_hash'] = info_hash
        item['magnet_uri'] = magnet
        item['download_link'] = magnet
        item['source_url'] = source_url
        item['title_raw'] = item.get('title_raw') or item.get('title') or ''
        item['title_clean'] = item.get('title_clean') or item.get('title') or ''
        # Local folder name remains content_key-safe for image storage.
        folder_id = content_key.replace(':', '_')
        item['video_id'] = folder_id
        self.csv_writer.writerow([
            content_key,
            folder_id,
            item.get('title_clean', ''),
            magnet,
            source_url,
            info_hash,
        ])
        self.csv_file.flush()
        spider.logger.info('分配 content_key: %s -> %s', content_key, item.get('title_clean', '')[:30])
        return item

    def close_spider(self, spider):
        if self.csv_file:
            self.csv_file.close()


class CaoliuImagesPipeline(ImagesPipeline):
    """Download images into content_key folders and validate quality."""

    def get_media_requests(self, item, info):
        image_urls = item.get('image_urls', []) or []
        video_id = item.get('video_id', 'unknown')
        for idx, image_url in enumerate(image_urls[:5]):
            yield Request(
                url=image_url,
                meta={'video_id': video_id, 'image_index': idx + 1},
                dont_filter=True,
            )

    def file_path(self, request, response=None, info=None, *, item=None):
        video_id = request.meta.get('video_id', 'unknown')
        image_index = request.meta.get('image_index', 1)
        url = request.url
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            ext = 'jpg'
        return f'{video_id}/image_{image_index:02d}.{ext}'

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        store = Path(self.store.basedir)
        abs_paths = []
        seen_hash = set()
        for relative in image_paths:
            path = store / relative
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                if width < 224 or height < 224:
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hash:
                    continue
                seen_hash.add(digest)
                abs_paths.append(str(path.resolve()))
            except Exception:
                continue
            if len(abs_paths) >= 5:
                break
        item['images'] = abs_paths
        if not abs_paths:
            raise DropItem(f"无有效图片: {item.get('content_key')}")
        return item


class PlatformIngestPipeline:
    """POST validated items to the preference platform after images are ready."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        download_dir: str,
        required: bool,
        retries: int,
        timeout: float,
    ):
        self.api_url = (api_url or '').rstrip('/')
        self.api_key = api_key or ''
        self.download_dir = Path(download_dir)
        self.required = required
        self.retries = max(0, retries)
        self.timeout = timeout
        self.audit_file = None
        self.audit_lock = threading.Lock()

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            api_url=crawler.settings.get('PLATFORM_API_URL') or os.environ.get('PLATFORM_API_URL', ''),
            api_key=crawler.settings.get('INGEST_API_KEY') or os.environ.get('INGEST_API_KEY', ''),
            download_dir=crawler.settings.get('CAOLIU_DOWNLOAD_DIR', './downloads'),
            required=crawler.settings.getbool('PLATFORM_INGEST_REQUIRED', False),
            retries=crawler.settings.getint('PLATFORM_INGEST_RETRIES', 3),
            timeout=crawler.settings.getfloat('PLATFORM_INGEST_TIMEOUT', 30),
        )

    def open_spider(self, spider):
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = (self.download_dir / 'platform_ingest.jsonl').open('a', encoding='utf-8')
        if not self.api_url:
            message = 'PLATFORM_API_URL 未配置，爬虫内容不会进入偏好平台'
            if self.required:
                raise RuntimeError(message)
            spider.logger.warning(message)
        elif not self.api_key:
            spider.logger.warning('INGEST_API_KEY 未配置，将仅在平台未启用入库鉴权时成功')

    def close_spider(self, spider):
        if self.audit_file:
            self.audit_file.close()

    def _audit(self, status: str, item, **details):
        if not self.audit_file:
            return
        record = {
            'status': status,
            'content_key': item.get('content_key'),
            'source_url': item.get('source_url'),
            'title': item.get('title_clean') or item.get('title'),
            'timestamp': time.time(),
            **details,
        }
        with self.audit_lock:
            self.audit_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.audit_file.flush()

    def _payload(self, item):
        media = [
            {'source_path': path, 'ordinal': ordinal}
            for ordinal, path in enumerate(item.get('images') or [], start=1)
        ]
        if not media:
            raise DropItem(f"平台入库缺少有效图片: {item.get('content_key')}")
        return {
            'content_key': item.get('content_key'),
            'source': 'crawler',
            'source_url': item.get('source_url') or '',
            'title_raw': item.get('title_raw') or '',
            'title_clean': item.get('title_clean') or '',
            'magnet_uri': item.get('magnet_uri') or item.get('download_link') or '',
            'info_hash': item.get('info_hash') or '',
            'media': media,
        }

    def _submit(self, payload):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f'{self.api_url}/api/v1/ingest/content',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'X-Ingest-Key': self.api_key,
            },
            method='POST',
        )
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode('utf-8', errors='ignore')
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(f'平台返回 HTTP {exc.code}: {detail}') from exc
                last_error = RuntimeError(f'平台返回 HTTP {exc.code}: {detail}')
            except Exception as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f'平台入库重试耗尽: {last_error}') from last_error

    def process_item(self, item, spider):
        if not self.api_url:
            self._audit('skipped', item, reason='PLATFORM_API_URL 未配置')
            return item
        payload = self._payload(item)
        deferred = deferToThread(self._submit, payload)

        def on_success(response):
            self._audit('succeeded', item, response=response)
            spider.logger.info(
                '平台入库成功 %s, created=%s, duplicate=%s, job=%s',
                item.get('content_key'),
                response.get('created'),
                response.get('duplicate'),
                response.get('prediction_job_id'),
            )
            return item

        def on_failure(failure):
            error = str(failure.value)
            self._audit('failed', item, error=error)
            spider.logger.error('平台入库失败 %s: %s', item.get('content_key'), error)
            if self.required:
                return failure
            return item

        return deferred.addCallbacks(on_success, on_failure)


class CaoliuPipeline:
    def process_item(self, item, spider):
        return item
