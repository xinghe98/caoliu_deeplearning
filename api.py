"""
视频吸引力预测 API 服务
"""

import asyncio
import ipaddress
import os
import secrets
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import httpx

from predict import Predictor

app = FastAPI(title="视频吸引力预测 API", version="1.0.0")

# CORS 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局预测器
predictor = None

MAX_BATCH_ITEMS = 100
MAX_UPLOAD_FILES = 32
MAX_URLS = 32
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_TOTAL_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 5
CHUNK_SIZE = 64 * 1024
IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def get_predictor():
    global predictor
    if predictor is None:
        model_path = os.environ.get("MODEL_PATH", "best_model.pth")
        predictor = Predictor(model_path)
    return predictor


async def require_api_key(x_api_key: Optional[str] = Header(None)):
    if os.environ.get("ALLOW_UNAUTHENTICATED_API", "").lower() in {"1", "true", "yes", "on"}:
        return
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Prediction API key is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _predict_folder(folder_path, title):
    return get_predictor().predict(folder_path, title)


def _predict_batch(video_list):
    return get_predictor().predict_batch(video_list, show_progress=False)


def _address_is_blocked(address):
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


async def _validate_public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Only public HTTP(S) image URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=400, detail="URL credentials are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname, port, type=socket.SOCK_STREAM
        )
        if not addresses or any(_address_is_blocked(item[4][0]) for item in addresses):
            raise HTTPException(status_code=400, detail="URL resolves to a non-public address")
    except HTTPException:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail="URL host could not be resolved") from exc
    return list(dict.fromkeys(item[4][0].split("%", 1)[0] for item in addresses))


def _pinned_request(url, address):
    parsed = urlsplit(url)
    host = f"[{address}]" if ":" in address else address
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    pinned_url = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))
    host_header = parsed.hostname
    default_port = 443 if parsed.scheme == "https" else 80
    if parsed.port is not None and parsed.port != default_port:
        host_header = f"{host_header}:{parsed.port}"
    return pinned_url, host_header, parsed.hostname.encode("idna")


async def _download_image(client, url, path, byte_limit):
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        addresses = await _validate_public_url(current_url)
        pinned_url, host_header, sni_hostname = _pinned_request(current_url, addresses[0])
        async with client.stream(
            "GET",
            pinned_url,
            headers={"Host": host_header},
            extensions={"sni_hostname": sni_hostname},
            follow_redirects=False,
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location or redirect_count == MAX_REDIRECTS:
                    raise HTTPException(status_code=400, detail="Invalid or excessive URL redirects")
                current_url = urljoin(current_url, location)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise HTTPException(status_code=400, detail="Image download failed") from exc
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            extension = IMAGE_CONTENT_TYPES.get(content_type)
            if extension is None:
                raise HTTPException(status_code=415, detail="URL did not return a supported image")
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > min(MAX_DOWNLOAD_BYTES, byte_limit):
                        raise HTTPException(status_code=413, detail="Downloaded image is too large")
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
            downloaded = 0
            with open(f"{path}{extension}", "wb") as output:
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES or downloaded > byte_limit:
                        raise HTTPException(status_code=413, detail="Downloaded image is too large")
                    output.write(chunk)
            return downloaded
    raise HTTPException(status_code=400, detail="Invalid URL redirect")


class PredictRequest(BaseModel):
    folder_path: str = Field(min_length=1, max_length=4096)
    title: str = Field(default="", max_length=512)


class PredictResponse(BaseModel):
    probability: float
    probability_good: float
    probability_bad: float
    prediction: int
    label: str
    confidence: float
    decision_threshold: float
    model_version: int


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class BatchPredictResponse(BaseModel):
    results: List[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(request: PredictRequest):
    """单个预测"""
    try:
        result = await anyio.to_thread.run_sync(_predict_folder, request.folder_path, request.title)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(require_api_key)])
async def predict_batch(request: BatchPredictRequest):
    """批量预测"""
    video_list = [(item.folder_path, item.title) for item in request.items]
    results = await anyio.to_thread.run_sync(_predict_batch, video_list)
    return {"results": results}


@app.post("/predict/upload", dependencies=[Depends(require_api_key)])
async def predict_upload(
    title: str = Form("", max_length=512),
    files: List[UploadFile] = File(...)
):
    """上传图片预测"""
    if not files or len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail="Too many uploaded files")
    with tempfile.TemporaryDirectory() as tmpdir:
        total_size = 0
        for i, file in enumerate(files):
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                raise HTTPException(status_code=415, detail="Unsupported image file type")
            path = os.path.join(tmpdir, f"image_{i}{ext}")
            file_size = 0
            with open(path, "wb") as f:
                while chunk := await file.read(CHUNK_SIZE):
                    file_size += len(chunk)
                    total_size += len(chunk)
                    if file_size > MAX_UPLOAD_BYTES or total_size > MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Uploaded image data is too large")
                    f.write(chunk)
        
        try:
            result = await anyio.to_thread.run_sync(_predict_folder, tmpdir, title)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class UrlPredictRequest(BaseModel):
    image_urls: list[str] = Field(min_length=1, max_length=MAX_URLS)
    title: str = Field(default="", max_length=512)


@app.post("/predict/url", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict_url(request: UrlPredictRequest):
    """通过图片URL预测
    
    参数:
        image_urls: 图片URL地址列表
        title: 视频标题
    
    返回:
        prediction: 预测结果 (0=不好看, 1=好看)
        label: 结果标签 ("好看" 或 "不好看")
        probability: 好看概率
        probability_good: 好看概率
        probability_bad: 不好看概率
        confidence: 置信度
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Pinned IPs must not share a pooled TLS connection across hostnames.
            limits = httpx.Limits(max_connections=5, max_keepalive_connections=0)
            async with httpx.AsyncClient(
                timeout=30.0, limits=limits, follow_redirects=False, trust_env=False
            ) as client:
                total_downloaded = 0
                for i, url in enumerate(request.image_urls):
                    remaining = MAX_TOTAL_DOWNLOAD_BYTES - total_downloaded
                    total_downloaded += await _download_image(
                        client, url, os.path.join(tmpdir, f"image_{i}"), remaining
                    )
            
            result = await anyio.to_thread.run_sync(_predict_folder, tmpdir, request.title)
            return result
        except HTTPException:
            raise
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"下载图片失败: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
