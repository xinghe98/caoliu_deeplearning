"""
视频吸引力预测 API 服务
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
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


def get_predictor():
    global predictor
    if predictor is None:
        model_path = os.environ.get("MODEL_PATH", "best_model.pth")
        predictor = Predictor(model_path)
    return predictor


class PredictRequest(BaseModel):
    folder_path: str
    title: str = ""


class PredictResponse(BaseModel):
    probability: float
    probability_good: float
    probability_bad: float
    prediction: int
    label: str
    confidence: float


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[dict]


@app.on_event("startup")
async def startup():
    get_predictor()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """单个预测"""
    try:
        result = get_predictor().predict(request.folder_path, request.title)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """批量预测"""
    video_list = [(item.folder_path, item.title) for item in request.items]
    results = get_predictor().predict_batch(video_list, show_progress=False)
    return {"results": results}


@app.post("/predict/upload")
async def predict_upload(
    title: str = Form(""),
    files: List[UploadFile] = File(...)
):
    """上传图片预测"""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, file in enumerate(files):
            ext = os.path.splitext(file.filename)[1] or ".jpg"
            path = os.path.join(tmpdir, f"image_{i}{ext}")
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        
        try:
            result = get_predictor().predict(tmpdir, title)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class UrlPredictRequest(BaseModel):
    image_urls: list[str]  # 支持多张图片URL
    title: str = ""


@app.post("/predict/url", response_model=PredictResponse)
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                for i, url in enumerate(request.image_urls):
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        
                        content_type = response.headers.get("content-type", "")
                        if "jpeg" in content_type or "jpg" in content_type:
                            ext = ".jpg"
                        elif "png" in content_type:
                            ext = ".png"
                        elif "gif" in content_type:
                            ext = ".gif"
                        elif "webp" in content_type:
                            ext = ".webp"
                        else:
                            ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                        
                        image_path = os.path.join(tmpdir, f"image_{i}{ext}")
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                    except Exception as e:
                        print(f"下载图片失败 {url}: {e}")
            
            result = get_predictor().predict(tmpdir, request.title)
            return result
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"下载图片失败: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
