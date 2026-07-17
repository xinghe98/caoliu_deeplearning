import asyncio
import os
import socket
import tempfile

import httpx
from fastapi.testclient import TestClient

import api


def test_health_is_public_and_predictions_fail_closed(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_API", raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        responses = [
            client.post("/predict", json={"folder_path": "images"}),
            client.post("/predict/batch", json={"items": [{"folder_path": "images"}]}),
            client.post("/predict/url", json={"image_urls": ["https://example.com/a.jpg"]}),
            client.post("/predict/upload", files={"files": ("a.jpg", b"image", "image/jpeg")}),
        ]
    assert all(response.status_code == 503 for response in responses)


def test_api_key_and_inference_thread_offload(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_API", raising=False)
    called = {}

    async def fake_run_sync(function, *args):
        called["function"] = function
        called["args"] = args
        return {
            "probability": 0.8, "probability_good": 0.8, "probability_bad": 0.2,
            "prediction": 1, "label": "good", "confidence": 0.8,
            "decision_threshold": 0.5, "model_version": 1,
        }

    monkeypatch.setattr(api.anyio.to_thread, "run_sync", fake_run_sync)
    with TestClient(api.app) as client:
        assert client.post("/predict", json={"folder_path": "images"}).status_code == 401
        response = client.post(
            "/predict", json={"folder_path": "images"}, headers={"X-API-Key": "secret"}
        )
    assert response.status_code == 200
    assert called == {"function": api._predict_folder, "args": ("images", "")}


def test_batch_and_url_counts_are_bounded(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    headers = {"X-API-Key": "secret"}
    with TestClient(api.app) as client:
        batch = client.post(
            "/predict/batch",
            json={"items": [{"folder_path": "x"}] * (api.MAX_BATCH_ITEMS + 1)},
            headers=headers,
        )
        urls = client.post(
            "/predict/url",
            json={"image_urls": ["https://example.com/a.jpg"] * (api.MAX_URLS + 1)},
            headers=headers,
        )
    assert batch.status_code == 422
    assert urls.status_code == 422


def test_url_validation_rejects_private_and_mixed_dns(monkeypatch):
    async def resolve(host, port, type):
        addresses = ["93.184.216.34"]
        if host == "private.test":
            addresses = ["127.0.0.1"]
        elif host == "mixed.test":
            addresses = ["93.184.216.34", "169.254.169.254"]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(loop, "getaddrinfo", resolve)
    monkeypatch.setattr(api.asyncio, "get_running_loop", lambda: loop)
    asyncio.run(api._validate_public_url("https://public.test/image.jpg"))
    for url in ("http://private.test/a", "https://mixed.test/a", "file:///etc/passwd"):
        try:
            asyncio.run(api._validate_public_url(url))
        except api.HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError(f"accepted unsafe URL: {url}")
    loop.close()


def test_redirect_target_is_revalidated_and_body_is_bounded(monkeypatch):
    validated = []

    async def validate(url):
        validated.append(url)
        if "127.0.0.1" in url:
            raise api.HTTPException(status_code=400, detail="blocked")
        return ["93.184.216.34"]

    monkeypatch.setattr(api, "_validate_public_url", validate)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1/a.jpg"})
    )

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        async def download_redirect():
            async with httpx.AsyncClient(transport=transport) as client:
                await api._download_image(client, "https://public.test/a.jpg", os.path.join(tmpdir, "a"), 100)

        try:
            asyncio.run(download_redirect())
        except api.HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("redirect target was not blocked")
        assert validated == ["https://public.test/a.jpg", "http://127.0.0.1/a.jpg"]

        monkeypatch.setattr(api, "_validate_public_url", lambda url: _public_address())
        oversized = httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "image/jpeg"}, content=b"12345")
        )

        async def bounded_download():
            async with httpx.AsyncClient(transport=oversized) as client:
                await api._download_image(client, "https://public.test/a.jpg", os.path.join(tmpdir, "b"), 4)

        try:
            asyncio.run(bounded_download())
        except api.HTTPException as exc:
            assert exc.status_code == 413
        else:
            raise AssertionError("oversized body was accepted")


async def _public_address():
    return ["93.184.216.34"]
