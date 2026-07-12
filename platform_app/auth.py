import hashlib
import json
import secrets
import time
from collections import defaultdict, deque
from datetime import timezone, timedelta

from fastapi import Cookie, Depends, Header, HTTPException, Request, Response
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_session
from .models import AuthSession, User, utcnow


password_hash = PasswordHash.recommended()
COOKIE_NAME = 'preference_platform_session'
CSRF_COOKIE_NAME = 'preference_platform_csrf'
SESSION_DAYS = 30
_login_failures: dict[str, deque[float]] = defaultdict(deque)


def digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def is_expired(value) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= utcnow()


def create_session(session: Session, user: User, response: Response) -> None:
    settings = get_settings()
    raw_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    session.add(AuthSession(
        user_id=user.id,
        token_hash=digest(raw_token),
        expires_at=utcnow() + timedelta(days=SESSION_DAYS),
    ))
    user.last_login_at = utcnow()
    session.commit()
    response.set_cookie(
        COOKIE_NAME,
        raw_token,
        httponly=True,
        samesite='strict',
        secure=settings.cookie_secure,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        path='/',
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        csrf_token,
        httponly=False,
        samesite='strict',
        secure=settings.cookie_secure,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        path='/',
    )


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path='/')
    response.delete_cookie(CSRF_COOKIE_NAME, path='/')


def require_user(
    session_token: str | None = Cookie(default=None, alias=COOKIE_NAME),
    session: Session = Depends(get_session),
) -> User:
    if not session_token:
        raise HTTPException(status_code=401, detail='请先登录')
    auth_session = session.scalar(select(AuthSession).where(AuthSession.token_hash == digest(session_token)))
    if auth_session is None or is_expired(auth_session.expires_at):
        raise HTTPException(status_code=401, detail='登录已过期')
    user = session.get(User, auth_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail='用户不可用')
    return user


def enforce_csrf(
    request: Request,
    csrf_header: str | None = Header(default=None, alias='X-CSRF-Token'),
    csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE_NAME),
) -> None:
    if not get_settings().csrf_enabled:
        return
    if request.method in {'GET', 'HEAD', 'OPTIONS'}:
        return
    if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
        raise HTTPException(status_code=403, detail='CSRF 校验失败')


def check_login_rate_limit(username: str) -> None:
    settings = get_settings()
    bucket = _login_failures[username]
    now = time.time()
    window = settings.login_rate_limit_window_seconds
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= settings.login_rate_limit_attempts:
        raise HTTPException(status_code=429, detail='登录尝试过多，请稍后再试')


def record_login_failure(username: str) -> None:
    _login_failures[username].append(time.time())


def clear_login_failures(username: str) -> None:
    _login_failures.pop(username, None)


def request_body_hash(payload: dict | str) -> str:
    if isinstance(payload, str):
        raw = payload
    else:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
