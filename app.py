import os
import hashlib
import hmac
import ipaddress
import json
import secrets
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from model import DEFAULT_MODEL_PATH, predict_image


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app)

_ANALYTICS_DB_PATH = os.getenv("ANALYTICS_DB_PATH", os.path.join(os.getcwd(), "analytics.sqlite3"))
_GEO_CACHE_TTL_SECONDS = int(os.getenv("GEO_CACHE_TTL_SECONDS", "86400"))
_geo_cache = {}
_DATABASE_URL = os.getenv("DATABASE_URL")
_ANALYTICS_RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "30"))
_analytics_last_cleanup_ts = 0.0

_API_KEY_PREFIX_LEN = int(os.getenv("API_KEY_PREFIX_LEN", "24"))
_API_KEY_HEADER = "X-API-Key"
_API_KEY_HMAC_ENV_PREFIX = "API_KEY_HMAC_SECRET_V"
_ADMIN_IP_ALLOWLIST = os.getenv("ADMIN_IP_ALLOWLIST", "")

_API_IP_SAFETY_LIMIT = os.getenv("API_IP_SAFETY_LIMIT", "60 per minute")

_auth_schema_ready = False

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
)
limiter.init_app(app)


def _sqlite_conn():
    conn = sqlite3.connect(_ANALYTICS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _pg_conn():
    import psycopg2

    return psycopg2.connect(_DATABASE_URL)


def _env_hmac_secret(version: int):
    return os.getenv(f"{_API_KEY_HMAC_ENV_PREFIX}{int(version)}")


def _parse_api_key(raw_key: str):
    k = (raw_key or "").strip()
    if not k:
        return None

    # Expected format: mw_live_<random> (or mw_test_<random> later)
    if "_" not in k:
        return None
    parts = k.split("_", 2)
    if len(parts) < 3:
        return None

    rand = parts[2]
    if not rand or len(rand) < _API_KEY_PREFIX_LEN:
        return None

    return rand[:_API_KEY_PREFIX_LEN], k


def _hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()


def _admin_allowlisted(ip: str) -> bool:
    allow = (_ADMIN_IP_ALLOWLIST or "").strip()
    if not allow:
        return True

    try:
        ip_obj = ipaddress.ip_address(ip)
    except Exception:
        return False

    for part in [p.strip() for p in allow.split(",") if p.strip()]:
        try:
            if "/" in part:
                if ip_obj in ipaddress.ip_network(part, strict=False):
                    return True
            else:
                if ip_obj == ipaddress.ip_address(part):
                    return True
        except Exception:
            continue
    return False


def _admin_guard():
    ip = _get_client_ip()
    if not _admin_allowlisted(ip):
        _admin_audit("admin_denied_ip", {"ip": ip})
        return jsonify({"status": "failure", "error": "Unauthorized"}), 401

    token = os.getenv("ADMIN_TOKEN")
    supplied = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token and supplied != token:
        _admin_audit("admin_denied_token", {"ip": ip})
        return jsonify({"status": "failure", "error": "Unauthorized"}), 401

    return None


def _require_api_key():
    raw = request.headers.get(_API_KEY_HEADER)
    parsed = _parse_api_key(raw or "")
    if not parsed:
        app.logger.info("auth_failed reason=missing_or_malformed_key")
        return None
    prefix, full = parsed

    if not _DATABASE_URL:
        app.logger.error("auth_failed reason=no_database_url")
        return None

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, key_hash, hmac_version, revoked_at, expires_at, plan_id
            FROM api_keys
            WHERE key_prefix = %s
            """,
            (prefix,),
        )
        row = cur.fetchone()
        if not row:
            app.logger.info("auth_failed reason=unknown_prefix")
            return None

        api_key_id, key_hash, hmac_version, revoked_at, expires_at, plan_id = row
        if revoked_at is not None:
            app.logger.info("auth_failed reason=revoked_key api_key_id=%s", api_key_id)
            return None
        if expires_at is not None and isinstance(expires_at, datetime):
            if expires_at < datetime.now(timezone.utc):
                app.logger.info("auth_failed reason=expired_key api_key_id=%s", api_key_id)
                return None

        secret = _env_hmac_secret(int(hmac_version or 1))
        if not secret:
            app.logger.error("auth_failed reason=missing_hmac_secret version=%s", hmac_version)
            return None

        calc = _hmac_sha256(secret, full)
        if not hmac.compare_digest(str(key_hash or ""), calc):
            app.logger.info("auth_failed reason=hmac_mismatch api_key_id=%s", api_key_id)
            return None

        cur.execute(
            """
            SELECT daily_limit, hourly_limit, burst_per_second
            FROM plans
            WHERE id = %s
            """,
            (plan_id,),
        )
        prow = cur.fetchone()
        if not prow:
            app.logger.error("auth_failed reason=missing_plan api_key_id=%s plan_id=%s", api_key_id, plan_id)
            return None

        daily_limit, hourly_limit, burst_per_second = prow
        g.api_key_id = int(api_key_id)
        g.api_plan = {
            "daily_limit": int(daily_limit),
            "hourly_limit": int(hourly_limit),
            "burst_per_second": int(burst_per_second),
        }
        return g.api_key_id
    finally:
        conn.close()


def _plan_limit_string():
    plan = getattr(g, "api_plan", None) or {}
    daily = int(plan.get("daily_limit", 0) or 0)
    hourly = int(plan.get("hourly_limit", 0) or 0)
    burst = int(plan.get("burst_per_second", 0) or 0)
    if daily <= 0 or hourly <= 0 or burst <= 0:
        return "1 per day"
    return f"{daily} per day; {hourly} per hour; {burst} per second"


def _admin_audit(action: str, details: dict | None = None):
    if not _DATABASE_URL:
        return
    try:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO admin_audit (ts_utc, action, actor, ip, details_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(timezone.utc),
                    action,
                    (request.headers.get("X-Admin-Actor") or ""),
                    _get_client_ip(),
                    (json.dumps(details) if details else None),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _maybe_cleanup_analytics():
    global _analytics_last_cleanup_ts

    now = time.time()
    if now - _analytics_last_cleanup_ts < 3600:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=_ANALYTICS_RETENTION_DAYS)

    try:
        if _DATABASE_URL:
            conn = _pg_conn()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM api_requests WHERE ts_utc < %s", (cutoff,))
                conn.commit()
            finally:
                conn.close()
        else:
            conn = _sqlite_conn()
            try:
                conn.execute("DELETE FROM api_requests WHERE ts_utc < ?", (cutoff.isoformat(),))
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass

    _analytics_last_cleanup_ts = now


def _ensure_analytics_schema():
    if _DATABASE_URL:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_requests (
                  id BIGSERIAL PRIMARY KEY,
                  ts_utc TIMESTAMPTZ NOT NULL,
                  ip TEXT,
                  method TEXT,
                  path TEXT,
                  status_code INTEGER,
                  country TEXT,
                  region TEXT,
                  city TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ts ON api_requests(ts_utc)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ip ON api_requests(ip)")
            cur.execute("ALTER TABLE api_requests ADD COLUMN IF NOT EXISTS api_key_id BIGINT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_api_key_id ON api_requests(api_key_id)")
            conn.commit()
        finally:
            conn.close()
        return

    conn = _sqlite_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_requests (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_utc TEXT NOT NULL,
              ip TEXT,
              method TEXT,
              path TEXT,
              status_code INTEGER,
              country TEXT,
              region TEXT,
              city TEXT,
              api_key_id INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ts ON api_requests(ts_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ip ON api_requests(ip)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_api_key_id ON api_requests(api_key_id)")
        conn.commit()
    finally:
        conn.close()


def _get_client_ip():
    ip = get_remote_address()
    if not ip:
        return None
    if ip == "127.0.0.1" and request.headers.get("X-Forwarded-For"):
        ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return ip


def _geo_lookup(ip):
    if not ip:
        return {"country": None, "region": None, "city": None}

    now = time.time()
    cached = _geo_cache.get(ip)
    if cached and (now - cached["ts"] < _GEO_CACHE_TTL_SECONDS):
        return cached["geo"]

    geo = {"country": None, "region": None, "city": None}
    try:
        import requests

        r = requests.get(f"https://ipapi.co/{ip}/json/", timeout=1.5)
        if r.ok:
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            geo = {
                "country": j.get("country_name") or j.get("country"),
                "region": j.get("region") or j.get("region_code"),
                "city": j.get("city"),
            }
    except Exception:
        pass

    _geo_cache[ip] = {"ts": now, "geo": geo}
    return geo


@app.before_request
def _analytics_before_request():
    _ensure_analytics_schema()
    _maybe_cleanup_analytics()


@app.after_request
def _analytics_after_request(response):
    path = request.path or ""
    if path.startswith("/health") or path.startswith("/admin") or path.startswith("/privacy"):
        return response

    ip = _get_client_ip()
    geo = _geo_lookup(ip)
    ts_dt = datetime.now(timezone.utc)

    try:
        if _DATABASE_URL:
            conn = _pg_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO api_requests (ts_utc, ip, method, path, status_code, country, region, city, api_key_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ts_dt,
                        ip,
                        request.method,
                        path,
                        int(getattr(response, "status_code", 0) or 0),
                        geo.get("country"),
                        geo.get("region"),
                        geo.get("city"),
                        int(getattr(g, "api_key_id", 0) or 0) or None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        else:
            conn = _sqlite_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO api_requests (ts_utc, ip, method, path, status_code, country, region, city, api_key_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_dt.isoformat(),
                        ip,
                        request.method,
                        path,
                        int(getattr(response, "status_code", 0) or 0),
                        geo.get("country"),
                        geo.get("region"),
                        geo.get("city"),
                        int(getattr(g, "api_key_id", 0) or 0) or None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass

    return response


@app.errorhandler(429)
def ratelimit_handler(e):
    headers = {}
    retry_after = getattr(e, "retry_after", None)
    if retry_after is not None:
        try:
            headers["Retry-After"] = str(int(retry_after))
        except Exception:
            pass

    return (jsonify({"error": "rate_limited"}), 429, headers)


@app.before_request
def _auth_before_request():
    path = request.path or ""

    if path.startswith("/admin"):
        guard = _admin_guard()
        if guard is not None:
            return guard
        return None

    if path.startswith("/health") or path.startswith("/privacy"):
        return None

    if path.startswith("/api/") or path == "/predict":
        _ensure_auth_schema()
        api_key_id = _require_api_key()
        if not api_key_id:
            return jsonify({"error": "invalid_api_key"}), 401

    return None


def _ensure_auth_schema():
    global _auth_schema_ready
    if not _DATABASE_URL:
        return

    if _auth_schema_ready:
        return

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
              id BIGSERIAL PRIMARY KEY,
              code TEXT UNIQUE NOT NULL,
              daily_limit INTEGER NOT NULL,
              hourly_limit INTEGER NOT NULL,
              burst_per_second INTEGER NOT NULL,
              created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
              id BIGSERIAL PRIMARY KEY,
              key_prefix TEXT UNIQUE NOT NULL,
              key_hash TEXT NOT NULL,
              hmac_version SMALLINT NOT NULL DEFAULT 1,
              label TEXT,
              plan_id BIGINT NOT NULL REFERENCES plans(id),
              created_at TIMESTAMPTZ NOT NULL,
              revoked_at TIMESTAMPTZ,
              expires_at TIMESTAMPTZ
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_audit (
              id BIGSERIAL PRIMARY KEY,
              ts_utc TIMESTAMPTZ NOT NULL,
              action TEXT NOT NULL,
              actor TEXT,
              ip TEXT,
              details_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_plan_id ON api_keys(plan_id)")

        cur.execute("SELECT id FROM plans WHERE code = %s", ("free",))
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT INTO plans (code, daily_limit, hourly_limit, burst_per_second, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                ("free", 100, 20, 10, datetime.now(timezone.utc)),
            )
        cur.execute("SELECT id FROM plans WHERE code = %s", ("pro",))
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT INTO plans (code, daily_limit, hourly_limit, burst_per_second, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                ("pro", 10000, 600, 20, datetime.now(timezone.utc)),
            )

        conn.commit()
        _auth_schema_ready = True
    finally:
        conn.close()


@app.get("/admin/dashboard")
def admin_dashboard():
    _ensure_auth_schema()
    _ensure_analytics_schema()
    if _DATABASE_URL:
        conn = _pg_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM api_requests")
            total_requests = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT ip) FROM api_requests WHERE ip IS NOT NULL AND ip != ''")
            total_users = cur.fetchone()[0]
            cur.execute(
                """
                SELECT COALESCE(country, 'Unknown') AS country, COUNT(*) AS c
                FROM api_requests
                GROUP BY COALESCE(country, 'Unknown')
                ORDER BY c DESC
                LIMIT 50
                """
            )
            by_country = [
                {"country": r[0], "c": r[1]}
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT ts_utc, ip, method, path, status_code, COALESCE(country, 'Unknown') AS country
                FROM api_requests
                ORDER BY id DESC
                LIMIT 100
                """
            )
            recent = [
                {
                    "ts_utc": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "ip": r[1],
                    "method": r[2],
                    "path": r[3],
                    "status_code": r[4],
                    "country": r[5],
                }
                for r in cur.fetchall()
            ]
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            total_requests = conn.execute("SELECT COUNT(*) AS c FROM api_requests").fetchone()["c"]
            total_users = conn.execute(
                "SELECT COUNT(DISTINCT ip) AS c FROM api_requests WHERE ip IS NOT NULL AND ip != ''"
            ).fetchone()["c"]
            by_country = conn.execute(
                """
                SELECT COALESCE(country, 'Unknown') AS country, COUNT(*) AS c
                FROM api_requests
                GROUP BY COALESCE(country, 'Unknown')
                ORDER BY c DESC
                LIMIT 50
                """
            ).fetchall()
            recent = conn.execute(
                """
                SELECT ts_utc, ip, method, path, status_code, COALESCE(country, 'Unknown') AS country
                FROM api_requests
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        finally:
            conn.close()

    def _get(row, key):
        return row[key] if hasattr(row, "keys") else row.get(key)

    country_rows = "".join(
        [
            f"<tr><td>{(_get(r, 'country') or '')}</td><td style='text-align:right'>{_get(r, 'c')}</td></tr>"
            for r in by_country
        ]
    )
    recent_rows = "".join(
        [
            "<tr>"
            f"<td>{(_get(r, 'ts_utc') or '')}</td>"
            f"<td>{(_get(r, 'ip') or '')}</td>"
            f"<td>{(_get(r, 'method') or '')}</td>"
            f"<td>{(_get(r, 'path') or '')}</td>"
            f"<td style='text-align:right'>{(_get(r, 'status_code') or '')}</td>"
            f"<td>{(_get(r, 'country') or '')}</td>"
            "</tr>"
            for r in recent
        ]
    )

    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>API Usage Dashboard</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
      .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }}
      .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }}
      .label {{ color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
      .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ border-bottom: 1px solid #f0f0f0; padding: 8px; font-size: 13px; vertical-align: top; }}
      th {{ text-align: left; color: #374151; font-weight: 600; }}
      h2 {{ margin-top: 18px; margin-bottom: 10px; font-size: 16px; }}
      .muted {{ color: #6b7280; font-size: 12px; }}
    </style>
  </head>
  <body>
    <h1 style=\"margin:0 0 6px 0\">API Usage Dashboard</h1>
    <div class=\"muted\">Counts are derived from request logs collected by the API.</div>

    <div class=\"kpis\">
      <div class=\"card\"><div class=\"label\">Total requests</div><div class=\"value\">{total_requests}</div></div>
      <div class=\"card\"><div class=\"label\">Total users (unique IPs)</div><div class=\"value\">{total_users}</div></div>
    </div>

    <div class=\"card\">
      <h2>Requests by country (top 50)</h2>
      <table>
        <thead><tr><th>Country</th><th style=\"text-align:right\">Requests</th></tr></thead>
        <tbody>
          {country_rows}
        </tbody>
      </table>
    </div>

    <div class=\"card\" style=\"margin-top:12px\">
      <h2>Recent requests (last 100)</h2>
      <table>
        <thead><tr><th>Time (UTC)</th><th>IP</th><th>Method</th><th>Path</th><th style=\"text-align:right\">Status</th><th>Country</th></tr></thead>
        <tbody>
          {recent_rows}
        </tbody>
      </table>
    </div>
  </body>
</html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.get("/admin/api-keys")
def admin_list_api_keys():
    _ensure_auth_schema()

    if not _DATABASE_URL:
        return jsonify({"status": "failure", "error": "DATABASE_URL not configured"}), 500

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT k.id, k.key_prefix, k.label, p.code, k.created_at, k.revoked_at, k.expires_at
            FROM api_keys k
            JOIN plans p ON p.id = k.plan_id
            ORDER BY k.id DESC
            LIMIT 200
            """
        )
        rows = cur.fetchall()
        keys = [
            {
                "id": int(r[0]),
                "key_prefix": r[1],
                "label": r[2],
                "plan": r[3],
                "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
                "revoked_at": (r[5].isoformat() if (r[5] is not None and hasattr(r[5], "isoformat")) else (str(r[5]) if r[5] is not None else None)),
                "expires_at": (r[6].isoformat() if (r[6] is not None and hasattr(r[6], "isoformat")) else (str(r[6]) if r[6] is not None else None)),
            }
            for r in rows
        ]
        _admin_audit("api_keys_list", {"count": len(keys)})
        return jsonify({"status": "success", "keys": keys})
    finally:
        conn.close()


@app.post("/admin/api-keys")
def admin_create_api_key():
    _ensure_auth_schema()

    if not _DATABASE_URL:
        return jsonify({"status": "failure", "error": "DATABASE_URL not configured"}), 500

    secret = _env_hmac_secret(1)
    if not secret:
        return jsonify({"status": "failure", "error": "API_KEY_HMAC_SECRET_V1 not configured"}), 500

    body = request.get_json(silent=True) or {}
    label = (body.get("label") or "").strip() or None
    plan_code = (body.get("plan") or "free").strip().lower()
    expires_at = body.get("expires_at")

    expires_dt = None
    if isinstance(expires_at, str) and expires_at.strip():
        try:
            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        except Exception:
            return jsonify({"status": "failure", "error": "Invalid expires_at (use ISO8601)"}), 400

    # raw key is only returned once
    raw_key = f"mw_live_{secrets.token_urlsafe(32)}"
    prefix = _parse_api_key(raw_key)[0]
    key_hash = _hmac_sha256(secret, raw_key)

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM plans WHERE code = %s", (plan_code,))
        prow = cur.fetchone()
        if not prow:
            return jsonify({"status": "failure", "error": "Unknown plan"}), 400
        plan_id = int(prow[0])

        cur.execute(
            """
            INSERT INTO api_keys (key_prefix, key_hash, hmac_version, label, plan_id, created_at, revoked_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, NULL, %s)
            RETURNING id
            """,
            (prefix, key_hash, 1, label, plan_id, datetime.now(timezone.utc), expires_dt),
        )
        new_id = int(cur.fetchone()[0])
        conn.commit()
        _admin_audit("api_key_created", {"api_key_id": new_id, "plan": plan_code, "label": label, "expires_at": expires_at})
        return jsonify(
            {
                "status": "success",
                "api_key_id": new_id,
                "api_key": raw_key,
                "key_prefix": prefix,
                "plan": plan_code,
            }
        )
    finally:
        conn.close()


@app.post("/admin/api-keys/<int:api_key_id>/revoke")
def admin_revoke_api_key(api_key_id: int):
    _ensure_auth_schema()
    if not _DATABASE_URL:
        return jsonify({"status": "failure", "error": "DATABASE_URL not configured"}), 500

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE api_keys SET revoked_at = %s WHERE id = %s AND revoked_at IS NULL", (datetime.now(timezone.utc), api_key_id))
        conn.commit()
        _admin_audit("api_key_revoked", {"api_key_id": api_key_id})
        return jsonify({"status": "success"})
    finally:
        conn.close()


@app.post("/admin/api-keys/<int:api_key_id>/rotate")
def admin_rotate_api_key(api_key_id: int):
    """Rotate key atomically: create replacement and revoke old in a single DB transaction."""

    _ensure_auth_schema()
    if not _DATABASE_URL:
        return jsonify({"status": "failure", "error": "DATABASE_URL not configured"}), 500

    secret = _env_hmac_secret(1)
    if not secret:
        return jsonify({"status": "failure", "error": "API_KEY_HMAC_SECRET_V1 not configured"}), 500

    raw_key = f"mw_live_{secrets.token_urlsafe(32)}"
    prefix = _parse_api_key(raw_key)[0]
    key_hash = _hmac_sha256(secret, raw_key)

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT plan_id, label, revoked_at, expires_at
                FROM api_keys
                WHERE id = %s
                FOR UPDATE
                """,
                (api_key_id,),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return jsonify({"status": "failure", "error": "Not found"}), 404
            plan_id, label, revoked_at, expires_at = row
            if revoked_at is not None:
                conn.rollback()
                return jsonify({"status": "failure", "error": "Already revoked"}), 400

            cur.execute(
                """
                INSERT INTO api_keys (key_prefix, key_hash, hmac_version, label, plan_id, created_at, revoked_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, NULL, %s)
                RETURNING id
                """,
                (prefix, key_hash, 1, label, int(plan_id), datetime.now(timezone.utc), expires_at),
            )
            new_id = int(cur.fetchone()[0])

            cur.execute("UPDATE api_keys SET revoked_at = %s WHERE id = %s", (datetime.now(timezone.utc), api_key_id))

            cur.execute(
                """
                INSERT INTO admin_audit (ts_utc, action, actor, ip, details_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(timezone.utc),
                    "api_key_rotated",
                    (request.headers.get("X-Admin-Actor") or ""),
                    _get_client_ip(),
                    json.dumps({"old_api_key_id": api_key_id, "new_api_key_id": new_id}),
                ),
            )

            conn.commit()
            return jsonify({"status": "success", "old_api_key_id": api_key_id, "new_api_key_id": new_id, "api_key": raw_key, "key_prefix": prefix})
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


@app.get("/privacy")
def privacy_policy():
    html = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Privacy Policy</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; max-width: 900px; }
      h1 { margin: 0 0 8px 0; }
      h2 { margin-top: 18px; font-size: 16px; }
      p, li { line-height: 1.5; }
      .muted { color: #6b7280; font-size: 12px; }
      code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h1>Privacy Policy</h1>
    <div class=\"muted\">Last updated: """ + datetime.now(timezone.utc).date().isoformat() + """</div>

    <h2>What data we collect</h2>
    <ul>
      <li>IP address</li>
      <li>Request metadata (timestamp, endpoint path, HTTP method, status code)</li>
      <li>Approximate location derived from IP (country/region/city), when available</li>
    </ul>

    <h2>Why we collect it</h2>
    <ul>
      <li>Rate limiting</li>
      <li>Abuse prevention</li>
      <li>Usage analytics (e.g., total usage and usage by geography)</li>
    </ul>

    <h2>Data retention</h2>
    <p>
      We retain analytics data for up to <strong>30 days</strong> and then delete older records.
    </p>

    <h2>Contact</h2>
    <p>
      If you have questions about this policy, please contact the service owner.
    </p>
  </body>
</html>"""

    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.get("/health")
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
@limiter.limit(_API_IP_SAFETY_LIMIT, key_func=get_remote_address)
@limiter.limit(_plan_limit_string, key_func=lambda: str(getattr(g, "api_key_id", "")))
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing multipart form field 'file'"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_path = request.args.get("model_path", DEFAULT_MODEL_PATH)

    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        result = predict_image(tmp_path, model_path=model_path)
        return jsonify(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/api/c2pa")
@limiter.limit(_API_IP_SAFETY_LIMIT, key_func=get_remote_address)
@limiter.limit(_plan_limit_string, key_func=lambda: str(getattr(g, "api_key_id", "")))
def api_c2pa():
    if "file" not in request.files:
        return jsonify({"error": "No file part", "status": "failure"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No selected file", "status": "failure"}), 400

    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        try:
            import c2pa  # type: ignore
        except Exception as e:
            return jsonify({"error": f"c2pa-python is not available: {str(e)}", "status": "failure"}), 500

        settings = None
        verify = request.args.get("verify", "false").lower() in ("1", "true", "yes")
        if verify:
            settings = c2pa.Settings.from_dict({"verify": {"verify_cert_anchors": False}})

        with c2pa.Context(settings) as context:
            with c2pa.Reader(tmp_path, context=context) as reader:
                detailed = reader.detailed_json()

        return jsonify({"status": "success", "c2pa": detailed})
    except Exception as e:
        return jsonify({"error": f"C2PA read failed: {str(e)}", "status": "failure"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/api/ai-detect")
@limiter.limit(_API_IP_SAFETY_LIMIT, key_func=get_remote_address)
@limiter.limit(_plan_limit_string, key_func=lambda: str(getattr(g, "api_key_id", "")))
def api_ai_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part", "status": "failure"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No selected file", "status": "failure"}), 400

    model_path = request.args.get("model_path", DEFAULT_MODEL_PATH)
    suffix = os.path.splitext(f.filename)[1] or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        result = predict_image(tmp_path, model_path=model_path)

        confidence = None
        if "proba" in result and isinstance(result["proba"], list) and len(result["proba"]) >= 2:
            confidence = {"real": float(result["proba"][0]), "fake": float(result["proba"][1])}

        return jsonify(
            {
                "status": "success",
                "prediction": result.get("label"),
                "confidence": confidence,
            }
        )
    except Exception as e:
        return jsonify({"error": f"AI Detection failed: {str(e)}", "status": "failure"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
