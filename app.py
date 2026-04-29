import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request
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

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5 per hour"],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
)
limiter.init_app(app)


def _db_conn():
    conn = sqlite3.connect(_ANALYTICS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_analytics_schema():
    conn = _db_conn()
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
              city TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ts ON api_requests(ts_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_requests_ip ON api_requests(ip)")
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


@app.after_request
def _analytics_after_request(response):
    path = request.path or ""
    if path.startswith("/health") or path.startswith("/admin"):
        return response

    ip = _get_client_ip()
    geo = _geo_lookup(ip)
    ts_utc = datetime.now(timezone.utc).isoformat()

    try:
        conn = _db_conn()
        try:
            conn.execute(
                """
                INSERT INTO api_requests (ts_utc, ip, method, path, status_code, country, region, city)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    ip,
                    request.method,
                    path,
                    int(getattr(response, "status_code", 0) or 0),
                    geo.get("country"),
                    geo.get("region"),
                    geo.get("city"),
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
    return (
        jsonify(
            {
                "status": "failure",
                "error": "Rate limit exceeded",
            }
        ),
        429,
    )


@app.get("/admin/dashboard")
def admin_dashboard():
    token = os.getenv("ADMIN_TOKEN")
    supplied = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token and supplied != token:
        return jsonify({"status": "failure", "error": "Unauthorized"}), 401

    _ensure_analytics_schema()
    conn = _db_conn()
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

    country_rows = "".join(
        [f"<tr><td>{r['country']}</td><td style='text-align:right'>{r['c']}</td></tr>" for r in by_country]
    )
    recent_rows = "".join(
        [
            "<tr>"
            f"<td>{r['ts_utc']}</td>"
            f"<td>{r['ip'] or ''}</td>"
            f"<td>{r['method']}</td>"
            f"<td>{r['path']}</td>"
            f"<td style='text-align:right'>{r['status_code']}</td>"
            f"<td>{r['country']}</td>"
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


@app.get("/health")
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
@limiter.limit("5 per hour")
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
@limiter.limit("5 per hour")
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
@limiter.limit("5 per hour")
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
