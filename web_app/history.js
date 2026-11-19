const API_BASE = "";
const JWT_KEY = "truthlens.web.jwt";
const API_KEY = "truthlens.web.apiKey";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function authHeaders() {
  const jwt = localStorage.getItem(JWT_KEY);
  const apiKey = sessionStorage.getItem(API_KEY);
  if (jwt) return { Authorization: `Bearer ${jwt}` };
  if (apiKey) return { "X-API-Key": apiKey };
  return {};
}

function bandBadge(row) {
  const band = String(row.result_band || "").toLowerCase();
  const label = row.result_label || row.result_band || "—";
  let cls = "neutral";
  if (band === "ai" || band === "likely_ai") cls = "ai";
  else if (band === "real" || band === "likely_real") cls = "clean";
  else if (band === "uncertain" || band === "credentials_found") cls = "dubious";
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function fmtTs(ts) {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? esc(ts) : esc(d.toISOString().replace("T", " ").slice(0, 19) + " UTC");
}

async function load() {
  const status = $("status");
  const table = $("table");
  const rows = $("rows");
  if (!localStorage.getItem(JWT_KEY) && !sessionStorage.getItem(API_KEY)) {
    status.innerHTML = 'Sign in on the <a href="/">app</a> to view your analysis history.';
    table.hidden = true;
    return;
  }
  status.textContent = "Loading…";
  const params = new URLSearchParams({ limit: "500" });
  const from = $("fromDate").value;
  const to = $("toDate").value;
  const type = $("typeFilter").value;
  if (from) params.set("start", from);
  if (to) params.set("end", `${to}T23:59:59Z`);
  if (type) params.set("analysis_type", type);

  try {
    const resp = await fetch(`${API_BASE}/api/my-analyses?${params}`, { headers: authHeaders() });
    const data = await resp.json().catch(() => null);
    if (!resp.ok || data?.status !== "success") throw new Error(data?.error || `Could not load history (${resp.status})`);
    let list = data.analyses || [];
    const channel = $("channelFilter").value;
    if (channel) list = list.filter((r) => String(r.channel || "") === channel);

    rows.innerHTML = list.map((r) => `
      <tr>
        <td>${fmtTs(r.ts_utc)}</td>
        <td>${esc(r.analysis_type || "")}</td>
        <td>${bandBadge(r)}${r.error ? `<div class="muted">${esc(String(r.error).slice(0, 120))}</div>` : ""}</td>
        <td>${r.p_ai === null || r.p_ai === undefined ? "—" : `${Math.round(Number(r.p_ai) * 100)}%`}</td>
        <td>${esc(r.backend || "—")}</td>
        <td>${esc(r.provenance_source || "—")}</td>
        <td>${esc(r.channel || "—")}</td>
        <td class="hash">${r.image_sha256 ? esc(String(r.image_sha256).slice(0, 12)) + "…" : "—"}</td>
      </tr>`).join("");

    table.hidden = !list.length;
    status.textContent = list.length ? `${list.length} record${list.length === 1 ? "" : "s"}${list.length >= 500 ? " (showing first 500)" : ""}` : "No analyses found for the selected filters.";
  } catch (e) {
    status.textContent = e.message || "Could not load history.";
    table.hidden = true;
  }
}

$("applyBtn").addEventListener("click", load);
$("refreshBtn").addEventListener("click", load);
load();
