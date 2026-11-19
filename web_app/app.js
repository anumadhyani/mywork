const API_BASE = "";
const JWT_STORAGE_KEY = "truthlens.web.jwt";
const EMAIL_STORAGE_KEY = "truthlens.web.email";
const API_KEY_SESSION_KEY = "truthlens.web.apiKey";
const MAX_FILES = 50;
const MAX_FILE_MB = 25;
const MAX_CONCURRENT = 2;

const state = {
  jwt: "",
  email: "",
  name: "",
  apiKey: "",
  items: [],
  history: [],
  historyLoading: false,
  historyError: "",
  running: false,
  nextId: 1,
};

const $ = (id) => document.getElementById(id);

const els = {
  userArea: $("userArea"),
  authPanel: $("authPanel"),
  workspace: $("workspace"),
  googleButton: $("googleButton"),
  authMessage: $("authMessage"),
  apiKeyInput: $("apiKeyInput"),
  apiKeyButton: $("apiKeyButton"),
  dropZone: $("dropZone"),
  browseButton: $("browseButton"),
  fileInput: $("fileInput"),
  analyzeButton: $("analyzeButton"),
  clearButton: $("clearButton"),
  includeC2pa: $("includeC2pa"),
  queueCount: $("queueCount"),
  completedCount: $("completedCount"),
  statusMessage: $("statusMessage"),
  usageText: $("usageText"),
  totalCount: $("totalCount"),
  syntheticCount: $("syntheticCount"),
  dubiousCount: $("dubiousCount"),
  cleanCount: $("cleanCount"),
  resultsSubtitle: $("resultsSubtitle"),
  emptyState: $("emptyState"),
  resultsGrid: $("resultsGrid"),
  historyStatus: $("historyStatus"),
  historyList: $("historyList"),
  refreshHistoryButton: $("refreshHistoryButton"),
  applyHistoryFilter: $("applyHistoryFilter"),
  historyFrom: $("historyFrom"),
  historyTo: $("historyTo"),
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  }[ch]));
}

function decodeJwtPayload(token) {
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const padded = part.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - part.length % 4) % 4);
    return JSON.parse(atob(padded));
  } catch {
    return null;
  }
}

function isJwtValid(token) {
  const payload = decodeJwtPayload(token);
  return Boolean(payload && payload.exp && payload.exp * 1000 > Date.now());
}

function setAuthMessage(message) {
  els.authMessage.textContent = message || "";
}

function setStatusMessage(message) {
  els.statusMessage.textContent = message || "";
}

function authHeaders() {
  const headers = { "X-Client-Channel": "web_app" };
  if (state.jwt) headers.Authorization = `Bearer ${state.jwt}`;
  else if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  return headers;
}

function signedIn() {
  return Boolean(state.jwt || state.apiKey);
}

function initials(name, email) {
  const source = (name || email || "U").trim();
  return source.split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
}

function renderAuth() {
  const authed = signedIn();
  els.authPanel.classList.toggle("hidden", authed);
  els.workspace.classList.toggle("hidden", !authed);

  if (!authed) {
    els.userArea.innerHTML = "";
    return;
  }

  const label = state.apiKey ? "API key session" : (state.email || "Signed in");
  els.userArea.innerHTML = `
    <div class="user-pill">
      <span class="avatar">${escapeHtml(initials(state.name, state.email))}</span>
      <span>
        <div class="user-name">${escapeHtml(state.name || "Signed in")}</div>
        <div class="user-email">${escapeHtml(label)}</div>
      </span>
    </div>
    <button id="signOutButton" type="button" class="ghost">Sign out</button>
  `;
  $("signOutButton").addEventListener("click", signOut);
}

function setSession(jwt, payload) {
  state.jwt = jwt;
  state.apiKey = "";
  state.email = payload?.email || "";
  state.name = payload?.name || "";
  localStorage.setItem(JWT_STORAGE_KEY, jwt);
  localStorage.setItem(EMAIL_STORAGE_KEY, state.email);
  sessionStorage.removeItem(API_KEY_SESSION_KEY);
  renderAuth();
  render();
  loadHistory();
  refreshUsage();
}

function setApiKeySession(key) {
  state.apiKey = key;
  state.jwt = "";
  state.email = "";
  state.name = "API key user";
  sessionStorage.setItem(API_KEY_SESSION_KEY, key);
  localStorage.removeItem(JWT_STORAGE_KEY);
  localStorage.removeItem(EMAIL_STORAGE_KEY);
  renderAuth();
  render();
  loadHistory();
  refreshUsage();
}

function signOut() {
  try {
    window.google?.accounts?.id?.disableAutoSelect();
  } catch {}
  state.jwt = "";
  state.email = "";
  state.name = "";
  state.apiKey = "";
  localStorage.removeItem(JWT_STORAGE_KEY);
  localStorage.removeItem(EMAIL_STORAGE_KEY);
  sessionStorage.removeItem(API_KEY_SESSION_KEY);
  state.history = [];
  state.historyLoading = false;
  state.historyError = "";
  renderAuth();
  render();
  renderHistory();
}

async function exchangeGoogleCredential(credential) {
  setAuthMessage("Signing in...");
  const resp = await fetch(`${API_BASE}/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: credential }),
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok || !data?.access_token) {
    throw new Error(data?.error || `Google sign-in failed (${resp.status})`);
  }
  setSession(data.access_token, decodeJwtPayload(data.access_token));
  setAuthMessage("");
}

function waitForGoogleIdentity(timeoutMs = 5000) {
  const started = Date.now();
  return new Promise((resolve) => {
    const check = () => {
      if (window.google?.accounts?.id || Date.now() - started >= timeoutMs) {
        resolve();
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  });
}

async function initGoogleSignIn() {
  try {
    const cfg = await fetch(`${API_BASE}/web-config`).then((r) => r.json());
    if (!cfg.google_client_id) {
      setAuthMessage("Google sign-in is not configured. Use an API key instead.");
      return;
    }

    await waitForGoogleIdentity();
    if (!window.google?.accounts?.id) {
      setAuthMessage("Google sign-in could not be loaded. Check your connection or use an API key.");
      return;
    }

    google.accounts.id.initialize({
      client_id: cfg.google_client_id,
      callback: (response) => {
        exchangeGoogleCredential(response.credential).catch((err) => {
          setAuthMessage(err.message || "Google sign-in failed.");
        });
      },
    });
    google.accounts.id.renderButton(els.googleButton, {
      theme: "filled_blue",
      size: "large",
      shape: "pill",
      width: 300,
    });
  } catch {
    setAuthMessage("Could not load sign-in configuration. Use an API key instead.");
  }
}

function fileSizeText(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function addFiles(fileList) {
  if (state.running) {
    setStatusMessage("Wait for the current batch to finish before adding more images.");
    return;
  }

  const files = Array.from(fileList || []);
  const remaining = MAX_FILES - state.items.length;
  const accepted = [];
  const rejected = [];

  files.forEach((file) => {
    const isImage = file.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(file.name);
    if (!isImage) {
      rejected.push(`${file.name}: not an image`);
    } else if (file.size > MAX_FILE_MB * 1024 * 1024) {
      rejected.push(`${file.name}: over ${MAX_FILE_MB} MB`);
    } else if (accepted.length < remaining) {
      accepted.push(file);
    } else {
      rejected.push(`${file.name}: batch limit is ${MAX_FILES}`);
    }
  });

  accepted.forEach((file) => {
    state.items.push({
      id: state.nextId++,
      file,
      name: file.name || "image",
      size: file.size,
      previewUrl: URL.createObjectURL(file),
      status: "queued",
      result: null,
      c2pa: null,
      error: "",
    });
  });

  if (rejected.length) {
    setStatusMessage(`${rejected.length} file${rejected.length === 1 ? "" : "s"} skipped. ${rejected[0]}`);
  } else if (accepted.length) {
    setStatusMessage(`${accepted.length} image${accepted.length === 1 ? "" : "s"} ready for analysis.`);
  }

  render();
}

function categoryForResult(result) {
  const band = String(result?.band || "").toLowerCase();
  if (band === "ai" || band === "likely_ai") return "synthetic";
  if (band === "uncertain") return "dubious";
  if (band === "real" || band === "likely_real") return "clean";
  return "neutral";
}

function effectiveCategory(item) {
  if (item.c2pa?.ai_provenance) return "synthetic";
  return categoryForResult(item.result);
}

function resultSentence(item) {
  if (item.c2pa?.ai_provenance) return "AI-generated content confirmed by provenance.";
  const result = item.result;
  if (!result) return "Waiting for analysis";
  const category = categoryForResult(result);
  if (category === "clean" || category === "dubious") return "No strong AI signals detected.";
  return result.prediction || result.label || "This image is likely AI-generated.";
}

function resultLabel(item) {
  if (item.c2pa?.ai_provenance) return "AI (provenance)";
  const category = categoryForResult(item.result);
  if (category === "synthetic") return "Likely AI";
  if (category === "dubious") return "Uncertain";
  if (category === "clean") return "No strong AI";
  return "Unknown";
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function aiProbability(item) {
  if (typeof item.result?.p_ai === "number") return item.result.p_ai;
  const proba = item.result?.proba;
  if (Array.isArray(proba) && proba.length > 1) return Number(proba[1]);
  return null;
}

function c2paOrigin(detailed) {
  if (!detailed) return null;
  if (typeof detailed === "string") {
    try {
      detailed = JSON.parse(detailed);
    } catch {
      return null;
    }
  }
  if (typeof detailed !== "object") return null;
  const manifestStore = detailed.manifest_store || detailed.manifestStore || detailed.manifests || null;
  const activeId = detailed.active_manifest || detailed.activeManifest || null;
  let active = manifestStore && activeId && manifestStore[activeId] ? manifestStore[activeId] : null;
  if (!active && manifestStore && typeof manifestStore === "object") {
    active = Object.values(manifestStore).find((manifest) => manifest && typeof manifest === "object") || null;
  }
  return detailed.origin || active?.claim_generator || active?.claimGenerator || detailed.claim_generator || detailed.claimGenerator || null;
}

const DIGITAL_SOURCE_TYPE_LABELS = {
  trainedAlgorithmicMedia: "AI-generated",
  compositeWithTrainedAlgorithmicMedia: "partially AI-generated",
  algorithmicallyEnhanced: "AI-enhanced",
  softwareImage: "software-generated",
  algorithmicMedia: "algorithmically generated",
  compositeSynthetic: "synthetic composite",
  digitalCapture: "camera capture",
  compositeCapture: "composite capture",
  screenCapture: "screen capture",
  virtualRecording: "virtual recording",
  negativeFilm: "scanned film",
  positiveFilm: "scanned film",
  print: "scanned print",
  minorHumanEdits: "minor human edits",
};
const AI_SOURCE_TYPE_PRIORITY = [
  "trainedAlgorithmicMedia",
  "compositeWithTrainedAlgorithmicMedia",
  "algorithmicallyEnhanced",
  "softwareImage",
];

function friendlyDigitalSourceType(types) {
  if (!Array.isArray(types)) return null;
  const ordered = [...types].sort(
    (a, b) => (AI_SOURCE_TYPE_PRIORITY.indexOf(a) + 1 || 99) - (AI_SOURCE_TYPE_PRIORITY.indexOf(b) + 1 || 99)
  );
  for (const t of ordered) if (DIGITAL_SOURCE_TYPE_LABELS[t]) return DIGITAL_SOURCE_TYPE_LABELS[t];
  return null;
}

const ACTION_LABELS = {
  "c2pa.created": "created",
  "c2pa.edited": "edited",
  "c2pa.published": "published",
  "c2pa.cropped": "cropped",
  "c2pa.resized": "resized",
  "c2pa.filtered": "filtered",
  "c2pa.color_adjustments": "color adjusted",
  "c2pa.drawing": "drawn",
  "c2pa.combined": "combined",
  "c2pa.placed": "placed",
  "c2pa.opened": "opened",
  "c2pa.repackaged": "repackaged",
  "c2pa.transcoded": "transcoded",
  "c2pa.removed": "removed",
  "c2pa.orientation": "reoriented",
  "c2pa.changed": "changed",
  "c2pa.saved": "saved",
  "c2pa.produced": "produced",
  "c2pa.unknown": "unknown changes",
};

function friendlyActions(actions) {
  if (!Array.isArray(actions)) return null;
  const mapped = actions
    .filter((a) => typeof a === "string" && a.trim())
    .map((a) => ACTION_LABELS[a] || a.replace(/^[a-z0-9.]+\./i, "").replace(/[_-]+/g, " "));
  return mapped.length ? mapped.join(", ") : null;
}

function provenanceText(item) {
  if (!els.includeC2pa.checked && !item.c2pa) return "Provenance check skipped";
  if (item.c2pa?.error) return "Provenance check unavailable";
  if (item.c2pa?.status === "success") {
    const source = item.c2pa.source || item.c2pa.origin || c2paOrigin(item.c2pa.c2pa);
    if (item.c2pa.credentials_found === false || item.c2pa.result_band === "no_credentials") {
      return "No provenance credentials found";
    }
    const parts = [source ? `Source: ${source}` : "Provenance credentials found (source unknown)"];
    const declared = friendlyDigitalSourceType(item.c2pa.digital_source_types);
    if (declared) parts.push(`Declared: ${declared}`);
    const actions = friendlyActions(item.c2pa.actions);
    if (actions) parts.push(`Actions: ${actions}`);
    return parts.join(" · ");
  }
  return "Provenance pending";
}

function formatHistoryTime(ts) {
  if (!ts) return "—";
  const date = new Date(ts);
  return Number.isNaN(date.getTime()) ? String(ts) : date.toLocaleString();
}

function historyResultText(row) {
  if (row.analysis_type === "c2pa") {
    return row.provenance_source
      ? `Source: ${row.provenance_source}`
      : (row.result_label || row.result_band || "Provenance check");
  }
  const result = row.result_label || row.result_band || "AI analysis";
  return row.provenance_source ? `${result} · Source: ${row.provenance_source}` : result;
}

function mergeHistoryRows(rows) {
  const merged = new Map();
  rows.forEach((row) => {
    const timestamp = new Date(row.ts_utc || 0).getTime();
    const bucket = Number.isNaN(timestamp) ? row.id : Math.floor(timestamp / 60000);
    const key = `${row.image_sha256 || `row-${row.id}`}-${bucket}`;
    const item = merged.get(key) || { ...row };
    if (row.analysis_type === "ai_detect") {
      Object.assign(item, row);
    } else if (row.analysis_type === "c2pa") {
      item.provenance_source = row.provenance_source || item.provenance_source;
      item.c2pa_band = row.result_band;
    }
    merged.set(key, item);
  });
  return [...merged.values()].sort((a, b) => new Date(b.ts_utc || 0) - new Date(a.ts_utc || 0));
}

function renderHistory() {
  if (!signedIn()) {
    els.historyStatus.textContent = "Sign in to load your analysis history.";
    els.historyList.innerHTML = "";
    return;
  }
  if (state.historyLoading) {
    els.historyStatus.textContent = "Loading analysis history...";
    return;
  }
  if (state.historyError) {
    els.historyStatus.textContent = state.historyError;
    return;
  }
  if (!state.history.length) {
    els.historyStatus.textContent = "No analyses yet.";
    els.historyList.innerHTML = "";
    return;
  }

  els.historyStatus.textContent = `Latest ${state.history.length} analysis${state.history.length === 1 ? "" : "es"}.`;
  els.historyList.innerHTML = state.history.map((row) => `
    <div class="history-item">
      <div>
        <div class="history-result">${escapeHtml(historyResultText(row))}</div>
        <div class="history-meta">${escapeHtml(row.analysis_type || "analysis")} · ${escapeHtml(row.channel || "api")}</div>
      </div>
      <div class="history-date">${escapeHtml(formatHistoryTime(row.ts_utc))}</div>
    </div>
  `).join("");
}

async function loadHistory() {
  if (!signedIn()) {
    state.history = [];
    state.historyError = "";
    renderHistory();
    return;
  }

  state.historyLoading = true;
  state.historyError = "";
  renderHistory();
  try {
    const params = new URLSearchParams({ limit: "500" });
    const from = els.historyFrom?.value;
    const to = els.historyTo?.value;
    if (from) params.set("start", from);
    if (to) params.set("end", `${to}T23:59:59Z`);
    const resp = await fetch(`${API_BASE}/api/my-analyses?${params}`, {
      headers: authHeaders(),
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok || data?.status !== "success") {
      throw new Error(data?.error || `Could not load history (${resp.status})`);
    }
    state.history = mergeHistoryRows(data.analyses || []);
  } catch (error) {
    state.history = [];
    state.historyError = error.message || "Could not load history.";
  } finally {
    state.historyLoading = false;
    renderHistory();
  }
}

function statusLabel(status) {
  if (status === "analyzing") return "Analyzing";
  if (status === "done") return "Done";
  if (status === "error") return "Error";
  return "Queued";
}

function renderCard(item) {
  const category = effectiveCategory(item);
  const pAi = aiProbability(item);
  const canRemove = !state.running && item.status !== "analyzing";

  return `
    <article class="result-card" data-id="${item.id}">
      <div class="thumb-wrap">
        <img src="${item.previewUrl}" alt="${escapeHtml(item.name)}" loading="lazy" />
        <span class="status-badge ${item.status}">${statusLabel(item.status)}</span>
        ${canRemove ? `<button class="danger remove-card" data-remove="${item.id}" title="Remove image">×</button>` : ""}
        ${item.status === "done" && !item.feedbackSent ? `<button class="feedback-icon" data-feedback="${item.id}" title="Share feedback on this result" type="button"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg></button>` : ""}
      </div>
      <div class="card-body">
        <div>
          <div class="file-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
          <div class="file-meta">${fileSizeText(item.size)}</div>
        </div>

        ${item.status === "analyzing" ? `<div class="progress-line"></div>` : ""}

        ${item.status === "done" ? `
          <div class="result-badge ${category}">${resultLabel(item)}</div>
          <div class="result-text">${escapeHtml(resultSentence(item))}</div>
          ${item.c2pa?.ai_provenance ? "" : `
          <div class="metric-grid">
            <div class="metric"><span>Band</span><strong>${escapeHtml(item.result?.band || "—")}</strong></div>
            <div class="metric"><span>AI probability</span><strong>${percent(pAi)}</strong></div>
            <div class="metric"><span>Model</span><strong>${escapeHtml(item.result?.backend || "—")}</strong></div>
            <div class="metric"><span>Uncertainty</span><strong>${item.result?.uncertain ? "Yes" : "No"}</strong></div>
          </div>`}
          <div class="provenance">${escapeHtml(provenanceText(item))}</div>
          ${category === "synthetic" ? `
            ${item.heatmap?.overlay_png_b64
              ? `<img class="heatmap-img" src="data:image/png;base64,${item.heatmap.overlay_png_b64}" alt="AI-signal heatmap" />
                 <div class="heatmap-note">${escapeHtml(item.heatmap.caveat || "Regions most influencing the AI score; not proof of edits.")}</div>`
              : `<button class="secondary heatmap-btn" data-heatmap="${item.id}" type="button" ${item.heatmapLoading ? "disabled" : ""}>${item.heatmapLoading ? "Generating heatmap..." : "Show AI-signal heatmap"}</button>
                 ${item.heatmap?.error ? `<div class="error-text">${escapeHtml(item.heatmap.error)}</div>` : ""}`}
          ` : ""}
          ${item.feedbackOpen && !item.feedbackSent ? `
            <div class="feedback-box">
              <textarea class="feedback-input" data-feedback-text="${item.id}" rows="3" maxlength="2000" placeholder="Was this result right or wrong? Tell us in up to 100 words"></textarea>
              <div class="feedback-row">
                <span class="feedback-count">0/100 words</span>
                <button class="secondary feedback-submit" data-feedback-submit="${item.id}" type="button" ${item.feedbackSending ? "disabled" : ""}>${item.feedbackSending ? "Sending..." : "Submit"}</button>
              </div>
              ${item.feedbackError ? `<div class="error-text">${escapeHtml(item.feedbackError)}</div>` : ""}
            </div>` : ""}
          ${item.feedbackSent ? `<div class="feedback-thanks">Thank you for the feedback!</div>` : ""}
        ` : ""}

        ${item.status === "queued" ? `<div class="result-badge neutral">Ready for analysis</div>` : ""}
        ${item.status === "error" ? `<div class="error-text">${escapeHtml(item.error || "Analysis failed")}${item.quotaExceeded ? ` <a href="/pricing">View pricing</a>` : ""}</div>` : ""}
      </div>
    </article>
  `;
}

function renderSummary() {
  const done = state.items.filter((item) => item.status === "done");
  const queued = state.items.filter((item) => item.status === "queued");
  const analyzing = state.items.filter((item) => item.status === "analyzing");
  const errors = state.items.filter((item) => item.status === "error");
  const synthetic = done.filter((item) => effectiveCategory(item) === "synthetic").length;
  const dubious = done.filter((item) => effectiveCategory(item) === "dubious").length;
  const clean = done.filter((item) => effectiveCategory(item) === "clean").length;

  els.queueCount.textContent = String(queued.length + analyzing.length);
  els.completedCount.textContent = String(done.length + errors.length);
  els.totalCount.textContent = String(state.items.length);
  els.syntheticCount.textContent = String(synthetic);
  els.dubiousCount.textContent = String(dubious);
  els.cleanCount.textContent = String(clean);

  if (!state.items.length) {
    els.resultsSubtitle.textContent = "Upload one or more images to begin.";
  } else if (analyzing.length) {
    els.resultsSubtitle.textContent = `${done.length + errors.length} of ${state.items.length} complete.`;
  } else if (queued.length) {
    els.resultsSubtitle.textContent = `${queued.length} image${queued.length === 1 ? "" : "s"} waiting for analysis.`;
  } else {
    els.resultsSubtitle.textContent = "Batch complete.";
  }
}

function render() {
  renderSummary();
  els.emptyState.classList.toggle("hidden", state.items.length > 0);
  els.resultsGrid.innerHTML = state.items.map(renderCard).join("");
  els.analyzeButton.disabled = !signedIn() || state.running || !state.items.some((item) => item.status === "queued" || item.status === "error");
  els.clearButton.disabled = state.running || !state.items.length;
  els.includeC2pa.disabled = state.running;
  els.fileInput.disabled = state.running;

  els.resultsGrid.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => removeItem(Number(button.dataset.remove)));
  });
  els.resultsGrid.querySelectorAll("[data-heatmap]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = state.items.find((i) => i.id === Number(button.dataset.heatmap));
      if (item && !item.heatmapLoading) generateHeatmap(item);
    });
  });
  els.resultsGrid.querySelectorAll("[data-feedback]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = state.items.find((i) => i.id === Number(button.dataset.feedback));
      if (item) {
        item.feedbackOpen = !item.feedbackOpen;
        render();
      }
    });
  });
  els.resultsGrid.querySelectorAll("[data-feedback-submit]").forEach((button) => {
    button.addEventListener("click", () => submitFeedback(Number(button.dataset.feedbackSubmit)));
  });
  els.resultsGrid.querySelectorAll("[data-feedback-text]").forEach((ta) => {
    ta.addEventListener("input", () => {
      const count = ta.value.trim().split(/\s+/).filter(Boolean).length;
      const counter = ta.closest(".feedback-box")?.querySelector(".feedback-count");
      if (counter) counter.textContent = `${count}/100 words`;
    });
  });
}

async function submitFeedback(id) {
  const item = state.items.find((i) => i.id === id);
  if (!item) return;
  const ta = els.resultsGrid.querySelector(`[data-feedback-text="${id}"]`);
  const text = (ta?.value || "").trim();
  const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
  if (!text) {
    item.feedbackError = "Please enter some feedback";
    render();
    return;
  }
  if (words > 100) {
    item.feedbackError = "Feedback must be 100 words or fewer";
    render();
    return;
  }
  item.feedbackSending = true;
  item.feedbackError = null;
  try {
    await parseApiResponse(await fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        feedback: text,
        filename: item.name,
        result_label: resultLabel(item),
        result_band: item.result?.band || null,
        channel: "web_app",
      }),
    }));
    item.feedbackSent = true;
    item.feedbackOpen = false;
  } catch (e) {
    item.feedbackError = e.message || "Could not send feedback";
  } finally {
    item.feedbackSending = false;
  }
  render();
}

function removeItem(id) {
  const index = state.items.findIndex((item) => item.id === id);
  if (index < 0 || state.items[index].status === "analyzing") return;
  URL.revokeObjectURL(state.items[index].previewUrl);
  state.items.splice(index, 1);
  render();
}

function clearItems() {
  if (state.running) return;
  state.items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
  state.items = [];
  setStatusMessage("");
  render();
}

async function parseApiResponse(resp) {
  const data = await resp.json().catch(() => null);
  if (resp.status === 401) {
    signOut();
    throw new Error("Session expired. Please sign in again.");
  }
  if (!resp.ok || data?.status === "failure") {
    throw new Error(data?.error || `Request failed (${resp.status})`);
  }
  return data;
}

async function analyzeItem(item, includeC2pa) {
  item.status = "analyzing";
  item.error = "";
  render();

  try {
    const aiForm = new FormData();
    aiForm.append("file", item.file, item.name);
    const aiResp = await fetch(`${API_BASE}/api/ai-detect`, {
      method: "POST",
      headers: authHeaders(),
      body: aiForm,
    });
    if (aiResp.status === 402) {
      const data = await aiResp.json().catch(() => ({}));
      item.quotaExceeded = true;
      throw new Error(`Free limit reached — ${data.free_limit || 5} image analyses used. \u20B9${data.price_per_image_inr || 10}/image after.`);
    }
    item.result = await parseApiResponse(aiResp);

    if (includeC2pa) {
      try {
        const c2paForm = new FormData();
        c2paForm.append("file", item.file, item.name);
        item.c2pa = await parseApiResponse(await fetch(`${API_BASE}/api/c2pa`, {
          method: "POST",
          headers: authHeaders(),
          body: c2paForm,
        }));
      } catch (error) {
        item.c2pa = { error: error.message || "C2PA check failed" };
      }
    }

    item.status = "done";
  } catch (error) {
    item.status = "error";
    item.error = error.message || "Analysis failed";
  } finally {
    render();
    refreshUsage();
  }
}

async function refreshUsage() {
  if (!els.usageText) return;
  try {
    const u = await parseApiResponse(await fetch(`${API_BASE}/api/usage`, { headers: authHeaders() }));
    if (u.used === null || u.used === undefined) {
      els.usageText.textContent = "";
      return;
    }
    const left = Math.max(0, (u.free_limit || 5) - u.used);
    els.usageText.innerHTML = left > 0
      ? `Free analyses left: ${left} of ${u.free_limit} · <a href="/pricing">Pricing</a>`
      : `Free analyses used (${u.used}/${u.free_limit}) · <a href="/pricing">View pricing</a>`;
  } catch {
    els.usageText.textContent = "";
  }
}

async function generateHeatmap(item) {
  item.heatmapLoading = true;
  render();
  try {
    const form = new FormData();
    form.append("file", item.file, item.name);
    const data = await parseApiResponse(await fetch(`${API_BASE}/api/ai-heatmap`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    }));
    item.heatmap = data;
  } catch (error) {
    item.heatmap = { error: error.message || "Heatmap failed" };
  } finally {
    item.heatmapLoading = false;
    render();
  }
}

async function analyzeQueue() {
  if (!signedIn()) {
    setStatusMessage("Sign in before running analysis.");
    return;
  }

  const pending = state.items.filter((item) => item.status === "queued" || item.status === "error");
  if (!pending.length) return;

  state.running = true;
  pending.forEach((item) => {
    if (item.status === "error") item.status = "queued";
    item.error = "";
  });
  setStatusMessage(`Analyzing ${pending.length} image${pending.length === 1 ? "" : "s"}...`);
  render();

  const includeC2pa = els.includeC2pa.checked;
  const queue = [...pending];
  const workers = Array.from({ length: Math.min(MAX_CONCURRENT, queue.length) }, async () => {
    while (queue.length) {
      const item = queue.shift();
      await analyzeItem(item, includeC2pa);
    }
  });
  await Promise.all(workers);

  const completed = state.items.filter((item) => item.status === "done").length;
  const failed = state.items.filter((item) => item.status === "error").length;
  setStatusMessage(`Finished. ${completed} completed${failed ? `, ${failed} failed` : ""}.`);
  state.running = false;
  render();
  loadHistory();
}

function bindEvents() {
  els.browseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    els.fileInput.click();
  });

  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.fileInput.click();
    }
  });
  els.fileInput.addEventListener("change", (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  });

  ["dragenter", "dragover"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("dragging");
    });
  });
  els.dropZone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

  els.analyzeButton.addEventListener("click", analyzeQueue);
  els.clearButton.addEventListener("click", clearItems);
  els.refreshHistoryButton.addEventListener("click", loadHistory);
  els.applyHistoryFilter?.addEventListener("click", loadHistory);
  els.apiKeyButton.addEventListener("click", () => {
    const key = els.apiKeyInput.value.trim();
    if (!key) {
      setAuthMessage("Enter an API key.");
      return;
    }
    setApiKeySession(key);
    setAuthMessage("");
  });
  els.apiKeyInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") els.apiKeyButton.click();
  });
}

function restoreSession() {
  const storedJwt = localStorage.getItem(JWT_STORAGE_KEY) || "";
  const storedApiKey = sessionStorage.getItem(API_KEY_SESSION_KEY) || "";
  if (storedJwt && isJwtValid(storedJwt)) {
    const payload = decodeJwtPayload(storedJwt);
    state.jwt = storedJwt;
    state.email = payload?.email || localStorage.getItem(EMAIL_STORAGE_KEY) || "";
    state.name = payload?.name || "";
  } else if (storedJwt) {
    localStorage.removeItem(JWT_STORAGE_KEY);
    localStorage.removeItem(EMAIL_STORAGE_KEY);
  }
  if (!state.jwt && storedApiKey) {
    state.apiKey = storedApiKey;
    state.name = "API key user";
  }
}

bindEvents();
restoreSession();
renderAuth();
render();
renderHistory();
initGoogleSignIn();
loadHistory();
