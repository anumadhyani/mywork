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
  totalCount: $("totalCount"),
  syntheticCount: $("syntheticCount"),
  dubiousCount: $("dubiousCount"),
  cleanCount: $("cleanCount"),
  resultsSubtitle: $("resultsSubtitle"),
  emptyState: $("emptyState"),
  resultsGrid: $("resultsGrid"),
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
  renderAuth();
  render();
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

function resultSentence(item) {
  const result = item.result;
  if (!result) return "Waiting for analysis";
  const category = categoryForResult(result);
  if (category === "clean" || category === "dubious") return "No strong AI signals detected.";
  return result.prediction || result.label || "This image is likely AI-generated.";
}

function resultLabel(item) {
  const category = categoryForResult(item.result);
  if (category === "synthetic") return "Likely AI";
  if (category === "dubious") return "Dubious";
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
  if (!detailed || typeof detailed !== "object") return null;
  const manifestStore = detailed.manifest_store || detailed.manifestStore || null;
  const activeId = detailed.active_manifest || detailed.activeManifest || null;
  const active = manifestStore && activeId && manifestStore[activeId] ? manifestStore[activeId] : null;
  return active?.claim_generator || active?.claimGenerator || detailed.claim_generator || detailed.claimGenerator || null;
}

function provenanceText(item) {
  if (!els.includeC2pa.checked && !item.c2pa) return "Provenance check skipped";
  if (item.c2pa?.error) return "Provenance check unavailable";
  if (item.c2pa?.c2pa) {
    const origin = c2paOrigin(item.c2pa.c2pa);
    return origin ? `Source: ${origin}` : "No provenance credentials found";
  }
  return "Provenance pending";
}

function statusLabel(status) {
  if (status === "analyzing") return "Analyzing";
  if (status === "done") return "Done";
  if (status === "error") return "Error";
  return "Queued";
}

function renderCard(item) {
  const category = categoryForResult(item.result);
  const pAi = aiProbability(item);
  const canRemove = !state.running && item.status !== "analyzing";

  return `
    <article class="result-card" data-id="${item.id}">
      <div class="thumb-wrap">
        <img src="${item.previewUrl}" alt="${escapeHtml(item.name)}" loading="lazy" />
        <span class="status-badge ${item.status}">${statusLabel(item.status)}</span>
        ${canRemove ? `<button class="danger remove-card" data-remove="${item.id}" title="Remove image">×</button>` : ""}
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
          <div class="metric-grid">
            <div class="metric"><span>Band</span><strong>${escapeHtml(item.result?.band || "—")}</strong></div>
            <div class="metric"><span>AI probability</span><strong>${percent(pAi)}</strong></div>
            <div class="metric"><span>Model</span><strong>${escapeHtml(item.result?.backend || "—")}</strong></div>
            <div class="metric"><span>Uncertainty</span><strong>${item.result?.uncertain ? "Yes" : "No"}</strong></div>
          </div>
          <div class="provenance">${escapeHtml(provenanceText(item))}</div>
        ` : ""}

        ${item.status === "queued" ? `<div class="result-badge neutral">Ready for analysis</div>` : ""}
        ${item.status === "error" ? `<div class="error-text">${escapeHtml(item.error || "Analysis failed")}</div>` : ""}
      </div>
    </article>
  `;
}

function renderSummary() {
  const done = state.items.filter((item) => item.status === "done");
  const queued = state.items.filter((item) => item.status === "queued");
  const analyzing = state.items.filter((item) => item.status === "analyzing");
  const errors = state.items.filter((item) => item.status === "error");
  const synthetic = done.filter((item) => categoryForResult(item.result) === "synthetic").length;
  const dubious = done.filter((item) => categoryForResult(item.result) === "dubious").length;
  const clean = done.filter((item) => categoryForResult(item.result) === "clean").length;

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
    item.result = await parseApiResponse(await fetch(`${API_BASE}/api/ai-detect`, {
      method: "POST",
      headers: authHeaders(),
      body: aiForm,
    }));

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
initGoogleSignIn();
