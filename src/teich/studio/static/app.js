/* Teich Studio frontend */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function api(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch (e) { /* empty body */ }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

let toastTimer = null;
function toast(message, type = "") {
  const node = $("#toast");
  node.textContent = message;
  node.className = type;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3800);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function displayText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch (e) {
    return String(value);
  }
}

function escapeHtml(value) {
  return displayText(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeMarkdownUrl(value) {
  const url = displayText(value).trim();
  if (/^(https?:|mailto:|#|\/)/i.test(url)) return url;
  return "#";
}

function inlineMarkdown(value) {
  let text = escapeHtml(value);
  const tokens = [];
  const stash = (html) => {
    const token = `\u0000md${tokens.length}\u0000`;
    tokens.push(html);
    return token;
  };

  text = text.replace(/`([^`]+)`/g, (_match, code) => stash(`<code>${code}</code>`));
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
    const safeUrl = escapeHtml(safeMarkdownUrl(url.replace(/&amp;/g, "&")));
    return stash(`<a href="${safeUrl}" target="_blank" rel="noreferrer">${label}</a>`);
  });
  text = text
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/(^|[^\*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");

  tokens.forEach((html, index) => {
    text = text.replaceAll(`\u0000md${index}\u0000`, html);
  });
  return text;
}

function markdownToHtml(value) {
  const lines = displayText(value).replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let paragraph = [];
  let listType = null;
  let inFence = false;
  let fenceLines = [];
  let fenceLang = "";

  const flushParagraph = () => {
    if (!paragraph.length) return;
    output.push(`<p>${paragraph.map(inlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (!listType) return;
    output.push(`</${listType}>`);
    listType = null;
  };
  const openList = (type) => {
    if (listType === type) return;
    closeList();
    listType = type;
    output.push(`<${type}>`);
  };
  const flushCode = () => {
    const language = fenceLang ? ` class="language-${escapeHtml(fenceLang)}"` : "";
    output.push(`<pre><code${language}>${escapeHtml(fenceLines.join("\n"))}</code></pre>`);
    fenceLines = [];
    fenceLang = "";
  };

  for (const rawLine of lines) {
    const fence = rawLine.match(/^\s*```([A-Za-z0-9_-]*)\s*$/);
    if (fence) {
      if (inFence) {
        flushCode();
        inFence = false;
      } else {
        flushParagraph();
        closeList();
        inFence = true;
        fenceLang = fence[1] || "";
      }
      continue;
    }
    if (inFence) {
      fenceLines.push(rawLine);
      continue;
    }

    if (!rawLine.trim()) {
      flushParagraph();
      closeList();
      continue;
    }

    const heading = rawLine.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = Math.min(heading[1].length + 2, 6);
      output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = rawLine.match(/^\s*[-*+]\s+(.+)$/);
    if (unordered) {
      flushParagraph();
      openList("ul");
      output.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = rawLine.match(/^\s*\d+[.)]\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      openList("ol");
      output.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    const quote = rawLine.match(/^\s*>\s?(.+)$/);
    if (quote) {
      flushParagraph();
      closeList();
      output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    closeList();
    paragraph.push(rawLine.trimEnd());
  }

  if (inFence) flushCode();
  flushParagraph();
  closeList();
  return output.join("");
}

function markdownNode(className, value) {
  const node = el("div", `${className} markdown-body`);
  node.innerHTML = markdownToHtml(value);
  return node;
}

function truncateText(value, max = 120) {
  const text = displayText(value);
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function get(obj, path, fallback) {
  let current = obj;
  for (const key of path.split(".")) {
    if (current == null || typeof current !== "object") return fallback;
    current = current[key];
  }
  return current == null ? fallback : current;
}

function normalizeSessionProvider(provider) {
  const normalized = String(provider || "pi").trim().toLowerCase();
  if (["claude", "claude_code"].includes(normalized)) return "claude-code";
  if (["hermes-agent", "hermes_agent"].includes(normalized)) return "hermes";
  return normalized || "pi";
}

function syncSessionProviderFields() {
  const isChat = state.sessionProvider === "chat";
  $("#sess-system-field").hidden = !isChat;
  $("#sess-repo-field").hidden = isChat;
}

function syncSessionProviderFromConfig() {
  state.sessionProvider = normalizeSessionProvider(get(state.config, "agent.provider", "pi"));
  syncSessionProviderFields();
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

const state = {
  status: null,
  config: {},
  prompts: [],
  promptsDirty: false,
  session: null,
  sessionSource: null,
  sessionProvider: "pi",
  term: null,
  termSocket: null,
  jobSource: null,
  job: null,
  extractSource: null,
  extractJob: null,
  extractEvents: [],
  datasetPreview: null,
  selectedDatasetRow: null,
  datasetEditor: null,
  hfUploadModal: null,
};

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function showView(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  if (name === "generate") refreshRunSummary();
  if (name === "extract") loadCurrentExtraction();
  if (name === "dataset") loadDatasetPreview();
  if (name === "interactive" && state.term) requestAnimationFrame(fitTerminal);
}

$$(".nav-item").forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));
document.addEventListener("click", (event) => {
  const goto = event.target.closest("[data-goto]");
  if (goto) { event.preventDefault(); showView(goto.dataset.goto); }
});

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function loadStatus() {
  try {
    state.status = await api("GET", "/api/status");
  } catch (err) {
    toast(`Failed to load status: ${err.message}`, "error");
    return;
  }
  const s = state.status;
  const pathEl = $("#project-path");
  pathEl.textContent = s.project_dir;
  pathEl.title = s.project_dir;

  const docker = $("#status-docker");
  docker.classList.toggle("ok", s.docker.available);
  docker.classList.toggle("bad", !s.docker.available);
  docker.title = s.docker.available ? `Docker ${s.docker.detail}` : (s.docker.detail || "Docker unavailable");

  const key = $("#status-key");
  key.classList.toggle("ok", s.api_key_present);
  key.classList.toggle("bad", !s.api_key_present);
  key.title = s.api_key_present ? "An API key is configured" : "No API key found in config or environment";

  const badge = $("#nav-prompts-count");
  badge.hidden = !(s.prompts_count > 0);
  badge.textContent = s.prompts_count > 0 ? s.prompts_count : "";

  $("#api-key-hint").innerHTML = s.api_key_present
    ? "✓ An API key was found (config or environment)."
    : "Tip: leave empty and set <code>TEICH_API_KEY</code> / <code>OPENROUTER_API_KEY</code> in your environment instead.";
}

// ---------------------------------------------------------------------------
// Setup view
// ---------------------------------------------------------------------------

const PROVIDER_TAGS = { agent: "Docker", chat: "API only" };

function renderProviderCards() {
  const grid = $("#provider-grid");
  grid.innerHTML = "";
  const selected = get(state.config, "agent.provider", "pi");
  for (const provider of state.status.providers) {
    const card = el("div", "provider-card" + (provider.id === selected ? " selected" : ""));
    const title = el("h3", null, provider.label);
    title.appendChild(el("span", "provider-tag", PROVIDER_TAGS[provider.kind] || ""));
    card.appendChild(title);
    card.appendChild(el("p", null, provider.description));
    card.addEventListener("click", () => {
      state.config.agent = state.config.agent || {};
      state.config.agent.provider = provider.id;
      renderProviderCards();
      syncApiProviderOptions();
    });
    grid.appendChild(card);
  }
}

const API_DEFAULT_URLS = {
  openrouter: "https://openrouter.ai/api/v1",
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com",
};
const DIRECT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "").toLowerCase();
}

function syncApiProviderOptions() {
  const select = $("#cfg-api-provider");
  const anthropicOption = select.querySelector('option[value="anthropic"]');
  const agentProvider = String(get(state.config, "agent.provider", "pi")).trim().toLowerCase();
  const isChat = agentProvider === "chat";
  if (anthropicOption) {
    anthropicOption.disabled = isChat;
    anthropicOption.hidden = isChat;
  }
  if (isChat && select.value === "anthropic") {
    select.value = "openrouter";
    $("#cfg-base-url").value = API_DEFAULT_URLS.openrouter;
    delete select.dataset.customValue;
  }
}

function fillConfigForm() {
  const c = state.config;
  renderProviderCards();
  $("#cfg-model").value = get(c, "model.model", "");
  $("#cfg-reasoning").value = get(c, "model.reasoning_effort", "") || "";
  const apiProvider = get(c, "api.provider", "openrouter");
  const select = $("#cfg-api-provider");
  select.value = ["openrouter", "openai", "anthropic"].includes(apiProvider) ? apiProvider : "custom";
  if (select.value === "custom") select.dataset.customValue = apiProvider;
  $("#cfg-base-url").value = get(c, "api.base_url", "") || "";
  syncApiProviderOptions();
  $("#cfg-api-key").value = get(c, "api.api_key", "") || "";
  $("#cfg-concurrency").value = get(c, "max_concurrency", 1);
  $("#cfg-timeout").value = get(c, "timeout_seconds", 600);
  $("#cfg-traces-dir").value = get(c, "output.traces_dir", "./output");
  $("#extract-output").value = get(c, "output.traces_dir", "./output") || "./output";
  $("#dataset-path").value = get(c, "output.traces_dir", "./output") || "./output";
  $("#cfg-pretty-name").value = get(c, "output.pretty_name", "");
  $("#cfg-dev-instructions").value = get(c, "developer_instructions", "") || "";
  $("#cfg-repo-id").value = get(c, "publish.repo_id", "") || "";
  $("#cfg-private").checked = Boolean(get(c, "publish.private", false));
}

$("#cfg-api-provider").addEventListener("change", (event) => {
  const value = event.target.value;
  if (value !== "custom") $("#cfg-base-url").value = API_DEFAULT_URLS[value] || "";
  syncApiProviderOptions();
});

function collectConfigUpdates() {
  const apiProviderSelect = $("#cfg-api-provider");
  const apiProvider = apiProviderSelect.value === "custom"
    ? (apiProviderSelect.dataset.customValue || "openai")
    : apiProviderSelect.value;
  return {
    agent: { provider: get(state.config, "agent.provider", "pi") },
    model: {
      model: $("#cfg-model").value.trim(),
      reasoning_effort: $("#cfg-reasoning").value || null,
    },
    api: {
      provider: apiProvider,
      base_url: $("#cfg-base-url").value.trim() || null,
      api_key: $("#cfg-api-key").value.trim() || null,
    },
    output: {
      traces_dir: $("#cfg-traces-dir").value.trim() || "./output",
      pretty_name: $("#cfg-pretty-name").value.trim() || "My Agent Traces",
    },
    publish: {
      repo_id: $("#cfg-repo-id").value.trim() || null,
      private: $("#cfg-private").checked,
    },
    max_concurrency: Math.max(1, parseInt($("#cfg-concurrency").value, 10) || 1),
    timeout_seconds: Math.max(30, parseInt($("#cfg-timeout").value, 10) || 600),
    developer_instructions: $("#cfg-dev-instructions").value.trim() || null,
  };
}

function isDirectAnthropicChatConfig(updates) {
  const provider = String(updates.api.provider || "").trim().toLowerCase();
  const agentProvider = String(updates.agent.provider || "").trim().toLowerCase();
  return agentProvider === "chat"
    && (provider === "anthropic" || normalizeBaseUrl(updates.api.base_url) === DIRECT_ANTHROPIC_BASE_URL);
}

$("#btn-save-config").addEventListener("click", async () => {
  const note = $("#save-note");
  note.textContent = "";
  const updates = collectConfigUpdates();
  if (!updates.model.model) {
    toast("Please set a model ID first", "error");
    return;
  }
  if (isDirectAnthropicChatConfig(updates)) {
    toast("Chat runs need an OpenAI-compatible API. Use OpenRouter, OpenAI, or a compatible custom base URL.", "error");
    return;
  }
  try {
    const result = await api("PUT", "/api/config", { config: updates });
    state.config = result.config;
    syncSessionProviderFromConfig();
    note.textContent = "Saved to config.yaml ✓";
    note.classList.remove("error");
    setTimeout(() => { note.textContent = ""; }, 3000);
    loadStatus();
    renderProviderSeg();
  } catch (err) {
    note.textContent = err.message;
    note.classList.add("error");
  }
});

async function loadConfig() {
  try {
    const result = await api("GET", "/api/config");
    state.config = result.config || {};
    syncSessionProviderFromConfig();
    fillConfigForm();
  } catch (err) {
    toast(`Failed to load config: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Prompts view
// ---------------------------------------------------------------------------

function markPromptsDirty() {
  state.promptsDirty = true;
  $("#prompts-actions").hidden = false;
}

function renderPrompts() {
  const list = $("#prompts-list");
  list.innerHTML = "";
  $("#prompts-empty").hidden = state.prompts.length > 0;
  $("#prompts-actions").hidden = !state.promptsDirty && state.prompts.length === 0;

  state.prompts.forEach((prompt, index) => {
    const card = el("div", "prompt-card");
    const head = el("div", "prompt-head");
    head.appendChild(el("span", "prompt-index", `#${index + 1}`));
    const badges = el("div", "prompt-badges");
    if (prompt.system) badges.appendChild(el("span", "badge", "system"));
    if (prompt.github_repo) badges.appendChild(el("span", "badge", prompt.github_repo));
    if (prompt.follow_up_prompts && prompt.follow_up_prompts.length) {
      badges.appendChild(el("span", "badge", `${prompt.follow_up_prompts.length} follow-ups`));
    }
    head.appendChild(badges);
    head.appendChild(el("div", "grow"));

    const toggleBtn = el("button", "link-btn", "details");
    const deleteBtn = el("button", "link-btn danger", "delete");
    deleteBtn.addEventListener("click", () => {
      state.prompts.splice(index, 1);
      markPromptsDirty();
      renderPrompts();
    });
    head.appendChild(toggleBtn);
    head.appendChild(deleteBtn);
    card.appendChild(head);

    const textArea = el("textarea", "prompt-text");
    textArea.rows = 2;
    textArea.placeholder = "What should the agent do?";
    textArea.value = prompt.prompt || "";
    textArea.addEventListener("input", () => { prompt.prompt = textArea.value; markPromptsDirty(); });
    card.appendChild(textArea);

    const extra = el("div", "prompt-extra");
    extra.hidden = !(prompt.system || prompt.github_repo || (prompt.follow_up_prompts || []).length);

    const systemInput = el("input");
    systemInput.type = "text";
    systemInput.placeholder = "System prompt (optional)";
    systemInput.value = prompt.system || "";
    systemInput.addEventListener("input", () => {
      if (systemInput.value.trim()) prompt.system = systemInput.value; else delete prompt.system;
      markPromptsDirty();
    });

    const repoInput = el("input");
    repoInput.type = "text";
    repoInput.placeholder = "GitHub repo: owner/repo (optional)";
    repoInput.value = prompt.github_repo || "";
    repoInput.addEventListener("input", () => {
      if (repoInput.value.trim()) prompt.github_repo = repoInput.value.trim(); else delete prompt.github_repo;
      markPromptsDirty();
    });

    const followUps = el("textarea", "span-2");
    followUps.rows = 2;
    followUps.placeholder = "Follow-up prompts — one per line (optional)";
    followUps.value = (prompt.follow_up_prompts || []).join("\n");
    followUps.addEventListener("input", () => {
      const lines = followUps.value.split("\n").map((line) => line.trim()).filter(Boolean);
      if (lines.length) prompt.follow_up_prompts = lines; else delete prompt.follow_up_prompts;
      markPromptsDirty();
    });

    extra.appendChild(systemInput);
    extra.appendChild(repoInput);
    extra.appendChild(followUps);
    card.appendChild(extra);

    toggleBtn.addEventListener("click", () => { extra.hidden = !extra.hidden; });
    list.appendChild(card);
  });
}

$("#btn-add-prompt").addEventListener("click", () => {
  state.prompts.push({ prompt: "" });
  markPromptsDirty();
  renderPrompts();
  const areas = $$("#prompts-list .prompt-text");
  if (areas.length) areas[areas.length - 1].focus();
});

$("#btn-save-prompts").addEventListener("click", async () => {
  const note = $("#prompts-save-note");
  note.textContent = "";
  const rows = state.prompts.filter((p) => (p.prompt || "").trim());
  try {
    const result = await api("PUT", "/api/prompts", { prompts: rows });
    state.prompts = result.prompts;
    state.promptsDirty = false;
    renderPrompts();
    note.textContent = "Saved ✓";
    note.classList.remove("error");
    setTimeout(() => { note.textContent = ""; }, 3000);
    loadStatus();
  } catch (err) {
    note.textContent = err.message;
    note.classList.add("error");
  }
});

$("#btn-upload-prompts").addEventListener("click", () => $("#prompts-file-input").click());
$("#prompts-file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (!/\.(jsonl|ndjson)$/i.test(file.name)) {
    event.target.value = "";
    toast("Prompt uploads must be JSONL or NDJSON files.", "error");
    return;
  }
  const text = await file.text();
  event.target.value = "";
  try {
    const result = await api("POST", "/api/prompts/import", {
      text,
      replace: $("#upload-replace").checked,
      filename: file.name,
    });
    state.prompts = result.prompts;
    state.promptsDirty = false;
    renderPrompts();
    toast(`Imported — now ${state.prompts.length} prompts`, "success");
    loadStatus();
  } catch (err) {
    toast(`Import failed: ${err.message}`, "error");
  }
});

async function loadPrompts() {
  try {
    const result = await api("GET", "/api/prompts");
    state.prompts = result.prompts || [];
    $("#prompts-path").textContent = result.path || "";
    renderPrompts();
  } catch (err) {
    toast(`Failed to load prompts: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Generate view
// ---------------------------------------------------------------------------

function refreshRunSummary() {
  const summary = $("#run-summary");
  summary.innerHTML = "";
  const stats = [
    [get(state.config, "agent.provider", "—"), "Agent"],
    [get(state.config, "model.model", "—"), "Model"],
    [String(configuredPromptCount()), "Prompts"],
    [String(get(state.config, "max_concurrency", 1)), "Parallel"],
  ];
  for (const [value, label] of stats) {
    const stat = el("div", "stat");
    stat.appendChild(el("b", null, value));
    stat.appendChild(el("span", null, label));
    summary.appendChild(stat);
  }
}

function configuredPromptCount() {
  const statusCount = Number(state.status && state.status.prompts_count);
  if (Number.isFinite(statusCount) && statusCount >= 0) return statusCount;
  const inlinePrompts = get(state.config, "prompts", []);
  const inlineCount = Array.isArray(inlinePrompts) ? inlinePrompts.length : 0;
  return state.prompts.length + inlineCount;
}

function renderJob() {
  const job = state.job;
  const card = $("#run-progress-card");
  if (!job) { card.hidden = true; return; }
  card.hidden = false;

  const running = job.status === "running" || job.status === "starting";
  $("#btn-start-run").hidden = running;
  $("#btn-stop-run").hidden = !running;

  const statusEl = $("#run-status");
  statusEl.innerHTML = "";
  const chip = el("span", `chip ${job.status}`, job.status);
  if (running) chip.classList.add("pulsing");
  statusEl.appendChild(chip);
  statusEl.appendChild(el("span", null, job.message || job.error || ""));

  const rows = $("#run-rows");
  rows.innerHTML = "";
  for (const prompt of job.prompts || []) {
    const row = el("div", "run-row");
    row.appendChild(el("span", `chip ${prompt.status}`, prompt.status));
    row.appendChild(el("span", "prompt-preview", prompt.prompt_preview || ""));
    const metaParts = [];
    if (prompt.metrics && prompt.metrics.total_tokens != null) metaParts.push(`${prompt.metrics.total_tokens} tok`);
    if (prompt.metrics && prompt.metrics.total_cost != null) metaParts.push(`$${prompt.metrics.total_cost.toFixed(4)}`);
    if (prompt.trace) metaParts.push(prompt.trace);
    if (prompt.error) metaParts.push(prompt.error);
    row.appendChild(el("span", "meta", metaParts.join(" · ")));
    rows.appendChild(row);
  }
}

function connectJobEvents() {
  if (state.jobSource) state.jobSource.close();
  const source = new EventSource("/api/generate/events?after=0");
  state.jobSource = source;
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.kind === "job_status") {
      state.job = state.job || { prompts: [] };
      state.job.status = data.status;
      state.job.message = data.text || "";
      state.job.error = data.error || null;
      if (["completed", "failed", "stopped"].includes(data.status)) refreshDatasetIfVisible();
    } else if (data.kind === "prompt_update") {
      state.job = state.job || { prompts: [] };
      const prompts = state.job.prompts;
      const existing = prompts.findIndex((p) => p.prompt_id === data.prompt_id);
      if (existing >= 0) prompts[existing] = data; else prompts.push(data);
      prompts.sort((a, b) => (a.prompt_index || 0) - (b.prompt_index || 0));
    }
    renderJob();
  };
  source.addEventListener("end", () => source.close());
}

$("#btn-start-run").addEventListener("click", async () => {
  if (configuredPromptCount() < 1) {
    toast("Add some prompts first", "error");
    showView("prompts");
    return;
  }
  try {
    state.job = await api("POST", "/api/generate", { resume: $("#run-resume").checked });
    state.job.prompts = state.job.prompts || [];
    renderJob();
    connectJobEvents();
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#btn-stop-run").addEventListener("click", async () => {
  try {
    await api("POST", "/api/generate/stop");
    toast("Stopping run — completed traces are kept");
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadCurrentJob() {
  try {
    const result = await api("GET", "/api/generate");
    if (result.job) {
      state.job = result.job;
      renderJob();
      if (["running", "starting"].includes(result.job.status)) connectJobEvents();
    }
  } catch (err) { /* no job yet */ }
}

// ---------------------------------------------------------------------------
// Extract view
// ---------------------------------------------------------------------------

function renderExtractJob() {
  const card = $("#extract-progress-card");
  const events = $("#extract-events");
  const job = state.extractJob;
  if (!job && !state.extractEvents.length) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const running = job && ["starting", "running"].includes(job.status);
  $("#btn-start-extract").disabled = Boolean(running);

  const statusEl = $("#extract-status");
  statusEl.innerHTML = "";
  const status = job ? job.status : "idle";
  const chip = el("span", `chip ${status}`, status);
  if (running) chip.classList.add("pulsing");
  statusEl.appendChild(chip);
  statusEl.appendChild(el("span", null, (job && (job.message || job.error)) || ""));

  events.innerHTML = "";
  for (const eventData of state.extractEvents) {
    const row = el("div", `event-row ${eventData.kind || ""}`);
    row.appendChild(el("span", "event-kind", eventData.status || eventData.kind || "event"));
    row.appendChild(el("span", "event-text", eventData.text || eventData.error || ""));
    events.appendChild(row);
  }
  if (job && job.result_files && job.result_files.length) {
    const files = el("div", "source-list");
    files.appendChild(el("div", "source-list-title", "Extracted files"));
    files.appendChild(el("pre", null, job.result_files.join("\n")));
    events.appendChild(files);
  }
}

function connectExtractEvents() {
  if (state.extractSource) state.extractSource.close();
  const source = new EventSource("/api/extract/events?after=0");
  state.extractSource = source;
  state.extractEvents = [];
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    state.extractEvents.push(data);
    if (data.kind === "extract_status") {
      state.extractJob = state.extractJob || {};
      state.extractJob.status = data.status;
      state.extractJob.message = data.text || "";
      state.extractJob.error = data.error || null;
      if (data.result_files) state.extractJob.result_files = data.result_files;
      if (["completed", "failed"].includes(data.status)) {
        refreshDatasetIfVisible();
        if (data.status === "completed") toast("Extraction complete", "success");
      }
    }
    renderExtractJob();
  };
  source.addEventListener("end", () => source.close());
}

async function detectExtractSources({ apply = false } = {}) {
  const provider = $("#extract-provider").value;
  try {
    const result = await api("GET", `/api/extract/sources?provider=${encodeURIComponent(provider)}`);
    const sources = result.sources || [];
    $("#extract-source-hint").textContent = sources.length
      ? `${sources.length} default path${sources.length === 1 ? "" : "s"} found`
      : "No default paths found. Paste a .claude/.codex/.pi/.hermes folder, Cursor workspaceStorage/globalStorage path, or provider data path.";
    if (apply) $("#extract-sources").value = sources.join("\n");
    return sources;
  } catch (err) {
    $("#extract-source-hint").textContent = err.message;
    if (apply) toast(err.message, "error");
    return [];
  }
}

$("#extract-provider").addEventListener("change", () => detectExtractSources());
$("#btn-detect-sources").addEventListener("click", () => detectExtractSources({ apply: true }));

$("#btn-start-extract").addEventListener("click", async () => {
  let sources = $("#extract-sources").value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!sources.length) sources = await detectExtractSources({ apply: true });
  const body = {
    provider: $("#extract-provider").value,
    output: $("#extract-output").value.trim() || "./output",
    sessions_dirs: sources,
    model: $("#extract-model").value.trim() || null,
    skip_anonymize: $("#extract-no-anon").checked,
  };
  try {
    state.extractJob = await api("POST", "/api/extract", body);
    state.extractEvents = [];
    renderExtractJob();
    connectExtractEvents();
    loadConfig();
  } catch (err) {
    toast(err.message, "error");
  }
});

async function loadCurrentExtraction() {
  detectExtractSources();
  try {
    const result = await api("GET", "/api/extract");
    if (result.job) {
      state.extractJob = result.job;
      renderExtractJob();
      if (["running", "starting"].includes(result.job.status)) connectExtractEvents();
    }
  } catch (err) { /* no extraction yet */ }
}

// ---------------------------------------------------------------------------
// Interactive — session start
// ---------------------------------------------------------------------------

function renderProviderSeg() {
  const seg = $("#sess-provider-seg");
  seg.innerHTML = "";
  for (const provider of state.status.providers) {
    const button = el("button", provider.id === state.sessionProvider ? "selected" : "", provider.label);
    button.addEventListener("click", () => {
      state.sessionProvider = provider.id;
      renderProviderSeg();
    });
    seg.appendChild(button);
  }
  syncSessionProviderFields();
  $("#sess-model").placeholder = get(state.config, "model.model", "Use configured model");
}

$("#btn-start-session").addEventListener("click", async () => {
  const provider = state.sessionProvider;
  const body = {
    provider,
    model: $("#sess-model").value.trim() || null,
    github_repo: provider === "chat" ? null : $("#sess-repo").value.trim() || null,
    system: provider === "chat" ? $("#sess-system").value.trim() || null : null,
  };
  $("#sess-note").textContent = "Launching…";
  try {
    const session = await api("POST", "/api/sessions", body);
    state.session = session;
    $("#sess-note").textContent = "";
    openSessionView(session);
  } catch (err) {
    $("#sess-note").textContent = "";
    toast(err.message, "error");
  }
});

function openSessionView(session) {
  $("#session-start").hidden = true;
  $("#session-live").hidden = false;
  const isTerminal = session.mode === "terminal";
  $("#term-window").hidden = !isTerminal;
  $("#chat-window").hidden = isTerminal;
  $("#term-banner").hidden = true;
  $("#chat-messages").innerHTML = "";
  $("#term-title-text").textContent = `${session.provider} · ${session.model} — /workspace`;
  setSessionStatus(session.status, "");
  connectSessionEvents(session.id);
  if (isTerminal) mountTerminal(session);
}

function setSessionStatus(status, message) {
  $("#sess-status-dot").className = `status-dot ${status}`;
  $("#sess-status-text").textContent = message || status;
  if (state.session) state.session.status = status;
  const busy = ["saving"].includes(status);
  const chatRunning = state.session && state.session.mode === "chat" && status === "running";
  $("#btn-save-session").disabled = busy;
  $("#btn-discard-session").disabled = busy || chatRunning;
  // chat composer state
  const sendDisabled = ["running", "starting", "saving", "finished", "discarded", "error"].includes(status);
  const sendBtn = $("#btn-send");
  if (sendBtn) sendBtn.disabled = sendDisabled;
  showTyping(["running"].includes(status) && state.session && state.session.mode === "chat");
}

// ---------------------------------------------------------------------------
// Interactive — native terminal
// ---------------------------------------------------------------------------

function fitTerminal() {
  if (state.term && state.term._fit) {
    try { state.term._fit.fit(); } catch (e) { /* not attached yet */ }
  }
}

function mountTerminal(session) {
  disposeTerminal();
  const container = $("#terminal");
  container.innerHTML = "";
  const term = new Terminal({
    fontFamily: "'Cascadia Code', 'SF Mono', Consolas, monospace",
    fontSize: 13,
    lineHeight: 1.25,
    cursorBlink: true,
    scrollback: 8000,
    theme: {
      background: "#101013",
      foreground: "#e8e8ec",
      cursor: "#f25c1a",
      cursorAccent: "#101013",
      selectionBackground: "rgba(242, 92, 26, 0.35)",
      black: "#1c1c21",
      red: "#ef5a5f",
      green: "#3ecf8e",
      yellow: "#e7bb4a",
      blue: "#6cb2f7",
      magenta: "#c792ea",
      cyan: "#5fd7d7",
      white: "#d6d6dc",
      brightBlack: "#5b5b66",
      brightRed: "#ff8085",
      brightGreen: "#62e2a8",
      brightYellow: "#ffd479",
      brightBlue: "#94c9ff",
      brightMagenta: "#dfb3ff",
      brightCyan: "#8af0f0",
      brightWhite: "#ffffff",
    },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term._fit = fit;
  term.open(container);
  state.term = term;
  requestAnimationFrame(() => {
    fit.fit();
    connectTerminalSocket(session, term);
  });
  window.addEventListener("resize", fitTerminal);
  term.writeln("\x1b[38;5;208m⌁ teich studio\x1b[0m — starting container and launching \x1b[1m" + session.provider + "\x1b[0m …");
  term.writeln("");
}

function connectTerminalSocket(session, term) {
  const cols = term.cols || 120;
  const rows = term.rows || 32;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/sessions/${session.id}/term?cols=${cols}&rows=${rows}`);
  state.termSocket = socket;
  $("#term-conn").textContent = "connecting…";

  socket.onopen = () => { $("#term-conn").textContent = `${cols}×${rows} · connected`; };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "stdout") {
      term.write(message.data);
    } else if (message.type === "exit") {
      term.writeln(`\r\n\x1b[38;5;208m${message.detail || "Session ended."}\x1b[0m`);
    } else if (message.type === "status") {
      $("#term-conn").textContent = message.detail || "waiting...";
    }
  };
  socket.onclose = () => { $("#term-conn").textContent = "disconnected"; };
  socket.onerror = () => { $("#term-conn").textContent = "connection error"; };

  term.onData((data) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stdin", data }));
    }
  });
}

function disposeTerminal() {
  if (state.termSocket) { try { state.termSocket.close(); } catch (e) {} state.termSocket = null; }
  if (state.term) { try { state.term.dispose(); } catch (e) {} state.term = null; }
  window.removeEventListener("resize", fitTerminal);
}

function showTermBanner(text, kind) {
  const banner = $("#term-banner");
  banner.textContent = text;
  banner.className = `term-banner ${kind || ""}`;
  banner.hidden = false;
}

// ---------------------------------------------------------------------------
// Interactive — session events (SSE) + chat mode
// ---------------------------------------------------------------------------

function connectSessionEvents(sessionId) {
  if (state.sessionSource) state.sessionSource.close();
  const source = new EventSource(`/api/sessions/${sessionId}/events?after=0`);
  state.sessionSource = source;
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleSessionEvent(data);
  };
  source.addEventListener("end", () => source.close());
}

function handleSessionEvent(data) {
  const session = state.session;
  if (!session) return;
  if (data.kind === "session_status") {
    setSessionStatus(data.status, data.text || "");
    if (session.mode === "terminal") {
      if (data.status === "exited") showTermBanner(data.text || "Agent CLI exited — save the trace or discard the session.", "");
      if (data.status === "error") showTermBanner(data.text || "Session failed.", "error");
    } else if (data.text && ["starting", "ready"].includes(data.status)) {
      appendChatNode(el("div", "msg status-line", data.text));
    }
    return;
  }
  if (data.kind === "session_saved") {
    if (session.mode === "terminal") {
      showTermBanner(`Trace saved to your dataset: ${data.text}`, "success");
    } else {
      appendChatNode(el("div", "msg saved-line", `Trace saved: ${data.text}`));
    }
    return;
  }
  if (session.mode === "chat") {
    const node = renderDisplayEvent(data);
    if (node) appendChatNode(node);
  } else if (data.kind === "error") {
    showTermBanner(data.text || "Error", "error");
  }
}

function appendChatNode(node) {
  showTyping(false);
  $("#chat-messages").appendChild(node);
  if (state.session && state.session.status === "running") showTyping(true);
  scrollChat();
}

function showTyping(show) {
  let typing = $("#chat-messages .typing");
  if (show && !typing) {
    typing = el("div", "typing");
    typing.innerHTML = 'agent is working <span class="dots"><span>●</span><span>●</span><span>●</span></span>';
    $("#chat-messages").appendChild(typing);
    scrollChat();
  } else if (!show && typing) {
    typing.remove();
  }
}

function scrollChat() {
  const box = $("#chat-messages");
  box.scrollTop = box.scrollHeight;
}

function renderDisplayEvent(data) {
  const kind = data.kind;
  const text = displayText(data.text);
  const name = displayText(data.name);
  if (kind === "user" || kind === "assistant") {
    if (!text.trim()) return null;
    const msg = el("div", `msg ${kind}`);
    msg.appendChild(markdownNode("msg-body", text));
    return msg;
  }
  if (kind === "thinking" || kind === "tool_call" || kind === "tool_result") {
    if (!text.trim() && !name.trim()) return null;
    const msg = el("div", `msg block ${kind}`);
    const details = el("details");
    const summary = el("summary");
    if (kind === "thinking") {
      summary.appendChild(el("span", null, "💭 thinking"));
    } else if (kind === "tool_call") {
      summary.appendChild(el("span", null, "⚒"));
      summary.appendChild(el("span", "tool-name", name || "tool"));
    } else {
      summary.appendChild(el("span", null, "↳ result"));
      if (name) summary.appendChild(el("span", "tool-name", name));
    }
    const previewText = text.replace(/\s+/g, " ").slice(0, 90);
    summary.appendChild(el("span", "muted", previewText));
    details.appendChild(summary);
    details.appendChild(markdownNode("block-body", text));
    msg.appendChild(details);
    return msg;
  }
  if (kind === "system") {
    if (!text.trim()) return null;
    const msg = el("div", "msg block system");
    const details = el("details");
    const summary = el("summary");
    summary.appendChild(el("span", null, "system prompt"));
    details.appendChild(summary);
    details.appendChild(markdownNode("block-body", text));
    msg.appendChild(details);
    return msg;
  }
  if (kind === "status" || kind === "log") return el("div", "msg status-line", text);
  if (kind === "error") return el("div", "msg error-line", text);
  if (kind === "session_saved") return el("div", "msg saved-line", `Trace saved: ${text}`);
  return null;
}

async function sendMessage() {
  const input = $("#composer-input");
  const text = input.value.trim();
  if (!text || !state.session) return;
  try {
    await api("POST", `/api/sessions/${state.session.id}/message`, { text });
    input.value = "";
  } catch (err) {
    toast(err.message, "error");
  }
}

$("#btn-send").addEventListener("click", sendMessage);
$("#composer-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

// ---------------------------------------------------------------------------
// Interactive — save / discard
// ---------------------------------------------------------------------------

$("#btn-save-session").addEventListener("click", async () => {
  if (!state.session) return;
  if (state.session.mode === "terminal" && !confirm("Save this session as a trace? The agent CLI will be stopped and the trace exported to your dataset.")) return;
  try {
    const result = await api("POST", `/api/sessions/${state.session.id}/save`);
    toast(`Trace saved: ${result.trace}`, "success");
    setTimeout(resetSessionView, 2200);
  } catch (err) {
    toast(err.message, "error");
  }
});

$("#btn-discard-session").addEventListener("click", async () => {
  if (!state.session) return;
  if (!confirm("Discard this session? Nothing will be saved to your dataset.")) return;
  try {
    await api("POST", `/api/sessions/${state.session.id}/discard`);
  } catch (err) { /* already gone */ }
  resetSessionView();
});

function resetSessionView() {
  if (state.sessionSource) { state.sessionSource.close(); state.sessionSource = null; }
  disposeTerminal();
  state.session = null;
  $("#session-live").hidden = true;
  $("#session-start").hidden = false;
  refreshDatasetIfVisible();
}

function refreshDatasetIfVisible() {
  const view = $("#view-dataset");
  if (view && view.classList.contains("active")) loadDatasetPreview();
}

// ---------------------------------------------------------------------------
// Dataset preview
// ---------------------------------------------------------------------------

function featureColumns(preview) {
  const names = (preview.dataset.features || []).map((feature) => feature.name);
  const preferred = ["prompt", "response", "model", "messages", "tools", "metadata"];
  const columns = preferred.filter((name) => names.includes(name));
  for (const name of names) {
    if (!columns.includes(name) && columns.length < 7) columns.push(name);
  }
  return columns;
}

function summarizeDatasetCell(value, column, preview = {}) {
  if (column === "messages" && Array.isArray(value)) return `${preview.message_count || 0} messages`;
  if (column === "tools" && Array.isArray(value)) return `${preview.tool_count || 0} tools`;
  if (Array.isArray(value)) return `${value.length} items`;
  if (value && typeof value === "object") return truncateText(Object.keys(value).join(", "), 80);
  return truncateText(value == null ? "" : value, 110);
}

function datasetPathQuery() {
  const query = new URLSearchParams();
  const path = $("#dataset-path").value.trim();
  if (path) query.set("path", path);
  return query;
}

function datasetRowUrl(rowIdx) {
  const query = datasetPathQuery();
  const suffix = query.toString();
  return `/api/dataset-preview/rows/${rowIdx}${suffix ? `?${suffix}` : ""}`;
}

function messageContentText(content) {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return displayText(content);
}

function renderDatasetSummary(preview) {
  const summary = $("#dataset-summary");
  summary.innerHTML = "";
  const dataset = preview.dataset || {};
  const stats = [
    [String(dataset.num_rows || 0), "Rows"],
    [String(preview.files.length || 0), "JSONL files"],
    [String((dataset.features || []).length), "Columns"],
    [String(dataset.complete_rows || 0), "Complete"],
  ];
  for (const [value, label] of stats) {
    const stat = el("div", "stat");
    stat.appendChild(el("b", null, value));
    stat.appendChild(el("span", null, label));
    summary.appendChild(stat);
  }
  const root = el("div", "dataset-root");
  root.appendChild(el("span", "muted", preview.root));
  if (preview.notes && preview.notes.length) root.appendChild(el("p", null, preview.notes[0]));
  summary.appendChild(root);

  const embedCard = $("#hf-embed-card");
  if (preview.hf_embed_url) {
    embedCard.hidden = false;
    $("#hf-embed-link").href = preview.hf_embed_url;
    $("#hf-embed-link").textContent = preview.hf_embed_url;
    $("#hf-embed-frame").src = preview.hf_embed_url;
  } else {
    embedCard.hidden = true;
    $("#hf-embed-frame").removeAttribute("src");
  }
}

function renderDatasetTable(preview) {
  const container = $("#dataset-table");
  container.innerHTML = "";
  const rows = preview.dataset.rows || [];
  if (!rows.length) {
    const empty = el("div", "empty-state small");
    empty.appendChild(el("div", "empty-icon", "▦"));
    empty.appendChild(el("p", null, preview.errors && preview.errors.length ? preview.errors[0] : "No preview rows found."));
    container.appendChild(empty);
    return;
  }
  const columns = featureColumns(preview);
  const table = el("table", "dataset-table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(el("th", "idx-col", "#"));
  for (const column of columns) headRow.appendChild(el("th", null, column));
  headRow.appendChild(el("th", "actions-col", ""));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const rowInfo of rows) {
    const tr = document.createElement("tr");
    tr.className = rowInfo.row_idx === state.selectedDatasetRow ? "selected" : "";
    tr.appendChild(el("td", "mono idx-col", String(rowInfo.row_idx)));
    for (const column of columns) {
      const cell = el("td");
      cell.appendChild(el("span", "dataset-cell", summarizeDatasetCell(rowInfo.row[column], column, rowInfo.preview || {})));
      tr.appendChild(cell);
    }
    const actions = el("td", "dataset-row-actions");
    const actionWrap = el("div", "dataset-row-actions-inner");
    const editBtn = el("button", "icon-btn", "✎");
    editBtn.type = "button";
    editBtn.title = "Edit row";
    editBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      openDatasetEditor(rowInfo);
    });
    const deleteBtn = el("button", "icon-btn danger", "×");
    deleteBtn.type = "button";
    deleteBtn.title = "Delete row";
    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteDatasetRow(rowInfo);
    });
    actionWrap.append(editBtn, deleteBtn);
    actions.appendChild(actionWrap);
    tr.appendChild(actions);
    tr.addEventListener("click", () => {
      state.selectedDatasetRow = rowInfo.row_idx;
      renderDatasetTable(preview);
      renderDatasetDetail(rowInfo);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
  const selected = rows.find((rowInfo) => rowInfo.row_idx === state.selectedDatasetRow) || rows[0];
  if (selected) {
    state.selectedDatasetRow = selected.row_idx;
    renderDatasetDetail(selected);
  }
}

function renderTrainingMessages(row) {
  const messages = Array.isArray(row.messages) ? row.messages : [];
  const wrap = el("div", "dataset-message-list");
  for (const message of messages) {
    if (!message || typeof message !== "object") continue;
    const thinkingText = message.thinking || message.reasoning_content || "";
    if (thinkingText.trim()) {
      const thinking = renderDisplayEvent({ kind: "thinking", text: thinkingText });
      if (thinking) wrap.appendChild(thinking);
    }
    const role = message.role || "assistant";
    const content = messageContentText(message.content);
    const kind = role === "tool" ? "tool_result" : role;
    if (content.trim()) {
      const node = renderDisplayEvent({ kind, text: content, name: message.name || message.tool_call_id || "" });
      if (node) wrap.appendChild(node);
    }
    if (Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        const fn = toolCall && (toolCall.function || toolCall);
        const toolNode = renderDisplayEvent({
          kind: "tool_call",
          name: fn && fn.name ? fn.name : "tool",
          text: fn && fn.arguments ? fn.arguments : JSON.stringify(toolCall, null, 2),
        });
        if (toolNode) wrap.appendChild(toolNode);
      }
    }
  }
  return wrap;
}

function renderDatasetDetail(rowInfo) {
  const detail = $("#dataset-detail");
  detail.innerHTML = "";
  const head = el("div", "preview-head");
  head.appendChild(el("span", "badge", `row ${rowInfo.row_idx}`));
  const preview = rowInfo.preview || {};
  if (preview.trace_type) head.appendChild(el("span", "muted", preview.trace_type));
  if (preview.model) head.appendChild(el("span", "muted", preview.model));
  const actions = el("div", "dataset-detail-actions");
  const editBtn = el("button", "icon-btn", "✎");
  editBtn.type = "button";
  editBtn.title = "Edit row";
  editBtn.addEventListener("click", () => openDatasetEditor(rowInfo));
  const deleteBtn = el("button", "icon-btn danger", "×");
  deleteBtn.type = "button";
  deleteBtn.title = "Delete row";
  deleteBtn.addEventListener("click", () => deleteDatasetRow(rowInfo));
  actions.append(editBtn, deleteBtn);
  head.appendChild(actions);
  detail.appendChild(head);
  detail.appendChild(renderTrainingMessages(rowInfo.row));

  const raw = el("details", "json-details");
  const summary = el("summary", null, "Row JSON");
  raw.appendChild(summary);
  raw.appendChild(el("pre", null, JSON.stringify(rowInfo.row, null, 2)));
  detail.appendChild(raw);
}

async function deleteDatasetRow(rowInfo) {
  const label = rowInfo && rowInfo.preview && rowInfo.preview.prompt
    ? truncateText(rowInfo.preview.prompt, 80)
    : `row ${rowInfo.row_idx}`;
  if (!confirm(`Delete ${label}? This removes the backing trace row or file.`)) return;
  try {
    await api("DELETE", datasetRowUrl(rowInfo.row_idx));
    toast(`Deleted row ${rowInfo.row_idx}`, "success");
    state.selectedDatasetRow = null;
    await loadDatasetPreview();
  } catch (err) {
    toast(err.message, "error");
  }
}

function closeDatasetEditor() {
  if (!state.datasetEditor) return;
  document.removeEventListener("keydown", state.datasetEditor.onKeydown);
  state.datasetEditor.overlay.remove();
  state.datasetEditor = null;
}

function toolCallView(toolCall) {
  const fn = toolCall && typeof toolCall === "object" && toolCall.function && typeof toolCall.function === "object"
    ? toolCall.function
    : toolCall;
  if (!fn || typeof fn !== "object") return null;
  const rawKey = Object.prototype.hasOwnProperty.call(fn, "arguments") ? "arguments"
    : Object.prototype.hasOwnProperty.call(fn, "input") ? "input"
      : "arguments";
  return {
    fn,
    name: displayText(fn.name || fn.toolName || ""),
    rawKey,
    rawArgs: fn[rawKey],
  };
}

function parseToolArguments(rawArgs) {
  if (typeof rawArgs === "string") {
    try {
      const parsed = JSON.parse(rawArgs);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch (e) {
      return null;
    }
  }
  return rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs) ? { ...rawArgs } : null;
}

function setToolArguments(view, args) {
  view.fn[view.rawKey] = typeof view.rawArgs === "string" ? JSON.stringify(args) : args;
}

function argPath(args) {
  return displayText(args.path || args.file_path || args.filePath || args.resource || args.filename || "");
}

function writeContentKey(args) {
  for (const key of ["content", "file_text", "fileText", "text", "contents"]) {
    if (typeof args[key] === "string") return key;
  }
  return null;
}

function editSteps(args) {
  const rawSteps = Array.isArray(args.edits) ? args.edits : [args];
  const steps = [];
  for (const step of rawSteps) {
    if (!step || typeof step !== "object") continue;
    const oldText = step.old_string ?? step.old_str ?? step.oldString ?? step.search ?? step.find;
    const newText = step.new_string ?? step.new_str ?? step.newString ?? step.replace ?? step.replacement ?? "";
    if (typeof oldText === "string" && oldText) steps.push({ oldText, newText: displayText(newText) });
  }
  return steps;
}

function collectToolCalls(message) {
  return Array.isArray(message && message.tool_calls) ? message.tool_calls : [];
}

function applyWriteEditMerge(row) {
  const messages = Array.isArray(row.messages) ? row.messages : [];
  const writes = [];
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex += 1) {
    const message = messages[messageIndex];
    if (!message || message.role !== "assistant") continue;
    for (const toolCall of collectToolCalls(message)) {
      const view = toolCallView(toolCall);
      if (!view) continue;
      const args = parseToolArguments(view.rawArgs);
      if (!args) continue;
      const name = view.name.toLowerCase();
      const contentKey = writeContentKey(args);
      const path = argPath(args);
      const isWrite = contentKey && /write|create/.test(name);
      if (isWrite) {
        writes.push({ messageIndex, view, args, contentKey, path });
        continue;
      }
      const steps = editSteps(args);
      const isEdit = steps.length && (/edit|replace|str_replace/.test(name) || steps.length);
      if (!isEdit) continue;
      const target = [...writes].reverse().find((write) => !path || !write.path || write.path === path);
      if (!target) continue;
      let content = target.args[target.contentKey];
      let applied = 0;
      for (const step of steps) {
        if (!content.includes(step.oldText)) {
          applied = 0;
          break;
        }
        content = content.replace(step.oldText, step.newText);
        applied += 1;
      }
      if (!applied) continue;
      target.args[target.contentKey] = content;
      setToolArguments(target.view, target.args);

      let keepEnd = target.messageIndex + 1;
      while (messages[keepEnd] && messages[keepEnd].role === "tool") keepEnd += 1;
      let removeEnd = messageIndex + 1;
      while (messages[removeEnd] && messages[removeEnd].role === "tool") removeEnd += 1;
      if (removeEnd > keepEnd) messages.splice(keepEnd, removeEnd - keepEnd);
      return { applied, path: path || target.path || "matching file" };
    }
  }
  return null;
}

function cloneDatasetRow(row) {
  return JSON.parse(JSON.stringify(row || {}));
}

function editorMessages(row) {
  if (!Array.isArray(row.messages)) row.messages = [];
  return row.messages;
}

function editorMessageHasPayload(message) {
  if (!message || typeof message !== "object") return false;
  if (typeof message.content === "string" && message.content.trim()) return true;
  if (message.thinking || message.reasoning_content) return true;
  return Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
}

function cleanupEditorMessage(row, messageIndex) {
  const messages = editorMessages(row);
  const message = messages[messageIndex];
  if (message && message.role === "assistant" && !editorMessageHasPayload(message)) {
    messages.splice(messageIndex, 1);
  }
}

function editorDisplayForItem(item) {
  if (item.type === "thinking") {
    return { kind: "thinking", text: item.text };
  }
  if (item.type === "tool_call") {
    return { kind: "tool_call", name: item.name || "tool", text: item.text };
  }
  const kind = item.role === "tool" ? "tool_result" : item.role;
  return { kind, text: item.text, name: item.name || "" };
}

function editorItems(editor) {
  const items = [];
  const row = editor.row;
  const messages = editorMessages(row);
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex += 1) {
    const message = messages[messageIndex];
    if (!message || typeof message !== "object") continue;
    const pending = editor.pendingEdit && editor.pendingEdit.messageIndex === messageIndex ? editor.pendingEdit : null;
    const pendingThinking = pending && pending.type === "thinking";
    const pendingContent = pending && pending.type === "content";
    const thinkingField = message.thinking
      ? "thinking"
      : message.reasoning_content || pendingThinking
        ? "reasoning_content"
        : null;
    const thinkingText = thinkingField ? displayText(message[thinkingField]) : "";
    if (thinkingField && (thinkingText.trim() || pendingThinking)) {
      items.push({
        type: "thinking",
        label: "Thinking",
        text: thinkingText,
        messageIndex,
        field: thinkingField,
        pending: Boolean(pendingThinking),
      });
    }
    const content = messageContentText(message.content);
    if (content.trim() || pendingContent) {
      items.push({
        type: "content",
        label: message.role || "assistant",
        role: message.role || "assistant",
        text: content,
        messageIndex,
        field: "content",
        name: message.name || message.tool_call_id || "",
        pending: Boolean(pendingContent),
      });
    }
    if (Array.isArray(message.tool_calls)) {
      for (let toolCallIndex = 0; toolCallIndex < message.tool_calls.length; toolCallIndex += 1) {
        const toolCall = message.tool_calls[toolCallIndex];
        const view = toolCallView(toolCall);
        if (!view) continue;
        items.push({
          type: "tool_call",
          label: "Tool call",
          text: displayText(view.rawArgs),
          name: view.name || "tool",
          messageIndex,
          toolCallIndex,
        });
      }
    }
  }
  return items;
}

function setEditorItemValue(editor, item, value) {
  const messages = editorMessages(editor.row);
  const message = messages[item.messageIndex];
  if (!message) return true;
  if (item.type === "thinking") {
    const field = item.field || "reasoning_content";
    if (value.trim()) {
      message[field] = value;
    } else {
      delete message[field];
    }
    return true;
  }
  if (item.type === "content") {
    message.content = value;
    return true;
  }
  if (item.type === "tool_call") {
    const toolCall = collectToolCalls(message)[item.toolCallIndex];
    const view = toolCallView(toolCall);
    if (!view) return false;
    if (typeof view.rawArgs === "string") {
      view.fn[view.rawKey] = value;
      return true;
    }
    try {
      view.fn[view.rawKey] = JSON.parse(value);
      return true;
    } catch (err) {
      toast("Tool arguments must stay valid JSON.", "error");
      return false;
    }
  }
  return false;
}

function deleteEditorItem(editor, item) {
  const messages = editorMessages(editor.row);
  const message = messages[item.messageIndex];
  if (!message) return;
  if (item.type === "thinking") {
    delete message[item.field || "reasoning_content"];
    cleanupEditorMessage(editor.row, item.messageIndex);
  } else if (item.type === "content") {
    if (message.role === "assistant" && (message.thinking || message.reasoning_content || collectToolCalls(message).length)) {
      message.content = "";
      cleanupEditorMessage(editor.row, item.messageIndex);
    } else {
      messages.splice(item.messageIndex, 1);
    }
  } else if (item.type === "tool_call") {
    const toolCalls = collectToolCalls(message);
    const [removed] = toolCalls.splice(item.toolCallIndex, 1);
    const removedId = removed && (removed.id || removed.tool_call_id);
    const next = messages[item.messageIndex + 1];
    if (next && next.role === "tool" && (!removedId || next.tool_call_id === removedId || next.id === removedId)) {
      messages.splice(item.messageIndex + 1, 1);
    }
    cleanupEditorMessage(editor.row, item.messageIndex);
  }
  renderDatasetEditorList(editor);
}

function addEditorItem(editor, afterMessageIndex, type) {
  const message = type === "thinking"
    ? { role: "assistant", content: "", reasoning_content: "" }
    : { role: "assistant", content: "" };
  const messages = editorMessages(editor.row);
  const insertAt = Math.max(0, Math.min(afterMessageIndex + 1, messages.length));
  messages.splice(insertAt, 0, message);
  editor.pendingEdit = {
    messageIndex: insertAt,
    type,
    field: type === "thinking" ? "reasoning_content" : "content",
  };
  renderDatasetEditorList(editor);
}

function showEditorAddMenu(editor, item, button) {
  const existing = button.parentElement.querySelector(".editor-add-menu");
  if (existing) {
    existing.remove();
    return;
  }
  $$(".editor-add-menu", editor.list).forEach((node) => node.remove());
  const menu = el("div", "editor-add-menu");
  const contentBtn = el("button", null, "Assistant content");
  contentBtn.type = "button";
  contentBtn.addEventListener("click", () => addEditorItem(editor, item.messageIndex, "content"));
  const thinkingBtn = el("button", null, "Thinking");
  thinkingBtn.type = "button";
  thinkingBtn.addEventListener("click", () => addEditorItem(editor, item.messageIndex, "thinking"));
  menu.append(contentBtn, thinkingBtn);
  button.parentElement.appendChild(menu);
}

function startEditorItemEdit(editor, item, card) {
  const node = card.querySelector(".editor-item-rendered");
  node.innerHTML = "";
  const textarea = el("textarea", "editor-inline-textarea");
  textarea.value = item.text;
  textarea.spellcheck = false;
  node.appendChild(textarea);
  const controls = card.querySelector(".editor-hover-actions");
  controls.innerHTML = "";
  const save = el("button", "icon-btn", "✓");
  save.type = "button";
  save.title = "Save edit";
  save.addEventListener("click", () => {
    if (!setEditorItemValue(editor, item, textarea.value)) return;
    editor.pendingEdit = null;
    cleanupEditorMessage(editor.row, item.messageIndex);
    renderDatasetEditorList(editor);
  });
  const cancel = el("button", "icon-btn", "×");
  cancel.type = "button";
  cancel.title = "Cancel edit";
  cancel.addEventListener("click", () => {
    if (item.pending) cleanupEditorMessage(editor.row, item.messageIndex);
    editor.pendingEdit = null;
    renderDatasetEditorList(editor);
  });
  controls.append(save, cancel);
  textarea.focus();
}

function renderDatasetEditorList(editor) {
  editor.list.innerHTML = "";
  const items = editorItems(editor);
  if (!items.length) {
    const empty = el("div", "empty-state small");
    empty.appendChild(el("div", "empty-icon", "▦"));
    empty.appendChild(el("p", null, "No messages in this row."));
    const addContent = el("button", "btn small", "Add assistant content");
    addContent.type = "button";
    addContent.addEventListener("click", () => addEditorItem(editor, -1, "content"));
    const addThinking = el("button", "btn small", "Add thinking");
    addThinking.type = "button";
    addThinking.addEventListener("click", () => addEditorItem(editor, -1, "thinking"));
    const actions = el("div", "row-actions");
    actions.append(addContent, addThinking);
    empty.appendChild(actions);
    editor.list.appendChild(empty);
    return;
  }
  for (const item of items) {
    const card = el("div", "editor-message-item");
    const rendered = el("div", "editor-item-rendered");
    const node = renderDisplayEvent(editorDisplayForItem(item));
    if (node) rendered.appendChild(node);
    const controls = el("div", "editor-hover-actions");
    const editBtn = el("button", "icon-btn", "✎");
    editBtn.type = "button";
    editBtn.title = "Edit";
    editBtn.addEventListener("click", () => startEditorItemEdit(editor, item, card));
    const deleteBtn = el("button", "icon-btn danger", "×");
    deleteBtn.type = "button";
    deleteBtn.title = "Delete";
    deleteBtn.addEventListener("click", () => deleteEditorItem(editor, item));
    const addBtn = el("button", "icon-btn", "+");
    addBtn.type = "button";
    addBtn.title = "Add after";
    addBtn.addEventListener("click", () => showEditorAddMenu(editor, item, addBtn));
    controls.append(editBtn, deleteBtn, addBtn);
    card.append(rendered, controls);
    editor.list.appendChild(card);
    if (item.pending) startEditorItemEdit(editor, item, card);
  }
}

function openDatasetEditor(rowInfo) {
  closeDatasetEditor();

  const editor = {
    row: cloneDatasetRow(rowInfo.row),
    rowInfo,
    overlay: null,
    onKeydown: null,
    list: null,
    pendingEdit: null,
  };
  const overlay = el("div", "modal-backdrop");
  editor.overlay = overlay;
  const modal = el("div", "dataset-editor");
  const head = el("div", "dataset-editor-head");
  const title = el("div", "dataset-editor-title");
  title.appendChild(el("span", "badge", `row ${rowInfo.row_idx}`));
  if (rowInfo.preview && rowInfo.preview.trace_type) title.appendChild(el("span", "muted", rowInfo.preview.trace_type));
  head.appendChild(title);
  const closeBtn = el("button", "icon-btn", "×");
  closeBtn.type = "button";
  closeBtn.title = "Close";
  closeBtn.addEventListener("click", closeDatasetEditor);
  head.appendChild(closeBtn);

  const body = el("div", "dataset-editor-body");
  const list = el("div", "dataset-editor-preview editable");
  editor.list = list;
  body.appendChild(list);

  const actions = el("div", "dataset-editor-actions");
  const mergeBtn = el("button", "btn small", "Merge write/edit");
  mergeBtn.type = "button";
  mergeBtn.addEventListener("click", () => {
    const result = applyWriteEditMerge(editor.row);
    if (!result) {
      toast("No mergeable write/edit pair found.", "error");
      return;
    }
    renderDatasetEditorList(editor);
    toast(`Merged ${result.applied} edit into ${result.path}`, "success");
  });
  const saveBtn = el("button", "btn primary", "Save row");
  saveBtn.type = "button";
  saveBtn.addEventListener("click", async () => {
    try {
      await api("PUT", datasetRowUrl(rowInfo.row_idx), { row: editor.row });
      toast(`Saved row ${rowInfo.row_idx}`, "success");
      closeDatasetEditor();
      await loadDatasetPreview({ selectRow: rowInfo.row_idx });
    } catch (err) {
      toast(err.message, "error");
    }
  });
  const cancelBtn = el("button", "btn", "Cancel");
  cancelBtn.type = "button";
  cancelBtn.addEventListener("click", closeDatasetEditor);
  actions.append(mergeBtn, el("div", "toolbar-spacer"), cancelBtn, saveBtn);

  modal.append(head, body, actions);
  overlay.appendChild(modal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeDatasetEditor();
  });
  const onKeydown = (event) => {
    if (event.key === "Escape") closeDatasetEditor();
  };
  editor.onKeydown = onKeydown;
  document.addEventListener("keydown", onKeydown);
  document.body.appendChild(overlay);
  state.datasetEditor = editor;
  renderDatasetEditorList(editor);
}

function closeHfUploadModal() {
  if (!state.hfUploadModal) return;
  document.removeEventListener("keydown", state.hfUploadModal.onKeydown);
  state.hfUploadModal.overlay.remove();
  state.hfUploadModal = null;
}

function openHfUploadModal() {
  closeHfUploadModal();
  const preview = state.datasetPreview || {};
  const overlay = el("div", "modal-backdrop");
  const modal = el("div", "dataset-upload-modal");
  const head = el("div", "dataset-editor-head");
  const title = el("div", "dataset-editor-title");
  title.appendChild(el("span", "badge", "Hugging Face"));
  if (preview.root) title.appendChild(el("span", "muted", preview.root));
  head.appendChild(title);
  const closeBtn = el("button", "icon-btn", "×");
  closeBtn.type = "button";
  closeBtn.title = "Close";
  closeBtn.addEventListener("click", closeHfUploadModal);
  head.appendChild(closeBtn);

  const body = el("div", "dataset-upload-body");
  const form = el("div", "form-grid");
  const repoField = el("label", "field");
  repoField.appendChild(el("span", null, "Repo ID"));
  const repoInput = document.createElement("input");
  repoInput.type = "text";
  repoInput.id = "hf-upload-repo-id";
  repoInput.placeholder = "username/my-dataset";
  repoInput.value = preview.repo_id || get(state.config, "publish.repo_id", "") || "";
  repoField.appendChild(repoInput);
  const tokenField = el("label", "field");
  tokenField.appendChild(el("span", null, "HF_TOKEN optional"));
  const tokenInput = document.createElement("input");
  tokenInput.type = "password";
  tokenInput.id = "hf-upload-token";
  tokenInput.placeholder = "Leave blank to use environment";
  tokenInput.autocomplete = "off";
  tokenField.appendChild(tokenInput);
  form.append(repoField, tokenField);
  body.appendChild(form);

  const actions = el("div", "dataset-editor-actions");
  const note = el("span", "save-note");
  const cancelBtn = el("button", "btn", "Cancel");
  cancelBtn.type = "button";
  cancelBtn.addEventListener("click", closeHfUploadModal);
  const uploadBtn = el("button", "btn primary", "Upload");
  uploadBtn.type = "button";
  uploadBtn.addEventListener("click", async () => {
    const repoId = repoInput.value.trim();
    if (!repoId) {
      note.textContent = "Repo ID is required.";
      note.classList.add("error");
      return;
    }
    note.textContent = "Uploading…";
    note.classList.remove("error");
    uploadBtn.disabled = true;
    try {
      const result = await api("POST", "/api/dataset-preview/upload", {
        path: $("#dataset-path").value.trim() || null,
        repo_id: repoId,
        hf_token: tokenInput.value.trim() || null,
      });
      toast(`Uploaded dataset: ${result.repo_id}`, "success");
      closeHfUploadModal();
      await loadConfig();
      await loadDatasetPreview();
    } catch (err) {
      note.textContent = err.message;
      note.classList.add("error");
      uploadBtn.disabled = false;
    }
  });
  actions.append(note, el("div", "toolbar-spacer"), cancelBtn, uploadBtn);

  modal.append(head, body, actions);
  overlay.appendChild(modal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeHfUploadModal();
  });
  const onKeydown = (event) => {
    if (event.key === "Escape") closeHfUploadModal();
    if (event.key === "Enter" && event.target === repoInput) uploadBtn.click();
  };
  state.hfUploadModal = { overlay, onKeydown };
  document.addEventListener("keydown", onKeydown);
  document.body.appendChild(overlay);
  repoInput.focus();
}

async function loadDatasetPreview(options = {}) {
  const path = $("#dataset-path").value.trim();
  const search = $("#dataset-search").value.trim();
  const query = new URLSearchParams();
  if (path) query.set("path", path);
  if (search) query.set("search", search);
  try {
    const preview = await api("GET", `/api/dataset-preview?${query.toString()}`);
    state.datasetPreview = preview;
    state.selectedDatasetRow = options.selectRow ?? null;
    renderDatasetSummary(preview);
    renderDatasetTable(preview);
  } catch (err) {
    toast(err.message, "error");
    $("#dataset-table").innerHTML = "";
    $("#dataset-detail").innerHTML = "";
  }
}

$("#btn-load-dataset").addEventListener("click", loadDatasetPreview);
$("#btn-upload-dataset").addEventListener("click", openHfUploadModal);
$("#dataset-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadDatasetPreview();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  await loadStatus();
  await loadConfig();
  await loadPrompts();
  syncSessionProviderFromConfig();
  renderProviderSeg();
  refreshRunSummary();
  loadCurrentJob();
  setInterval(loadStatus, 60000);
}

init();
