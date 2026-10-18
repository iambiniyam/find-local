const form = document.getElementById("search-form");
const input = document.getElementById("q");
const clearButton = document.getElementById("clear");
const submitButton = document.getElementById("submit");
const resultsShell = document.getElementById("results-shell");
const resultsTitle = document.getElementById("results-title");
const results = document.getElementById("results");
const resultMeta = document.getElementById("result-meta");
const warmNote = document.getElementById("warm-note");
const resultsHead = document.querySelector(".results-head");
const toast = document.getElementById("toast");
const chips = Array.from(document.querySelectorAll(".chip"));
const filterDetails = document.getElementById("filter-more");
const filterBadge = document.getElementById("active-filters");
const resetFilters = document.getElementById("reset-filters");
const filterInputs = {
  since: document.getElementById("since"),
  until: document.getElementById("until"),
  folder: document.getElementById("folder"),
  min_size: document.getElementById("min-size"),
  max_size: document.getElementById("max-size"),
};
const FILTER_KEYS = Object.keys(filterInputs);

const state = {
  kind: "",
  controller: null,
  stats: null,
  lastQuery: "",
};

const EXAMPLES = [
  "invoice from March",
  "a leaf with orange spots",
  "offline privacy notes",
  "that error screenshot",
];

const ICONS = {
  text: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"></path><path d="M14 3v6h5"></path><path d="M10 13h7M10 17h5"></path></svg>',
  pdf: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"></path><path d="M14 3v6h5"></path><path d="M9 17v-4h1.6a1.4 1.4 0 0 1 0 2.8H9"></path></svg>',
  image: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"></rect><circle cx="9" cy="10" r="1.6"></circle><path d="m4 18 5-5 4 4 3-3 4 4"></path></svg>',
};

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatBytes(bytes) {
  if (!bytes || bytes < 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

function splitPath(path) {
  const clean = String(path || "");
  const index = clean.lastIndexOf("/");
  if (index === -1) return { name: clean, dir: "" };
  return { name: clean.slice(index + 1), dir: clean.slice(0, index) };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlight(text, query) {
  const terms = Array.from(
    new Set(
      query
        .toLowerCase()
        .split(/\s+/)
        .map((term) => term.replace(/[^\p{L}\p{N}]+/gu, ""))
        .filter((term) => term.length > 2)
    )
  );
  if (!terms.length) return escapeHtml(text);

  const pattern = new RegExp(`\\b(${terms.map(escapeRegExp).join("|")})\\b`, "gi");
  let output = "";
  let cursor = 0;
  for (const match of String(text).matchAll(pattern)) {
    const index = match.index ?? 0;
    output += escapeHtml(text.slice(cursor, index));
    output += `<mark>${escapeHtml(match[0])}</mark>`;
    cursor = index + match[0].length;
  }
  output += escapeHtml(text.slice(cursor));
  return output;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
}

function setBusy(busy) {
  resultsShell.setAttribute("aria-busy", busy ? "true" : "false");
}

function setActiveChip(button) {
  chips.forEach((chip) => {
    const active = chip === button;
    chip.classList.toggle("is-active", active);
    chip.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function collectFilters() {
  const values = {};
  FILTER_KEYS.forEach((key) => {
    const field = filterInputs[key];
    const value = field ? field.value.trim() : "";
    if (value) values[key] = value;
  });
  return values;
}

function updateFilterBadge() {
  if (!filterBadge) return;
  const count = Object.keys(collectFilters()).length;
  filterBadge.hidden = count === 0;
  filterBadge.textContent = count === 1 ? "1 active" : `${count} active`;
}

function renderSkeleton() {
  resultsHead.hidden = false;
  resultMeta.textContent = "Searching";
  results.innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
}

function renderError(message) {
  resultsHead.hidden = false;
  resultMeta.textContent = "";
  results.innerHTML = `
    <div class="error" role="alert">
      <h3>Search failed</h3>
      <p>${escapeHtml(message || "The local server did not answer.")}</p>
      <button class="btn btn-primary" type="button" id="retry">Try again</button>
    </div>`;
  const retry = document.getElementById("retry");
  if (retry) retry.addEventListener("click", () => search({ focus: true }));
}

function renderInitial() {
  if (resultsHead) resultsHead.hidden = true;
  const stats = state.stats;
  const empty = !stats || stats.files === 0;
  const title = empty ? "Your index is empty" : "Start with a description";
  const body = empty
    ? "Index a folder first. It stays on this machine."
    : "Try a phrase that describes the content, not the file name.";
  const examples = EXAMPLES.map(
    (example) => `<button class="example" type="button" data-example="${escapeHtml(example)}">${escapeHtml(example)}</button>`
  ).join("");
  const command = empty
    ? `<div class="command"><code>find-local index ~/Documents</code><button class="btn-ghost" type="button" data-copy="find-local index ~/Documents">Copy</button></div>`
    : "";
  results.innerHTML = `
    <div class="empty">
      <h3>${title}</h3>
      <p>${body}</p>
      ${examples ? `<div class="examples">${examples}</div>` : ""}
      ${command}
    </div>`;
  resultMeta.textContent = stats
    ? `${stats.files} files, ${stats.chunks} text chunks, ${stats.images} images indexed`
    : "";
}

function renderResults(data, query) {
  resultsHead.hidden = false;
  const count = data.count || 0;
  const took = typeof data.took_ms === "number" ? data.took_ms : 0;
  resultMeta.textContent = count
    ? `${count} result${count === 1 ? "" : "s"} in ${took} ms`
    : `No results in ${took} ms`;

  if (!count) {
    results.innerHTML = `
      <div class="empty">
        <h3>Nothing matched</h3>
        <p>Try a shorter phrase, remove a filter, or index another folder.</p>
        <div class="command"><code>find-local index ~/Documents</code><button class="btn-ghost" type="button" data-copy="find-local index ~/Documents">Copy</button></div>
      </div>`;
    return;
  }

  const maxScore = Math.max(...data.results.map((item) => Number(item.score) || 0), 0.0001);
  results.innerHTML = data.results
    .map((item) => {
      const { name, dir } = splitPath(item.path);
      const kind = item.kind === "image" ? "image" : item.kind === "pdf" ? "pdf" : "text";
      const label = kind === "pdf" ? "PDF" : kind === "image" ? "Image" : "Text";
      const thumb =
        kind === "image"
          ? `<img class="result-thumb" src="/thumb?path=${encodeURIComponent(item.path)}" alt="" loading="lazy">`
          : `<span class="result-icon" aria-hidden="true">${ICONS[kind]}</span>`;
      const score = Number(item.score) || 0;
      const width = Math.max(4, Math.round((score / maxScore) * 100));
      const size = formatBytes(item.size);
      return `
        <article class="result${kind === "image" ? " result--image" : ""}">
          ${thumb}
          <div class="result-body">
            <h3 class="result-title"><span>${escapeHtml(name)}</span><span class="kind">${label}</span></h3>
            <p class="result-path" title="${escapeHtml(item.path)}">${escapeHtml(dir)}${size ? ` · ${size}` : ""}</p>
            ${item.snippet ? `<p class="snippet">${highlight(item.snippet, query)}</p>` : ""}
          </div>
          <div class="result-side">
            <div class="score" aria-label="Relevance ${score.toFixed(2)}">
              <span class="score-value">${score.toFixed(2)}</span>
              <span class="score-track"><span class="score-fill" style="width:${width}%"></span></span>
            </div>
            <button class="btn-ghost" type="button" data-copy="${escapeHtml(item.path)}">Copy path</button>
          </div>
        </article>`;
    })
    .join("");
}

async function search({ focus = false } = {}) {
  const query = input.value.trim();
  state.lastQuery = query;
  clearButton.hidden = !query;

  if (!query) {
    renderInitial();
    updateUrl();
    return;
  }

  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  setBusy(true);
  renderSkeleton();

  const params = new URLSearchParams({ q: query, limit: "12" });
  if (state.kind) params.set("kind", state.kind);
  const filters = collectFilters();
  FILTER_KEYS.forEach((key) => {
    if (filters[key]) params.set(key, filters[key]);
  });

  try {
    const response = await fetch(`/api/search?${params.toString()}`, {
      signal: state.controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Server returned ${response.status}`);
    }
    const data = await response.json();
    state.stats = data.stats || state.stats;
    renderResults(data, query);
    if (focus) resultsTitle.focus();
    updateUrl();
  } catch (error) {
    if (error.name !== "AbortError") renderError(error.message);
  } finally {
    if (!state.controller.signal.aborted) setBusy(false);
  }
}

function updateUrl() {
  const params = new URLSearchParams(window.location.search);
  if (input.value.trim()) params.set("q", input.value.trim());
  else params.delete("q");
  if (state.kind) params.set("kind", state.kind);
  else params.delete("kind");
  const filters = collectFilters();
  FILTER_KEYS.forEach((key) => {
    if (filters[key]) params.set(key, filters[key]);
    else params.delete(key);
  });
  const next = `${window.location.pathname}${params.toString() ? `?${params}` : ""}`;
  window.history.replaceState(null, "", next);
}

function debounce(callback, wait) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), wait);
  };
}

const debouncedSearch = debounce(() => {
  if (input.value.trim().length >= 2) search();
}, 320);

function onFilterChange() {
  updateFilterBadge();
  if (input.value.trim()) search();
  else updateUrl();
}

const debouncedFilterChange = debounce(onFilterChange, 350);

Object.values(filterInputs).forEach((field) => {
  if (!field) return;
  field.addEventListener("input", () => {
    updateFilterBadge();
    debouncedFilterChange();
  });
  field.addEventListener("change", onFilterChange);
});

if (resetFilters) {
  resetFilters.addEventListener("click", () => {
    Object.values(filterInputs).forEach((field) => {
      if (field) field.value = "";
    });
    updateFilterBadge();
    if (input.value.trim()) search();
    else updateUrl();
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  search({ focus: true });
});

input.addEventListener("input", () => {
  clearButton.hidden = !input.value;
  if (!input.value.trim()) {
    renderInitial();
    updateUrl();
    return;
  }
  debouncedSearch();
});

clearButton.addEventListener("click", () => {
  input.value = "";
  clearButton.hidden = true;
  state.lastQuery = "";
  renderInitial();
  updateUrl();
  input.focus();
});

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    setActiveChip(chip);
    state.kind = chip.dataset.kind || "";
    if (input.value.trim()) search();
  });
});

document.addEventListener("click", (event) => {
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    const value = copy.getAttribute("data-copy");
    navigator.clipboard
      .writeText(value)
      .then(() => showToast("Copied to clipboard"))
      .catch(() => showToast("Could not copy"));
    return;
  }
  const example = event.target.closest("[data-example]");
  if (example) {
    input.value = example.getAttribute("data-example");
    clearButton.hidden = false;
    search({ focus: true });
  }
});

document.addEventListener("keydown", (event) => {
  const tag = document.activeElement ? document.activeElement.tagName : "";
  const typing = tag === "INPUT" || tag === "TEXTAREA" || document.activeElement.isContentEditable;
  if (event.key === "/" && !typing) {
    event.preventDefault();
    input.focus();
    input.select();
  }
  if (event.key === "Escape" && document.activeElement === input) {
    input.value = "";
    clearButton.hidden = true;
    renderInitial();
    updateUrl();
  }
});

async function waitForWarm() {
  while (true) {
    try {
      const response = await fetch("/api/status");
      if (!response.ok) break;
      const payload = await response.json();
      state.stats = payload.stats || state.stats;
      if (!payload.warming) {
        if (warmNote) warmNote.hidden = true;
        if (!input.value.trim()) renderInitial();
        return;
      }
    } catch (error) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  if (warmNote) warmNote.hidden = true;
}

async function start() {
  try {
    const response = await fetch("/api/status");
    if (response.ok) {
      const payload = await response.json();
      state.stats = payload.stats || null;
      if (payload.warming && warmNote) warmNote.hidden = false;
    }
  } catch (error) {
    state.stats = null;
  }
  waitForWarm();

  const params = new URLSearchParams(window.location.search);
  const initial = params.get("q");
  const kind = params.get("kind");
  if (kind && ["text", "pdf", "image"].includes(kind)) {
    state.kind = kind;
    const chip = chips.find((item) => item.dataset.kind === kind);
    if (chip) setActiveChip(chip);
  }
  FILTER_KEYS.forEach((key) => {
    const value = params.get(key);
    const field = filterInputs[key];
    if (!value || !field) return;
    if ((key === "since" || key === "until") && !/^\d{4}-\d{2}-\d{2}$/.test(value)) return;
    field.value = value;
  });
  updateFilterBadge();
  if (filterDetails && Object.keys(collectFilters()).length) filterDetails.open = true;
  if (initial) {
    input.value = initial;
    clearButton.hidden = false;
    search();
  } else {
    renderInitial();
  }
}

start();
