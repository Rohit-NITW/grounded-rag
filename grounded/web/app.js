"use strict";
const $ = (id) => document.getElementById(id);
const state = { token: "", documents: [], selected: new Set(), answer: null, busy: false, configured: false };
const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status}`); }
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "Check the request and try again.";
    throw new Error(message);
  }
  return data;
}

function showError(error) { $("error").textContent = error.message; $("error").hidden = false; }
function updateScope() {
  $("scope-label").textContent = state.selected.size ? `Searching ${state.selected.size} selected document(s)` : "Searching all documents";
}

async function refresh() {
  const [config, data] = await Promise.all([api("/api/config"), api("/api/documents")]);
  if (!state.configured) {
    $("top-k").value = config.top_k;
    $("top-k-value").textContent = config.top_k;
    state.configured = true;
  }
  $("provider-badge").textContent = config.provider === "demo" ? "● Extractive preview" : `● ${config.provider} · ${config.model}`;
  $("mode-note").hidden = config.provider !== "demo";
  $("mode-note").textContent = "DEMO MODE · Showing matching excerpts with lexical hash vectors. Select Ollama or OpenAI in .env and re-index to use neural embeddings and LLM answers.";
  $("upload-hint").textContent = `PDF, TXT, MD · ${config.max_upload_mb} MB per file`;
  state.documents = data.documents;
  const existing = new Set(state.documents.map(d => d.id));
  state.selected.forEach(id => { if (!existing.has(id)) state.selected.delete(id); });
  $("doc-count").textContent = state.documents.length;
  const chunks = state.documents.reduce((sum, doc) => sum + doc.chunk_count, 0);
  $("index-status").textContent = `${state.documents.length} documents · ${chunks} passages`;
  $("document-list").replaceChildren();
  if (!state.documents.length) {
    $("document-list").append(element("p", "small muted", "No sources yet. Upload a file, or run the sample ingestion command in the README."));
  }
  state.documents.forEach(doc => {
    const row = element("div", "document");
    const checkbox = element("input"); checkbox.type = "checkbox"; checkbox.id = `doc-${doc.id}`;
    checkbox.checked = state.selected.has(doc.id);
    checkbox.addEventListener("change", () => { checkbox.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id); updateScope(); });
    const label = element("label"); label.htmlFor = checkbox.id; label.title = doc.filename;
    label.append(element("span", "document-name", doc.filename), element("span", "document-meta", `${doc.chunk_count} passages · ${(doc.size_bytes / 1024).toFixed(1)} KB`));
    const remove = element("button", "delete-button", "×"); remove.title = `Delete ${doc.filename}`; remove.setAttribute("aria-label", remove.title);
    remove.addEventListener("click", async () => {
      if (!confirm(`Remove ${doc.filename} and its passages from this index?`)) return;
      remove.disabled = true;
      try { await api(`/api/documents/${doc.id}`, { method: "DELETE" }); await refresh(); }
      catch (error) { showError(error); remove.disabled = false; }
    });
    row.append(checkbox, label, remove); $("document-list").append(row);
  });
  updateScope();
}

async function upload(files) {
  if (!files.length) return;
  $("error").hidden = true;
  $("file-input").disabled = true;
  const errors = [];
  let completed = 0;
  for (const file of files) {
    $("upload-status").textContent = `Indexing ${file.name}…`;
    const form = new FormData(); form.append("file", file);
    try { await api("/api/documents", { method: "POST", body: form }); completed++; }
    catch (error) { errors.push(`${file.name}: ${error.message}`); }
  }
  $("upload-status").textContent = `${completed} of ${files.length} file(s) processed.`;
  $("file-input").disabled = false; $("file-input").value = "";
  if (errors.length) showError(new Error(errors.join("\n")));
  try { await refresh(); } catch (error) { showError(error); }
}

function renderAnswer(result) {
  state.answer = result;
  $("empty-state").hidden = true; $("answer-panel").hidden = false;
  $("answer-label").textContent = result.insufficient_evidence ? "INSUFFICIENT EVIDENCE" : result.provider === "demo" ? "MATCHING EXCERPTS" : "GROUNDED ANSWER";
  $("answered-question").textContent = result.question;
  $("answer-content").replaceChildren();
  if (!result.claims.length) $("answer-content").append(element("p", "answer-paragraph", result.answer));
  result.claims.forEach(claim => {
    const paragraph = element("p", "answer-paragraph", claim.text);
    claim.source_ids.forEach(id => {
      const citation = element("button", "citation", id);
      citation.setAttribute("aria-label", `View source ${id}`);
      citation.addEventListener("click", () => { const card = $(`evidence-${id}`); card?.scrollIntoView({ behavior: "smooth", block: "center" }); card?.focus({ preventScroll: true }); });
      paragraph.append(citation);
    });
    $("answer-content").append(paragraph);
  });
  const timing = result.timings_ms;
  const metrics = [`${result.strategy} retrieval`, `${timing.retrieval} ms search`, `${timing.generation} ms generation`, result.model];
  if (result.usage.input_tokens) metrics.push(`${result.usage.input_tokens} input / ${result.usage.output_tokens} output tokens`);
  $("answer-metrics").replaceChildren(...metrics.map(value => element("span", "", value)));
  $("evidence-count").textContent = result.sources.length;
  $("evidence-list").replaceChildren();
  if (!result.sources.length) $("evidence-list").append(element("p", "small muted", "No passages passed the relevance threshold. Try selecting different documents or a more specific question."));
  result.sources.forEach(source => {
    const card = element("article", "evidence-card"); card.id = `evidence-${source.source_id}`; card.tabIndex = -1;
    const heading = element("h3"); heading.append(element("span", "", source.source_id), document.createTextNode(source.filename));
    const details = element("details"); const summary = element("summary", "", "Inspect retrieval scores");
    details.append(summary, element("div", "score-line", `Rank score ${source.score} · Cosine ${source.cosine} · BM25 ${source.bm25}. Scores are not probabilities.`));
    card.append(heading, element("span", "page", `Page ${source.page} · ${source.chunk_id.slice(0, 8)}`), element("p", "", source.text), details);
    $("evidence-list").append(card);
  });
}

$("question-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (state.busy) return;
  const question = $("question").value.trim(); if (question.length < 2) return;
  state.busy = true; $("ask-button").disabled = true; $("ask-button").textContent = "Finding evidence…"; $("error").hidden = true;
  try {
    const result = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, strategy: $("strategy").value, top_k: Number($("top-k").value), document_ids: [...state.selected] }) });
    renderAnswer(result);
  } catch (error) { showError(error); }
  finally { state.busy = false; $("ask-button").disabled = false; $("ask-button").textContent = "Ask Grounded ↗"; }
});
$("question").addEventListener("keydown", event => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("question-form").requestSubmit(); });
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => { $("question").value = button.dataset.question; $("question-form").requestSubmit(); }));
$("file-input").addEventListener("change", event => upload([...event.target.files]));
$("drop-zone").addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("file-input").click(); } });
$("drop-zone").addEventListener("dragover", event => { event.preventDefault(); $("drop-zone").classList.add("dragover"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragover"));
$("drop-zone").addEventListener("drop", event => { event.preventDefault(); $("drop-zone").classList.remove("dragover"); upload([...event.dataTransfer.files]); });
$("clear-filter").addEventListener("click", () => { state.selected.clear(); document.querySelectorAll(".document input").forEach(box => { box.checked = false; }); updateScope(); });
$("top-k").addEventListener("input", () => { $("top-k-value").textContent = $("top-k").value; });
$("toggle-settings").addEventListener("click", () => {
  const expanded = document.querySelector(".sidebar").classList.toggle("settings-open");
  $("toggle-settings").setAttribute("aria-expanded", String(expanded));
});
$("connect").addEventListener("click", async () => { state.token = $("access-token").value.trim(); $("error").hidden = true; try { await refresh(); } catch (error) { showError(error); } });
$("export-answer").addEventListener("click", () => {
  if (!state.answer) return;
  const r = state.answer;
  const content = `# ${r.question}\n\n${r.answer}\n\n## Sources\n\n${r.sources.map(s => `### [${s.source_id}] ${s.filename} — page ${s.page}\n\n${s.text}`).join("\n\n")}\n\nProvider: ${r.provider}; model: ${r.model}; strategy: ${r.strategy}\n`;
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown;charset=utf-8" }));
  const link = element("a"); link.href = url; link.download = "grounded-answer.md"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refresh().catch(error => { $("provider-badge").textContent = "Connection needed"; $("index-status").textContent = "Unable to load index"; document.querySelector(".access").open = true; showError(error); });
