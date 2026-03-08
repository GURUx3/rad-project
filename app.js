/**
 * RADIUS.NLP v11.0
 * PURE TEXT CLASSIFICATION ENGINE
 */

const app = {
  config: {
    apiBase: "http://127.0.0.1:8000",
    storageThreadsKey: "radius_nlp_threads_v3",
    storageActiveThreadKey: "radius_nlp_active_thread_v3",
  },

  state: {
    isProcessing: false,
    threads: [],
    activeThreadId: null,
  },

  init() {
    console.log("Radiology copilot ready");
    this.setupEventListeners();
    this.loadPersistedState();
    this.renderThreadList();
    this.renderActiveThread();
    this.checkApiHealth();

    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    }
  },

  setupEventListeners() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const newThreadBtn = document.getElementById("btn-new-thread");
    const threadList = document.getElementById("thread-list");
    const findingsText = document.getElementById("findings-text");

    if (dropZone && fileInput) {
      dropZone.onclick = () => fileInput.click();

      fileInput.onchange = (e) => {
        const file = e.target.files[0];
        if (file) this.handleFile(file);
      };

      dropZone.ondragover = (e) => {
        e.preventDefault();
        dropZone.classList.add("is-dragging");
      };
      dropZone.ondragleave = () => {
        dropZone.classList.remove("is-dragging");
      };
      dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove("is-dragging");
        const file = e.dataTransfer.files[0];
        if (file) this.handleFile(file);
      };
    }

    if (newThreadBtn) {
      newThreadBtn.onclick = () => {
        this.state.activeThreadId = null;
        this.persistState();
        this.renderThreadList();
        this.renderActiveThread();
        findingsText.focus();
      };
    }

    if (threadList) {
      threadList.onclick = (event) => {
        const trigger = event.target.closest("[data-thread-id]");
        if (!trigger) return;
        const threadId = trigger.getAttribute("data-thread-id");
        if (!threadId) return;
        this.state.activeThreadId = threadId;
        this.persistState();
        this.renderThreadList();
        this.renderActiveThread();
      };
    }

    if (findingsText) {
      findingsText.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
          event.preventDefault();
          this.execute();
        }
      });
    }
  },

  loadPersistedState() {
    try {
      const rawThreads = localStorage.getItem(this.config.storageThreadsKey);
      const rawActiveId = localStorage.getItem(
        this.config.storageActiveThreadKey,
      );

      this.state.threads = rawThreads ? JSON.parse(rawThreads) : [];
      this.state.threads = Array.isArray(this.state.threads)
        ? this.state.threads
        : [];

      if (
        rawActiveId &&
        this.state.threads.some((thread) => thread.id === rawActiveId)
      ) {
        this.state.activeThreadId = rawActiveId;
      } else if (this.state.threads.length > 0) {
        this.state.activeThreadId = this.state.threads[0].id;
      }
    } catch (error) {
      console.warn("Could not load persisted threads", error);
      this.state.threads = [];
      this.state.activeThreadId = null;
    }
  },

  persistState() {
    try {
      localStorage.setItem(
        this.config.storageThreadsKey,
        JSON.stringify(this.state.threads),
      );
      localStorage.setItem(
        this.config.storageActiveThreadKey,
        this.state.activeThreadId || "",
      );
    } catch (error) {
      console.warn("Could not persist threads", error);
    }
  },

  async checkApiHealth() {
    const pill = document.getElementById("connection-pill");
    if (!pill) return;

    try {
      const response = await fetch(`${this.config.apiBase}/health`);
      const data = await response.json();
      if (!response.ok || data.engine !== "ML_READY") {
        pill.textContent = "API Degraded";
        pill.className = "connection-pill bad";
        return;
      }

      pill.textContent = "API Online";
      pill.className = "connection-pill ok";
    } catch (error) {
      pill.textContent = "API Offline";
      pill.className = "connection-pill bad";
    }
  },

  async handleFile(file) {
    const fileLabel = document.getElementById("file-label");
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      fileLabel.innerText = "Reading PDF...";
      try {
        const text = await this.extractTextFromPDF(file);
        document.getElementById("findings-text").value = text;
        fileLabel.innerText = `Loaded: ${file.name}`;
      } catch (err) {
        alert("Could not read this PDF file.");
        fileLabel.innerText = "Upload .txt or .pdf";
      }
    } else if (file.type === "text/plain" || file.name.endsWith(".txt")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById("findings-text").value = e.target.result;
        fileLabel.innerText = `Loaded: ${file.name}`;
      };
      reader.readAsText(file);
    }
  },

  async extractTextFromPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer })
      .promise;
    let fullText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      fullText += content.items.map((item) => item.str).join(" ") + "\n";
    }
    return fullText.trim();
  },

  async execute() {
    if (this.state.isProcessing) return;

    const findings = (
      document.getElementById("findings-text").value || ""
    ).trim();
    if (!findings) {
      alert("Please enter report text before analyzing.");
      return;
    }

    const payload = {
      findings: findings,
      patient_name: "N/A",
      patient_dob: "N/A",
      patient_id: "REF-UNKNOWN",
      referring_physician: "N/A",
      hospital_name: "CityCare Medical Center",
    };

    this.setLoading(true);

    try {
      const response = await fetch(`${this.config.apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let message = "Unable to reach analysis API";
        try {
          const errorBody = await response.json();
          if (errorBody && errorBody.detail) {
            message = errorBody.detail;
          }
        } catch (e) {}
        throw new Error(message);
      }

      const report = await response.json();
      this.saveReportIntoThread(findings, report);

      document.getElementById("findings-text").value = "";
      document.getElementById("file-label").innerText = "Upload .txt or .pdf";
    } catch (error) {
      console.error(error);
      alert("Analysis failed: " + error.message);
    } finally {
      this.setLoading(false);
    }
  },

  setLoading(active) {
    this.state.isProcessing = active;
    const btn = document.getElementById("btn-execute");
    btn.disabled = active;
    btn.innerText = active ? "Analyzing…" : "Analyze Report";
  },

  saveReportIntoThread(findings, report) {
    const nowIso = new Date().toISOString();
    const patientThreadKey =
      report.patient_thread_key || this.fallbackPatientKey(report, findings);

    let thread = this.state.threads.find(
      (item) => item.patientThreadKey === patientThreadKey,
    );
    if (!thread) {
      thread = {
        id: this.makeId("thread"),
        patientThreadKey: patientThreadKey,
        patientDisplayName:
          report.patient_display_name ||
          report.patient_name ||
          "Unidentified Patient",
        patientId: report.patient_id || "REF-UNKNOWN",
        patientDob: report.patient_dob || "N/A",
        identitySource: report.patient_identity_source || "unknown",
        hospitalName: report.hospital_name || "Unknown facility",
        createdAt: nowIso,
        updatedAt: nowIso,
        messages: [],
      };
      this.state.threads.unshift(thread);
    }

    thread.patientDisplayName =
      report.patient_display_name ||
      thread.patientDisplayName ||
      "Unidentified Patient";
    thread.patientId = report.patient_id || thread.patientId || "REF-UNKNOWN";
    thread.patientDob = report.patient_dob || thread.patientDob || "N/A";
    thread.identitySource =
      report.patient_identity_source || thread.identitySource || "unknown";
    thread.hospitalName =
      report.hospital_name || thread.hospitalName || "Unknown facility";
    thread.updatedAt = nowIso;

    thread.messages.push({
      id: this.makeId("msg"),
      role: "user",
      content: findings,
      timestamp: nowIso,
    });
    thread.messages.push({
      id: this.makeId("msg"),
      role: "assistant",
      report: report,
      timestamp: report.timestamp || nowIso,
    });
    thread.messages = thread.messages.slice(-80);

    this.state.threads = this.state.threads
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
      .slice(0, 120);

    this.state.activeThreadId = thread.id;
    this.persistState();
    this.renderThreadList();
    this.renderActiveThread();
  },

  renderThreadList() {
    const threadList = document.getElementById("thread-list");
    threadList.innerHTML = "";

    if (!this.state.threads.length) {
      const empty = document.createElement("div");
      empty.className = "thread-empty";
      empty.textContent = "No patient threads yet.";
      threadList.appendChild(empty);
      return;
    }

    this.state.threads.forEach((thread) => {
      const reportCount = thread.messages.filter(
        (m) => m.role === "assistant",
      ).length;
      const lastAssistant = [...thread.messages]
        .reverse()
        .find((m) => m.role === "assistant");
      const lastDiagnosis = lastAssistant?.report?.diagnosis || "No result";
      const lastTime = this.formatTimestamp(thread.updatedAt);

      const button = document.createElement("button");
      button.className = `thread-item${thread.id === this.state.activeThreadId ? " active" : ""}`;
      button.setAttribute("data-thread-id", thread.id);
      button.innerHTML = `
                <div class="thread-name">${this.escapeHtml(thread.patientDisplayName || "Unidentified Patient")}</div>
                <div class="thread-sub">ID: ${this.escapeHtml(thread.patientId || "REF-UNKNOWN")} · DOB: ${this.escapeHtml(thread.patientDob || "N/A")}</div>
                <div class="thread-meta">${reportCount} report${reportCount === 1 ? "" : "s"} · ${this.escapeHtml(lastDiagnosis)} · ${this.escapeHtml(lastTime)}</div>
            `;
      threadList.appendChild(button);
    });
  },

  renderActiveThread() {
    const chatFeed = document.getElementById("chat-feed");
    const title = document.getElementById("active-thread-title");
    const meta = document.getElementById("active-thread-meta");

    chatFeed.innerHTML = "";

    if (!this.state.activeThreadId) {
      title.textContent = "New Case";
      meta.textContent =
        "Paste a report and run analysis. Patient identity is auto-detected.";
      chatFeed.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M9 12h6M9 16h6M9 8h3M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>
                        </svg>
                    </div>
                    <h3>Ready for a new report</h3>
                    <p>Submit any radiology text. Threads are created and grouped by extracted patient identity.</p>
                </div>
            `;
      return;
    }

    const thread = this.state.threads.find(
      (t) => t.id === this.state.activeThreadId,
    );
    if (!thread) {
      this.state.activeThreadId = null;
      this.persistState();
      this.renderThreadList();
      this.renderActiveThread();
      return;
    }

    title.textContent = thread.patientDisplayName || "Unidentified Patient";
    meta.textContent = `ID: ${thread.patientId || "—"} · DOB: ${thread.patientDob || "—"} · Source: ${thread.identitySource || "—"} · ${thread.hospitalName || "—"}`;

    thread.messages.forEach((msg) => {
      if (msg.role === "user")
        chatFeed.appendChild(this.renderUserMessage(msg));
      else if (msg.role === "assistant")
        chatFeed.appendChild(this.renderAssistantMessage(msg.report || {}));
    });

    chatFeed.scrollTop = chatFeed.scrollHeight;
  },

  renderUserMessage(message) {
    const row = document.createElement("div");
    row.className = "message-row user";
    const bubble = document.createElement("article");
    bubble.className = "user-bubble";
    bubble.innerHTML = `
            <div class="user-label">Report Input · ${this.escapeHtml(this.formatTimestamp(message.timestamp))}</div>
            <div class="user-text">${this.escapeHtml(message.content || "")}</div>
        `;
    row.appendChild(bubble);
    return row;
  },

  renderAssistantMessage(report) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const card = document.createElement("article");
    card.className = "assistant-card";

    const probabilityMap = report.probability_distribution || {};
    const probabilityEntries = Object.entries(probabilityMap)
      .map(([label, value]) => ({ label, value: this.toNumber(value) }))
      .sort((a, b) => b.value - a.value);

    const diagnosis = report.diagnosis || "--";
    const severity = report.severity || "--";
    const triage = report.triage_level || "--";
    const quality = `${this.toNumber(report.quality_score).toFixed(2)}%`;
    const latency = `${this.toNumber(report.latency_ms).toFixed(0)} ms`;
    const tokenCount = report.findings_tokens ?? "--";

    const abnormal =
      probabilityEntries.find((e) => e.label.toLowerCase() === "abnormal")
        ?.value || 0;
    const normal =
      probabilityEntries.find((e) => e.label.toLowerCase() === "normal")
        ?.value || 0;

    const regionsHtml = this.renderInlineTags(report.affected_regions, "tag");
    const conditionsHtml = this.renderInlineTags(
      report.suspected_conditions,
      "tag",
    );
    const measurementsHtml = this.renderSimpleList(
      report.measurements,
      "No measurements extracted.",
    );
    const keyFindingsHtml = this.renderSimpleList(
      report.key_findings,
      "No key findings extracted.",
    );
    const recommendationsHtml = this.renderSimpleList(
      report.recommendations,
      "No recommendations generated.",
    );
    const codingLines = this.asList(report.coding)
      .map((e) => {
        if (!e || typeof e !== "object") return "";
        return `${e.code || "--"} (${e.system || "--"}) — ${e.description || "--"}`;
      })
      .filter(Boolean);
    const codingHtml = this.renderSimpleList(
      codingLines,
      "No coding suggestions.",
    );
    const negatedHtml = this.renderInlineTags(
      report.negated_findings,
      "tag warn",
    );
    const regionalPairs = this.renderKvPairs(report.regional_involvement, "%");
    const flagsHtml = this.renderSimpleList(
      report.quality_flags,
      "No quality flags.",
    );

    const probabilityExtra = probabilityEntries.length
      ? `<div class="kv">${probabilityEntries
          .map(
            (e) =>
              `<div class="kv-row"><span>${this.escapeHtml(e.label)}</span><span>${(e.value * 100).toFixed(2)}%</span></div>`,
          )
          .join("")}</div>`
      : `<p class="muted">No probability distribution available.</p>`;

    card.innerHTML = `
            <div class="assistant-head">
                <div>
                    <div class="assistant-title">Clinical Analysis Report</div>
                    <div class="assistant-sub">
                        ${this.escapeHtml(report.patient_display_name || report.patient_name || "Unidentified Patient")}
                        · Identity: ${this.escapeHtml(report.patient_identity_source || "unknown")}
                    </div>
                </div>
                <div class="assistant-id">
                    ${this.escapeHtml(report.execution_id || "--")}<br>
                    ${this.escapeHtml(this.formatTimestamp(report.timestamp))}
                </div>
            </div>

            <div class="card-body">
                <div class="diagnosis-band">
                    <div>
                        <div class="diagnosis-main">${this.escapeHtml(diagnosis)}</div>
                        <div class="chip-row">
                            <span class="${this.chipClassForValue(severity)}">Severity: ${this.escapeHtml(severity)}</span>
                            <span class="${this.chipClassForValue(triage)}">Triage: ${this.escapeHtml(triage)}</span>
                            <span class="chip">Area: ${this.escapeHtml(report.affected_area || "--")}</span>
                        </div>
                    </div>
                </div>

                <div class="probability-grid">
                    <div class="prob-line">
                        <div class="prob-meta"><span>Abnormal</span><span>${(abnormal * 100).toFixed(2)}%</span></div>
                        <div class="prob-track"><div class="prob-fill" style="width:${(abnormal * 100).toFixed(2)}%"></div></div>
                    </div>
                    <div class="prob-line">
                        <div class="prob-meta"><span>Normal</span><span>${(normal * 100).toFixed(2)}%</span></div>
                        <div class="prob-track"><div class="prob-fill" style="width:${(normal * 100).toFixed(2)}%"></div></div>
                    </div>
                </div>

                <div class="mini-stats">
                    <div class="stat"><span class="stat-key">Latency</span><span class="stat-val">${this.escapeHtml(latency)}</span></div>
                    <div class="stat"><span class="stat-key">Quality</span><span class="stat-val">${this.escapeHtml(quality)}</span></div>
                    <div class="stat"><span class="stat-key">Tokens</span><span class="stat-val">${this.escapeHtml(String(tokenCount))}</span></div>
                    <div class="stat"><span class="stat-key">Facility</span><span class="stat-val">${this.escapeHtml(report.hospital_name || "--")}</span></div>
                </div>

                <div class="report-grid">
                    <section class="panel">
                        <h4>Probability Map</h4>
                        ${probabilityExtra}
                    </section>
                    <section class="panel">
                        <h4>Detected Regions</h4>
                        <div class="inline-tags">${regionsHtml}</div>
                        <div class="kv" style="margin-top:8px">${regionalPairs}</div>
                    </section>
                    <section class="panel">
                        <h4>Suspected Conditions</h4>
                        <div class="inline-tags">${conditionsHtml}</div>
                        <h4 style="margin-top:10px">Measurements</h4>
                        ${measurementsHtml}
                    </section>
                    <section class="panel">
                        <h4>Context</h4>
                        <div class="kv">
                            <div class="kv-row"><span>Indication</span><span>${this.escapeHtml(report.indication || "--")}</span></div>
                            <div class="kv-row"><span>Technique</span><span>${this.escapeHtml(report.technique || "--")}</span></div>
                            <div class="kv-row"><span>Modality</span><span>${this.escapeHtml(this.asList(report.modality).join(", ") || "--")}</span></div>
                            <div class="kv-row"><span>Laterality</span><span>${this.escapeHtml(this.asList(report.laterality).join(", ") || "--")}</span></div>
                        </div>
                    </section>
                    <section class="panel">
                        <h4>Key Findings</h4>
                        ${keyFindingsHtml}
                    </section>
                    <section class="panel">
                        <h4>Recommendations</h4>
                        ${recommendationsHtml}
                    </section>
                    <section class="panel">
                        <h4>Coding Suggestions</h4>
                        ${codingHtml}
                    </section>
                    <section class="panel">
                        <h4>Negated Findings</h4>
                        <div class="inline-tags">${negatedHtml}</div>
                        <h4 style="margin-top:10px">Quality Flags</h4>
                        ${flagsHtml}
                    </section>
                </div>
            </div>
        `;

    row.appendChild(card);
    return row;
  },

  renderSimpleList(items, fallback) {
    const values = this.asList(items);
    if (!values.length)
      return `<p class="muted">${this.escapeHtml(fallback)}</p>`;
    return `<ul class="text-list">${values.map((v) => `<li>${this.escapeHtml(String(v))}</li>`).join("")}</ul>`;
  },

  renderInlineTags(items, className) {
    const values = this.asList(items);
    if (!values.length) return `<span class="muted">None</span>`;
    return values
      .map(
        (v) =>
          `<span class="${className}">${this.escapeHtml(String(v))}</span>`,
      )
      .join("");
  },

  renderKvPairs(obj, suffix) {
    const entries = Object.entries(obj || {});
    if (!entries.length)
      return `<div class="kv-row"><span>No values</span><span>—</span></div>`;
    return entries
      .map(([k, v]) => {
        const n = this.toNumber(v);
        return `<div class="kv-row"><span>${this.escapeHtml(k)}</span><span>${n.toFixed(2)}${suffix || ""}</span></div>`;
      })
      .join("");
  },

  chipClassForValue(value) {
    const n = String(value || "").toLowerCase();
    if (n.includes("high") || n.includes("urgent") || n.includes("stat"))
      return "chip warn";
    if (n.includes("low") || n.includes("routine") || n.includes("normal"))
      return "chip ok";
    return "chip";
  },

  fallbackPatientKey(report, findings) {
    if (report && report.patient_id && report.patient_id !== "REF-UNKNOWN") {
      return `id-${report.patient_id}`.toLowerCase();
    }
    const source = String(findings || "").slice(0, 200);
    let hash = 0;
    for (let i = 0; i < source.length; i++) {
      hash = (hash << 5) - hash + source.charCodeAt(i);
      hash |= 0;
    }
    return `unknown-${Math.abs(hash)}`;
  },

  makeId(prefix) {
    if (window.crypto && window.crypto.randomUUID)
      return `${prefix}-${window.crypto.randomUUID()}`;
    return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
  },

  toNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  },

  asList(value) {
    return Array.isArray(value)
      ? value.filter((i) => i !== null && i !== undefined && i !== "")
      : [];
  },

  formatTimestamp(value) {
    if (!value) return "--";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  },

  escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  },
};

document.addEventListener("DOMContentLoaded", () => app.init());
