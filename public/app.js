const API_BASE = `${window.location.origin}/api/v1`;
const API_ROOT = window.location.origin;
const API_KEY = "dev_key";

// ============================
// TAB NAVIGATION
// ============================

function switchTab(tabName) {
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));

    const panel = document.getElementById(`tab-${tabName}`);
    if (panel) { panel.classList.remove("hidden"); panel.classList.add("active"); }

    const link = document.getElementById(`nav-${tabName}`);
    if (link) link.classList.add("active");

    if (tabName === "metrics") loadMetrics();
    if (tabName === "models") loadModels();
}

// ============================
// HEALTH CHECK
// ============================

async function checkHealth() {
    const statusEl = document.getElementById("health-status");
    const dot = document.getElementById("status-dot");
    try {
        const res = await fetch(`${API_ROOT}/health`);
        if (res.ok) {
            statusEl.textContent = "Gateway Online";
            dot.classList.add("healthy");
        } else {
            statusEl.textContent = "Gateway Degraded";
            dot.classList.remove("healthy");
        }
    } catch {
        statusEl.textContent = "Gateway Unreachable";
        dot.classList.remove("healthy");
    }
}

// ============================
// PLAYGROUND — ANALYZE
// ============================

async function handleAnalyze() {
    const prompt = document.getElementById("prompt-input").value;
    const maxTokens = document.getElementById("max-tokens").value;
    const temperature = document.getElementById("temperature").value;
    const preferredModel = document.getElementById("preferred-model").value;

    if (!prompt.trim()) { alert("Please enter a prompt."); return; }

    document.getElementById("results-content").classList.add("hidden");
    document.getElementById("results-empty").classList.add("hidden");
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("submit-btn").disabled = true;

    try {
        const payload = { prompt, max_tokens: parseInt(maxTokens), temperature: parseFloat(temperature) };
        if (preferredModel) payload.preferred_model = preferredModel;

        const response = await fetch(`${API_BASE}/routing/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`API Error: ${response.status} ${await response.text()}`);

        const data = await response.json();

        document.getElementById("res-model").textContent = data.selected_model;
        document.getElementById("res-strategy").textContent = data.strategy_used;
        document.getElementById("res-latency").textContent = `${data.estimated_latency.toFixed(1)} ms`;
        document.getElementById("res-cost").textContent = `$${data.estimated_cost.toFixed(6)}`;
        document.getElementById("res-text").textContent = data.response_text || "No response generated";

        const altList = document.getElementById("res-alternatives");
        altList.innerHTML = "";
        (data.alternative_candidates || []).forEach(model => {
            const li = document.createElement("li");
            li.textContent = model;
            altList.appendChild(li);
        });

        document.getElementById("results-content").classList.remove("hidden");

    } catch (error) {
        alert(`Failed to execute: ${error.message}`);
        console.error(error);
        document.getElementById("results-empty").classList.remove("hidden");
    } finally {
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("submit-btn").disabled = false;
    }
}

// ============================
// METRICS TAB
// ============================

async function loadMetrics() {
    const content = document.getElementById("metrics-content");
    const loader = document.getElementById("metrics-loading");
    content.style.opacity = "0";
    loader.classList.remove("hidden");

    try {
        const res = await fetch(`${API_BASE}/monitoring/metrics`, {
            headers: { "X-API-Key": API_KEY }
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Overview cards
        const overview = document.getElementById("metrics-overview");
        overview.innerHTML = `
            <div class="metric-overview-card">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
                <div class="metric-overview-value">${data.total_requests.toLocaleString()}</div>
                <div class="metric-overview-label">Total Requests</div>
            </div>
            <div class="metric-overview-card success">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                <div class="metric-overview-value">${(data.success_rate * 100).toFixed(1)}%</div>
                <div class="metric-overview-label">Success Rate</div>
            </div>
            <div class="metric-overview-card ${data.error_rate > 0.1 ? 'danger' : ''}">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/></svg>
                <div class="metric-overview-value">${(data.error_rate * 100).toFixed(1)}%</div>
                <div class="metric-overview-label">Error Rate</div>
            </div>
            <div class="metric-overview-card">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <div class="metric-overview-value">${data.avg_latency_ms.toFixed(0)} ms</div>
                <div class="metric-overview-label">Avg Latency</div>
            </div>
            <div class="metric-overview-card">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                <div class="metric-overview-value">${(data.cache_hit_rate * 100).toFixed(1)}%</div>
                <div class="metric-overview-label">Cache Hit Rate</div>
            </div>
        `;

        // Per-model stats
        const perModel = document.getElementById("per-model-stats");
        const stats = data.per_model_stats || {};
        if (Object.keys(stats).length === 0) {
            perModel.innerHTML = `<p style="color:var(--text-secondary);padding:20px 0;">No per-model data yet. Run some requests in the Playground first.</p>`;
        } else {
            perModel.innerHTML = Object.entries(stats).map(([model, s]) => `
                <div class="model-stat-card">
                    <div class="model-stat-name">${model.split("/").pop()}</div>
                    <div class="model-stat-full-id">${model}</div>
                    <div class="model-stat-row"><span>Requests</span><strong>${s.requests}</strong></div>
                    <div class="model-stat-row"><span>Error Rate</span><strong>${(s.error_rate * 100).toFixed(1)}%</strong></div>
                    <div class="model-stat-row"><span>Avg Latency</span><strong>${s.avg_latency_ms.toFixed(0)} ms</strong></div>
                </div>
            `).join("");
        }

        loader.classList.add("hidden");
        content.style.opacity = "1";

    } catch (err) {
        loader.classList.add("hidden");
        content.style.opacity = "1";
        document.getElementById("metrics-overview").innerHTML =
            `<p style="color:#f87171;grid-column:1/-1;">Failed to load metrics: ${err.message}. Make sure the backend is running.</p>`;
    }
}

// ============================
// MODELS REGISTRY TAB
// ============================

async function loadModels() {
    const grid = document.getElementById("models-grid");
    const loader = document.getElementById("models-loading");
    grid.style.opacity = "0";
    loader.classList.remove("hidden");

    try {
        const res = await fetch(`${API_BASE}/models`, {
            headers: { "X-API-Key": API_KEY }
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const models = await res.json();

        const tierColors = { fast: "#22d3ee", capable: "#a78bfa", standard: "#34d399", premium: "#fb923c" };
        const tierIcons = { fast: "⚡", capable: "🧠", standard: "📦", premium: "💎" };

        grid.innerHTML = models.map(m => `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-tier-badge" style="background:${(tierColors[m.tier] || "#64748b")}22;color:${tierColors[m.tier] || "#94a3b8"}">
                        ${tierIcons[m.tier] || "🤖"} ${m.tier.toUpperCase()}
                    </span>
                    <span class="model-status-pill ${m.status === 'available' ? 'online' : 'offline'}">${m.status}</span>
                </div>
                <h3 class="model-name">${m.id.split("/").pop()}</h3>
                <p class="model-org">${m.id.split("/")[0] || m.id}</p>
                <p class="model-desc">${m.description || "No description available."}</p>
                <div class="model-meta">
                    <span>Max Tokens: <strong>${m.max_tokens ? m.max_tokens.toLocaleString() : "—"}</strong></span>
                    <span>Cost/1K: <strong>${m.cost_per_1k_tokens ? "$" + m.cost_per_1k_tokens.toFixed(4) : "—"}</strong></span>
                </div>
                <div class="model-id-chip">${m.id}</div>
            </div>
        `).join("");

        loader.classList.add("hidden");
        grid.style.opacity = "1";

    } catch (err) {
        loader.classList.add("hidden");
        grid.style.opacity = "1";
        grid.innerHTML = `<p style="color:#f87171;">Failed to load models: ${err.message}</p>`;
    }
}

// ============================
// DYNAMIC MODEL DROPDOWN
// ============================

// Custom Dropdown Event Listeners
document.addEventListener("click", (e) => {
    const wrapper = document.getElementById("model-selector");
    if (!wrapper) return;
    
    if (e.target.closest(".custom-select-trigger")) {
        wrapper.classList.toggle("open");
    } 
    else if (e.target.closest(".custom-option")) {
        const option = e.target.closest(".custom-option");
        const value = option.dataset.value;
        const title = option.querySelector(".option-title").textContent;
        const icon = option.querySelector(".option-icon").textContent;
        
        document.querySelector(".custom-select-value").innerHTML = `${icon} <span style="margin-left:4px">${title}</span>`;
        document.getElementById("preferred-model").value = value;
        
        wrapper.querySelectorAll(".custom-option").forEach(opt => opt.classList.remove("selected"));
        option.classList.add("selected");
        wrapper.classList.remove("open");
    } 
    else if (!e.target.closest(".custom-select-wrapper")) {
        wrapper.classList.remove("open");
    }
});

async function populateModelDropdown() {
    const container = document.querySelector(".custom-options-container");
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE}/models`, {
            headers: { "X-API-Key": API_KEY }
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const models = await res.json();

        const firstOptionHtml = `
            <div class="custom-option selected" data-value="">
                <div class="option-icon">🎯</div>
                <div class="option-content">
                    <span class="option-title">Auto-Route</span>
                    <span class="option-desc">Intelligent Selection</span>
                </div>
            </div>`;
        
        const tierIcons = { fast: "⚡", capable: "🧠", standard: "📦", premium: "💎" };

        const optionsHtml = models.map(m => {
            const shortId = m.id.split("/").pop();
            const icon = tierIcons[m.tier] || "🤖";
            return `
                <div class="custom-option" data-value="${m.id}">
                    <div class="option-icon">${icon}</div>
                    <div class="option-content">
                        <span class="option-title">${shortId}</span>
                        <span class="option-desc">${m.tier.toUpperCase()} • ${m.cost_per_1k_tokens ? '$'+m.cost_per_1k_tokens.toFixed(4)+'/1k' : 'Free'}</span>
                    </div>
                </div>
            `;
        }).join("");
        
        container.innerHTML = firstOptionHtml + optionsHtml;

    } catch (err) {
        console.warn("Failed to populate model dropdown:", err);
    }
}

// ============================
// INIT
// ============================

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setInterval(checkHealth, 30000);

    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            switchTab(link.dataset.tab);
        });
    });

    document.getElementById("submit-btn").addEventListener("click", handleAnalyze);
    document.getElementById("refresh-metrics-btn").addEventListener("click", loadMetrics);

    // Load available models into the Force Model dropdown
    populateModelDropdown();
});
