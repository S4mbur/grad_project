/**
 * SkinSight – Frontend Application  (v3 – multi-model, 4-class)
 * ==============================================================
 * Manages file upload with model selection, analysis polling,
 * 4-class result display, ensemble breakdown,
 * OpenSeadragon slide viewer, and heatmap overlay.
 */

// ─────────────────────── State ─────────────────────────
const state = {
    currentJobId: null,
    pollInterval: null,
    viewer: null,
    heatmapVisible: false,
    heatmapOverlayId: null,
    analysisStartTime: null,
    history: [],
    slideInfo: null,
    topTilesData: [],
    modelsData: null,    // from /api/models
    selectedModel: 'ensemble_3_best',
};

const API = '';

// ─────────────────────── Upload ────────────────────────

function initUpload() {
    const zone = document.getElementById('upload-zone');

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) uploadFile(file);
    event.target.value = '';
}

async function uploadFile(file) {
    const validExts = ['.tif', '.tiff', '.svs', '.ndpi', '.mrxs', '.scn'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
        showToast(`Unsupported format: ${ext}`, 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024 * 1024) {
        showToast('File too large (max 10 GB)', 'error');
        return;
    }

    const modelKey = state.selectedModel;
    const modelName = getModelDisplayName(modelKey);
    showToast(`Uploading ${file.name} (model: ${modelName})...`, 'info');
    state.analysisStartTime = Date.now();

    const form = new FormData();
    form.append('slide', file);
    form.append('model', modelKey);

    try {
        const resp = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
        if (!resp.ok) {
            const err = await resp.json();
            showToast(err.error || 'Upload failed', 'error');
            return;
        }

        const data = await resp.json();
        state.currentJobId = data.job_id;
        showToast(`Analysis started with ${data.model}: ${data.job_id}`, 'success');

        showAnalysisStatus(file.name);
        startPolling(data.job_id);
        addToHistory(data.job_id, file.name, 'processing', null, modelName);

    } catch (err) {
        showToast('Network error: ' + err.message, 'error');
    }
}

// ────────────────── Model Selection ────────────────────

async function loadModels() {
    try {
        const resp = await fetch(`${API}/api/models`);
        if (!resp.ok) return;
        const data = await resp.json();
        state.modelsData = data;

        const select = document.getElementById('model-select');
        select.innerHTML = '';

        // Add ensemble presets first (top priority)
        if (data.ensembles && data.ensembles.length > 0) {
            const ensGroup = document.createElement('optgroup');
            ensGroup.label = '🏆 Ensemble (MelFN=0)';
            data.ensembles.forEach(e => {
                const opt = document.createElement('option');
                opt.value = e.key;
                opt.textContent = `${e.display} (F1: ${(e.f1 * 100).toFixed(1)}%)`;
                ensGroup.appendChild(opt);
            });
            select.appendChild(ensGroup);
        }

        // Group individual models by backbone
        const groups = {};
        data.models.forEach(m => {
            const g = m.group || 'Other';
            if (!groups[g]) groups[g] = [];
            groups[g].push(m);
        });

        // Order: Phikon first, then ConvNeXt, DINOv2, ResNet
        const groupOrder = ['Phikon', 'ConvNeXt-Base', 'ConvNeXt-Small', 'DINOv2', 'ResNet'];
        const sortedGroups = groupOrder.filter(g => groups[g]);
        Object.keys(groups).forEach(g => {
            if (!sortedGroups.includes(g)) sortedGroups.push(g);
        });

        sortedGroups.forEach(groupName => {
            const models = groups[groupName];
            if (!models) return;
            const optGroup = document.createElement('optgroup');
            optGroup.label = `🧠 ${groupName}`;
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.key;
                const fnLabel = m.mel_fn !== undefined && m.mel_fn !== '?' ? ` · MelFN=${m.mel_fn}` : '';
                opt.textContent = `${m.display} (F1: ${(m.f1 * 100).toFixed(1)}%${fnLabel})`;
                if (!m.available) {
                    opt.disabled = true;
                    opt.textContent += ' ⚠';
                }
                optGroup.appendChild(opt);
            });
            select.appendChild(optGroup);
        });

        // Set default
        select.value = data.default || 'ensemble_3_best';
        state.selectedModel = select.value;
        updateModelInfo(select.value);

        // Listen for changes
        select.addEventListener('change', (e) => {
            state.selectedModel = e.target.value;
            updateModelInfo(e.target.value);
        });

    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

function updateModelInfo(key) {
    const infoEl = document.getElementById('model-info');
    const f1El = document.getElementById('model-f1');
    const descEl = document.getElementById('model-desc');

    // Check ensemble presets first
    const ensemble = state.modelsData?.ensembles?.find(e => e.key === key);
    if (ensemble) {
        f1El.textContent = `F1: ${(ensemble.f1 * 100).toFixed(1)}% · AUC: ${(ensemble.auc * 100).toFixed(1)}% · MelFN: 0 ✓`;
        descEl.textContent = ensemble.description;
        return;
    }

    const model = state.modelsData?.models?.find(m => m.key === key);
    if (model) {
        const fnText = model.mel_fn !== undefined && model.mel_fn !== '?' ? ` · MelFN: ${model.mel_fn}` : '';
        f1El.textContent = `F1: ${(model.f1 * 100).toFixed(1)}% · AUC: ${(model.auc * 100).toFixed(1)}%${fnText}`;
        descEl.textContent = model.description;
    }
}

function getModelDisplayName(key) {
    const ensemble = state.modelsData?.ensembles?.find(e => e.key === key);
    if (ensemble) return ensemble.name;
    const model = state.modelsData?.models?.find(m => m.key === key);
    return model ? model.display : key;
}

// ────────────────────── Polling ─────────────────────────

function startPolling(jobId) {
    if (state.pollInterval) clearInterval(state.pollInterval);
    state.pollInterval = setInterval(() => pollStatus(jobId), 1500);
}

async function pollStatus(jobId) {
    try {
        const resp = await fetch(`${API}/api/status/${jobId}`);
        if (!resp.ok) return;

        const data = await resp.json();
        updateProgress(data);

        if (data.status === 'completed') {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
            onAnalysisComplete(jobId, data);
        } else if (data.status === 'error') {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
            showToast('Analysis failed: ' + (data.message || 'Unknown error'), 'error');
            hideAnalysisStatus();
            updateHistoryItem(jobId, 'error');
        }

        if (data.slide_info) {
            const info = data.slide_info;
            state.slideInfo = info;
            document.getElementById('info-dimensions').textContent =
                `${info.width?.toLocaleString()} × ${info.height?.toLocaleString()}`;
            document.getElementById('info-mpp').textContent = info.mpp || '—';
        }

    } catch (err) {
        console.error('Poll error:', err);
    }
}

// ──────────────────── Progress UI ──────────────────────

function showAnalysisStatus(filename) {
    const el = document.getElementById('analysis-status');
    el.classList.add('visible');
    document.getElementById('progress-filename').textContent = filename;
    document.getElementById('progress-percent').textContent = '0%';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-message').textContent = 'Starting analysis...';
    document.getElementById('result-panel').classList.remove('visible');
}

function hideAnalysisStatus() {
    document.getElementById('analysis-status').classList.remove('visible');
}

function updateProgress(data) {
    const pct = data.progress || 0;
    document.getElementById('progress-percent').textContent = pct + '%';
    document.getElementById('progress-bar').style.width = pct + '%';
    if (data.message) {
        document.getElementById('progress-message').textContent = data.message;
    }
}

// ──────────────── Analysis Complete ────────────────────

function onAnalysisComplete(jobId, data) {
    hideAnalysisStatus();

    const result = data.result;
    if (!result) return;

    if (data.slide_info) {
        state.slideInfo = data.slide_info;
    }

    // Show result panel
    const panel = document.getElementById('result-panel');
    panel.classList.add('visible');

    // Model badge
    const badge = document.getElementById('result-model-badge');
    badge.style.display = 'flex';
    document.getElementById('result-model-name').textContent = result.model_used || 'Unknown';

    // Model in info bar
    document.getElementById('info-model').textContent = result.model_used || '—';

    // Prediction card
    const card = document.getElementById('prediction-card');
    const classKey = _getClassKey(result.prediction);
    card.className = 'prediction-card ' + classKey + ' slide-up';
    document.getElementById('prediction-value').textContent = result.prediction;

    const maxProb = Math.max(...Object.values(result.probabilities));
    document.getElementById('prediction-confidence').textContent =
        `Confidence: ${(maxProb * 100).toFixed(1)}%`;

    // Probability bars (4 classes)
    const probs = result.probabilities;
    animateProbBar('normal', probs['Normal/Benign'] || 0);
    animateProbBar('bcc', probs['BCC'] || 0);
    animateProbBar('scc', probs['SCC'] || 0);
    animateProbBar('melanoma', probs['Melanoma'] || 0);

    // Ensemble details
    const ensembleDiv = document.getElementById('ensemble-details');
    if (result.ensemble_details && result.ensemble_details.length > 0) {
        ensembleDiv.style.display = 'block';
        const bd = document.getElementById('ensemble-breakdown');
        bd.innerHTML = result.ensemble_details.map(m => {
            const mProbs = m.probabilities;
            const topProb = Math.max(...Object.values(mProbs));
            const topClass = Object.entries(mProbs).sort((a, b) => b[1] - a[1])[0][0];
            return `
                <div class="ensemble-row">
                    <span class="ensemble-model">${m.model}</span>
                    <span class="ensemble-pred ${_getClassKey(m.prediction)}">${m.prediction}</span>
                    <span class="ensemble-conf">${(topProb * 100).toFixed(0)}%</span>
                </div>
            `;
        }).join('');
    } else {
        ensembleDiv.style.display = 'none';
    }

    // Stats
    document.getElementById('stat-tiles').textContent = result.n_tiles || '—';
    const elapsed = state.analysisStartTime
        ? Math.round((Date.now() - state.analysisStartTime) / 1000) + 's'
        : '—';
    document.getElementById('stat-time').textContent = elapsed;

    // Top tiles
    renderTopTiles(result.top_tiles || [], jobId);

    // Init viewer
    initSlideViewer(jobId);

    // Show slide info bar
    document.getElementById('slide-info-bar').classList.add('active');

    // Show export button
    document.getElementById('btn-export').style.display = '';

    // Update history
    updateHistoryItem(jobId, 'completed', result.prediction);

    showToast(`Analysis complete: ${result.prediction} (${result.model_used})`, 'success');
}

function _getClassKey(prediction) {
    const p = (prediction || '').toLowerCase();
    if (p.includes('normal') || p.includes('benign')) return 'normal';
    if (p.includes('bcc')) return 'bcc';
    if (p.includes('scc')) return 'scc';
    if (p.includes('melanoma')) return 'melanoma';
    return 'normal';
}

function animateProbBar(cls, value) {
    const pct = (value * 100).toFixed(1);
    const bar = document.getElementById(`prob-bar-${cls}`);
    const val = document.getElementById(`prob-val-${cls}`);

    requestAnimationFrame(() => {
        bar.style.width = pct + '%';
        val.textContent = pct + '%';
    });
}

// ─────────────────── Top Tiles ─────────────────────────

function renderTopTiles(tiles, jobId) {
    const grid = document.getElementById('top-tiles-grid');
    grid.innerHTML = '';
    state.topTilesData = tiles;

    tiles.forEach((tile) => {
        const div = document.createElement('div');
        div.className = 'top-tile fade-in';
        div.style.animationDelay = (tile.rank * 0.05) + 's';
        div.onclick = () => openTileModal(tile, jobId);
        div.ondblclick = (e) => {
            e.stopPropagation();
            navigateToTile(tile);
        };
        div.title = `Click: preview  ·  Double-click: go to location`;

        div.innerHTML = `
            <img src="${API}${tile.image_url}" alt="Tile ${tile.rank}" loading="lazy">
            <span class="tile-rank">#${tile.rank}</span>
            <span class="tile-attn">${(tile.attention * 100).toFixed(1)}%</span>
            <span class="tile-locate" title="Navigate to this tile on slide">📍</span>
        `;

        const pin = div.querySelector('.tile-locate');
        pin.addEventListener('click', (e) => {
            e.stopPropagation();
            navigateToTile(tile);
        });

        grid.appendChild(div);
    });
}

// ─────────────── Navigate to Tile on Viewer ────────────

function navigateToTile(tile) {
    if (!state.viewer || !state.slideInfo) {
        showToast('Slide viewer not ready', 'error');
        return;
    }

    const slideW = state.slideInfo.width;
    const coord = tile.coord;
    const tileRealSize = coord.read_size || (coord.size * (coord.level_ds || 1));

    const vpCenterX = (coord.x + tileRealSize / 2) / slideW;
    const vpCenterY = (coord.y + tileRealSize / 2) / slideW;

    const contextFactor = 5;
    const regionWidth = (tileRealSize * contextFactor) / slideW;
    const targetZoom = 1 / regionWidth;

    const viewportPoint = new OpenSeadragon.Point(vpCenterX, vpCenterY);
    state.viewer.viewport.panTo(viewportPoint, false);
    state.viewer.viewport.zoomTo(targetZoom, viewportPoint, false);

    _flashTileMarker(coord, slideW, tileRealSize);
    showToast(`Navigated to Tile #${tile.rank}`, 'info');
}

function _flashTileMarker(coord, slideW, tileRealSize) {
    const existing = document.getElementById('tile-nav-marker');
    if (existing) existing.remove();

    const marker = document.createElement('div');
    marker.id = 'tile-nav-marker';
    marker.style.cssText = `
        border: 3px solid #ff4444;
        background: rgba(255, 68, 68, 0.15);
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.5), inset 0 0 20px rgba(255, 68, 68, 0.1);
        border-radius: 4px;
        pointer-events: none;
        transition: opacity 0.5s ease;
    `;

    const vpX = coord.x / slideW;
    const vpY = coord.y / slideW;
    const vpW = tileRealSize / slideW;
    const vpH = tileRealSize / slideW;

    const rect = new OpenSeadragon.Rect(vpX, vpY, vpW, vpH);
    state.viewer.addOverlay({ element: marker, location: rect });

    setTimeout(() => { marker.style.opacity = '0'; }, 2500);
    setTimeout(() => { state.viewer.removeOverlay(marker); }, 3200);
}

// ─────────────────── Tile Modal ────────────────────────

function openTileModal(tile, jobId) {
    const modal = document.getElementById('tile-modal');
    modal.classList.add('visible');

    document.getElementById('modal-tile-title').textContent =
        `Tile #${tile.rank} – Attention: ${(tile.attention * 100).toFixed(2)}%`;
    document.getElementById('modal-tile-image').src = `${API}${tile.image_url}`;

    const infoDiv = document.getElementById('modal-tile-info');
    infoDiv.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${tile.rank}</div>
            <div class="stat-label">Rank</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${(tile.attention * 100).toFixed(3)}%</div>
            <div class="stat-label">Attention Weight</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${tile.coord.x}, ${tile.coord.y}</div>
            <div class="stat-label">Position (px)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${tile.coord.size}×${tile.coord.size}</div>
            <div class="stat-label">Tile Size</div>
        </div>
    `;

    const navBtn = document.getElementById('modal-nav-btn');
    if (navBtn) {
        navBtn.onclick = () => {
            closeTileModal();
            navigateToTile(tile);
        };
        navBtn.style.display = '';
    }
}

function closeTileModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('tile-modal').classList.remove('visible');
}

// ───────────────── Slide Viewer ────────────────────────

function initSlideViewer(jobId) {
    document.getElementById('viewer-placeholder').style.display = 'none';

    const viewerEl = document.getElementById('slide-viewer');
    viewerEl.classList.add('active');

    const tileSource = `${API}/api/results/${jobId}/dzi/slide.dzi`;

    if (state.viewer) {
        const oldOverlay = document.getElementById('osd-heatmap-overlay');
        if (oldOverlay) {
            state.viewer.removeOverlay(oldOverlay);
            oldOverlay.remove();
        }
        state.heatmapOverlayId = null;

        const navMarker = document.getElementById('tile-nav-marker');
        if (navMarker) {
            state.viewer.removeOverlay(navMarker);
            navMarker.remove();
        }

        state.viewer.world.removeAll();
        state.viewer.open(tileSource);
    } else {
        state.viewer = OpenSeadragon({
            id: 'slide-viewer',
            tileSources: tileSource,
            prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/',
            showNavigationControl: false,
            showNavigator: true,
            navigatorPosition: 'TOP_RIGHT',
            navigatorSizeRatio: 0.15,
            navigatorAutoFade: true,
            animationTime: 0.6,
            blendTime: 0.3,
            minZoomImageRatio: 0.5,
            maxZoomPixelRatio: 4,
            visibilityRatio: 0.8,
            gestureSettingsMouse: { scrollToZoom: true },
            background: '#0a0a0f',
            immediateRender: true,
            imageLoaderLimit: 4,
            timeout: 30000,
        });
    }

    document.getElementById('viewer-controls').classList.add('active');

    state.heatmapVisible = false;
    document.getElementById('heatmap-toggle').checked = false;
    const heatBtn = document.getElementById('btn-heatmap-viewer');
    if (heatBtn) heatBtn.classList.remove('active');
}

// ───────────────── Heatmap ─────────────────────────────

function toggleHeatmap(enabled) {
    if (!state.currentJobId || !state.viewer) return;

    state.heatmapVisible = enabled;
    const heatBtn = document.getElementById('btn-heatmap-viewer');

    if (enabled) {
        if (!state.heatmapOverlayId) {
            const img = document.createElement('img');
            img.src = `${API}/api/results/${state.currentJobId}/heatmap_only`;
            img.id = 'osd-heatmap-overlay';
            img.style.opacity = '0.6';
            img.style.pointerEvents = 'none';
            img.style.transition = 'opacity 0.4s ease';

            state.viewer.addOverlay({
                element: img,
                location: state.viewer.world.getItemAt(0).getBounds(),
            });
            state.heatmapOverlayId = 'osd-heatmap-overlay';
        } else {
            const el = document.getElementById('osd-heatmap-overlay');
            if (el) el.style.opacity = '0.6';
        }
        if (heatBtn) heatBtn.classList.add('active');
    } else {
        const el = document.getElementById('osd-heatmap-overlay');
        if (el) el.style.opacity = '0';
        if (heatBtn) heatBtn.classList.remove('active');
    }
}

function toggleHeatmapFromViewer() {
    const toggle = document.getElementById('heatmap-toggle');
    toggle.checked = !toggle.checked;
    toggleHeatmap(toggle.checked);
}

// ───────────────── Viewer Controls ─────────────────────

function zoomIn() {
    if (state.viewer) {
        state.viewer.viewport.zoomTo(state.viewer.viewport.getZoom() * 1.5);
    }
}

function zoomOut() {
    if (state.viewer) {
        state.viewer.viewport.zoomTo(state.viewer.viewport.getZoom() / 1.5);
    }
}

function resetView() {
    if (state.viewer) {
        state.viewer.viewport.goHome();
    }
}

// ───────────────── History ─────────────────────────────

function addToHistory(jobId, filename, status, prediction, model) {
    state.history.unshift({ jobId, filename, status, prediction, model, createdAt: new Date() });
    renderHistory();
}

function updateHistoryItem(jobId, status, prediction) {
    const item = state.history.find(h => h.jobId === jobId);
    if (item) {
        item.status = status;
        if (prediction) item.prediction = prediction;
    }
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');
    if (state.history.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <div class="icon">📂</div>
                <p>No analyses yet.<br>Upload a slide to get started.</p>
            </div>`;
        return;
    }

    list.innerHTML = state.history.map(h => {
        const badgeClass = h.status === 'completed'
            ? _getClassKey(h.prediction || '')
            : h.status;
        const isActive = h.jobId === state.currentJobId ? 'active' : '';
        const time = h.createdAt
            ? new Date(h.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '';
        const statusText = h.status === 'completed'
            ? h.prediction
            : h.status === 'processing' ? 'Analyzing...' : h.status;
        const modelText = h.model ? ` · ${h.model}` : '';

        return `
            <div class="history-item ${isActive}" onclick="loadHistoryItem('${h.jobId}')">
                <div class="history-badge ${badgeClass}"></div>
                <div class="history-info">
                    <div class="history-name">${h.filename}</div>
                    <div class="history-meta">${statusText}${modelText} · ${time}</div>
                </div>
            </div>
        `;
    }).join('');
}

async function loadHistoryItem(jobId) {
    state.currentJobId = jobId;
    renderHistory();

    try {
        const resp = await fetch(`${API}/api/status/${jobId}`);
        if (!resp.ok) return;
        const data = await resp.json();

        if (data.status === 'completed' && data.result) {
            onAnalysisComplete(jobId, data);
        } else if (data.status === 'processing') {
            showAnalysisStatus(data.filename || 'Unknown');
            startPolling(jobId);
        }
    } catch (err) {
        showToast('Failed to load results', 'error');
    }
}

// ───────────────── Export ──────────────────────────────

async function exportResults() {
    if (!state.currentJobId) {
        showToast('No analysis to export', 'error');
        return;
    }
    window.open(`${API}/api/results/${state.currentJobId}/export`, '_blank');
}

// ───────────────── Toasts ──────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
        <span class="toast-text">${message}</span>
    `;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ───────────────── Keyboard Shortcuts ──────────────────

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeTileModal();
    }
    if (e.key === 'h' && !e.ctrlKey && !e.metaKey && e.target.tagName !== 'INPUT') {
        const toggle = document.getElementById('heatmap-toggle');
        toggle.checked = !toggle.checked;
        toggleHeatmap(toggle.checked);
    }
});

// ───────────────── Init ────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    renderHistory();
    loadModels();
});
