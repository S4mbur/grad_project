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
    currentHeatmapVariant: 'attention',
    heatmapViews: [],
    analysisStartTime: null,
    history: [],
    slideInfo: null,
    topTilesData: [],
    currentResult: null,
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
    const thresholdEl = document.getElementById('model-threshold');
    const descEl = document.getElementById('model-desc');

    const ensemble = state.modelsData?.ensembles?.find(e => e.key === key);
    if (ensemble) {
        f1El.textContent = `F1: ${(ensemble.f1 * 100).toFixed(1)}% | AUC: ${(ensemble.auc * 100).toFixed(1)}% | MelFN: 0`;
        thresholdEl.textContent = formatThresholdPolicy(ensemble.threshold_policy || null);
        descEl.textContent = ensemble.description;
        return;
    }

    const model = state.modelsData?.models?.find(m => m.key === key);
    if (model) {
        const fnText = model.mel_fn !== undefined && model.mel_fn !== '?' ? ` | MelFN: ${model.mel_fn}` : '';
        f1El.textContent = `F1: ${(model.f1 * 100).toFixed(1)}% | AUC: ${(model.auc * 100).toFixed(1)}%${fnText}`;
        thresholdEl.textContent = formatThresholdPolicy(model.threshold_policy || null);
        descEl.textContent = model.description;
    }
}

function formatThresholdPolicy(policy) {
    if (!policy || !policy.available) {
        return 'Threshold: default multiclass decision';
    }
    if (policy.label) {
        return String(policy.label);
    }

    const parts = [];
    if (policy.melanoma_safe_threshold !== null && policy.melanoma_safe_threshold !== undefined) {
        parts.push(`Mel review threshold ${Number(policy.melanoma_safe_threshold).toFixed(2)}`);
    }
    if (policy.selection_basis) {
        parts.push(String(policy.selection_basis));
    }
    if (!parts.length) {
        parts.push('Threshold registry loaded');
    }
    return parts.join(' | ');
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
    state.currentResult = result;

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
    const classKey = result.prediction_key || _getClassKey(result.raw_prediction || result.prediction);
    card.className = 'prediction-card ' + classKey + ' slide-up';
    document.getElementById('prediction-value').textContent = result.prediction;

    const maxProb = Math.max(...Object.values(result.probabilities));
    const safety = result.safety || null;
    const displayConfidence = safety && safety.confidence !== undefined
        ? safety.confidence
        : maxProb;
    document.getElementById('prediction-confidence').textContent =
        `Confidence: ${(displayConfidence * 100).toFixed(1)}%`;
    renderSafetyPanel(safety);

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
    document.getElementById('top-tiles-title').textContent = result.top_tiles_title || 'Top Attention Tiles';
    renderTopTiles(result.top_tiles || [], jobId);
    renderRetrievalPanel(result.retrieval || null);

    // Init viewer
    initSlideViewer(jobId);
    renderHeatmapViewControls(result.heatmap_views || [{ key: 'attention', label: 'Attention' }], result.default_heatmap_view || 'attention');

    // Show slide info bar
    document.getElementById('slide-info-bar').classList.add('active');

    // Show export button
    document.getElementById('btn-export').style.display = '';

    // Update history
    updateHistoryItem(jobId, 'completed', result.prediction, classKey);

    showToast(`Analysis complete: ${result.prediction} (${result.model_used})`, 'success');
}

function _getClassKey(prediction) {
    const p = (prediction || '').toLowerCase();
    if (p.includes('abstain') || p.includes('expert review')) return 'abstain';
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

function _riskBadgeClass(level) {
    const v = (level || '').toLowerCase();
    if (v.includes('urgent')) return 'risk-urgent';
    if (v.includes('high')) return 'risk-high';
    if (v.includes('moderate')) return 'risk-moderate';
    return 'risk-low';
}

function _oodBadgeClass(level) {
    const v = (level || '').toLowerCase();
    if (v.includes('strong')) return 'ood-strong';
    if (v.includes('moderate')) return 'ood-moderate';
    return 'ood-low';
}

function renderSafetyPanel(safety) {
    const panel = document.getElementById('safety-panel');
    if (!panel || !safety) {
        if (panel) panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';

    const riskBadge = document.getElementById('safety-risk-badge');
    riskBadge.className = `safety-badge ${_riskBadgeClass(safety.risk_level)}`;
    riskBadge.textContent = safety.risk_level || 'Unknown Risk';

    const abstainBadge = document.getElementById('safety-abstain-badge');
    abstainBadge.className = `safety-badge ${safety.abstain_recommended ? 'abstain-on' : 'abstain-off'}`;
    abstainBadge.textContent = safety.abstain_recommended ? 'Abstain Recommended' : 'No Abstain';

    const ood = safety.ood || {};
    const oodBadge = document.getElementById('safety-ood-badge');
    oodBadge.className = `safety-badge ${_oodBadgeClass(ood.ood_level)}`;
    oodBadge.textContent = ood.available ? `OOD ${ood.ood_level || 'low'}` : 'OOD N/A';

    const rawNote = safety.raw_prediction && safety.display_prediction && safety.raw_prediction !== safety.display_prediction
        ? ` Raw model prediction: ${safety.raw_prediction}.`
        : '';
    document.getElementById('safety-summary').textContent =
        (safety.recommendation || 'No safety recommendation.') + rawNote;

    document.getElementById('safety-uncertainty').textContent = `${((safety.uncertainty || 0) * 100).toFixed(1)}%`;
    document.getElementById('safety-margin').textContent = `${((safety.margin || 0) * 100).toFixed(1)}%`;
    document.getElementById('safety-melanoma-prob').textContent = `${((safety.melanoma_probability || 0) * 100).toFixed(1)}%`;
    document.getElementById('safety-disagreement').textContent =
        safety.ensemble_disagreement === null || safety.ensemble_disagreement === undefined
            ? 'N/A'
            : `${(safety.ensemble_disagreement * 100).toFixed(1)}%`;
    document.getElementById('safety-ood-score').textContent =
        ood.ood_score === null || ood.ood_score === undefined
            ? 'N/A'
            : `${(ood.ood_score * 100).toFixed(1)}%`;
    document.getElementById('safety-id-support').textContent =
        safety.id_support_score === null || safety.id_support_score === undefined
            ? 'N/A'
            : `${(safety.id_support_score * 100).toFixed(1)}%`;
    document.getElementById('safety-score').textContent =
        safety.safety_score === null || safety.safety_score === undefined
            ? 'N/A'
            : `${(safety.safety_score * 100).toFixed(1)}%`;
    const calibration = safety.calibration || {};
    const calibrationText = calibration.available
        ? `T=${Number(calibration.temperature || 1).toFixed(2)}`
        : 'Not loaded';
    document.getElementById('safety-calibration').textContent = calibrationText;
    document.getElementById('safety-threshold').textContent = formatThresholdPolicy(safety.threshold_policy || null);

    const flags = [];
    (safety.reasons || []).forEach(r => flags.push(r));
    if (safety.melanoma_first_guard) flags.push('Melanoma-first safeguard active');
    if (safety.hard_case_candidate) flags.push('Hard-case candidate');
    if (ood.available && ood.nearest_class) {
        flags.push(`Nearest in-distribution class: ${ood.nearest_class}`);
    }
    if (calibration.available && calibration.ece_after !== undefined && calibration.ece_after !== null) {
        flags.push(`Calibrated ECE: ${(Number(calibration.ece_after) * 100).toFixed(1)}%`);
    }
    if (safety.threshold_policy && safety.threshold_policy.threshold_triggered) {
        flags.push('Tuned melanoma review threshold was triggered');
    }
    if (safety.raw_prediction && safety.display_prediction && safety.raw_prediction !== safety.display_prediction) {
        flags.push(`Raw model prediction: ${safety.raw_prediction}`);
    }
    if (!flags.length) flags.push('No active safety flags');

    document.getElementById('safety-flags').innerHTML = flags
        .map(flag => `<span class="safety-flag">${flag}</span>`)
        .join('');
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
            <span class="tile-attn">${((tile.shared_score ?? tile.attention) * 100).toFixed(1)}%</span>
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

function renderRetrievalPanel(retrieval) {
    const panel = document.getElementById('retrieval-panel');
    const grid = document.getElementById('retrieval-grid');
    const hardGrid = document.getElementById('hard-retrieval-grid');
    const hardTitle = document.getElementById('hard-retrieval-title');
    const summary = document.getElementById('retrieval-summary');

    if (!panel || !grid || !hardGrid || !hardTitle || !summary) return;

    if (!retrieval || !retrieval.available) {
        panel.style.display = 'none';
        grid.innerHTML = '';
        hardGrid.innerHTML = '';
        hardTitle.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    summary.textContent = `${retrieval.bank_display || retrieval.bank_key} • ${retrieval.bank_size || 0} reference cases • ${retrieval.hard_case_count || 0} hard melanoma cases`;
    grid.innerHTML = _renderRetrievalCards(retrieval.similar_cases || [], 'No similar cases available.');

    const hardCases = retrieval.hard_melanoma_matches || [];
    hardTitle.style.display = hardCases.length ? 'block' : 'none';
    hardGrid.innerHTML = hardCases.length
        ? _renderRetrievalCards(hardCases, '')
        : '';
}

function _renderRetrievalCards(items, emptyText = '') {
    if (!items || !items.length) {
        if (!emptyText) return '';
        return `
            <div class="retrieval-empty">
                <div class="icon">R</div>
                <p>${emptyText}</p>
            </div>
        `;
    }

    return items.map(item => `
        <div class="retrieval-card">
            <img class="retrieval-thumb" src="${API}${item.thumbnail_url}" alt="${item.filename}" loading="lazy">
            <div class="retrieval-card-body">
                <div class="retrieval-card-top">
                    <span class="retrieval-label ${_getClassKey(item.true_label)}">${item.true_label}</span>
                    <span class="retrieval-similarity">${((item.similarity || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="retrieval-file">${item.filename || item.slide_id}</div>
                <div class="retrieval-meta">${item.source || 'unknown source'}</div>
                <div class="retrieval-flags">
                    ${item.is_hard_melanoma ? '<span class="retrieval-flag hard">Hard melanoma</span>' : ''}
                </div>
                <div class="retrieval-actions">
                    <button class="retrieval-compare-btn" type="button" onclick="compareRetrievedCase('${item.slide_id}')">Compare</button>
                </div>
            </div>
        </div>
    `).join('');
}

function _resolveHeatmapVariant(resultLike, preferredVariant) {
    const views = (resultLike && resultLike.heatmap_views) || [];
    const keys = views.map(v => v.key);
    if (preferredVariant && keys.includes(preferredVariant)) return preferredVariant;
    return resultLike.default_heatmap_view || (keys[0] || 'attention');
}

function _resultHeatmapUrl(jobId, resultLike, preferredVariant) {
    const variant = _resolveHeatmapVariant(resultLike, preferredVariant);
    if (!variant || variant === 'attention' || variant === 'default') {
        return `${API}/api/results/${jobId}/heatmap`;
    }
    return `${API}/api/results/${jobId}/heatmap/${variant}`;
}

function _renderCompareTiles(tiles) {
    if (!tiles || !tiles.length) {
        return '<div class="compare-empty">No top-tile data available.</div>';
    }
    return tiles.slice(0, 4).map(tile => `
        <div class="compare-tile">
            <img src="${API}${tile.image_url}" alt="Tile ${tile.rank}" loading="lazy">
            <span class="compare-tile-rank">#${tile.rank}</span>
            <span class="compare-tile-score">${((tile.shared_score ?? tile.attention) * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

function _renderCompareSummary(title, meta, jobId, resultLike, preferredVariant) {
    const safety = resultLike.safety || {};
    return `
        <div class="compare-column">
            <div class="compare-card-header">
                <div>
                    <div class="compare-title">${title}</div>
                    <div class="compare-subtitle">${meta}</div>
                </div>
                <div class="compare-badge ${_getClassKey(resultLike.prediction || resultLike.raw_prediction)}">${resultLike.prediction}</div>
            </div>
            <div class="compare-stats">
                <div class="compare-stat"><span>Decision</span><strong>${resultLike.decision_status || 'predicted'}</strong></div>
                <div class="compare-stat"><span>Risk</span><strong>${safety.risk_level || 'N/A'}</strong></div>
                <div class="compare-stat"><span>Safety</span><strong>${safety.safety_score !== undefined ? `${(safety.safety_score * 100).toFixed(1)}%` : 'N/A'}</strong></div>
                <div class="compare-stat"><span>Mel Prob.</span><strong>${safety.melanoma_probability !== undefined ? `${(safety.melanoma_probability * 100).toFixed(1)}%` : 'N/A'}</strong></div>
            </div>
            <img class="compare-heatmap" src="${_resultHeatmapUrl(jobId, resultLike, preferredVariant)}" alt="${title} heatmap">
            <div class="compare-tiles">${_renderCompareTiles(resultLike.top_tiles || [])}</div>
        </div>
    `;
}

async function compareRetrievedCase(slideId) {
    if (!state.currentJobId || !state.currentResult) {
        showToast('Analyze a slide first to open comparison mode', 'error');
        return;
    }

    const modal = document.getElementById('compare-modal');
    const body = document.getElementById('compare-modal-body');
    const title = document.getElementById('compare-modal-title');
    modal.classList.add('visible');
    title.textContent = 'Similar Case Comparison';
    body.innerHTML = '<div class="compare-loading">Loading retrieved case analysis...</div>';

    try {
        const resp = await fetch(`${API}/api/retrieval/cases/${slideId}/compare?model=${encodeURIComponent(state.selectedModel)}`);
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || 'Failed to build comparison view');
        }

        const currentMeta = `${state.currentResult.model_used || state.selectedModel} • ${state.currentResult.raw_prediction || state.currentResult.prediction}`;
        const retrievedMeta = `${data.model_display} • ${data.true_label} • ${data.source}`;
        const preferredVariant = state.currentHeatmapVariant || (state.currentResult.default_heatmap_view || 'attention');

        body.innerHTML = `
            <div class="compare-layout">
                ${_renderCompareSummary('Current Case', currentMeta, state.currentJobId, state.currentResult, preferredVariant)}
                ${_renderCompareSummary('Retrieved Case', retrievedMeta, data.job_id, data.result, preferredVariant)}
            </div>
        `;
    } catch (err) {
        body.innerHTML = `<div class="compare-error">${err.message || 'Failed to load comparison.'}</div>`;
        showToast('Failed to load retrieved case comparison', 'error');
    }
}

function closeCompareModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('compare-modal').classList.remove('visible');
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
    const primaryScore = tile.shared_score ?? tile.attention;
    const primaryLabel = tile.shared_score !== undefined ? 'Shared Focus' : 'Attention';

    document.getElementById('modal-tile-title').textContent =
        `Tile #${tile.rank} – ${primaryLabel}: ${(primaryScore * 100).toFixed(2)}%`;
    document.getElementById('modal-tile-image').src = `${API}${tile.image_url}`;

    const infoDiv = document.getElementById('modal-tile-info');
    const sharedBlock = tile.shared_score !== undefined ? `
        <div class="stat-card">
            <div class="stat-value">${(tile.shared_score * 100).toFixed(3)}%</div>
            <div class="stat-label">Shared Score</div>
        </div>` : '';
    const consensusBlock = tile.consensus_score !== undefined ? `
        <div class="stat-card">
            <div class="stat-value">${(tile.consensus_score * 100).toFixed(3)}%</div>
            <div class="stat-label">Consensus</div>
        </div>` : '';
    const disagreementBlock = tile.disagreement_score !== undefined ? `
        <div class="stat-card">
            <div class="stat-value">${(tile.disagreement_score * 100).toFixed(3)}%</div>
            <div class="stat-label">Disagreement</div>
        </div>` : '';
    infoDiv.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${tile.rank}</div>
            <div class="stat-label">Rank</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${(primaryScore * 100).toFixed(3)}%</div>
            <div class="stat-label">${primaryLabel}</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${tile.coord.x}, ${tile.coord.y}</div>
            <div class="stat-label">Position (px)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${tile.coord.size}×${tile.coord.size}</div>
            <div class="stat-label">Tile Size</div>
        </div>
        ${sharedBlock}
        ${consensusBlock}
        ${disagreementBlock}
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
    state.currentHeatmapVariant = state.currentHeatmapVariant || 'attention';
    document.getElementById('heatmap-toggle').checked = false;
    const heatBtn = document.getElementById('btn-heatmap-viewer');
    if (heatBtn) heatBtn.classList.remove('active');
}

// ───────────────── Heatmap ─────────────────────────────

function _heatmapOverlayUrl(jobId, variant) {
    if (!variant || variant === 'attention' || variant === 'default') {
        return `${API}/api/results/${jobId}/heatmap_only`;
    }
    return `${API}/api/results/${jobId}/heatmap_only/${variant}`;
}

function renderHeatmapViewControls(views, defaultVariant = 'attention') {
    const wrap = document.getElementById('heatmap-view-controls');
    const buttons = document.getElementById('heatmap-view-buttons');
    state.heatmapViews = views || [];
    state.currentHeatmapVariant = defaultVariant || 'attention';

    if (!views || views.length <= 1) {
        wrap.style.display = 'none';
        buttons.innerHTML = '';
        return;
    }

    wrap.style.display = 'block';
    buttons.innerHTML = views.map(view => `
        <button
            class="heatmap-view-btn ${view.key === state.currentHeatmapVariant ? 'active' : ''}"
            data-variant="${view.key}"
            type="button"
            onclick="setHeatmapVariant('${view.key}')"
            title="${view.description || view.label}"
        >${view.label}</button>
    `).join('');
}

function setHeatmapVariant(variant) {
    state.currentHeatmapVariant = variant;
    const buttons = document.querySelectorAll('.heatmap-view-btn');
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.variant === variant);
    });

    if (state.heatmapVisible) {
        toggleHeatmap(true);
    }
}

function toggleHeatmap(enabled) {
    if (!state.currentJobId || !state.viewer) return;

    state.heatmapVisible = enabled;
    const heatBtn = document.getElementById('btn-heatmap-viewer');

    if (enabled) {
        const overlaySrc = _heatmapOverlayUrl(state.currentJobId, state.currentHeatmapVariant);
        if (!state.heatmapOverlayId) {
            const img = document.createElement('img');
            img.src = overlaySrc;
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
            if (el) {
                if (el.src !== overlaySrc) {
                    el.src = overlaySrc;
                }
                el.style.opacity = '0.6';
            }
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

function addToHistory(jobId, filename, status, prediction, model, predictionKey=null) {
    state.history.unshift({ jobId, filename, status, prediction, model, predictionKey, createdAt: new Date() });
    renderHistory();
}

function updateHistoryItem(jobId, status, prediction, predictionKey=null) {
    const item = state.history.find(h => h.jobId === jobId);
    if (item) {
        item.status = status;
        if (prediction) item.prediction = prediction;
        if (predictionKey) item.predictionKey = predictionKey;
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
            ? (h.predictionKey || _getClassKey(h.prediction || ''))
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
        closeCompareModal();
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
