const statusEl = document.getElementById('status');
const infoEl = document.getElementById('model-info');
const modelOptionsEl = document.getElementById('model-options');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const queueStatus = document.getElementById('queue-status');
const queueListEl = document.getElementById('queue-list');
const textVariantGrid = document.getElementById('text-variant-grid');
const cropsVariantGrid = document.getElementById('crops-variant-grid');
const boundingVariantGrid = document.getElementById('bounding-variant-grid');
const tabPlain = document.getElementById('tab-plain');
const tabMarkdown = document.getElementById('tab-markdown');
const modal = document.getElementById('modal');
const modalImage = document.getElementById('modal-image');
const modalClose = document.getElementById('modal-close');
const modalCopyBtn = document.getElementById('modal-copy');
const modalDownloadBtn = document.getElementById('modal-download');
const historyListEl = document.getElementById('history-list');
const iconSpriteContainer = document.getElementById('icon-sprite');
const inputPreviewSection = document.getElementById('input-preview-section');
const inputPreviewGrid = document.getElementById('input-preview-grid');
const promptInput = document.getElementById('prompt-input');
const promptResetBtn = document.getElementById('prompt-reset');
const defaultPrompt = promptInput?.defaultValue || '';
const progressWrapper = document.getElementById('progress-wrapper');
const progressBar = document.getElementById('progress-bar');
const MODEL_PRIORITIES = {
  yomitoku: 0,
  deepseek: 1,
  'deepseek-4bit': 2,
};
const MODEL_STATUS_LABELS = {
  pending: '待機中',
  running: '処理中',
  success: '完了',
  error: '失敗',
};
const MODEL_STATUS_CLASSES = {
  pending: 'pending',
  running: 'running',
  success: 'success',
  error: 'error',
};

const MODEL_SELECTION_STORAGE_KEY = 'ocr.selectedModels';
const MODEL_OPTIONS_STORAGE_KEY = 'ocr.modelOptions';

const modelTimerHandles = new Map();

function buildPlaceholderDataUrl(label) {
  const safeLabel = (label || '').toUpperCase().replace(/[^A-Z0-9+]/g, '').slice(0, 6) || 'FILE';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="140" height="100" viewBox="0 0 140 100" fill="none">
    <rect x="3" y="3" width="134" height="94" rx="12" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.18)" stroke-width="3" />
    <text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle" font-size="32" font-family="'Segoe UI', sans-serif" fill="rgba(255,255,255,0.82)">${safeLabel}</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

const FILE_PLACEHOLDER_IMAGE = buildPlaceholderDataUrl('FILE');
const PDF_PLACEHOLDER_IMAGE = buildPlaceholderDataUrl('PDF');

function isPdfFile(file) {
  if (!file) return false;
  const type = (file.type || '').toLowerCase();
  if (type === 'application/pdf') {
    return true;
  }
  const name = (file.name || '').toLowerCase();
  return name.endsWith('.pdf');
}

function formatAggregateName(files) {
  if (!files || !files.length) {
    return 'upload';
  }
  const first = files[0].name || `upload-${Date.now()}`;
  if (files.length === 1) {
    return first;
  }
  return `${first} 他${files.length - 1}件`;
}

function createQueueItem(files) {
  const normalized = Array.from(files || []).filter(Boolean);
  if (!normalized.length) {
    return null;
  }
  const id = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const previews = normalized.map((file, index) => {
    const previewId = `${id}-${index}`;
    const name = file.name || `upload-${index + 1}`;
    const pdf = isPdfFile(file);
    const url = pdf ? null : URL.createObjectURL(file);
    return {
      id: previewId,
      name,
      url,
      status: '待機中',
      type: pdf ? 'pdf' : 'image',
    };
  });

  return {
    id,
    files: normalized,
    name: formatAggregateName(normalized),
    previewEntries: previews,
    historyId: null,
  };
}

function escapeCssIdentifier(value) {
  if (window.CSS && typeof window.CSS.escape === 'function') {
    return window.CSS.escape(value);
  }
  return String(value).replace(/[^a-zA-Z0-9_-]/g, (char) => `\\${char}`);
}

function createStatusInfo(status = 'pending') {
  return {
    status,
    startedAt: null,
    finishedAt: null,
    elapsedSeconds: null,
  };
}

function getNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function computeElapsedSeconds(info) {
  if (!info) return null;
  if (info.status === 'running' && typeof info.startedAt === 'number') {
    const seconds = Math.max(0, (getNow() - info.startedAt) / 1000);
    info.elapsedSeconds = seconds;
    return seconds;
  }
  if (typeof info.elapsedSeconds === 'number') {
    return info.elapsedSeconds;
  }
  return null;
}

function formatStatusLabel(info) {
  if (!info) return '';
  const statusKey = info.status;
  const statusLabel = MODEL_STATUS_LABELS[statusKey] || statusKey || '';
  const elapsed = computeElapsedSeconds(info);
  if (elapsed == null) {
    return statusLabel;
  }
  if (statusKey === 'running') {
    return `${statusLabel} (${elapsed.toFixed(1)}秒経過)`;
  }
  return `${statusLabel} (${elapsed.toFixed(1)}秒)`;
}

function updateModelTimerDisplay(key) {
  const info = modelStatusMap.get(key);
  if (!info) {
    stopModelTimer(key);
    return;
  }
  const selector = `.model-option-status[data-model-key="${escapeCssIdentifier(key)}"]`;
  const badge = document.querySelector(selector);
  if (badge) {
    badge.textContent = formatStatusLabel(info);
  }
  updateModelInfo();
}

function startModelTimer(key) {
  stopModelTimer(key);
  updateModelTimerDisplay(key);
  const handle = setInterval(() => {
    const info = modelStatusMap.get(key);
    if (!info || info.status !== 'running' || typeof info.startedAt !== 'number') {
      stopModelTimer(key);
      updateModelTimerDisplay(key);
      return;
    }
    updateModelTimerDisplay(key);
  }, 200);
  modelTimerHandles.set(key, handle);
}

function stopModelTimer(key) {
  const handle = modelTimerHandles.get(key);
  if (handle) {
    clearInterval(handle);
    modelTimerHandles.delete(key);
  }
}

function stopAllModelTimers() {
  modelTimerHandles.forEach((handle) => {
    clearInterval(handle);
  });
  modelTimerHandles.clear();
}

let currentTextMode = 'markdown';
let processing = false;
const queue = [];
let currentItem = null;
let lastResult = null;
let modalContext = null;
let historyEntries = [];
let activeHistoryId = null;
const inputPreviews = [];
const textCardRegistry = [];
let historyInputImages = [];
let progressTimer = null;
let progressValue = 0;
let progressSession = 0;
let activeProgressSession = 0;
let availableModels = [];
let selectedModels = new Set();
const modelOptionSelections = new Map();
let serverStatusText = 'モデル初期化待機中…';
let modelStatusMap = new Map();
let activeAggregate = null;
const MATH_LINEBREAK_ENVS = new Set([
  'align',
  'align*',
  'alignat',
  'alignat*',
  'aligned',
  'aligned*',
  'alignedat',
  'alignedat*',
  'cases',
  'gather',
  'gather*',
  'gathered',
  'multline',
  'multline*',
  'split',
]);

function labelForModel(key) {
  const descriptor = availableModels.find((item) => item.key === key);
  if (descriptor) {
    return descriptor.label;
  }
  if (!key) {
    return '不明なモデル';
  }
  return key;
}

function sortModelKeys(keys) {
  return [...keys].sort((a, b) => {
    const aPriority = MODEL_PRIORITIES[a] ?? 99;
    const bPriority = MODEL_PRIORITIES[b] ?? 99;
    if (aPriority === bPriority) {
      return a.localeCompare(b);
    }
    return aPriority - bPriority;
  });
}

function formatModelStatusText() {
  if (!modelStatusMap.size) {
    return null;
  }
  const parts = sortModelKeys(modelStatusMap.keys()).map((key) => {
    const info = modelStatusMap.get(key) || createStatusInfo('pending');
    const label = formatStatusLabel(info) || info.status || '';
    return `${labelForModel(key)}:${label}`;
  });
  return parts.join(' / ');
}

function updateModelInfo(statusText = null) {
  if (typeof statusText === 'string') {
    serverStatusText = statusText;
  }
  if (!infoEl) {
    return;
  }

  const statusSummary = formatModelStatusText();
  if (statusSummary) {
    infoEl.textContent = `サーバー状態: ${serverStatusText} / モデル進行状況: ${statusSummary}`;
    return;
  }

  const selectedLabels = getSelectedModels().map(labelForModel).join(', ') || '未選択';
  infoEl.textContent = `サーバー状態: ${serverStatusText} / 選択中: ${selectedLabels}`;
}

function persistModelSelection() {
  try {
    const stored = JSON.stringify([...selectedModels]);
    localStorage.setItem(MODEL_SELECTION_STORAGE_KEY, stored);
  } catch (error) {
    console.debug('モデル選択の保存に失敗しました', error);
  }
}

function persistModelOptions() {
  try {
    const serialized = {};
    modelOptionSelections.forEach((options, modelKey) => {
      if (!modelKey || !options || typeof options !== 'object') {
        return;
      }
      serialized[modelKey] = { ...options };
    });
    localStorage.setItem(MODEL_OPTIONS_STORAGE_KEY, JSON.stringify(serialized));
  } catch (error) {
    console.debug('モデルオプションの保存に失敗しました', error);
  }
}

function loadStoredModelOptions() {
  try {
    const stored = localStorage.getItem(MODEL_OPTIONS_STORAGE_KEY);
    if (!stored) {
      return;
    }
    const parsed = JSON.parse(stored);
    if (!parsed || typeof parsed !== 'object') {
      return;
    }
    Object.entries(parsed).forEach(([modelKey, options]) => {
      if (!modelKey || !options || typeof options !== 'object') {
        return;
      }
      modelOptionSelections.set(modelKey, { ...options });
    });
  } catch (error) {
    console.debug('モデルオプションの読み込みに失敗しました', error);
  }
}

function ensureModelOptionDefaults() {
  const validModelKeys = new Set(availableModels.map((model) => model.key));
  [...modelOptionSelections.keys()].forEach((key) => {
    if (!validModelKeys.has(key)) {
      modelOptionSelections.delete(key);
    }
  });

  availableModels.forEach((model) => {
    if (!model || !model.key) {
      return;
    }
    const current = { ...(modelOptionSelections.get(model.key) || {}) };
    if (!Array.isArray(model.options) || !model.options.length) {
      modelOptionSelections.set(model.key, current);
      return;
    }
    model.options.forEach((option) => {
      if (!option || !option.key) {
        return;
      }
      if (typeof current[option.key] !== 'boolean') {
        current[option.key] = !!option.default;
      }
    });
    modelOptionSelections.set(model.key, current);
  });

  persistModelOptions();
}

function getModelOptionValue(modelKey, optionKey) {
  const options = modelOptionSelections.get(modelKey);
  if (!options || typeof options !== 'object') {
    return false;
  }
  return !!options[optionKey];
}

function setModelOptionValue(modelKey, optionKey, value) {
  const options = { ...(modelOptionSelections.get(modelKey) || {}) };
  options[optionKey] = !!value;
  modelOptionSelections.set(modelKey, options);
  persistModelOptions();
}

function buildModelOptionPayload(modelKey) {
  const options = modelOptionSelections.get(modelKey);
  if (!options || typeof options !== 'object') {
    return null;
  }
  const normalized = {};
  let hasValue = false;
  Object.entries(options).forEach(([key, value]) => {
    if (typeof value === 'boolean') {
      normalized[key] = value;
      hasValue = true;
    }
  });
  if (!hasValue) {
    return null;
  }
  return { [modelKey]: normalized };
}

function renderModelOptions() {
  if (!modelOptionsEl) {
    return;
  }

  modelOptionsEl.innerHTML = '';
  const fragment = document.createDocumentFragment();

  if (!availableModels.length) {
    const fallback = document.createElement('p');
    fallback.className = 'metadata';
    fallback.textContent = '利用できるモデル情報を取得できませんでした。';
    fragment.appendChild(fallback);
    modelOptionsEl.appendChild(fragment);
    updateModelInfo();
    return;
  }

  availableModels.forEach((model) => {
    const option = document.createElement('div');
    option.className = 'model-option';

    const mainRow = document.createElement('label');
    mainRow.className = 'model-option-main';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = model.key;
    checkbox.checked = selectedModels.has(model.key);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        selectedModels.add(model.key);
      } else if (selectedModels.size > 1) {
        selectedModels.delete(model.key);
      } else {
        checkbox.checked = true;
      }
      updateModelInfo();
      persistModelSelection();
    });

    const labelContainer = document.createElement('div');
    labelContainer.className = 'model-option-text';

    const label = document.createElement('span');
    label.className = 'model-option-label';
    label.textContent = model.label;
    labelContainer.appendChild(label);

    if (model.description) {
      const description = document.createElement('span');
      description.className = 'model-option-description';
      description.textContent = model.description;
      labelContainer.appendChild(description);
    }

    mainRow.append(checkbox, labelContainer);
    option.appendChild(mainRow);

    const info = modelStatusMap.get(model.key);
    const statusKey = info?.status;
    if (statusKey) {
      const statusBadge = document.createElement('span');
      const className = MODEL_STATUS_CLASSES[statusKey] || 'pending';
      statusBadge.className = `model-option-status model-status-${className}`;
      statusBadge.dataset.modelKey = model.key;
      statusBadge.dataset.status = statusKey;
      const labelText = formatStatusLabel(info) || MODEL_STATUS_LABELS[statusKey] || statusKey;
      statusBadge.textContent = labelText;
      option.appendChild(statusBadge);
    }

    if (Array.isArray(model.options) && model.options.length) {
      const settings = document.createElement('div');
      settings.className = 'model-option-settings';
      model.options.forEach((opt) => {
        if (!opt || !opt.key) {
          return;
        }
        const optLabel = document.createElement('label');
        optLabel.className = 'model-suboption';

        const optCheckbox = document.createElement('input');
        optCheckbox.type = 'checkbox';
        optCheckbox.value = opt.key;
        optCheckbox.checked = getModelOptionValue(model.key, opt.key);
        optCheckbox.addEventListener('change', () => {
          setModelOptionValue(model.key, opt.key, optCheckbox.checked);
        });

        const optBody = document.createElement('div');
        optBody.className = 'model-suboption-body';

        const optLabelText = document.createElement('span');
        optLabelText.className = 'model-suboption-label';
        optLabelText.textContent = opt.label;
        optBody.appendChild(optLabelText);

        if (opt.description) {
          const optDesc = document.createElement('span');
          optDesc.className = 'model-suboption-description';
          optDesc.textContent = opt.description;
          optBody.appendChild(optDesc);
        }

        optLabel.append(optCheckbox, optBody);
        settings.appendChild(optLabel);
      });
      option.appendChild(settings);
    }

    fragment.appendChild(option);
  });

  modelOptionsEl.appendChild(fragment);
  updateModelInfo();
}

function resetModelStatuses(models) {
  stopAllModelTimers();
  modelStatusMap = new Map();
  sortModelKeys(models).forEach((key) => {
    modelStatusMap.set(key, createStatusInfo('pending'));
  });
  updateModelInfo();
  renderModelOptions();
}

function setModelStatus(key, status) {
  if (!key) {
    return;
  }

  const now = getNow();
  const existing = modelStatusMap.get(key);
  const info = existing ? { ...existing } : createStatusInfo();
  info.status = status;

  if (status === 'running') {
    if (typeof info.startedAt !== 'number') {
      info.startedAt = now;
    }
    info.finishedAt = null;
    modelStatusMap.set(key, info);
    updateModelInfo();
    renderModelOptions();
    startModelTimer(key);
    return;
  }

  if (typeof info.startedAt === 'number' && info.startedAt !== null) {
    info.elapsedSeconds = Math.max(0, (now - info.startedAt) / 1000);
  }
  if (status === 'pending') {
    info.startedAt = null;
    info.finishedAt = null;
    info.elapsedSeconds = null;
  } else {
    info.finishedAt = now;
  }

  modelStatusMap.set(key, info);
  stopModelTimer(key);
  updateModelInfo();
  renderModelOptions();
  updateModelTimerDisplay(key);
}

function clearModelStatuses() {
  stopAllModelTimers();
  modelStatusMap.clear();
  updateModelInfo();
  renderModelOptions();
}

function loadStoredModelSelection() {
  try {
    const stored = localStorage.getItem(MODEL_SELECTION_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length) {
        selectedModels = new Set(parsed);
      }
    }
  } catch (error) {
    console.debug('モデル選択の読み込みに失敗しました', error);
  }
}

function ensureModelSelection() {
  selectedModels = new Set([...selectedModels].filter((model) => availableModels.some((item) => item.key === model)));
  if (!selectedModels.size && availableModels.length) {
    const defaultModel = availableModels.find((item) => item.key === 'deepseek') || availableModels[0];
    if (defaultModel) {
      selectedModels.add(defaultModel.key);
    }
  }
}

function getSelectedModels() {
  return sortModelKeys(selectedModels);
}

async function fetchModels() {
  let models = [];
  try {
    const res = await fetch('/api/models');
    if (!res.ok) {
      throw new Error(res.statusText);
    }
    const data = await res.json();
    if (Array.isArray(data) && data.length) {
      models = data;
    }
  } catch (error) {
    console.warn('モデルリストの取得に失敗しました。既定値を使用します。', error);
    models = [
      {
        key: 'yomitoku',
        label: 'YomiToku Document Analyzer',
        description: '',
        options: [
          {
            key: 'figure_letter',
            label: '絵や図の中の文字も抽出',
            description: 'YomiToku の --figure_letter オプション相当。',
          },
        ],
      },
      { key: 'deepseek', label: 'DeepSeek OCR', description: '', options: [] },
      {
        key: 'deepseek-4bit',
        label: 'DeepSeek OCR (4-bit Quantized)',
        description: 'BitsAndBytes 4-bit build (CUDA GPU required).',
        options: [],
      },
    ];
  }

  models.sort((a, b) => {
    const aPriority = MODEL_PRIORITIES[a.key] ?? 99;
    const bPriority = MODEL_PRIORITIES[b.key] ?? 99;
    if (aPriority === bPriority) {
      return a.key.localeCompare(b.key);
    }
    return aPriority - bPriority;
  });

  availableModels = models;
  ensureModelSelection();
  ensureModelOptionDefaults();
  updateModelInfo('モデル一覧取得済み');
  renderModelOptions();
  persistModelSelection();
}

function isLikelyMathContent(source) {
  if (!source) {
    return false;
  }

  const text = source.trim();
  if (!text) {
    return false;
  }

  if (/\\begin\{[^}]+}/.test(text) || /\\[\[(]/.test(text)) {
    return true;
  }

  let score = 0;

  if (/\\[a-zA-Z]+/.test(text)) {
    score += 2;
  }

  if (/[_^]\s*[({\[]?[-+0-9a-zA-Z\\]/.test(text)) {
    score += 2;
  } else if (/[_^]/.test(text)) {
    score += 1;
  }

  if (/[=<>±≈≠≤≥]/.test(text)) {
    score += 1;
  }

  if (/[0-9]/.test(text) && /[a-zA-Z]/.test(text)) {
    score += 1;
  }

  if (/[∑∏√∞→←∀∃∂∫⊂⊆×·⋅∇]/.test(text)) {
    score += 2;
  }

  if (/\\(frac|sum|int|operatorname|mathcal|mathbf|mathrm|left|right|cdot|times|pm|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)/.test(text)) {
    score += 2;
  }

  if (/\b(?:sin|cos|tan|log|exp|min|max)\b/i.test(text)) {
    score += 1;
  }

  return score >= 3;
}

function sanitizeMathContent(source) {
  if (!source) {
    return source;
  }

  const normalized = source.replace(/\\n/g, '\n');
  const placeholder = '\uFFF0';
  const escapedText = normalized.replace(/\\text\{([^{}]*)\}/g, (match, inner) => {
    const preserved = inner.replace(/\\_/g, placeholder);
    const escaped = preserved.replace(/_/g, '\\_').replace(new RegExp(placeholder, 'g'), '\\_');
    return `\\text{${escaped}}`;
  });
  const normalizedMath = ensureDisplayMathLineBreaks(escapedText);
  return ensureMathLineBreaks(normalizedMath);
}

function ensureDisplayMathLineBreaks(math) {
  if (!math || !/\\begin\{(align\*?|aligned|array|cases|matrix)/.test(math)) {
    return math;
  }

  return math.replace(/\\begin\{(align\*?|aligned|array|cases|matrix)\}([\s\S]*?)\\end\{\1\}/g, (match, envName, body) => {
    const lines = body.split('\n');
    if (lines.length <= 1) {
      return match;
    }

    const updated = lines.map((line, index) => {
      if (index === lines.length - 1) {
        return line;
      }

      return /\\\\\s*$/.test(line)
        ? line
        : `${line.replace(/\s*$/, '')} \\`;
    }).join('\n');

    return `\\begin{${envName}}${updated}\\end{${envName}}`;
  });
}

function ensureMathLineBreaks(math) {
  if (!math) {
    return math;
  }

  const envPattern = /\\begin\{([a-zA-Z*]+)\}([\s\S]*?)\\end\{\1\}/g;

  return math.replace(envPattern, (match, env, body) => {
    if (!MATH_LINEBREAK_ENVS.has(env)) {
      return match;
    }

    const normalizedBody = body.replace(/\\[ \t]+(?=\S)/g, () => '\\' + '\n');
    const rows = normalizedBody.split(/\r?\n/);
    const newline = normalizedBody.includes('\r\n') ? '\r\n' : '\n';
    let lastNonEmpty = -1;

    for (let i = rows.length - 1; i >= 0; i -= 1) {
      if (rows[i].trim()) {
        lastNonEmpty = i;
        break;
      }
    }

    if (lastNonEmpty === -1) {
      return match;
    }

    const breakPattern = /\\\\(\[[^\]]*])?(?:\\[a-zA-Z]+|[,;:!?])?$/;

    const processed = rows.map((row, index) => {
      if (index >= lastNonEmpty) {
        return row;
      }

      if (!row.trim()) {
        return row;
      }

      const trimmedEnd = row.trimEnd();
      if (breakPattern.test(trimmedEnd)) {
        return row;
      }

      const trailingWhitespace = row.slice(trimmedEnd.length);
      return `${trimmedEnd}\\\\${trailingWhitespace}`;
    });

    const updatedBody = processed.join(newline);
    return match.replace(body, updatedBody);
  });
}



async function pingServer() {
  try {
    const res = await fetch('/api/ping');
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    updateModelInfo(`オンライン (${data.status})`);
  } catch (error) {
    updateModelInfo('サーバーに接続できません');
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? '#ff6b6b' : '#cfd8dc';
}

function showProgress() {
  if (!progressWrapper || !progressBar) return;
  progressWrapper.classList.remove('hidden');
  progressWrapper.setAttribute('aria-hidden', 'false');
}

function hideProgress(force = false) {
  if (!progressWrapper || !progressBar) return;
  if (!force && processing) return;
  progressWrapper.classList.add('hidden');
  progressWrapper.setAttribute('aria-hidden', 'true');
  progressBar.style.width = '0%';
  progressValue = 0;
  if (force) {
    activeProgressSession = 0;
  }
}

function startFakeProgress() {
  if (!progressWrapper || !progressBar) return 0;
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  progressValue = 0;
  progressBar.style.width = '0%';
  showProgress();
  progressSession += 1;
  activeProgressSession = progressSession;
  const sessionId = progressSession;

  progressTimer = setInterval(() => {
    const target = progressValue < 55
      ? progressValue + (Math.random() * 6 + 3)
      : progressValue < 80
        ? progressValue + (Math.random() * 4 + 1)
        : progressValue + (Math.random() * 1.2 + 0.1);
    progressValue = Math.min(target, 95);
    progressBar.style.width = `${progressValue.toFixed(1)}%`;
  }, 400);

  return sessionId;
}

function settleProgress(sessionId, success) {
  if (!progressWrapper || !progressBar) return;
  if (sessionId === 0 || sessionId !== activeProgressSession) {
    return;
  }
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  progressValue = success ? 100 : Math.max(progressValue, 98);
  progressBar.style.width = `${progressValue}%`;
  const delay = success ? 500 : 1200;
  setTimeout(() => {
    if (sessionId === activeProgressSession) {
      hideProgress(true);
    }
  }, delay);
}

function findPreview(id) {
  return inputPreviews.find((entry) => entry.id === id);
}

function setHistoryInputImages(images, fallbackName = 'ocr', historyId = 'history') {
  historyInputImages = [];
  if (Array.isArray(images)) {
    images.forEach((image, index) => {
      if (!image || !image.url) {
        return;
      }
      const baseName = image.name || fallbackName || `input-${index + 1}`;
      const identifier = image.path || image.url || `${index}`;
      historyInputImages.push({
        id: `history-${historyId}-${identifier}`,
        name: baseName,
        url: image.url,
        status: '完了',
        order: index,
      });
    });
  }
  renderPreviews();
}

function clearHistoryInputImages() {
  if (!historyInputImages.length) {
    return;
  }
  historyInputImages = [];
  renderPreviews();
}

function extractElapsedSeconds(variant) {
  if (!variant) {
    return null;
  }
  if (typeof variant.elapsedSeconds === 'number') {
    return variant.elapsedSeconds;
  }
  if (variant.metadata && typeof variant.metadata.elapsed_seconds === 'number') {
    return variant.metadata.elapsed_seconds;
  }
  return null;
}

function applyHistoryModelStatuses(variants) {
  if (processing || queue.length) {
    return;
  }
  if (!Array.isArray(variants) || !variants.length) {
    clearModelStatuses();
    return;
  }

  const updated = new Map();
  variants.forEach((variant) => {
    const key = variant?.key || variant?.model;
    if (!key) {
      return;
    }
    const info = createStatusInfo('success');
    const elapsed = extractElapsedSeconds(variant);
    if (typeof elapsed === 'number') {
      info.elapsedSeconds = elapsed;
    }
    updated.set(key, info);
  });

  const coverageSource = modelStatusMap.size
    ? [...modelStatusMap.keys()]
    : availableModels.map((model) => model.key);
  const coverageKeys = sortModelKeys([...new Set(coverageSource.filter(Boolean))]);

  coverageKeys.forEach((key) => {
    if (!updated.has(key)) {
      updated.set(key, createStatusInfo('pending'));
    }
  });

  stopAllModelTimers();
  modelStatusMap = updated;
  updateModelInfo();
  renderModelOptions();
}

function dedupeInputPreviews() {
  const seen = new Set();
  for (let index = inputPreviews.length - 1; index >= 0; index -= 1) {
    const entry = inputPreviews[index];
    if (!entry || !entry.id) {
      inputPreviews.splice(index, 1);
      continue;
    }
    if (seen.has(entry.id)) {
      const [removed] = inputPreviews.splice(index, 1);
      if (removed?.url && removed.url.startsWith('blob:')) {
        URL.revokeObjectURL(removed.url);
      }
      continue;
    }
    seen.add(entry.id);
  }
}

function clearLiveInputPreviews() {
  for (let index = inputPreviews.length - 1; index >= 0; index -= 1) {
    const entry = inputPreviews[index];
    if (!entry || entry.status === '履歴') {
      continue;
    }
    const [removed] = inputPreviews.splice(index, 1);
    if (removed?.url && removed.url.startsWith('blob:')) {
      URL.revokeObjectURL(removed.url);
    }
  }
}

function renderPreviews() {
  if (!inputPreviewSection || !inputPreviewGrid) return;
  dedupeInputPreviews();

  const historyItems = historyInputImages
    .slice()
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((entry, index) => ({ ...entry, renderOrder: index }));

  const currentItems = [...inputPreviews]
    .sort((a, b) => (b.addedAt || 0) - (a.addedAt || 0))
    .map((entry, index) => ({ ...entry, renderOrder: historyItems.length + index }));

  const combined = [...historyItems, ...currentItems];
  const hasItems = combined.length > 0;
  inputPreviewSection.classList.toggle('hidden', !hasItems);

  if (!hasItems) {
    inputPreviewGrid.innerHTML = '';
    return;
  }

  const fragment = document.createDocumentFragment();

  combined.forEach((entry) => {
    const item = document.createElement('div');
    item.className = 'preview-item';
    if (entry.status) {
      item.dataset.status = entry.status;
    } else if (item.dataset.status) {
      delete item.dataset.status;
    }

    if (entry.status) {
      const badge = document.createElement('span');
      badge.className = 'preview-badge';
      badge.textContent = entry.status;
      item.appendChild(badge);
    }

    const img = document.createElement('img');
    img.src = entry.url || FILE_PLACEHOLDER_IMAGE;
    img.alt = `${entry.name || '入力画像'} プレビュー`;
    img.className = 'preview-thumb';
    if (entry.url) {
      img.addEventListener('click', () => {
        openModal({ type: 'input', url: entry.url, name: entry.name });
      });
    }
    item.appendChild(img);

    const label = document.createElement('div');
    label.className = 'preview-label';
    label.textContent = entry.name || '入力画像';
    item.appendChild(label);

    fragment.appendChild(item);
  });

  inputPreviewGrid.innerHTML = '';
  inputPreviewGrid.appendChild(fragment);
}

function addInputPreview({ id, name, url, status, type }) {
  if (!id) return;
  const previewUrl = url || (type === 'pdf' ? PDF_PLACEHOLDER_IMAGE : FILE_PLACEHOLDER_IMAGE);
  const existing = findPreview(id);
  if (existing) {
    existing.status = status;
    existing.name = name;
    existing.url = previewUrl;
    renderPreviews();
    return;
  }

  inputPreviews.push({ id, name, url: previewUrl, status, type, addedAt: Date.now() });
  renderPreviews();
}

function updatePreviewStatus(id, status) {
  const entry = findPreview(id);
  if (!entry) return;
  entry.status = status;
  renderPreviews();
}

function updateQueueItemPreviewStatus(item, status) {
  if (!item || !Array.isArray(item.previewEntries)) {
    return;
  }
  let mutated = false;
  item.previewEntries.forEach((preview) => {
    if (!preview) return;
    preview.status = status;
    const entry = findPreview(preview.id);
    if (entry) {
      entry.status = status;
      mutated = true;
    }
  });
  if (mutated) {
    renderPreviews();
  }
}

function removeQueueItemPreviews(item) {
  if (!item || !Array.isArray(item.previewEntries)) {
    return;
  }
  const targetIds = new Set(
    item.previewEntries
      .map((preview) => preview?.id)
      .filter((value) => typeof value === 'string')
  );
  if (!targetIds.size) {
    return;
  }
  for (let index = inputPreviews.length - 1; index >= 0; index -= 1) {
    const entry = inputPreviews[index];
    if (!entry || !targetIds.has(entry.id)) {
      continue;
    }
    const [removed] = inputPreviews.splice(index, 1);
    if (removed?.url && removed.url.startsWith('blob:')) {
      URL.revokeObjectURL(removed.url);
    }
  }
  item.previewEntries = [];
  renderPreviews();
}

function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function renderMath(container, attempt = 0) {
  if (!container) {
    return;
  }

  fallbackMathRender(container);

  const renderFn = window.renderMathInElement;
  if (typeof renderFn === 'function') {
    try {
      renderFn(container, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
        strict: 'ignore',
      });
    } catch (error) {
      console.error('Failed to render math', error);
    }
    return;
  }

  if (attempt >= 20) {
    console.warn('KaTeX auto-render not ready after retries');
    return;
  }

  setTimeout(() => renderMath(container, attempt + 1), 200);
}

function normalizeMathBlocks(markdown) {
  if (!markdown) {
    return markdown;
  }

  const wrapDisplayMatch = (prefix, indent, body, fallback) => {
    const trimmed = (body ?? '').trim();
    if (!trimmed || /^\\\[|^\$\$/.test(trimmed)) {
      return fallback;
    }
    if (!isLikelyMathContent(trimmed)) {
      return fallback;
    }

    const cleaned = sanitizeMathContent(trimmed);
    if (!cleaned) {
      return fallback;
    }

    const safePrefix = prefix || '';
    const safeIndent = indent || '';
    return `${safePrefix}${safeIndent}\\[\n${cleaned}\n${safeIndent}\\]`;
  };

  let processed = markdown;

  const bracketPattern = /(^|\r?\n)([ \t]*)\[\s*([\s\S]*?)\s*\](?=\r?\n|$)/g;
  processed = processed.replace(bracketPattern, (match, prefix, indent, content) =>
    wrapDisplayMatch(prefix, indent, content, match)
  );

  const leftRightPattern = /(^|\r?\n)([ \t]*)(\\left[\s\S]*?\\right[\.\}\)\]]?)(?=\r?\n|$)/g;
  processed = processed.replace(leftRightPattern, (match, prefix, indent, content) =>
    wrapDisplayMatch(prefix, indent, content, match)
  );

  const environmentPattern = /(^|\r?\n)([ \t]*)(\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\})(?=\r?\n|$)/g;
  processed = processed.replace(environmentPattern, (match, prefix, indent, content) =>
    wrapDisplayMatch(prefix, indent, content, match)
  );

  return processed;
}

function normalizeMathNodes(container) {
  if (!container) {
    return;
  }

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node || !node.textContent) {
      continue;
    }

    let parent = node.parentNode;
    let skip = false;
    while (parent) {
      if (parent.nodeType === Node.ELEMENT_NODE && ['CODE', 'PRE', 'TEXTAREA'].includes(parent.tagName)) {
        skip = true;
        break;
      }
      parent = parent.parentNode;
    }
    if (skip) {
      continue;
    }

    const text = node.textContent;
    if (!text) {
      continue;
    }

    const leading = text.match(/^\s*/)?.[0] ?? '';
    const trailing = text.match(/\s*$/)?.[0] ?? '';
    const trimmed = text.slice(leading.length, text.length - trailing.length);

    if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) {
      continue;
    }
    if (trimmed.startsWith('\\[') || trimmed.startsWith('$$')) {
      continue;
    }

    const body = trimmed.slice(1, -1).trim();
    if (!isLikelyMathContent(body)) {
      continue;
    }

    const inner = sanitizeMathContent(body);
    if (!inner) {
      continue;
    }

    const newline = text.includes('\r\n') ? '\r\n' : '\n';
    node.textContent = `${leading}\\[${newline}${inner}${newline}\\]${trailing}`;
  }
}

function normalizeMathElements(container) {
  if (!container) {
    return;
  }

  const elements = container.querySelectorAll('p, div');
  elements.forEach((element) => {
    element.querySelectorAll('br').forEach((br) => {
      const replacement = document.createTextNode('\\\\\n');
      br.replaceWith(replacement);
    });

    const text = element.textContent;
    if (!text) {
      return;
    }

    const trimmed = text.trim();
    if (trimmed.length < 2 || (!trimmed.startsWith('[') && !trimmed.startsWith('\\['))) {
      return;
    }

    let body = trimmed;
    if (body.startsWith('\\[') && body.endsWith('\\]')) {
      body = body.slice(2, -2).trim();
    } else if (body.startsWith('[') && body.endsWith(']')) {
      body = body.slice(1, -1).trim();
    } else {
      return;
    }

    if (!isLikelyMathContent(body)) {
      return;
    }

    const cleaned = sanitizeMathContent(body);
    if (!cleaned) {
      return;
    }

    const newline = text.includes('\r\n') ? '\r\n' : '\n';
    element.textContent = `\\[${newline}${cleaned}${newline}\\]`;
  });
}

function fallbackMathRender(container) {
  if (!container) {
    return;
  }

  const katexLib = window.katex;
  if (!katexLib || typeof katexLib.renderToString !== 'function') {
    return;
  }

  const doc = container.ownerDocument;
  const walker = doc.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodes = [];

  while (walker.nextNode()) {
    nodes.push(walker.currentNode);
  }

  nodes.forEach((node) => {
    if (!node?.textContent) {
      return;
    }

    const parentElement = node.parentElement;
    if (!parentElement) {
      return;
    }

    if (parentElement.closest('code, pre, textarea, script, style') || parentElement.closest('.katex')) {
      return;
    }

    const text = node.textContent;
    if (!text) {
      return;
    }

    const hasExplicitDelim = text.includes('\\[') || text.includes('\\(');
    if (!hasExplicitDelim) {
      const trimmedCheck = text.trim();
      let candidate = trimmedCheck;
      if (trimmedCheck.startsWith('[') && trimmedCheck.endsWith(']')) {
        candidate = trimmedCheck.slice(1, -1).trim();
      }
      if (!isLikelyMathContent(candidate)) {
        return;
      }
    }

    const regex = /\\\[([\s\S]*?)\\\]|\\\(([^]*?)\\\)/g;
    let match;
    let lastIndex = 0;
    const fragments = [];

    while ((match = regex.exec(text)) !== null) {
      const matchStart = match.index;
      if (matchStart > lastIndex) {
        fragments.push({ type: 'text', value: text.slice(lastIndex, matchStart) });
      }

      const matchText = match[0];
      const content = (match[1] ?? match[2] ?? '').trim();
      const displayMode = match[1] !== undefined;
      fragments.push({ type: 'math', raw: matchText, content, displayMode });
      lastIndex = match.index + matchText.length;
    }

    if (fragments.length === 0) {
      const trimmed = text.trim();
      let candidate = trimmed;
      if (trimmed.startsWith('\\[') && trimmed.endsWith('\\]')) {
        candidate = trimmed.slice(2, -2).trim();
      } else if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        candidate = trimmed.slice(1, -1).trim();
      }

      if (!isLikelyMathContent(candidate)) {
        return;
      }

      const inner = sanitizeMathContent(candidate);
      if (!inner) {
        return;
      }

      try {
        const html = katexLib.renderToString(inner, {
          displayMode: true,
          throwOnError: false,
          strict: 'ignore',
        });
        const temp = doc.createElement('div');
        temp.innerHTML = html;
        const frag = doc.createDocumentFragment();
        while (temp.firstChild) {
          frag.appendChild(temp.firstChild);
        }
        const parent = node.parentNode;
        parent.insertBefore(frag, node);
        parent.removeChild(node);
      } catch (error) {
        console.error('KaTeX fallback render failed', error);
      }
      return;
    }

    if (lastIndex < text.length) {
      fragments.push({ type: 'text', value: text.slice(lastIndex) });
    }

    const parent = node.parentNode;
    const reference = node;

    fragments.forEach((fragment) => {
      if (fragment.type === 'text') {
        if (fragment.value) {
          parent.insertBefore(doc.createTextNode(fragment.value), reference);
        }
        return;
      }

      const mathSource = sanitizeMathContent(fragment.content);
      let html = fragment.raw;
      if (mathSource) {
        try {
          html = katexLib.renderToString(mathSource, {
            displayMode: fragment.displayMode,
            throwOnError: false,
            strict: 'ignore',
          });
        } catch (error) {
          console.error('KaTeX fallback render failed', error);
        }
      }

      const temp = doc.createElement('div');
      temp.innerHTML = html;
      const frag = doc.createDocumentFragment();
      while (temp.firstChild) {
        frag.appendChild(temp.firstChild);
      }
      parent.insertBefore(frag, reference);
    });

    parent.removeChild(node);
  });
}

function transformMarkdown(markdown, variant) {
  if (!markdown) {
    return markdown;
  }

  let processed = normalizeMathBlocks(markdown);

  const replacements = new Map();

  const addMapping = (rawKey, value) => {
    if (!rawKey || !value) return;
    const normalized = rawKey.replace(/\\/g, '/');
    const base = normalized.split('?')[0];
    const variants = new Set([
      rawKey,
      normalized,
      base,
      base.replace(/^\.\/?/, ''),
      base.replace(/^\.\.?\//, ''),
      base.replace(/^artifacts\//, ''),
      base.replace(/^\.\/?artifacts\//, ''),
    ]);
    const name = base.split('/').pop();
    if (name) {
      variants.add(name);
    }
    variants.forEach((variantKey) => {
      if (variantKey) {
        replacements.set(variantKey, value);
      }
    });
  };

  if (variant && Array.isArray(variant.crops)) {
    variant.crops.forEach((crop, index) => {
      if (!crop || !crop.url) return;
      if (crop.path) addMapping(crop.path, crop.url);
      if (crop.name) {
        addMapping(crop.name, crop.url);
        addMapping(`images/${crop.name}`, crop.url);
        addMapping(`artifacts/${crop.name}`, crop.url);
      }
      addMapping(String(index), crop.url);
      addMapping(`images/${index}`, crop.url);
      addMapping(`images/${index}.jpg`, crop.url);
      addMapping(`images/${index}.jpeg`, crop.url);
      addMapping(`images/${index}.png`, crop.url);
    });
  }

  if (variant?.boundingUrl) {
    const boundingPath = variant?.metadata?.bounding_image_path || variant?.metadata?.boundingImagePath;
    if (boundingPath) {
      addMapping(boundingPath, variant.boundingUrl);
      const name = boundingPath.split('/').pop();
      if (name) addMapping(name, variant.boundingUrl);
    }
  }

  if (!replacements.size) {
    return processed;
  }

  const resolveAssetPath = (source) => {
    if (!source) return null;
    const candidates = new Set();
    const trimmed = source.trim();
    const noQuery = trimmed.split('?')[0];
    candidates.add(trimmed);
    candidates.add(noQuery);
    candidates.add(noQuery.replace(/^\.\/?/, ''));
    candidates.add(noQuery.replace(/^\.\.?\//, ''));
    candidates.add(noQuery.replace(/\\/g, '/'));
    candidates.add(noQuery.replace(/\\/g, '/').replace(/^\.\/?/, ''));
    candidates.add(noQuery.replace(/\\/g, '/').replace(/^\.\.?\//, ''));
    const name = noQuery.split('/').pop();
    if (name) candidates.add(name);
    for (const key of candidates) {
      if (replacements.has(key)) {
        return replacements.get(key);
      }
    }
    return null;
  };

  processed = processed.replace(/!\[(.*?)\]\(([^)]+)\)/g, (match, alt, path) => {
    const resolved = resolveAssetPath(path);
    if (resolved) {
      return `![${alt}](${resolved})`;
    }
    return match;
  });

  processed = processed.replace(/<img\s+([^>]*?)src=["']([^"']+)["']([^>]*?)>/gi, (match, before, src, after) => {
    const resolved = resolveAssetPath(src);
    if (!resolved) {
      return match;
    }
    return match.replace(src, resolved);
  });

  return processed;
}

function clearResults() {
  closeModal();
  if (textVariantGrid) {
    textVariantGrid.innerHTML = '<p class="metadata">結果はまだありません。</p>';
  }
  if (cropsVariantGrid) {
    cropsVariantGrid.innerHTML = '<p class="metadata">切り出し画像はまだありません。</p>';
  }
  if (boundingVariantGrid) {
    boundingVariantGrid.innerHTML = '<p class="metadata">バウンディング画像はまだありません。</p>';
  }
  textCardRegistry.length = 0;
  lastResult = null;
  clearHistoryInputImages();
}

function updateQueueStatus() {
  if (queueStatus) {
    queueStatus.textContent = '';
  }
  renderQueue();
  renderPreviews();
}

function inferExtensionFromDataUrl(source, fallback = 'png') {
  if (!source) return fallback;
  if (source.startsWith('data:')) {
    const match = source.match(/^data:(.+?);/);
    if (!match) return fallback;
    const mime = match[1];
    if (mime.includes('jpeg')) return 'jpg';
    if (mime.includes('png')) return 'png';
    if (mime.includes('gif')) return 'gif';
    if (mime.includes('webp')) return 'webp';
    return fallback;
  }

  const name = source.split('?')[0]?.split('#')[0];
  const ext = name?.split('.').pop();
  if (!ext) return fallback;
  return ext;
}

function addImageControls(wrapper, crop, filename, index) {
  if (!crop || !crop.url) return;
  const { url, name } = crop;
  const controls = document.createElement('div');
  controls.className = 'img-actions action-buttons';
  const copyBtn = document.createElement('button');
  copyBtn.className = 'icon-button mini';
  copyBtn.title = 'コピー';
  copyBtn.innerHTML = '<svg><use href="#icon-copy" /></svg>';
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    copyImageToClipboard(url).then(() => setStatus('画像をコピーしました。'));
  });
  const downloadBtn = document.createElement('button');
  downloadBtn.className = 'icon-button mini';
  downloadBtn.title = 'ダウンロード';
  downloadBtn.innerHTML = '<svg><use href="#icon-download" /></svg>';
  downloadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const ext = inferExtensionFromDataUrl(url);
    const label = name || `crop-${index + 1}`;
    downloadDataUrl(url, `${filename}-${label}.${ext}`);
  });
  controls.append(copyBtn, downloadBtn);
  wrapper.appendChild(controls);
}


function deriveBaseName(name) {
  if (!name) return 'ocr';
  const trimmed = name.trim();
  if (!trimmed) return 'ocr';
  return trimmed.replace(/\.[^.]+$/, '');
}

function normalizeBoundingImages(images, fallbackUrl = null, label = 'bounding') {
  const result = [];
  if (Array.isArray(images)) {
    images.forEach((item, index) => {
      if (!item) return;
      const url = item.url || null;
      if (!url) return;
      const name = item.name || `${label || 'bounding'}-${index + 1}`;
      const path = item.path || url;
      result.push({ name, path, url });
    });
  }

  if (!result.length && fallbackUrl) {
    result.push({
      name: `${label || 'bounding'}-1`,
      path: fallbackUrl,
      url: fallbackUrl,
    });
  }

  return result;
}

function normalizeVariants(data) {
  if (Array.isArray(data?.variants) && data.variants.length) {
    return data.variants.map((variant, index) => ({
      key: variant.model || `model-${index}`,
      label: variant.label || labelForModel(variant.model || `model-${index}`),
      textPlain: variant.text_plain || '',
      textMarkdown: variant.text_markdown || '',
      boundingUrl: variant.bounding_image_url || null,
      boundingImages: normalizeBoundingImages(variant.bounding_images, variant.bounding_image_url, variant.label),
      crops: Array.isArray(variant.crops) ? variant.crops : [],
      previewUrl: variant.preview_image_url || null,
      metadata: variant.metadata || {},
      preview: variant.preview || '',
      elapsedSeconds: typeof variant.elapsed_seconds === 'number' ? variant.elapsed_seconds : null,
    }));
  }

  return [
    {
      key: data?.metadata?.model || 'deepseek',
      label: data?.metadata?.model_label || labelForModel(data?.metadata?.model || 'deepseek'),
      textPlain: data?.text_plain || '',
      textMarkdown: data?.text_markdown || '',
      boundingUrl: data?.bounding_image_url || null,
      boundingImages: normalizeBoundingImages(data?.bounding_images, data?.bounding_image_url, data?.metadata?.model_label || data?.metadata?.model),
      crops: Array.isArray(data?.crops) ? data.crops : [],
      previewUrl: data?.preview_image_url || null,
      metadata: data?.metadata || {},
      preview: data?.preview || '',
      elapsedSeconds: typeof data?.metadata?.elapsed_seconds === 'number' ? data.metadata.elapsed_seconds : null,
    },
  ];
}

function createIconButton(iconId, title, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'icon-button';
  if (title) {
    button.title = title;
  }
  button.innerHTML = `<svg><use href="#icon-${iconId}" /></svg>`;
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    onClick();
  });
  return button;
}

function createVariantHeader(variant, context, extraMeta = '') {
  const header = document.createElement('div');
  header.className = 'variant-card-header';

  const title = document.createElement('span');
  title.className = 'variant-card-title';
  title.textContent = variant.label;
  header.appendChild(title);

  const metaParts = [];
  if (extraMeta) {
    metaParts.push(extraMeta);
  }
  if (typeof variant.elapsedSeconds === 'number') {
    metaParts.push(`処理: ${variant.elapsedSeconds.toFixed(2)}秒`);
  }
  if (context.createdLabel && context.createdLabel !== '-') {
    metaParts.push(`作成: ${context.createdLabel}`);
  }
  if (metaParts.length) {
    const meta = document.createElement('span');
    meta.className = 'variant-card-meta';
    meta.textContent = metaParts.join(' / ');
    header.appendChild(meta);
  }

  return header;
}

function renderTextVariants(variants, context) {
  if (!textVariantGrid) return;
  textVariantGrid.innerHTML = '';
  textCardRegistry.length = 0;

  if (!variants.length) {
    const empty = document.createElement('p');
    empty.className = 'metadata';
    empty.textContent = 'テキスト出力はありませんでした。';
    textVariantGrid.appendChild(empty);
    return;
  }

  variants.forEach((variant) => {
    const card = document.createElement('div');
    card.className = 'variant-card text-card';
    card.dataset.model = variant.key;

    const header = createVariantHeader(
      variant,
      context,
      `クロップ: ${variant.crops.length}件`
    );

    const actions = document.createElement('div');
    actions.className = 'variant-card-actions';
    actions.appendChild(createIconButton('copy', 'テキストをコピー', () => copyVariantText(variant)));
    actions.appendChild(createIconButton('download', 'テキストをダウンロード', () => downloadVariantText(variant, context.downloadBase)));

    const content = document.createElement('div');
    content.className = 'variant-card-content';

    const markdownContainer = document.createElement('div');
    markdownContainer.className = 'variant-markdown';

    const plainContainer = document.createElement('textarea');
    plainContainer.className = 'variant-plain hidden';
    plainContainer.readOnly = true;

    const processedMarkdown = transformMarkdown(variant.textMarkdown, variant);
    if (processedMarkdown) {
      markdownContainer.innerHTML = window.marked.parse(processedMarkdown);
      normalizeMathNodes(markdownContainer);
      normalizeMathElements(markdownContainer);
      renderMath(markdownContainer);
    } else {
      markdownContainer.innerHTML = '<p>マークダウン出力はありません。</p>';
    }

    plainContainer.value = variant.textPlain || variant.textMarkdown || '';

    content.append(markdownContainer, plainContainer);
    card.append(header, actions, content);
    textVariantGrid.appendChild(card);

    textCardRegistry.push({
      card,
      variant,
      markdownEl: markdownContainer,
      plainEl: plainContainer,
      filename: context.downloadBase,
    });
  });

  updateTextCardsDisplay();
}

function renderCropVariants(variants, context) {
  if (!cropsVariantGrid) return;
  cropsVariantGrid.innerHTML = '';

  if (!variants.length) {
    const empty = document.createElement('p');
    empty.className = 'metadata';
    empty.textContent = '切り出し画像はありませんでした。';
    cropsVariantGrid.appendChild(empty);
    return;
  }

  variants.forEach((variant) => {
    const card = document.createElement('div');
    card.className = 'variant-card crop-card';
    card.dataset.model = variant.key;

    const header = createVariantHeader(
      variant,
      context,
      `切り出し: ${variant.crops.length}件`
    );

    card.appendChild(header);

    if (!variant.crops.length) {
      const empty = document.createElement('p');
      empty.className = 'metadata';
      empty.textContent = '切り出し画像はありませんでした。';
      card.appendChild(empty);
      cropsVariantGrid.appendChild(card);
      return;
    }

    const grid = document.createElement('div');
    grid.className = 'crops-grid';

    variant.crops.forEach((crop, index) => {
      if (!crop || !crop.url) {
        return;
      }
      const wrapper = document.createElement('div');
      wrapper.className = 'crop-item';
      const img = document.createElement('img');
      img.src = crop.url;
      img.alt = `${variant.label} クロップ ${index + 1}`;
      img.dataset.index = index;
      img.addEventListener('click', () => openModal({ type: 'crop', url: crop.url, name: crop.name, variant: variant.key, index }));
      wrapper.appendChild(img);
      addImageControls(wrapper, crop, `${context.downloadBase}-${variant.key}`, index);
      grid.appendChild(wrapper);
    });

    card.appendChild(grid);
    cropsVariantGrid.appendChild(card);
  });
}

function renderBoundingVariants(variants, context) {
  if (!boundingVariantGrid) return;
  boundingVariantGrid.innerHTML = '';

  variants.forEach((variant) => {
    const card = document.createElement('div');
    card.className = 'variant-card bounding-card';
    card.dataset.model = variant.key;

    const boundingImages = Array.isArray(variant.boundingImages)
      ? variant.boundingImages
      : normalizeBoundingImages(variant.bounding_images, variant.boundingUrl, variant.label);

    const header = createVariantHeader(
      variant,
      context,
      boundingImages.length ? `バウンディング: ${boundingImages.length}件` : ''
    );
    card.appendChild(header);

    if (!boundingImages.length) {
      const empty = document.createElement('p');
      empty.className = 'metadata';
      empty.textContent = 'バウンディング画像は生成されませんでした。';
      card.appendChild(empty);
      boundingVariantGrid.appendChild(card);
      return;
    }

    let currentIndex = 0;

    const viewer = document.createElement('div');
    viewer.className = 'bounding-carousel';

    const frame = document.createElement('div');
    frame.className = 'bounding-frame';

    const image = document.createElement('img');
    image.className = 'bounding-image';
    frame.appendChild(image);

    frame.addEventListener('click', () => {
      const current = boundingImages[currentIndex];
      if (!current || !current.url) return;
      openModal({
        type: 'bounding',
        url: current.url,
        name: current.name || `${context.downloadBase}-${variant.key}-bounding-${currentIndex + 1}`,
      });
    });

    const indicator = document.createElement('div');
    indicator.className = 'bounding-indicator';

    const actions = document.createElement('div');
    actions.className = 'variant-card-actions bounding-actions';

    const copyBtn = createIconButton('copy', 'バウンディング画像をコピー', async () => {
      const current = boundingImages[currentIndex];
      if (!current || !current.url) return;
      try {
        await copyImageToClipboard(current.url);
        setStatus('バウンディング画像をコピーしました。');
      } catch (error) {
        console.error(error);
        setStatus('画像をコピーできませんでした。', true);
      }
    });

    const downloadBtn = createIconButton('download', 'バウンディング画像をDL', () => {
      const current = boundingImages[currentIndex];
      if (!current || !current.url) return;
      const ext = inferExtensionFromDataUrl(current.url, 'jpg');
      downloadDataUrl(current.url, `${context.downloadBase}-${variant.key}-bounding-${currentIndex + 1}.${ext}`);
    });

    actions.append(copyBtn, downloadBtn);

    const nav = document.createElement('div');
    nav.className = 'bounding-nav';

    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'button secondary mini nav-button';
    prevBtn.textContent = '← 前へ';

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'button secondary mini nav-button';
    nextBtn.textContent = '次へ →';

    nav.append(prevBtn, indicator, nextBtn);

    const updateViewer = (index) => {
      const total = boundingImages.length;
      if (!total) {
        image.src = '';
        indicator.textContent = '';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
      }
      const bounded = Math.max(0, Math.min(index, total - 1));
      currentIndex = bounded;
      const current = boundingImages[currentIndex];
      image.src = current?.url || '';
      image.alt = `${variant.label} のバウンディング画像 ${currentIndex + 1}/${total}`;
      indicator.textContent = `${currentIndex + 1} / ${total}`;
      prevBtn.disabled = currentIndex === 0;
      nextBtn.disabled = currentIndex === total - 1;
    };

    prevBtn.addEventListener('click', () => {
      if (currentIndex <= 0) return;
      updateViewer(currentIndex - 1);
    });

    nextBtn.addEventListener('click', () => {
      if (currentIndex >= boundingImages.length - 1) return;
      updateViewer(currentIndex + 1);
    });

    viewer.append(frame, nav);

    card.append(viewer, actions);

    updateViewer(0);

    boundingVariantGrid.appendChild(card);
  });
}

function displayResult(data, filename, historyId = null, createdAt = null) {
  const firstDisplay = !lastResult;
  const variants = normalizeVariants(data);
  const effectiveFilename = filename || data?.filename || 'ocr';
  const downloadBase = deriveBaseName(effectiveFilename);
  const createdTimestamp = createdAt || data?.created_at || null;
  const context = {
    filename: effectiveFilename,
    downloadBase,
    createdAt: createdTimestamp,
    createdLabel: formatDate(createdTimestamp),
    warnings: Array.isArray(data?.metadata?.warnings) ? data.metadata.warnings.filter(Boolean) : [],
  };

  lastResult = {
    raw: data,
    variants,
    filename: downloadBase,
    originalFilename: effectiveFilename,
    historyId,
    createdAt: createdTimestamp,
  };

  if (Object.prototype.hasOwnProperty.call(data || {}, 'input_images')) {
    const inputImages = Array.isArray(data.input_images) ? data.input_images : [];
    setHistoryInputImages(inputImages, effectiveFilename, historyId || data?.history_id || 'history');
  }

  applyHistoryModelStatuses(variants);

  renderTextVariants(variants, context);
  renderCropVariants(variants, context);
  renderBoundingVariants(variants, context);

  const modelSummary = variants.map((variant) => `${variant.label}`).join(', ');
  const cropSummary = variants.map((variant) => `${variant.label}:${variant.crops.length}`).join(' / ');
  const warningText = context.warnings.length ? ` / 警告: ${context.warnings.join(' / ')}` : '';
  setStatus(`ファイル: ${effectiveFilename} / モデル: ${modelSummary} / クロップ: ${cropSummary}${warningText}`);

  if (firstDisplay) {
    setTab('markdown');
  } else {
    updateTextCardsDisplay();
  }
}

function resetActiveAggregate(filename) {
  activeAggregate = {
    filename: filename || null,
    variants: new Map(),
    warnings: new Set(),
    createdAt: null,
    historyIds: [],
    inputImages: [],
  };
}

function ensureVariantObject(source, fallbackKey) {
  const modelKey = source?.model || source?.key || fallbackKey;
  const label = source?.label || labelForModel(modelKey);
  const boundingUrlSource = source?.bounding_image_url || source?.boundingUrl || null;
  const boundingImages = normalizeBoundingImages(
    source?.bounding_images || source?.boundingImages,
    boundingUrlSource,
    label
  );
  const resolvedBoundingUrl = boundingUrlSource || (boundingImages.length ? boundingImages[0].url : null);

  const variant = {
    model: modelKey,
    label,
    text_plain: source?.text_plain || source?.textPlain || '',
    text_markdown: source?.text_markdown || source?.textMarkdown || '',
    bounding_image_url: resolvedBoundingUrl,
    bounding_images: boundingImages,
    crops: Array.isArray(source?.crops) ? source.crops : [],
    preview_image_url: source?.preview_image_url || source?.previewUrl || null,
    preview: source?.preview || '',
    metadata: source?.metadata || {},
    elapsed_seconds: typeof source?.elapsed_seconds === 'number'
      ? source.elapsed_seconds
      : (typeof source?.metadata?.elapsed_seconds === 'number' ? source.metadata.elapsed_seconds : null),
  };
  return variant;
}

function mergeAggregateResponse(data, modelKey = null) {
  if (!data) {
    return;
  }
  if (!activeAggregate) {
    resetActiveAggregate(data.filename || null);
  }
  if (!activeAggregate.filename && data.filename) {
    activeAggregate.filename = data.filename;
  }
  if (!activeAggregate.createdAt && data.created_at) {
    activeAggregate.createdAt = data.created_at;
  }

  const variants = Array.isArray(data.variants) ? data.variants : [];
  variants.forEach((variant, index) => {
    const key = variant?.model || variant?.key || modelKey || `model-${index}`;
    if (!key) {
      return;
    }
    activeAggregate.variants.set(key, ensureVariantObject(variant, key));
  });

  if (!activeAggregate.variants.size) {
    const metadata = data?.metadata || {};
    const fallbackKey = modelKey
      || metadata.primary_model
      || (Array.isArray(metadata.models) ? metadata.models[0] : null)
      || data?.model
      || 'deepseek';
    activeAggregate.variants.set(fallbackKey, ensureVariantObject(data, fallbackKey));
  }

  const warnings = Array.isArray(data?.metadata?.warnings) ? data.metadata.warnings : [];
  warnings.forEach((warning) => {
    if (warning) {
      activeAggregate.warnings.add(warning);
    }
  });

  if (data.history_id) {
    if (!activeAggregate.historyIds.includes(data.history_id)) {
      activeAggregate.historyIds.push(data.history_id);
    }
  }

  if (Array.isArray(data.input_images)) {
    activeAggregate.inputImages = data.input_images;
  }
}

function renderAggregateResult(fallbackFilename) {
  if (!activeAggregate) {
    return;
  }

  const orderedKeys = sortModelKeys(activeAggregate.variants.keys());
  const variants = orderedKeys
    .map((key, index) => {
      const variant = activeAggregate.variants.get(key);
      if (!variant) {
        return null;
      }
      const resolvedKey = variant.model || variant.key || key || `model-${index}`;
      return { ...variant, model: resolvedKey };
    })
    .filter(Boolean);

  if (!variants.length) {
    return;
  }

  const aggregatedData = {
    filename: activeAggregate.filename || fallbackFilename || null,
    created_at: activeAggregate.createdAt || null,
    variants,
    metadata: {
      warnings: [...activeAggregate.warnings],
    },
    input_images: activeAggregate.inputImages || [],
  };

  const historyId = activeAggregate.historyIds[activeAggregate.historyIds.length - 1] || null;
  displayResult(aggregatedData, aggregatedData.filename || fallbackFilename, historyId, aggregatedData.created_at);
}

function updateTextCardsDisplay() {
  textCardRegistry.forEach((entry) => {
    if (!entry.markdownEl || !entry.plainEl) {
      return;
    }
    if (currentTextMode === 'plain') {
      entry.markdownEl.classList.add('hidden');
      entry.plainEl.classList.remove('hidden');
    } else {
      entry.markdownEl.classList.remove('hidden');
      entry.plainEl.classList.add('hidden');
    }
  });
}

function copyVariantText(variant) {
  const text = currentTextMode === 'plain'
    ? (variant.textPlain || variant.textMarkdown || '')
    : (variant.textMarkdown || variant.textPlain || '');

  if (!text) {
    setStatus('コピーするテキストがありません。', true);
    return;
  }

  navigator.clipboard.writeText(text)
    .then(() => setStatus('テキストをコピーしました。'))
    .catch(() => setStatus('テキストをコピーできませんでした。', true));
}

function downloadTextFile(content, filename, mime = 'text/plain') {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  downloadDataUrl(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function downloadVariantText(variant, filenameBase) {
  const ext = currentTextMode === 'plain' ? 'txt' : 'md';
  const text = currentTextMode === 'plain'
    ? (variant.textPlain || variant.textMarkdown || '')
    : (variant.textMarkdown || variant.textPlain || '');

  if (!text) {
    setStatus('ダウンロードするテキストがありません。', true);
    return;
  }

  const safeBase = filenameBase || 'ocr';
  const filename = `${safeBase}-${variant.key}.${ext}`;
  const mime = ext === 'md' ? 'text/markdown' : 'text/plain';
  downloadTextFile(text, filename, mime);
}
function handleFiles(files) {
  const normalized = Array.from(files || []).filter(Boolean);
  if (!normalized.length) {
    return;
  }
  if (currentItem) {
    removeQueueItemPreviews(currentItem);
  }
  if (queue.length) {
    queue.forEach(removeQueueItemPreviews);
  }
  clearLiveInputPreviews();
  renderPreviews();

  clearHistoryInputImages();

  const item = createQueueItem(normalized);
  if (!item) {
    return;
  }
  queue.push(item);
  item.previewEntries.forEach((preview) => {
    addInputPreview({
      id: preview.id,
      name: preview.name,
      url: preview.url,
      status: preview.status,
      type: preview.type,
    });
  });
  updateQueueStatus();
  processQueue();
}

async function processQueue() {
  if (processing || queue.length === 0) {
    return;
  }
  processing = true;
  currentItem = queue.shift() || null;
  if (currentItem) {
    updateQueueItemPreviewStatus(currentItem, '処理中');
  }
  const sessionId = startFakeProgress();
  updateQueueStatus();

  if (currentItem) {
    const success = await uploadFile(currentItem);
    updateQueueItemPreviewStatus(currentItem, success ? '完了' : '失敗');
    settleProgress(sessionId, success);
    if (success) {
      removeQueueItemPreviews(currentItem);
    }
  }

  processing = false;
  currentItem = null;
  updateQueueStatus();

  if (queue.length > 0) {
    processQueue();
  }
}

fileInput.addEventListener('change', (event) => {
  handleFiles(event.target.files);
  fileInput.value = '';
});

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(event.dataTransfer.files);
});

window.addEventListener('paste', (event) => {
  const items = event.clipboardData?.items;
  if (!items) {
    return;
  }
  const files = [];
  for (const item of items) {
    if (item.kind === 'file') {
      const file = item.getAsFile();
      if (file) {
        files.push(file);
      }
    }
  }
  if (files.length) {
    handleFiles(files);
  }
});

function setTab(mode) {
  currentTextMode = mode;
  if (mode === 'plain') {
    tabPlain.classList.add('active');
    tabMarkdown.classList.remove('active');
  } else {
    tabPlain.classList.remove('active');
    tabMarkdown.classList.add('active');
  }
  updateTextCardsDisplay();
}

tabPlain.addEventListener('click', () => setTab('plain'));
tabMarkdown.addEventListener('click', () => setTab('markdown'));

modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => {
  if (e.target === modal) closeModal();
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
    closeModal();
  }
});

modalCopyBtn.addEventListener('click', async () => {
  if (!modalContext) return;
  await copyImageToClipboard(modalContext.url);
  setStatus('画像をコピーしました。');
});

modalDownloadBtn.addEventListener('click', () => {
  if (!modalContext) return;
  let filename = 'ocr-output';

  if (modalContext.type === 'crop') {
    const ext = inferExtensionFromDataUrl(modalContext.url, 'png');
    const suffix = typeof modalContext.index === 'number' ? `crop-${modalContext.index + 1}` : 'crop';
    const base = lastResult?.filename || modalContext.name || 'ocr';
    filename = `${base}-${suffix}.${ext}`;
  } else if (modalContext.type === 'bounding') {
    const ext = inferExtensionFromDataUrl(modalContext.url, 'jpg');
    const base = lastResult?.filename || modalContext.name || 'ocr';
    filename = `${base}-bounding.${ext}`;
  } else if (modalContext.type === 'input') {
    filename = modalContext.name || 'input-image';
    if (!/\.[a-z0-9]+$/i.test(filename)) {
      filename = `${filename}.png`;
    }
  } else {
    const ext = inferExtensionFromDataUrl(modalContext.url, 'png');
    filename = `${lastResult?.filename || 'ocr'}-${modalContext.type || 'image'}.${ext}`;
  }

  downloadDataUrl(modalContext.url, filename);
});

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename || 'ocr-output';
  a.click();
}

async function copyImageToClipboard(dataUrl) {
  if (!navigator.clipboard || !window.ClipboardItem) {
    setStatus('このブラウザは画像コピーに対応していません。', true);
    return;
  }
  try {
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    const item = new ClipboardItem({ [blob.type]: blob });
    await navigator.clipboard.write([item]);
  } catch (error) {
    console.error(error);
    setStatus('画像をクリップボードにコピーできませんでした。', true);
  }
}

function openModal(ctx) {
  modalContext = ctx;
  modalImage.src = ctx.url;
  modal.classList.remove('hidden');
}

function closeModal() {
  modal.classList.add('hidden');
  modalImage.src = '';
  modalContext = null;
}

function renderQueue() {
  if (!queueListEl) return;
  queueListEl.innerHTML = '';

  if (processing && currentItem) {
    const processingItem = document.createElement('div');
    processingItem.className = 'queue-item processing';
    processingItem.innerHTML = `
      <span class="queue-name">${currentItem.name}</span>
      <span class="queue-state">処理中</span>
    `;
    queueListEl.appendChild(processingItem);
  }

  if (queue.length) {
    queue.forEach((item) => {
      const entry = document.createElement('div');
      entry.className = 'queue-item';
      entry.innerHTML = `
        <span class="queue-name">${item.name}</span>
        <span class="queue-state">待機中</span>
      `;
      queueListEl.appendChild(entry);
    });
  }

  if (!processing && queue.length === 0) {
    queueListEl.innerHTML = '';
  }
}

function loadIcons() {
  if (!iconSpriteContainer) return;
  fetch('/static/icons.svg')
    .then((res) => {
      if (!res.ok) throw new Error(res.statusText);
      return res.text();
    })
    .then((svg) => {
      iconSpriteContainer.innerHTML = svg;
    })
    .catch((err) => console.error('Failed to load icons', err));
}

async function uploadModelVariant(item, modelKey, promptValue) {
  const label = labelForModel(modelKey);
  const formData = new FormData();
  (item.files || []).forEach((file) => {
    formData.append('files', file);
  });
  if (promptValue) {
    formData.append('prompt', promptValue);
  }
  formData.append('models', modelKey);
  if (item.historyId) {
    formData.append('history_id', item.historyId);
  }

  const optionPayload = buildModelOptionPayload(modelKey);
  if (optionPayload) {
    formData.append('model_options', JSON.stringify(optionPayload));
  }

  setModelStatus(modelKey, 'running');
  setStatus(`${label} を解析中…`);

  try {
    const response = await fetch('/api/ocr', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || response.statusText);
    }

    const data = await response.json();
    if (!item.historyId && data.history_id) {
      item.historyId = data.history_id;
    } else if (data.history_id) {
      item.historyId = data.history_id;
    }
    mergeAggregateResponse(data, modelKey);
    renderAggregateResult(item.name);
    if (data.history_id) {
      activeHistoryId = data.history_id;
    }
    await fetchHistory(false);
    setModelStatus(modelKey, 'success');
    setStatus(`${label} の解析が完了しました。`);
    return true;
  } catch (error) {
    console.error(error);
    setModelStatus(modelKey, 'error');
    renderAggregateResult(item.name);
    setStatus(`${label} の解析に失敗しました: ${error.message}`, true);
    return false;
  }
}

async function uploadFile(item) {
  const models = getSelectedModels();
  if (!models.length) {
    setStatus('解析に使用するモデルが選択されていません。', true);
    return false;
  }

  setStatus(`${item.name} を解析中…`);
  resetActiveAggregate(item.name);
  resetModelStatuses(models);

  const promptValue = promptInput?.value?.trim() || '';
  let overallSuccess = true;

  for (const modelKey of models) {
    const success = await uploadModelVariant(item, modelKey, promptValue);
    if (!success) {
      overallSuccess = false;
    }
  }

  if (overallSuccess) {
    setStatus(`${item.name} の解析が完了しました。`);
  } else {
    setStatus(`${item.name} の解析で一部のモデルが失敗しました。`, true);
  }
  return overallSuccess;
}

async function fetchHistory(autoSelect = true) {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    if (!Array.isArray(data)) {
      historyEntries = [];
    } else {
      historyEntries = data;
    }

    const hasActive = activeHistoryId && historyEntries.some((entry) => entry.id === activeHistoryId);
    if (!hasActive && !historyEntries.length) {
      activeHistoryId = null;
      clearResults();
    }

    renderHistory();

    if (!historyEntries.length) {
      return;
    }

    if (autoSelect && (!hasActive || !activeHistoryId)) {
      await selectHistory(historyEntries[0].id);
    } else if (!hasActive && historyEntries.length) {
      activeHistoryId = historyEntries[0].id;
      renderHistory();
    } else {
      renderHistory();
    }
  } catch (error) {
    console.error('Failed to load history', error);
  }
}

function renderHistory() {
  historyListEl.innerHTML = '';
  if (!historyEntries.length) {
    const empty = document.createElement('p');
    empty.className = 'history-empty';
    empty.textContent = '履歴はまだありません。';
    historyListEl.appendChild(empty);
    return;
  }

  historyEntries.forEach((entry) => {
    const item = document.createElement('div');
    item.className = 'history-item';
    if (entry.id === activeHistoryId) {
      item.classList.add('active');
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'history-entry';

    if (entry.preview_image_url) {
      const thumb = document.createElement('img');
      thumb.className = 'history-thumb';
      thumb.src = entry.preview_image_url;
      thumb.alt = `${entry.filename || entry.id} プレビュー`;
      button.appendChild(thumb);
    }

    const textWrap = document.createElement('div');
    textWrap.className = 'history-text';
    const name = document.createElement('div');
    name.className = 'history-name';
    name.textContent = entry.filename || entry.id;
    const time = document.createElement('div');
    time.className = 'history-time';
    time.textContent = formatDate(entry.created_at);
    const preview = document.createElement('div');
    preview.className = 'history-preview';
    preview.textContent = entry.preview || '';
    textWrap.append(name, time, preview);
    button.appendChild(textWrap);
    button.addEventListener('click', () => selectHistory(entry.id));

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'icon-button mini history-delete';
    deleteBtn.title = '履歴を削除';
    deleteBtn.innerHTML = '<svg><use href="#icon-trash" /></svg>';
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteHistoryEntry(entry.id, entry.filename || entry.id);
    });

    item.append(button, deleteBtn);
    historyListEl.appendChild(item);
  });
}

async function selectHistory(id, forceReload = true) {
  if (!id) return;
  if (activeHistoryId !== id) {
    activeHistoryId = id;
    renderHistory();
  }
  if (forceReload) {
    await loadHistoryEntry(id);
  }
}

async function loadHistoryEntry(id) {
  try {
    const res = await fetch(`/api/history/${id}`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    displayResult(data, data.filename || id, data.history_id || id, data.created_at);
    setStatus(`${data.filename || id} を読み込みました。`);
  } catch (error) {
    console.error('failed to load history entry', error);
    setStatus('履歴の読み込みに失敗しました。', true);
  }
}

async function deleteHistoryEntry(id, filename) {
  if (!confirm(`${filename} を削除しますか？`)) {
    return;
  }
  try {
    const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(res.statusText);
    if (activeHistoryId === id) {
      activeHistoryId = null;
      clearResults();
    }
    await fetchHistory();
    setStatus(`${filename} を削除しました。`);
  } catch (error) {
    console.error('failed to delete history entry', error);
    setStatus('履歴の削除に失敗しました。', true);
  }
}

loadStoredModelSelection();
loadStoredModelOptions();
updateModelInfo();
fetchModels();
clearResults();
setTab('markdown');
pingServer();
updateQueueStatus();
fetchHistory();
loadIcons();

if (promptResetBtn && promptInput) {
  promptResetBtn.addEventListener('click', () => {
    promptInput.value = defaultPrompt;
    setStatus('プロンプトを既定に戻しました。');
  });
}
