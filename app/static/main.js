const statusEl = document.getElementById('status');
const infoEl = document.getElementById('model-info');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const queueStatus = document.getElementById('queue-status');
const queueListEl = document.getElementById('queue-list');
const plainText = document.getElementById('plain-text');
const markdownPanel = document.getElementById('markdown-panel');
const markdownRender = document.getElementById('markdown-render');
const cropsGrid = document.getElementById('crops-grid');
const boundingImage = document.getElementById('bounding-image');
const metadataPanel = document.getElementById('metadata-panel');
const tabPlain = document.getElementById('tab-plain');
const tabMarkdown = document.getElementById('tab-markdown');
const copyTextBtn = document.getElementById('copy-text');
const downloadTextBtn = document.getElementById('download-text');
const cropActions = document.getElementById('crop-actions');
const copyAllCropsBtn = document.getElementById('copy-all-crops');
const downloadAllCropsBtn = document.getElementById('download-all-crops');
const boundingActions = document.getElementById('bounding-actions');
const copyBoundingBtn = document.getElementById('copy-bounding');
const downloadBoundingBtn = document.getElementById('download-bounding');
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
const MAX_PREVIEW_COLUMNS = 5;
const MAX_PREVIEW_ROWS = 2;

let currentTextMode = 'markdown';
let processing = false;
const queue = [];
let currentItem = null;
let lastResult = null;
let modalContext = null;
let historyEntries = [];
let activeHistoryId = null;
const inputPreviews = [];
let progressTimer = null;
let progressValue = 0;
let progressSession = 0;
let activeProgressSession = 0;

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
  return ensureDisplayMathLineBreaks(escapedText);
}

function ensureDisplayMathLineBreaks(math) {
  if (!math || !/\\begin\{(?:align|aligned|array|cases|matrix)/.test(math)) {
    return math;
  }

  return math.replace(/\n(\s*)(?!\\)/g, (match, whitespace, offset) => {
    const rest = math.slice(offset + match.length).trimStart();
    if (rest.startsWith('\\end')) {
      return `\n${whitespace}`;
    }
    return `\n${whitespace}\\ `;
  });
}

async function pingServer() {
  try {
    const res = await fetch('/api/ping');
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    infoEl.textContent = `サーバー状態: ${data.status}`;
  } catch (error) {
    infoEl.textContent = 'サーバーに接続できません';
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

function pruneInputPreviews() {
  const activeIds = new Set();
  if (currentItem) {
    activeIds.add(currentItem.id);
  }
  queue.forEach((item) => activeIds.add(item.id));

  while (inputPreviews.length > 12) {
    const removableIndex = inputPreviews.findIndex((entry) => !activeIds.has(entry.id));
    if (removableIndex === -1) {
      break;
    }
    const [removed] = inputPreviews.splice(removableIndex, 1);
    if (removed?.url) {
      URL.revokeObjectURL(removed.url);
    }
  }
}

function renderPreviews() {
  if (!inputPreviewSection || !inputPreviewGrid) return;
  pruneInputPreviews();

  const items = [...inputPreviews].sort((a, b) => (b.addedAt || 0) - (a.addedAt || 0));
  const visibleItems = items.slice(0, MAX_PREVIEW_COLUMNS * MAX_PREVIEW_ROWS);
  const hasItems = visibleItems.length > 0;
  inputPreviewSection.classList.toggle('hidden', !hasItems);

  if (!hasItems) {
    inputPreviewGrid.innerHTML = '';
    return;
  }

  const fragment = document.createDocumentFragment();

  visibleItems.forEach((entry, index) => {
    const item = document.createElement('div');
    item.className = 'preview-item';
    item.dataset.status = entry.status;

    const columnIndex = (index % MAX_PREVIEW_COLUMNS) + 1;
    const rowIndex = Math.floor(index / MAX_PREVIEW_COLUMNS) + 1;
    item.style.gridColumn = String(columnIndex);
    item.style.gridRow = String(rowIndex);

    if (entry.status) {
      const badge = document.createElement('span');
      badge.className = 'preview-badge';
      badge.textContent = entry.status;
      item.appendChild(badge);
    }

    const img = document.createElement('img');
    img.src = entry.url;
    img.alt = `${entry.name} プレビュー`;
    img.className = 'preview-thumb';
    img.addEventListener('click', () => {
      openModal({ type: 'input', url: entry.url, name: entry.name });
    });
    item.appendChild(img);

    const label = document.createElement('div');
    label.className = 'preview-label';
    label.textContent = entry.name;
    item.appendChild(label);

    fragment.appendChild(item);
  });

  inputPreviewGrid.innerHTML = '';
  inputPreviewGrid.appendChild(fragment);
}

function addInputPreview({ id, name, url, status }) {
  if (!id || !url) return;
  const existing = findPreview(id);
  if (existing) {
    existing.status = status;
    existing.name = name;
    renderPreviews();
    return;
  }

  inputPreviews.push({ id, name, url, status, addedAt: Date.now() });
  renderPreviews();
}

function updatePreviewStatus(id, status) {
  const entry = findPreview(id);
  if (!entry) return;
  entry.status = status;
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

  const pattern = /(^|\r?\n)([ \t]*)\[\s*([\s\S]*?)\s*\](?=\r?\n|$)/g;
  return markdown.replace(pattern, (match, prefix, indent, content) => {
    if (!content || !/\\/.test(content)) {
      return match;
    }

    const trimmed = content.trim();
    if (!trimmed) {
      return match;
    }

    const cleaned = sanitizeMathContent(trimmed);
    return `${prefix}${indent}\\[\n${cleaned}\n${indent}\\]`;
  });
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
    if (!text || (!text.includes('[') && !text.includes(']'))) {
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

    const inner = sanitizeMathContent(trimmed.slice(1, -1).trim());
    if (!inner || !inner.includes('\\')) {
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
    const text = element.textContent;
    if (!text) {
      return;
    }

    const trimmed = text.trim();
    if (trimmed.length < 4 || !trimmed.endsWith(']') || !trimmed.includes('\\')) {
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

    if (!body) {
      return;
    }

    const newline = text.includes('\r\n') ? '\r\n' : '\n';
    const cleaned = sanitizeMathContent(body);
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
      if (!(trimmedCheck.startsWith('[') && trimmedCheck.endsWith(']') && trimmedCheck.includes('\\'))) {
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
      if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        const inner = sanitizeMathContent(trimmed.slice(1, -1).trim());
        if (inner && inner.includes('\\')) {
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
        }
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

function transformMarkdown(markdown, crops) {
  if (!markdown) {
    return markdown;
  }

  let processed = normalizeMathBlocks(markdown);

  if (!Array.isArray(crops) || !crops.length) {
    return processed;
  }

  const replacements = new Map();
  crops.forEach((crop, index) => {
    if (!crop || !crop.url) return;
    if (crop.name) {
      replacements.set(crop.name, crop.url);
      replacements.set(`images/${crop.name}`, crop.url);
    }
    replacements.set(`${index}`, crop.url);
    replacements.set(`images/${index}`, crop.url);
    replacements.set(`images/${index}.jpg`, crop.url);
    replacements.set(`images/${index}.jpeg`, crop.url);
    replacements.set(`images/${index}.png`, crop.url);
  });

  return processed.replace(/!\[(.*?)\]\(([^)]+)\)/g, (match, alt, path) => {
    const key = path.trim();
    const direct = replacements.get(key);
    if (direct) {
      return `![${alt}](${direct})`;
    }
    const file = key.split('/').pop();
    if (file && replacements.has(file)) {
      return `![${alt}](${replacements.get(file)})`;
    }
    return match;
  });
}

function clearResults() {
  closeModal();
  if (plainText) {
    plainText.value = '';
  }
  markdownRender.innerHTML = '';
  cropsGrid.innerHTML = '<p>切り出し画像はまだありません。</p>';
  boundingImage.src = '';
  boundingImage.alt = '';
  metadataPanel.textContent = '';
  boundingActions.hidden = true;
  cropActions.hidden = true;
  lastResult = null;
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

function displayResult(data, filename, historyId = null, createdAt = null) {
  const firstDisplay = !lastResult;
  lastResult = { data, filename, historyId, createdAt };

  if (plainText) {
    plainText.value = data.text_plain || '';
  }
  const markdownText = data.text_markdown || '';
  const cropList = Array.isArray(data.crops) ? data.crops.filter((crop) => crop && crop.url) : [];
  const processedMarkdown = transformMarkdown(markdownText, cropList);
  markdownRender.innerHTML = processedMarkdown
    ? window.marked.parse(processedMarkdown)
    : '<p>マークダウン出力はありません。</p>';
  normalizeMathNodes(markdownRender);
  normalizeMathElements(markdownRender);
  renderMath(markdownRender);

  if (cropList.length > 0) {
    cropsGrid.innerHTML = '';
    cropList.forEach((crop, index) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'crop-item';
      const img = document.createElement('img');
      img.src = crop.url;
      img.alt = `Crop ${index + 1}`;
      img.dataset.index = index;
      img.addEventListener('click', () => openModal({ type: 'crop', url: crop.url, index }));
      wrapper.appendChild(img);
      addImageControls(wrapper, crop, filename, index);
      cropsGrid.appendChild(wrapper);
    });
    cropActions.hidden = false;
  } else {
    cropsGrid.innerHTML = '<p>切り出し画像はありませんでした。</p>';
    cropActions.hidden = true;
  }

  if (data.bounding_image_url) {
    boundingImage.src = data.bounding_image_url;
    boundingImage.alt = `${filename} のバウンディングボックス`;
    boundingActions.hidden = false;
  } else {
    boundingImage.src = '';
    boundingImage.alt = 'バウンディング画像は生成されませんでした';
    boundingActions.hidden = true;
  }

  const cropCount = cropList.length;
  const createdText = formatDate(createdAt || data.created_at);
  const inputName = data.metadata?.input || filename;
  metadataPanel.textContent = `ファイル: ${inputName} / クロップ数: ${cropCount} / 作成: ${createdText}`;

  if (firstDisplay) {
    setTab('markdown');
  }
}

function handleFiles(files) {
  if (!files || files.length === 0) {
    return;
  }
  Array.from(files).forEach((file) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const name = file.name || `upload-${Date.now()}`;
    const previewUrl = URL.createObjectURL(file);
    queue.push({
      id,
      file,
      name,
      previewUrl,
    });
    addInputPreview({ id, name, url: previewUrl, status: '待機中' });
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
  if (currentItem?.id) {
    updatePreviewStatus(currentItem.id, '処理中');
  }
  const sessionId = startFakeProgress();
  updateQueueStatus();

  if (currentItem) {
    const success = await uploadFile(currentItem);
    if (currentItem.id) {
      updatePreviewStatus(currentItem.id, success ? '完了' : '失敗');
    }
    settleProgress(sessionId, success);
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
  for (const item of items) {
    if (item.kind === 'file') {
      const file = item.getAsFile();
      if (file) {
        const id = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
        const name = file.name || `paste-${Date.now()}`;
        const previewUrl = URL.createObjectURL(file);
        queue.push({
          id,
          file,
          name,
          previewUrl,
        });
        addInputPreview({ id, name, url: previewUrl, status: '待機中' });
      }
    }
  }
  updateQueueStatus();
  processQueue();
});

function setTab(mode) {
  currentTextMode = mode;
  if (mode === 'plain') {
    tabPlain.classList.add('active');
    tabMarkdown.classList.remove('active');
    if (plainText) {
      plainText.classList.remove('hidden');
    }
    markdownPanel.classList.add('hidden');
  } else {
    tabPlain.classList.remove('active');
    tabMarkdown.classList.add('active');
    if (plainText) {
      plainText.classList.add('hidden');
    }
    markdownPanel.classList.remove('hidden');
  }
}

tabPlain.addEventListener('click', () => setTab('plain'));
tabMarkdown.addEventListener('click', () => setTab('markdown'));

copyTextBtn.addEventListener('click', () => {
  let text = '';
  if (currentTextMode === 'plain') {
    text = plainText?.value || '';
  } else if (lastResult?.data?.text_markdown) {
    text = lastResult.data.text_markdown;
  }
  if (text) {
    navigator.clipboard.writeText(text).then(() => setStatus('テキストをコピーしました。'));
  }
});

downloadTextBtn.addEventListener('click', () => {
  const isPlain = currentTextMode === 'plain';
  const text = isPlain ? (plainText?.value || '') : (lastResult?.data?.text_markdown || '');
  if (!text) {
    return;
  }
  const ext = isPlain ? 'txt' : 'md';
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ocr.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
});

copyAllCropsBtn.addEventListener('click', async () => {
  if (!lastResult?.data?.crops?.length) return;
  const first = lastResult.data.crops[0];
  if (!first?.url) return;
  await copyImageToClipboard(first.url);
  setStatus('最初の切り出し画像をコピーしました。');
});

downloadAllCropsBtn.addEventListener('click', () => {
  if (!lastResult?.data?.crops?.length) return;
  lastResult.data.crops.forEach((crop, index) => {
    if (!crop?.url) return;
    const ext = inferExtensionFromDataUrl(crop.url, 'png');
    const label = crop.name || `crop-${index + 1}`;
    downloadDataUrl(crop.url, `${lastResult.filename}-${label}.${ext}`);
  });
});

copyBoundingBtn.addEventListener('click', async () => {
  if (!lastResult?.data?.bounding_image_url) return;
  await copyImageToClipboard(lastResult.data.bounding_image_url);
  setStatus('バウンディング画像をコピーしました。');
});

downloadBoundingBtn.addEventListener('click', () => {
  if (!lastResult?.data?.bounding_image_url) return;
  const ext = inferExtensionFromDataUrl(lastResult.data.bounding_image_url, 'jpg');
  downloadDataUrl(lastResult.data.bounding_image_url, `${lastResult.filename}-bounding.${ext}`);
});

boundingImage.addEventListener('click', () => {
  if (!lastResult?.data?.bounding_image_url) return;
  openModal({ type: 'bounding', url: lastResult.data.bounding_image_url });
});

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

async function uploadFile(item) {
  setStatus(`${item.name} を解析中…`);
  const formData = new FormData();
  formData.append('file', item.file);
  const promptValue = promptInput?.value?.trim();
  if (promptValue) {
    formData.append('prompt', promptValue);
  }

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
    displayResult(data, item.name, data.history_id, data.created_at);
    if (data.history_id) {
      activeHistoryId = data.history_id;
    }
    await fetchHistory(false);
    setStatus(`${item.name} の解析が完了しました。`);
    return true;
  } catch (error) {
    console.error(error);
    setStatus(`${item.name} の解析に失敗しました: ${error.message}`, true);
    return false;
  }
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
