const statusEl = document.getElementById('status');
const infoEl = document.getElementById('model-info');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const queueStatus = document.getElementById('queue-status');
const queueListEl = document.getElementById('queue-list');
const plainText = document.getElementById('plain-text');
const markdownPanel = document.getElementById('markdown-panel');
const markdownRaw = document.getElementById('markdown-raw');
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
const refreshHistoryBtn = document.getElementById('refresh-history');
const iconSpriteContainer = document.getElementById('icon-sprite');

let currentTextMode = 'markdown';
let processing = false;
const queue = [];
let currentItem = null;
let lastResult = null;
let modalContext = null;
let historyEntries = [];
let activeHistoryId = null;

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

function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function transformMarkdown(markdown, crops) {
  if (!markdown || !Array.isArray(crops) || !crops.length) {
    return markdown;
  }
  return markdown.replace(/!\[\]\(images\/(\d+)\.(?:jpe?g|png)\)/g, (match, index) => {
    const idx = Number.parseInt(index, 10);
    if (Number.isNaN(idx) || !crops[idx]) {
      return match;
    }
    return `![](${crops[idx]})`;
  });
}

function clearResults() {
  closeModal();
  plainText.value = '';
  markdownRaw.value = '';
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
  if (processing) {
    queueStatus.textContent = `処理中: ${currentItem?.name || '---'} / 残り ${queue.length} 件`;
  } else if (queue.length > 0) {
    queueStatus.textContent = `待機中: ${queue.length} 件`;
  } else {
    queueStatus.textContent = '';
  }
  renderQueue();
}

function inferExtensionFromDataUrl(dataUrl, fallback = 'png') {
  const match = dataUrl.match(/^data:(.+?);/);
  if (!match) return fallback;
  const mime = match[1];
  if (mime.includes('jpeg')) return 'jpg';
  if (mime.includes('png')) return 'png';
  if (mime.includes('gif')) return 'gif';
  if (mime.includes('webp')) return 'webp';
  return fallback;
}

function addImageControls(wrapper, url, filename, index) {
  const controls = document.createElement('div');
  controls.className = 'img-actions action-buttons';
  const copyBtn = document.createElement('button');
  copyBtn.className = 'button secondary mini';
  copyBtn.textContent = 'コピー';
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    copyImageToClipboard(url).then(() => setStatus('画像をコピーしました。'));
  });
  const downloadBtn = document.createElement('button');
  downloadBtn.className = 'button secondary mini';
  downloadBtn.textContent = 'DL';
  downloadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const ext = inferExtensionFromDataUrl(url);
    downloadDataUrl(url, `${filename}-crop-${index + 1}.${ext}`);
  });
  controls.append(copyBtn, downloadBtn);
  wrapper.appendChild(controls);
}

function displayResult(data, filename, historyId = null, createdAt = null) {
  const firstDisplay = !lastResult;
  lastResult = { data, filename, historyId, createdAt };

  plainText.value = data.text_plain || '';
  const markdownText = data.text_markdown || '';
  const processedMarkdown = transformMarkdown(markdownText, data.crops || []);
  markdownRaw.value = markdownText;
  markdownRender.innerHTML = processedMarkdown
    ? window.marked.parse(processedMarkdown)
    : '<p>マークダウン出力はありません。</p>';

  if (Array.isArray(data.crops) && data.crops.length > 0) {
    cropsGrid.innerHTML = '';
    data.crops.forEach((url, index) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'crop-item';
      const img = document.createElement('img');
      img.src = url;
      img.alt = `Crop ${index + 1}`;
      img.dataset.index = index;
      img.addEventListener('click', () => openModal({ type: 'crop', url, index }));
      wrapper.appendChild(img);
      addImageControls(wrapper, url, filename, index);
      cropsGrid.appendChild(wrapper);
    });
    cropActions.hidden = false;
  } else {
    cropsGrid.innerHTML = '<p>切り出し画像はありませんでした。</p>';
    cropActions.hidden = true;
  }

  if (data.bounding_image) {
    boundingImage.src = data.bounding_image;
    boundingImage.alt = `${filename} のバウンディングボックス`;
    boundingActions.hidden = false;
  } else {
    boundingImage.src = '';
    boundingImage.alt = 'バウンディング画像は生成されませんでした';
    boundingActions.hidden = true;
  }

  const cropCount = Array.isArray(data.crops) ? data.crops.length : 0;
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
    queue.push({
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      file,
      name: file.name || `upload-${Date.now()}`,
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
  updateQueueStatus();

  if (currentItem) {
    await uploadFile(currentItem);
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
        queue.push({
          id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
          file,
          name: file.name || `paste-${Date.now()}`,
        });
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
    plainText.classList.remove('hidden');
    markdownPanel.classList.add('hidden');
  } else {
    tabPlain.classList.remove('active');
    tabMarkdown.classList.add('active');
    plainText.classList.add('hidden');
    markdownPanel.classList.remove('hidden');
  }
}

tabPlain.addEventListener('click', () => setTab('plain'));
tabMarkdown.addEventListener('click', () => setTab('markdown'));

copyTextBtn.addEventListener('click', () => {
  const text = currentTextMode === 'plain' ? plainText.value : markdownRaw.value;
  if (text) {
    navigator.clipboard.writeText(text).then(() => setStatus('テキストをコピーしました。'));
  }
});

downloadTextBtn.addEventListener('click', () => {
  const isPlain = currentTextMode === 'plain';
  const text = isPlain ? plainText.value : markdownRaw.value;
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
  await copyImageToClipboard(lastResult.data.crops[0]);
  setStatus('最初の切り出し画像をコピーしました。');
});

downloadAllCropsBtn.addEventListener('click', () => {
  if (!lastResult?.data?.crops?.length) return;
  lastResult.data.crops.forEach((url, index) => {
    const ext = inferExtensionFromDataUrl(url);
    downloadDataUrl(url, `${lastResult.filename}-crop-${index + 1}.${ext}`);
  });
});

copyBoundingBtn.addEventListener('click', async () => {
  if (!lastResult?.data?.bounding_image) return;
  await copyImageToClipboard(lastResult.data.bounding_image);
  setStatus('バウンディング画像をコピーしました。');
});

downloadBoundingBtn.addEventListener('click', () => {
  if (!lastResult?.data?.bounding_image) return;
  const ext = inferExtensionFromDataUrl(lastResult.data.bounding_image, 'jpg');
  downloadDataUrl(lastResult.data.bounding_image, `${lastResult.filename}-bounding.${ext}`);
});

boundingImage.addEventListener('click', () => {
  if (!lastResult?.data?.bounding_image) return;
  openModal({ type: 'bounding', url: lastResult.data.bounding_image });
});

modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => {
  if (e.target === modal) closeModal();
});

modalCopyBtn.addEventListener('click', async () => {
  if (!modalContext) return;
  await copyImageToClipboard(modalContext.url);
  setStatus('画像をコピーしました。');
});

modalDownloadBtn.addEventListener('click', () => {
  if (!modalContext) return;
  const ext = inferExtensionFromDataUrl(modalContext.url);
  const suffix = modalContext.type === 'crop' ? `crop-${modalContext.index + 1}` : 'bounding';
  const name = `${lastResult?.filename || 'ocr'}-${suffix}.${ext}`;
  downloadDataUrl(modalContext.url, name);
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
  } catch (error) {
    console.error(error);
    setStatus(`${item.name} の解析に失敗しました: ${error.message}`, true);
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

    if (entry.preview_image) {
      const thumb = document.createElement('img');
      thumb.className = 'history-thumb';
      thumb.src = entry.preview_image;
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

refreshHistoryBtn.addEventListener('click', () => {
  fetchHistory();
});

clearResults();
setTab('markdown');
pingServer();
updateQueueStatus();
fetchHistory();
loadIcons();
