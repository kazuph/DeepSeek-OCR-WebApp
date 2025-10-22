const statusEl = document.getElementById('status');
const infoEl = document.getElementById('model-info');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const plainText = document.getElementById('plain-text');
const markdownPanel = document.getElementById('markdown-panel');
const markdownRaw = document.getElementById('markdown-raw');
const markdownRender = document.getElementById('markdown-render');
const cropsGrid = document.getElementById('crops-grid');
const boundingImage = document.getElementById('bounding-image');
const metadataPanel = document.getElementById('metadata-panel');
const tabPlain = document.getElementById('tab-plain');
const tabMarkdown = document.getElementById('tab-markdown');

async function pingServer() {
  try {
    const res = await fetch('/api/ping');
    if (res.ok) {
      const data = await res.json();
      infoEl.textContent = `サーバー状態: ${data.status}`;
    } else {
      infoEl.textContent = 'サーバーに接続できません';
    }
  } catch (error) {
    infoEl.textContent = 'サーバーに接続できません';
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? '#ff6b6b' : '#cfd8dc';
}

function clearResults() {
  plainText.value = '';
  markdownRaw.value = '';
  markdownRender.innerHTML = '';
  cropsGrid.innerHTML = '<p>切り出し画像はまだありません。</p>';
  boundingImage.src = '';
  boundingImage.alt = '';
  metadataPanel.textContent = '';
}

async function uploadFile(file) {
  setStatus('解析を開始します…');
  const formData = new FormData();
  formData.append('file', file);

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
    displayResult(data, file.name);
    setStatus('解析が完了しました。');
  } catch (error) {
    console.error(error);
    setStatus(`エラーが発生しました: ${error.message}`, true);
  }
}

function displayResult(data, filename) {
  plainText.value = data.text_plain || '';
  const markdownText = data.text_markdown || '';
  markdownRaw.value = markdownText;
  markdownRender.innerHTML = markdownText ? window.marked.parse(markdownText) : '<p>マークダウン出力はありません。</p>';

  if (Array.isArray(data.crops) && data.crops.length > 0) {
    cropsGrid.innerHTML = '';
    data.crops.forEach((url, index) => {
      const wrapper = document.createElement('div');
      const img = document.createElement('img');
      img.src = url;
      img.alt = `Crop ${index + 1}`;
      wrapper.appendChild(img);
      cropsGrid.appendChild(wrapper);
    });
  } else {
    cropsGrid.innerHTML = '<p>切り出し画像はありませんでした。</p>';
  }

  if (data.bounding_image) {
    boundingImage.src = data.bounding_image;
    boundingImage.alt = `${filename} のバウンディングボックス`;
  } else {
    boundingImage.src = '';
    boundingImage.alt = 'バウンディング画像は生成されませんでした';
  }

  const meta = data.metadata || {};
  const cropCount = Array.isArray(data.crops) ? data.crops.length : 0;
  metadataPanel.textContent = `ファイル: ${meta.input || filename} / クロップ数: ${cropCount}`;
}

function handleFiles(files) {
  if (!files || files.length === 0) {
    return;
  }
  clearResults();
  uploadFile(files[0]);
}

fileInput.addEventListener('change', (event) => {
  handleFiles(event.target.files);
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
        clearResults();
        uploadFile(file);
        break;
      }
    }
  }
});

function setTab(mode) {
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

clearResults();
setTab('plain');
pingServer();
