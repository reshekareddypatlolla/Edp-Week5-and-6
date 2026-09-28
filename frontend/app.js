const imageInput = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const previewImage = document.querySelector('#preview-image');
const emptyPreview = document.querySelector('#empty-preview');
const generateButton = document.querySelector('#generate-button');
const captionText = document.querySelector('#caption-text');
const captionPlaceholder = document.querySelector('#caption-placeholder');
const errorMessage = document.querySelector('#error-message');
const loadingOverlay = document.querySelector('#loading-overlay');
const captionTools = document.querySelector('#caption-tools');
const apiUrl = (window.FRAME_PHRASE_API_URL || '').trim().replace(/\/$/, '');
let selectedFile = null;
let previewUrl = '';
let generatedCaption = '';

const serviceStatus = document.querySelector('#service-status');
const serviceLabel = document.querySelector('#service-label');
const serviceNote = document.querySelector('#service-note');
if (apiUrl) {
  serviceStatus.classList.add('is-ready');
  serviceLabel.textContent = 'Caption service connected';
  serviceNote.classList.add('is-hidden');
} else {
  serviceLabel.textContent = 'Caption service not connected';
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  captionPlaceholder.hidden = true;
  captionText.hidden = true;
  captionTools.hidden = true;
}

function clearResult() {
  generatedCaption = '';
  errorMessage.hidden = true;
  captionText.hidden = true;
  captionPlaceholder.hidden = false;
  captionTools.hidden = true;
  captionPlaceholder.textContent = 'A clear description is just one step away.';
}

function useImage(file) {
  clearResult();
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showError('Choose a JPG, PNG, or WEBP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('This image is larger than 10 MB. Choose a smaller file.');
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
  emptyPreview.hidden = true;
  document.querySelector('#file-detail').textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  document.querySelector('#drop-title').textContent = 'Choose a different photo';
  document.querySelector('#upload-hint').innerHTML = 'or <span class="browse-link">browse files</span> · up to 10 MB';
  document.querySelector('#remove-image').hidden = false;
  document.querySelector('#result-label').textContent = 'IMAGE READY';
  generateButton.disabled = false;
}

dropzone.addEventListener('click', () => imageInput.click());
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    imageInput.click();
  }
});
imageInput.addEventListener('change', () => useImage(imageInput.files?.[0]));
for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('is-dragging');
  });
}
dropzone.addEventListener('drop', (event) => useImage(event.dataTransfer.files?.[0]));
document.querySelector('#remove-image').addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
  previewImage.removeAttribute('src');
  previewImage.hidden = true;
  emptyPreview.hidden = false;
  document.querySelector('#file-detail').textContent = 'No image selected';
  document.querySelector('#drop-title').textContent = 'Drop a photo here';
  document.querySelector('#remove-image').hidden = true;
  document.querySelector('#result-label').textContent = 'WAITING FOR IMAGE';
  generateButton.disabled = true;
  clearResult();
});

generateButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  if (!apiUrl) {
    showError('Caption generation needs a deployed API. Add its URL to config.js and redeploy this frontend.');
    return;
  }
  const formData = new FormData();
  formData.append('image', selectedFile);
  generateButton.disabled = true;
  loadingOverlay.hidden = false;
  errorMessage.hidden = true;
  captionPlaceholder.hidden = true;
  captionText.hidden = true;
  captionTools.hidden = true;
  try {
    const response = await fetch(apiUrl, { method: 'POST', body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Caption service returned ${response.status}.`);
    if (typeof payload.caption !== 'string' || !payload.caption.trim()) {
      throw new Error('The caption service response must include a non-empty "caption" string.');
    }
    generatedCaption = payload.caption.trim();
    captionText.textContent = generatedCaption;
    captionText.hidden = false;
    captionTools.hidden = false;
    document.querySelector('#result-label').textContent = 'CAPTION READY';
  } catch (error) {
    showError(error instanceof TypeError
      ? 'Could not reach the caption service. Check its URL and allow this frontend origin in the API CORS settings.'
      : error.message || 'Caption generation failed. Please try again.');
  } finally {
    loadingOverlay.hidden = true;
    generateButton.disabled = false;
  }
});

document.querySelector('#copy-button').addEventListener('click', async (event) => {
  try {
    await navigator.clipboard.writeText(generatedCaption);
    event.currentTarget.textContent = 'Copied';
    window.setTimeout(() => { event.currentTarget.textContent = 'Copy'; }, 1400);
  } catch {
    showError('Clipboard access is unavailable in this browser. Select and copy the caption instead.');
  }
});
document.querySelector('#download-button').addEventListener('click', () => {
  const blob = new Blob([`${generatedCaption}\n`], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'image-caption.txt';
  link.click();
  URL.revokeObjectURL(url);
});
