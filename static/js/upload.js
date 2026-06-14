// static/js/upload.js — 文件上传模块

export function setupUpload({ fileInput, uploadBtn, uploadOverlay, dropzone, closeBtn, progressFill, uploadStatus, onUploadComplete }) {
    uploadBtn.addEventListener('click', () => {
        uploadOverlay.hidden = false;
    });

    closeBtn.addEventListener('click', () => {
        uploadOverlay.hidden = true;
    });

    uploadBtn.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = [...e.dataTransfer.files];
        for (const file of files) await uploadFile(file);
    });

    fileInput.addEventListener('change', async () => {
        const files = [...fileInput.files];
        for (const file of files) await uploadFile(file);
        fileInput.value = '';
    });

    async function uploadFile(file) {
        const progressDiv = document.getElementById('uploadProgress');
        const status = document.getElementById('uploadStatus');
        const fill = document.getElementById('progressFill');

        progressDiv.hidden = false;
        status.textContent = `正在上传: ${file.name}...`;
        fill.style.width = '50%';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            fill.style.width = '100%';

            if (data.status === 'indexed') {
                status.textContent = `✅ ${file.name} — ${data.chunks} 分片, ${data.tables} 表格, ${data.images} 图片`;
            } else {
                status.textContent = `❌ ${file.name} — ${data.error || '上传失败'}`;
            }

            if (onUploadComplete) await onUploadComplete();
        } catch (err) {
            status.textContent = `❌ ${file.name} — ${err.message}`;
            fill.style.width = '100%';
        }

        setTimeout(() => { progressDiv.hidden = true; }, 3000);
    }
}
