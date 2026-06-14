// static/js/app.js — 应用入口 & 知识库管理

import { initMarkdown } from './markdown.js';
import { setupUpload } from './upload.js';
import { setupChat } from './chat.js';

document.addEventListener('DOMContentLoaded', () => {
    initMarkdown();

    loadDocuments();

    setupUpload({
        fileInput: document.getElementById('fileInput'),
        uploadBtn: document.getElementById('uploadBtn'),
        uploadOverlay: document.getElementById('uploadOverlay'),
        dropzone: document.getElementById('uploadDropzone'),
        closeBtn: document.getElementById('closeUpload'),
        progressFill: document.getElementById('progressFill'),
        uploadStatus: document.getElementById('uploadStatus'),
        onUploadComplete: loadDocuments,
    });

    setupChat({
        messageInput: document.getElementById('messageInput'),
        sendBtn: document.getElementById('sendBtn'),
        chatMessages: document.getElementById('chatMessages'),
        welcomeMessage: document.getElementById('welcomeMessage'),
    });

    const docSearch = document.getElementById('docSearch');
    docSearch.addEventListener('input', () => loadDocuments(docSearch.value));
});

async function loadDocuments(keyword = '') {
    const docList = document.getElementById('docList');
    try {
        const params = new URLSearchParams({ page: '1', page_size: '50' });
        if (keyword) params.set('keyword', keyword);
        const resp = await fetch('/api/documents?' + params);
        const data = await resp.json();

        if (data.documents && data.documents.length > 0) {
            docList.innerHTML = data.documents.map(d =>
                '<div class="doc-item">' +
                '<div class="doc-name">' + escapeHtml(d.filename) + '</div>' +
                '<div class="doc-meta">' + d.file_type + ' · ' + d.chunks + ' 分片 · ' + formatDate(d.created_at) + '</div>' +
                '</div>'
            ).join('');
        } else {
            docList.innerHTML = '<p class="empty-hint">暂无文档，点击 + 上传</p>';
        }
    } catch (err) {
        console.error('加载文档列表失败:', err);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(isoStr) {
    try { return new Date(isoStr).toLocaleDateString('zh-CN'); } catch { return ''; }
}
