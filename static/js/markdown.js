// static/js/markdown.js — Markdown 渲染（KaTeX + Mermaid + highlight.js）

export function initMarkdown() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
    }
    if (typeof mermaid !== 'undefined') {
        mermaid.initialize({ startOnLoad: false, theme: 'default' });
    }
}

export function renderMarkdown(text) {
    if (!text) return '';
    if (typeof marked === 'undefined') return escapeHtml(text);
    try {
        return marked.parse(text);
    } catch {
        return escapeHtml(text);
    }
}

export function highlightBlocks(container) {
    if (typeof hljs !== 'undefined' && container) {
        container.querySelectorAll('pre code:not(.hljs)').forEach(block => {
            hljs.highlightElement(block);
        });
    }
    if (typeof mermaid !== 'undefined' && container) {
        container.querySelectorAll('pre code.language-mermaid').forEach(async (block) => {
            try {
                const id = 'mermaid-' + Math.random().toString(36).slice(2, 8);
                const { svg } = await mermaid.render(id, block.textContent);
                block.parentElement.outerHTML = svg;
            } catch {}
        });
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
