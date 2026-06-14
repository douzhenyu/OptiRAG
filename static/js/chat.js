// static/js/chat.js — 对话管理 & SSE 流处理

import { renderMarkdown, highlightBlocks } from './markdown.js';

export function setupChat({ messageInput, sendBtn, chatMessages, welcomeMessage }) {
    const sessionId = 'session_' + Math.random().toString(36).slice(2, 10) + '_' + Date.now();
    let isStreaming = false;

    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isStreaming) return;

        if (welcomeMessage) welcomeMessage.hidden = true;

        addMessage('user', text);
        messageInput.value = '';

        isStreaming = true;
        sendBtn.disabled = true;

        try {
            const isExperiment = /实验|方案|设计|光路|搭建|测量/.test(text);

            if (isExperiment) {
                await streamExperiment(text);
            } else {
                await quickChat(text);
            }
        } catch (err) {
            addMessage('assistant', `❌ 出错: ${err.message}`);
        } finally {
            isStreaming = false;
            sendBtn.disabled = false;
            messageInput.focus();
        }
    }

    async function quickChat(text) {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, question: text }),
        });
        const data = await resp.json();
        addMessage('assistant', data.answer || '（无回复）');
    }

    async function streamExperiment(text) {
        const msgEl = addMessage('assistant', '', true);
        const contentEl = msgEl.querySelector('.msg-content');
        let fullText = '';

        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, question: text }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                try {
                    const event = JSON.parse(line.slice(5).trim());

                    if (event.type === 'plan') {
                        fullText += renderPlanCard(event);
                    } else if (event.type === 'step_complete') {
                        fullText += `\n✅ **${event.step}**\n${event.result ? event.result.slice(0, 300) + '...' : ''}\n`;
                    } else if (event.type === 'replan' && event.action === 'replan') {
                        fullText += `\n🔄 调整计划: ${(event.new_steps || []).join(', ')}\n`;
                    } else if (event.type === 'report') {
                        fullText = event.report;
                    } else if (event.type === 'done') {
                        if (event.response) fullText = event.response;
                    } else if (event.type === 'error') {
                        fullText += `\n❌ ${event.message}\n`;
                    }

                    contentEl.innerHTML = renderMarkdown(fullText);
                    highlightBlocks(contentEl);
                    scrollToBottom();
                } catch {}
            }
        }

        msgEl.classList.remove('streaming');
        contentEl.innerHTML = renderMarkdown(fullText);
        highlightBlocks(contentEl);
    }

    function renderPlanCard(event) {
        let html = '<div class="plan-card"><div class="plan-header">📋 ' + event.message + '</div>';
        (event.plan || []).forEach((step, i) => {
            html += '<div class="plan-step pending">' + (i === 0 ? '🔄' : '⏳') + ' ' + (i + 1) + '. ' + step + '</div>';
        });
        html += '</div>\n';
        return html;
    }

    function addMessage(type, content, isStreamingMsg = false) {
        const div = document.createElement('div');
        div.className = 'message ' + type + (isStreamingMsg ? ' streaming' : '');
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        if (type === 'assistant' && !isStreamingMsg) {
            bubble.innerHTML = renderMarkdown(content);
            highlightBlocks(bubble);
        } else if (type === 'assistant' && isStreamingMsg) {
            const inner = document.createElement('div');
            inner.className = 'msg-content';
            inner.textContent = content;
            bubble.appendChild(inner);
        } else {
            bubble.textContent = content;
        }

        div.appendChild(bubble);
        chatMessages.appendChild(div);
        scrollToBottom();
        return div;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    return { sendMessage };
}
