// Global state
let isGenerating = false;
let abortController = null;

// DOM Elements
const messagesWrapper = document.getElementById('messagesWrapper');
const welcomeContainer = document.getElementById('welcomeContainer');
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sendIcon = document.getElementById('sendIcon');
const stopIcon = document.getElementById('stopIcon');
const newChatBtn = document.getElementById('newChatBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const resetSettingsBtn = document.getElementById('resetSettingsBtn');
const sidebarStatusDot = document.getElementById('sidebarStatusDot');
const sidebarStatusText = document.getElementById('sidebarStatusText');
const tempSlider = document.getElementById('tempSlider');
const lengthSlider = document.getElementById('lengthSlider');
const tempValue = document.getElementById('tempValue');
const lengthValue = document.getElementById('lengthValue');
const streamToggle = document.getElementById('streamToggle');

const contextBarFill = document.getElementById('contextBarFill');
const contextUsed = document.getElementById('contextUsed');
const contextMax = document.getElementById('contextMax');
const contextPercentage = document.getElementById('contextPercentage');

let totalTokensUsed = 0;
const MAX_CONTEXT = 32768;

document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setupEventListeners();
    autoResizeTextarea();
    setInterval(checkStatus, 30000);
});

function setupEventListeners() {
    sendBtn.onclick = handleSend;

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating && messageInput.value.trim()) handleSend();
        }
    });

    messageInput.addEventListener('input', () => {
        autoResizeTextarea();
        if (!isGenerating) sendBtn.disabled = !messageInput.value.trim();
    });

    newChatBtn.addEventListener('click', newChat);

    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            messageInput.value = card.dataset.prompt;
            messageInput.focus();
            autoResizeTextarea();
            sendBtn.disabled = false;
        });
    });

    settingsBtn.addEventListener('click', () => settingsModal.style.display = 'flex');
    closeSettingsBtn.addEventListener('click', () => settingsModal.style.display = 'none');
    saveSettingsBtn.addEventListener('click', () => { updateConfig(); settingsModal.style.display = 'none'; });
    settingsModal.addEventListener('click', (e) => { if (e.target === settingsModal) settingsModal.style.display = 'none'; });
    tempSlider.addEventListener('input', (e) => tempValue.textContent = e.target.value);
    lengthSlider.addEventListener('input', (e) => lengthValue.textContent = e.target.value);
    resetSettingsBtn.addEventListener('click', resetSettings);
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

// ============================================================================
// STREAMING STATE MACHINE
// States: idle → thinking → acknowledging → tool_calling → answering → done
// ============================================================================

// Detects tool name from acknowledgment text to show right icon
function detectToolFromText(text) {
    const lower = text.toLowerCase();
    if (lower.includes('weather') || lower.includes('temperature') || lower.includes('forecast'))
        return { name: 'get_weather', label: 'Fetching weather', icon: '🌤️' };
    if (lower.includes('knowledge') || lower.includes('search') || lower.includes('confluence') || lower.includes('looking up'))
        return { name: 'search_confluence', label: 'Searching knowledge base', icon: '📚' };
    return null;
}

// Creates the tool-in-progress badge shown after acknowledgment
function createToolProgressBadge(tool) {
    const badge = document.createElement('div');
    badge.className = 'tool-progress-badge';
    badge.id = 'toolProgressBadge';
    badge.innerHTML = `
        <div class="tool-progress-inner">
            <span class="tool-progress-icon">${tool.icon}</span>
            <span class="tool-progress-label">${tool.label}</span>
            <div class="tool-progress-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
        <div class="tool-progress-bar"><div class="tool-progress-bar-fill"></div></div>
    `;
    return badge;
}

// Creates the "generating answer" pulse shown while final answer streams
function createAnsweringBadge() {
    const badge = document.createElement('div');
    badge.className = 'answering-badge';
    badge.id = 'answeringBadge';
    badge.innerHTML = `
        <span class="answering-pulse"></span>
        <span class="answering-label">Generating answer</span>
    `;
    return badge;
}

// ============================================================================
// API Functions
// ============================================================================

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        if (data.model_loaded) {
            sidebarStatusDot.classList.add('ready');
            sidebarStatusText.textContent = 'Ready';
        } else {
            sidebarStatusText.textContent = 'Loading...';
        }
        if (data.context_window) contextMax.textContent = data.context_window.toLocaleString();

        // Show confluence KB chunk count if available
        if (data.confluence_chunks !== undefined) {
            const kbEl = document.getElementById('kbChunks');
            if (kbEl) kbEl.textContent = data.confluence_chunks.toLocaleString();
        }
    } catch (error) {
        sidebarStatusText.textContent = 'Error';
    }
}

async function updateConfig() {
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temperature: parseFloat(tempSlider.value),
                max_length: parseInt(lengthSlider.value)
            })
        });
    } catch (e) { console.error('Config update failed', e); }
}

function resetSettings() {
    tempSlider.value = 0.7; lengthSlider.value = 200;
    tempValue.textContent = '0.7'; lengthValue.textContent = '200';
    streamToggle.checked = true;
    updateConfig();
}

function estimateTokens(text) { return Math.ceil(text.length / 4); }

function updateContextUsage(additionalTokens) {
    totalTokensUsed += additionalTokens;
    contextUsed.textContent = totalTokensUsed.toLocaleString();
    const pct = (totalTokensUsed / MAX_CONTEXT) * 100;
    contextPercentage.textContent = `${pct.toFixed(1)}%`;
    contextBarFill.style.width = `${Math.min(pct, 100)}%`;
    contextBarFill.classList.remove('warning', 'danger');
    if (pct > 80) contextBarFill.classList.add('danger');
    else if (pct > 60) contextBarFill.classList.add('warning');
}

function resetContextUsage() {
    totalTokensUsed = 0;
    contextUsed.textContent = '0';
    contextPercentage.textContent = '0%';
    contextBarFill.style.width = '0%';
    contextBarFill.classList.remove('warning', 'danger');
}

// ============================================================================
// Message Handling
// ============================================================================

async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isGenerating) return;

    if (welcomeContainer.style.display !== 'none') {
        welcomeContainer.style.display = 'none';
        messagesContainer.style.display = 'block';
    }

    addMessage('user', message);
    messageInput.value = '';
    autoResizeTextarea();

    isGenerating = true;
    sendBtn.disabled = false;
    sendIcon.style.display = 'none';
    stopIcon.style.display = 'flex';
    sendBtn.onclick = handleStop;

    try {
        if (streamToggle.checked) {
            await generateStreaming(message);
        } else {
            await generateRegular(message);
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            addMessage('assistant', 'Sorry, an error occurred. Please try again.');
        }
    } finally {
        resetSendButton();
    }
}

function resetSendButton() {
    isGenerating = false;
    sendIcon.style.display = 'block';
    stopIcon.style.display = 'none';
    sendBtn.disabled = true;
    sendBtn.onclick = handleSend;
    messageInput.focus();
}

async function generateRegular(message) {
    const typingMsg = addTypingIndicator();
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, stream: false })
    });
    if (!response.ok) throw new Error('Generation failed');
    const data = await response.json();
    typingMsg.remove();
    addMessage('assistant', data.response);
}

async function generateStreaming(message) {
    abortController = new AbortController();

    // ── Create the assistant message bubble ──────────────────────────────────
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text" id="streamingText"></div>
            <div class="message-meta" id="streamingMeta"></div>
        </div>`;
    messagesContainer.appendChild(messageDiv);
    messagesWrapper.scrollTop = messagesWrapper.scrollHeight;

    const textEl = messageDiv.querySelector('#streamingText');
    const metaEl = messageDiv.querySelector('#streamingMeta');

    // Show initial thinking dots
    textEl.innerHTML = `<div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;

    let fullText = '';
    let phase = 'thinking';     // thinking → acknowledging → tool_calling → answering
    let detectedTool = null;
    let toolBadge = null;
    let answerBadge = null;
    let blankLineCount = 0;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, stream: true }),
            signal: abortController.signal
        });

        if (!response.ok) throw new Error('Generation failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                let data;
                try { data = JSON.parse(line.slice(6)); }
                catch (e) { continue; }

                if (data.error) {
                    textEl.textContent = `Error: ${data.error}`;
                    return;
                }

                if (data.token !== undefined) {
                    const token = data.token;
                    fullText += token;

                    // ── Phase: THINKING → ACKNOWLEDGING ─────────────────────
                    if (phase === 'thinking') {
                        phase = 'acknowledging';
                        textEl.innerHTML = ''; // clear dots
                        textEl.textContent = '';
                    }

                    // ── Phase: ACKNOWLEDGING — stream the ack text ───────────
                    if (phase === 'acknowledging') {

                        // Check for our sentinel marker
                        if (fullText.includes('[TOOL_EXECUTING]')) {
                            // Extract clean ack text before the sentinel
                            const ackClean = fullText.split('[TOOL_EXECUTING]')[0].trim();
                            textEl.textContent = ackClean;

                            phase = 'tool_calling';
                            detectedTool = detectToolFromText(ackClean) || { name: 'tool', label: 'Processing', icon: '⚙️' };
                            toolBadge = createToolProgressBadge(detectedTool);
                            metaEl.appendChild(toolBadge);
                            messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
                        } else {
                            // Still streaming ack — show it without the sentinel chars
                            textEl.textContent = fullText.replace('[TOOL_EXECUTING]', '');
                        }
                    }

                    // ── Phase: TOOL_CALLING → ANSWERING ─────────────────────
                    // When new non-whitespace tokens arrive after tool badge shown
                    else if (phase === 'tool_calling') {
                        if (token.trim().length > 0) {
                            phase = 'answering';

                            // Remove tool badge, add answering badge
                            if (toolBadge) { toolBadge.remove(); toolBadge = null; }
                            answerBadge = createAnsweringBadge();
                            metaEl.appendChild(answerBadge);

                            // Reset text to show only final answer
                            const ackText = textEl.textContent;
                            textEl.textContent = ackText;
                            // Start fresh text tracking for final answer
                            const answerStart = fullText.length - token.length;
                            textEl.dataset.answerStart = answerStart;
                        }
                    }

                    // ── Phase: ANSWERING — stream final answer ───────────────
                    else if (phase === 'answering') {
                        const start = parseInt(textEl.dataset.answerStart || 0);
                        const answerText = fullText.slice(start);

                        // Rebuild: ack text + separator + answer
                        const ackEnd = fullText.indexOf('\n\n');
                        if (ackEnd > -1) {
                            const ack = fullText.slice(0, ackEnd).trim();
                            textEl.textContent = ack + '\n\n' + answerText;
                        } else {
                            textEl.textContent = fullText;
                        }
                        messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
                    }
                }

                // ── DONE ─────────────────────────────────────────────────────
                if (data.done) {
                    // Remove any lingering badges
                    if (toolBadge) toolBadge.remove();
                    if (answerBadge) answerBadge.remove();

                    // Render final markdown
                    textEl.innerHTML = parseMarkdown(fullText.trim());
                    setTimeout(() => {
                        textEl.querySelectorAll('pre code').forEach(b => window.hljs && hljs.highlightElement(b));
                    }, 10);

                    updateContextUsage(estimateTokens(fullText));
                    messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
                    return;
                }
            }
        }

        // Stream ended without done signal
        if (toolBadge) toolBadge.remove();
        if (answerBadge) answerBadge.remove();
        if (fullText.trim()) {
            textEl.innerHTML = parseMarkdown(fullText.trim());
            setTimeout(() => {
                textEl.querySelectorAll('pre code').forEach(b => window.hljs && hljs.highlightElement(b));
            }, 10);
        }

    } catch (error) {
        if (toolBadge) toolBadge.remove();
        if (answerBadge) answerBadge.remove();
        throw error;
    }
}

function handleStop() {
    if (abortController) { abortController.abort(); abortController = null; }
    document.querySelectorAll('.tool-progress-badge, .answering-badge').forEach(el => el.remove());
    resetSendButton();
}

// ============================================================================
// UI Helpers
// ============================================================================

function addMessage(role, text) {
    const messageDiv = createMessageElement(role, text);
    updateContextUsage(estimateTokens(text));
    return messageDiv;
}

function parseMarkdown(text) {
    const escapeHtml = str => str
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

    let html = text;

    html = html.replace(/```(\w+)?\s*\n?([\s\S]*?)```/g, (match, lang, code) => {
        const language = (lang || 'plaintext').trim().toLowerCase();
        const escapedCode = escapeHtml(code.trim());
        return `<div class="code-block-wrapper">
            <div class="code-block-header">
                <span class="code-language">${language}</span>
                <button class="copy-code-btn" onclick="copyCodeToClipboard(this, event)">Copy</button>
            </div>
            <pre><code class="language-${language}">${escapedCode}</code></pre>
        </div>`;
    });

    html = html.replace(/`([^`\n]+)`/g, (m, code) => `<code>${escapeHtml(code)}</code>`);

    // Convert URLs to links
    html = html.replace(/(https?:\/\/[^\s<>"]+)/g, '<a href="$1" target="_blank" rel="noopener" class="msg-link">$1</a>');

    // Newlines to <br> outside code blocks
    html = html.replace(/\n/g, '<br>');

    return html;
}

function copyCodeToClipboard(button, event) {
    if (event) { event.preventDefault(); event.stopPropagation(); }
    const code = button.closest('.code-block-wrapper')?.querySelector('code');
    if (!code) return;
    navigator.clipboard.writeText(code.textContent).then(() => {
        button.textContent = 'Copied!';
        button.classList.add('copied');
        setTimeout(() => { button.textContent = 'Copy'; button.classList.remove('copied'); }, 2000);
    });
}

function createMessageElement(role, text, isStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const textEl = document.createElement('div');
    textEl.className = 'message-text';

    if (role === 'assistant' && !isStreaming && text) {
        textEl.innerHTML = parseMarkdown(text);
        setTimeout(() => {
            textEl.querySelectorAll('pre code').forEach(b => window.hljs && hljs.highlightElement(b));
        }, 10);
    } else {
        textEl.textContent = text;
    }

    content.appendChild(textEl);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesContainer.appendChild(messageDiv);
    messagesWrapper.scrollTop = messagesWrapper.scrollHeight;

    return messageDiv;
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>`;
    messagesContainer.appendChild(messageDiv);
    messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
    return messageDiv;
}

function newChat() {
    messagesContainer.innerHTML = '';
    messagesContainer.style.display = 'none';
    welcomeContainer.style.display = 'flex';
    messageInput.value = '';
    autoResizeTextarea();
    sendBtn.disabled = true;
    resetContextUsage();
}