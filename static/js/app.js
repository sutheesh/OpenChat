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

// Context tracking elements
const contextBarFill = document.getElementById('contextBarFill');
const contextUsed = document.getElementById('contextUsed');
const contextMax = document.getElementById('contextMax');
const contextPercentage = document.getElementById('contextPercentage');

// Context tracking
let totalTokensUsed = 0;
const MAX_CONTEXT = 32768; // Qwen 2.5 context window

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    setupEventListeners();
    autoResizeTextarea();
    
    // Check status periodically
    setInterval(checkStatus, 30000);
});

function setupEventListeners() {
    // Send message - use onclick for dynamic switching between send/stop
    sendBtn.onclick = handleSend;
    
    // Enter to send
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating && messageInput.value.trim()) {
                handleSend();
            }
        }
    });
    
    // Auto-resize and enable/disable send button
    messageInput.addEventListener('input', () => {
        autoResizeTextarea();
        if (!isGenerating) {
            sendBtn.disabled = !messageInput.value.trim();
        }
    });
    
    // New chat
    newChatBtn.addEventListener('click', newChat);
    
    // Suggestion cards
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.dataset.prompt;
            messageInput.value = prompt;
            messageInput.focus();
            autoResizeTextarea();
            sendBtn.disabled = false;
        });
    });
    
    // Settings modal
    settingsBtn.addEventListener('click', () => {
        settingsModal.style.display = 'flex';
    });
    
    closeSettingsBtn.addEventListener('click', () => {
        settingsModal.style.display = 'none';
    });
    
    saveSettingsBtn.addEventListener('click', () => {
        updateConfig();
        settingsModal.style.display = 'none';
    });
    
    // Close modal on background click
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });
    
    // Settings sliders
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
    });
    
    lengthSlider.addEventListener('input', (e) => {
        lengthValue.textContent = e.target.value;
    });
    
    resetSettingsBtn.addEventListener('click', resetSettings);
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

// ============================================================================
// Context Window Tracking
// ============================================================================

function estimateTokens(text) {
    /**
     * Rough estimation: 1 token ≈ 4 characters for English
     * This is approximate but good enough for UI display
     */
    return Math.ceil(text.length / 4);
}

function updateContextUsage(additionalTokens) {
    totalTokensUsed += additionalTokens;
    
    // Update display
    contextUsed.textContent = totalTokensUsed.toLocaleString();
    
    const percentage = (totalTokensUsed / MAX_CONTEXT) * 100;
    contextPercentage.textContent = `${percentage.toFixed(1)}%`;
    
    // Update progress bar
    contextBarFill.style.width = `${Math.min(percentage, 100)}%`;
    
    // Change color based on usage
    contextBarFill.classList.remove('warning', 'danger');
    if (percentage > 80) {
        contextBarFill.classList.add('danger');
    } else if (percentage > 60) {
        contextBarFill.classList.add('warning');
    }
    
    // Warn if approaching limit
    if (percentage > 90 && percentage <= 95) {
        console.warn('Context window is 90% full. Consider starting a new chat.');
    } else if (percentage > 95) {
        console.error('Context window nearly full! Start a new chat to avoid truncation.');
    }
}

function resetContextUsage() {
    totalTokensUsed = 0;
    contextUsed.textContent = '0';
    contextPercentage.textContent = '0%';
    contextBarFill.style.width = '0%';
    contextBarFill.classList.remove('warning', 'danger');
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
        
        // Update context max if provided by backend
        if (data.context_window) {
            contextMax.textContent = data.context_window.toLocaleString();
        }
    } catch (error) {
        console.error('Status check failed:', error);
        sidebarStatusText.textContent = 'Error';
    }
}

async function updateConfig() {
    const config = {
        temperature: parseFloat(tempSlider.value),
        max_length: parseInt(lengthSlider.value)
    };
    
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
    } catch (error) {
        console.error('Failed to update config:', error);
    }
}

function resetSettings() {
    tempSlider.value = 0.7;
    lengthSlider.value = 200;
    tempValue.textContent = '0.7';
    lengthValue.textContent = '200';
    streamToggle.checked = true;
    updateConfig();
}

// ============================================================================
// Message Handling
// ============================================================================

async function handleSend() {
    const message = messageInput.value.trim();
    
    if (!message || isGenerating) return;
    
    // Hide welcome, show messages
    if (welcomeContainer.style.display !== 'none') {
        welcomeContainer.style.display = 'none';
        messagesContainer.style.display = 'block';
    }
    
    // Add user message
    addMessage('user', message);
    
    // Clear input
    messageInput.value = '';
    autoResizeTextarea();
    sendBtn.disabled = false;  // Keep enabled as stop button
    
    // Change button to stop
    isGenerating = true;
    sendIcon.style.display = 'none';
    stopIcon.style.display = 'flex';
    
    // Change click handler to stop
    sendBtn.onclick = handleStop;
    
    // Generate response
    try {
        if (streamToggle.checked) {
            await generateStreaming(message);
        } else {
            await generateRegular(message);
        }
    } catch (error) {
        console.error('Generation error:', error);
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
    sendBtn.disabled = true;  // Disabled until user types
    
    // Change click handler back to send
    sendBtn.onclick = handleSend;
    
    messageInput.focus();
}

async function generateRegular(message) {
    const typingMsg = addTypingIndicator();
    
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: message,
            stream: false
        })
    });
    
    if (!response.ok) {
        throw new Error('Generation failed');
    }
    
    const data = await response.json();
    typingMsg.remove();
    addMessage('assistant', data.response);
}

async function generateStreaming(message) {
    abortController = new AbortController();
    
    const typingMsg = addTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                stream: true
            }),
            signal: abortController.signal
        });
        
        if (!response.ok) {
            throw new Error('Generation failed');
        }
        
        // Remove typing indicator
        typingMsg.remove();
        
        // Create message element for streaming (plain text mode)
        const messageDiv = createMessageElement('assistant', '', true);
        const textElement = messageDiv.querySelector('.message-text');
        let fullText = '';
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.token) {
                            fullText += data.token;
                            // Show plain text while streaming
                            textElement.textContent = fullText;
                            messagesWrapper.scrollTop = messagesWrapper.scrollHeight;
                        }
                        
                        if (data.done) {
                            // Parse markdown and highlight after completion
                            textElement.innerHTML = parseMarkdown(fullText);
                            setTimeout(() => {
                                textElement.querySelectorAll('pre code').forEach((block) => {
                                    if (window.hljs) {
                                        hljs.highlightElement(block);
                                    }
                                });
                            }, 10);
                            return;
                        }
                    } catch (e) {
                        console.error('JSON parse error:', e);
                    }
                }
            }
        }
        
        // If stream ends without 'done', still format
        if (fullText) {
            textElement.innerHTML = parseMarkdown(fullText);
            setTimeout(() => {
                textElement.querySelectorAll('pre code').forEach((block) => {
                    if (window.hljs) {
                        hljs.highlightElement(block);
                    }
                });
            }, 10);
        }
        
    } catch (error) {
        typingMsg.remove();
        throw error;
    }
}

function handleStop() {
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    
    // Remove any typing indicators that might be stuck
    const typingIndicators = messagesContainer.querySelectorAll('.typing-indicator');
    typingIndicators.forEach(indicator => {
        indicator.closest('.message')?.remove();
    });
    
    resetSendButton();
}

// ============================================================================
// UI Helpers
// ============================================================================

function addMessage(role, text) {
    const messageDiv = createMessageElement(role, text);
    
    // Estimate and update token count
    const tokens = estimateTokens(text);
    updateContextUsage(tokens);
    
    return messageDiv;
}

function parseMarkdown(text) {
    // Escape HTML to prevent XSS
    const escapeHtml = (str) => {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    };
    
    let html = text;
    
    // Parse code blocks with language: ```language\ncode\n```
    html = html.replace(/```(\w+)?\s*\n?([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang ? lang.trim().toLowerCase() : 'plaintext';
        const cleanCode = code.trim();
        const escapedCode = escapeHtml(cleanCode);
        
        return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-language">${language}</span><button class="copy-code-btn" onclick="copyCodeToClipboard(this, event)">Copy</button></div><pre><code class="language-${language}">${escapedCode}</code></pre></div>`;
    });
    
    // Parse inline code: `code`
    html = html.replace(/`([^`\n]+)`/g, (match, code) => {
        return `<code>${escapeHtml(code)}</code>`;
    });
    
    // Convert newlines to <br> for regular text (not in code blocks)
    // html = html.replace(/\n/g, '<br>');
    
    return html;
}

// ===== COPY CODE FUNCTION - FIXED VERSION =====

function copyCodeToClipboard(button, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    try {
        const wrapper = button.closest('.code-block-wrapper');
        if (!wrapper) {
            console.error('Code wrapper not found');
            return;
        }
        
        const codeBlock = wrapper.querySelector('code');
        if (!codeBlock) {
            console.error('Code block not found');
            return;
        }
        
        const text = codeBlock.textContent;
        
        // Copy to clipboard
        navigator.clipboard.writeText(text).then(() => {
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.classList.add('copied');
            
            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('copied');
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy code:', err);
            button.textContent = 'Failed';
            setTimeout(() => {
                button.textContent = 'Copy';
            }, 2000);
        });
    } catch (error) {
        console.error('Copy error:', error);
    }
}

// ===== CREATE MESSAGE ELEMENT - UPDATED =====

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
    
    // For assistant messages, parse markdown unless streaming
    if (role === 'assistant' && !isStreaming && text) {
        textEl.innerHTML = parseMarkdown(text);
        
        // Apply syntax highlighting after a short delay
        setTimeout(() => {
            textEl.querySelectorAll('pre code').forEach((block) => {
                if (window.hljs) {
                    hljs.highlightElement(block);
                }
            });
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
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    
    content.appendChild(typingDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
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
    
    // Reset context usage
    resetContextUsage();
}