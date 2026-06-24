/* script.js */
// Imports Hugging Face Transformers.js engine execution environment utilizing JSDeliver CDN distribution
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Set pipeline environment allocation parameters to cache assets locally 
env.allowLocalModels = false;

// Application Context States
let chatEngine = null;
let activeChatId = null;
let conversationHistory = [];
let voiceRecognition = null;
let totalTokensGenerated = 0;

// Application Selected Configuration Parameters
const MODEL_IDENTIFIER = 'Xenova/Qwen1.5-0.5B-Chat'; 

// ==========================================================================
// Initialization and Core Event Bindings
// ==========================================================================
document.addEventListener('DOMContentLoaded', async () => {
    setupDOMEventHooks();
    initSpeechRecognitionPipeline();
    loadSavedConversationsFromDisk();
    await bootstrapNeuralEngine();
});

function setupDOMEventHooks() {
    // Structural Node Accessors
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const exportChatBtn = document.getElementById('exportChatBtn');
    const openSidebarBtn = document.getElementById('openSidebarBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const voiceInputBtn = document.getElementById('voiceInputBtn');

    // Dynamic Textarea Sizing and Key Evaluation Execution
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            commitUserQueryStream();
        }
    });

    sendBtn.addEventListener('click', commitUserQueryStream);
    newChatBtn.addEventListener('click', () => createNewChatSessionContext());
    themeToggleBtn.addEventListener('click', toggleInterfaceThemePalette);
    clearHistoryBtn.addEventListener('click', purgeAllChatHistoriesOnDisk);
    exportChatBtn.addEventListener('click', exportActiveChatToTextBlob);
    
    openSidebarBtn.addEventListener('click', () => document.getElementById('sidebar').classList.add('open'));
    closeSidebarBtn.addEventListener('click', () => document.getElementById('sidebar').classList.remove('open'));
    voiceInputBtn.addEventListener('click', toggleVoiceDictationPipeline);
}

// ==========================================================================
// Neural Network Infrastructure (Transformers.js Local Loading)
// ==========================================================================
async function bootstrapNeuralEngine() {
    const overlay = document.getElementById('loaderOverlay');
    const barFill = document.getElementById('progressBarFill');
    const progressStatus = document.getElementById('progressStatus');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');

    try {
        // Initialization Pipeline Progress Interceptor Callbacks
        chatEngine = await pipeline('text-generation', MODEL_IDENTIFIER, {
            progress_callback: (data) => {
                if (data.status === 'downloading') {
                    const loaded = data.loaded || 0;
                    const total = data.total || 1;
                    const percentage = Math.round((loaded / total) * 100);
                    barFill.style.width = `${percentage}%`;
                    progressStatus.innerText = `Fetching weights: ${data.file} (${percentage}%)`;
                } else if (data.status === 'done') {
                    progressStatus.innerText = `Caching binary artifact layers...`;
                }
            }
        });

        // Toggle Application State upon Resolution
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 400);

        statusIndicator.className = 'status-indicator online';
        statusText.innerText = 'Online / Local Node';
    } catch (error) {
        console.error('System bootstrap fault encountered:', error);
        progressStatus.innerText = `Critical Loading Fault: ${error.message}`;
        statusIndicator.className = 'status-indicator error';
        statusText.innerText = 'Error Mapping Engine';
    }
}

// ==========================================================================
// User Input Orchestration and Token Generation Engines
// ==========================================================================
async function commitUserQueryStream() {
    const inputField = document.getElementById('chatInput');
    const query = inputField.value.trim();

    if (!query || !chatEngine) return;

    // Flush Input Controls Immediately
    inputField.value = '';
    inputField.style.height = 'auto';

    if (!activeChatId) {
        createNewChatSessionContext(query.substring(0, 24));
    }

    appendMessageToInterface('user', query);
    saveCurrentConversationState();

    // Construct UI elements for streaming feedback response loops
    const messageWindow = document.getElementById('chatWindow');
    const welcomeScreen = document.getElementById('welcomeScreen');
    if (welcomeScreen) welcomeScreen.remove();

    const rowElement = document.createElement('div');
    rowElement.className = 'message-row ai';
    
    const bubbleElement = document.createElement('div');
    bubbleElement.className = 'message-bubble';
    
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    
    bubbleElement.appendChild(indicator);
    rowElement.appendChild(bubbleElement);
    messageWindow.appendChild(rowElement);
    autoScrollToLatestNode();

    try {
        // Map conversational history context matching standard formats
        const formattedPrompt = transformHistoryContextToModelPrompt(query);
        
        // Execute Client Sandbox Execution Loop
        const inferenceOutput = await chatEngine(formattedPrompt, {
            max_new_tokens: 512,
            temperature: 0.7,
            do_sample: true,
            top_k: 50,
            top_p: 0.95,
        });

        const fullGeneration = inferenceOutput[0].generated_text;
        const parsedAIResponse = parseGeneratedOutputBlob(fullGeneration, formattedPrompt);

        // Erase processing indicators and parse finalized generation array structures
        indicator.remove();
        
        // Custom Markdown Processing using Marked.js Configured Architecture
        configureMarkdownEngineParser();
        bubbleElement.innerHTML = marked.parse(parsedAIResponse);
        
        // Assign dynamic timestamps and append elements to history arrays
        appendMetadataBlockToMessageNode(bubbleElement);
        
        conversationHistory.push({ role: 'assistant', content: parsedAIResponse, timestamp: new Date().toLocaleTimeString() });
        saveCurrentConversationState();
        
        // Increment Token Matrix Estimator Metrics
        updateEstimatedTokenMetrics(query.length / 4 + parsedAIResponse.length / 4);

    } catch (err) {
        console.error('Inference pipeline exception context:', err);
        indicator.remove();
        bubbleElement.innerHTML = `<span style="color: #ff4d4d;"><i class="fas fa-exclamation-triangle"></i> Local inference context dropped out: ${err.message}</span>`;
    }

    autoScrollToLatestNode();
}

function transformHistoryContextToModelPrompt(newestQuery) {
    // Maps standard structural schemas to format tokens specifically for Qwen execution templates
    let builtPrompt = "";
    conversationHistory.forEach(msg => {
        const entity = msg.role === 'user' ? 'user' : 'assistant';
        builtPrompt += `<|im_start|>${entity}\n${msg.content}<|im_end|>\n`;
    });
    builtPrompt += `<|im_start|>user\n${newestQuery}<|im_end|>\n<|im_start| style="display: none;">assistant\n`;
    return builtPrompt;
}

function parseGeneratedOutputBlob(generatedResponseText, inputPromptContext) {
    // Isolates newly computed model strings from context wrappers
    if (generatedResponseText.startsWith(inputPromptContext)) {
        let cleanSubstring = generatedResponseText.substring(inputPromptContext.length);
        return cleanSubstring.replace(/<\|im_end\|>/g, '').replace(/<\|im_start\|>/g, '').trim();
    }
    return generatedResponseText;
}

// ==========================================================================
// Markdown Engine Config and Syntax Highlighting Hooks
// ==========================================================================
function configureMarkdownEngineParser() {
    const customRenderer = new marked.Renderer();
    
    // Override standard code block structures to introduce copying button widgets
    customRenderer.code = function(codeText, codeLanguage) {
        const validatedLanguage = codeLanguage || 'plaintext';
        const highlightedCode = hljs.getLanguage(validatedLanguage) 
            ? hljs.highlight(codeText, { language: validatedLanguage }).value 
            : hljs.highlightAuto(codeText).value;

        return `
            <div class="code-block-wrapper">
                <div class="code-block-header">
                    <span>${validatedLanguage.toUpperCase()}</span>
                    <button class="copy-code-btn" onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').innerText); this.innerHTML='<i class=\'fas fa-check\'></i> Copied'; setTimeout(() => this.innerHTML='<i class=\'fas fa-copy\'></i> Copy Code', 2000)">
                        <i class="fas fa-copy"></i> Copy Code
                    </button>
                </div>
                <pre><code class="hljs ${validatedLanguage}">${highlightedCode}</code></pre>
            </div>
        `;
    };
    marked.setOptions({ renderer: customRenderer });
}

// ==========================================================================
// Context Interface Node Builders
// ==========================================================================
function appendMessageToInterface(role, messageText) {
    const chatWindow = document.getElementById('chatWindow');
    const welcomeScreen = document.getElementById('welcomeScreen');
    if (welcomeScreen) welcomeScreen.remove();

    const row = document.createElement('div');
    row.className = `message-row ${role === 'user' ? 'user' : 'ai'}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (role === 'user') {
        bubble.innerText = messageText;
    } else {
        configureMarkdownEngineParser();
        bubble.innerHTML = marked.parse(messageText);
    }

    appendMetadataBlockToMessageNode(bubble);
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    
    conversationHistory.push({ role, content: messageText, timestamp: new Date().toLocaleTimeString() });
    autoScrollToLatestNode();
}

function appendMetadataBlockToMessageNode(targetBubbleNode) {
    const metaDataBlock = document.createElement('div');
    metaDataBlock.className = 'message-metadata';
    metaDataBlock.innerHTML = `<span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>`;
    targetBubbleNode.appendChild(metaDataBlock);
}

function autoScrollToLatestNode() {
    const chatWindow = document.getElementById('chatWindow');
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function updateEstimatedTokenMetrics(tokenDeltaValue) {
    totalTokensGenerated += Math.ceil(tokenDeltaValue);
    document.getElementById('tokenCounter').innerText = totalTokensGenerated;
}

// ==========================================================================
// Web Speech API Voice Interface Subsystem
// ==========================================================================
function initSpeechRecognitionPipeline() {
    const SpeechObject = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechObject) {
        document.getElementById('voiceInputBtn').style.display = 'none';
        return;
    }
    
    voiceRecognition = new SpeechObject();
    voiceRecognition.continuous = false;
    voiceRecognition.interimResults = false;
    voiceRecognition.lang = 'en-US';

    voiceRecognition.onstart = () => {
        document.getElementById('voiceInputBtn').classList.add('active');
    };

    voiceRecognition.onresult = (event) => {
        const speechToTextValue = event.results[0][0].transcript;
        const inputField = document.getElementById('chatInput');
        inputField.value = (inputField.value + " " + speechToTextValue).trim();
        inputField.dispatchEvent(new Event('input')); // Trigger dynamic autogrow recalculations
    };

    voiceRecognition.onerror = (err) => {
        console.error('Speech synthesis module error mapping:', err);
        document.getElementById('voiceInputBtn').classList.remove('active');
    };

    voiceRecognition.onend = () => {
        document.getElementById('voiceInputBtn').classList.remove('active');
    };
}

function toggleVoiceDictationPipeline() {
    if (!voiceRecognition) return;
    
    const isCurrentlyRecording = document.getElementById('voiceInputBtn').classList.contains('active');
    if (isCurrentlyRecording) {
        voiceRecognition.stop();
    } else {
        voiceRecognition.start();
    }
}

// ==========================================================================
// Storage Hub Management & Multiple Session Architecture
// ==========================================================================
function createNewChatSessionContext(explicitTitleString = null) {
    // Flush current standard contextual frame references
    conversationHistory = [];
    activeChatId = 'chat_' + Date.now();
    
    const windowContainer = document.getElementById('chatWindow');
    windowContainer.innerHTML = `
        <div class="welcome-screen" id="welcomeScreen">
            <div class="welcome-icon"><i class="fas fa-brain"></i></div>
            <h2>Welcome to NexusAI</h2>
            <p>A fully decentralized, private local LLM framework operating entirely inside your browser sandbox.</p>
        </div>
    `;

    const dynamicTitle = explicitTitleString || 'Unassigned Prompt';
    
    // Save New Chat Session to disk immediately
    let structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    structuralDatabase[activeChatId] = {
        title: dynamicTitle,
        messages: [],
        timestamp: new Date().toLocaleDateString()
    };
    localStorage.setItem('nexus_chats', JSON.stringify(structuralDatabase));
    
    renderChatHistorySidebarPanel();
}

function saveCurrentConversationState() {
    if (!activeChatId) return;
    let structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    
    if (structuralDatabase[activeChatId]) {
        structuralDatabase[activeChatId].messages = conversationHistory;
        // Dynamically compute runtime names using first user sequence prompt structures if unassigned
        if (structuralDatabase[activeChatId].title === 'Unassigned Prompt' && conversationHistory.length > 0) {
            structuralDatabase[activeChatId].title = conversationHistory[0].content.substring(0, 26) + '...';
        }
        localStorage.setItem('nexus_chats', JSON.stringify(structuralDatabase));
        renderChatHistorySidebarPanel();
    }
}

function loadSavedConversationsFromDisk() {
    const structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    const sessionIds = Object.keys(structuralDatabase);
    
    if (sessionIds.length > 0) {
        // Load the most recently structured session entry by default sequence parsing
        activeChatId = sessionIds.sort().reverse()[0];
        hydrateChatWindowWithTargetHistory(activeChatId);
    } else {
        createNewChatSessionContext();
    }
}

function renderChatHistorySidebarPanel() {
    const targetHistoryListContainer = document.getElementById('historyList');
    targetHistoryListContainer.innerHTML = '';
    
    const structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    
    Object.keys(structuralDatabase).sort().reverse().forEach(keyId => {
        const chatMetadata = structuralDatabase[keyId];
        const listItem = document.createElement('button');
        listItem.className = `history-item ${keyId === activeChatId ? 'active' : ''}`;
        listItem.setAttribute('data-id', keyId);
        
        listItem.innerHTML = `
            <i class="far fa-comment-alt"></i>
            <span class="chat-title-text">${chatMetadata.title}</span>
            <span class="delete-chat-btn" data-id="${keyId}"><i class="fas fa-trash"></i></span>
        `;
        
        // Context selection interception routine mapping
        listItem.addEventListener('click', (e) => {
            if (e.target.closest('.delete-chat-btn')) {
                e.stopPropagation();
                purgeSpecificChatSessionContext(keyId);
            } else {
                hydrateChatWindowWithTargetHistory(keyId);
            }
        });
        
        targetHistoryListContainer.appendChild(listItem);
    });
}

function hydrateChatWindowWithTargetHistory(targetSessionKeyId) {
    activeChatId = targetSessionKeyId;
    const structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    const selectedSession = structuralDatabase[targetSessionKeyId];
    
    const windowContainer = document.getElementById('chatWindow');
    windowContainer.innerHTML = '';
    conversationHistory = [];

    if (!selectedSession || selectedSession.messages.length === 0) {
        windowContainer.innerHTML = `
            <div class="welcome-screen" id="welcomeScreen">
                <div class="welcome-icon"><i class="fas fa-brain"></i></div>
                <h2>Welcome to NexusAI</h2>
                <p>A fully decentralized, private local LLM framework operating entirely inside your browser sandbox.</p>
            </div>
        `;
    } else {
        selectedSession.messages.forEach(msg => {
            appendMessageToInterface(msg.role, msg.content);
        });
    }
    
    renderChatHistorySidebarPanel();
    document.getElementById('sidebar').classList.remove('open'); // Close responsive menus
}

function purgeSpecificChatSessionContext(targetSessionKeyId) {
    let structuralDatabase = JSON.parse(localStorage.getItem('nexus_chats') || '{}');
    delete structuralDatabase[targetSessionKeyId];
    localStorage.setItem('nexus_chats', JSON.stringify(structuralDatabase));
    
    if (activeChatId === targetSessionKeyId) {
        activeChatId = null;
        loadSavedConversationsFromDisk();
    } else {
        renderChatHistorySidebarPanel();
    }
}

function purgeAllChatHistoriesOnDisk() {
    if (confirm('Execute complete localized dataset wipe? All chats will be deleted.')) {
        localStorage.removeItem('nexus_chats');
        activeChatId = null;
        createNewChatSessionContext();
    }
}

// ==========================================================================
// Operational System Features (Theme Changing & Output Exporting)
// ==========================================================================
function toggleInterfaceThemePalette() {
    const documentRootNode = document.documentElement;
    const trackingAttribute = documentRootNode.getAttribute('data-theme');
    const targetNewThemeMode = trackingAttribute === 'dark' ? 'light' : 'dark';
    
    documentRootNode.setAttribute('data-theme', targetNewThemeMode);
    
    // Refresh Icon and Display Configuration Parameters
    const labelField = document.getElementById('themeToggleText');
    const iconField = document.getElementById('themeToggleBtn').querySelector('i');
    
    if (targetNewThemeMode === 'light') {
        labelField.innerText = 'Dark Mode';
        iconField.className = 'fas fa-sun';
    } else {
        labelField.innerText = 'Light Mode';
        iconField.className = 'fas fa-moon';
    }
}

function exportActiveChatToTextBlob() {
    if (conversationHistory.length === 0) {
        alert('Empty structural scope bounds detected. Cannot compile empty text log outputs.');
        return;
    }
    
    let consolidatedTextOutput = `==================================================\n`;
    consolidatedTextOutput += `NexusAI Client Chat Export Log Stream\n`;
    consolidatedTextOutput += `Session Key Address: ${activeChatId}\n`;
    consolidatedTextOutput += `Export Context Generation Timestamp: ${new Date().toLocaleString()}\n`;
    consolidatedTextOutput += `==================================================\n\n`;

    conversationHistory.forEach(msg => {
        const identifierHeader = msg.role === 'user' ? 'USER ENTITY' : 'NEXUSAI NODE';
        consolidatedTextOutput += `[${msg.timestamp || 'UNKNOWN'}] ${identifierHeader}:\n${msg.content}\n\n`;
    });

    const outputBlobInstance = new Blob([consolidatedTextOutput], { type: 'text/plain;charset=utf-8' });
    const virtualDownloadNode = document.createElement('a');
    virtualDownloadNode.href = URL.createObjectURL(outputBlobInstance);
    virtualDownloadNode.download = `NexusAI_Export_${activeChatId || Date.now()}.txt`;
    
    document.body.appendChild(virtualDownloadNode);
    virtualDownloadNode.click();
    document.body.removeChild(virtualDownloadNode);
}