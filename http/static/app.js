const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  ws: null,
  sessionId: null,
  mode: localStorage.getItem('assistant-thinking-mode') || 'auto',
  autoplay: localStorage.getItem('assistant-autoplay') !== 'false',
  speed: Number(localStorage.getItem('assistant-speed') || 1),
  generating: false,
  currentTurn: null,
  recorder: null,
  recordingChunks: [],
  audioContext: null,
  audioCursor: 0,
  audioQueue: Promise.resolve(),
  audioSources: new Set(),
  playbackEpoch: 0,
  responseId: 0,
  blockedResponseId: -1,
  stopping: false,
  pendingAudio: 0,
  micStarting: false,
  recordingEpoch: 0,
};

function show(element, visible = true) { element.classList.toggle('hidden', !visible); }
function escapeText(value) { return value == null ? '' : String(value); }
function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  show(element, true);
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => show(element, false), 2600);
}

function setMode(mode) {
  state.mode = ['auto', 'fast', 'slow'].includes(mode) ? mode : 'auto';
  localStorage.setItem('assistant-thinking-mode', state.mode);
  $$('.mode-option').forEach(button => button.classList.toggle('active', button.dataset.mode === state.mode));
  $('#default-mode').value = state.mode;
  $('#mode-label').textContent = modeName(state.mode);
}

function modeName(mode) {
  return mode === 'auto' ? '自动协同' : (mode === 'slow' ? '慢思考（调试）' : '快思考（调试）');
}

function cooperationLabel(plan) {
  if (plan?.decision === 'clarify') return '快路径 · 补充条件';
  if (plan?.decision === 'handoff') return '待人工处理';
  return plan?.resolved_mode === 'slow' ? '快→慢 · 检索分析' : '快路径 · 简明回答';
}

function setGenerating(value, label = '正在思考') {
  state.generating = value;
  show($('#send'), !value);
  syncStopControls();
  show($('#stage'), value);
  $('#stage-label').textContent = label;
}

function collapseWelcome(collapse = true) {
  $('#welcome').classList.toggle('collapsed', collapse);
}

function syncStopControls() {
  show($('#stop'), state.generating || state.audioSources.size > 0 || state.pendingAudio > 0);
  $('#stop').setAttribute('aria-label', state.generating ? '停止回答' : '停止播放');
}

function stopReply() {
  state.stopping = true;
  state.blockedResponseId = Math.max(state.blockedResponseId, state.responseId);
  send({type: 'interrupt'});
  cancelRecording();
  stopAudio();
  if (state.currentTurn) {
    state.currentTurn._body.classList.remove('generating');
    renderAnswerMarkdown(state.currentTurn._body, state.currentTurn._rawText);
    state.currentTurn._timing.textContent = '已停止';
    state.currentTurn = null;
  }
  setGenerating(false);
}

function scrollBottom() {
  const messages = $('#messages');
  requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
}

function renderAnswerMarkdown(element, source) {
  const raw = escapeText(source);
  if (!window.marked || !window.DOMPurify || !window.renderMathInElement) {
    element.textContent = raw;
    return;
  }
  try {
    const html = marked.parse(raw, {gfm: true, breaks: true});
    element.innerHTML = DOMPurify.sanitize(html, {
      USE_PROFILES: {html: true},
      FORBID_TAGS: ['style', 'iframe', 'object', 'embed'],
      FORBID_ATTR: ['style'],
    });
    renderMathInElement(element, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false},
        {left: '$', right: '$', display: false},
      ],
      throwOnError: false,
      strict: false,
      trust: false,
    });
  } catch (error) {
    console.warn('Markdown/KaTeX render failed', error);
    element.textContent = raw;
  }
}

function createTurn(question, mode, options = {}) {
  collapseWelcome(true);
  const turn = document.createElement('article');
  turn.className = 'turn';
  const questionRow = document.createElement('div');
  questionRow.className = 'question';
  const questionText = document.createElement('p');
  questionText.textContent = question;
  questionRow.append(questionText);

  const answer = document.createElement('section');
  answer.className = 'answer';
  const head = document.createElement('div');
  head.className = 'answer-head';
  const modeLabel = document.createElement('span');
  modeLabel.textContent = options.metadata?.cooperation ? cooperationLabel(options.metadata.cooperation) : modeName(mode);
  const timing = document.createElement('span');
  timing.textContent = options.status || '正在生成';
  head.append(modeLabel, timing);
  const body = document.createElement('div');
  body.className = `answer-body${options.complete ? '' : ' generating'}`;
  const rawText = options.text || '';
  if (options.complete) renderAnswerMarkdown(body, rawText);
  else body.textContent = rawText;
  const actions = document.createElement('div');
  actions.className = 'answer-actions';
  actions.innerHTML = '<button data-action="speak">▷ 播放</button><button data-action="copy">复制</button><button data-action="retry">重新生成</button><button class="source-chip" data-action="sources">查看回答依据</button>';
  answer.append(head, body, actions);
  turn.append(questionRow, answer);
  turn._question = question;
  turn._body = body;
  turn._rawText = rawText;
  turn._timing = timing;
  turn._metadata = options.metadata || {};
  turn._mode = mode;
  turn._route = modeLabel;
  $('#messages').append(turn);
  scrollBottom();
  return turn;
}

function renderHistory(turns) {
  $('#messages').replaceChildren();
  if (!turns.length) {
    collapseWelcome(false);
    return;
  }
  collapseWelcome(true);
  turns.forEach(item => {
    const elapsed = item.metadata?.handoff?.status === 'pending'
      ? '待人工处理（尚未接入人工客服）'
      : (item.first_chunk_ms ? `首字 ${Math.round(item.first_chunk_ms)} ms` : (item.status || '已完成'));
    const mode = item.metadata?.web?.mode || 'auto';
    createTurn(item.user_text, mode, {
      text: item.assistant_text,
      complete: true,
      status: elapsed,
      metadata: item.metadata || {},
    });
  });
}

function renderSessions(items) {
  const list = $('#session-list');
  list.replaceChildren();
  items.forEach(item => {
    const row = document.createElement('div');
    row.className = `session-item${item.session_id === state.sessionId ? ' active' : ''}`;
    row.dataset.sessionId = item.session_id;
    const button = document.createElement('button');
    button.className = 'session-load';
    button.type = 'button';
    const title = document.createElement('strong');
    title.textContent = item.title;
    const meta = document.createElement('span');
    meta.textContent = `${item.turn_count} 轮对话`;
    button.append(title, meta);
    const remove = document.createElement('button');
    remove.className = 'session-delete';
    remove.type = 'button';
    remove.dataset.action = 'delete-session';
    remove.setAttribute('aria-label', `删除会话：${item.title}`);
    remove.textContent = '×';
    row.append(button, remove);
    list.append(row);
  });
}

function syncSessionSelection() {
  $$('#session-list .session-item').forEach(row => {
    row.classList.toggle('active', row.dataset.sessionId === state.sessionId);
  });
}

function renderEvidence(metadata = {}) {
  const panel = $('#evidence-content');
  panel.replaceChildren();
  const sources = metadata.knowledge_mode === 'rag'
    ? (metadata.evidence_sources || metadata.sources || []) : (metadata.sources || []);
  const summary = document.createElement('div');
  summary.className = 'evidence-summary';
  if (metadata.knowledge_mode === 'rag') {
    summary.textContent = `提供给模型的检索资料（非逐条结论核验） · 检索 ${Math.round(metadata.retrieval_ms || 0)} ms · 最高相似度 ${(metadata.top_score || 0).toFixed(3)}`;
  } else if (metadata.knowledge_mode === 'handoff') {
    summary.textContent = '当前保险资料证据不足，未使用模型知识补充。待人工处理，尚未接入人工客服；下列候选片段未作为回答依据。';
  } else if (metadata.knowledge_mode === 'model_fallback') {
    summary.textContent = '这是旧规则下保存的模型知识补充回答，并非保险资料证据；新提问已改为证据不足时待人工处理。';
  } else if (metadata.knowledge_mode === 'clarify') {
    summary.textContent = '缺少必要的用户条件，快路径先询问一个问题。补充后继续检索，当前没有生成业务结论。';
  } else if (metadata.knowledge_mode === 'service') {
    summary.textContent = '这是助手身份或会话服务说明，不是保险条款结论，无需资料引用。';
  } else {
    summary.textContent = '当前回答没有可确认的保险资料证据。';
  }
  panel.append(summary);
  if (metadata.cooperation) {
    const plan = metadata.cooperation;
    const details = document.createElement('p');
    details.className = 'evidence-summary';
    const names = {insured_person:'被保险人',age:'年龄',social_insurance:'社保',payment:'缴费方式',
      health:'健康情况',consultation_type:'咨询类型',event_reason:'出险原因'};
    const conditions = Object.entries(plan.slots || {}).map(([key, slot]) => {
      const value = slot.value == null ? '待确认' : String(slot.value);
      return `${names[key] || key}：${value}${slot.status === 'scenario' ? '（假设条件）' : ''}`;
    }).join('；');
    details.textContent = `协同路径：${cooperationLabel(plan)} · 条件版本 ${plan.context_version}。用户表述（非条款证据）：${conditions || '暂无已提取条件'}。`;
    panel.append(details);
  }
  sources.forEach((source, index) => {
    const card = document.createElement('article');
    card.className = 'source-card';
    const header = document.createElement('header');
    const page = document.createElement('span');
    const sourceName = source.source_name || 'PDF';
    const location = source.location || `第 ${source.pdf_page} 页`;
    page.textContent = `${sourceName} · ${location}`;
    const score = document.createElement('span');
    score.textContent = `相似度 ${Number(source.score || 0).toFixed(3)}`;
    header.append(page, score);
    const text = document.createElement('p');
    text.textContent = escapeText(source.text).slice(0, 360);
    card.append(header, text);
    panel.append(card);
  });
  if (!sources.length) {
    const empty = document.createElement('p');
    empty.className = 'evidence-empty';
    empty.textContent = '没有可展示的保险资料片段。';
    panel.append(empty);
  }
  openPanel('evidence');
}

function openPanel(name) {
  const panel = name === 'history' ? $('#history-panel') : $('#evidence-panel');
  if (name === 'evidence') $('#app').classList.remove('evidence-hidden');
  panel.classList.add('open');
  if (innerWidth <= 1050) show($('#backdrop'), true);
}
function closePanels() {
  $$('.sidebar').forEach(panel => panel.classList.remove('open'));
  show($('#backdrop'), false);
}

function closeEvidence() {
  $('#evidence-panel').classList.remove('open');
  $('#app').classList.add('evidence-hidden');
  show($('#backdrop'), false);
}

function send(payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    toast('服务尚未连接');
    return false;
  }
  if (['chat', 'new_session', 'load_session', 'clear_session', 'delete_session'].includes(payload.type)) {
    cancelRecording();
  }
  state.ws.send(JSON.stringify(payload));
  return true;
}

async function ensureAudio() {
  if (!state.audioContext || state.audioContext.state === 'closed') {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    state.audioCursor = 0;
  }
  if (state.audioContext.state === 'suspended') await state.audioContext.resume();
}

function stopAudio() {
  state.playbackEpoch++;
  state.audioSources.forEach(source => { try { source.stop(); } catch (_) {} });
  state.audioSources.clear();
  state.audioCursor = 0;
  state.audioQueue = Promise.resolve();
  state.pendingAudio = 0;
  syncStopControls();
}

async function playPcm(base64Pcm, sampleRate, epoch = state.playbackEpoch) {
  await ensureAudio();
  if (epoch !== state.playbackEpoch) return;
  const raw = atob(base64Pcm);
  const count = Math.floor(raw.length / 2);
  const buffer = state.audioContext.createBuffer(1, count, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let index = 0; index < count; index++) {
    let value = raw.charCodeAt(index * 2) | (raw.charCodeAt(index * 2 + 1) << 8);
    if (value >= 32768) value -= 65536;
    channel[index] = value / 32768;
  }
  const source = state.audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioContext.destination);
  const startAt = Math.max(state.audioCursor, state.audioContext.currentTime + .06);
  source.start(startAt);
  state.audioCursor = startAt + buffer.duration;
  state.audioSources.add(source);
  syncStopControls();
  source.onended = () => { state.audioSources.delete(source); syncStopControls(); };
}

function enqueuePcm(base64Pcm, sampleRate) {
  const epoch = state.playbackEpoch;
  state.pendingAudio++;
  syncStopControls();
  state.audioQueue = state.audioQueue.then(() => {
    if (epoch === state.playbackEpoch) return playPcm(base64Pcm, sampleRate, epoch);
  }).catch(error => {
    if (epoch === state.playbackEpoch) toast(error.message);
  }).finally(() => {
    if (epoch === state.playbackEpoch) {
      state.pendingAudio = Math.max(0, state.pendingAudio - 1);
      syncStopControls();
    }
  });
}

function handleMessage(message) {
  if (state.stopping && !['response_epoch','interrupted','ready','history','sessions','service','error'].includes(message.type)) return;
  if (message.response_id != null && message.response_id <= state.blockedResponseId) return;
  if (message.response_id != null) {
    if (message.response_id < state.responseId) return;
    if (message.response_id > state.responseId) {
      state.responseId = message.response_id;
      stopAudio();
      if (state.currentTurn) {
        state.currentTurn._body.classList.remove('generating');
        renderAnswerMarkdown(state.currentTurn._body, state.currentTurn._rawText);
        state.currentTurn._timing.textContent = '已打断';
        state.currentTurn = null;
      }
      setGenerating(false);
    }
  }
  switch (message.type) {
    case 'response_epoch':
      break;
    case 'service':
      $('#connection-label').textContent = message.state === 'ready' ? '已就绪' : '正在加载模型';
      break;
    case 'ready':
      state.responseId = 0;
      state.blockedResponseId = -1;
      state.stopping = false;
      state.sessionId = message.session_id;
      syncSessionSelection();
      $('#connection-label').textContent = '已就绪';
      break;
    case 'sessions':
      renderSessions(message.items || []);
      break;
    case 'history':
      cancelRecording();
      state.sessionId = message.session_id;
      syncSessionSelection();
      state.currentTurn = null;
      setGenerating(false);
      stopAudio();
      renderHistory(message.turns || []);
      closePanels();
      break;
    case 'stage':
      setGenerating(true, message.label || '正在处理');
      break;
    case 'cooperation':
      if (state.currentTurn) {
        state.currentTurn._route.textContent = cooperationLabel(message.plan);
        state.currentTurn._metadata.cooperation = message.plan;
      }
      setGenerating(true, cooperationLabel(message.plan));
      break;
    case 'turn_start':
      setGenerating(true, message.from_voice ? '识别完成，正在回答' : '正在检索并思考');
      stopAudio();
      state.currentTurn = createTurn(message.question, message.mode);
      break;
    case 'text_delta':
      if (state.currentTurn) {
        state.currentTurn._rawText += message.text;
        state.currentTurn._body.textContent = state.currentTurn._rawText;
        scrollBottom();
      }
      break;
    case 'audio_reset':
      stopAudio();
      break;
    case 'audio_chunk':
      enqueuePcm(message.pcm, message.sample_rate);
      break;
    case 'audio_end':
      setGenerating(false);
      break;
    case 'text_done':
      if (state.currentTurn) {
        state.currentTurn._rawText = message.text;
        state.currentTurn._metadata = message.metadata || {};
        state.currentTurn._body.classList.remove('generating');
        renderAnswerMarkdown(state.currentTurn._body, message.text);
        const elapsed = message.metadata?.voice?.text_completed_ms;
        state.currentTurn._timing.textContent = elapsed ? `文字完成 ${Math.round(elapsed)} ms` : '文字已完成';
      }
      setGenerating(true, '文字已完成，正在合成语音');
      break;
    case 'turn_done':
      if (state.currentTurn) {
        state.currentTurn._rawText = message.text;
        state.currentTurn._body.classList.remove('generating');
        renderAnswerMarkdown(state.currentTurn._body, state.currentTurn._rawText);
        state.currentTurn._metadata = message.metadata || {};
        if (message.metadata?.cooperation) state.currentTurn._route.textContent = cooperationLabel(message.metadata.cooperation);
        const first = message.first_chunk_ms || message.metadata?.voice?.llm_first_text_ms;
        state.currentTurn._timing.textContent = message.metadata?.handoff?.status === 'pending'
          ? '待人工处理（尚未接入人工客服）'
          : (first ? `首字 ${Math.round(first)} ms · ${message.status}` : message.status);
      }
      state.currentTurn = null;
      setGenerating(false);
      break;
    case 'interrupted':
      state.stopping = false;
      if (state.currentTurn) {
        state.currentTurn._body.classList.remove('generating');
        renderAnswerMarkdown(state.currentTurn._body, state.currentTurn._rawText);
        state.currentTurn._timing.textContent = '已打断';
      }
      state.currentTurn = null;
      setGenerating(false);
      stopAudio();
      break;
    case 'error':
      toast(message.message || '请求失败');
      if (state.currentTurn) state.currentTurn._body.classList.remove('generating');
      setGenerating(false);
      break;
  }
}

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  state.ws = ws;
  $('#connection-label').textContent = '正在连接';
  ws.onmessage = event => handleMessage(JSON.parse(event.data));
  ws.onclose = event => {
    cancelRecording();
    stopAudio();
    $('#connection-label').textContent = event.code === 4401 ? '认证失效' : '连接已断开';
    setGenerating(false);
    if (event.code === 4401) {
      show($('#app'), false);
      show($('#login'), true);
    } else {
      setTimeout(connect, 2200);
    }
  };
}

async function blobToBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const block = 0x8000;
  for (let index = 0; index < bytes.length; index += block) {
    binary += String.fromCharCode(...bytes.subarray(index, index + block));
  }
  return btoa(binary);
}

function updateMicrophoneUi() {
  const recording = state.recorder?.state === 'recording';
  const label = recording ? '结束录音并发送' : '开始录音';
  $$('.mic, .hero-mic').forEach(button => {
    button.classList.toggle('recording', recording);
    button.setAttribute('aria-label', label);
    button.setAttribute('title', label);
  });
  $('#hero-mic b').textContent = recording ? '点击结束并发送' : '点击开始录音';
  $('#voice-status').textContent = recording ? '正在录音 · 再次点击结束并发送' : '点击录音，再次点击结束并发送';
}

function cancelRecording() {
  state.recordingEpoch++;
  if (state.recorder) {
    state.recorder.cancelled = true;
    if (state.recorder.state === 'recording') state.recorder.stop();
  }
  updateMicrophoneUi();
}

function toggleMicrophone() {
  return toggleRecording();
}

async function toggleRecording() {
  if (state.recorder?.state === 'recording') {
    state.recorder.stop();
    return;
  }
  if (state.micStarting) return;
  if (state.ws?.readyState !== WebSocket.OPEN) { toast('请等待服务连接'); return; }
  state.micStarting = true;
  const epoch = state.recordingEpoch;
  const ws = state.ws;
  const session = state.sessionId;
  const mode = state.mode;
  const speed = state.speed;
  let stream;
  try {
    await ensureAudio();
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    if (epoch !== state.recordingEpoch || ws !== state.ws || ws.readyState !== WebSocket.OPEN) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type)) || '';
    const chunks = [];
    const recorder = new MediaRecorder(stream, mime ? {mimeType: mime} : undefined);
    state.recordingChunks = chunks;
    state.recorder = recorder;
    let timeout;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = async () => {
      clearTimeout(timeout);
      stream.getTracks().forEach(track => track.stop());
      updateMicrophoneUi();
      try {
        if (recorder.cancelled) return;
        const blob = new Blob(chunks, {type: recorder.mimeType});
        const audio = await blobToBase64(blob);
        if (recorder.cancelled || epoch !== state.recordingEpoch || ws !== state.ws || session !== state.sessionId) return;
        if (send({type: 'voice', audio, mime: blob.type, mode, speed})) setGenerating(true, '正在上传录音');
      } catch (error) { toast(error.message); }
    };
    recorder.start(250);
    updateMicrophoneUi();
    toast('正在录音，再点一次结束并发送');
    timeout = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 30000);
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    updateMicrophoneUi();
    toast(`无法使用麦克风：${error.message}`);
  } finally { state.micStarting = false; }
}

async function initialize() {
  updateMicrophoneUi();
  setMode(state.mode);
  $('#autoplay').checked = state.autoplay;
  $('#speech-speed').value = state.speed;
  $('#speed-label').textContent = `${state.speed.toFixed(1)}×`;
  try {
    const auth = await fetch('/api/auth').then(response => response.json());
    if (auth.required && !auth.authenticated) {
      show($('#login'), true);
    } else {
      show($('#app'), true);
      connect();
    }
  } catch (_) {
    show($('#app'), true);
    connect();
  }
}

$('#login-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('#login-error').textContent = '';
  const response = await fetch('/api/login', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({password: $('#password').value}),
  });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    $('#login-error').textContent = result.detail || '登录失败';
    return;
  }
  show($('#login'), false);
  show($('#app'), true);
  connect();
});

$('#composer').addEventListener('submit', async event => {
  event.preventDefault();
  const text = $('#prompt').value.trim();
  if (!text) return;
  await ensureAudio().catch(() => {});
  if (send({type: 'chat', text, mode: state.mode, autoplay: state.autoplay, speed: state.speed})) {
    $('#prompt').value = '';
    $('#prompt').style.height = 'auto';
  }
});

$('#prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#composer').requestSubmit();
  }
});
$('#prompt').addEventListener('input', event => {
  event.target.style.height = 'auto';
  event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`;
});

$$('.mode-option').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
$('#mic').addEventListener('click', toggleMicrophone);
$('#hero-mic').addEventListener('click', toggleMicrophone);
$('#stop').addEventListener('click', stopReply);
$('#new-chat').addEventListener('click', () => send({type: 'new_session'}));
$('#history-open').addEventListener('click', () => openPanel('history'));
$('#settings-open').addEventListener('click', () => show($('#settings'), true));
$('#settings-close').addEventListener('click', () => show($('#settings'), false));
$('#backdrop').addEventListener('click', closePanels);
$('#history-panel .close-panel').addEventListener('click', closePanels);
$('#evidence-panel .close-panel').addEventListener('click', closeEvidence);

$('#session-list').addEventListener('click', event => {
  const row = event.target.closest('.session-item');
  if (!row) return;
  if (event.target.closest('[data-action="delete-session"]')) {
    const title = row.querySelector('strong')?.textContent || '该会话';
    if (confirm(`确定永久删除“${title}”及其全部记录吗？`)) {
      send({type: 'delete_session', session_id: row.dataset.sessionId});
    }
  } else if (event.target.closest('.session-load')) {
    send({type: 'load_session', session_id: row.dataset.sessionId});
  }
});

$('#messages').addEventListener('click', async event => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const turn = button.closest('.turn');
  const action = button.dataset.action;
  if (action === 'speak') {
    await ensureAudio().catch(() => {});
    stopAudio();
    if (send({type: 'speak', text: turn._rawText, speed: state.speed})) setGenerating(true, '正在合成语音');
  } else if (action === 'copy') {
    await navigator.clipboard.writeText(turn._rawText);
    toast('已复制');
  } else if (action === 'retry') {
    send({type: 'chat', text: turn._question, mode: turn._mode, autoplay: state.autoplay, speed: state.speed});
  } else if (action === 'sources') {
    renderEvidence(turn._metadata);
  }
});

$('#default-mode').addEventListener('change', event => setMode(event.target.value));
window.addEventListener('pagehide', cancelRecording);
$('#autoplay').addEventListener('change', event => {
  state.autoplay = event.target.checked;
  localStorage.setItem('assistant-autoplay', state.autoplay);
});
$('#speech-speed').addEventListener('input', event => {
  state.speed = Number(event.target.value);
  $('#speed-label').textContent = `${state.speed.toFixed(1)}×`;
  localStorage.setItem('assistant-speed', state.speed);
});
$('#clear-chat').addEventListener('click', () => {
  if (confirm('确定清空当前对话吗？')) {
    send({type: 'clear_session'});
    show($('#settings'), false);
  }
});

initialize();
