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
  updateEvidenceChip(turn);
  $('#messages').append(turn);
  scrollBottom();
  return turn;
}

function updateEvidenceChip(turn) {
  const chip = turn.querySelector('[data-action="sources"]');
  if (!chip) return;
  const count = Number(turn._metadata?.evidence_audit?.unverified_claims || 0);
  chip.textContent = count > 0 ? `查看回答依据（${count}条待核实）` : '查看回答依据';
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

const EVIDENCE_GENERIC_WORDS = new Set([
  '什么', '怎么', '如何', '是否', '可以', '请问', '告诉', '具体', '相关', '问题',
  '回答', '依据', '来源', '资料', '哪个', '哪些', '多少', '这个', '那个',
  '刚才', '之前', '给出', '根据', '当前', '如果', '以及', '还有', '需要',
]);

function foldEvidenceText(value) {
  // ASCII-only folding preserves character offsets for Chinese and punctuation.
  return value.replace(/[A-Z]/g, char => char.toLowerCase());
}

function evidenceKeywords(question, sourceTexts) {
  const words = [];
  const written = escapeText(question);
  if (typeof Intl.Segmenter === 'function') {
    const segmenter = new Intl.Segmenter('zh', {granularity: 'word'});
    for (const part of segmenter.segment(written)) {
      if (part.isWordLike) words.push(part.segment);
    }
  } else {
    words.push(...(written.match(/[A-Za-z0-9]+/g) || []));
  }
  // Chinese segmenters may split domain terms such as “年缴费率” into single
  // characters. Add short source-confirmed phrases so they remain findable.
  for (const block of written.match(/[\u3400-\u9fff]{2,}/g) || []) {
    for (let size = Math.min(5, block.length); size >= 2; size--) {
      for (let start = 0; start <= block.length - size; start++) {
        words.push(block.slice(start, start + size));
      }
    }
  }
  words.push(...(written.match(/\d{1,3}\s*岁/g) || []));
  const available = sourceTexts.map(foldEvidenceText);
  return [...new Set(words.map(word => word.trim()).filter(word =>
    word.length >= 2 && !EVIDENCE_GENERIC_WORDS.has(word) &&
    !/^\d$/.test(word) && available.some(text => text.includes(foldEvidenceText(word)))
  ))].sort((left, right) => right.length - left.length).slice(0, 8);
}

function highlightEvidenceText(element, source, keywords) {
  const text = escapeText(source);
  const folded = foldEvidenceText(text);
  const ranges = [];
  for (const keyword of keywords) {
    const needle = foldEvidenceText(keyword);
    if (!needle) continue;
    let from = 0;
    while (from < folded.length) {
      const start = folded.indexOf(needle, from);
      if (start < 0) break;
      ranges.push({start, end: start + needle.length});
      from = start + needle.length;
    }
  }
  ranges.sort((left, right) => left.start - right.start || right.end - left.end);
  if (!ranges.length) {
    element.textContent = text;
    return [];
  }
  element.replaceChildren();
  const marks = [];
  let cursor = 0;
  for (const range of ranges) {
    if (range.start < cursor) continue;
    if (range.start > cursor) {
      const plain = document.createElement('span');
      plain.textContent = text.slice(cursor, range.start);
      element.append(plain);
    }
    const mark = document.createElement('mark');
    mark.className = 'evidence-highlight';
    mark.textContent = text.slice(range.start, range.end);
    element.append(mark);
    marks.push(mark);
    cursor = range.end;
  }
  if (cursor < text.length) {
    const plain = document.createElement('span');
    plain.textContent = text.slice(cursor);
    element.append(plain);
  }
  return marks;
}

function renderEvidence(metadata = {}, question = '', answer = '') {
  const panel = $('#evidence-content');
  panel.replaceChildren();
  let sources = metadata.knowledge_mode === 'rag'
    ? (metadata.evidence_sources || metadata.sources || []) : (metadata.sources || []);
  // Older turns may contain a generated premium next to unrelated retrieved
  // policy clauses. Never present those clauses as proof of the quoted amount.
  const premium = metadata.knowledge_mode === 'rag'
    ? escapeText(answer).match(/([\d,]+)\s*元\s*\/\s*(年|月)/) : null;
  if (premium) {
    const amount = premium[1].replace(/,/g, '');
    const period = premium[2] === '年' ? '年缴' : '月缴';
    const numeric = new RegExp(`(^|\\D)${amount}(?!\\d)`);
    sources = sources.filter(source => {
      const name = escapeText(source.source_name);
      const body = escapeText(source.text).replace(/,/g, '');
      return name.includes('费率') && name.includes(period) && numeric.test(body);
    });
  }
  const missingPremiumBasis = Boolean(premium && !sources.length);
  const summary = document.createElement('div');
  summary.className = 'evidence-summary';
  if (missingPremiumBasis) {
    summary.textContent = '这条已保存的保费回答缺少可核验的对应费率表依据，原检索片段不能证明该金额。请重新提问；仍无法核实时需转人工处理。';
  } else if (metadata.evidence_audit?.unverified_claims) {
    summary.textContent = `回答完成后发现 ${metadata.evidence_audit.unverified_claims} 条保险结论未匹配到足够的原文依据，需人工核实；下方只展示与结论匹配的候选片段。`;
  } else if (metadata.evidence_audit?.matched_claims) {
    summary.textContent = `已按 ${metadata.evidence_audit.matched_claims} 条保险结论自动匹配原文片段；这不是独立的语义真伪核验，请核对适用条件。`;
  } else if (metadata.knowledge_mode === 'rag' && metadata.cooperation?.stages?.includes('fast:exact_rate_lookup')) {
    summary.textContent = `本轮基础费率按当前记录的用户条件从下列费率表读取；用户条件不是保险条款证据。检索 ${Math.round(metadata.retrieval_ms || 0)} ms`;
  } else if (metadata.knowledge_mode === 'rag') {
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
  const sourceRecords = [];
  sources.forEach((source, index) => {
    const card = document.createElement('article');
    card.className = 'source-card';
    const header = document.createElement('header');
    const page = document.createElement('span');
    const sourceName = source.source_name || 'PDF';
    const location = source.location || `第 ${source.pdf_page} 页`;
    const title = `${sourceName} · ${location}`;
    page.textContent = title;
    const score = document.createElement('span');
    score.textContent = `相似度 ${Number(source.score || 0).toFixed(3)}`;
    header.append(page, score);
    const text = document.createElement('p');
    const body = escapeText(source.text);
    text.textContent = body;
    card.append(header, text);
    panel.append(card);
    sourceRecords.push({id: source.chunk_id, card, page, title, text, body});
  });
  if (Array.isArray(metadata.evidence_claims) && metadata.evidence_claims.length) {
    const claimList = document.createElement('div');
    claimList.className = 'evidence-claims';
    for (const claim of metadata.evidence_claims) {
      const item = document.createElement('article');
      item.className = `evidence-claim ${claim.status === 'source_match' ? 'matched' : 'unverified'}`;
      const statement = document.createElement('p');
      statement.textContent = claim.claim;
      const label = document.createElement('small');
      label.textContent = claim.status !== 'source_match'
        ? '待人工核实：未找到足够匹配的原文'
        : claim.support_kind === 'base_exception_inference'
          ? '依据一般规则与例外规则推导（两条原文均列于下方，请核对适用条件）'
          : '对应原文片段（自动匹配）';
      item.append(statement, label);
      for (const id of claim.source_ids || []) {
        const source = sourceRecords.find(record => record.id === id);
        if (!source) continue;
        const link = document.createElement('button');
        link.type = 'button';
        link.textContent = source.title;
        link.addEventListener('click', () => source.card.scrollIntoView({block: 'center', behavior: 'smooth'}));
        item.append(link);
      }
      claimList.append(item);
    }
    panel.insertBefore(claimList, sourceRecords[0]?.card || null);
  }
  if (sourceRecords.length) {
    const search = document.createElement('div');
    search.className = 'evidence-search';
    const input = document.createElement('input');
    input.type = 'search';
    input.maxLength = 80;
    input.placeholder = '搜索依据关键词';
    input.setAttribute('aria-label', '搜索回答依据中的关键词');
    const next = document.createElement('button');
    next.type = 'button';
    next.textContent = '下一处';
    next.setAttribute('aria-label', '跳到下一处关键词');
    const status = document.createElement('small');
    status.setAttribute('role', 'status');
    search.append(input, next, status);
    panel.insertBefore(search, sourceRecords[0].card);

    const effectiveQuestion = metadata.query || metadata.cooperation?.query || question;
    const automatic = evidenceKeywords(effectiveQuestion, sourceRecords.flatMap(record => [record.title, record.body]));
    let marks = [];
    let active = -1;
    const refresh = () => {
      const typed = input.value.trim();
      const terms = typed ? [...new Set(typed.split(/[\s,，、;；]+/).filter(Boolean))].slice(0, 8) : automatic;
      marks = sourceRecords.flatMap(record => [
        ...highlightEvidenceText(record.page, record.title, terms),
        ...highlightEvidenceText(record.text, record.body, terms),
      ]);
      active = -1;
      next.disabled = !marks.length;
      status.textContent = typed
        ? (marks.length ? `找到 ${marks.length} 处匹配` : '未找到匹配关键词')
        : (marks.length ? `根据问题及有效上下文自动高亮 ${marks.length} 处` : '输入关键词可查找片段');
    };
    const jumpNext = () => {
      if (!marks.length) return;
      if (active >= 0) marks[active].classList.remove('active');
      active = (active + 1) % marks.length;
      marks[active].classList.add('active');
      marks[active].scrollIntoView({block: 'center', behavior: 'smooth'});
    };
    input.addEventListener('input', refresh);
    input.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); jumpNext(); }
    });
    next.addEventListener('click', jumpNext);
    refresh();
  }
  if (!sources.length) {
    const empty = document.createElement('p');
    empty.className = 'evidence-empty';
    empty.textContent = missingPremiumBasis ? '未找到与该保费匹配的费率表片段。' : '没有可展示的保险资料片段。';
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

function primeAudioFromGesture() {
  // Browsers may keep resume() pending until a user gesture starts a source.
  // Do both synchronously in the Play click; never delay the TTS request while
  // waiting for resume(), or another click could be required to hear it.
  try {
    if (!state.audioContext || state.audioContext.state === 'closed') {
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      state.audioCursor = 0;
    }
    const context = state.audioContext;
    if (context.state === 'suspended') {
      Promise.resolve(context.resume()).catch(error => toast(error.message || '无法启动语音播放'));
    }
    const silence = context.createBuffer(1, 1, context.sampleRate || 22050);
    const source = context.createBufferSource();
    source.buffer = silence;
    source.connect(context.destination);
    source.start(0);
    return true;
  } catch (error) {
    toast(error.message || '无法启动语音播放');
    return false;
  }
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
        updateEvidenceChip(state.currentTurn);
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
        updateEvidenceChip(state.currentTurn);
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
  // Stop the old voice before requesting microphone permission. Cancelling an
  // unfinished answer also prevents later TTS chunks from speaking over ASR.
  if (state.generating || state.currentTurn || state.audioSources.size || state.pendingAudio) {
    if (state.stopping) stopAudio();
    else stopReply();
  } else {
    stopAudio();
  }
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
    stopAudio();
    if (primeAudioFromGesture() && send({type: 'speak', text: turn._rawText, speed: state.speed})) {
      setGenerating(true, '正在合成语音');
    }
  } else if (action === 'copy') {
    await navigator.clipboard.writeText(turn._rawText);
    toast('已复制');
  } else if (action === 'retry') {
    send({type: 'chat', text: turn._question, mode: turn._mode, autoplay: state.autoplay, speed: state.speed});
  } else if (action === 'sources') {
    renderEvidence(turn._metadata, turn._question, turn._rawText);
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
