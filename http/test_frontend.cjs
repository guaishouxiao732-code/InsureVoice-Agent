// Optional JavaScript regressions: run with Node.js (no browser required).
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');

const element = () => ({hidden:false, attributes:{},classList: {toggle(name,value){if(name==='hidden')this.hidden=value;},remove(){},add(){}},
  children: [],append(...children) {this.children.push(...children);},
  replaceChildren(...children) {this.children = children;},
  addEventListener(){},setAttribute(name,value){this.attributes[name]=value;},textContent:'',value:'',style:{}});
const nodes = new Map();
const context = vm.createContext({
  document: {querySelector: s => {if (!nodes.has(s)) nodes.set(s, element());return nodes.get(s);},
    querySelectorAll: () => [],createElement: element},
  localStorage: {getItem: () => null,setItem(){}},
  window: {addEventListener(){}},
  WebSocket: {OPEN: 1},
  requestAnimationFrame: callback => callback(),innerWidth: 1280,
  setTimeout, clearTimeout, console, atob: x => Buffer.from(x, 'base64').toString('binary'),
});
const app = fs.readFileSync(path.join(__dirname, 'static/app.js'), 'utf8');
vm.runInContext(app.replace(/initialize\(\);\s*$/, ''), context);
(async () => {
  // Audio initialization may finish after interruption; no old source may start.
  let resume;
  context.resumePromise = new Promise(resolve => {resume = resolve;});
  vm.runInContext(`state.audioContext = {state:'suspended',resume:()=>resumePromise,
    createBuffer:()=>{throw Error('stale audio was scheduled')}};`, context);
  const playing = vm.runInContext(`playPcm('AAA=',16000)`, context);
  vm.runInContext('stopAudio()', context);
  resume();
  await playing;
  vm.runInContext(`handleMessage({type:'response_epoch',response_id:5});
    handleMessage({type:'audio_chunk',response_id:4,pcm:'AAA=',sample_rate:16000});`, context);
  assert.equal(vm.runInContext('state.responseId', context), 5);
  vm.runInContext(`renderEvidence({knowledge_mode:'handoff',handoff:{status:'pending'},sources:[]});`, context);
  assert.match(nodes.get('#evidence-content').children[0].textContent, /待人工处理/);
  assert.doesNotMatch(nodes.get('#evidence-content').children[0].textContent, /补充内容来自/);
  vm.runInContext(`renderHistory([{user_text:'资料外问题',assistant_text:'需转人工处理',
    status:'completed',metadata:{knowledge_mode:'handoff',handoff:{status:'pending'}}}]);`, context);
  assert.match(nodes.get('#messages').children[0]._timing.textContent, /待人工处理/);
  vm.runInContext(`state.currentTurn = createTurn('资料外问题','fast');
    handleMessage({type:'turn_done',status:'completed',text:'需转人工处理',
      metadata:{knowledge_mode:'handoff',handoff:{status:'pending'}}});`, context);
  assert.match(nodes.get('#messages').children.at(-1)._timing.textContent, /尚未接入人工客服/);
  assert.equal(vm.runInContext('state.mode', context), 'auto');
  vm.runInContext(`state.currentTurn = createTurn('复杂问题','auto');
    handleMessage({type:'cooperation',plan:{resolved_mode:'slow',decision:'answer'}});`, context);
  assert.match(nodes.get('#messages').children.at(-1)._route.textContent, /快→慢/);
  vm.runInContext(`renderEvidence({knowledge_mode:'clarify',cooperation:{resolved_mode:'fast',
    decision:'clarify',context_version:2,slots:{age:{value:65,status:'explicit'}}}});`, context);
  assert.match(nodes.get('#evidence-content').children[0].textContent, /补充后继续检索/);
  assert.match(nodes.get('#evidence-content').children[1].textContent, /年龄：65/);
  assert.doesNotMatch(app, /listen_start|listen_stop|AudioWorklet|toggleListening/);
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  assert.doesNotMatch(html, /value="continuous"|id="voice-input-mode"/);
  vm.runInContext('updateMicrophoneUi()', context);
  assert.equal(nodes.get('#hero-mic b').textContent, '点击开始录音');
  context.sent = [];
  vm.runInContext(`state.ws={readyState:1,send:x=>sent.push(JSON.parse(x))};
    state.audioSources.add({stop(){}});handleMessage({type:'turn_done',text:'完整回答',status:'completed',metadata:{}});`,context);
  assert.equal(nodes.get('#stop').attributes['aria-label'],'停止播放');
  assert.equal(nodes.get('#stop').classList.hidden,false);
  vm.runInContext(`stopReply();handleMessage({type:'audio_chunk',response_id:5,pcm:'AAA=',sample_rate:16000});`,context);
  assert.equal(vm.runInContext('state.audioSources.size',context),0);
  assert.equal(vm.runInContext('state.generating',context),false);
  assert.equal(context.sent.at(-1).type,'interrupt');
  vm.runInContext(`renderEvidence({knowledge_mode:'rag',sources:[{source_name:'未选中资料'}],
    evidence_sources:[{source_name:'已选中资料',location:'第1页',text:'实际注入模型的片段'}]})`,context);
  assert.equal(nodes.get('#evidence-content').children.length,2);
  assert.match(nodes.get('#evidence-content').children[1].children[0].children[0].textContent,/已选中资料/);
  console.log('PASS: stop remains visible during playback; immediate local cancellation rejects stale audio; evidence uses actual selected sources');
  for (const savedMode of ['continuous', 'recording', null]) {
    const preferenceContext = vm.createContext({...context,
      localStorage: {getItem: key => key.startsWith('assistant-voice-mode') ? savedMode : null, setItem(){}},
    });
    vm.runInContext(app.replace(/initialize\(\);\s*$/, ''), preferenceContext);
    assert.equal(vm.runInContext('typeof toggleListening', preferenceContext), 'undefined');
  }
  let permissionCalls = 0;
  const stoppedTracks = [];
  context.navigator = {mediaDevices: {getUserMedia: async () => {
    permissionCalls++;
    return {getTracks: () => [{stop: () => stoppedTracks.push(true)}]};
  }}};
  context.Blob = Blob;
  context.btoa = x => Buffer.from(x, 'binary').toString('base64');
  context.sent = [];
  context.recorders = [];
  context.MediaRecorder = class {
    static isTypeSupported() {return true;}
    constructor(stream, options) {
      this.mimeType = options.mimeType;
      this.state = 'inactive';
      context.recorders.push(this);
    }
    start() {this.state = 'recording';}
    stop() {
      this.state = 'inactive';
      this.ondataavailable({data: new Blob(['recording-test'])});
      this.stopped = this.onstop();
    }
  };
  vm.runInContext("state.audioContext = {state:'running'}; state.ws = {readyState:1,send:x=>sent.push(JSON.parse(x))}", context);
  const starting = vm.runInContext('toggleMicrophone()', context);
  await vm.runInContext('toggleMicrophone()', context); // Repeated click while permission is pending.
  await starting;
  assert.equal(permissionCalls, 1);
  assert.equal(context.recorders.length, 1);
  assert.equal(context.sent.length, 0, 'starting recording must not submit');
  assert.equal(nodes.get('#hero-mic b').textContent, '点击结束并发送');
  await vm.runInContext('toggleMicrophone()', context);
  await context.recorders[0].stopped;
  assert.equal(context.sent.length, 1);
  assert.equal(context.sent[0].type, 'voice');
  assert.equal(Buffer.from(context.sent[0].audio, 'base64').toString(), 'recording-test');
  assert.equal(stoppedTracks.length, 1);
  assert.equal(nodes.get('#hero-mic b').textContent, '点击开始录音');
  await vm.runInContext('toggleMicrophone()', context);
  vm.runInContext('cancelRecording()', context);
  await context.recorders[1].stopped;
  assert.equal(context.sent.length, 1, 'cancelled recording must not submit');
  assert.equal(stoppedTracks.length, 2);
  console.log('PASS: only click-to-record; old continuous preferences ignored; second click submits once; cancellation releases microphone without upload');
  console.log('PASS: asynchronous playback cancellation; stale response rejection');
  console.log('PASS: pending handoff in live answer, restored history and evidence panel');
  console.log('PASS: default auto mode, cooperative route updates and user condition display');
})().catch(error => {console.error(error);process.exitCode = 1;});
