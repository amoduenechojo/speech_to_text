const TARGET_SAMPLE_RATE = 16000;
// The mic worklet fires every ~128 native-rate samples, but the backend's VAD
// needs 512-sample frames to score anything at all — so we accumulate
// downsampled audio here and only flush over the socket once we have enough.
const SEND_CHUNK_SAMPLES = 1600; // 100ms at 16kHz

let sendBuffer = new Float32Array(0);

const captionEl = document.getElementById("caption");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-btn");
const refreshBtn = document.getElementById("refresh-btn");
const levelBar = document.getElementById("level-bar");

let audioContext = null;
let socket = null;
let sourceSampleRate = 48000;

function wsUrl() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/ws/transcribe`;
}

function setStatus(text, ok) {
  statusEl.textContent = text;
  statusEl.classList.toggle("status--ok", Boolean(ok));
  statusEl.classList.toggle("status--error", ok === false);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderCaption(committed, pending) {
  captionEl.innerHTML =
      `<span class="caption__committed">${escapeHtml(committed)}</span> ` +
      `<span class="caption__pending">${escapeHtml(pending)}</span>`;
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) return buffer;
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(buffer.length / ratio);
  const result = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, buffer.length - 1);
    const frac = srcIndex - i0;
    result[i] = buffer[i0] * (1 - frac) + buffer[i1] * frac;
  }
  return result;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function updateLevel(float32Array) {
  let sum = 0;
  for (let i = 0; i < float32Array.length; i++) sum += float32Array[i] * float32Array[i];
  const rms = Math.sqrt(sum / float32Array.length);
  const pct = Math.min(100, Math.round(rms * 400));
  levelBar.style.width = `${pct}%`;
}

function connectSocket() {
  const s = new WebSocket(wsUrl());
  s.binaryType = "arraybuffer";

  s.onopen = () => setStatus("Listening — audio stays on this server", true);
  s.onclose = () => setStatus("Disconnected", false);
  s.onerror = () => setStatus("Connection error", false);
  s.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "transcript") {
      renderCaption(data.committed, data.pending);
    }
  };

  return s;
}

async function start() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { autoGainControl: true, noiseSuppression: true, echoCancellation: true },
  });

  // Ask for 16kHz directly so the browser's own audio pipeline does the
  // sample-rate conversion (with a proper anti-aliasing filter) before the
  // worklet ever sees a sample. Not every OS/hardware combo honors the
  // requested rate, so we check what we actually got and only fall back to
  // manual (unfiltered) downsampling in downsampleBuffer() if it didn't take.
  audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  sourceSampleRate = audioContext.sampleRate;
  if (sourceSampleRate !== TARGET_SAMPLE_RATE) {
    console.warn(
        `AudioContext ignored requested sampleRate; running at ${sourceSampleRate}Hz and ` +
        `falling back to manual downsampling to ${TARGET_SAMPLE_RATE}Hz.`
    );
  }

  await audioContext.audioWorklet.addModule("assets/audio-processor.js");
  const source = audioContext.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioContext, "pcm-capture-processor");

  socket = connectSocket();

  sendBuffer = new Float32Array(0);

  worklet.port.onmessage = (event) => {
    const chunk = event.data;
    updateLevel(chunk);
    const downsampled = downsampleBuffer(chunk, sourceSampleRate, TARGET_SAMPLE_RATE);

    const merged = new Float32Array(sendBuffer.length + downsampled.length);
    merged.set(sendBuffer);
    merged.set(downsampled, sendBuffer.length);
    sendBuffer = merged;

    while (sendBuffer.length >= SEND_CHUNK_SAMPLES) {
      const toSend = sendBuffer.subarray(0, SEND_CHUNK_SAMPLES);
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(floatTo16BitPCM(toSend));
      }
      sendBuffer = sendBuffer.subarray(SEND_CHUNK_SAMPLES);
    }
  };

  source.connect(worklet);
  startBtn.disabled = true;
  startBtn.textContent = "Listening…";
  refreshBtn.disabled = false;
}

startBtn.addEventListener("click", () => {
  setStatus("Connecting…");
  start().catch((err) => setStatus(`Mic error: ${err.message}`, false));
});

refreshBtn.addEventListener("click", () => {
  renderCaption("", "");
  sendBuffer = new Float32Array(0);
  if (socket) {
    socket.close();
    socket = connectSocket();
  }
});