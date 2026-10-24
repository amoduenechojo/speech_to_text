// Runs on the audio rendering thread; hands raw mic frames back to the main
// thread via the worklet port so app.js can resample/encode without glitching.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
