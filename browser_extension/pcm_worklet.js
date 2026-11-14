class IgnaitCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = (options && options.processorOptions) || {};
    this.targetRate = 16000;
    this.frameSamples = 512; // 32 ms @ 16 kHz (exacto para Silero VAD v5)
    this.enabled = Boolean(processorOptions.enabled);
    this._phase = 0.0;
    this._accum = new Int16Array(this.frameSamples);
    this._accumPos = 0;

    this.port.onmessage = (event) => {
      const msg = event.data;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "setEnabled") {
        this.enabled = Boolean(msg.enabled);
        if (!this.enabled) {
          this._phase = 0.0;
          this._accumPos = 0;
        }
      }
    };
  }

  process(inputs) {
    if (!this.enabled) {
      return true;
    }
    const input = inputs && inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) {
      return true;
    }
    const nCh = input.length;
    const blockLen = ch0.length;
    const step = sampleRate / this.targetRate;

    let idx = this._phase;
    while (idx < blockLen) {
      const i0 = Math.floor(idx);
      const frac = idx - i0;
      const i1 = i0 + 1 < blockLen ? i0 + 1 : i0;

      let s0 = 0.0;
      let s1 = 0.0;
      for (let c = 0; c < nCh; c++) {
        const ch = input[c];
        s0 += ch[i0] || 0.0;
        s1 += ch[i1] || 0.0;
      }
      const mono = (s0 * (1.0 - frac) + s1 * frac) / nCh;
      const clamped = mono > 1.0 ? 1.0 : mono < -1.0 ? -1.0 : mono;
      this._accum[this._accumPos++] =
        clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      if (this._accumPos >= this.frameSamples) {
        const outBuf = this._accum.buffer;
        this.port.postMessage(outBuf, [outBuf]);
        this._accum = new Int16Array(this.frameSamples);
        this._accumPos = 0;
      }
      idx += step;
    }

    this._phase = idx - blockLen;
    return true;
  }
}

registerProcessor("ignait-capture-processor", IgnaitCaptureProcessor);
