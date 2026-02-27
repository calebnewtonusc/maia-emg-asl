/**
 * Sliding window buffer for real-time BLE EMG data.
 *
 * - Decodes 16-byte BLE packets (8xint16 big-endian)
 * - Maintains a 40-sample sliding window with 50% overlap
 * - Emits Float32Array windows (320 values: 40x8) ready for ONNX inference
 */

const N_CHANNELS = 8;
const WINDOW_SAMPLES = 40;
const HOP_SAMPLES = 20; // 50% overlap
const FLOAT_SCALE = 1 / 32768;

export type WindowCallback = (window: Float32Array) => void;

export class EMGWindowBuffer {
  private buffer: Float32Array;
  private writePtr: number = 0;
  private samplesUntilEmit: number = HOP_SAMPLES;
  private primed: boolean = false;
  private onWindow: WindowCallback;

  constructor(onWindow: WindowCallback) {
    this.buffer = new Float32Array(WINDOW_SAMPLES * N_CHANNELS);
    this.onWindow = onWindow;
  }

  /** Feed a 16-byte BLE packet (8 x int16 big-endian). */
  feed(packet: DataView): void {
    if (packet.byteLength < 16) return;

    // Decode one sample (8 channels x int16)
    const sample = new Float32Array(N_CHANNELS);
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      sample[ch] = packet.getInt16(ch * 2, false) * FLOAT_SCALE; // big-endian
    }

    // Write into circular buffer
    const offset = (this.writePtr % WINDOW_SAMPLES) * N_CHANNELS;
    this.buffer.set(sample, offset);
    this.writePtr++;

    if (this.writePtr < WINDOW_SAMPLES) return; // not enough data yet
    this.primed = true;

    this.samplesUntilEmit--;
    if (this.samplesUntilEmit <= 0) {
      this.samplesUntilEmit = HOP_SAMPLES;
      this.emitWindow();
    }
  }

  private emitWindow(): void {
    // Reconstruct ordered window from circular buffer
    const window = new Float32Array(WINDOW_SAMPLES * N_CHANNELS);
    const start = this.writePtr % WINDOW_SAMPLES;
    for (let i = 0; i < WINDOW_SAMPLES; i++) {
      const src = ((start + i) % WINDOW_SAMPLES) * N_CHANNELS;
      const dst = i * N_CHANNELS;
      window.set(this.buffer.subarray(src, src + N_CHANNELS), dst);
    }
    this.onWindow(window);
  }

  reset(): void {
    this.buffer.fill(0);
    this.writePtr = 0;
    this.samplesUntilEmit = HOP_SAMPLES;
    this.primed = false;
  }
}
