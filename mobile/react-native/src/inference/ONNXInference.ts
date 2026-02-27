/**
 * On-device ONNX inference for ASL classification.
 * Uses onnxruntime-react-native.
 *
 * Usage:
 *   const engine = await ONNXInference.getInstance();
 *   const result = await engine.predict(window); // Float32Array of 320 values
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';

const N_CLASSES = 26;
const ASL_CLASSES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const CONFIDENCE_THRESHOLD = 0.75;
const DEBOUNCE_MS = 300;
const INPUT_SIZE = 40 * 8; // WINDOW_SAMPLES x N_CHANNELS

export interface PredictionResult {
  class: string | null;
  confidence: number;
  probabilities: Float32Array;
  latencyMs: number;
}

export class ONNXInference {
  private static instance: ONNXInference | null = null;
  private session: InferenceSession | null = null;
  private lastEmitTime: number = 0;

  private constructor() {}

  static async getInstance(): Promise<ONNXInference> {
    if (!ONNXInference.instance) {
      ONNXInference.instance = new ONNXInference();
      await ONNXInference.instance.loadModel();
    }
    return ONNXInference.instance;
  }

  private async loadModel(): Promise<void> {
    try {
      // Model should be bundled at assets/model/asl_emg_classifier.onnx
      this.session = await InferenceSession.create(
        require('../../../assets/model/asl_emg_classifier.onnx'),
        { executionProviders: ['cpu'] }
      );
      console.log('[MAIA] ONNX model loaded');
    } catch (err) {
      console.error('[MAIA] Failed to load ONNX model:', err);
      this.session = null;
    }
  }

  async predict(window: Float32Array): Promise<PredictionResult | null> {
    if (!this.session) return null;

    // Debounce
    const now = Date.now();
    if (now - this.lastEmitTime < DEBOUNCE_MS) return null;

    const t0 = performance.now();
    try {
      const inputTensor = new Tensor('float32', window, [1, INPUT_SIZE]);
      const feeds = { input: inputTensor };
      const results = await this.session.run(feeds);
      const logits = results['output'].data as Float32Array;
      const probs = softmax(logits);
      const maxIdx = argmax(probs);
      const confidence = probs[maxIdx];
      const latencyMs = performance.now() - t0;

      if (confidence >= CONFIDENCE_THRESHOLD) {
        this.lastEmitTime = now;
        return {
          class: ASL_CLASSES[maxIdx],
          confidence,
          probabilities: probs,
          latencyMs,
        };
      }
      return {
        class: null,
        confidence,
        probabilities: probs,
        latencyMs,
      };
    } catch (err) {
      console.error('[MAIA] Inference error:', err);
      return null;
    }
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  static reset(): void {
    ONNXInference.instance = null;
  }
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exp = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exp[i] = Math.exp(logits[i] - max);
    sum += exp[i];
  }
  return exp.map(v => v / sum);
}

function argmax(arr: Float32Array): number {
  let maxIdx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[maxIdx]) maxIdx = i;
  }
  return maxIdx;
}
