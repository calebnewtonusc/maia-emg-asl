/**
 * MAIA On-Device Inference
 *
 * Quick start:
 *   import { ONNXInference, EMGWindowBuffer } from './inference';
 *
 *   const engine = await ONNXInference.getInstance();
 *   const buffer = new EMGWindowBuffer(async (window) => {
 *     const result = await engine.predict(window);
 *     if (result?.class) console.log('ASL:', result.class);
 *   });
 *
 *   // On each BLE packet (DataView of 16 bytes):
 *   buffer.feed(blePacket);
 */

export { ONNXInference } from './ONNXInference';
export type { PredictionResult } from './ONNXInference';
export { EMGWindowBuffer } from './EMGWindowBuffer';
export type { WindowCallback } from './EMGWindowBuffer';
