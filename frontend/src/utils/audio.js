// Audio helpers (pure functions, unit-tested).

// Linear-resample a Float32Array to a target sample rate.
export function resample(samples, fromRate, toRate = 16000) {
  if (fromRate === toRate) return samples;
  const n = Math.floor((samples.length * toRate) / fromRate);
  const out = new Float32Array(n);
  const ratio = (samples.length - 1) / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, samples.length - 1);
    out[i] = samples[lo] + (samples[hi] - samples[lo]) * (pos - lo);
  }
  return out;
}

export function rms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

// Float32Array -> base64 of little-endian float32 bytes (what the backend accepts).
export function float32ToBase64(samples) {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}
