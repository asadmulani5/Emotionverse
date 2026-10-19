import { float32ToBase64, resample, rms } from './audio';

test('resample keeps same-rate audio unchanged', () => {
  const s = new Float32Array([0.1, 0.2, 0.3]);
  expect(resample(s, 16000, 16000)).toBe(s);
});

test('resample changes length by the rate ratio', () => {
  const s = new Float32Array(48000);
  expect(resample(s, 48000, 16000).length).toBe(16000);
});

test('rms of empty and simple signals', () => {
  expect(rms(new Float32Array([]))).toBe(0);
  expect(rms(new Float32Array([1, -1]))).toBeCloseTo(1);
});

test('float32ToBase64 encodes little-endian bytes', () => {
  expect(float32ToBase64(new Float32Array([0]))).toBe('AAAAAA==');
});
