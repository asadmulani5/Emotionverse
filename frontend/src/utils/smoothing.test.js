import { smooth } from './smoothing';

test('returns null for no history', () => {
  expect(smooth([])).toBeNull();
});

test('averages distributions and picks the dominant', () => {
  const out = smooth([{ happy: 1, sad: 0 }, { happy: 0.5, sad: 0.5 }]);
  expect(out.emotions.happy).toBeCloseTo(0.75);
  expect(out.emotions.sad).toBeCloseTo(0.25);
  expect(out.dominant).toBe('happy');
});
