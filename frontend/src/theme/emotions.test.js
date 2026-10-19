import { canonical, pct, titleCase } from './emotions';

test('maps model labels to canonical labels', () => {
  expect(canonical('Joy')).toBe('happy');
  expect(canonical('calm')).toBe('neutral');
  expect(canonical('sad')).toBe('sad');
  expect(canonical(null)).toBeNull();
});

test('titleCase and pct', () => {
  expect(titleCase('sad')).toBe('Sad');
  expect(titleCase('')).toBe('');
  expect(pct(0.456)).toBe('46%');
  expect(pct(undefined)).toBe('0%');
});
