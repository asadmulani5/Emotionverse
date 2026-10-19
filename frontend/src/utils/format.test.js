import { formatDateTime, formatDuration } from './format';

test('formats durations', () => {
  expect(formatDuration(5)).toBe('5s');
  expect(formatDuration(65)).toBe('1m 5s');
  expect(formatDuration(-3)).toBe('0s');
  expect(formatDuration(undefined)).toBe('0s');
});

test('formats date-times and handles bad input', () => {
  expect(formatDateTime(null)).toBe('—');
  expect(formatDateTime('not a date')).toBe('—');
  expect(formatDateTime('2026-10-17T18:37:02Z')).not.toBe('—');
});
