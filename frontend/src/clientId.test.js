import { getClientId } from './clientId';

beforeEach(() => window.localStorage.clear());

test('creates a valid id, stores it and reuses it', () => {
  const id = getClientId();
  expect(id).toMatch(/^[A-Za-z0-9_-]{16,64}$/);
  expect(getClientId()).toBe(id);
  expect(window.localStorage.getItem('ev_client_id')).toBe(id);
});

test('keeps an existing stored id', () => {
  window.localStorage.setItem('ev_client_id', 'abcdefghijklmnop1234');
  expect(getClientId()).toBe('abcdefghijklmnop1234');
});

test('replaces an invalid stored id', () => {
  window.localStorage.setItem('ev_client_id', 'bad');
  expect(getClientId()).not.toBe('bad');
});
