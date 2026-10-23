import { readPref, writePref } from './storage';

// A random private id for this browser. The backend stores sessions with it and
// only returns sessions to the same id, so visitors can't see each other's history.
const KEY = 'ev_client_id';
const VALID = /^[A-Za-z0-9_-]{16,64}$/;
let memoryId = null; // fallback when localStorage is blocked

function randomId() {
  const c = typeof window !== 'undefined' ? window.crypto : undefined;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  const bytes = new Uint8Array(16);
  if (c && typeof c.getRandomValues === 'function') c.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function getClientId() {
  const stored = readPref(KEY);
  if (stored && VALID.test(stored)) return stored;
  if (!memoryId) memoryId = randomId();
  writePref(KEY, memoryId);
  return memoryId;
}
