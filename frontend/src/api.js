import { getClientId } from './clientId';
import { BACKEND } from './config';

async function getJson(path) {
  let res;
  try {
    // The private per-browser id: the server only returns this browser's sessions.
    res = await fetch(`${BACKEND}${path}`, { headers: { 'X-Client-Id': getClientId() } });
  } catch {
    throw new Error(`Can't reach the server at ${BACKEND}.`);
  }
  if (res.status === 503) throw new Error('Session history is unavailable because the database is not connected.');
  if (res.status === 404) throw new Error('That session no longer exists.');
  if (!res.ok) throw new Error(`Server error (${res.status}).`);
  return res.json();
}

export const listSessions = () => getJson('/sessions?limit=50').then((d) => d.sessions || []);
export const getSession = (id) => getJson(`/sessions/${encodeURIComponent(id)}`);
