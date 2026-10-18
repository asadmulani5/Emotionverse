// localStorage can throw (private mode, blocked site data); never let that break the app.
export function readPref(key, fallback = null) {
  try {
    const v = window.localStorage.getItem(key);
    return v === null ? fallback : v;
  } catch {
    return fallback;
  }
}

export function writePref(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

export const CONSENT_KEY = 'ev_consent_v1';
export const THEME_KEY = 'ev_theme';
