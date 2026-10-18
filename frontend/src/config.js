// Backend URL comes from an env var at build time (set in Vercel / .env.local).
export const BACKEND = (process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');

// Client-side send throttle. Frames are also sent stop-and-wait: never while one is in flight.
export const MIN_SEND_INTERVAL_MS = Number(process.env.REACT_APP_MIN_SEND_INTERVAL_MS || 1000);
export const RESULT_TIMEOUT_MS = 8000;
export const SMOOTHING_WINDOW = 5;
