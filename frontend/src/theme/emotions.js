// Canonical expression labels used by the backend fusion engine, with display colours.
export const EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust'];

export const EMOTION_COLORS = {
  happy: '#16a34a',
  sad: '#2563eb',
  angry: '#dc2626',
  neutral: '#6b7280',
  surprise: '#d97706',
  fear: '#9333ea',
  disgust: '#0d9488',
};

// Per-model labels -> canonical (mirrors backend/fusion.py LABEL_MAP)
const LABEL_MAP = {
  sadness: 'sad', joy: 'happy', anger: 'angry', love: 'happy',
  fearful: 'fear', surprised: 'surprise', calm: 'neutral',
};

export const canonical = (label) => (label ? LABEL_MAP[label.toLowerCase()] || label.toLowerCase() : label);
export const titleCase = (s) => (s ? s[0].toUpperCase() + s.slice(1) : '');
export const pct = (v) => `${Math.round((v || 0) * 100)}%`;
