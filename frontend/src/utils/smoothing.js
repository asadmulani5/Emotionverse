// Average the last N emotion distributions to stop the display flickering.
export function smooth(history) {
  if (!history.length) return null;
  const avg = {};
  history.forEach((e) => {
    Object.entries(e).forEach(([k, v]) => {
      avg[k] = (avg[k] || 0) + v;
    });
  });
  Object.keys(avg).forEach((k) => {
    avg[k] /= history.length;
  });
  const dominant = Object.keys(avg).reduce((a, b) => (avg[a] >= avg[b] ? a : b));
  return { emotions: avg, dominant };
}
