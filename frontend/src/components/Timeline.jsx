import { EMOTION_COLORS, titleCase } from '../theme/emotions';

export default function Timeline({ entries }) {
  return (
    <section className="card full" aria-label="Expression timeline">
      <h2>Timeline</h2>
      {entries.length === 0 ? (
        <p className="muted small-text">Changes in the detected expression will appear here.</p>
      ) : (
        <ul className="timeline">
          {[...entries].reverse().map((e) => (
            <li key={e.id}>
              <span className="pip" style={{ background: EMOTION_COLORS[e.emotion] || 'var(--muted)' }} />
              <span className="muted">{e.time}</span>
              <span>{titleCase(e.emotion)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
