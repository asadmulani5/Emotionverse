import { EMOTION_COLORS, EMOTIONS, pct, titleCase } from '../theme/emotions';
import { formatDuration } from '../utils/format';

/** Summary computed by the backend (summary.py): dominant expression, average mix, spikes. */
export default function SessionSummary({ summary }) {
  if (!summary || !summary.dominant) {
    return (
      <section className="card full" aria-label="Session summary">
        <h2>Summary</h2>
        <p className="muted small-text">No readings were saved in this session.</p>
      </section>
    );
  }
  const dist = summary.distribution || {};
  const used = Object.entries(summary.modalities_used || {})
    .map(([m, n]) => `${titleCase(m)} (${n})`)
    .join(', ') || '—';
  const mm = summary.mismatch_count || 0;

  return (
    <section className="card full" aria-label="Session summary">
      <h2>Summary</h2>
      <p style={{ margin: '0 0 10px' }}>
        Most detected expression:{' '}
        <strong style={{ color: EMOTION_COLORS[summary.dominant] }}>{titleCase(summary.dominant)}</strong>{' '}
        <span className="muted">
          · {summary.n_results} readings · {formatDuration(summary.duration_s)} · {mm} mismatch{mm === 1 ? '' : 'es'}
        </span>
      </p>
      <div className="seg" role="img" aria-label="Average expression mix over the session">
        {EMOTIONS.filter((e) => dist[e] > 0).map((e) => (
          <span key={e} title={`${titleCase(e)}: ${pct(dist[e])}`} style={{ width: `${dist[e] * 100}%`, background: EMOTION_COLORS[e] }} />
        ))}
      </div>
      <p className="muted small-text" style={{ margin: '6px 0 12px' }}>Inputs used: {used}</p>
      <h3 className="small-text muted" style={{ marginBottom: 6 }}>Notable spikes</h3>
      {summary.spikes && summary.spikes.length ? (
        <ul className="timeline">
          {summary.spikes.map((s) => (
            <li key={`${s.emotion}-${s.offset_s}`}>
              <span className="pip" style={{ background: EMOTION_COLORS[s.emotion] || 'var(--muted)' }} />
              <span className="muted">+{formatDuration(s.offset_s)}</span>
              <span>{titleCase(s.emotion)} ({pct(s.score)})</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small-text">No sharp spikes above the session average.</p>
      )}
    </section>
  );
}
