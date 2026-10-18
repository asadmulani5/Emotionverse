import { canonical, pct, titleCase } from '../theme/emotions';

export function describePair(p) {
  return `${titleCase(p.a)} shows ${titleCase(canonical(p.a_emotion))} (${pct(p.a_confidence)}) while ${titleCase(p.b)} shows ${titleCase(canonical(p.b_emotion))} (${pct(p.b_confidence)})`;
}

/** Shows the current mismatch (if any) and the last few that happened this session. */
export default function MismatchAlert({ current, recent }) {
  const pairs = current && current.detected ? current.pairs : [];
  return (
    <section className="card full" aria-label="Mismatch alerts">
      <h2>Mismatch alerts</h2>
      {pairs.length > 0 ? (
        <div className="banner warn" role="alert">
          <strong>Signals disagree.</strong>{' '}
          {pairs.map(describePair).join('; ')}. Neither is necessarily wrong; it can be worth noticing.
        </div>
      ) : (
        <p className="muted small-text">No strong disagreement between face, voice and text right now.</p>
      )}
      {recent.length > 0 && (
        <ul className="timeline" style={{ marginTop: 10 }} aria-label="Recent mismatches">
          {[...recent].reverse().map((m) => (
            <li key={m.id}>
              <span className="pip" style={{ background: 'var(--warn)' }} />
              <span className="muted">{m.time}</span>
              <span>{m.text}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
