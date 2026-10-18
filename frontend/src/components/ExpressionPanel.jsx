import { EMOTIONS, EMOTION_COLORS, pct, titleCase } from '../theme/emotions';

/** `smoothed` = { emotions: {happy: .., ...}, dominant } or null. Wording: "detected expression", never "you feel". */
export default function ExpressionPanel({ smoothed }) {
  const emotions = smoothed ? smoothed.emotions : null;
  const dominant = smoothed ? smoothed.dominant : null;
  return (
    <section className="card" aria-label="Detected expression">
      <h2>Detected expression</h2>
      <div className="readout" data-testid="dominant">
        {dominant ? (
          <>
            <span style={{ color: EMOTION_COLORS[dominant] }}>{titleCase(dominant)}</span>{' '}
            <span className="muted" style={{ fontSize: '1rem' }}>{pct(emotions[dominant])}</span>
          </>
        ) : (
          <span className="muted">Waiting for input…</span>
        )}
        <span className="label">An estimate from facial, vocal and text cues, not proof of how anyone feels.</span>
      </div>
      <div className="bars">
        {EMOTIONS.map((e) => (
          <div className="bar-row" key={e}>
            <span>{titleCase(e)}</span>
            <div className="bar-track" role="presentation">
              <div className="bar-fill" style={{ width: pct(emotions ? emotions[e] : 0), background: EMOTION_COLORS[e] }} />
            </div>
            <span className="muted">{pct(emotions ? emotions[e] : 0)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
