import { canonical, EMOTION_COLORS, pct, titleCase } from '../theme/emotions';

const MODALITIES = [
  { key: 'face', label: 'Face' },
  { key: 'voice', label: 'Voice' },
  { key: 'text', label: 'Text' },
];
const SEG_COLORS = { face: '#4f46e5', voice: '#0891b2', text: '#c026d3' };

function topOf(res) {
  if (!res || !res.emotions) return null;
  const dom = res.dominant;
  return { emotion: canonical(dom), score: res.emotions[dom] };
}

function whyMissing(res) {
  if (!res || Object.keys(res).length === 0) return 'not sent (off or no input)';
  if (res.error) return res.error;
  return null;
}

/** Explains the latest fused reading: how much each modality contributed to the top expression. */
export default function ContributionPanel({ result }) {
  const fused = result && result.fused;
  if (!fused || !fused.dominant) {
    return (
      <section className="card full" aria-label="Explanation">
        <h2>Why this reading?</h2>
        <p className="muted small-text">Once a reading arrives, you'll see which inputs shaped it.</p>
      </section>
    );
  }
  const dom = fused.dominant;
  const total = fused.emotions[dom] || 0;
  const used = fused.used || [];
  return (
    <section className="card full" aria-label="Explanation">
      <h2>Why this reading?</h2>
      <p style={{ margin: '0 0 8px' }}>
        Top detected expression: <strong style={{ color: EMOTION_COLORS[dom] }}>{titleCase(dom)}</strong> ({pct(total)}).
        Share of that score contributed by each input:
      </p>
      <div className="seg" role="img" aria-label="Contribution of each input to the top expression">
        {used.map((m) => {
          const c = (fused.contributions[m] || {})[dom] || 0;
          return <span key={m} title={`${m}: ${pct(total ? c / total : 0)}`} style={{ width: `${total ? (c / total) * 100 : 0}%`, background: SEG_COLORS[m] }} />;
        })}
      </div>
      <table className="table" style={{ marginTop: 10 }}>
        <thead>
          <tr><th>Input</th><th>Weight</th><th>Its top expression</th><th>Share of result</th></tr>
        </thead>
        <tbody>
          {MODALITIES.map(({ key, label }) => {
            const isUsed = used.includes(key);
            const top = topOf(result[key]);
            const c = isUsed ? (fused.contributions[key] || {})[dom] || 0 : 0;
            return (
              <tr key={key}>
                <td><span className="pip" style={{ display: 'inline-block', background: SEG_COLORS[key], marginRight: 6 }} />{label}</td>
                <td>{isUsed ? pct(fused.weights[key]) : '–'}</td>
                <td>{isUsed && top ? `${titleCase(top.emotion)} (${pct(top.score)})` : <span className="muted">{whyMissing(result[key]) || 'unused'}</span>}</td>
                <td>{isUsed ? pct(total ? c / total : 0) : '–'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small-text" style={{ margin: '8px 0 0' }}>
        Weights are re-balanced across the inputs that returned a result.
      </p>
    </section>
  );
}
