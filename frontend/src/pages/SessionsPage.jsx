import { useCallback, useEffect, useState } from 'react';
import { getSession, listSessions } from '../api';
import SessionReplay from '../components/SessionReplay';
import SessionSummary from '../components/SessionSummary';
import { formatDateTime, formatDuration } from '../utils/format';

export default function SessionsPage() {
  const [sessions, setSessions] = useState(null);
  const [showEmpty, setShowEmpty] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loadingId, setLoadingId] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setSessions(await listSessions());
    } catch (e) {
      setError(e.message);
      setSessions([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const open = async (id) => {
    setLoadingId(id);
    setError(null);
    try {
      setSelected(await getSession(id));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingId(null);
    }
  };

  if (selected) {
    return (
      <>
        <div className="status-row" style={{ justifyContent: 'space-between' }}>
          <button className="btn" onClick={() => setSelected(null)}>← All sessions</button>
          <span className="muted small-text">{formatDateTime(selected.session.started_at)}</span>
        </div>
        <div className="grid">
          <SessionSummary summary={selected.summary} />
          <SessionReplay results={selected.results} />
        </div>
      </>
    );
  }

  const all = sessions || [];
  const visible = all.filter((s) => showEmpty || s.n_results > 0);
  const hidden = all.length - visible.length;

  return (
    <section className="card" aria-label="Past sessions">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}>Past sessions</h2>
        <button className="btn small" onClick={load}>Refresh</button>
      </div>
      {error && <div className="banner error" role="alert" style={{ marginBottom: 10 }}>{error}</div>}
      {sessions === null ? (
        <p className="muted small-text">Loading…</p>
      ) : visible.length === 0 ? (
        <p className="muted small-text">No saved sessions yet. Start a live session, then come back here.</p>
      ) : (
        <ul className="list">
          {visible.map((s) => {
            const dur = s.ended_at ? (new Date(s.ended_at) - new Date(s.started_at)) / 1000 : null;
            return (
              <li key={s.session_id}>
                <button onClick={() => open(s.session_id)} disabled={loadingId === s.session_id}>
                  <span>{formatDateTime(s.started_at)}</span>
                  <span className="muted small-text">
                    {s.n_results} readings · {dur === null ? 'in progress' : formatDuration(dur)}
                    {loadingId === s.session_id ? ' · loading…' : ''}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {hidden > 0 && !showEmpty && (
        <button className="btn small" style={{ marginTop: 10 }} onClick={() => setShowEmpty(true)}>
          Show {hidden} empty session{hidden === 1 ? '' : 's'}
        </button>
      )}
    </section>
  );
}
