import { useEffect, useState } from 'react';
import ConsentScreen from './components/ConsentScreen';
import LivePage from './pages/LivePage';
import SessionsPage from './pages/SessionsPage';
import { CONSENT_KEY, readPref, THEME_KEY, writePref } from './storage';

const THEMES = ['system', 'light', 'dark'];

export default function App() {
  const [consent, setConsent] = useState(() => readPref(CONSENT_KEY) === 'granted');
  const [declined, setDeclined] = useState(false);
  const [theme, setTheme] = useState(() => readPref(THEME_KEY, 'system'));
  const [page, setPage] = useState('live');

  useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    writePref(THEME_KEY, theme);
  }, [theme]);

  const accept = () => {
    writePref(CONSENT_KEY, 'granted');
    setDeclined(false);
    setConsent(true);
  };
  const revoke = () => {
    writePref(CONSENT_KEY, null);
    setConsent(false);
  };

  const tab = (id, label) => (
    <button className="tab" aria-current={page === id ? 'page' : undefined} onClick={() => setPage(id)}>
      {label}
    </button>
  );

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>EmotionVerse</h1>
          <p>A private emotion mirror for self-reflection</p>
        </div>
        <nav className="nav" aria-label="Main">
          {consent && (
            <>
              {tab('live', 'Live')}
              {tab('sessions', 'Sessions')}
            </>
          )}
          <button
            className="btn"
            onClick={() => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length])}
            aria-label={`Theme: ${theme}. Click to change.`}
          >
            Theme: {theme}
          </button>
        </nav>
      </header>

      {!consent ? (
        <ConsentScreen declined={declined} onAccept={accept} onDecline={() => setDeclined(true)} />
      ) : page === 'live' ? (
        <LivePage />
      ) : (
        <SessionsPage />
      )}

      <p className="footer">
        Detected expressions are estimates, not facts about feelings.
        {consent && (
          <>
            {' '}
            <button className="btn small" onClick={revoke}>Privacy &amp; consent</button>
          </>
        )}
      </p>
    </div>
  );
}
