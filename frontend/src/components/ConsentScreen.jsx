import { useEffect, useRef } from 'react';

/** First-load privacy notice. Nothing is captured or connected until the user agrees. */
export default function ConsentScreen({ declined, onAccept, onDecline }) {
  const btn = useRef(null);
  useEffect(() => {
    btn.current?.focus();
  }, []);

  if (declined) {
    return (
      <div className="overlay">
        <div className="modal" role="dialog" aria-modal="true" aria-labelledby="consent-title">
          <h2 id="consent-title">No problem</h2>
          <p>EmotionVerse hasn't accessed your camera, microphone or text. Nothing was captured or sent.</p>
          <div className="actions">
            <button ref={btn} className="btn primary" onClick={onAccept}>Review again</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="overlay">
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="consent-title">
        <h2 id="consent-title">Before you start</h2>
        <p>
          EmotionVerse is a <strong>personal self-reflection tool</strong>: a mirror that shows
          the expression your face, voice and words appear to show.
        </p>
        <ul>
          <li><strong>Estimates, not feelings.</strong> Results are the detected expression, not proof of how you feel.</li>
          <li><strong>Only for yourself.</strong> Do not use it to monitor employees, students or anyone who hasn't agreed. This is prohibited in workplaces and schools under the EU AI Act.</li>
          <li><strong>You choose the inputs.</strong> Camera, microphone and text can each be switched off at any time.</li>
          <li><strong>Where data goes.</strong> Frames, audio and text are sent to the EmotionVerse server for analysis and are not stored. If you run it locally, they never leave your computer. A hosted demo sends them to its server, so run it locally for full privacy.</li>
          <li><strong>What is saved.</strong> Only emotion scores with timestamps, never images, audio or what you typed, so you can replay past sessions.</li>
        </ul>
        <div className="actions">
          <button className="btn" onClick={onDecline}>No thanks</button>
          <button ref={btn} className="btn primary" onClick={onAccept}>I understand, continue</button>
        </div>
      </div>
    </div>
  );
}
