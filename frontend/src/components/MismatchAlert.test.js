import { describePair } from './MismatchAlert';

test('describes a disagreement in plain words', () => {
  const text = describePair({
    a: 'face', a_emotion: 'joy', a_confidence: 0.9,
    b: 'text', b_emotion: 'sadness', b_confidence: 0.8,
  });
  expect(text).toBe('Face shows Happy (90%) while Text shows Sad (80%)');
});
