import { render, screen } from '@testing-library/react';
import SessionSummary from './SessionSummary';

test('shows an empty state when nothing was saved', () => {
  render(<SessionSummary summary={null} />);
  expect(screen.getByText(/No readings were saved/)).toBeInTheDocument();
});

test('renders dominant expression, counts and spikes', () => {
  render(
    <SessionSummary
      summary={{
        dominant: 'happy',
        n_results: 10,
        duration_s: 65,
        mismatch_count: 1,
        distribution: { happy: 0.7, sad: 0.3 },
        modalities_used: { face: 10 },
        spikes: [{ emotion: 'surprise', offset_s: 5, score: 0.9 }],
      }}
    />
  );
  expect(screen.getByText('Happy')).toBeInTheDocument();
  expect(screen.getByText(/10 readings · 1m 5s · 1 mismatch$/)).toBeInTheDocument();
  expect(screen.getByText(/Face \(10\)/)).toBeInTheDocument();
  expect(screen.getByText('Surprise (90%)')).toBeInTheDocument();
});
