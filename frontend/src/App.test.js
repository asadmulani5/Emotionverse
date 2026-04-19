import { render, screen } from '@testing-library/react';
import App from './App';

test('renders EmotionVerse title', () => {
  render(<App />);
  const title = screen.getByText(/EmotionVerse/i);
  expect(title).toBeInTheDocument();
});