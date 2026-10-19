import { fireEvent, render, screen } from '@testing-library/react';
import ConsentScreen from './ConsentScreen';

test('accept and decline call the right handlers', () => {
  const onAccept = jest.fn();
  const onDecline = jest.fn();
  render(<ConsentScreen declined={false} onAccept={onAccept} onDecline={onDecline} />);
  fireEvent.click(screen.getByText('I understand, continue'));
  fireEvent.click(screen.getByText('No thanks'));
  expect(onAccept).toHaveBeenCalledTimes(1);
  expect(onDecline).toHaveBeenCalledTimes(1);
});

test('declined state confirms nothing was captured', () => {
  render(<ConsentScreen declined onAccept={() => {}} onDecline={() => {}} />);
  expect(screen.getByText('No problem')).toBeInTheDocument();
  expect(screen.getByText(/Nothing was captured or sent/)).toBeInTheDocument();
});
