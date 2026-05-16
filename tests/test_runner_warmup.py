"""Unit tests for the continuous-carrier helper in :mod:`runner.runner`.

Phase F replaced Phase E's warmup-then-stop pattern with continuous
carrier traffic spanning warmup + measurement, with a single special
case: ``cold_idle_reference: true`` skips the carrier entirely so we
keep a cold-baseline data point for the paper §5.2 contrast.

These tests don't bring up a real network; they patch
``BackgroundTraffic`` and assert the call shape of
``make_continuous_carrier``. The full end-to-end behaviour is
exercised by the Phase F pilot run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from runner.runner import make_continuous_carrier


def test_carrier_zero_rate_returns_none() -> None:
    """``rate_mbps <= 0`` is the cold-idle path: no BG instantiated."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        result = make_continuous_carrier(
            net=object(),
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=0,
        )
        assert result is None
        bg_class.assert_not_called()


def test_carrier_negative_rate_returns_none() -> None:
    """Defensive: negative rate is also a no-op."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        result = make_continuous_carrier(
            net=object(),
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=-1,
        )
        assert result is None
        bg_class.assert_not_called()


def test_carrier_positive_rate_starts_bg_and_returns_instance() -> None:
    """Positive rate constructs BackgroundTraffic with that rate and starts it."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        instance = MagicMock()
        bg_class.return_value = instance

        result = make_continuous_carrier(
            net="dummy-net",
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=25,
        )

        bg_class.assert_called_once()
        kwargs = bg_class.call_args.kwargs
        assert kwargs["rate_mbps"] == 25
        assert kwargs["sender_host"] == "h1"
        assert kwargs["receiver_host"] == "h2"
        assert kwargs["sender_ip"] == "10.0.0.1"
        assert kwargs["receiver_ip"] == "10.0.0.2"
        instance.start.assert_called_once_with()
        # Critically: stop is NOT called here — caller is responsible.
        instance.stop.assert_not_called()
        assert result is instance


def test_carrier_caller_owns_stop_lifecycle() -> None:
    """The helper does not call ``stop()``; the workload's ``finally`` does."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        instance = MagicMock()
        bg_class.return_value = instance

        carrier = make_continuous_carrier(
            net="dummy",
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=1,
        )
        # Simulate the workload finishing and stopping the carrier itself.
        carrier.stop()
        instance.start.assert_called_once()
        instance.stop.assert_called_once_with()
