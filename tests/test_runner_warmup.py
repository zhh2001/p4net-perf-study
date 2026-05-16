"""Unit tests for the unified warmup helper in :mod:`runner.runner`.

These tests don't bring up a real network; they patch
``BackgroundTraffic`` and ``time.sleep`` and assert the call sequence
``start(rate=warmup_rate_mbps) → sleep(warmup_seconds) → stop()`` is
what the helper produces. The full end-to-end behaviour is exercised
by the Phase E pilot run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runner.runner import do_unified_warmup


def test_warmup_zero_seconds_is_noop() -> None:
    """``warmup_seconds <= 0`` must not instantiate ``BackgroundTraffic``."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        do_unified_warmup(
            net=object(),
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            warmup_seconds=0.0,
            warmup_rate_mbps=1,
        )
        bg_class.assert_not_called()


def test_warmup_zero_rate_is_noop() -> None:
    """``warmup_rate_mbps <= 0`` must not instantiate ``BackgroundTraffic``."""
    with patch("runner.runner.BackgroundTraffic") as bg_class:
        do_unified_warmup(
            net=object(),
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            warmup_seconds=10.0,
            warmup_rate_mbps=0,
        )
        bg_class.assert_not_called()


def test_warmup_call_sequence() -> None:
    """Verify start(rate=...) → sleep(seconds) → stop() in that order."""
    with (
        patch("runner.runner.BackgroundTraffic") as bg_class,
        patch("runner.runner.time.sleep") as sleep_mock,
    ):
        instance = MagicMock()
        bg_class.return_value = instance

        do_unified_warmup(
            net="dummy-net",
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            warmup_seconds=2.5,
            warmup_rate_mbps=7,
        )

        # BackgroundTraffic instantiated exactly once with the warmup rate
        bg_class.assert_called_once()
        kwargs = bg_class.call_args.kwargs
        assert kwargs["rate_mbps"] == 7
        assert kwargs["sender_host"] == "h1"
        assert kwargs["receiver_host"] == "h2"
        assert kwargs["sender_ip"] == "10.0.0.1"
        assert kwargs["receiver_ip"] == "10.0.0.2"

        # Order: start → sleep → stop
        instance.start.assert_called_once_with()
        sleep_mock.assert_called_once_with(2.5)
        instance.stop.assert_called_once_with()

        # Lifecycle ordering: start before sleep before stop
        assert instance.start.call_count == 1
        assert instance.stop.call_count == 1


def test_warmup_stops_bg_even_when_sleep_raises() -> None:
    """If the warmup is interrupted mid-sleep, BackgroundTraffic.stop()
    must still be called so the iperf3 child processes don't leak."""
    with (
        patch("runner.runner.BackgroundTraffic") as bg_class,
        patch("runner.runner.time.sleep", side_effect=KeyboardInterrupt),
    ):
        instance = MagicMock()
        bg_class.return_value = instance

        with pytest.raises(KeyboardInterrupt):
            do_unified_warmup(
                net="dummy-net",
                sender_host="h1",
                receiver_host="h2",
                sender_ip="10.0.0.1",
                receiver_ip="10.0.0.2",
                warmup_seconds=10.0,
                warmup_rate_mbps=1,
            )
        instance.start.assert_called_once()
        instance.stop.assert_called_once()
