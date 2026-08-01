"""Independent communication watchdog for the ROS 2 control boundary."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class WatchdogDecision:
    """Health decision consumed by the safety fallback path."""

    healthy: bool
    fallback_required: bool
    reasons: tuple[str, ...]
    max_age_s: float
    ages_s: dict[str, float]


class CommunicationWatchdog:
    """Track required ROS channels without depending on policy/RL code.

    ``mark`` is called by message callbacks. ``evaluate`` is called from a
    timer and returns a fail-closed decision when any required channel is
    missing or older than its configured timeout.
    """

    def __init__(
        self,
        timeouts_s: dict[str, float],
        *,
        required_channels: Iterable[str] = ("pose", "wrench"),
        fallback_on_timeout: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not timeouts_s:
            raise ValueError("at least one watchdog timeout is required")
        if any(value <= 0.0 for value in timeouts_s.values()):
            raise ValueError("watchdog timeouts must be positive")
        self._timeouts_s = {str(name): float(value) for name, value in timeouts_s.items()}
        self._required = tuple(str(name) for name in required_channels)
        if not self._required or any(name not in self._timeouts_s for name in self._required):
            raise ValueError("required watchdog channels must have configured timeouts")
        self._fallback_on_timeout = bool(fallback_on_timeout)
        self._clock = clock
        self._last_seen: dict[str, float] = {}

    @property
    def required_channels(self) -> tuple[str, ...]:
        return self._required

    def mark(self, channel: str, timestamp_s: float | None = None) -> None:
        """Record a fresh message for a named channel."""

        if channel not in self._timeouts_s:
            raise KeyError(f"unknown watchdog channel: {channel}")
        stamp = float(self._clock() if timestamp_s is None else timestamp_s)
        if stamp != stamp or stamp in (float("inf"), float("-inf")):
            raise ValueError("watchdog timestamp must be finite")
        self._last_seen[channel] = stamp

    def reset(self) -> None:
        self._last_seen.clear()

    def evaluate(self, now_s: float | None = None) -> WatchdogDecision:
        """Return current health and a fail-safe fallback requirement."""

        now = float(self._clock() if now_s is None else now_s)
        ages: dict[str, float] = {}
        reasons: list[str] = []
        for channel in self._required:
            last = self._last_seen.get(channel)
            age = float("inf") if last is None else max(0.0, now - last)
            ages[channel] = age
            if last is None:
                reasons.append(f"{channel}_missing")
            elif age > self._timeouts_s[channel]:
                reasons.append(f"{channel}_timeout")
        healthy = not reasons
        finite_ages = [age for age in ages.values() if age != float("inf")]
        max_age = max(ages.values()) if ages else 0.0
        if not finite_ages and ages:
            max_age = float("inf")
        return WatchdogDecision(
            healthy=healthy,
            fallback_required=(not healthy and self._fallback_on_timeout),
            reasons=tuple(reasons),
            max_age_s=max_age,
            ages_s=ages,
        )
