from __future__ import annotations

from ._native_counter import NativeCounter


class UniversalFlopCounter:
    """
    Python interface for the native FLOP/BOP counter.

    ATen interception and operation counting are performed
    by the native RecordFunction-based core.
    """

    def __init__(self) -> None:
        self._native = NativeCounter()

    def start(self) -> UniversalFlopCounter:
        """Start native ATen operator interception."""
        self._native.start()
        return self

    def stop(self) -> UniversalFlopCounter:
        """Stop native ATen operator interception."""
        self._native.stop()
        return self

    def reset(self) -> UniversalFlopCounter:
        """Reset all native counters and diagnostic data."""
        self._native.reset()
        return self

    def pause(self) -> UniversalFlopCounter:
        """Temporarily suspend counting without removing the callback."""
        self._native.pause()
        return self

    def resume(self) -> UniversalFlopCounter:
        """Resume counting after a pause."""
        self._native.resume()
        return self

    def add_flops(self, value: int) -> UniversalFlopCounter:
        """Add a manual FLOP contribution."""
        numeric_value = int(value)

        if numeric_value < 0:
            raise ValueError("FLOP contribution cannot be negative.")

        self._native.add_flops(numeric_value)
        return self

    def add_bops(self, value: int) -> UniversalFlopCounter:
        """Add a manual BOP contribution."""
        numeric_value = int(value)

        if numeric_value < 0:
            raise ValueError("BOP contribution cannot be negative.")

        self._native.add_bops(numeric_value)
        return self

    @property
    def flops(self) -> int:
        return int(self._native.total_flops)

    @property
    def bops(self) -> int:
        return int(self._native.total_bops)

    @property
    def active(self) -> bool:
        return bool(self._native.active)

    @property
    def paused(self) -> bool:
        return bool(self._native.paused)

    @paused.setter
    def paused(self, value: bool) -> None:
        if bool(value):
            self.pause()
        else:
            self.resume()

    @property
    def event_count(self) -> int:
        return int(self._native.event_count)

    @property
    def operator_flops(self) -> dict[str, int]:
        return {
            str(name): int(value)
            for name, value in self._native.operator_flops.items()
        }

    @property
    def operator_bops(self) -> dict[str, int]:
        return {
            str(name): int(value)
            for name, value in self._native.operator_bops.items()
        }

    @property
    def operator_errors(self) -> dict[str, str]:
        return {
            str(name): str(value)
            for name, value in self._native.operator_errors.items()
        }

    def __enter__(self) -> UniversalFlopCounter:
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        self.stop()
        return False