"""Non-blocking omniBCI WebSocket input for four-second SSVEP trials.

The worker thread owns asyncio and the WebSocket.  It only parses transport
messages and enqueues typed events.  ``poll`` is deliberately non-blocking and
is the sole place that mutates the ring buffer and trial collector.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
import queue
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .config import LiveConfig
from .protocol import CANONICAL_CHANNELS, EEGWindow


UINT32_MAX = (1 << 32) - 1
UINT32_MODULUS = 1 << 32


class LiveProtocolError(ValueError):
    """A malformed or incompatible omniBCI stream message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LiveSampleFrame:
    sequence: int
    valid: bool
    values_uv: np.ndarray
    received_monotonic_s: float

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or not 0 <= self.sequence <= UINT32_MAX:
            raise ValueError("sequence must be an unsigned 32-bit integer")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")
        values = np.asarray(self.values_uv, dtype=np.float32)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("values_uv must contain exactly 8 finite values")
        values = values.copy()
        values.flags.writeable = False
        object.__setattr__(self, "values_uv", values)
        if not isinstance(self.received_monotonic_s, (int, float)) or not math.isfinite(
            float(self.received_monotonic_s)
        ):
            raise ValueError("received_monotonic_s must be finite")


@dataclass(frozen=True)
class LiveStreamMetadata:
    sample_rate_hz: float
    source_channel_names: tuple[str, ...]
    mapped_channel_names: tuple[str, ...]


@dataclass(frozen=True)
class ParsedLiveMessage:
    metadata: LiveStreamMetadata
    frames: tuple[LiveSampleFrame, ...]


@dataclass(frozen=True)
class LiveSourceEvent:
    kind: str
    monotonic_s: float
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class LiveSourceStatus:
    connection_state: str
    metadata: LiveStreamMetadata | None
    latest_sequence: int | None
    last_valid_frame_monotonic_s: float | None
    stream_gap_count: int
    invalid_frame_count: int
    reconnect_count: int
    queue_overflow_count: int
    protocol_error_count: int
    collected_samples: int
    expected_samples: int
    trial_active: bool
    start_sequence: int | None
    end_sequence: int | None
    window_ready_monotonic_s: float | None
    trial_failure_reason: str | None
    recent_anomaly: str | None
    worker_alive: bool

    @property
    def received_samples(self) -> int:
        """Alias used by the persisted live-trial contract."""
        return self.collected_samples


class LiveEEGSource(Protocol):
    def start(self) -> None: ...
    def poll(self) -> None: ...
    def begin_trial(self, stimulus_start_monotonic_s: float) -> None: ...
    def try_take_completed_window(self) -> EEGWindow | None: ...
    def abort_trial(self, reason: str) -> None: ...
    def status(self) -> LiveSourceStatus: ...
    def take_events(self) -> list[LiveSourceEvent]: ...
    def close(self) -> None: ...


def _required(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise LiveProtocolError("protocol_error", f"omniBCI message is missing {name}")
    return payload[name]


def parse_omnibci_message(
    message: str | bytes,
    live_config: LiveConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> ParsedLiveMessage:
    """Parse one actual omniBCI ``/v1/stream`` batch into mapped sample frames.

    The current Rust protocol has no separate unit field.  Its signal field is
    explicitly named ``values_uv``; no other value field or implicit unit is
    accepted here.
    """
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LiveProtocolError("protocol_error", "WebSocket binary payload is not UTF-8 JSON") from exc
    if not isinstance(message, str):
        raise LiveProtocolError("protocol_error", "WebSocket message must be JSON text")
    try:
        payload = json.loads(message)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveProtocolError("protocol_error", f"invalid omniBCI JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LiveProtocolError("protocol_error", "omniBCI message root must be an object")
    if _required(payload, "type") != "eeg":
        raise LiveProtocolError("protocol_error", "omniBCI message type must be 'eeg'")
    if "unit" in payload and payload["unit"] != "uV":
        raise LiveProtocolError("metadata_mismatch", "unit must be exactly 'uV'")

    rate = _required(payload, "sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
        raise LiveProtocolError("metadata_mismatch", "sample_rate_hz must be a finite number")
    rate = float(rate)
    if rate != live_config.expected_sample_rate_hz or rate != 250.0:
        raise LiveProtocolError("metadata_mismatch", f"sample_rate_hz must be exactly 250, got {rate:g}")

    raw_names = _required(payload, "channel_names")
    if not isinstance(raw_names, list) or not all(isinstance(name, str) and name for name in raw_names):
        raise LiveProtocolError("metadata_mismatch", "channel_names must be a list of non-empty strings")
    source_names = tuple(raw_names)
    if len(source_names) != 8:
        raise LiveProtocolError("metadata_mismatch", f"channel_names must contain exactly 8 channels, got {len(source_names)}")
    if source_names != live_config.source_channel_names:
        raise LiveProtocolError(
            "metadata_mismatch",
            "channel_names do not match configured physical source order: "
            f"expected {live_config.source_channel_names}, got {source_names}",
        )
    if set(live_config.channel_map) != set(source_names):
        raise LiveProtocolError("metadata_mismatch", "channel_map keys do not match stream channel_names")
    mapped_values = tuple(live_config.channel_map[name] for name in source_names)
    if len(set(mapped_values)) != 8 or set(mapped_values) != set(CANONICAL_CHANNELS):
        raise LiveProtocolError("metadata_mismatch", "channel_map must cover each canonical SSVEP channel exactly once")
    source_index_for_target = {
        target: source_names.index(source_name)
        for source_name, target in live_config.channel_map.items()
    }
    reorder = tuple(source_index_for_target[target] for target in CANONICAL_CHANNELS)
    metadata = LiveStreamMetadata(rate, source_names, CANONICAL_CHANNELS)

    raw_frames = _required(payload, "frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise LiveProtocolError("protocol_error", "frames must be a non-empty list")
    frames: list[LiveSampleFrame] = []
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise LiveProtocolError("protocol_error", f"frames[{index}] must be an object")
        sequence = _required(raw_frame, "sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence <= UINT32_MAX:
            raise LiveProtocolError("protocol_error", f"frames[{index}].sequence must be a u32")
        valid = _required(raw_frame, "valid")
        if not isinstance(valid, bool):
            raise LiveProtocolError("protocol_error", f"frames[{index}].valid must be a boolean")
        # The actual protocol's field name carries the only unit declaration.
        values = _required(raw_frame, "values_uv")
        if not isinstance(values, list) or len(values) != 8:
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must contain exactly 8 values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must be numeric")
        values_array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values_array).all():
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must be finite")
        frames.append(
            LiveSampleFrame(
                sequence=sequence,
                valid=valid,
                values_uv=values_array[list(reorder)],
                received_monotonic_s=clock(),
            )
        )
    return ParsedLiveMessage(metadata, tuple(frames))


class LiveRingBuffer:
    """A fixed-capacity, sample-major diagnostics buffer with u32 continuity."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._frames: deque[LiveSampleFrame] = deque(maxlen=capacity)
        self.latest_sequence: int | None = None

    def append(self, frame: LiveSampleFrame) -> str | None:
        anomaly: str | None = None
        if self.latest_sequence is not None:
            delta = (frame.sequence - self.latest_sequence) % UINT32_MODULUS
            if delta == 0:
                anomaly = "duplicate_sequence"
            elif delta == 1:
                anomaly = None
            elif delta < UINT32_MODULUS // 2:
                anomaly = "sequence_gap"
            else:
                anomaly = "sequence_backwards"
        self._frames.append(frame)
        self.latest_sequence = frame.sequence
        return anomaly

    def clear(self, *, reset_sequence: bool = False) -> None:
        self._frames.clear()
        if reset_sequence:
            self.latest_sequence = None

    def __len__(self) -> int:
        return len(self._frames)


@dataclass(frozen=True)
class _WorkerEvent:
    kind: str
    monotonic_s: float
    frame: LiveSampleFrame | None = None
    metadata: LiveStreamMetadata | None = None
    reason: str | None = None
    detail: str | None = None


class OmniBCIWebSocketSource:
    """Threaded omniBCI receiver and main-thread SSVEP trial collector."""

    def __init__(
        self,
        server_url: str,
        live_config: LiveConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        websocket_connect: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(server_url, str) or not server_url.startswith(("ws://", "wss://")):
            raise ValueError("server_url must be a ws:// or wss:// URL")
        self.server_url = server_url
        self.config = live_config
        self.clock = clock
        self._websocket_connect = websocket_connect
        self._queue: queue.Queue[_WorkerEvent] = queue.Queue(maxsize=live_config.queue_capacity)
        self._overflow_lock = threading.Lock()
        self._pending_overflows = 0
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._active_websocket: Any | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._started = False
        self._events: deque[LiveSourceEvent] = deque()
        capacity = int(round(live_config.ring_buffer_seconds * live_config.expected_sample_rate_hz))
        self.ring = LiveRingBuffer(capacity)
        self._connection_state = "not_started"
        self._metadata: LiveStreamMetadata | None = None
        self._last_valid_frame_s: float | None = None
        self._gap_count = 0
        self._invalid_count = 0
        self._reconnect_count = 0
        self._overflow_count = 0
        self._protocol_error_count = 0
        self._recent_anomaly: str | None = None
        self._trial_active = False
        self._trial_start_s: float | None = None
        self._trial_frames: list[LiveSampleFrame] = []
        self._trial_received_samples = 0
        self._trial_start_sequence: int | None = None
        self._trial_end_sequence: int | None = None
        self._trial_failure_reason: str | None = None
        self._completed_window: EEGWindow | None = None
        self._window_ready_s: float | None = None

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("live source is closed")
        if self._started:
            return
        self._started = True
        self._connection_state = "connecting"
        self._thread = threading.Thread(target=self._thread_main, name="ssvep-omnibci-websocket", daemon=True)
        self._thread.start()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._worker())
        except BaseException as exc:
            self._put_event(
                _WorkerEvent("worker_failed", self.clock(), reason="stream_disconnected", detail=f"{type(exc).__name__}: {exc}")
            )

    async def _worker(self) -> None:
        if self._websocket_connect is None:
            from websockets.asyncio.client import connect

            websocket_connect = connect
        else:
            websocket_connect = self._websocket_connect
        self._loop = asyncio.get_running_loop()
        self._worker_task = asyncio.current_task()
        self._async_stop = asyncio.Event()
        delay = self.config.reconnect_initial_delay_s
        connection_number = 0
        while not self._stop.is_set():
            self._put_event(_WorkerEvent("connecting", self.clock()))
            try:
                async with websocket_connect(self.server_url, open_timeout=1.0, close_timeout=1.0) as websocket:
                    self._active_websocket = websocket
                    connection_number += 1
                    kind = "connected" if connection_number == 1 else "reconnected"
                    self._put_event(_WorkerEvent(kind, self.clock()))
                    delay = self.config.reconnect_initial_delay_s
                    last_metadata: LiveStreamMetadata | None = None
                    while not self._stop.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.25)
                        except asyncio.TimeoutError:
                            continue
                        try:
                            parsed = parse_omnibci_message(message, self.config, clock=self.clock)
                        except LiveProtocolError as exc:
                            self._put_event(
                                _WorkerEvent("protocol_error", self.clock(), reason=exc.code, detail=str(exc))
                            )
                            continue
                        if parsed.metadata != last_metadata:
                            self._put_event(
                                _WorkerEvent(
                                    "metadata" if last_metadata is None else "metadata_changed",
                                    self.clock(),
                                    metadata=parsed.metadata,
                                    reason=None if last_metadata is None else "metadata_mismatch",
                                )
                            )
                            last_metadata = parsed.metadata
                        for frame in parsed.frames:
                            self._put_event(_WorkerEvent("frame", frame.received_monotonic_s, frame=frame))
            except asyncio.CancelledError:
                break
            except BaseException as exc:
                if self._stop.is_set():
                    break
                self._put_event(
                    _WorkerEvent("disconnected", self.clock(), reason="stream_disconnected", detail=f"{type(exc).__name__}: {exc}")
                )
            else:
                if not self._stop.is_set():
                    self._put_event(_WorkerEvent("disconnected", self.clock(), reason="stream_disconnected"))
            if self._stop.is_set():
                break
            assert self._async_stop is not None
            try:
                await asyncio.wait_for(self._async_stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2.0, self.config.reconnect_max_delay_s)
            self._active_websocket = None
        self._put_event(_WorkerEvent("closed", self.clock()))

    def _put_event(self, event: _WorkerEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._overflow_lock:
                self._pending_overflows += 1

    def _consume_pending_overflows(self) -> int:
        with self._overflow_lock:
            count = self._pending_overflows
            self._pending_overflows = 0
        return count

    def poll(self) -> None:
        overflows = self._consume_pending_overflows()
        if overflows:
            self._overflow_count += overflows
            self._record_anomaly("queue_overflow", f"{overflows} worker events could not be queued")
            self._invalidate_trial("queue_overflow")
        # Bounded by queue_capacity, performs no I/O and never waits.
        for _ in range(self.config.queue_capacity):
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

    def _handle_event(self, event: _WorkerEvent) -> None:
        if event.kind in {"connecting", "connected", "reconnected", "disconnected", "closed", "worker_failed"}:
            self._connection_state = {
                "worker_failed": "disconnected",
                "closed": "closed",
                "reconnected": "connected",
            }.get(event.kind, event.kind)
            if event.kind == "reconnected":
                self._reconnect_count += 1
                self.ring.clear()
                self._record_anomaly("stream_reconnected", event.detail)
                self._invalidate_trial("stream_reconnected", event.monotonic_s)
            elif event.kind in {"disconnected", "worker_failed"}:
                self.ring.clear()
                self._record_anomaly("stream_disconnected", event.detail)
                self._invalidate_trial("stream_disconnected", event.monotonic_s)
            self._events.append(LiveSourceEvent(event.kind, event.monotonic_s, event.reason, event.detail))
            return
        if event.kind == "protocol_error":
            self._protocol_error_count += 1
            reason = event.reason or "protocol_error"
            self._record_anomaly(reason, event.detail)
            self._invalidate_trial(reason, event.monotonic_s)
            self._events.append(LiveSourceEvent("protocol_error", event.monotonic_s, reason, event.detail))
            return
        if event.kind in {"metadata", "metadata_changed"}:
            assert event.metadata is not None
            if self._metadata is not None and event.metadata != self._metadata:
                self._record_anomaly("metadata_mismatch", "stream metadata changed")
                self._invalidate_trial("metadata_mismatch", event.monotonic_s)
                self.ring.clear()
            self._metadata = event.metadata
            detail = event.detail or json.dumps(
                {
                    "sample_rate_hz": event.metadata.sample_rate_hz,
                    "source_channel_names": event.metadata.source_channel_names,
                    "mapped_channel_names": event.metadata.mapped_channel_names,
                }
            )
            self._events.append(LiveSourceEvent(event.kind, event.monotonic_s, event.reason, detail))
            return
        if event.kind == "frame":
            assert event.frame is not None
            self._handle_frame(event.frame)

    def _handle_frame(self, frame: LiveSampleFrame) -> None:
        anomaly = self.ring.append(frame)
        if anomaly is not None:
            self._gap_count += 1
            self._record_anomaly(anomaly, f"sequence={frame.sequence}")
            self._invalidate_trial(anomaly, frame.received_monotonic_s)
        if not frame.valid:
            self._invalid_count += 1
            self._record_anomaly("invalid_frame", f"sequence={frame.sequence}")
            self._invalidate_trial("invalid_frame", frame.received_monotonic_s)
            return
        self._last_valid_frame_s = frame.received_monotonic_s
        if (
            not self._trial_active
            or self._trial_failure_reason is not None
            or self._trial_start_s is None
            or frame.received_monotonic_s < self._trial_start_s
        ):
            return
        if self._trial_start_sequence is None:
            self._trial_start_sequence = frame.sequence
        else:
            expected = (self._trial_frames[-1].sequence + 1) % UINT32_MODULUS
            if frame.sequence != expected:
                # Normally caught by the global ring first; retain a local guard.
                self._gap_count += 1
                self._record_anomaly("sequence_gap", f"trial expected={expected}, got={frame.sequence}")
                self._invalidate_trial("sequence_gap", frame.received_monotonic_s)
                return
        self._trial_frames.append(frame)
        self._trial_received_samples += 1
        if len(self._trial_frames) == self.config.trial_samples:
            self._trial_end_sequence = frame.sequence
            data = np.stack([item.values_uv for item in self._trial_frames], axis=1)
            start_s = self._trial_frames[0].received_monotonic_s
            # EEGWindow uses a half-open time range; receive timestamps remain
            # diagnostics and are not claimed as hardware acquisition times.
            end_s = start_s + self.config.trial_samples / self.config.expected_sample_rate_hz
            self._completed_window = EEGWindow(
                data=data,
                sample_rate_hz=self.config.expected_sample_rate_hz,
                channel_names=list(CANONICAL_CHANNELS),
                start_time_s=start_s,
                end_time_s=end_s,
            )
            self._window_ready_s = self.clock()
            self._trial_active = False
            self._trial_frames = []
            self._events.append(LiveSourceEvent("window_ready", self._window_ready_s))

    def begin_trial(self, stimulus_start_monotonic_s: float) -> None:
        if not self._started or self._closed:
            raise RuntimeError("live source must be started before begin_trial")
        if not isinstance(stimulus_start_monotonic_s, (int, float)) or not math.isfinite(
            float(stimulus_start_monotonic_s)
        ):
            raise ValueError("stimulus_start_monotonic_s must be finite")
        self._trial_active = True
        self._trial_start_s = float(stimulus_start_monotonic_s)
        self._trial_frames = []
        self._trial_received_samples = 0
        self._trial_start_sequence = None
        self._trial_end_sequence = None
        self._trial_failure_reason = None
        self._completed_window = None
        self._window_ready_s = None
        if self._connection_state != "connected" or self._metadata is None:
            self._invalidate_trial("stream_disconnected")
        self._events.append(LiveSourceEvent("trial_collection_started", self.clock()))

    def try_take_completed_window(self) -> EEGWindow | None:
        window = self._completed_window
        self._completed_window = None
        return window

    def abort_trial(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("abort reason must be a non-empty string")
        self._invalidate_trial(reason)

    def _invalidate_trial(self, reason: str, event_time_s: float | None = None) -> None:
        if self._trial_start_s is not None and event_time_s is not None and event_time_s < self._trial_start_s:
            return
        if self._trial_active or self._completed_window is not None:
            self._trial_failure_reason = reason
            self._trial_active = False
            self._completed_window = None
            self._trial_frames = []
            self._events.append(LiveSourceEvent("trial_invalidated", self.clock(), reason=reason))

    def _record_anomaly(self, reason: str, detail: str | None) -> None:
        self._recent_anomaly = reason
        self._events.append(LiveSourceEvent("stream_anomaly", self.clock(), reason, detail))

    def take_events(self) -> list[LiveSourceEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def status(self) -> LiveSourceStatus:
        return LiveSourceStatus(
            connection_state=self._connection_state,
            metadata=self._metadata,
            latest_sequence=self.ring.latest_sequence,
            last_valid_frame_monotonic_s=self._last_valid_frame_s,
            stream_gap_count=self._gap_count,
            invalid_frame_count=self._invalid_count,
            reconnect_count=self._reconnect_count,
            queue_overflow_count=self._overflow_count,
            protocol_error_count=self._protocol_error_count,
            collected_samples=self._trial_received_samples,
            expected_samples=self.config.trial_samples,
            trial_active=self._trial_active,
            start_sequence=self._trial_start_sequence,
            end_sequence=self._trial_end_sequence,
            window_ready_monotonic_s=self._window_ready_s,
            trial_failure_reason=self._trial_failure_reason,
            recent_anomaly=self._recent_anomaly,
            worker_alive=self._thread.is_alive() if self._thread is not None else False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._loop is not None and self._async_stop is not None:
            try:
                def stop_worker() -> None:
                    self._async_stop.set()
                    # A network peer that doesn't participate in the close
                    # handshake must not hold the PsychoPy shutdown path.
                    transport = getattr(self._active_websocket, "transport", None)
                    if transport is not None:
                        transport.abort()
                    if self._worker_task is not None:
                        self._worker_task.cancel()

                self._loop.call_soon_threadsafe(stop_worker)
            except RuntimeError:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        self._connection_state = "closed"
        self._invalidate_trial("source_closed")
