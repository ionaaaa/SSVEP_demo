"""Non-blocking omniBCI input and sequence-aligned SSVEP trial collection.

The WebSocket and optional HTTP trigger workers never call PsychoPy.  The
PsychoPy thread calls :meth:`poll`, which is the only place that mutates the
ring buffer or trial state.  Signal samples remain raw microvolts throughout
this module: parsing only validates and reorders channels.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import math
import queue
import socket
import threading
import time
from typing import Any, Callable, Literal, Mapping, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from .config import LiveConfig
from .protocol import CANONICAL_CHANNELS, EEGWindow


UINT32_MAX = (1 << 32) - 1
UINT32_MODULUS = 1 << 32
TRIGGER_CODE = 1
# Confirmed from omniBCI-R: the response echoes the sequence of the sample to
# which the queued event is written, so this sequence is the inclusive start.
TRIGGER_SEQUENCE_SEMANTICS = "event_aligned_sample_inclusive"
AlignmentMode = Literal["buffer_sequence", "trigger"]


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
        if not isinstance(self.received_monotonic_s, (int, float)) or not math.isfinite(float(self.received_monotonic_s)):
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
    trial_id: int | None
    attempt_id: int | None
    start_sequence: int | None
    end_sequence: int | None
    window_ready_monotonic_s: float | None
    trial_failure_reason: str | None
    recent_anomaly: str | None
    worker_alive: bool
    alignment_mode_requested: str | None
    alignment_mode_used: str | None
    stimulus_flip_monotonic_s: float | None
    sequence_at_flip: int | None
    trigger_request_monotonic_s: float | None
    trigger_response_monotonic_s: float | None
    trigger_http_latency_ms: float | None
    trigger_sequence: int | None
    trigger_sequence_semantics: str | None

    @property
    def received_samples(self) -> int:
        return self.collected_samples

    @property
    def window_start_sequence(self) -> int | None:
        return self.start_sequence

    @property
    def window_end_sequence(self) -> int | None:
        return self.end_sequence


class LiveEEGSource(Protocol):
    def start(self) -> None: ...
    def poll(self) -> None: ...
    def mark_stimulus_flip(
        self, *, trial_id: int, attempt_id: int, stimulus_start_monotonic_s: float
    ) -> None: ...
    def begin_trial(
        self,
        *,
        trial_id: int,
        attempt_id: int,
        stimulus_start_monotonic_s: float,
        start_sequence: int,
        alignment_mode: AlignmentMode,
    ) -> None: ...
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
    """Parse an actual ``/v1/stream`` batch without transforming its signal."""
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LiveProtocolError("protocol_error", "WebSocket binary payload is not UTF-8 JSON") from exc
    if not isinstance(message, str):
        raise LiveProtocolError("protocol_error", "WebSocket message must be JSON text")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
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
            f"channel_names must match configured physical source order {live_config.source_channel_names}, got {source_names}",
        )
    if set(live_config.channel_map) != set(source_names):
        raise LiveProtocolError("metadata_mismatch", "channel_map keys do not match stream channel_names")
    mapped = tuple(live_config.channel_map[name] for name in source_names)
    if len(set(mapped)) != 8 or set(mapped) != set(CANONICAL_CHANNELS):
        raise LiveProtocolError("metadata_mismatch", "channel_map must cover each canonical SSVEP channel exactly once")
    source_for_target = {target: source_names.index(source) for source, target in live_config.channel_map.items()}
    reorder = tuple(source_for_target[target] for target in CANONICAL_CHANNELS)
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
        values = _required(raw_frame, "values_uv")
        if not isinstance(values, list) or len(values) != 8:
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must contain exactly 8 values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must be numeric")
        raw_uv = np.asarray(values, dtype=np.float32)
        if not np.isfinite(raw_uv).all():
            raise LiveProtocolError("protocol_error", f"frames[{index}].values_uv must be finite")
        # Deliberately only reorder: no filtering, centering, reference, scale,
        # normalization, or conversion from microvolts to volts.
        frames.append(LiveSampleFrame(sequence, valid, raw_uv[list(reorder)], clock()))
    return ParsedLiveMessage(metadata, tuple(frames))


@dataclass(frozen=True)
class RingExtraction:
    state: Literal["pending", "complete", "failed"]
    frames: tuple[LiveSampleFrame, ...] = ()
    reason: str | None = None


class LiveRingBuffer:
    """Fixed-capacity sample buffer supporting exact u32 sequence extraction."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._frames: deque[LiveSampleFrame] = deque(maxlen=capacity)
        self.latest_sequence: int | None = None
        self.latest_valid_sequence: int | None = None

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
        if frame.valid:
            self.latest_valid_sequence = frame.sequence
        return anomaly

    def extract(self, start_sequence: int, count: int) -> RingExtraction:
        if not 0 <= start_sequence <= UINT32_MAX or count <= 0:
            raise ValueError("invalid sequence range")
        frames = tuple(self._frames)
        start_index = next((index for index, frame in enumerate(frames) if frame.sequence == start_sequence), None)
        if start_index is None:
            if self.latest_sequence is not None:
                distance = (self.latest_sequence - start_sequence) % UINT32_MODULUS
                if distance < UINT32_MODULUS // 2:
                    return RingExtraction("failed", reason="sequence_not_in_buffer")
            return RingExtraction("pending")

        selected: list[LiveSampleFrame] = []
        expected = start_sequence
        for frame in frames[start_index:]:
            if frame.sequence != expected:
                return RingExtraction("failed", tuple(selected), "sequence_gap")
            if not frame.valid:
                return RingExtraction("failed", tuple(selected), "invalid_frame")
            selected.append(frame)
            if len(selected) == count:
                return RingExtraction("complete", tuple(selected))
            expected = (expected + 1) % UINT32_MODULUS
        return RingExtraction("pending", tuple(selected))

    def clear(self, *, reset_sequence: bool = False) -> None:
        self._frames.clear()
        if reset_sequence:
            self.latest_sequence = None
            self.latest_valid_sequence = None

    def __len__(self) -> int:
        return len(self._frames)


@dataclass(frozen=True)
class _TriggerRequest:
    trial_id: int
    attempt_id: int
    queued_monotonic_s: float


@dataclass(frozen=True)
class _TriggerResult:
    request: _TriggerRequest
    started_monotonic_s: float
    completed_monotonic_s: float
    payload: Mapping[str, Any] | None = None
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _WorkerEvent:
    kind: str
    monotonic_s: float
    frame: LiveSampleFrame | None = None
    metadata: LiveStreamMetadata | None = None
    trigger: _TriggerResult | None = None
    reason: str | None = None
    detail: str | None = None


def _post_trigger_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> Mapping[str, Any]:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib_error.HTTPError as exc:
        raise LiveProtocolError("trigger_http_error", f"Trigger HTTP {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LiveProtocolError("trigger_timeout", "Trigger request timed out") from exc
    except urllib_error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise LiveProtocolError("trigger_timeout", "Trigger request timed out") from exc
        raise LiveProtocolError("trigger_http_error", f"Trigger request failed: {exc.reason}") from exc
    try:
        result = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiveProtocolError("trigger_invalid_json", "Trigger response is not valid JSON") from exc
    if not isinstance(result, Mapping):
        raise LiveProtocolError("trigger_invalid_json", "Trigger response root must be an object")
    return result


class OmniBCIWebSocketSource:
    """Threaded omniBCI receiver and main-thread sequence-range collector."""

    def __init__(
        self,
        server_url: str,
        live_config: LiveConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        websocket_connect: Callable[..., Any] | None = None,
        trigger_http_post: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(server_url, str) or not server_url.startswith(("ws://", "wss://")):
            raise ValueError("server_url must be a ws:// or wss:// URL")
        if live_config.alignment_mode not in {"buffer_sequence", "trigger"}:
            raise ValueError("alignment_mode must be buffer_sequence or trigger")
        if live_config.alignment_mode == "trigger" and not live_config.trigger_url:
            raise ValueError("trigger mode requires trigger_url")
        self.server_url = server_url
        self.config = live_config
        self.clock = clock
        self._websocket_connect = websocket_connect
        self._trigger_http_post = trigger_http_post or _post_trigger_json
        self._queue: queue.Queue[_WorkerEvent] = queue.Queue(maxsize=live_config.queue_capacity)
        self._trigger_queue: queue.Queue[_TriggerRequest | None] = queue.Queue(maxsize=16)
        self._overflow_lock = threading.Lock()
        self._pending_overflows = 0
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._active_websocket: Any | None = None
        self._thread: threading.Thread | None = None
        self._trigger_thread: threading.Thread | None = None
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
        self._reset_trial_state()

    def _reset_trial_state(self) -> None:
        self._trial_active = False
        self._trial_id: int | None = None
        self._attempt_id: int | None = None
        self._trial_start_s: float | None = None
        self._trial_received_samples = 0
        self._trial_start_sequence: int | None = None
        self._trial_end_sequence: int | None = None
        self._trial_failure_reason: str | None = None
        self._completed_window: EEGWindow | None = None
        self._window_ready_s: float | None = None
        self._alignment_requested: str | None = None
        self._alignment_used: str | None = None
        self._stimulus_flip_s: float | None = None
        self._sequence_at_flip: int | None = None
        self._trigger_request_s: float | None = None
        self._trigger_response_s: float | None = None
        self._trigger_latency_ms: float | None = None
        self._trigger_sequence: int | None = None
        self._trigger_semantics: str | None = None

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("live source is closed")
        if self._started:
            return
        self._started = True
        self._connection_state = "connecting"
        self._thread = threading.Thread(target=self._thread_main, name="ssvep-omnibci-websocket", daemon=True)
        self._thread.start()
        if self.config.alignment_mode == "trigger":
            self._trigger_thread = threading.Thread(target=self._trigger_worker, name="ssvep-omnibci-trigger", daemon=True)
            self._trigger_thread.start()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._worker())
        except BaseException as exc:
            self._put_event(_WorkerEvent("worker_failed", self.clock(), reason="stream_disconnected", detail=f"{type(exc).__name__}: {exc}"))

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
                    self._put_event(_WorkerEvent("connected" if connection_number == 1 else "reconnected", self.clock()))
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
                            self._put_event(_WorkerEvent("protocol_error", self.clock(), reason=exc.code, detail=str(exc)))
                            continue
                        if parsed.metadata != last_metadata:
                            self._put_event(_WorkerEvent(
                                "metadata" if last_metadata is None else "metadata_changed",
                                self.clock(), metadata=parsed.metadata,
                                reason=None if last_metadata is None else "metadata_mismatch",
                            ))
                            last_metadata = parsed.metadata
                        for frame in parsed.frames:
                            self._put_event(_WorkerEvent("frame", frame.received_monotonic_s, frame=frame))
            except asyncio.CancelledError:
                break
            except BaseException as exc:
                if self._stop.is_set():
                    break
                self._put_event(_WorkerEvent("disconnected", self.clock(), reason="stream_disconnected", detail=f"{type(exc).__name__}: {exc}"))
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

    def _trigger_worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            started = self.clock()
            self._put_event(_WorkerEvent("trigger_started", started, trigger=_TriggerResult(item, started, started)))
            try:
                # omniBCI-R rejects unknown request fields, so identity is
                # carried in its existing label rather than a new protocol.
                payload = self._trigger_http_post(
                    str(self.config.trigger_url),
                    {"code": TRIGGER_CODE, "label": f"ssvep trial_id={item.trial_id} attempt_id={item.attempt_id}"},
                    self.config.trigger_timeout_s,
                )
            except LiveProtocolError as exc:
                completed = self.clock()
                result = _TriggerResult(item, started, completed, reason=exc.code, detail=str(exc))
                self._put_event(_WorkerEvent("trigger_failed", completed, trigger=result, reason=exc.code, detail=str(exc)))
            except BaseException as exc:
                completed = self.clock()
                result = _TriggerResult(item, started, completed, reason="trigger_http_error", detail=f"{type(exc).__name__}: {exc}")
                self._put_event(_WorkerEvent("trigger_failed", completed, trigger=result, reason=result.reason, detail=result.detail))
            else:
                completed = self.clock()
                result = _TriggerResult(item, started, completed, payload=payload)
                self._put_event(_WorkerEvent("trigger_response", completed, trigger=result))

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
        for _ in range(self.config.queue_capacity):
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self._try_collect_from_ring()

    def _handle_event(self, event: _WorkerEvent) -> None:
        if event.kind in {"connecting", "connected", "reconnected", "disconnected", "closed", "worker_failed"}:
            self._connection_state = {"worker_failed": "disconnected", "closed": "closed", "reconnected": "connected"}.get(event.kind, event.kind)
            if event.kind == "reconnected":
                self._reconnect_count += 1
                self.ring.clear(reset_sequence=True)
                self._record_anomaly("stream_reconnected", event.detail)
                self._invalidate_trial("stream_reconnected", event.monotonic_s)
            elif event.kind in {"disconnected", "worker_failed"}:
                self.ring.clear(reset_sequence=True)
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
                self.ring.clear(reset_sequence=True)
            self._metadata = event.metadata
            detail = event.detail or json.dumps({
                "sample_rate_hz": event.metadata.sample_rate_hz,
                "source_channel_names": event.metadata.source_channel_names,
                "mapped_channel_names": event.metadata.mapped_channel_names,
            })
            self._events.append(LiveSourceEvent(event.kind, event.monotonic_s, event.reason, detail))
            return
        if event.kind == "frame":
            assert event.frame is not None
            self._handle_frame(event.frame)
            # Finalize as soon as the exact inclusive end sequence arrives.
            # Later frames in the same WebSocket batch are outside this trial
            # and must neither be collected nor retroactively invalidate it.
            self._try_collect_from_ring()
            return
        if event.kind in {"trigger_started", "trigger_response", "trigger_failed"}:
            assert event.trigger is not None
            self._handle_trigger_event(event.kind, event.trigger)

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

    def _handle_trigger_event(self, kind: str, result: _TriggerResult) -> None:
        identity = (result.request.trial_id, result.request.attempt_id)
        if identity != (self._trial_id, self._attempt_id) or not self._trial_active:
            self._events.append(LiveSourceEvent(
                "trigger_response_discarded", result.completed_monotonic_s,
                reason="stale_trigger_response", detail=f"trial_id={identity[0]} attempt_id={identity[1]}",
            ))
            return
        if kind == "trigger_started":
            self._events.append(LiveSourceEvent(
                "trigger_request_started", result.started_monotonic_s,
                detail=json.dumps({"trial_id": identity[0], "attempt_id": identity[1]}),
            ))
            return
        self._trigger_response_s = result.completed_monotonic_s
        self._trigger_latency_ms = (result.completed_monotonic_s - result.started_monotonic_s) * 1000.0
        if kind == "trigger_failed":
            reason = result.reason or "trigger_http_error"
            self._events.append(LiveSourceEvent(
                "trigger_alignment_failed", result.completed_monotonic_s, reason,
                json.dumps({"trial_id": identity[0], "attempt_id": identity[1], "error": result.detail}),
            ))
            self._invalidate_trial(reason, result.completed_monotonic_s)
            return
        self._events.append(LiveSourceEvent("trigger_response_received", result.completed_monotonic_s, detail=json.dumps(result.payload)))
        payload = result.payload
        sequence = payload.get("sequence") if isinstance(payload, Mapping) else None
        if not isinstance(payload, Mapping) or payload.get("accepted") is not True or isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence <= UINT32_MAX:
            self._events.append(LiveSourceEvent("trigger_alignment_failed", result.completed_monotonic_s, "trigger_invalid_response"))
            self._invalidate_trial("trigger_invalid_response", result.completed_monotonic_s)
            return
        self._trigger_sequence = sequence
        self._trigger_semantics = TRIGGER_SEQUENCE_SEMANTICS
        self._set_alignment(sequence, "trigger")
        self._events.append(LiveSourceEvent(
            "trigger_alignment_ready", result.completed_monotonic_s,
            detail=json.dumps({
                "trial_id": identity[0], "attempt_id": identity[1],
                "sequence": sequence, "semantics": TRIGGER_SEQUENCE_SEMANTICS,
            }),
        ))

    def _prepare_trial(self, trial_id: int, attempt_id: int, stimulus_start_monotonic_s: float) -> bool:
        if not self._started or self._closed:
            raise RuntimeError("live source must be started before beginning a trial")
        if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
            raise ValueError("trial_id must be a non-negative integer")
        if isinstance(attempt_id, bool) or not isinstance(attempt_id, int) or attempt_id < 0:
            raise ValueError("attempt_id must be a non-negative integer")
        if not isinstance(stimulus_start_monotonic_s, (int, float)) or not math.isfinite(float(stimulus_start_monotonic_s)):
            raise ValueError("stimulus_start_monotonic_s must be finite")
        self._reset_trial_state()
        self._trial_active = True
        self._trial_id = trial_id
        self._attempt_id = attempt_id
        self._trial_start_s = float(stimulus_start_monotonic_s)
        self._alignment_requested = self.config.alignment_mode
        self._stimulus_flip_s = float(stimulus_start_monotonic_s)
        if self._connection_state != "connected" or self._metadata is None:
            self._invalidate_trial("stream_disconnected")
            return False
        return True

    def mark_stimulus_flip(self, *, trial_id: int, attempt_id: int, stimulus_start_monotonic_s: float) -> None:
        """Snapshot or enqueue alignment work; performs no I/O and no EEG copy."""
        if not self._prepare_trial(trial_id, attempt_id, stimulus_start_monotonic_s):
            return
        self._sequence_at_flip = self.ring.latest_valid_sequence
        self._events.append(LiveSourceEvent(
            "stimulus_flip_sequence_snapshotted", self.clock(),
            detail=json.dumps({
                "trial_id": trial_id, "attempt_id": attempt_id,
                "stimulus_flip_monotonic_s": stimulus_start_monotonic_s,
                "sequence_at_flip": self._sequence_at_flip,
            }),
        ))
        if self.config.alignment_mode == "buffer_sequence":
            if self._sequence_at_flip is None:
                self._invalidate_trial("no_sequence_at_flip")
                return
            self._set_alignment((self._sequence_at_flip + 1) % UINT32_MODULUS, "buffer_sequence")
            return
        queued_s = self.clock()
        request = _TriggerRequest(trial_id, attempt_id, queued_s)
        self._trigger_request_s = queued_s
        try:
            self._trigger_queue.put_nowait(request)
        except queue.Full:
            self._events.append(LiveSourceEvent("trigger_alignment_failed", queued_s, "trigger_queue_overflow"))
            self._invalidate_trial("trigger_queue_overflow")
            return
        self._events.append(LiveSourceEvent(
            "trigger_request_queued", queued_s,
            detail=json.dumps({"trial_id": trial_id, "attempt_id": attempt_id}),
        ))

    def begin_trial(
        self,
        *,
        trial_id: int,
        attempt_id: int,
        stimulus_start_monotonic_s: float,
        start_sequence: int,
        alignment_mode: AlignmentMode,
    ) -> None:
        """Start collection from an explicit inclusive sequence (test/API hook)."""
        if alignment_mode not in {"buffer_sequence", "trigger"}:
            raise ValueError("invalid alignment_mode")
        if isinstance(start_sequence, bool) or not isinstance(start_sequence, int) or not 0 <= start_sequence <= UINT32_MAX:
            raise ValueError("start_sequence must be a u32")
        if self._prepare_trial(trial_id, attempt_id, stimulus_start_monotonic_s):
            self._set_alignment(start_sequence, alignment_mode)

    def _set_alignment(self, start_sequence: int, mode: AlignmentMode) -> None:
        self._trial_start_sequence = start_sequence
        self._trial_end_sequence = (start_sequence + self.config.trial_samples - 1) % UINT32_MODULUS
        self._alignment_used = mode
        self._events.append(LiveSourceEvent(
            "trial_collection_started", self.clock(),
            detail=json.dumps({"start_sequence": self._trial_start_sequence, "end_sequence": self._trial_end_sequence, "alignment_mode": mode}),
        ))
        self._try_collect_from_ring()

    def _try_collect_from_ring(self) -> None:
        if not self._trial_active or self._trial_failure_reason is not None or self._trial_start_sequence is None:
            return
        extraction = self.ring.extract(self._trial_start_sequence, self.config.trial_samples)
        self._trial_received_samples = len(extraction.frames)
        if extraction.state == "failed":
            self._invalidate_trial(extraction.reason or "sequence_gap")
            return
        if extraction.state != "complete":
            return
        # This is the only sample-major -> channel-major conversion. Values
        # remain the raw float32 microvolts parsed from values_uv.
        data = np.stack([frame.values_uv for frame in extraction.frames], axis=1)
        first = extraction.frames[0]
        self._completed_window = EEGWindow(
            data=data,
            sample_rate_hz=self.config.expected_sample_rate_hz,
            channel_names=list(CANONICAL_CHANNELS),
            start_time_s=first.received_monotonic_s,
            # EEGWindow is half-open and the exact sequence count/sample rate
            # defines its duration. Per-frame values are WebSocket receive
            # times, not hardware timestamps, and must not introduce network
            # jitter into the decoder's four-second contract.
            end_time_s=first.received_monotonic_s + self.config.trial_samples / self.config.expected_sample_rate_hz,
        )
        self._trial_received_samples = self.config.trial_samples
        self._window_ready_s = self.clock()
        self._trial_active = False
        self._events.append(LiveSourceEvent(
            "live_window_ready", self._window_ready_s,
            detail=json.dumps({
                "trial_id": self._trial_id, "attempt_id": self._attempt_id,
                "start_sequence": self._trial_start_sequence,
                "end_sequence": self._trial_end_sequence,
                "alignment_mode": self._alignment_used,
            }),
        ))

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
        if self._trial_active:
            self._trial_failure_reason = reason
            self._trial_active = False
            self._completed_window = None
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
            trial_id=self._trial_id,
            attempt_id=self._attempt_id,
            start_sequence=self._trial_start_sequence,
            end_sequence=self._trial_end_sequence,
            window_ready_monotonic_s=self._window_ready_s,
            trial_failure_reason=self._trial_failure_reason,
            recent_anomaly=self._recent_anomaly,
            worker_alive=(self._thread.is_alive() if self._thread else False) or (self._trigger_thread.is_alive() if self._trigger_thread else False),
            alignment_mode_requested=self._alignment_requested,
            alignment_mode_used=self._alignment_used,
            stimulus_flip_monotonic_s=self._stimulus_flip_s,
            sequence_at_flip=self._sequence_at_flip,
            trigger_request_monotonic_s=self._trigger_request_s,
            trigger_response_monotonic_s=self._trigger_response_s,
            trigger_http_latency_ms=self._trigger_latency_ms,
            trigger_sequence=self._trigger_sequence,
            trigger_sequence_semantics=self._trigger_semantics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._trigger_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._loop is not None and self._async_stop is not None:
            try:
                def stop_worker() -> None:
                    self._async_stop.set()
                    transport = getattr(self._active_websocket, "transport", None)
                    if transport is not None:
                        transport.abort()
                    if self._worker_task is not None:
                        self._worker_task.cancel()
                self._loop.call_soon_threadsafe(stop_worker)
            except RuntimeError:
                pass
        for thread in (self._thread, self._trigger_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        self._connection_state = "closed"
        self._invalidate_trial("source_closed")
