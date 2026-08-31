"""A minimal, display-independent virtual car and optional PsychoPy view."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

from .control import ControlCommand, DispatchDecision


@dataclass
class VirtualCarController:
    """Small deterministic controller used to exercise the safety boundary.

    Coordinates use a conventional Cartesian frame: zero degrees points right
    and positive angles turn toward the top of the display.  The class draws
    nothing, so it can be fully tested without a display server.
    """

    x: float = 0.0
    y: float = 0.0
    heading_degrees: float = 90.0
    is_moving: bool = False
    active_command: ControlCommand | None = None
    bounds: tuple[float, float, float, float] = (-0.85, 0.85, -0.70, 0.70)
    turn_degrees: float = 20.0
    forward_step: float = 0.12

    def __post_init__(self) -> None:
        if len(self.bounds) != 4 or self.bounds[0] >= self.bounds[1] or self.bounds[2] >= self.bounds[3]:
            raise ValueError("bounds must be (min_x, max_x, min_y, max_y) with increasing limits")
        if not math.isfinite(self.turn_degrees) or self.turn_degrees <= 0:
            raise ValueError("turn_degrees must be a positive finite number")
        if not math.isfinite(self.forward_step) or self.forward_step <= 0:
            raise ValueError("forward_step must be a positive finite number")
        self.heading_degrees = float(self.heading_degrees) % 360.0
        self.x, self.y = self._clamped_position(self.x, self.y)

    def execute(self, command: ControlCommand) -> None:
        """Apply one simple movement update; UNKNOWN is never accepted."""
        if not isinstance(command, ControlCommand):
            raise ValueError("VirtualCarController.execute requires a ControlCommand")
        if command is ControlCommand.UNKNOWN:
            raise ValueError("UNKNOWN must not be sent to VirtualCarController.execute")
        if command is ControlCommand.STOP:
            self.stop()
            return
        if command is ControlCommand.LEFT:
            self.heading_degrees = (self.heading_degrees + self.turn_degrees) % 360.0
        elif command is ControlCommand.RIGHT:
            self.heading_degrees = (self.heading_degrees - self.turn_degrees) % 360.0
        elif command is ControlCommand.FORWARD:
            radians = math.radians(self.heading_degrees)
            requested_x = self.x + self.forward_step * math.cos(radians)
            requested_y = self.y + self.forward_step * math.sin(radians)
            self.x, self.y = self._clamped_position(requested_x, requested_y)
            if (self.x, self.y) != (requested_x, requested_y):
                self.stop()
                return
        self.is_moving = True
        self.active_command = command

    def stop(self) -> None:
        """Idempotently stop the virtual car."""
        self.is_moving = False
        self.active_command = None

    def state(self) -> dict[str, Any]:
        """Return JSON-friendly controller state for a decision log."""
        return {
            "x": self.x,
            "y": self.y,
            "heading_degrees": self.heading_degrees,
            "is_moving": self.is_moving,
            "active_command": self.active_command.value if self.active_command else None,
        }

    def _clamped_position(self, x: float, y: float) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = self.bounds
        return (min(max(float(x), min_x), max_x), min(max(float(y), min_y), max_y))


class ControlDecisionLogger:
    """Small, flushed JSONL control-decision log following the session style."""

    def __init__(self, output_root: Path, config_path: Path, metadata: dict[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = output_root / f"session_{stamp}"
        suffix = 1
        while self.directory.exists():
            self.directory = output_root / f"session_{stamp}_{suffix:02d}"
            suffix += 1
        self.directory.mkdir(parents=True)
        shutil.copy2(config_path, self.directory / "config.yaml")
        self.metadata = metadata
        (self.directory / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._file = (self.directory / "decisions.jsonl").open("a", encoding="utf-8")

    def write(
        self,
        decision: DispatchDecision,
        controller: VirtualCarController,
        *,
        prediction_timestamp_s: float | None,
        confidence: float | None,
    ) -> None:
        record = {
            "received_timestamp_s": decision.timestamp_s,
            "prediction_timestamp_s": prediction_timestamp_s,
            "received_command": decision.received_command.value,
            "confidence": confidence,
            "effective_command": decision.effective_command.value,
            "confirmation_count": decision.confirmation_count,
            "action": decision.action,
            "reason": decision.reason,
            "controller_state": controller.state(),
            **controller.state(),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class PsychoPyVirtualCarView:
    """The deliberately simple drawing half of the virtual-car demonstration."""

    def __init__(self, win: Any, visual: Any, confirmations_required: int) -> None:
        self._confirmations_required = confirmations_required
        self._car = visual.Rect(win, width=0.12, height=0.08, fillColor="royalblue", lineColor="white")
        self._heading = visual.Line(win, start=(0, 0), end=(0, 0.12), lineColor="yellow", lineWidth=4)
        self._position = visual.TextStim(win, pos=(0, 0.85), height=0.045, color="white")
        self._motion = visual.TextStim(win, pos=(0, 0.75), height=0.04, color="white")
        self._confidence = visual.TextStim(win, pos=(0, 0.66), height=0.035, color="white")
        self._decision = visual.TextStim(win, pos=(0, -0.82), height=0.032, color="white", wrapWidth=1.8)
        self._help = visual.TextStim(
            win,
            text="1 LEFT   2 RIGHT   3 FORWARD   4 STOP   U low confidence   Esc exit",
            pos=(0, -0.92),
            height=0.026,
            color="white",
            wrapWidth=1.8,
        )

    def draw(
        self,
        car: VirtualCarController,
        decision: DispatchDecision | None,
        confidence: float | None,
    ) -> None:
        self._car.pos = (car.x, car.y)
        radians = math.radians(car.heading_degrees)
        self._heading.start = (car.x, car.y)
        self._heading.end = (car.x + 0.13 * math.cos(radians), car.y + 0.13 * math.sin(radians))
        self._position.text = f"Position: ({car.x:.2f}, {car.y:.2f})   heading: {car.heading_degrees:.0f}°"
        active = car.active_command.value if car.active_command else "STOPPED"
        self._motion.text = f"Motion: {active}"
        self._confidence.text = "Latest confidence: --" if confidence is None else f"Latest confidence: {confidence:.2f}"
        if decision is None:
            self._decision.text = "Decision: waiting for keyboard simulation"
        elif decision.action == "pending":
            self._decision.text = (
                f"Decision: {decision.effective_command.value} confirmation "
                f"{decision.confirmation_count}/{self._confirmations_required}"
            )
        else:
            self._decision.text = (
                f"Decision: {decision.action} / {decision.reason} / "
                f"confirmation {decision.confirmation_count}/{self._confirmations_required}"
            )
        for stimulus in (
            self._car,
            self._heading,
            self._position,
            self._motion,
            self._confidence,
            self._decision,
            self._help,
        ):
            stimulus.draw()
