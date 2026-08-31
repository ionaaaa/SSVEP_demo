import os

import pytest

from ssvep_demo.control import ControlCommand
from ssvep_demo.virtual_car import VirtualCarController


def test_virtual_car_turns_moves_and_stops() -> None:
    car = VirtualCarController(heading_degrees=90.0, turn_rate_degrees_s=30.0, speed_units_s=0.2)
    car.execute(ControlCommand.LEFT)
    car.update(1.0)
    assert (car.heading_degrees, car.is_moving, car.active_command) == (120.0, True, ControlCommand.LEFT)
    car.execute(ControlCommand.RIGHT)
    car.update(1.0)
    assert car.heading_degrees == 90.0
    car.execute(ControlCommand.FORWARD)
    car.update(1.0)
    assert car.y == pytest.approx(0.2)
    assert car.active_command is ControlCommand.FORWARD
    car.stop()
    car.stop()
    assert not car.is_moving and car.active_command is None


def test_virtual_car_rejects_unknown_and_stops_at_bounds() -> None:
    car = VirtualCarController(x=0.84, heading_degrees=0.0, speed_units_s=0.2)
    car.execute(ControlCommand.FORWARD)
    car.update(1.0)
    assert car.x == pytest.approx(0.85)
    assert not car.is_moving and car.active_command is None
    with pytest.raises(ValueError, match="UNKNOWN"):
        car.execute(ControlCommand.UNKNOWN)


def test_virtual_car_backward_is_continuous() -> None:
    car = VirtualCarController(heading_degrees=0.0, speed_units_s=0.25)
    car.execute(ControlCommand.BACKWARD)
    car.update(0.4)
    assert car.x == pytest.approx(-0.1)


def test_virtual_car_module_import_does_not_create_a_psychopy_window() -> None:
    import ssvep_demo.virtual_car as module

    assert "psychopy" not in module.__dict__


@pytest.mark.skipif(not os.environ.get("RUN_PSYCHOPY_SMOKE"), reason="requires a local display and manual PsychoPy smoke run")
def test_virtual_car_gui_smoke_is_manual_only() -> None:
    pytest.skip("Run scripts/run_virtual_car_demo.py manually with a display")
