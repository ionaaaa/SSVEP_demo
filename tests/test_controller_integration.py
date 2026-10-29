from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time
from types import ModuleType

import pytest

from scripts.run_ssvep_demo import (
    apply_controller_cli_overrides,
    arguments,
    main,
)
from ssvep_demo.config import BluetoothControllerConfig, ControllerConfig, load_config
from ssvep_demo.control import ControlCommand, SafeCommandDispatcher
from ssvep_demo.controller_factory import (
    ControllerFaultError,
    ControllerRuntime,
    ControllerStartupError,
    cleanup_controller,
    create_controller,
)
from ssvep_demo.protocol import DecodeResult
from ssvep_demo.session_logging import resolved_config_dict
from ssvep_demo.live_demo import SSVEPLiveDemoRunner
from ssvep_demo.synthetic_demo import SSVEPSyntheticDemoRunner
from ssvep_demo.virtual_car import VirtualCarController


CONFIG_PATH = Path(__file__).parents[1] / "config" / "ssvep_demo.yaml"


class FakeBluetoothController:
    def __init__(
        self,
        *,
        device_name: str | None = None,
        device_id: str | None = None,
        io_timeout_s: float = 3.0,
        calls: list[object] | None = None,
        close_error: BaseException | None = None,
        fault_on_execute: bool = False,
    ) -> None:
        self.device_name = device_name
        self.device_id = device_id
        self.io_timeout_s = io_timeout_s
        self.state = "DISCONNECTED"
        self.last_error: str | None = None
        self.calls = calls if calls is not None else []
        self.close_error = close_error
        self.fault_on_execute = fault_on_execute

    def connect(self, timeout_s: float) -> None:
        self.calls.append(("connect", timeout_s))
        self.state = "READY"

    def execute(self, command: ControlCommand) -> None:
        self.calls.append(("execute", command))
        if self.fault_on_execute:
            self.state = "FAULT"
            self.last_error = "simulated write failure"

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self, timeout_s: float) -> None:
        self.calls.append(("close", timeout_s))
        self.state = "CLOSED"
        if self.close_error is not None:
            raise self.close_error


class FixedDecoder:
    def decode(self, _window) -> DecodeResult:
        return DecodeResult(
            8.0,
            ControlCommand.LEFT,
            {8.0: 1.0, 10.0: 0.0, 12.0: 0.0, 15.0: 0.0},
            1.0,
            time.monotonic(),
        )


class ExplodingDecoder:
    def decode(self, _window) -> DecodeResult:
        raise RuntimeError("decoder exploded")


def _bluetooth_config(**updates) -> ControllerConfig:
    bluetooth = BluetoothControllerConfig(
        device_name="paired-car",
        device_id="exact-id",
        connect_timeout_s=7.0,
        io_timeout_s=2.0,
        close_timeout_s=1.0,
    )
    bluetooth = replace(bluetooth, **updates)
    return ControllerConfig(type="bluetooth", bluetooth=bluetooth)


def _runner(
    tmp_path: Path,
    controller: FakeBluetoothController,
    *,
    decoder=FixedDecoder(),
) -> SSVEPSyntheticDemoRunner:
    config = replace(load_config(CONFIG_PATH), controller=_bluetooth_config())
    controller.state = "READY"
    runtime = ControllerRuntime(controller, config.controller)
    runner = SSVEPSyntheticDemoRunner(
        config,
        CONFIG_PATH,
        decoder_type="cca",
        output_dir=tmp_path,
        max_trials=1,
        confirmations_required=1,
        controller_runtime_factory=lambda: runtime,
    )
    runner._make_decoder = lambda: decoder  # type: ignore[method-assign]
    return runner


def _install_fake_psychopy(monkeypatch: pytest.MonkeyPatch, window_factory) -> None:
    psychopy = ModuleType("psychopy")
    psychopy.__version__ = "test"  # type: ignore[attr-defined]
    visual = ModuleType("psychopy.visual")
    visual.Window = window_factory  # type: ignore[attr-defined]
    event = ModuleType("psychopy.event")
    psychopy.visual = visual  # type: ignore[attr-defined]
    psychopy.event = event  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psychopy", psychopy)
    monkeypatch.setitem(sys.modules, "psychopy.visual", visual)
    monkeypatch.setitem(sys.modules, "psychopy.event", event)


def test_cli_defaults_remain_none_and_yaml_virtual_wins() -> None:
    args = arguments([])
    config = apply_controller_cli_overrides(load_config(CONFIG_PATH), args)

    assert args.controller is None
    assert args.car_device_name is None
    assert args.car_device_id is None
    assert config.controller.type == "virtual"


def test_cli_controller_overrides_yaml_but_unspecified_car_fields_do_not() -> None:
    yaml_config = replace(load_config(CONFIG_PATH), controller=_bluetooth_config())
    args = arguments(["--controller", "bluetooth", "--car-io-timeout-s", "4.5"])

    config = apply_controller_cli_overrides(yaml_config, args)

    assert config.controller.type == "bluetooth"
    assert config.controller.bluetooth.device_name == "paired-car"
    assert config.controller.bluetooth.device_id == "exact-id"
    assert config.controller.bluetooth.connect_timeout_s == 7.0
    assert config.controller.bluetooth.io_timeout_s == 4.5
    assert config.controller.bluetooth.close_timeout_s == 1.0


def test_cli_parses_all_bluetooth_values_and_resolved_config_records_them() -> None:
    args = arguments(
        [
            "--controller", "bluetooth",
            "--car-device-name", "renamed-car",
            "--car-device-id", "id-2",
            "--car-connect-timeout-s", "9",
            "--car-io-timeout-s", "2.5",
            "--car-close-timeout-s", "1.5",
        ]
    )
    config = apply_controller_cli_overrides(load_config(CONFIG_PATH), args)

    saved = resolved_config_dict(config)
    assert saved["controller"] == {
        "type": "bluetooth",
        "bluetooth": {
            "device_name": "renamed-car",
            "device_id": "id-2",
            "connect_timeout_s": 9.0,
            "io_timeout_s": 2.5,
            "close_timeout_s": 1.5,
        },
    }


@pytest.mark.parametrize("flag", ["--car-device-name", "--car-device-id"])
def test_virtual_controller_rejects_explicit_bluetooth_options(flag: str) -> None:
    args = arguments(["--controller", "virtual", flag, "unexpected"])

    with pytest.raises(ValueError, match="cannot be used"):
        apply_controller_cli_overrides(load_config(CONFIG_PATH), args)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--car-connect-timeout-s", "0"),
        ("--car-io-timeout-s", "-1"),
        ("--car-close-timeout-s", "nan"),
    ],
)
def test_cli_rejects_non_positive_or_non_finite_timeouts(flag: str, value: str) -> None:
    args = arguments(["--controller", "bluetooth", flag, value])

    with pytest.raises(ValueError, match="positive finite"):
        apply_controller_cli_overrides(load_config(CONFIG_PATH), args)


def test_replay_bluetooth_is_rejected_before_replay_file_or_hardware() -> None:
    with pytest.raises(SystemExit, match="disabled in replay mode for safety"):
        main(["--mode", "replay", "--controller", "bluetooth"])


def test_virtual_factory_does_not_import_bluetooth_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "ssvep_demo.bluetooth_car_controller", raising=False)

    runtime = create_controller(ControllerConfig(), mode="synthetic")

    assert isinstance(runtime.controller, VirtualCarController)
    assert "ssvep_demo.bluetooth_car_controller" not in sys.modules


def test_bluetooth_factory_receives_final_config_connects_and_uses_device_id_priority() -> None:
    created: list[FakeBluetoothController] = []

    def factory(**kwargs):
        controller = FakeBluetoothController(**kwargs)
        created.append(controller)
        return controller

    runtime = create_controller(_bluetooth_config(), mode="live", bluetooth_factory=factory)

    assert runtime.controller is created[0]
    assert created[0].device_name == "paired-car"
    assert created[0].device_id == "exact-id"
    assert created[0].io_timeout_s == 2.0
    assert created[0].calls == [("connect", 7.0)]
    assert runtime.state == "READY"


def test_bluetooth_factory_failure_has_no_virtual_fallback() -> None:
    class FailingController(FakeBluetoothController):
        def connect(self, timeout_s: float) -> None:
            raise TimeoutError("not found")

    with pytest.raises(ControllerStartupError, match="exact-id.*TimeoutError: not found"):
        create_controller(_bluetooth_config(), mode="synthetic", bluetooth_factory=FailingController)


def test_real_bluetooth_factory_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import ssvep_demo.bluetooth_car_controller as bluetooth_module

    monkeypatch.setattr(bluetooth_module.sys, "platform", "linux")

    with pytest.raises(ControllerStartupError, match="requires Windows 10/11"):
        create_controller(_bluetooth_config(), mode="synthetic")


def test_controller_factory_runs_before_psychopy_window_and_window_failure_closes_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    controller = FakeBluetoothController(calls=calls)
    config = replace(load_config(CONFIG_PATH), controller=_bluetooth_config())
    runtime = ControllerRuntime(controller, config.controller)

    def make_runtime() -> ControllerRuntime:
        calls.append("controller_factory")
        controller.state = "READY"
        return runtime

    def make_window(**_kwargs):
        calls.append("window")
        raise RuntimeError("window failed")

    _install_fake_psychopy(monkeypatch, make_window)
    runner = SSVEPSyntheticDemoRunner(
        config,
        CONFIG_PATH,
        output_dir=tmp_path,
        controller_runtime_factory=make_runtime,
    )

    with pytest.raises(RuntimeError, match="window failed"):
        runner.run()

    assert calls[0:2] == ["controller_factory", "window"]
    assert calls[-1] == ("close", 1.0)


def test_connect_failure_does_not_create_window_or_start_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window_created = False

    def make_window(**_kwargs):
        nonlocal window_created
        window_created = True
        raise AssertionError("window must not be created")

    _install_fake_psychopy(monkeypatch, make_window)
    config = replace(load_config(CONFIG_PATH), controller=_bluetooth_config())
    runner = SSVEPSyntheticDemoRunner(
        config,
        CONFIG_PATH,
        output_dir=tmp_path,
        controller_runtime_factory=lambda: (_ for _ in ()).throw(ControllerStartupError("connect failed")),
    )

    with pytest.raises(ControllerStartupError, match="connect failed"):
        runner.run()
    assert not window_created


def test_live_controller_factory_also_runs_before_psychopy_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    controller = FakeBluetoothController(calls=calls)
    controller.state = "READY"
    config = replace(load_config(CONFIG_PATH), controller=_bluetooth_config())
    runtime = ControllerRuntime(controller, config.controller)

    def make_runtime() -> ControllerRuntime:
        calls.append("controller_factory")
        return runtime

    def make_window(**_kwargs):
        calls.append("window")
        raise RuntimeError("window failed")

    _install_fake_psychopy(monkeypatch, make_window)
    runner = SSVEPLiveDemoRunner(
        config,
        CONFIG_PATH,
        server_url="ws://127.0.0.1:8766/v1/stream",
        output_dir=tmp_path,
        live_source=object(),  # Window creation fails before source methods are used.
        controller_runtime_factory=make_runtime,
    )

    with pytest.raises(RuntimeError, match="window failed"):
        runner.run()

    assert calls[0:2] == ["controller_factory", "window"]
    assert calls[-1] == ("close", 1.0)


def test_offline_decoder_uses_dispatcher_with_the_same_controller_and_closes_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ssvep_demo.synthetic_demo as synthetic_module

    calls: list[object] = []
    controller = FakeBluetoothController(calls=calls)
    captured: list[object] = []
    real_dispatcher = SafeCommandDispatcher

    def dispatcher_factory(selected, *args, **kwargs):
        captured.append(selected)
        return real_dispatcher(selected, *args, **kwargs)

    monkeypatch.setattr(synthetic_module, "SafeCommandDispatcher", dispatcher_factory)
    _runner(tmp_path, controller).run_offline()

    assert captured == [controller]
    assert any(call[0] == "execute" for call in calls if isinstance(call, tuple))
    assert calls[-2:] == ["stop", ("close", 1.0)]


def test_controller_close_failure_does_not_mask_decoder_failure(tmp_path: Path) -> None:
    controller = FakeBluetoothController(close_error=OSError("close exploded"))

    with pytest.raises(RuntimeError, match="decoder exploded"):
        _runner(tmp_path, controller, decoder=ExplodingDecoder()).run_offline()


def test_offline_setup_failure_still_closes_bluetooth_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ssvep_demo.synthetic_demo as synthetic_module

    controller = FakeBluetoothController()

    def fail_logger(*_args, **_kwargs):
        raise RuntimeError("logger setup failed")

    monkeypatch.setattr(synthetic_module, "UnifiedSessionLogger", fail_logger)

    with pytest.raises(RuntimeError, match="logger setup failed"):
        _runner(tmp_path, controller).run_offline()

    assert controller.calls[-1] == ("close", 1.0)


def test_bluetooth_fault_aborts_session_and_rejects_further_trials(tmp_path: Path) -> None:
    controller = FakeBluetoothController(fault_on_execute=True)

    with pytest.raises(ControllerFaultError, match="simulated write failure"):
        _runner(tmp_path, controller).run_offline()

    execute_calls = [call for call in controller.calls if isinstance(call, tuple) and call[0] == "execute"]
    assert len(execute_calls) == 1


def test_cleanup_closes_dispatcher_before_controller_and_returns_close_error() -> None:
    calls: list[object] = []
    controller = FakeBluetoothController(calls=calls, close_error=OSError("close failed"))
    controller.state = "READY"
    runtime = ControllerRuntime(controller, _bluetooth_config())
    dispatcher = SafeCommandDispatcher(controller, load_config(CONFIG_PATH).control, load_config(CONFIG_PATH).commands)

    error = cleanup_controller(dispatcher, runtime)

    assert isinstance(error, OSError)
    assert calls[-2:] == ["stop", ("close", 1.0)]
