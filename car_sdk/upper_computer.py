"""Small Windows GUI for direct Bluetooth control of the STM32 car."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import tkinter as tk
from tkinter import ttk

from bluetooth_car import BluetoothCar, BluetoothCarDevice, discover_devices


class AsyncWorker:
    """Keep all WinRT Bluetooth objects on one dedicated asyncio thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def submit(self, coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


class CarController:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("STM32 小车蓝牙控制")
        self.root.resizable(False, False)
        self.worker = AsyncWorker()
        self.car = BluetoothCar()
        self.devices: list[BluetoothCarDevice] = []
        self.connected = False
        self.busy = False
        self.closing = False
        self.pressed: set[str] = set()
        self.events: queue.Queue = queue.Queue()
        self.direction_buttons: list[ttk.Button] = []

        frame = ttk.Frame(root, padding=16)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="蓝牙设备").grid(row=0, column=0, sticky="w")
        self.device_box = ttk.Combobox(frame, width=35, state="readonly")
        self.device_box.grid(row=1, column=0, columnspan=2, pady=(4, 8), sticky="ew")
        self.refresh_button = ttk.Button(frame, text="刷新设备", command=self.refresh)
        self.refresh_button.grid(row=1, column=2, padx=(8, 0))
        self.connect_button = ttk.Button(frame, text="连接", command=self.connect)
        self.connect_button.grid(row=2, column=0, sticky="ew")
        self.disconnect_button = ttk.Button(
            frame, text="断开", command=self.disconnect, state="disabled"
        )
        self.disconnect_button.grid(row=2, column=1, sticky="ew", padx=8)
        self.status = tk.StringVar(value="正在检查已配对设备的实时蓝牙串口服务…")
        ttk.Label(frame, textvariable=self.status, wraplength=400).grid(
            row=3, column=0, columnspan=3, pady=(10, 12), sticky="w"
        )

        pad = ttk.Frame(frame)
        pad.grid(row=4, column=0, columnspan=3)
        self._make_direction_button(pad, "↑ 前进", "forward", 0, 1)
        self._make_direction_button(pad, "← 左转", "left", 1, 0)
        self.stop_button = ttk.Button(pad, text="■ 停止", command=self.send_stop, width=12)
        self.stop_button.grid(row=1, column=1, padx=4, pady=4, ipady=8)
        self.stop_button.state(["disabled"])
        self._make_direction_button(pad, "右转 →", "right", 1, 2)
        self._make_direction_button(pad, "↓ 后退", "backward", 2, 1)

        ttk.Label(
            frame,
            text="按住方向按钮行驶，松开即发送停止。首次测试请架空车轮。",
        ).grid(row=5, column=0, columnspan=3, pady=(12, 0), sticky="w")

        for key_name, command in (
            ("Up", "forward"), ("Down", "backward"),
            ("Left", "left"), ("Right", "right"),
        ):
            self.root.bind(f"<{key_name}>", lambda event, cmd=command: self.press(cmd))
            self.root.bind(f"<KeyRelease-{key_name}>",
                           lambda event, cmd=command: self.release(cmd))

        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.root.bind("<FocusOut>", lambda _event: self.send_stop())
        self.root.after(50, self.process_events)
        self.refresh()

    def _make_direction_button(self, parent, text, command, row, column) -> None:
        button = ttk.Button(parent, text=text, width=12)
        button.grid(row=row, column=column, padx=4, pady=4, ipady=8)
        button.bind("<ButtonPress-1>", lambda event: self.press(command))
        button.bind("<ButtonRelease-1>", lambda event: self.release(command))
        button.state(["disabled"])
        self.direction_buttons.append(button)

    def _submit(self, coroutine, success, failure=None) -> None:
        future = self.worker.submit(coroutine)

        def finished(done):
            try:
                result = done.result()
                self.events.put((success, result))
            except Exception as exc:
                self.events.put((failure or self.show_error, exc))

        future.add_done_callback(finished)
        return future

    def process_events(self) -> None:
        while True:
            try:
                callback, value = self.events.get_nowait()
            except queue.Empty:
                break
            callback(value)
        if not self.closing:
            self.root.after(50, self.process_events)

    def refresh(self) -> None:
        self.refresh_button.state(["disabled"])
        self.status.set("正在查询当前可用的蓝牙串口服务…")
        self._submit(discover_devices(), self.devices_loaded)

    def devices_loaded(self, devices: list[BluetoothCarDevice]) -> None:
        self.devices = devices
        self.device_box["values"] = [device.label for device in devices]
        default = next(
            (i for i, device in enumerate(devices) if device.name == "OmniBCI_Car_Kit"), 0
        )
        if devices:
            self.device_box.current(default)
            self.status.set(f"发现 {len(devices)} 个设备。选择 OmniBCI_Car_Kit 后连接。")
        else:
            self.status.set(
                "当前没有发现可连接的蓝牙串口服务。关闭手机控制 APP 和串口助手，"
                "确认小车已开机后再刷新。"
            )
        self.refresh_button.state(["!disabled"])

    def selected_device(self) -> BluetoothCarDevice | None:
        index = self.device_box.current()
        return self.devices[index] if 0 <= index < len(self.devices) else None

    def connect(self) -> None:
        device = self.selected_device()
        if device is None:
            self.status.set("请先选择蓝牙设备。")
            return
        self.busy = True
        self.connect_button.state(["disabled"])
        self.refresh_button.state(["disabled"])
        self.status.set(f"正在直接连接 {device.label}…")
        self.pending_future = self._submit(
            self.car.connect(device), self.connected_ok, self.operation_failed
        )

    def connected_ok(self, _result) -> None:
        self.busy = False
        self.connected = True
        self.connect_button.state(["disabled"])
        self.disconnect_button.state(["!disabled"])
        self.refresh_button.state(["disabled"])
        self.set_direction_buttons(True)
        self.status.set("已连接。连接时已发送停止指令；可以按住方向按钮控制。")

    def disconnect(self) -> None:
        self.busy = True
        self.disable_controls()
        self.status.set("正在发送停止并断开蓝牙…")
        self.pending_future = self._submit(
            self.car.disconnect(), self.disconnected_ok, self.operation_failed
        )

    def disconnected_ok(self, _result) -> None:
        self.connected = self.busy = False
        self.disconnect_button.state(["disabled"])
        self.connect_button.state(["!disabled"])
        self.refresh_button.state(["!disabled"])
        self.status.set("已断开。")

    def press(self, command: str) -> None:
        if not self.connected or self.busy or command in self.pressed:
            return
        self.pressed.add(command)
        self._submit(getattr(self.car, command)(), self.command_sent, self.operation_failed)

    def release(self, command: str) -> None:
        if command not in self.pressed:
            return
        self.pressed.discard(command)
        self.send_stop()

    def send_stop(self) -> None:
        self.pressed.clear()
        if self.connected and not self.busy:
            self._submit(self.car.stop(), self.command_sent, self.operation_failed)

    def command_sent(self, _result) -> None:
        pass

    def disable_controls(self) -> None:
        self.set_direction_buttons(False)
        self.connect_button.state(["disabled"])
        self.disconnect_button.state(["disabled"])
        self.refresh_button.state(["disabled"])

    def set_direction_buttons(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        for button in self.direction_buttons:
            button.state(state)
        self.stop_button.state(state)

    def operation_failed(self, error: Exception) -> None:
        self.busy = False
        self.connected = self.car.connected
        self.set_direction_buttons(self.connected)
        self.connect_button.state(["disabled"] if self.connected else ["!disabled"])
        self.disconnect_button.state(["!disabled"] if self.connected else ["disabled"])
        self.refresh_button.state(["!disabled"] if not self.connected else ["disabled"])
        self.status.set(f"操作失败：{error}")

    def show_error(self, error: Exception) -> None:
        self.refresh_button.state(["!disabled"])
        self.status.set(f"蓝牙搜索失败：{error}")

    def close_window(self) -> None:
        if self.closing:
            return
        self.closing = True
        def cleanup(_future=None):
            future = self.worker.submit(self.car.disconnect())

            def finish(_done):
                self.worker.loop.call_soon_threadsafe(self.worker.loop.stop)
                try:
                    self.root.after(0, self.root.destroy)
                except tk.TclError:
                    pass

            future.add_done_callback(finish)

        pending = getattr(self, "pending_future", None)
        if self.busy and pending is not None and not pending.done():
            pending.add_done_callback(cleanup)
        elif self.connected:
            cleanup()
        else:
            self.worker.loop.call_soon_threadsafe(self.worker.loop.stop)
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CarController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
