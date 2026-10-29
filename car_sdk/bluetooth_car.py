"""Direct Windows Bluetooth RFCOMM controller for the supplied car firmware."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from winrt.windows.devices.bluetooth import BluetoothCacheMode, BluetoothDevice
from winrt.windows.devices.bluetooth.rfcomm import RfcommServiceId
from winrt.windows.devices.enumeration import DeviceAccessStatus, DeviceInformation
from winrt.windows.networking.sockets import StreamSocket
from winrt.windows.storage.streams import DataWriter


COMMANDS = {
    "forward": b"ONA",
    "backward": b"ONB",
    "left": b"ONC",
    "right": b"OND",
    "stop": b"ONF",
}
NO_SERVICE = (
    "Windows 当前没有发现此设备正在提供蓝牙串口服务。请关闭手机控制 APP 和串口助手，"
    "确认小车与蓝牙模块已开机，再点“刷新设备”。"
)


@dataclass(frozen=True)
class BluetoothCarDevice:
    name: str
    address: str

    @property
    def label(self) -> str:
        return f"{self.name} — {self.address}"


def _address_from_device_id(device_id: str) -> str:
    try:
        return device_id.rsplit("-", 1)[1].lower()
    except (IndexError, AttributeError) as exc:
        raise ValueError(f"Unrecognized Bluetooth device ID: {device_id}") from exc


async def _live_services(address: str):
    numeric_address = int(address.replace(":", ""), 16)
    device = await BluetoothDevice.from_bluetooth_address_async(numeric_address)
    if device is None:
        return None, None, []
    try:
        result = await device.get_rfcomm_services_for_id_with_cache_mode_async(
            RfcommServiceId.serial_port, BluetoothCacheMode.UNCACHED
        )
        return device.name, device, list(result.services)
    except Exception:
        device.close()
        raise


async def discover_devices() -> list[BluetoothCarDevice]:
    """List paired devices that currently advertise a Serial Port Profile."""
    selector = BluetoothDevice.get_device_selector_from_pairing_state(True)
    paired = await DeviceInformation.find_all_async_aqs_filter(selector)
    found = []
    for info in paired:
        try:
            address = _address_from_device_id(info.id)
        except ValueError:
            continue
        name, bluetooth_device, services = await _live_services(address)
        try:
            if name and services:
                found.append(BluetoothCarDevice(name, address))
        finally:
            for service in services:
                service.close()
            if bluetooth_device is not None:
                bluetooth_device.close()
        # Services can be empty when the device is off or another app owns it.
    return sorted(found, key=lambda d: (d.name.casefold(), d.address))


class BluetoothCar:
    """Async direct RFCOMM client; does not use a Windows virtual COM port."""

    def __init__(self) -> None:
        self._service = None
        self._device = None
        self._socket = None
        self._writer = None
        self._last_write: float | None = None
        self._send_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._writer is not None

    async def connect(self, device: BluetoothCarDevice) -> None:
        if self.connected:
            raise RuntimeError("Already connected; disconnect before selecting another device")

        _name, bluetooth_device, services = await _live_services(device.address)
        if not services:
            if bluetooth_device is not None:
                bluetooth_device.close()
            raise ConnectionError(NO_SERVICE)
        service, *unused = services
        for stale in unused:
            stale.close()
        socket = StreamSocket()
        try:
            access = await service.request_access_async()
            if access != DeviceAccessStatus.ALLOWED:
                raise PermissionError(f"Windows denied Bluetooth access: {access}")
            await socket.connect_with_protection_level_async(
                service.connection_host_name,
                service.connection_service_name,
                service.protection_level,
            )
            writer = DataWriter(socket.output_stream)
            self._device, self._service, self._socket, self._writer = (
                bluetooth_device, service, socket, writer
            )
            await self.stop()
        except Exception:
            socket.close()
            service.close()
            bluetooth_device.close()
            self._device = None
            self._service = self._socket = self._writer = None
            raise

    async def send(self, command: str) -> None:
        try:
            payload = COMMANDS[command]
        except KeyError as exc:
            raise ValueError(f"Unsupported car command: {command!r}") from exc
        if not self.connected:
            raise ConnectionError("Not connected to a car")

        async with self._send_lock:
            if self._last_write is not None:
                delay = 0.15 - (asyncio.get_running_loop().time() - self._last_write)
                if delay > 0:
                    await asyncio.sleep(delay)
            writer = self._writer
            if writer is None:
                raise ConnectionError("Bluetooth connection is closed")
            writer.write_bytes(payload)
            await writer.store_async()
            await writer.flush_async()
            self._last_write = asyncio.get_running_loop().time()

    async def forward(self) -> None:
        await self.send("forward")

    async def backward(self) -> None:
        await self.send("backward")

    async def left(self) -> None:
        await self.send("left")

    async def right(self) -> None:
        await self.send("right")

    async def stop(self) -> None:
        await self.send("stop")

    async def disconnect(self) -> None:
        error = None
        if self.connected:
            try:
                await self.stop()
            except Exception as exc:
                error = exc
        writer, socket, service, device = (
            self._writer, self._socket, self._service, self._device
        )
        self._writer = self._socket = self._service = self._device = None
        for resource in (writer, socket, service, device):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    error = error or exc
        if error:
            raise error
