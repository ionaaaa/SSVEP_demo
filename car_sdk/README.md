# OmniBCI 小车蓝牙 SDK

这个发行包只提供 Windows 蓝牙控制接口。程序通过 RFCOMM/SPP 直接连接小车，不需要 COM 口、串口线或串口助手。

## 安装

Windows 10/11、Python 3.11–3.13。在 Windows 蓝牙设置中配对 `OmniBCI_Car_Kit`，然后可从源码目录进行 editable install：

```powershell
python -m pip install -e .\car_sdk
```

安装会自动拉取 SDK 所需的 Windows 蓝牙运行组件；这些依赖带有 Windows 平台条件。

## 在其他 Python 程序中使用

```python
import asyncio
from bluetooth_car import BluetoothCar, discover_devices


async def main():
    devices = await discover_devices()
    car_device = next(
        (device for device in devices if device.name == "OmniBCI_Car_Kit"),
        None,
    )
    if car_device is None:
        raise RuntimeError("未找到已配对且可用的小车蓝牙设备")

    car = BluetoothCar()
    await car.connect(car_device)  # 连接后 SDK 会先发送停止指令
    try:
        await car.forward()
        await asyncio.sleep(0.5)
        await car.stop()
    finally:
        await car.disconnect()


asyncio.run(main())
```

首次测试请架空车轮。方向指令会持续生效，直到发送 `stop()` 或断开；断联时无法保证小车立即停车。程序应在 `finally` 中断开连接。

## 接口

- `await discover_devices()`：返回当前已配对且提供蓝牙串口服务的设备列表。
- `await car.connect(device)`：连接设备；连接成功后先发停止指令。
- `await car.forward()`、`backward()`、`left()`、`right()`：发送持续运动指令。
- `await car.stop()`：发送停止指令。
- `await car.disconnect()`：尽力发送停止指令并释放连接。

小车需已开机，且不能同时被手机控制 APP、串口助手或其他程序占用。蓝牙协议没有设备状态回执；发送成功只表示数据已交给蓝牙连接，不代表小车确认执行。
