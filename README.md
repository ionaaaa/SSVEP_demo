# SSVEP Demo

这是一个用于验证 SSVEP（稳态视觉诱发电位）软件链路的 Python 项目。项目包含：

- PsychoPy 多频闪烁刺激界面；
- synthetic、replay 和 live 三种 EEG 输入模式；
- FFT 和 CCA 两种解码器；
- 带置信度、连续确认、运动时限和输入超时保护的 `SafeCommandDispatcher`；
- 虚拟小车控制器；
- Windows Classic Bluetooth RFCOMM/SPP 真实小车控制器；
- EEG 窗口归档、trial CSV、结构化事件和 session summary。

> **安全警告**
>
> 使用真实蓝牙小车前，务必先把车轮架空，并保证附近没有人员、宠物或易损物品。蓝牙日志中的 `sent` 只表示 SDK/Windows 蓝牙栈写入成功，不代表小车已确认执行，也不代表真实位置发生了变化。

## 1. 支持范围

| 模式 | EEG 来源 | Virtual controller | Bluetooth controller | 是否需要 GUI |
|---|---|---:|---:|---:|
| `synthetic` | 项目生成的模拟 EEG | 支持 | 支持 | 可选 |
| `live` | omniBCI WebSocket | 支持 | 支持 | 必须 |
| `replay` | 已保存的 `eeg_windows.npz` | 支持 | **禁止** | 可选 |

`replay + bluetooth` 被明确禁止，避免历史预测在用户未预期时驱动真实小车。

## 2. 环境要求

推荐环境：

- Python 3.11；
- Windows 10/11：运行 PsychoPy 或真实蓝牙小车；
- Linux/macOS：可以运行无 GUI synthetic、replay 和大部分自动测试，但不能创建真实蓝牙控制器；
- 支持 OpenGL 的本地显示器：运行 PsychoPy GUI；
- 真实 live 模式还需要一个正在运行的 omniBCI WebSocket 服务。

真实小车 SDK 支持 Python 3.11–3.13。为同时满足主项目、PsychoPy 和蓝牙 SDK，建议直接使用 Python 3.11。

### NumPy 版本说明

当前 FFT decoder 使用了 NumPy 1.x 的 `np.trapz`。如果需要运行 FFT decoder 或完整测试集，请暂时使用：

```bash
python -m pip install "numpy>=1.24,<2"
```

CCA decoder 不经过这条 FFT 代码路径。

## 3. 获取代码并创建虚拟环境

进入项目根目录，也就是包含本 README 和 `pyproject.toml` 的目录：

```bash
cd /d/repo/SSVEP
```

创建虚拟环境：

```bash
python -m venv .venv
```

Git Bash 激活：

```bash
source .venv/Scripts/activate
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

升级安装工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装项目

### 4.1 安装主项目、PsychoPy 和测试依赖

```bash
python -m pip install "numpy>=1.24,<2"
python -m pip install -e ".[stimulus,test]"
```

如果机器没有显示器，且只需要运行无 GUI synthetic/replay，可只安装基础依赖：

```bash
python -m pip install "numpy>=1.24,<2"
python -m pip install -e ".[test]"
```

### 4.2 仅在需要真实蓝牙小车时安装 SDK

真实蓝牙只支持 Windows 10/11：

```bash
python -m pip install -e ./car_sdk
```

安装后可以做不连接设备的导入检查：

```bash
python -c "import bluetooth_car; print('car_sdk import OK')"
```

不要把 `car_sdk/.venv-bt` 加入 `PYTHONPATH`，也不要手工修改 `sys.path`。项目使用 editable install 查找 SDK。

## 5. 第一次运行：无 GUI synthetic 冒烟测试

这是拿到项目后最推荐的第一条命令。它不打开 PsychoPy、不连接 EEG、不连接真实小车：

```bash
python scripts/run_ssvep_demo.py \
  --mode synthetic \
  --controller virtual \
  --decoder cca \
  --max-trials 4 \
  --no-gui
```

成功后会打印一个 session 目录，例如：

```text
outputs/ssvep_demo/session_20261106T010203Z
```

至少检查以下文件是否存在：

```text
config.yaml
events.jsonl
trials.csv
frame_intervals.csv
eeg_windows.npz
summary.json
```

## 6. 运行 PsychoPy synthetic + virtual

```bash
python scripts/run_ssvep_demo.py \
  --mode synthetic \
  --controller virtual \
  --decoder cca \
  --max-trials 4 \
  --windowed
```

界面按键：

- `Enter`：开始 session；
- `P`：暂停或恢复；暂停会立即 STOP 并清除旧运动；
- `Esc`：安全停止并退出；
- 关闭窗口：等价于安全中止。

`synthetic` 使用当前 trial 的目标频率生成模拟 EEG，只能验证软件链路，不能证明真实人体 SSVEP、真实 EEG 同步或真实小车控制质量。

## 7. 使用本地 fake WebSocket 验证 live 软件链路

项目自带一个模拟 omniBCI WebSocket 服务。它适合验证网络接收、8 通道映射、连续序列和 EEG 归档，但生成的数据不是人体 SSVEP。

终端 1：启动 fake server。

```bash
python scripts/fake_omnibci_ws.py \
  --host 127.0.0.1 \
  --port 8766 \
  --scenario continuous
```

终端 2：使用 virtual controller 运行 live。

```bash
python scripts/run_ssvep_demo.py \
  --mode live \
  --source omnibci-websocket \
  --server-url ws://127.0.0.1:8766/v1/stream \
  --alignment-mode buffer-sequence \
  --controller virtual \
  --decoder cca \
  --max-trials 1 \
  --windowed
```

fake server 还支持以下场景：

```text
continuous  连续有效帧
gap         注入 sequence gap
invalid     注入 invalid frame
reconnect   主动断开一次以测试重连
```

例如：

```bash
python scripts/fake_omnibci_ws.py --scenario gap
```

> 不要用 fake EEG 判断 decoder 准确率，也不要把 fake server 和真实蓝牙小车组合用于运动验证。

## 8. 运行真实 omniBCI live EEG

live 模式要求外部程序已经在指定地址提供 omniBCI WebSocket `/v1/stream`。默认示例地址是：

```text
ws://127.0.0.1:8766/v1/stream
```

先使用 virtual controller 验证真实 EEG 链路：

```bash
python scripts/run_ssvep_demo.py \
  --mode live \
  --source omnibci-websocket \
  --server-url ws://127.0.0.1:8766/v1/stream \
  --alignment-mode buffer-sequence \
  --controller virtual \
  --decoder cca \
  --max-trials 4 \
  --windowed
```

当前标准通道顺序是：

```text
PO7, PO3, POz, PO4, PO8, O1, Oz, O2
```

服务端实际通道名和映射由 `config/ssvep_demo.yaml` 中的 `live.source_channel_names` 与 `live.channel_map` 决定。服务端采样率必须与配置中的 `250 Hz` 一致。

### Live 对齐模式

- `buffer-sequence`：在刺激翻转时读取接收缓冲区 sequence，属于软件侧对齐；
- `trigger`：使用外部 omniBCI trigger HTTP endpoint 返回的设备 sequence。

Trigger 模式需要额外参数：

```bash
python scripts/run_ssvep_demo.py \
  --mode live \
  --source omnibci-websocket \
  --server-url ws://127.0.0.1:8766/v1/stream \
  --alignment-mode trigger \
  --trigger-url http://127.0.0.1:8766/v1/trigger \
  --controller virtual \
  --decoder cca \
  --max-trials 4 \
  --windowed
```

软件对齐或 trigger sequence 都不等同于光子级刺激起始验证；设备级同步仍需现场测量。

## 9. 使用真实蓝牙小车

### 9.1 Windows 配对和准备

1. 给小车上电。
2. 在 Windows 蓝牙设置中完成配对。
3. 记下系统显示的精确设备名称，例如 `OmniBCI_Car_Kit`。
4. 关闭可能占用蓝牙串口服务的手机 APP、串口助手和其他控制程序。
5. 安装 `car_sdk`：

   ```bash
   python -m pip install -e ./car_sdk
   ```

6. **架空车轮。**

设备 ID/蓝牙地址比名称优先。如果同名设备超过一个，请显式使用 `--car-device-id`。

### 9.2 Synthetic + Bluetooth 硬件链路测试

```bash
python scripts/run_ssvep_demo.py \
  --mode synthetic \
  --controller bluetooth \
  --car-device-name "OmniBCI_Car_Kit" \
  --decoder cca \
  --confirmations-required 1 \
  --max-trials 4 \
  --windowed
```

连接顺序是：创建 session → 连接小车 → 先确认发送 STOP → 创建 PsychoPy 窗口 → 开始 trial。连接失败不会创建刺激窗口，也不会自动回退到 virtual。

### 9.3 Live EEG + Bluetooth

请先完成 `live + virtual` 验证，再运行：

```bash
python scripts/run_ssvep_demo.py \
  --mode live \
  --source omnibci-websocket \
  --server-url ws://127.0.0.1:8766/v1/stream \
  --alignment-mode buffer-sequence \
  --controller bluetooth \
  --car-device-name "OmniBCI_Car_Kit" \
  --decoder cca \
  --confirmations-required 1 \
  --max-trials 4 \
  --windowed
```

当前运动映射：

| SSVEP 频率 | 控制命令 | 蓝牙 payload |
|---:|---|---|
| 8 Hz | LEFT | `ONC` |
| 10 Hz | RIGHT | `OND` |
| 12 Hz | FORWARD | `ONA` |
| 15 Hz | BACKWARD | `ONB` |
| 安全停车 | STOP | `ONF` |

默认运动最长持续 `0.5 s`，之后 dispatcher 自动发送 STOP。

### 连续确认的重要说明

`control.confirmations_required` 表示需要连续几次相同、有效且高置信度的预测。`control.max_confirmation_gap_s` 限制两次确认间隔。

当前 live 流程每个 trial 只产生一次解码结果，而完整 trial 间隔通常大于默认的 `1 s`。因此：

- 使用默认 `confirmations_required: 2` 时，结果可能一直停留在 `awaiting_confirmation`；
- `--confirmations-required 1` 适合架空车轮的软件/硬件链路检查；
- 真实落地运行前应先评估准确率、置信度分布和误动作风险，不应仅为“让车动”而降低安全门槛。

## 10. Replay 已保存的 EEG

从已有 session 中选择 `eeg_windows.npz`：

```bash
python scripts/run_ssvep_demo.py \
  --mode replay \
  --controller virtual \
  --replay-file outputs/ssvep_demo/session_YYYYMMDDTHHMMSSZ/eeg_windows.npz \
  --decoder cca \
  --no-gui
```

只 replay 指定 trial：

```bash
python scripts/run_ssvep_demo.py \
  --mode replay \
  --controller virtual \
  --replay-file outputs/ssvep_demo/session_YYYYMMDDTHHMMSSZ/eeg_windows.npz \
  --trial-ids 0,2,3 \
  --decoder cca \
  --no-gui
```

Replay 会创建新的输出 session，不会修改原始 EEG archive，也不会重新显示历史刺激。

## 11. 配置文件

默认配置位于：

```text
config/ssvep_demo.yaml
```

配置优先级：

```text
显式 CLI 参数 > YAML 配置 > dataclass 默认值
```

主要配置块：

- `stimulus.*`：频率、trial/cue/rest 时长、窗口和颜色；
- `acquisition.*`：采样率和 decoder 通道顺序；
- `decoder.*`：默认 decoder、谐波数和带通范围；
- `commands.*`：频率到控制命令的映射；
- `control.*`：置信度、连续确认、运动时限、输入超时；
- `controller.*`：virtual/bluetooth 选择与蓝牙超时；
- `synthetic_demo.*`：模拟 EEG 的 SNR、随机种子和确认次数；
- `live.*`：WebSocket 通道、trial 样本数、对齐和重连参数；
- `ui.*`：CJK 字体覆盖。

需要保留默认配置时，可以复制一份：

```bash
cp config/ssvep_demo.yaml config/ssvep_demo.local.yaml
```

然后通过以下参数使用：

```bash
python scripts/run_ssvep_demo.py --config config/ssvep_demo.local.yaml ...
```

不要在配置或日志中保存蓝牙 PIN、Windows 凭据或其他敏感信息。

## 12. 安全调度语义

所有 decoder 输出都经过：

```text
DecodeResult → SafeCommandDispatcher → Controller
```

以下情况会触发 STOP 或拒绝运动：

- UNKNOWN 或频率/命令不匹配；
- 置信度低于阈值；
- prediction 过期或乱序；
- 连续确认未完成；
- 输入超过 `input_timeout_s` 没有有效更新；
- 运动达到 `command_duration_s`；
- Pause、Escape、窗口关闭、trial 失败或异常退出；
- Bluetooth controller 进入 FAULT；
- dispatcher 已关闭。

`input_timeout` 是安全 watchdog 发出的 STOP，不表示蓝牙网络 I/O 超时。低置信度已经 STOP 后，watchdog 仍可能在后续 frame 再次报告 `input_timeout`。

## 13. Session 输出和日志诊断

默认输出目录：

```text
outputs/ssvep_demo/
```

每次运行创建独立 session：

```text
session_YYYYMMDDTHHMMSSZ/
├── config.yaml
├── events.jsonl
├── trials.csv
├── frame_intervals.csv
├── eeg_windows.npz
└── summary.json
```

文件说明：

- `config.yaml`：本次运行最终解析后的配置；
- `events.jsonl`：逐行独立 JSON 的生命周期、EEG 和 controller 事件；
- `trials.csv`：每个 trial 的预测、置信度、dispatcher 决策和 controller 状态；
- `frame_intervals.csv`：PsychoPy frame interval 和 suspected dropped frame；
- `eeg_windows.npz`：无 pickle 的 `[trial, channel, sample]` EEG archive；
- `summary.json`：session 最终状态、准确率、连接状态和事件计数。

Controller 命令事件语义：

| 事件 | 含义 |
|---|---|
| `controller_command_requested` | dispatcher 请求 controller 执行方向命令 |
| `controller_command_queued` | Bluetooth mailbox 接受命令 |
| `controller_command_coalesced` | 旧运动被 latest-wins 或 STOP 替换 |
| `controller_command_send_started` | worker 开始调用 SDK |
| `controller_command_sent` | SDK/主机蓝牙栈写入成功返回 |
| `controller_command_failed` | SDK 写入失败或超时 |
| `controller_stop_requested` | 安全层或生命周期请求 STOP |

排查“小车为什么没动”时，依次查看：

1. `trials.csv` 的 `confidence`、`dispatcher_action`、`dispatcher_reason`；
2. `summary.json` 的 `controller_requested_command_count`；
3. `events.jsonl` 中是否出现方向命令的 requested → queued → send_started → sent；
4. `controller_failed_command_count`、`controller_last_error` 和 `controller_final_state`。

常见判断：

- `low_confidence`：置信度未通过；
- `awaiting_confirmation`：还缺少连续相同预测；
- `input_timeout`：有效输入超时，安全停车；
- `command_duration_elapsed`：运动时限到达，正常停车；
- 只有 `ONF`：只发送了 STOP，没有方向命令通过安全层；
- `sent` 但车没有物理动作：检查车轮、电源、设备占用和小车端状态。

## 14. 其他脚本

只运行视觉刺激：

```bash
python scripts/run_ssvep_stimulus.py --windowed
```

键盘控制虚拟小车：

```bash
python scripts/run_virtual_car_demo.py --windowed
```

按键映射：`1=LEFT`、`2=RIGHT`、`3=FORWARD`、`4=BACKWARD`、`U=低置信度`、`Esc=退出`。

离线评估 synthetic decoder：

```bash
python scripts/evaluate_synthetic_decoder.py \
  --decoder cca \
  --trials-per-frequency 20 \
  --snr-db -20 -10 0 10
```

结果保存在 `results/`。

## 15. 运行测试

```bash
pytest -q
python -m compileall -q ssvep_demo scripts tests
git diff --check
```

部分 PsychoPy smoke test 需要真实显示器，会在无显示器环境中 skip。测试不会自动连接真实蓝牙或驱动小车。

如果使用 NumPy 2.x，FFT 相关测试目前会因 `np.trapz` 被移除而失败；请使用前述 `numpy<2` 环境，或仅在明确修改 FFT 兼容性后升级。

## 16. 常见问题

### PsychoPy 无法导入

确认安装了 stimulus extra：

```bash
python -m pip install -e ".[stimulus]"
```

并确认当前终端使用的是项目虚拟环境：

```bash
python -c "import sys; print(sys.executable)"
```

### `Monitor specification not found`

这是 PsychoPy 未找到显示器校准配置时创建临时 monitor 的警告，通常不会阻止运行。需要精确视觉角度时，应在 PsychoPy Monitor Center 中配置真实显示器尺寸和观察距离。

### 出现大量 dropped frame

优先尝试：

- 关闭录屏、浏览器动画和占用 GPU/CPU 的程序；
- 使用全屏模式；
- 关闭系统节能和动态刷新率；
- 确认 PsychoPy 使用独立显卡；
- 不要把 Bluetooth 连接、文件扫描或其他阻塞工作放进 frame loop。

大量掉帧会影响刺激时序和真实 SSVEP 质量，但不等同于蓝牙发送失败。

### 找不到蓝牙设备

检查：

- Windows 已完成配对；
- 设备名称完全一致；
- 小车已上电；
- 没有其他程序占用 RFCOMM/SPP；
- `car_sdk` 已安装到当前 Python 环境；
- 同名设备超过一个时改用 `--car-device-id`。

### Bluetooth 连接成功但车不动

先查看日志，不要直接降低安全阈值。最常见原因是：

- `low_confidence`；
- `awaiting_confirmation`；
- 只有 STOP `ONF` 被发送；
- 方向命令只持续了默认 `0.5 s`；
- SDK 写入成功，但小车端没有执行。

### Session 为什么显示 `failed` 或 `aborted`

- `completed`：计划 trial 正常结束；
- `aborted`：Escape、窗口关闭、KeyboardInterrupt 或用户主动停止；
- `failed`：连接失败、controller FAULT、EEG 无法完成、decoder 异常或其他运行错误。

即使蓝牙连接失败，项目也会尽量保留 `config.yaml`、`events.jsonl`、空 `trials.csv`、合法空 EEG archive 和 `summary.json`，便于定位问题。

## 17. 项目结构

```text
SSVEP/
├── car_sdk/                    # Windows Classic Bluetooth RFCOMM/SPP SDK
├── config/                     # 默认和兼容配置
├── scripts/                    # 主入口、fake server 和辅助脚本
├── ssvep_demo/                 # 刺激、解码、控制器、live source、日志
├── tests/                      # 无硬件自动测试和显示器 smoke test
├── outputs/                    # 运行 session（git ignored）
├── results/                    # 离线评估结果（git ignored）
├── pyproject.toml
└── README.md
```

主入口帮助：

```bash
python scripts/run_ssvep_demo.py --help
```
