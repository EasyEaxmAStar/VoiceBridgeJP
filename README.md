# VoiceBridge JP

在 Windows 上将普通话转换成日语语音，并通过虚拟麦克风送入 Zoom、Teams 等会议软件。

VoiceBridge JP 使用本地语音识别、翻译和语音合成模型。模型下载完成后，推理可离线运行。桌面界面提供翻译与原声切换、语速调整、用户词典、分句控制和摄像头效果。

## 功能

- **普通话 → 日语**：Qwen3-ASR 识别中文，OPUS-MT 在 CPU 上翻译，GPT-SoVITS 合成日语。
- **翻译 / 原声切换**：TRANSLATE 输出合成日语；DIRECT 将麦克风原声送入 VB-CABLE。
- **热待机**：模型保持加载，支持 Start、Pause、Resume、Stop。
- **用户词典**：分别设置中文或英文源词、日语译名和假名读音，支持导入、导出和热更新。初始词典为空。
- **分句控制**：自动、手动及混合模式，可调整静音时长、最长话语和检测阈值。
- **日语语体**：面向商务场景的敬体处理与校验，未通过的内容仅在界面提示。
- **有序播放**：等待队列最多 16 条话语，显示队列数量和等待时间；满队列时拒绝新话语，继续处理已接收内容。
- **摄像头效果**：麦克风声音起始时触发马赛克、跳帧和冻结，日语实际开始播放后恢复清晰画面。支持 OBS Virtual Camera。
- **接口与文件处理**：提供 WebSocket 服务和 WAV 处理模块，协议见 [PROTOCOL.md](PROTOCOL.md)。

## 工作方式

```text
物理麦克风
   ↓
分句 → Qwen3-ASR → OPUS-MT → 日语语体与术语处理
                                    ↓
                              GPT-SoVITS
                                    ↓
                             CABLE Input
                                    ↓
                     会议软件选择 CABLE Output
```

识别按话语进行。自动模式在停顿后提交，手动模式由用户结束当前句；麦克风可以继续采集后续内容。Pause、Stop 和模式切换会清理失效任务，防止旧语音进入新路线。

## 系统与硬件

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11，原生运行 |
| 应用环境 | Python 3.12，64 位 |
| TTS 环境 | Python 3.11，独立虚拟环境 |
| GPU | NVIDIA CUDA GPU；已有测量使用 16GB 显存 |
| 内存 | 已有测量使用 64GB RAM |
| 音频 | 物理麦克风、VB-CABLE、FFmpeg |
| 摄像头（可选） | 物理摄像头、OBS Virtual Camera、独立 Python 环境 |

开发与性能测量平台为 **Intel i9-13900H、RTX 4090 Laptop 16GB、64GB RAM、Windows 11**。其他配置的资源占用和延迟需按实际使用情况评估。

## 模型

| 用途 | 模型 | 运行方式 |
|---|---|---|
| 中文识别 | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | CUDA BF16 / SDPA |
| 中文到日语 | [OPUS-MT zh-ja](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zh-ja) | CPU |
| 日语合成 | [GPT-SoVITS v2ProPlus](https://github.com/RVC-Boss/GPT-SoVITS) | CUDA FP16，常驻 Python 3.11 worker |
| 翻译恢复（可选） | Qwen3-4B-Instruct-2507 | 本地独立 worker |

默认 TTS 使用官方预训练权重：`s1v3.ckpt` 和 `v2Pro/s2Gv2ProPlus.pth`。源码版本、模型版本和文件哈希记录在 [gptsovits.lock.json](gptsovits.lock.json)。

GPT-SoVITS 需要一段参考音频来确定说话方式。准备一段有使用授权的日语 WAV，以及音频中实际说出的准确文本。参考要求：

- 长度 3–10 秒。
- 采样率 16–48kHz，单声道或双声道。
- 声音清晰，没有削波、句子截断或明显背景噪声。

## 安装

以下命令均在项目根目录的 PowerShell 中执行。先安装 Python 3.12、Python 3.11、Git 和 FFmpeg，并确认 `ffmpeg -version` 可用。NVIDIA 驱动需支持 CUDA 12.4 运行时。

### 1. 创建应用环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt -r requirements-asr-cuda.txt -r requirements-gui.txt
```

主环境负责 GUI、ASR 和翻译，OPUS-MT 在 CPU 上运行。上游 `pyproject.toml` 还包含其他平台和替代后端的依赖；Windows 默认安装使用上面的依赖文件即可。

### 2. 创建 TTS 环境

```powershell
py -3.11 -m venv .venv-gptsovits
.\.venv-gptsovits\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gptsovits\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv-gptsovits\Scripts\python.exe -m pip install -r requirements-gptsovits.txt
```

获取对应版本的 GPT-SoVITS 源码：

```powershell
New-Item -ItemType Directory -Force .model-cache | Out-Null
git clone https://github.com/RVC-Boss/GPT-SoVITS .model-cache/GPT-SoVITS
git -C .model-cache/GPT-SoVITS checkout b4570b876f5c086819ba84e3215472733c6a7b70
```

运行时会核对源码版本和模型哈希。更新 GPT-SoVITS 时，需要同步检查兼容性和模型清单。

### 3. 下载模型与语言资源

```powershell
.\.venv\Scripts\python.exe tools/download_models.py --asr --opus --tts
.\.venv\Scripts\python.exe tools/setup_gptsovits_nltk.py
```

ASR 和 OPUS-MT 使用标准 Hugging Face 缓存。GPT-SoVITS 文件保存到 `.model-cache/GPT-SoVITS/GPT_SoVITS/pretrained_models`。NLTK 资源供英语发音前端使用，日语中包含英文词或技术术语时也会用到。

下载阶段需要联网；运行阶段使用已缓存文件。模型、参考录音和虚拟环境均不包含在源码中。

### 4. 安装音频设备

从 [VB-Audio](https://vb-audio.com/Cable/) 安装 VB-CABLE，按安装程序要求完成设置。

| 位置 | 选择的设备 |
|---|---|
| VoiceBridge JP 输入 | 物理麦克风，Windows WASAPI |
| VoiceBridge JP 输出 | CABLE Input |
| 会议软件麦克风 | CABLE Output |

设备编号可能在重启或重新连接后变化，界面按设备身份选择。

## 启动与使用

```powershell
.\tools\start_voicebridge_gui.ps1
```

也可以直接运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m voicebridge.gui
```

首次启动时选择参考 WAV、填写准确日语文本，并选择物理麦克风。配置保存在 `config/public_tts.local.json`。模型加载、预热完成后，界面显示 READY。

1. 选择 **TRANSLATE**，点击 **Start**，开始说普通话。
2. 在会议软件中选择 **CABLE Output**，接收合成的日语。
3. 需要直接使用原声时切换到 **DIRECT**。
4. **Pause / Resume** 暂停和恢复当前路线；**Stop** 停止音频并保留模型待机。
5. 退出窗口会关闭本窗口拥有的模型、音频和摄像头进程。

高级设置中可以编辑词典、调整语速和分句参数。参考音频需要更换时，关闭应用并移走 `config/public_tts.local.json`，再次启动即可重新配置。环境变量配置见 [.env.example](.env.example)，显式环境变量优先。

### 词典与分句

词典将“日语显示文字”和“合成读音”分开。例如，技术词可以在字幕中保留英文，在语音中使用指定假名读音。中文人名还可以提供拼音，用于可选的 ASR 上下文提示。

自动分句适合自然停顿；手动或混合模式适合长句、演讲和需要精确控制提交时机的场景。连续输入快于日语处理和播放速度时，等待时间会增加。队列达到 12/16 时显示预警，满队列时仅拒绝新到话语。

### 可选：摄像头效果

```powershell
py -3.12 -m venv .venv-camera
.\.venv-camera\Scripts\python.exe -m pip install --only-binary=:all: --require-hashes -r requirements-camera.lock.txt
```

安装 [OBS Studio](https://obsproject.com/) 后，在 VoiceBridge JP 的摄像头设置中选择物理摄像头和 **OBS Virtual Camera**，再在会议软件中选择同一虚拟摄像头。

- 推荐从 **1280×720、30FPS、CPU BGR** 开始。
- 默认按麦克风声音起始触发，也可切回 ASR 开始触发。
- 强烈卡顿通过重复帧模拟冻结，虚拟输出仍保持目标发送帧率。
- 马赛克、冻结范围和强度可以独立调整。
- 实际非豁免音频开始播放后恢复清晰画面；快捷回复 `quick.hai` 保持豁免。
- NV12 为实验路径，存在色差，每次启动需确认。

摄像头处理在 CPU 上运行。暗光或曝光条件可能影响实际采集帧率；界面分别显示物理采集和虚拟发送帧率。

### 可选：Qwen 翻译恢复

默认的 OPUS Fast 可以独立运行。启用 Qwen 恢复后端需要另行准备 Qwen3-4B-Instruct-2507 模型和对应校验清单。它会增加 GPU 占用，未准备模型时界面显示后端不可用。

## 性能参考

以下为 Windows 11、i9-13900H、RTX 4090 Laptop 16GB、64GB RAM 平台的历史测量。完整链路测量使用了微调版 GPT-SoVITS；表中的 TTS、整体延迟和资源占用用于说明该平台的运行情况，官方预训练分支尚未重新测量。

720p 摄像头开启、模型温态驻留，每种视频路径各测 4 次短句。P50 为中位数，P95 为第 95 百分位。

| 阶段 | CPU BGR：P50 / P95 | NV12：P50 / P95 |
|---|---:|---:|
| ASR 识别 | 0.484 / 0.514 秒 | 0.474 / 0.515 秒 |
| CPU 翻译 | 0.192 / 0.193 秒 | 0.193 / 0.195 秒 |
| TTS 请求到父进程收到首段 PCM | 0.773 / 0.941 秒 | 0.784 / 0.791 秒 |
| ASR 开始到音频播放回调 | 1.506 / 2.317 秒 | 1.522 / 1.552 秒 |

其他记录：

| 指标 | 测量结果 |
|---|---|
| 单独 ASR 短 WAV 温态推理 | 约 0.285 秒 |
| 单独 ASR 峰值 reserved 显存 | 约 1832MiB |
| 完整驻留栈的整卡显存采样峰值 | 约 11822MiB |
| 进程 RSS 加总峰值 | 约 8GB，含可能重复计算的共享页 |
| 明亮条件下的 720p 摄像头交付 | 约 29.69FPS |
| 暗光条件下的摄像头交付 | 约 9.9FPS，虚拟输出约 30FPS |
| 强卡顿的完整冻结区间 | 8 段记录约 538–736ms |

这些延迟从表中标注的阶段起点计算，不是从开口到会议远端听到的总延迟。样本量较小，BGR 与 NV12 的结果可用于排查开销，不足以判断哪条路径整体更快。

原完整项目最近一次非 E2E 回归为 **891 项通过**；当前通用 TTS 分支完成了静态语法和配置检查，尚未重新进行完整运行验收。

## 当前限制

- ASR 为话语级识别，长句需要等待提交；连续输入可能形成播放积压。
- 长话语播放饥饿和 VB-CABLE 间歇性零插入问题仍在排查。
- 新麦克风起始触发的实机视频延迟，以及会议客户端接收效果，尚待完整验收。
- 日语敬体与术语处理可能拒绝部分表达，可在界面查看原因并调整词典。
- 首次安装需准备模型、参考音频和相应驱动；不同设备的性能会有差异。

## 项目结构

```text
voicebridge/                 桌面界面、音频路由、词典与摄像头
ort/                         服务、协议、模型后端与文件处理
tools/                       启动、模型下载与资源准备
config/                      配置与空的初始发音词典
licenses/                    第三方许可原文与来源
requirements-*.txt           按环境划分的依赖文件
gptsovits.lock.json           GPT-SoVITS 源码及模型校验清单
PROTOCOL.md                  WebSocket 事件协议
```

本地配置、录音、模型和运行缓存默认由 `.gitignore` 排除。

## 致谢与许可

本项目基于 [JacobLinCool/open-realtime-translate](https://github.com/JacobLinCool/open-realtime-translate)，原作者为 [@JacobLinCool](https://github.com/JacobLinCool)。VoiceBridge JP 在其语音翻译服务基础上增加了 Windows 桌面界面、音频路由、日语处理和摄像头联动功能。


主要第三方组件包括 GPT-SoVITS（MIT）、Qwen3-ASR（Apache 2.0）、OPUS-MT zh-ja（CC BY 4.0）、Qt/PySide6、pyvirtualcam（GPLv2）、OBS、FFmpeg 和 VB-CABLE，各自适用其许可。完整来源、版本及分发注意事项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，依赖清单见 [THIRD_PARTY_COMPONENTS.json](THIRD_PARTY_COMPONENTS.json)。
