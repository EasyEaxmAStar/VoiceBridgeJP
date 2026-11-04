# 第三方组件、模型与再分发提醒

核查日期：2026-11-04。以下是针对本源码目录的来源记录与发布提醒，
不是覆盖整个供应链的法律保证。明确的源码授权阻塞项先看
[LICENSE_STATUS.md](LICENSE_STATUS.md)。

## 代码来源与本分支修改

基础服务来自 [JacobLinCool/open-realtime-translate](https://github.com/JacobLinCool/open-realtime-translate)。
项目级授权未确认。本地ZIP无法提供精确的上游提交对应关系。
Windows路由、GUI、词典覆盖、语体检查、队列和摄像头功能是在该基础上继续实现。
本分支另将个人声线选择改为官方预训练适配、增加公共参考配置窗口、移除个人词表和
私有预热录音依赖。原有协议与后端扩展源码保留。不能把这些修改描述为上游官方版本。

## 默认运行链路

| 组件 | 来源及许可依据 | 状态与提醒 |
|---|---|---|
| GPT-SoVITS源码 | [固定提交LICENSE](https://github.com/RVC-Boss/GPT-SoVITS/blob/b4570b876f5c086819ba84e3215472733c6a7b70/LICENSE) | MIT，版权RVC-Boss。已取回原文到licenses/，复制或分发实质部分时保留版权与许可全文 |
| 官方GPT-SoVITS预训练文件 | [固定revision模型卡](https://huggingface.co/lj1995/GPT-SoVITS/blob/8f8857563c040d542de108a2c18b0425abd6c9c5/README.md) | 模型卡标MIT；实际使用s1v3和v2Pro/s2Gv2ProPlus，不附权重。清单还包含说话人编码器、Chinese HuBERT与RoBERTa文件；仓库标签不代替辅助资源的独立声明核对 |
| Qwen3-ASR-0.6B | [模型卡](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)、[源码许可](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Apache2.0。分发时保留许可及适用NOTICE/归属，注明修改；本包不包含模型 |
| OPUS-MT zh-ja | [准确模型卡](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zh-ja) | CC BY4.0，作者/发布方Helsinki-NLP、OPUS-MT项目。保留作者与模型链接、许可链接、必要归属和修改说明；不能按其他OPUS模型的许可推断此模型 |
| PyTorch/torchaudio | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)、[audio LICENSE](https://github.com/pytorch/audio/blob/main/LICENSE) | 依具体wheel及所带第三方库保留声明；NVIDIA运行库另有条款。不随本包提供wheel |
| Transformers/Accelerate/Hugging Face Hub | [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)、[Accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE)、[Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) | 按实际版本的Apache2.0及其NOTICE处理；模型许可不是工具库许可 |
| SentencePiece | [LICENSE](https://github.com/google/sentencepiece/blob/master/LICENSE) | Apache2.0，分发时保留原声明 |
| NumPy/SciPy及其他直接依赖 | [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt)、[SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | 除项目主许可外还可能有BLAS等原生库声明；完整二进制审计不能仅看包名 |

`gptsovits.lock.json`记录源码/模型固定revision、实际9个文件及SHA256。
官方模型是条件式TTS，不含一段任意声音的使用授权；声音所有者、录音制作人、
表演/隐私及录音本身的权利需由使用者另行解决。参考WAV不得因模型MIT就公开上传。
本包不提供开发者声音、训练数据、嵌入或个人模型。

## GUI、音频、视频和系统组件

| 组件 | 官方来源 | 提醒 |
|---|---|---|
| PySide6/Qt Essentials/Shiboken | [Qt开源许可义务](https://www.qt.io/development/open-source-lgpl-obligations)、[PySide许可](https://doc.qt.io/qtforpython-6/licenses.html) | LGPLv3/GPLv3/商业选项依模块和发行方式而异。将来打包Qt库需提供适用文本、允许库替换/调试并满足相关源码要求；本次不分发DLL/wheel |
| sounddevice / PortAudio | [sounddevice LICENSE](https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE)、[PortAudio](https://www.portaudio.com/license.html) | 包与所带DLL均保留各自许可/版权；Windows设备驱动不因Python包许可而变为开源 |
| SoundFile / libsndfile | [SoundFile](https://github.com/bastibe/python-soundfile/blob/master/LICENSE)、[libsndfile](https://github.com/libsndfile/libsndfile/blob/master/COPYING) | Python包装与原生音频库分别核对；libsndfile为LGPL体系，不要统一写成MIT |
| python-soxr / libsoxr | [python-soxr](https://github.com/dofuuz/python-soxr)、[libsoxr](https://sourceforge.net/projects/soxr/) | 原生重采样库具有自己的LGPL许可条件 |
| FFmpeg | [官方法律说明](https://ffmpeg.org/legal.html) | LGPL2.1+为基础，启用GPL组件等会改变整份构建的条件；确认具体exe的configure选项。ffmpeg-python包装的许可不覆盖exe。不附exe |
| OpenCV | [4.x LICENSE](https://github.com/opencv/opencv/blob/4.x/LICENSE) | 现代版本Apache2.0；wheel中的编解码器/依赖另查 |
| pyvirtualcam0.15.0 | [固定版本LICENSE](https://github.com/letmaik/pyvirtualcam/blob/v0.15.0/LICENSE) | GPLv2。许可原文已保存。不能因摄像头独立进程就声称主程序免除copyleft；分发组合或二进制前需确认适用范围 |
| OBS Virtual Camera | [OBS COPYING](https://github.com/obsproject/obs-studio/blob/master/COPYING) | OBS为GPLv2或后续版本；不附OBS/驱动安装包，用户从官方安装 |
| VB-CABLE | [VB-Audio许可](https://vb-audio.com/Services/licensing.htm) | Donationware，不是MIT/Apache开源驱动。专业使用与再分发看厂商条款；需要捆绑分发时另取得许可，不转发下载包或付费链接 |
| Windows / NVIDIA CUDA | [Microsoft条款](https://www.microsoft.com/licensing/terms/)、[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) | 平台和运行库有独立条款，本项目没有授予它们的复制权 |

## 日语/英语文本前端与资源

GPT-SoVITS依赖pyopenjtalk-plus/Open JTalk、其日语词典、g2p-en、NLTK及CMU发音词典/tagger，
以及jieba、pypinyin、cn2an、fast-langdetect/其语言识别模型等。它们的Python包、词典与
模型可能具有不同许可证。源码分发不包含下载后的词典或模型。

- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus)：核查其随包Open JTalk/词典说明，不应把前端的整个资源集合等同一个MIT声明。
- [NLTK](https://github.com/nltk/nltk/blob/develop/LICENSE.txt)为Apache2.0；[NLTK数据集许可索引](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md)明确不同语料分别适用条款。
- `setup_gptsovits_nltk.py`只取固定数据版本的cmudict和两个tagger；上线或捆绑其下载物时需保存该版本index.xml及每包许可，不能只附NLTK软件许可证。
- [g2p-en](https://github.com/Kyubyong/g2p)、[jieba](https://github.com/fxsjy/jieba)、[fast-langdetect](https://github.com/LlmKira/fast-langdetect)及其他包的精确版本线索见组件清单；本轮未把所有传递资源逐一认定为已完成二进制再分发审计。

## 保留但默认不启动的替代后端

源码保留Qwen翻译恢复以及Whisper、TranslateGemma、Qwen3-TTS、Kokoro、Supertonic等
原有后端扩展，不包含这些模型或默认启用它们。若主动启用：

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)、Qwen3-TTS模型发布页各自标Apache2.0；应核对实际下载revision。
- [TranslateGemma](https://huggingface.co/google/translategemma-4b-it)适用Google Gemma条款，不能用Apache/MIT代替。
- [Whisper](https://github.com/openai/whisper/blob/main/LICENSE)与faster-whisper/Systran模型分别核对其发布方条款。
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)发布页标Apache2.0；[Supertonic-3](https://huggingface.co/Supertone/supertonic-3)有OpenRAIL-M使用/分发要求。
- macOS say使用系统声音，受Apple条款约束，不属于此Windows默认配置。

## 发布文件与未确认事项

`THIRD_PARTY_COMPONENTS.json`列出当前requirements文件中的153个直接/锁定依赖声明，
并保存本机元数据可取得的版本、许可字段和官方项目链接，不附本机路径。
它不是一个已证实无遗漏的传递依赖SBOM，也不把模糊/缺失元数据当作许可通过。
`licenses/SOURCES.json`记录实际取回的关键许可原文、URL和SHA256。

正式发布前仍需：补齐基础项目授权；决定并核对适当的修改代码许可；核实GPL组合关系；
若制作安装包，审计实际wheel、DLL、FFmpeg、驱动、字体/词典/模型的再分发要求。
不要将README的历史个人版性能或891项测试结果说成此通用分支已重新验收的结果。
