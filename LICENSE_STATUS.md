# 项目许可状态：公开再分发尚未确认

核查日期：2026-11-04。本文件是状态说明，不是许可证，也不授予新的权利。

本项目继承 [JacobLinCool/open-realtime-translate](https://github.com/JacobLinCool/open-realtime-translate)
的源码。开发用副本来自ZIP，没有Git历史；本地未找到项目根许可证。
本轮核查了上游公开仓库、README的“Model licenses”部分、pyproject页面，
以及原始`main/LICENSE`、`LICENSE.md`、`LICENSE.txt`、`COPYING`、`COPYING.md`
五个常见许可路径（均404）；上游`pyproject.toml`可读取但没有license字段。
GitHub仓库API请求被限流（403）。
模型许可表只说明第三方模型条款，并不构成对该仓库源码的项目级授权。

**结论：尚未取得可核实的继承源码公开再分发许可。**
不能据此断言作者永远不允许，也不能因仓库公开或可下载而视为已授权。
本目录没有添加一个会覆盖上游权利的MIT/Apache/GPL根许可证。

公开上传GitHub之前，需要取得上游明确许可证或适用的书面授权，核实其覆盖
当前继承版本及修改版公开发布的范围，并保留版权与修改声明。不要把
GPT-SoVITS的MIT许可证或Qwen的Apache许可证当作该项目源码的授权。
目前没有代替你联系作者、发送消息或进行公共上传。

即使上游授权补齐，仍需处理摄像头GPL组件、Qt、模型、驱动和二进制构建的
各自要求。仅发布源码、由用户自行安装依赖，可以避免直接捆绑许多二进制，
但不等于自动解除全部组合/派生作品义务。

详细组件来源及已确认/未确认范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
