# 智能保险业务助手

项目仓库：[InsureVoice-Agent](https://github.com/guaishouxiao732-code/InsureVoice-Agent)

![保险复杂咨询语音 Agent 目标方案流程图](9ab8d5ac-08ef-478f-acb4-c89c434cb1b5.png)

流程图展示完整目标方案；其中全双工自然插话、逐项证据核验、业务记录接口和真实人工工单属于后续目标，当前实现边界见下文。

基于 Qwen3-4B-Instruct-2507 的本地保险咨询原型，组合快慢路径协同、RAG、SQLite 上下文及离线 ASR/TTS，并提供电脑和手机自适应网页。

本仓库只发布项目源码、前端静态资源、测试和说明文档。**不包含任何模型权重、保险知识资料、向量索引、聊天记录或实际访问密码。** 克隆代码后需要自行准备这些运行资源；RAG 是检索增强生成，不是模型训练或微调。

## 当前功能与边界

- 自动快慢协同：本地规则识别意图、更新年龄/社保/被保险人等条件，必要时先澄清；简单问题进入快路径，复杂问题进入慢路径。两个路径共享一个 Qwen 实例，不是两个模型并行推理。
- 上下文：SQLite 保存会话、历史、来源元数据和轮次状态，支持恢复、删除及会话隔离。
- RAG：解析 PDF 文本层、Excel 和 Markdown，BGE 向量化，FAISS 检索，再将相关片段和问题交给 Qwen。
- 语音输入：点击开始录音，再次点击结束并发送；SenseVoice 离线识别，支持中文、英文及中英混合。
- 语音输出：CosyVoice 中文女声；首段流式合成，后续段在播放缓冲充足时使用完整合成以改善长句清晰度，不足时继续流式；仅朗读正文。英文缩写与单词由离线词典和词形自动区分，专有名词可配置覆盖读法。
- 文字提速：文字读取与语音合成分开执行，完整文字不再等待每段 TTS。普通问题优先简洁，不能通过硬截断省略必要条件。
- 手动停止：生成中、合成中、音频尚未播放完及历史回答重播时均可停止；立即清空待播音频并屏蔽迟到结果，后台安全释放任务资源。
- 回答依据：检索来源、文件位置、片段预览及协同条件放在独立面板，不要求模型在正文生成来源列表。
- 安全兜底：证据不足时不使用模型知识补充，提示待人工处理；尚未接入真实人工队列，不声称已转接成功。
- 网页：响应式布局、历史会话高亮与删除、可关闭的回答依据面板、固定密码和短期公网测试入口。

当前网页已移除边录边发送、持续监听及语音自动插话。手动停止**不等于全双工自然打断**。逐项证据支持校验、完整保险规则引擎、授权保单/理赔查询和真实人工交接仍待完善。

比赛的首响 ≤3 秒、复杂完整回答 ≤10 秒、准确率 ≥90% 和自然打断成功率 ≥95% 均需要独立验收；本仓库不宣称已全面达标。单个报价样例约1.74秒显示完整文字，不能替代语音首响、完整播报或 P95 统计。长答案播报可能超过10秒。

## 项目结构

```text
api.py                    Qwen 文本模型调用接口
context_manager.py        SQLite 多轮上下文与取消
cooperative.py            快慢路由、条件维护与澄清
rag.py                    资料解析、检索和证据门控
build_rag_index.py        构建知识索引
speech.py                 ASR/TTS Python 接口
voice_assistant.py        文字/语音流水线
cosyvoice_server.py        独立环境中的本机 TTS 服务
chat*.py                  命令行使用示例
http/                     网页服务、静态资源及网页测试
http/static/vendor/       本地 Markdown、公式和安全渲染资源
tests/                    核心功能回归测试
docs/                     设计思路和技术使用指南
environment.yml           主 Conda 环境
environment-cosyvoice.yml  独立 TTS Conda 环境
project_config.py         可配置路径与 TTS 环境发现
.env.example              不含秘密的配置示例
```

`live_voice.py` 及其单元测试属于保留的内部辅助代码，当前网页不提供连续语音入口。

## 新机器部署（Linux / NVIDIA GPU）

下列命令按顺序执行。默认分工为 Qwen 使用 GPU 1、CosyVoice 使用 GPU 0，ASR 和检索使用 CPU。先运行 `nvidia-smi` 检查设备；若设备数量或显存不同，必须调整配置，不能直接套用默认性能结果。

### 1. 获取代码与系统依赖

```bash
git clone https://github.com/guaishouxiao732-code/InsureVoice-Agent.git
cd InsureVoice-Agent
sudo apt-get update
sudo apt-get install -y ffmpeg poppler-utils git curl wget
```

系统需已安装 Conda 和支持所用 PyTorch CUDA 运行时的 NVIDIA 驱动。以上系统依赖命令用于 Ubuntu/Debian；Windows 或其他发行版需要对应安装方法。

### 2. 创建 Conda 环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --file environment.yml
conda env create --file environment-cosyvoice.yml
conda activate cosyvoice
python -m pip install --no-build-isolation openai-whisper==20231117
conda activate qwen3-vl
```

已有同名环境时，用 `conda env update --name qwen3-vl --file environment.yml` 更新主环境；不要反复创建或删除正在使用的环境。Whisper 单独安装是为了兼容其旧构建脚本，需先完成 TTS 环境依赖安装。

### 配置路径与设备

默认路径均相对于项目根目录，不依赖开发机器的用户名、微信目录或 Conda 安装位置。可以复制配置模板后编辑自己的 `.env`：

```bash
cp .env.example .env
```

在启动程序或构建索引之前，从项目根目录加载它：

```bash
set -a
source .env
set +a
```

每个新终端都要加载一次；Python 不会自动执行 `.env`。此文件是可信的本机 Shell 配置，不要执行来历不明的配置文件。实际 `.env` 被 Git 忽略，模板中没有真实密码。

| 环境变量 | 用途 / 默认路径 |
|---|---|
| `QWEN_MODEL_PATH` | Qwen 权重：`models/Qwen3-4B-Instruct-2507` |
| `ASR_MODEL_PATH` | SenseVoice INT8 权重目录 |
| `TTS_MODEL_PATH` | CosyVoice 权重：`models/speech/CosyVoice-300M-SFT` |
| `COSYVOICE_CODE_PATH` | 上游源码：`third_party/CosyVoice` |
| `COSYVOICE_PYTHON` | 独立 TTS 环境的 Python；空值时自动寻找同一 Conda 安装中的 `cosyvoice` 环境 |
| `RAG_EMBEDDING_MODEL` | BGE 权重：`models/bge-small-zh-v1.5` |
| `KNOWLEDGE_SOURCE` | 本机知识目录：`/home/scai/桌面/fyx/knowledge`；迁移机器后可覆盖 |
| `RAG_INDEX_DIR` | 索引：`data/rag/insurance` |
| `RAG_RATE_TRANSCRIPTION` | 可选人工核对费率转录路径 |
| `CONTEXT_DB_PATH` / `WEB_DB_PATH` | 命令行 / 网页数据库；默认为 `data/` 下不同文件 |
| `QWEN_DEVICE` / `TTS_DEVICE` | 网页设备：默认 `cuda:1` / `cuda:0` |
| `TTS_SERVER_URL` | 可选本机 CosyVoice 服务地址；通常留空，独立测试服务可设置为 `http://127.0.0.1:8767` |

配置应在导入模型模块之前设置；启动后修改变量需要重启对应服务。Python 接口中显式传入的模型路径优先于默认值。`.env` 中有空格的绝对路径需加引号。

### 3. 单独准备模型（不会上传到 GitHub）

需要以下目录：

| 用途 | 模型与获取来源 | 放置目录 |
|---|---|---|
| 文本生成 | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `models/Qwen3-4B-Instruct-2507` |
| 中文检索 | [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | `models/bge-small-zh-v1.5` |
| 离线识别 | [sherpa-onnx SenseVoice INT8 2024-07-17](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html) | `models/speech/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` |
| 语音合成 | [CosyVoice-300M-SFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) | `models/speech/CosyVoice-300M-SFT` |

可从 Hugging Face 下载 Qwen、BGE 和 CosyVoice；SenseVoice 使用 sherpa-onnx 官方转换包：

```bash
conda activate qwen3-vl
python - <<'PY'
from huggingface_hub import snapshot_download
for repo_id, destination in [
    ('Qwen/Qwen3-4B-Instruct-2507', 'models/Qwen3-4B-Instruct-2507'),
    ('BAAI/bge-small-zh-v1.5', 'models/bge-small-zh-v1.5'),
    ('FunAudioLLM/CosyVoice-300M-SFT', 'models/speech/CosyVoice-300M-SFT'),
]:
    snapshot_download(repo_id=repo_id, local_dir=destination)
PY
mkdir -p models/speech
wget -P models/speech https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar -xjf models/speech/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2 -C models/speech
```

这些下载会占用数 GB 磁盘和网络流量；如已有完整权重目录，可直接从原主机复制，无需重复下载。ASR 目录必须有 `model.int8.onnx` 和 `tokens.txt`；TTS 目录必须有 `cosyvoice.yaml`、`llm.pt`、`flow.pt`、`hift.pt` 和 `spk2info.pt`。请遵守各模型自身的使用许可。

### 4. 获取 CosyVoice 上游代码

```bash
mkdir -p third_party
git clone https://github.com/QwenAudio/CosyVoice.git third_party/CosyVoice
git -C third_party/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C third_party/CosyVoice submodule update --init --recursive
```

`third_party/`、早期 Qwen3-VL 代码和研究参考仓库不随源码包发布；当前文本模型接口不依赖早期 Qwen3-VL 目录。CosyVoice 运行时依赖上面获取的源码及 Matcha-TTS 子模块。

### 5. 准备知识库并构建索引

将自己有权使用的 PDF、Excel 或 Markdown 放在本机资料目录。不要公开真实客户资料、内部条款资产或业务记录。

```bash
conda activate qwen3-vl
python build_rag_index.py
```

本机默认从 `/home/scai/桌面/fyx/knowledge` 读取资料；迁移机器后通过 `KNOWLEDGE_SOURCE` 或 `--source` 指定新目录。默认输出为 `data/rag/insurance`。读取索引时会校验源文件和哈希，因此知识原文件也必须保留在该机器上；迁移机器后应重新构建索引。更改知识库目录后必须重新构建索引，否则现有索引仍指向旧路径。

图片型 PDF 的数字表格不能仅靠文本提取保证识别。若复现保险费率场景，请自行准备经授权、人工核对的 `data/rag/insurance_fee_rates.md`，与原始费率 PDF 一起核对并重建索引。仓库不附带这份业务资产，也不保证换一套资料后仍能得到原示例报价。

### 6. 启动本机 TTS

在第一个终端中，从项目根目录执行，并保持此终端运行：

```bash
conda activate cosyvoice
CUDA_VISIBLE_DEVICES=0 python cosyvoice_server.py --host 127.0.0.1 --port 8766 --model models/speech/CosyVoice-300M-SFT --physical-device cuda:0
```

TTS Python 会自动寻找同一 Conda 安装中的 `cosyvoice` 环境。如果使用自定义环境目录，在主环境启动前设置 `COSYVOICE_PYTHON=/absolute/path/to/cosyvoice/bin/python`，或在接口中显式配置 `TTSClient(server_python=...)`。手动启动 TTS 时，也要先从根目录加载自己的 `.env`；命令行 `--model` 路径会覆盖 `TTS_MODEL_PATH`。切换 TTS GPU 时，需要让物理设备、服务端口和网页 `TTS_DEVICE` 一致；默认端口为 `8766 + GPU编号`。

### 7. 启动网页

在第二个终端中，从项目根目录执行：

```bash
conda activate qwen3-vl
cd http
python app.py
```

本机浏览器打开 `http://127.0.0.1:8000`。`/healthz` 中 `model` 为 `ready` 才表示模型已加载并预热；页面可先打开，但加载期间提问会等待。不要同时启动多个占用 8000 端口的网页进程。

可选设备配置：

```bash
QWEN_DEVICE=cuda:1 TTS_DEVICE=cuda:0 python app.py
```

单 GPU 机器可将两个设备都设为 `cuda:0`，但必须保证显存足够，并重新验收资源竞争带来的延迟。当前服务主要用于单用户原型，不是多租户生产平台。

## 手机与异地访问

如果只是在另一台设备使用，**不需要安装模型或运行指令**，用浏览器访问服务端产生的 HTTPS 公网网址即可。模型依然在主机运行，主机、TTS、网页进程和隧道需保持运行。

短期测试可使用 Cloudflare Quick Tunnel。Linux x86_64 机器可先获取客户端：

```bash
mkdir -p http/.tools
curl --fail --location https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 --output http/.tools/cloudflared
chmod +x http/.tools/cloudflared
```

在本机创建 `http/.web.env`，填写一行 `WEB_PASSWORD=你自行设置的强密码`，替换示例值并限制该文件权限。不要把真实密码写入 README、提交到 Git 或发给无关人员。启动器会读取此文件；未配置时只会临时生成密码，不保证下一次启动相同。

停止之前的 `app.py`（网页终端按 Ctrl+C，不停止独立 TTS），再运行：

```bash
conda activate qwen3-vl
cd http
python public.py
```

终端会显示 `https://....trycloudflare.com` 和访问密码。远程浏览器必须使用 HTTPS 并允许麦克风；同一主机内的会话目前共享，不提供独立用户登录隔离。重启隧道后网址会变化，旧网址可能失效；长期部署需要另行配置固定域名、稳定隧道或服务器。网页更新后电脑按 Ctrl+Shift+R 强制刷新。

## 调用示例

普通流式文本接口，无需准备 RAG 和语音模型：

```python
from api import Qwen3Client

client = Qwen3Client(device='cuda:1')
messages = [{'role': 'user', 'content': '请简要介绍你自己。'}]
for text in client.stream_chat(messages, max_new_tokens=256):
    print(text, end='', flush=True)
```

多轮、RAG、快慢协同及语音接口的详细示例见 [技术使用指南](docs/技术使用指南.md)。网页操作和交互规则见 [网页说明](http/README.md)。[解题思路](docs/保险复杂咨询语音Agent_解题思路.md) 描述比赛目标和后续方案，不能作为当前已实现或全部达标的证明。

## 验证与测试

主环境安装后，以下回归测试不需要下载权重或启动真实模型：

```bash
conda activate qwen3-vl
python -m unittest discover -s tests -v
python -m unittest discover -s http -p test_app.py -v
node http/test_frontend.cjs
```

最后一条需要 Node.js。其中3项保留的旧 VAD 实测需要额外 Silero 权重/私有录音，资源缺失时自动跳过，不计为通过的实测。其余回归覆盖配置路径、上下文、取消、RAG 兜底、快慢条件维护、录音交互和前端停止。测试数量以当前运行结果为准；通过回归不等于保险业务准确率和真机语音指标达标。

可选浏览器测试需要 Playwright/Chromium、已预热的完整服务及相应业务资料。使用独立测试服务/数据库，不要直接把生产会话当作测试数据：

```bash
TEST_WEB_URL=http://127.0.0.1:8011 python http/test_speed_stop_browser.py
TEST_WEB_URL=http://127.0.0.1:8011 python http/test_cooperative_browser.py
TEST_WEB_URL=http://127.0.0.1:8011 TEST_MIC_WAV=/absolute/path/to/test.wav python http/test_recording_browser.py
```

8011 仅为独立测试服务的示例端口，不会由上述命令自动启动。录音测试 WAV 应包含单句语音后足够静音；报价和协同测试依赖原业务样例，换知识库时需调整验收问题。另应实测完整文字、首个实际可听音频、全部音频发送完成及实际播放结束，不能混用计时口径。

## 安全与发布规则

`.gitignore` 排除 `models/`、权重格式、`data/`、`knowledge/`、真实环境变量文件、数据库、日志、录音、缓存、上游仓库和本地打包产物。发布前仍应审查文件清单；忽略规则不能清除已经提交的敏感历史。

本项目不包含真实人工客服接口。检索门控仅作初筛，回答依据展示的资料不代表每条结论已经核验。TTS 停止是安全取消，不会强制杀死 CUDA 内核；下一轮可能短暂等待资源释放。公网入口仅面向受控比赛测试，正式部署前需增加身份与会话隔离、权限、审计及安全加固。

## 上游项目

- [Qwen](https://github.com/QwenLM/Qwen3)：文本模型生态。
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：离线语音识别运行时。
- [CosyVoice](https://github.com/QwenAudio/CosyVoice)：语音合成源码和预置音色。
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)、[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)：检索方案参考，不是运行必需服务。
- 前端本地资源为 Marked、DOMPurify 和 KaTeX；请保留其版权及许可说明。模型和第三方资源适用各自上游许可，本仓库没有替它们重新授权。
