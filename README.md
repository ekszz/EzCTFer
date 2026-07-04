# EzCTFer

基于LangGraph的多LLM CTF解题框架，面向Web、Pwn、Reverse、Crypto、Misc等场景，提供双模型协作“结对解题”、MCP工具扩展、Skills、RAG、HITL(Human in the Loop)和实时Web进度查看能力。

![MONITOR](img/monitor.jpg)

## 核心特性

### 双模型结队解题模式

支持2个LLM同时并行解题，线程之间可以共享关键发现、同步情报并协作推进；任一线程拿到突破口后，另一线程会在后续轮次中继承这些发现继续深入。

### 支持常见API协议、主流模型与代理

支持OpenAI-compatible、Anthropic-compatible等常见API接入方式，可方便接入GPT、DeepSeek、GLM、MiniMax、Qwen等主流模型；每个LLM都可以单独配置代理。

### 支持MCP扩展，覆盖二进制与移动端分析

内置MCP客户端，支持`stdio`/`sse`协议；自带IDA与JADX扩展，适合二进制、JAR、APK等题型的辅助分析。

### 支持Skills，并内置PUA Skill

支持从`skills`目录动态加载Skill，并在解题过程中通过工具按需调用；仓库默认内置`pua` Skill。

### 支持HITL(Human in the Loop)

支持人类在解题过程中实时介入，通过Web监控台向指定线程注入提示、经验、修正方向或额外指令。

### 支持RAG知识管理

可选启用本地RAG知识检索能力，为漏洞分析、利用思路和题型知识补充上下文。

### 支持Web监控台

运行时默认启动Web监控台，可实时查看轮次进度、对话内容、工具调用、重大发现和最终Flag提交情况。

### 友好支持CLI、无人工交互与Docker

支持命令行直接传入题面、`--quiet`无人工交互运行，以及Docker部署方式，便于批量化、自动化或远程运行。

## 架构总览

### 结队解题模式

线程之间通过「共享发现」协同推进，人类（HITL）可随时通过 Web 监控台向指定线程注入提示或指令：

```text
+-------------------------------+          +-------------------------------+          +-------------------------------+
|           Thread 1            |          |         Shared Findings       |          |           Thread 2            |
|                               |          |                               |          |                               |
|  Round 1                      |          |  Key discoveries              |          |  Round 1                      |
|  - LLM call                   |  write   |  collected from both threads  |  write   |  - LLM call                   |
|  - Dialogue                   |--------->|                               |<---------|  - Dialogue                   |
|  - tool_call                  |          |  Read at the beginning of     |          |  - tool_call                  |
|  - Tool result                |          |  each new round               |          |  - Tool result                |
|  - Summary / Reflection       |          |                               |          |  - Summary / Reflection       |
|          |                    |<---------|  Shared context for both      |--------->|          |                    |
|          v                    |   read   |  threads                      |   read   |          v                    |
|                               |          |                               |          |                               |
|  Round 2                      |          +-------------------------------+          |  Round 2                      |
|  - LLM call                   |                                                     |  - LLM call                   |
|  - Dialogue                   |                                                     |  - Dialogue                   |
|  - tool_call                  |          +-------------------------------+          |  - tool_call                  |
|  - Tool result                |<---------|    Human In The Loop (HITL)   |--------->|  - Tool result                |
|  - Summary / Reflection       |  inject  |     Hint / Guide / Command    |  inject  |  - Summary / Reflection       |
|          |                    |          +-------------------------------+          |          |                    |
|          v                    |                                                     |          v                    |
|  ...                          |                                                     |  ...                          |
|                               |                                                     |                               |
|  submit_flag (via tool)       |                                                     |  submit_flag (via tool)       |
+---------------+---------------+                                                     +---------------+---------------+
                |                                                                                     |
                | tool_call                                                                           | tool_call
                +------------------------------------------+------------------------------------------+
                                                           |
                                                           v
            +-------------+   +-------------+   +-------------+   +--------------------------------------+
            |     MCP     |   |   Skills    |   |     RAG     |   |             Local Tools              |
            |  - IDA pro  |   |  - PUA      |   |  - Writeups |   |  - python_exec / python_pip          |
            |  - Jadx     |   |  - ...      |   |  - Payloads |   |  - command execution                 |
            |             |   |             |   |  - ...      |   |  - file read / write / dir listing   |
            |             |   |             |   |             |   |  - HTTP request                      |
            |             |   |             |   |             |   |  - ...                               |
            +-------------+   +-------------+   +-------------+   +--------------------------------------+
```

### 探索图模式

启动方式：`uv run ezctfer --graph`（与 `--dual-thread` 互斥）。图模式把解题过程建模成一张**有向图**，由两个专职角色协作，按"图扩展逐步探索"的方式向前推进。

#### 设计思想：图扩展逐步探索

- **节点 = 已有发现**，**边 = 探索方向**。整张图从唯一的 `ROOT`（题目描述）出发，按 `发现─>探索─>新发现` 的方式**增量生长**。
- **不做一次性全局规划**：每确认一条新发现，就基于全部已知洞察**重新规划**待探索方向——保留有效方向、删除已失效方向、补充新方向。
- 这种"边走边看"的增量扩展天然具备**纠偏能力**：发现路径偏离，下一轮就会自动调整优先级与方向，避免在错误路径上空耗。

#### 角色分工

| 角色 | 职责                           | 可用工具 | 产出                            |
| --- |------------------------------| --- |-------------------------------|
| **Reasoner（推理者）** | 只思考、不操作：读整图快照 → 反思进展 → 重规划方向 | 仅 `submit_missions` | 一组带优先级（1-10）的 Mission，替换所有 pending |
| **Explorer（探索者）** | 只操作、不规划：领取一个探索方向 → 工具落地验证    | 命令/文件/HTTP/python 等 + `submit_insight` / `submit_flag` | 一条发现，或直接提交 flag               |

#### 运行机制

1 个 Reasoner 与最多 2 个并发 Explorer 围绕同一张线程安全的 `TaskGraph` 协作；Reason 结束后立即调度空闲 Explorer，每个 Explorer 收尾后又触发下一次 Reason，循环直至找到 flag：

```text
 ┌──────────────────────── 共享 TaskGraph（线程安全） ────────────────────────┐
 │   ROOT ──Mission──▶ Insight ──Mission──▶ Insight ──Mission──▶ Insight …    │
 └────▲────────────────────────────────────────────────────▲──────────────────┘
      │ submit_missions                            submit_insight
      │ （替换 pending）                        （收尾边 + 新 Insight）
 ┌────┴─────────┐                                ┌─────────┴────────┐
 │   Reasoner   │ Reason结束 ⇒调度Explorer       │   Explorer ×2    │
 │ 读图→反思→   │ ─────────────────────────▶     │ 取最高优先级     │
 │ 重规划(只想) │ ◀─────────────────────────     │ pending→工具验证 │
 └──────────────┘  结束 ⇒触发下一次Reason        └──────────────────┘
```

#### 图示例

下图是一条主路径的扩展过程（边 = Mission，标注优先级）。`concluded` 的边已产出洞察；`exploring` 正被某个 Explorer 走；`pending` 等待被领取。每确认一个新 Insight，就会从它长出新的 Mission：

```text
        +-----------+
        |   ROOT    |   题目描述（初始唯一信息）
        +-----+-----+
              |  M-001 (p:9)  侦察首页
              +------------[ M-002 (p:5) 路径扫描 ]--> (pending)  待领取
              v
        +-----------+
        |   M-001   |   "登录页在 /login"
        +-----+-----+
              |  M-003 (p:8)  注入测试
              +-------[ M-005 (p:4) ]--> (exploring)  某 Explorer 正在走
              v
        +-----------+
        |   M-003   |   "登录接口存在 SQL 注入"   ← concluded
        +-----+-----+
              |  M-004 (p:9)  利用注入读数据
              v
        +-----------+
        |   M-006   |   "flag{...}"   --> submit_flag  ✅
        +-----------+
```

## 快速开始

### 1. 安装uv

```bash
pip install uv
```

### 2. 安装项目依赖

```bash
uv sync
```

### 3. 准备配置文件

```bash
cp .env.example .env
cp mcp.json.example mcp.json
```

### 4. 配置 `.env`

`.env.example`已包含完整注释。下面给一个适合双模型协作的最小示例：

```env
LLM_1_NAME=gpt
LLM_1_API_KEY=your-api-key-here
LLM_1_API_URL=https://api.openai.com/v1
LLM_1_MODEL=gpt-5.4
LLM_1_TIMEOUT=120
LLM_1_API_TYPE=openai
LLM_1_EXTRA={"use_responses_api":true,"use_previous_response_id":false,"reasoning":{"effort":"high","summary":"auto"}}

LLM_2_NAME=deepseek
LLM_2_API_KEY=your-deepseek-api-key-here
LLM_2_API_URL=https://api.deepseek.com
LLM_2_MODEL=deepseek-reasoner
LLM_2_TIMEOUT=120
LLM_2_API_TYPE=deepseek
# LLM_2_PROXY=http://127.0.0.1:7890

DUAL_THREAD_0_LLM=1
DUAL_THREAD_1_LLM=2
MAX_ITERATIONS=200
MAX_ROUNDS=10
```

说明：

- `LLM_1`、`LLM_2`、`LLM_3`...的序号可以不连续，程序会按序号从小到大加载。
- `LLM_{n}_PROXY`只作用于对应模型；未配置时保持`httpx`默认代理行为。
- `LLM_{n}_EXTRA`必须是JSON对象，可用于透传`reason`、`reasoning`、自定义请求头或其他服务商扩展字段。
- 如果未指定`SINGLE_THREAD_LLM`、`DUAL_THREAD_0_LLM`、`DUAL_THREAD_1_LLM`，程序会从已配置模型中随机选择。

### 5. 配置MCP（可选）

如果需要逆向或移动端辅助工具，请按需编辑`mcp.json`。默认字段和示例请直接查看仓库中的[mcp.json.example](mcp.json.example)或[mcp.json](mcp.json)，这里不再重复粘贴默认值。

- `--ida`不带参数时，对应启用`ida_pro_mcp`。
- `--jadx`不带参数时，对应启用`jadx_mcp`。
- `--ida ARGS`会先在本地启动`idalib-mcp`服务，再启用`idalib_mcp`。
- `--jadx TARGET`会先执行`jadx-gui TARGET`，再启用`jadx_mcp`。

## 配置文件说明

### LLM相关变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_{n}_NAME` | 是 | 模型名称标识，用于日志和路由显示 |
| `LLM_{n}_API_KEY` | 是 | API密钥 |
| `LLM_{n}_API_URL` | 是 | API基础地址 |
| `LLM_{n}_MODEL` | 是 | 模型名 |
| `LLM_{n}_TIMEOUT` | 否 | 请求超时时间，默认 `120` 秒 |
| `LLM_{n}_API_TYPE` | 否 | API类型：`openai`、`anthropic`、`deepseek`，默认`openai` |
| `LLM_{n}_PROXY` | 否 | 当前LLM专用代理地址，例如`http://127.0.0.1:7890` |
| `LLM_{n}_EXTRA` | 否 | 额外模型初始化参数，必须是JSON对象 |

### `LLM_{n}_EXTRA` 说明

`LLM_{n}_EXTRA`会与代码中的默认参数递归合并，适合补充或覆盖底层客户端参数。

- 仅支持JSON对象。
- 嵌套字段会递归合并，例如`default_headers`、`model_kwargs`、`reasoning`。
- 某个字段设为`null`时，会删除默认值。
- OpenAI-compatible客户端可通过`use_responses_api`、`use_previous_response_id`、`reasoning`等字段切换Responses API能力。
- Anthropic/DeepSeek/其他兼容服务也可以通过`EXTRA`继续透传底层参数。

补充说明：

- 未配置`LLM_{n}_PROXY`时，对应LLM会保持`httpx`默认行为，可继续继承系统`HTTP_PROXY`/`HTTPS_PROXY`。
- 使用`deepseek-reasoner`一类推理模型时，建议设置`LLM_{n}_API_TYPE=deepseek`。
- 只支持`/v1/responses`的OpenAI-compatible服务，保持`API_TYPE=openai`，并在`EXTRA`中启用`use_responses_api`。

### 应用级变量

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `MAX_ITERATIONS` | 否 | `120` | LangGraph图执行步数上限；一次工具调用通常会消耗约2步，建议200-300 |
| `MAX_ITERATIONS_GRAPH` | 否 | 复用 `MAX_ITERATIONS` | 探索图模式单轮对话上限；优先级 `MAX_ITERATIONS_GRAPH` > `MAX_ITERATIONS` > `100` |
| `MAX_ROUNDS` | 否 | `10` | LLM切换轮数上限，建议5-10 |
| `MAX_ROUNDS_GRAPH` | 否 | 复用 `MAX_ROUNDS` | 探索图模式思考最大轮数；优先级 `MAX_ROUNDS_GRAPH` > `MAX_ROUNDS` > `10` |
| `MCP_CONFIG_FILEPATH` | 否 | `./mcp.json` | MCP配置文件路径 |
| `RAG_DATA_ROOT` | 否 | `./rag` | RAG 模块数据根目录，内部约定使用`data/`、`db/`、`models/`三个子目录 |
| `SKILLS_SCAN_PATH` | 否 | `./skills` | 额外扫描的 skills 目录（支持相对/绝对路径，相对当前工作目录）；其下每个子目录是一个 skill。未配置时回退扫描当前工作目录下的`skills/`。内置 skills 始终加载；同名 skill 以本目录覆盖内置 |
| `SINGLE_THREAD_LLM` | 否 | 随机 | 单线程模式固定使用的`LLM_X`序号 |
| `DUAL_THREAD_0_LLM` | 否 | 随机 | 双线程模式下线程1使用的`LLM_X`序号 |
| `DUAL_THREAD_1_LLM` | 否 | 随机 | 双线程模式下线程2使用的`LLM_X`序号 |

### RAG目录约定

当启用RAG时，程序会从`RAG_DATA_ROOT`读取和写入以下目录结构：

- `data/`：`--init-rag`使用的知识库原始资料目录
- `db/`：生成的索引和存储文件目录
- `models/`：本地embedding模型目录，默认使用`models/all-MiniLM-L6-v2`

如果未配置 `RAG_DATA_ROOT`，默认使用当前项目目录下的 `rag/`。

## 运行

推荐直接使用项目脚本入口：

```bash
uv run ezctfer
```

如果更习惯模块方式，也可以使用：

```bash
uv run python -m ezctfer
```

常用启动方式：

```bash
# 标准模式
uv run ezctfer

# 指定配置文件
uv run ezctfer --config .env.ctf

# 直接通过命令行传入题目描述
uv run ezctfer --prompt "分析这个Web题并尝试拿到flag，靶机地址是http://123.45.67.8:8080/"

# CLI无人工交互模式
uv run ezctfer --prompt "分析这个Web题并尝试拿到flag，靶机地址是http://123.45.67.8:8080/" --quiet

# 启用双线程结队模式
uv run ezctfer --dual-thread

# 初始化RAG知识库
uv run ezctfer --init-rag

# 启用RAG
uv run ezctfer --rag

# 启用IDA Pro MCP
uv run ezctfer --ida

# 启用JADX MCP
uv run ezctfer --jadx

# 启动JADX GUI，并把目标文件传给`jadx-gui`
uv run ezctfer --jadx "samples/app.apk"

# 启动idalib MCP，并把目标文件路径附加给idalib-mcp
uv run ezctfer --ida "samples/challenge.exe"

# 可组合使用
uv run ezctfer --rag --dual-thread --quiet --jadx
```

运行期间会默认启动Web监控页面：`http://localhost:8000`。

## 命令行参数说明

| 参数 | 说明 |
| --- | --- |
| `-h`, `--help` | 显示帮助信息 |
| `--config PATH` | 指定 `.env` 配置文件路径 |
| `--demo` | 运行内置演示题目 |
| `--ida [ARGS]` | 不带参数时启用`mcp.json`中的`ida_pro_mcp`；带`ARGS`时改为启动本地`idalib-mcp`服务，并将`ARGS`按空格拆分后附加到`idalib-mcp`命令后面 |
| `--jadx [TARGET]` | 不带参数时启用`mcp.json`中的`jadx_mcp`；带`TARGET`时先执行`jadx-gui TARGET`，再启用`jadx_mcp` |
| `--rag` | 启用本地RAG检索工具，并向Agent注入`retrieve_knowledge`工具 |
| `--init-rag` | 忽略其它参数，仅执行知识库初始化 |
| `--dual-thread` | 启用双线程并行解题，并基于情报共享协同推进 |
| `--debug` | 输出debug级别日志 |
| `--prompt TEXT` | 直接传入题目描述，跳过交互式多行输入 |
| `--quiet` | 找到flag时自动确认；程序结束后等待10秒自动退出 |
| `--no-writeup` | 禁用writeup生成 |

## 内置工具

| 工具 | 说明 |
| --- | --- |
| `execute_command` | 执行系统命令 |
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件 |
| `list_directory` | 列出目录内容 |
| `http_request` | 发送HTTP请求 |
| `python_exec` | 在沙箱Python环境中执行脚本 |
| `python_pip` | 在沙箱Python环境中安装Python包 |
| `record_finding` | 记录重要发现，供后续轮次或线程复用 |
| `submit_flag` | 提交找到的flag |
| `get_skill` | 获取已安装Skill的方法论与内容 |
| `retrieve_knowledge` | 从本地索引检索知识库，仅在启用`--rag`后可用 |

## 致谢

- PUA Skill来源于开源项目[pua](https://github.com/tanweai/pua)。
- 感谢相关项目作者提供的实现思路与基础能力。
- 纵有天枢穷尽万象，仍庆幸生而为人，有爱可寻，有未知可探。

## 免责声明

- 本项目仅面向CTF竞赛辅助解题、研究、学习，禁止任何非法用途。
- 如您在使用本项目的过程中存在任何非法行为，您需自行承担相应后果。
- 除非您已充分阅读、完全理解并接受本协议，否则，请您不要使用本项目。
- 因大模型的不确定性，建议在容器或虚拟环境中运行。
