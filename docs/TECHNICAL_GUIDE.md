# SkillEngine 技术架构全景文档

> **Skills are new software. CLIs are new API. Agents are new OS.**
> **Multi-modal is new UI. Vibe coding is new product management.**

---

## 目录

1. [设计哲学：Skill First](#1-设计哲学skill-first)
2. [整体架构概览](#2-整体架构概览)
3. [核心体系详解](#3-核心体系详解)
   - [3.1 Skills 技能体系](#31-skills-技能体系)
   - [3.2 Agent 代理体系](#32-agent-代理体系)
   - [3.3 Tools 工具体系](#33-tools-工具体系)
   - [3.4 Memory 记忆体系](#34-memory-记忆体系)
   - [3.5 MCP / Extensions 扩展体系](#35-mcp--extensions-扩展体系)
4. [支撑体系](#4-支撑体系)
   - [4.1 事件总线与 Hooks (EventBus)](#41-事件总线与-hooks-eventbus)
   - [4.2 上下文管理 (ContextManager)](#42-上下文管理-contextmanager)
   - [4.3 会话持久化 (Session)](#43-会话持久化-session)
   - [4.4 包管理 (Packages)](#44-包管理-packages)
   - [4.5 跨模型适配 (Adapters)](#45-跨模型适配-adapters)
   - [4.6 TUI 终端界面体系](#46-tui-终端界面体系)
   - [4.7 Prompt Caching 缓存策略](#47-prompt-caching-缓存策略)
5. [完整执行流程：Skill 的动态加载与运行](#5-完整执行流程skill-的动态加载与运行)
6. [项目亮点与创新分析](#6-项目亮点与创新分析)
7. [与其他 Agent 哲学的对比](#7-与其他-agent-哲学的对比)
8. [学习路线图](#8-学习路线图)

---

## 1. 设计哲学：Skill First

### 1.1 四大 Agent 哲学

当前 Agent 领域存在四种主流设计哲学：

| 哲学类型 | 代表框架 | 核心特点 | Agent 自主性 | 典型场景 |
|---------|---------|---------|-------------|---------|
| **Tool First**（工具优先） | LangChain, Semantic Kernel | 给 agent 大量工具，prompt 里写 "use these tools when appropriate"，模型自由选择调用顺序 | 中高（模型自由选工具/顺序，但易混乱） | 探索性任务、RAG + tool 混合，但容易 token 爆炸、模型选错工具 |
| **Workflow First**（工作流优先） | LangGraph, CrewAI, n8n + AI nodes, Power Automate | 把 agent 任务建模成 graph / DAG / 角色团队 / 固定 SOP，步骤明确、可视化、可控、易审计 | 低→中（偏低，图/角色限制强） | 生产级、企业场景、需要可靠性/可观测性/成本控制时首选 |
| **Skill First**（技能/扩展优先） | **SkillEngine**, pi-mono/Pi, nanoclaw 变体 | 只给极少原语工具（4 个），所有高级能力外包给 skill 文件/扩展，prompt 极短，靠 caching + 模型推理 | **高**（裸 ReAct loop，几乎完全靠模型自己推理/决定何时结束） | 极简、高效 token、省钱、coding agent 场景最强，但可控性较弱 |
| **Conversation First**（对话/协作优先） | AutoGen, OpenAI Swarm | 代理间通过消息/对话协作，动态路由、辩论式推理 | 中高（对话动态，但可加 supervisor 降自主性） | 研究、复杂推理、多角色辩论场景 |

### 1.2 Skill First 的核心理念

```
传统 Tool First:  给 Agent 100 个 tools → 模型选择困难 → token 爆炸 → 推理混乱
Skill First:       给 Agent 4 个原语 + 动态注入的 Skill prompt → 模型自由推理 → 高效精准
```

**关键洞察**：Skill 不是 function calling 的 tool，而是注入到 system prompt 中的 **指导知识**。LLM 读取 Skill 内容后，使用少量内置工具（主要是 bash）来完成任务。

这是理解 SkillEngine 的关键区别：

| 概念 | Tool（传统工具） | Skill（技能） |
|------|-----------------|-------------|
| **注入位置** | function calling schema | system prompt 文本 |
| **模型感知** | JSON schema 描述 | 自然语言知识 + 最佳实践 |
| **执行方式** | 模型返回结构化调用 | 模型理解后用 bash 等原语执行 |
| **扩展成本** | 需要写代码 + 注册函数 | 只需写一个 Markdown 文件 |
| **Token 开销** | 每个 tool 都有 schema 占 token | Skill 内容可缓存，按需加载 |

### 1.3 "Skills are new software" 的隐喻

```
传统软件生态:    操作系统 → 应用程序 → 用户操作
Agent 新生态:    Agent (新 OS) → Skills (新软件) → 自然语言指令

传统 API 调用:   HTTP Client → REST API → JSON Response
Agent 新范式:    Agent → CLI 命令 (新 API) → 文本输出

传统 UI:         图形界面 → 鼠标点击 → 视觉反馈
Agent 新 UI:     多模态输入 → 自然语言 → 结构化输出
```

---

## 2. 整体架构概览

### 2.1 Agent + Skills + 虚拟机 架构

SkillEngine 的架构可以类比为一台"Agent 虚拟机"：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agent Configuration                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │              Core System Prompt                      │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                                                             │
│  Equipped Skills:                                                           │
│  ┌─────────┐ ┌─────────┐ ┌───────────┐ ┌─────┐ ┌──────┐ ┌──────┐        │
│  │bigquery │ │  docx   │ │nda-review │ │ pdf │ │ pptx │ │ xlsx │ ...    │
│  └─────────┘ └─────────┘ └───────────┘ └─────┘ └──────┘ └──────┘        │
│                                                                  ─────→    │
│  Equipped MCP Servers:                                   use computer     │
│  ○ MCP server 1                                                           │
│  ○ MCP server 2                                          ┌───────────────┐│
│  ○ MCP server 3                                          │ Agent Virtual ││
│       │  │  │                                            │   Machine     ││
│       ▼  ▼  ▼                                            │               ││
│  ┌────────────────────────────────────┐                  │ ┌────┐┌──────┐││
│  │   Remote MCP Servers               │                  │ │Bash││Python│││
│  │   (elsewhere on the internet)      │                  │ └────┘└──────┘││
│  └────────────────────────────────────┘                  │ ┌──────┐      ││
│                                                          │ │Node.js│     ││
│                                                          │ └──────┘      ││
│                                                          │               ││
│                                                          │ File System:  ││
│                                                          │ skills/       ││
│                                                          │  bigquery/    ││
│                                                          │   SKILL.md    ││
│                                                          │   rules.md    ││
│                                                          │  pdf/         ││
│                                                          │   SKILL.md    ││
│                                                          │   forms.md    ││
│                                                          │   extract.py  ││
│                                                          └───────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

**要点**：Skill 目录不仅包含 SKILL.md，还可以包含辅助文件（数据源配置、模板、脚本等），这些都存在于 Agent 的"虚拟文件系统"中，Agent 可以通过 `read` 工具访问。

### 2.2 Agent Loop 执行模型

```
┌──────────────────────────────────────────┐
│  GOAL                                     │
│  "handle this lead" / 用户任务描述        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  AGENT LOOP                               │
│  observe → think → act → learn → repeat  │
└──────┬───────────┬───────────┬───────────┘
       │           │           │
       ▼           ▼           ▼
┌────────────┐┌──────────┐┌───────────────┐
│ SUBAGENTS  ││ SKILLS   ││ TOOLS         │
│            ││          ││               │
│ code-      ││ lead-    ││ Built-in:     │
│  reviewer  ││  research││ Read, Write,  │
│ test-      ││          ││ Bash, Grep... │
│  runner    ││ email-   ││               │
│ researcher ││  drafting││ MCP:          │
│            ││          ││ Zapier, DBs,  │
│ (parallel, ││ (domain  ││ APIs...       │
│  isolated) ││ expertise││               │
│            ││  auto-   ││ Custom:       │
│            ││ invoked) ││ your functions│
└────────────┘└──────────┘└───────────────┘
       │           │           │
       └───────────┴───────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  HOOKS                                    │
│  guard rails, logging, human-in-the-loop │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  STRUCTURED OUTPUT                        │
│  validated JSON matching your schema      │
└──────────────────────────────────────────┘
```

**三条并行路径**：
- **Subagents**：并行隔离的子代理（代码审查、测试运行、调研）
- **Skills**：领域专业知识，自动根据上下文调用
- **Tools**：内置工具 + MCP 服务器 + 自定义函数

### 2.3 系统分层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户交互层                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ CLI/TUI  │  │  Web UI  │  │ RPC Mode │  │ JSON Mode│           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       └──────────────┴──────────────┴──────────────┘                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      AgentRunner（代理运行器）                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ EventBus    │  │ContextManager│  │SessionManager│               │
│  │ (事件+Hooks)│  │ (上下文管理) │  │ (会话持久化) │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      SkillsEngine（技能引擎）                        │
│                                                                     │
│  Skill Files ──→ [Loader] ──→ [Filter] ──→ [Snapshot] ──→ Prompt  │
│  (SKILL.md)     解析技能      资格检查      缓存快照     注入LLM   │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Markdown │  │ Default  │  │   Bash   │  │ OpenAI / │           │
│  │ Loader   │  │ Filter   │  │ Runtime  │  │ Anthropic│           │
│  └──────────┘  └──────────┘  └──────────┘  │ Adapter  │           │
│                                             └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                         外部可插拔体系                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Memory   │  │Extensions│  │ Packages │  │ Context  │           │
│  │(OpenViking)│ │ (MCP+插件)│ │ (包管理) │  │  Files   │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│  ┌──────────┐  ┌──────────┐                                        │
│  │ Themes   │  │PromptTpl │                                        │
│  │ (主题)   │  │ (模板)   │                                        │
│  └──────────┘  └──────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心体系详解

### 3.1 Skills 技能体系

Skills 是整个系统的核心。一个 Skill 本质上是一个**目录**，包含主描述文件和辅助资源。

#### 3.1.1 Skill 目录结构

一个 Skill 不只是一个文件，而是一个完整的目录：

```
skills/
├── bigquery/
│   ├── SKILL.md           # 主文件：YAML frontmatter + Markdown 指令
│   ├── datasources.md     # 辅助：可用的数据源列表
│   └── rules.md           # 辅助：查询规则和约束
│
├── docx/
│   ├── SKILL.md
│   ├── ooxml/             # 辅助：模板和规范
│   ├── spec.md
│   └── editing.md
│
├── pdf/
│   ├── SKILL.md
│   ├── forms.md           # 辅助：表单处理指南
│   ├── reference.md       # 辅助：参考文档
│   └── extract_fields.py  # 辅助：可执行脚本（Action）
│
└── nda-review/
    └── SKILL.md           # 简单 Skill：只有主文件
```

**核心理解**：SKILL.md 是 Agent 的"入口说明书"，辅助文件是 Agent 可以用 `read` 工具按需查阅的"参考手册"。Agent 的 system prompt 中只注入 SKILL.md 的内容，辅助文件在需要时才读取，实现了知识的按需加载。

#### 3.1.2 SKILL.md 定义格式

```yaml
# skills/github/SKILL.md
---
name: github
description: "Interact with GitHub repositories, issues, and PRs using the gh CLI"
metadata:
  emoji: "🐙"
  homepage: "https://cli.github.com"
  author: "Alex Zhang"
  version: "1.0.0"
  tags: ["git", "devops"]
  primary_env: "GITHUB_TOKEN"        # 主 API Key（Filter 用此检查环境变量）
  requires:
    bins: ["gh"]                      # 所有必须存在的二进制（AND 逻辑）
    any_bins: ["npm", "pnpm"]         # 至少一个存在（OR 逻辑）
    env: ["GITHUB_TOKEN"]             # 必须的环境变量
    os: ["darwin", "linux"]           # 支持的操作系统
  install:                            # 自动安装建议
    - kind: brew
      id: gh
      label: "GitHub CLI (Homebrew)"
      bins: ["gh"]
      os: ["darwin", "linux"]
  invocation:
    user_invocable: true              # 用户可通过 /github 直接调用
    disable_model_invocation: false   # 是否从 LLM system prompt 隐藏
    require_confirmation: false       # 执行前是否需要用户确认
actions:                              # 确定性脚本动作（非 LLM 推理）
  create-issue:
    script: "scripts/create-issue.sh"
    description: "Create a new GitHub issue"
    output: "json"
    params:
      - name: title
        type: string
        required: true
        position: 1
      - name: body
        type: string
        required: false
---
# GitHub CLI Skill

You have access to the GitHub CLI (`gh`).
Use it to manage repositories, issues, pull requests, and more.

## Common Operations

- List issues: `gh issue list`
- Create PR: `gh pr create --title "..." --body "..."`
- View PR status: `gh pr status`

## Best Practices

- Always check `gh auth status` before operations
- Use `--json` flag for structured output when parsing results
```

#### 3.1.3 Skill 数据模型

```python
@dataclass
class Skill:
    name: str                    # 唯一标识符（如 "github"）
    description: str             # 一行描述（供 LLM 概览用）
    content: str                 # 完整 Markdown 内容（注入 system prompt）
    file_path: Path              # SKILL.md 的完整路径
    base_dir: Path               # Skill 目录路径（用于解析辅助文件的相对路径）
    source: SkillSource          # 来源：BUNDLED / MANAGED / WORKSPACE / PLUGIN / EXTRA
    metadata: SkillMetadata      # 扩展元数据（见下）
    actions: dict[str, SkillAction]  # 确定性动作（脚本映射）

    def content_hash(self) -> str:
        """SHA256 哈希的前 16 位，用于缓存失效检测"""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def skill_key(self) -> str:
        """配置查找键（可自定义，默认 = name）"""
        return self.metadata.skill_key or self.name

@dataclass
class SkillMetadata:
    always: bool = False          # 始终包含（跳过 Filter 检查）
    skill_key: str | None = None  # 自定义配置查找键
    primary_env: str | None = None # 主要环境变量（如 "GITHUB_TOKEN"）
    emoji: str | None = None      # 显示图标
    homepage: str | None = None
    author: str | None = None
    version: str | None = None
    tags: list[str] = []          # 分类标签
    requires: SkillRequirements   # 运行要求
    install: list[SkillInstallSpec] = []  # 安装说明
    invocation: SkillInvocationPolicy = ... # 调用策略

@dataclass
class SkillRequirements:
    bins: list[str] = []          # ALL 必须存在
    any_bins: list[str] = []      # 至少 ONE 存在
    env: list[str] = []           # 必须的环境变量
    config: list[str] = []        # 必须的配置路径
    os: list[str] = []            # 支持的操作系统（darwin/linux/win32）
```

#### 3.1.4 Skill 来源优先级（覆盖机制）

```
BUNDLED (内置)  →  MANAGED (~/.skillengine/skills)  →  WORKSPACE (./skills)  →  EXTRA
    低优先级                                                                高优先级
```

后加载的目录覆盖先加载的同名 Skill。这意味着：
- 框架内置 github Skill → 用户全局自定义版覆盖 → 项目级版本再覆盖
- 类似 Linux 配置文件 `/etc/` → `~/.config/` → `./.config/` 的优先级机制

#### 3.1.5 Skill 加载管线

```
                ┌──────────────────────────────────────────────┐
                │            MarkdownSkillLoader                │
                │                                              │
SKILL.md ──────→│ 1. 读取文件内容                              │
                │ 2. 分离 YAML frontmatter 和 Markdown body   │
                │ 3. 解析 metadata, requirements, actions      │
                │ 4. 构建 Skill 对象                          │
                └──────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────────┐
                │            DefaultSkillFilter                 │
                │                                              │
                │ 短路检查（按序，首个失败即跳过该 Skill）：     │
                │ ① always=true? → 直接通过                    │
                │ ② 配置中 enabled=false? → 排除               │
                │ ③ 在 exclude_skills 列表中? → 排除           │
                │ ④ 在 bundled allowlist 中? → 检查            │
                │ ⑤ OS 匹配? (sys.platform)                   │
                │ ⑥ 所有 bins 存在? (shutil.which)            │
                │ ⑦ any_bins 至少一个存在?                     │
                │ ⑧ 环境变量存在?（三级查找）                   │
                │ ⑨ 配置路径存在?                              │
                └──────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────────┐
                │            SkillSnapshot（不可变快照）         │
                │                                              │
                │ skills: list[Skill]    # 合格的技能列表      │
                │ prompt: str            # 预格式化的 Prompt    │
                │ version: int           # 版本号（缓存键）    │
                │ timestamp: float       # 创建时间            │
                │ source_dirs: list[Path] # 扫描的目录         │
                └──────────────────────────────────────────────┘
```

**环境变量三级查找**（Filter 中的 `_has_env` 逻辑）：
```
检查环境变量 "GITHUB_TOKEN" 是否可用:
  Level 1: os.environ["GITHUB_TOKEN"] 存在?
  Level 2: 配置文件 entries.github.api_key 存在?（映射到 primary_env）
  Level 3: 配置文件 entries.github.env.GITHUB_TOKEN 存在?
```

#### 3.1.6 Prompt 格式化

Skill 内容格式化为三种格式注入 LLM system prompt：

**XML 格式（默认，LLM 解析最佳）：**
```xml
<skills>
  <skill name="github" emoji="🐙">
    <description>Interact with GitHub using gh CLI</description>
    <content>
      You have access to the GitHub CLI (`gh`)...
    </content>
  </skill>
</skills>
```

**Markdown 格式** / **JSON 格式** 也可按需选择。

#### 3.1.7 Skill 的 /command 调用

用户可以通过 `/skill-name` 直接调用 Skill，类似 Slash Command：

```python
# 用户输入: "/github create an issue for the login bug"
# 内部处理:
def _check_skill_invocation(self, user_input: str) -> Skill | None:
    match = re.match(r"^/(\S+)", user_input.strip())
    if match:
        return self.get_skill(match.group(1))
    return None

# 匹配到 github skill 后，注入完整 skill 内容到消息中
skill_context = (
    f"[User invoked skill: /{skill.name}]\n\n"
    f"<skill-content name=\"{skill.name}\">\n"
    f"{skill.content}\n"
    f"</skill-content>\n\n"
    f"User input: {user_input}"
)
```

---

### 3.2 Agent 代理体系

AgentRunner 是最上层的编排器，实现了完整的 **observe → think → act → learn → repeat** 循环。

#### 3.2.1 Agent 配置

```python
@dataclass
class AgentConfig:
    # LLM 设置
    model: str = "MiniMax-M2.1"           # 模型 ID
    base_url: str | None = None           # API base URL（默认 OPENAI_BASE_URL）
    api_key: str | None = None            # API Key（默认 OPENAI_API_KEY）
    temperature: float = 0.0              # 采样温度
    max_tokens: int = 8192                # 最大输出 token

    # Agent 行为
    max_turns: int = 50                   # 最大工具调用轮次
    enable_tools: bool = True             # 启用 function calling
    enable_reasoning: bool = False        # 启用推理模式
    auto_execute: bool = True             # 自动执行工具调用（无需确认）

    # 思考 & 传输
    thinking_level: ThinkingLevel = None  # off/minimal/low/medium/high/xhigh
    transport: Transport = "sse"          # sse/websocket/auto

    # Skills
    skill_dirs: list[Path] = []           # 技能目录列表
    watch_skills: bool = False            # 文件监听热重载
    system_prompt: str = ""               # 基础系统提示词

    # 缓存 & 上下文
    cache_retention: str = "short"        # none/short/long
    session_id: str | None = None         # 会话 ID（缓存键）
    load_context_files: bool = True       # 自动发现 AGENTS.md / CLAUDE.md
```

#### 3.2.2 Agent 执行循环（完整流程图）

```
用户输入 "帮我修复登录 bug"
  │
  ▼
┌─────────────────────────────────────────────────┐
│ 1. emit(INPUT) → 可转换/拦截/直接处理            │
│ 2. 检查 /skill-name 调用? → 若是则注入 skill    │
│ 3. 构建消息列表 (system prompt + history + user) │
│ 4. emit(AGENT_START) → Memory 创建 session      │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │ emit(TURN_START)    │◄──────────────────┐
          │ context 压缩检查    │                    │
          │ emit(CONTEXT_TRANSFORM) → Memory 同步   │
          │ 调用 LLM (stream)  │                    │
          └──────────┬──────────┘                    │
                     │                               │
          ┌──────────▼──────────┐                    │
          │ LLM 返回响应        │                    │
          │ emit(TURN_END)      │                    │
          └──────────┬──────────┘                    │
                     │                               │
              有 tool_calls?                         │
             ┌──YES──┴──NO──┐                        │
             │              │                        │
    ┌────────▼────────┐     ▼                        │
    │emit(BEFORE_TOOL) │  返回最终响应                │
    │→ Hooks 可阻止    │  emit(AGENT_END)            │
    │→ 可修改参数      │  → Memory commit            │
    │                  │                             │
    │ 执行工具(bash)   │                             │
    │ → BashRuntime    │                             │
    │ → 流式输出       │                             │
    │ → 支持中止       │                             │
    │                  │                             │
    │emit(AFTER_TOOL)  │                             │
    │→ 可修改结果      │                             │
    │                  │                             │
    │ 检查 steering?   │                             │
    │ 检查 follow_up?  │                             │
    │ 检查 abort?      │                             │
    └────────┬────────┘                              │
             │                                       │
             │  turn < max_turns?                    │
             └──────────YES──────────────────────────┘
```

#### 3.2.3 三种响应模式

```python
# 模式 1: 同步等待完整响应
response: AgentMessage = await agent.chat("Fix the bug in auth.py")
print(response.content)

# 模式 2: 流式文本增量（最简单）
async for delta in agent.chat_stream("Explain this code"):
    print(delta, end="", flush=True)

# 模式 3: 结构化事件流（最强大，TUI/Web 用）
async for event in agent.chat_stream_events("Refactor this"):
    match event.type:
        case "thinking_delta": show_thinking(event.content)  # 思考过程
        case "text_delta":     append_text(event.content)    # 文本输出
        case "tool_call_start": show_tool(event.tool_name)   # 工具调用开始
        case "tool_call_delta": update_args(event.args_delta)# 参数流式解析
        case "tool_result":    show_result(event.content)    # 工具结果
        case "done":           finish(event.finish_reason)   # 完成
```

#### 3.2.4 中断、转向与追加

```python
# 紧急中止：立即停止所有操作（杀掉正在运行的 subprocess）
agent.abort()

# 转向：注入新指令，当前工具执行完成后切换方向
agent.steer("Stop refactoring, focus on the failing tests instead")

# 追加：在当前循环结束后追加新任务
agent.follow_up("Also run the linter after fixing")
```

#### 3.2.5 多模态消息支持

```python
@dataclass
class AgentMessage:
    role: str                              # "user" / "assistant" / "system" / "tool"
    content: str | list[TextContent | ImageContent]  # 支持文本 + 图片混合
    tool_calls: list[dict] = []            # LLM 发起的工具调用
    reasoning: str | None = None           # 思考过程（支持 thinking 的模型）
    token_usage: TokenUsage | None = None  # Token 用量统计

# 多模态示例
message = AgentMessage(
    role="user",
    content=[
        TextContent(type="text", text="这个 UI 有什么问题？"),
        ImageContent(type="image", data="base64...", mime_type="image/png"),
    ]
)
```

---

### 3.3 Tools 工具体系

Skill First 的核心原则：**极少内置工具 + Skills 知识 = 无限能力。**

#### 3.3.1 内置原语工具（仅 4 个 function calling tools）

这是 Agent 通过 function calling 可以调用的全部工具：

| 工具 | 功能 | 关键参数 |
|------|------|---------|
| `execute` | 执行单条 bash 命令 | command, timeout, cwd |
| `execute_script` | 执行多行脚本 | script, timeout, cwd |
| `read` | 读取文件内容 | path, offset, limit |
| `write` | 写入文件 | path, content |

**为什么只有 4 个？**

```
传统方式：                              Skill First 方式：
tool: search_github_issues()          skill: "Use `gh issue list --json` to search issues"
tool: create_github_pr()              skill: "Use `gh pr create` to create PRs"
tool: run_pytest()        ────→       skill: "Use `pytest -v` to run tests"
tool: deploy_to_k8s()                 skill: "Use `kubectl apply -f` to deploy"
...100+ tools                         ...全用 bash execute
```

bash 是万能的。与其为每个 API 写 wrapper tool（增加 token 开销），不如教模型怎么用 CLI。这就是 **"CLIs are new API"** 的实践。

#### 3.3.2 扩展工具集（给 TUI/CLI 使用，非 function calling）

```python
# ToolRegistry 管理的工具（用于内部逻辑，不暴露给 LLM function calling）
create_coding_tools()     # → Read, Write, Edit, Bash（代码编辑场景）
create_read_only_tools()  # → Read, Grep, Find, Ls（只读分析场景）
create_all_tools()        # → 全部 7 个
```

| 内部工具 | 功能亮点 |
|---------|---------|
| `BashTool` | 100K 字符输出限制，120s 超时，流式输出 |
| `ReadTool` | cat -n 格式行号，base64 图片支持，2000 行默认 |
| `WriteTool` | 自动创建父目录，返回字节数统计 |
| `EditTool` | 精确字符串替换，模糊匹配提示（difflib），统一 diff 输出 |
| `FindTool` | Glob 模式，git ls-files 优先尊重 .gitignore |
| `GrepTool` | 优先用 ripgrep 加速，回退到 Python regex |
| `LsTool` | 递归 1000 条安全限制，长格式显示 |

#### 3.3.3 工具执行引擎（BashRuntime）

BashRuntime 实现了高性能的命令执行，有两条路径：

```
                    execute(command, timeout, on_output, abort_signal)
                                      │
                                      ▼
                            ┌─────────────────┐
                            │ 是否需要流式输出 │
                            │   或中止支持？   │
                            └────┬────────┬───┘
                                 │        │
                            NO   │        │ YES
                                 ▼        ▼
                          ┌──────────┐ ┌──────────────────────┐
                          │ 快速路径 │ │     流式路径          │
                          │          │ │                      │
                          │communicate│ │ 并发任务：           │
                          │ (等待完成)│ │ ├─ _read_stream(stdout)
                          │          │ │ ├─ _read_stream(stderr)
                          │          │ │ └─ _watch_abort()     │
                          └──────────┘ │                      │
                                       │ asyncio.wait(timeout)│
                                       └──────────────────────┘
```

**关键特性**：
- **协作式中止**：通过 `asyncio.Event` 信号触发 `process.kill()`
- **流式输出回调**：逐行读取，实时回调 `on_output(line)`
- **输出截断**：1MB 上限防止内存溢出
- **超时控制**：1-600 秒可配置

---

### 3.4 Memory 记忆体系

基于 OpenViking 上下文数据库的跨会话持久化记忆，通过 Extension 机制接入。

#### 3.4.1 架构

```
┌───────────────────────────────────────────────────────┐
│                    AgentRunner                         │
│                                                       │
│  ┌───────────────┐  ┌──────────────────────────────┐ │
│  │   EventBus    │  │     Extension Manager        │ │
│  │               │  │                              │ │
│  │ AGENT_START ──┼──┼→ MemoryHooks.on_agent_start │ │
│  │ CONTEXT_──────┼──┼→ MemoryHooks.on_context_    │ │
│  │  TRANSFORM    │  │    transform                │ │
│  │ AGENT_END ────┼──┼→ MemoryHooks.on_agent_end   │ │
│  └───────────────┘  │                              │ │
│                     │  LLM 可用的 Memory Tools:    │ │
│                     │  ├─ recall_memory (搜索记忆)  │ │
│                     │  ├─ save_memory (保存记忆)    │ │
│                     │  ├─ explore_memory (浏览记忆) │ │
│                     │  └─ add_knowledge (索引文件)  │ │
│                     └──────────────┬───────────────┘ │
└────────────────────────────────────┼─────────────────┘
                                     │ HTTP (async httpx)
                          ┌──────────▼──────────┐
                          │   OpenViking Server  │
                          │  (上下文数据库)       │
                          │  localhost:1933       │
                          └──────────────────────┘
```

#### 3.4.2 记忆工具详解

```python
# 1. 回忆 — 语义搜索历史对话和知识
recall_memory(query="用户的编码偏好", scope="user", limit=5)
# scope: "user"(用户级) / "session"(会话级) / "project"(项目级)

# 2. 保存 — 持久化重要信息（由 LLM 主动判断何时保存）
save_memory(content="用户偏好 Python type hints 和 dataclass", category="preferences")

# 3. 浏览 — 文件系统式浏览记忆库
explore_memory(uri="/users/sawzhang/preferences", recursive=True)

# 4. 知识索引 — 将本地文件内容索引到记忆库
add_knowledge(path="/path/to/architecture.md", reason="项目架构参考")
```

#### 3.4.3 透明生命周期管理

Memory 系统通过 EventBus 钩子实现完全透明的记忆管理：

```
AGENT_START → 自动创建 OpenViking session
     │
     ▼
每次 CONTEXT_TRANSFORM → 增量同步新消息到 OpenViking
     │                   （避免上下文压缩时丢失历史）
     ▼
AGENT_END → 同步剩余消息 + commit（触发知识提取）
```

**关键设计**：`_synced_message_count` 追踪已同步数量，只同步增量，避免重复。

---

### 3.5 MCP / Extensions 扩展体系

Extensions 是 SkillEngine 的插件系统，连接 MCP 服务器和自定义工具。

#### 3.5.1 Extensions vs Skills vs Tools 的区别

```
┌─────────────────────────────────────────────────────────────────┐
│                          能力来源矩阵                            │
│                                                                 │
│  Skills (知识层)     → 注入 system prompt，教模型做事           │
│  ├─ bigquery skill   → "用 bq 命令查 BigQuery"                │
│  └─ pdf skill        → "用 poppler 工具处理 PDF"              │
│                                                                 │
│  Tools (执行层)      → function calling，模型直接调用           │
│  ├─ execute          → 内置：执行 bash 命令                    │
│  ├─ recall_memory    → Extension 注册：搜索记忆                │
│  └─ search_jira      → Extension 注册：搜索 Jira             │
│                                                                 │
│  MCP Servers (连接层) → 远程服务器提供的工具                     │
│  ├─ Zapier MCP       → 自动化工作流                            │
│  ├─ Database MCP     → 数据库查询                              │
│  └─ Custom API MCP   → 自定义 API 封装                         │
│                                                                 │
│  Extensions (集成层)  → 注册工具、命令、钩子的插件               │
│  ├─ memory extension → 注册 4 个 memory tools + 3 个 hooks    │
│  ├─ jira extension   → 注册 search/create/update tools        │
│  └─ theme extension  → 注册 theme 命令                         │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.5.2 Extension API

```python
class ExtensionAPI:
    """Extension 可以使用的接口"""
    def register_tool(self, tool: ToolInfo): ...       # 注册 function calling 工具
    def register_command(self, cmd: CommandInfo): ...   # 注册 /command
    def get_event_bus(self) -> EventBus: ...            # 获取事件总线
    def get_engine(self) -> SkillsEngine: ...           # 获取技能引擎

class ToolInfo:
    name: str              # 工具名（LLM 可见）
    description: str       # 功能描述
    parameters: dict       # JSON Schema
    handler: Callable      # async handler(args) → str

class CommandInfo:
    name: str              # 命令名（如 "status"）
    description: str       # 描述
    handler: Callable      # 处理函数
    source: str            # 来源标识
```

#### 3.5.3 Memory 接入示例（Extension 的完整接入模式）

```python
async def setup_memory(agent: AgentRunner, config: MemoryConfig):
    """Memory 系统通过 Extension 机制接入 Agent — 体现了插件化设计"""

    # 1. 创建客户端 + 健康检查
    client = OpenVikingClient(config.base_url, config.api_key)
    if not await client.available:
        return None  # 优雅降级：Memory 不可用时 Agent 正常运行

    # 2. 初始化扩展管理器
    ext_manager = agent.engine.init_extensions()

    # 3. 注册 4 个 memory tools（LLM 通过 function calling 调用）
    state = MemoryState(client=client)
    for tool_factory in [make_recall_tool, make_save_tool,
                         make_explore_tool, make_knowledge_tool]:
        ext_manager.register_tool(tool_factory(state))

    # 4. 注册 3 个生命周期钩子（透明运行，无需 LLM 感知）
    hooks = MemoryHooks(client, state)
    bus = agent.events
    bus.on(AGENT_START, hooks.on_agent_start, priority=10)      # 创建 session
    bus.on(CONTEXT_TRANSFORM, hooks.on_context_transform, priority=100)  # 增量同步
    bus.on(AGENT_END, hooks.on_agent_end, priority=90)          # 提交+提取

    return client
```

---

## 4. 支撑体系

### 4.1 事件总线与 Hooks (EventBus)

EventBus 是整个系统的"神经网络"，所有生命周期事件通过它传播，支持拦截和修改。

```python
class EventBus:
    def on(self, event: str, handler, priority: int = 0, source: str = ""):
        """注册处理器。priority 越小越先执行。支持 sync 和 async handler"""
        ...
    async def emit(self, event: str, data) -> list[Any]:
        """触发事件，按优先级调用所有 handler，收集返回值"""
        ...
    def off_by_source(self, source: str) -> int:
        """按来源批量移除 handler（Extension 卸载时用）"""
        ...
```

**完整事件生命周期：**

| 事件 | 触发时机 | Handler 返回值能做什么 |
|------|---------|---------------------|
| `INPUT` | 用户输入 | `action="transform"` 改写输入 / `action="handled"` 直接响应 |
| `AGENT_START` | Agent 开始处理 | 初始化资源（Memory session 等） |
| `TURN_START` | 每次 LLM 调用前 | 记录轮次信息 |
| `CONTEXT_TRANSFORM` | 消息发送给 LLM 前 | **注入/删减/重排消息** |
| `TURN_END` | LLM 返回后 | 分析响应、统计 |
| `BEFORE_TOOL_CALL` | 工具执行前 | **阻止执行** / **修改参数**（guard rails） |
| `AFTER_TOOL_RESULT` | 工具执行后 | **修改结果**（审计/过滤） |
| `TOOL_EXECUTION_UPDATE` | 工具实时输出 | 流式展示进度 |
| `AGENT_END` | Agent 完成 | 清理资源、提交记忆 |
| `SESSION_START/END` | 会话生命周期 | 持久化管理 |
| `MODEL_CHANGE` | 模型切换 | 记录变更 |
| `COMPACTION` | 上下文压缩 | 同步记忆到外部存储 |

**Hooks 实战示例（Guard Rails）：**

```python
# Guard Rail: 阻止危险命令
@bus.on(BEFORE_TOOL_CALL, priority=0)
async def safety_guard(event: BeforeToolCallEvent):
    cmd = event.args.get("command", "")
    if any(danger in cmd for danger in ["rm -rf /", "DROP TABLE", "format c:"]):
        return ToolCallEventResult(block=True, reason="Dangerous command blocked")

# Logging: 记录所有工具调用
@bus.on(AFTER_TOOL_RESULT, priority=100)
async def audit_log(event: AfterToolResultEvent):
    log.info(f"Tool: {event.tool_name}, Duration: {event.turn}ms")

# Human-in-the-loop: 敏感操作确认
@bus.on(BEFORE_TOOL_CALL, priority=5)
async def require_approval(event: BeforeToolCallEvent):
    if "deploy" in event.args.get("command", ""):
        approved = await ask_user_confirmation("Deploy command detected. Proceed?")
        if not approved:
            return ToolCallEventResult(block=True, reason="User denied deployment")

# Context Injection: 动态注入项目规则
@bus.on(CONTEXT_TRANSFORM, priority=50)
async def inject_rules(event: ContextTransformEvent):
    messages = list(event.messages)
    messages.insert(1, AgentMessage(role="system", content="Rule: Always use Python 3.12+"))
    return ContextTransformEventResult(messages=messages)
```

---

### 4.2 上下文管理 (ContextManager)

管理 LLM 上下文窗口预算，防止长对话超出限制。

```python
class ContextManager:
    context_window: int = 128_000     # 模型上下文窗口大小
    reserve_tokens: int = 4096        # 输出预留 token
    threshold: float = 0.9            # 使用率达到 90% 时触发压缩
    compactor: ContextCompactor       # 压缩策略

    @property
    def budget_tokens(self) -> int:
        return self.context_window - self.reserve_tokens

    def should_compact(self, messages) -> bool:
        return self.usage_fraction(messages) >= self.threshold
```

**两种压缩策略：**

```
策略 1: TokenBudgetCompactor（默认）
  原始: [sys] [u1] [a1] [u2] [a2] [u3] [a3] [u4] [a4]  ← 超出 90%
  压缩: [sys]                    [u3] [a3] [u4] [a4]     ← 从前往后删除

策略 2: SlidingWindowCompactor（保留最近 N 轮）
  设置: max_turns = 3
  原始: [sys] [u1] [a1] [u2] [a2] [u3] [a3] [u4] [a4]
  压缩: [sys]           [u2] [a2] [u3] [a3] [u4] [a4]   ← 保留最近 3 轮
```

**Token 估算**：`estimate_tokens(text) → len(text) / 4`（启发式近似）

---

### 4.3 会话持久化 (Session)

基于 JSONL 的 append-only 会话存储，支持**分支对话树**。

#### 存储结构

```
~/.skillengine/sessions/{cwd-hash-16}/
  ├── {session-id-1}.jsonl
  └── {session-id-2}.jsonl    ← fork 产生的分支会话
```

#### 条目类型（8 种）

```json
{"type": "header", "id": "h1", "version": 1, "cwd": "/project"}
{"type": "message", "id": "m1", "parent_id": "h1", "role": "user", "content": "Fix bug"}
{"type": "message", "id": "m2", "parent_id": "m1", "role": "assistant", "content": "..."}
{"type": "model_change", "id": "mc1", "parent_id": "m2", "new_model": "claude-opus-4-6"}
{"type": "compaction", "id": "c1", "parent_id": "mc1", "summary": "...", "tokens_before": 50000}
{"type": "branch_summary", "id": "b1", "from_id": "m2", "summary": "探索了另一种方案"}
{"type": "label", "id": "l1", "target_id": "m3", "label": "checkpoint-before-refactor"}
{"type": "custom", "id": "x1", "custom_type": "analysis", "data": {...}}
```

#### 分支（Fork）与时间旅行

```
       h1 ─── m1 ─── m2 ─── m3 ─── m4  (主线)
                        │
                        └── m5 ─── m6    (分支：从 m2 开始的新方向)

# 创建分支
new_session = session_manager.fork(entry_id="m2")

# 回到过去某个点
session_manager.navigate(entry_id="m2")  # 移动 leaf 指针
```

---

### 4.4 包管理 (Packages)

多来源 Skill/Extension/Theme/Prompt 的发现和解析。

```python
class PackageManager:
    user_dir   = ~/.skillengine/packages/     # 用户级（全局）
    project_dir = ./.skillengine/packages/    # 项目级

    def resolve(self, sources=None) -> list[ResolvedPackage]:
        """自动发现 + 手动指定来源"""
        # 1. 扫描 user_dir 和 project_dir
        # 2. 解析 cwd 的 pyproject.toml
        # 3. 解析显式 sources
```

**来源类型：**
```python
parse_source("./my-skills/")                              → local
parse_source("my-skill-pack")                             → pypi
parse_source("git+https://github.com/user/skills.git")    → git
```

**Manifest（`pyproject.toml`）：**
```toml
[tool.skillengine]
skills = ["skills/**"]
extensions = ["ext/**/*.py"]
themes = ["themes/*.yaml"]
prompts = ["prompts/*.md"]
```

---

### 4.5 跨模型适配 (Adapters)

运行时切换 LLM 提供商，自动处理消息格式差异。

```python
# 运行时切换
agent.switch_model("claude-opus-4-6", adapter_name="anthropic")

# 消息自动转换
transform_messages(messages, target_provider="anthropic", source_provider="openai")
```

**自动处理的差异：**

| 差异 | OpenAI | Anthropic | 转换策略 |
|------|--------|-----------|---------|
| Tool Call ID 长度 | 450+ 字符 | 最大 64 | SHA-256 哈希截断 |
| Thinking blocks | 不支持 | 原生支持 | 转为 text 前缀 |
| System message | messages 数组中 | 独立 system 参数 | 自动提取 |
| 孤立 tool_calls | 可能出现 | 必须有 result | 插入合成空 result |
| Thinking budget | reasoning_effort | adaptive thinking | 统一 ThinkingLevel 映射 |

**ThinkingLevel 统一映射：**
```
ThinkingLevel:  off → minimal → low → medium → high → xhigh
Anthropic:      (无)   low      low   medium   high   max
OpenAI:         (无)   low      low   medium   high   high
Token Budget:    0     1024    2048   8192    16384  16384
```

---

### 4.6 TUI 终端界面体系

完整的终端 UI 框架，从零构建（不依赖 curses/blessed）。

```
┌────────────────────────────────────────────────┐
│                TUI Architecture                 │
│                                                │
│  TUIRenderer (差分渲染引擎)                     │
│  ├─ 只重绘变化的行                              │
│  ├─ DEC 2026 同步输出（无撕裂）                 │
│  └─ 终端大小变化自动重绘                        │
│                                                │
│  Container (垂直容器)                           │
│  ├─ MarkdownWidget (只读 Markdown 渲染)        │
│  ├─ InputWidget (单行输入 + readline 快捷键)   │
│  ├─ EditorWidget (多行编辑器 + 自动换行)       │
│  ├─ SelectList (列表选择 + 模糊过滤)          │
│  └─ Overlay (模态弹窗栈)                       │
│                                                │
│  AutoComplete (自动补全)                        │
│  ├─ FileAutocomplete (@文件路径)               │
│  ├─ CommandAutocomplete (/命令)                │
│  └─ CombinedAutocomplete (组合)                │
│                                                │
│  KeybindingsManager (快捷键)                   │
│  ├─ 可自定义 ~/.skillengine/keybindings.json      │
│  └─ 默认: Ctrl+C 中断, Ctrl+L 清屏, etc.     │
│                                                │
│  Theme (主题系统)                               │
│  ├─ 53 个颜色键 (core/message/markdown/syntax) │
│  ├─ JSON + 变量引用                            │
│  └─ 发现: ~/.skillengine/themes/ + .skillengine/     │
└────────────────────────────────────────────────┘
```

**主题变量解析**（巧妙的颜色复用机制）：
```json
{
  "name": "catppuccin-mocha",
  "variables": {
    "base": "#1e1e2e",
    "blue": "#89b4fa",
    "text": "#cdd6f4"
  },
  "colors": {
    "background": "base",       ← 引用 variables 中的 "base"
    "primary": "text",          ← 引用 variables 中的 "text"
    "accent": "blue",           ← 引用 variables 中的 "blue"
    "error": "#f38ba8"          ← 直接使用 hex 值
  }
}
```

---

### 4.7 Prompt Caching 缓存策略

针对不同 LLM 提供商的缓存配置：

```python
# Anthropic: 利用 ephemeral cache_control
cache_retention="short"  → {"type": "ephemeral"}              # 默认 TTL
cache_retention="long"   → {"type": "ephemeral", "ttl": "1h"} # 1 小时

# OpenAI: 利用 prompt_cache_key
cache_retention="short"  → {"prompt_cache_key": session_id}
cache_retention="long"   → {"prompt_cache_key": session_id, "prompt_cache_retention": "24h"}
```

**缓存对 Skill First 的价值**：
- System prompt 中的 Skills 内容在多轮对话中保持不变
- Anthropic 的 prompt caching 可以大幅减少重复传输的 token 费用
- `SkillSnapshot.version` 确保 Skill 变化时缓存正确失效

---

## 5. 完整执行流程：Skill 的动态加载与运行

以下以 **"帮我创建一个 GitHub Issue，标题是 'Fix login bug'"** 为例，展示完整的端到端流程。

### Phase 1: 初始化

```python
agent = await create_agent(
    skill_dirs=[Path("~/.skillengine/skills"), Path("./skills")],
    system_prompt="You are a helpful coding assistant.",
    model="claude-sonnet-4-20250514",
)
# 可选：接入 Memory 系统
client = await setup_memory(agent, MemoryConfig())
```

**内部发生了什么：**
```
AgentRunner.__init__()
 ├─ SkillsEngine(config, loader, filter, runtime)
 │   ├─ loader = MarkdownSkillLoader()
 │   ├─ filter = DefaultSkillFilter()
 │   └─ runtime = BashRuntime(shell="/bin/bash", default_timeout=30s, max_output=1MB)
 ├─ EventBus()
 ├─ ContextManager(context_window=200000, threshold=0.9)
 ├─ AdapterRegistry() → 注册 OpenAI + Anthropic adapters
 └─ load_context_files() → 向上遍历找 AGENTS.md / CLAUDE.md

setup_memory()
 ├─ OpenVikingClient → HTTP 健康检查
 ├─ ExtensionManager.register_tool() × 4 (recall/save/explore/knowledge)
 └─ EventBus.on() × 3 (agent_start/context_transform/agent_end)
```

### Phase 2: Skill 动态加载（get_snapshot 首次调用）

```
engine.get_snapshot(force_reload=False)
  │
  │  缓存为空 → 触发完整加载
  │
  ├─ load_skills()
  │   ├─ 扫描 ~/.skillengine/skills/
  │   │   ├─ github/SKILL.md
  │   │   │   ├─ 读取文件内容 (1.2KB)
  │   │   │   ├─ 正则分离: --- YAML --- + Markdown body
  │   │   │   ├─ yaml.safe_load(frontmatter)
  │   │   │   ├─ 解析 metadata:
  │   │   │   │   emoji="🐙", primary_env="GITHUB_TOKEN"
  │   │   │   │   requires.bins=["gh"]
  │   │   │   │   requires.env=["GITHUB_TOKEN"]
  │   │   │   ├─ 解析 actions: create-issue → scripts/create-issue.sh
  │   │   │   └─ → Skill(name="github", source=MANAGED, content_hash="a3f2b1...")
  │   │   │
  │   │   ├─ docker/SKILL.md → Skill(name="docker", source=MANAGED)
  │   │   └─ python/SKILL.md → Skill(name="python", source=MANAGED)
  │   │
  │   └─ 扫描 ./skills/
  │       └─ my-tool/SKILL.md → Skill(name="my-tool", source=WORKSPACE)
  │
  ├─ filter_skills([github, docker, python, my-tool], context)
  │   │
  │   │  context = FilterContext(
  │   │    platform="darwin",
  │   │    available_bins={"gh", "python3", "node", ...},
  │   │    env_vars={"GITHUB_TOKEN": "ghp_xxx", "PATH": "...", ...},
  │   │    config_values={}
  │   │  )
  │   │
  │   ├─ github:
  │   │   ① always=false → 继续
  │   │   ② enabled? → 没有配置禁用 → 继续
  │   │   ③ 在 exclude_skills? → 不在 → 继续
  │   │   ④ bundled allowlist? → 非 bundled → 跳过检查
  │   │   ⑤ os=["darwin", "linux"] → 当前 darwin ✓
  │   │   ⑥ bins=["gh"] → shutil.which("gh") ✓ 存在
  │   │   ⑦ any_bins=[] → 无要求 ✓
  │   │   ⑧ env=["GITHUB_TOKEN"] → os.environ 中存在 ✓
  │   │   ⑨ config=[] → 无要求 ✓
  │   │   → 合格 ✓
  │   │
  │   ├─ docker:
  │   │   ⑥ bins=["docker"] → shutil.which("docker") → None ✗
  │   │   → 不合格 (reason: "missing required binary: docker")
  │   │
  │   ├─ python: → 合格 ✓
  │   └─ my-tool: → 合格 ✓
  │
  ├─ format_prompt([github, python, my-tool], format="xml")
  │   └─ _format_xml() → "<skills><skill name='github'>...</skill>...</skills>"
  │
  └─ → SkillSnapshot(
       skills=[github, python, my-tool],     # 3 个合格 Skill
       prompt="<skills>...(~3KB)...",         # 预格式化 XML
       version=1,                             # 版本号（缓存键）
       timestamp=1708000000.0
     )
```

### Phase 3: 构建完整 System Prompt

```python
agent.build_system_prompt()
```

```
最终 system prompt 组成:

┌──────────────────────────────────────────────────────┐
│  1. 基础 System Prompt                               │
│  "You are a helpful coding assistant."               │
├──────────────────────────────────────────────────────┤
│  2. Context Files (自动发现, 按优先级排列)            │
│  # ~/.skillengine/AGENTS.md (全局)                      │
│  "Always use Python 3.12+ ..."                       │
│  # ./CLAUDE.md (项目级, 更高优先级)                   │
│  "This project uses FastAPI + PostgreSQL ..."        │
├──────────────────────────────────────────────────────┤
│  3. Skills Prompt (预格式化 XML)                     │
│  <skills>                                            │
│    <skill name="github" emoji="🐙">                  │
│      <description>Interact with GitHub...</description│
│      <content>                                        │
│        You have access to the GitHub CLI (`gh`)...   │
│        ## Common Operations                          │
│        - List issues: `gh issue list`                │
│        - Create PR: `gh pr create ...`               │
│      </content>                                      │
│    </skill>                                          │
│    <skill name="python">...</skill>                  │
│    <skill name="my-tool">...</skill>                 │
│  </skills>                                           │
└──────────────────────────────────────────────────────┘
```

### Phase 4: 用户请求 → Agent Loop

```python
response = await agent.chat("帮我创建一个 GitHub Issue，标题是 'Fix login bug'")
```

```
─── Step 1: INPUT 事件 ───
emit(INPUT, InputEvent(user_input="帮我创建一个 GitHub Issue..."))
  └─ 无 handler 拦截 → action="continue"
_check_skill_invocation("帮我创建...") → None（不以 / 开头）
history.append(AgentMessage(role="user", content="帮我创建..."))

─── Step 2: AGENT_START ───
emit(AGENT_START, AgentStartEvent(model="claude-sonnet-4-20250514", turn=0))
  └─ MemoryHooks.on_agent_start → client.create_session() → session_id="sess_abc123"

─── Step 3: Turn 1 — 调用 LLM ───
emit(TURN_START, TurnStartEvent(turn=1, message_count=2))
context_manager.should_compact(messages) → False (usage=12% < 90%)
emit(CONTEXT_TRANSFORM, ContextTransformEvent(messages=[sys, user], turn=1))
  └─ MemoryHooks.on_context_transform → sync 2 new messages to OpenViking

_call_llm(messages=[system_prompt, user_message], stream=True)
  └─ AnthropicAdapter.chat_stream_events()
      ├─ 将 tools=[execute, execute_script, read, write] + [recall_memory, ...] 转为 Anthropic 格式
      ├─ 附加 cache_control={"type": "ephemeral"} 到 system prompt
      └─ client.messages.stream(...) → AsyncIterator[StreamEvent]

LLM 思考后返回:
  text: "我来帮你创建这个 GitHub Issue。让我先确认 gh 认证状态。"
  tool_calls: [{
    id: "toolu_01ABC",
    type: "function",
    function: {
      name: "execute",
      arguments: "{\"command\": \"gh auth status\", \"timeout\": 10}"
    }
  }]

emit(TURN_END, TurnEndEvent(turn=1, has_tool_calls=True, tool_call_count=1))

─── Step 4: 工具执行 ───
emit(BEFORE_TOOL_CALL, BeforeToolCallEvent(
    tool_name="execute", args={"command": "gh auth status"}, turn=1))
  └─ safety_guard handler → 不含危险命令 → 无拦截

_execute_tool(tool_call):
  engine.execute(command="gh auth status", timeout=10)
    └─ BashRuntime.execute()
        ├─ asyncio.create_subprocess_shell("gh auth status", ...)
        ├─ 快速路径: process.communicate(timeout=10)
        ├─ exit_code=0, stdout="✓ Logged in to github.com as sawzhang"
        └─ → ExecutionResult(success=True, output="✓ Logged in...", duration_ms=423)

emit(AFTER_TOOL_RESULT, AfterToolResultEvent(
    tool_name="execute", result="✓ Logged in...", turn=1))

history.append(tool_result_message)

─── Step 5: Turn 2 — LLM 读取结果 + 执行创建 ───
emit(TURN_START, TurnStartEvent(turn=2, message_count=4))

LLM 返回:
  text: "已确认认证状态。现在创建 Issue。"
  tool_calls: [{
    function: {
      name: "execute",
      arguments: "{\"command\": \"gh issue create --title 'Fix login bug' --body 'Login functionality has a bug that needs to be fixed.' --repo sawzhang/myproject\"}"
    }
  }]

→ 工具执行:
  BashRuntime: /bin/bash -c "gh issue create --title 'Fix login bug' ..."
  → stdout: "https://github.com/sawzhang/myproject/issues/42"
  → ExecutionResult(success=True, exit_code=0, duration_ms=1523)

─── Step 6: Turn 3 — LLM 生成最终回复 ───
LLM 返回:
  text: "已成功创建 GitHub Issue #42！\n\n链接: https://github.com/sawzhang/myproject/issues/42\n\n标题: 'Fix login bug'"
  tool_calls: []  ← 无更多工具调用

emit(TURN_END, TurnEndEvent(turn=3, has_tool_calls=False))

─── Step 7: 完成 ───
无 tool_calls + 无 steering + 无 follow_up → 退出循环

emit(AGENT_END, AgentEndEvent(
    total_turns=3, finish_reason="complete", user_input="帮我创建..."))
  └─ MemoryHooks.on_agent_end:
      ├─ sync remaining 3 messages to OpenViking
      └─ client.commit_session("sess_abc123") → 触发知识提取

→ 返回 AgentMessage(
    role="assistant",
    content="已成功创建 GitHub Issue #42！...",
    token_usage=TokenUsage(input=3200, output=180, cache_read=2800)
  )
```

### Phase 5: 热重载（后续修改 SKILL.md 时）

```
用户编辑 skills/github/SKILL.md (添加新命令提示)
  │
  ▼
watchfiles.awatch() 检测到文件变化
  │
  ├─ 250ms 防抖等待（合并连续保存）
  │
  ├─ 过滤: path.name == "SKILL.md" → 是相关变化
  │
  ├─ engine.invalidate_cache()
  │   └─ self._snapshot = None  # 清除缓存
  │
  └─ 触发回调 → 通知 UI 显示 "Skills reloaded"
      └─ 下次 chat() 调用时:
          get_snapshot() → 缓存为空 → 重新执行完整加载管线
          → 新 SkillSnapshot(version=2, ...)
```

---

## 6. 项目亮点与创新分析

### 6.1 架构创新

| 亮点 | 说明 | 工程价值 |
|------|------|---------|
| **Skill as Directory** | Skill 不是单个文件，而是目录（SKILL.md + 辅助资源） | Agent 可按需 `read` 辅助文件，实现知识的惰性加载 |
| **Filter Short-circuit** | 9 级资格检查，任一失败立即跳过 | 毫秒级过滤，避免无效 Skill 进入 prompt |
| **Snapshot Immutability** | SkillSnapshot 不可变 + version 追踪 | 安全缓存，version 变化即缓存失效 |
| **Content Hashing** | SHA-256 哈希前 16 位标识内容变化 | 精确检测 Skill 内容变化，避免不必要的重加载 |
| **Thread-safe Env** | `contextvars.ContextVar` + `threading.Lock` | 并发执行多个 Skill 时互不干扰的环境变量隔离 |

### 6.2 设计模式精要

| 模式 | 应用场景 | 效果 |
|------|---------|------|
| **策略模式** | Loader/Filter/Runtime/Adapter 全部可替换 | 4 个子系统完全可插拔 |
| **观察者模式** | EventBus 事件总线 | 解耦核心逻辑与扩展逻辑 |
| **管线模式** | Load → Filter → Format → Cache → Inject | 清晰的数据流转 |
| **上下文管理器** | env_context 安全环境变量注入/恢复 | 零泄漏的环境变量管理 |
| **工厂方法** | `create_agent()`, `AgentConfig.from_env()` | 简化初始化 |
| **装饰器模式** | `@bus.on(EVENT)` 事件注册 | 优雅的钩子注册 |

### 6.3 Token 经济学

```
传统 Tool First (100 个 tools):
  每个 tool schema ≈ 200 tokens
  100 tools × 200 = 20,000 tokens/请求 (仅 tool 定义)
  + 每次请求重传全部 schema

Skill First (4 个原语 + Skills prompt):
  4 个 tool schema ≈ 800 tokens
  Skills prompt ≈ 3,000 tokens（但可 prompt caching）
  首次: 3,800 tokens
  后续: 3,800 tokens（Anthropic cache hit → ≈380 tokens 实际计费）

节省: 每次请求节省 ~80% token 费用
```

### 6.4 可扩展性设计

```
添加新能力的三条路径:

路径 1: 写 SKILL.md（零代码，最简单）
  → 教 Agent 新知识/最佳实践
  → 例: 添加 kubernetes skill = 写 SKILL.md 描述 kubectl 用法

路径 2: 注册 Extension Tool（需要代码，更强控制）
  → 给 Agent 新的 function calling tool
  → 例: 添加数据库查询 = 注册 query_db tool + handler

路径 3: 接入 MCP Server（远程服务，最灵活）
  → 连接远程 API/服务
  → 例: 接入 Zapier MCP = 访问 5000+ 应用的自动化能力
```

### 6.5 与 Claude Code 的对照

SkillEngine 的架构可以看到许多与 Claude Code (Anthropic 官方 CLI) 相似的设计理念：

| 特性 | Claude Code | SkillEngine |
|------|------------|---------|
| 技能系统 | Slash commands + CLAUDE.md | SKILL.md + /command |
| 内置工具 | Read, Write, Edit, Bash, Grep... | execute, read, write + 扩展 |
| 上下文管理 | 自动压缩对话 | ContextManager + 多策略 |
| 事件系统 | Hooks (pre/post commit) | EventBus (12+ 事件类型) |
| MCP 支持 | MCP servers | Extensions + MCP |
| 会话管理 | 对话历史 | JSONL + 分支树 |
| 思考模式 | Thinking levels | ThinkingLevel 映射 |

**差异点**：SkillEngine 的 Skill 是**更完整的目录结构**（SKILL.md + 辅助文件 + 脚本），而非单纯的 prompt 注入。它支持 Actions（确定性脚本执行），可以绕过 LLM 直接执行预定义操作。

---

## 7. 与其他 Agent 哲学的对比

| 维度 | Tool First | Workflow First | **Skill First** | Conversation First |
|------|-----------|---------------|-----------------|-------------------|
| **代表** | LangChain, Semantic Kernel | LangGraph, CrewAI, n8n | **SkillEngine**, pi-mono | AutoGen, OpenAI Swarm |
| **核心思路** | 给大量 tools，模型选择调用 | 建模为 DAG/图/SOP | **极少原语 + Skill 文件扩展** | 代理间消息协作 |
| **Prompt 体积** | 大（tool schema 多） | 中（步骤描述） | **小（Skill 内容 + 缓存）** | 中（对话协议） |
| **Agent 自主性** | 中高（选工具易混乱） | 低→中（图/角色限制强） | **高（裸 ReAct loop）** | 中高（可加 supervisor） |
| **Token 效率** | 低（每个 tool 有 schema） | 中 | **高（prompt caching）** | 中 |
| **可控性** | 弱（模型选错工具） | 强（预定义流程） | **中（靠模型推理）** | 中（可加约束） |
| **扩展方式** | 注册新 tool 函数 | 添加新 node/edge | **添加 SKILL.md 文件** | 添加新 agent |
| **适用场景** | RAG + tool 混合 | 生产级/企业 | **coding agent/个人 agent** | 研究/多角色辩论 |

### Skill First 的优势

1. **零代码扩展** — 添加能力只需写一个 Markdown 文件
2. **Token 高效** — Skill 内容利用 prompt caching，成本降低 80%+
3. **知识驱动** — 不仅描述"能做什么"，还教会模型"怎么做"和"最佳实践"
4. **热重载** — 修改 SKILL.md 文件后 250ms 内自动生效
5. **自然组合** — 模型根据上下文自动选择和组合多个 Skills
6. **渐进增强** — 从单个 SKILL.md 到完整目录（含辅助文件/脚本），能力可渐进扩展

### Skill First 的权衡

1. **可控性弱于 Workflow** — 模型自行决定使用路径
2. **依赖模型能力** — 需要 Claude/GPT-4 级别的强推理模型
3. **调试不透明** — 模型的 Skill 选择和组合逻辑隐含在推理中
4. **非确定性** — 同样的输入可能产生不同的执行路径

---

## 8. 学习路线图

### 按模块阅读顺序

```
第 1 层: 核心概念 (理解 Skill First 哲学)
├─ models.py              → Skill 数据模型
├─ config.py              → 配置结构
└─ loaders/markdown.py    → Skill 文件解析

第 2 层: 引擎管线 (理解加载→过滤→运行)
├─ engine.py              → 核心引擎编排
├─ filters/default.py     → 资格检查逻辑
└─ runtime/bash.py        → 命令执行引擎

第 3 层: Agent 运行时 (理解 ReAct Loop)
├─ agent.py               → AgentRunner 完整实现
├─ events.py              → 事件总线和数据类
└─ context.py             → 上下文窗口管理

第 4 层: 适配与扩展 (理解可插拔设计)
├─ adapters/base.py       → LLM 适配器抽象
├─ adapters/anthropic.py  → Anthropic 实现
├─ adapters/transform.py  → 跨提供商消息转换
└─ cache.py               → Prompt 缓存策略

第 5 层: 外部体系 (理解完整生态)
├─ memory/                → 跨会话记忆
├─ session/               → 会话持久化 + 分支
├─ packages/              → 包管理和发现
├─ tools/                 → 扩展工具集
├─ tui/                   → 终端 UI 框架
└─ web/                   → Web UI 后端

第 6 层: 入口与集成
├─ cli.py                 → CLI 命令入口
├─ modes/                 → 运行模式 (interactive/json/rpc)
└─ context_files.py       → AGENTS.md / CLAUDE.md 发现
```

### 关键文件行数参考

| 文件 | 行数 | 复杂度 | 核心价值 |
|------|------|--------|---------|
| `agent.py` | ~1100 | 高 | Agent 完整循环 + 流式 + 工具执行 |
| `engine.py` | ~600 | 中 | 加载管线 + 环境管理 + 文件监听 |
| `events.py` | ~350 | 中 | 类型化事件 + EventBus |
| `models.py` | ~200 | 低 | 所有数据结构 |
| `runtime/bash.py` | ~250 | 中 | 双路径执行 + 中止 + 流式 |
| `adapters/anthropic.py` | ~200 | 中 | Anthropic 特性 (thinking, cache) |
| `session/manager.py` | ~200 | 中 | JSONL 持久化 + 分支 |
| `memory/hooks.py` | ~100 | 低 | 透明记忆管理 |

---

> **总结**：SkillEngine 代表了一种以 **"知识注入"替代"工具堆砌"** 的 Agent 设计哲学。它通过 Markdown 目录定义技能、极少的内置原语工具、事件驱动的 Hooks 生命周期、可插拔的四层子系统（Loader / Filter / Runtime / Adapter）、以及 MCP/Extension 扩展机制，构建了一个极简但功能完备的 Agent 运行时。
>
> 在 "Skills are new software, Agents are new OS" 的范式下，SkillEngine 展示了个人 Agent 学习项目的理想架构 — 足够简单可以完全理解，足够完整可以真正使用。

---

## 9. 三步实现路线图（Phase 1 → Phase 2 → Phase 3）

如果要从零实现这套系统，可以分三个阶段，每个阶段都能独立运行和验证。

---

### Phase 1: 最小可用引擎 — Skills Engine Core

> **目标**：实现 Skill 的定义 → 加载 → 过滤 → Prompt 生成 → Bash 执行
> **核心文件**：models.py, config.py, engine.py, loaders/, filters/, runtime/bash.py

```
Phase 1 架构:

SKILL.md ──→ [MarkdownSkillLoader] ──→ [DefaultSkillFilter] ──→ SkillSnapshot
                                                                      │
                                                              format_prompt()
                                                                      │
                                                                      ▼
用户 ──→ 手动拼接 System Prompt + Skills Prompt ──→ 任意 LLM API
                                                         │
                                                    tool_call: execute
                                                         │
                                                         ▼
                                                   [BashRuntime]
                                                    执行 CLI 命令
```

#### Phase 1 实现清单

| 步骤 | 模块 | 要实现的内容 | 验证标准 |
|------|------|------------|---------|
| 1.1 | **models.py** | `Skill`, `SkillMetadata`, `SkillRequirements`, `SkillSnapshot`, `SkillSource` 数据类 | 能创建 Skill 实例并序列化 |
| 1.2 | **config.py** | `SkillsConfig`, `SkillEntryConfig`；支持 YAML 加载 | 能从 YAML 文件加载配置 |
| 1.3 | **loaders/markdown.py** | `MarkdownSkillLoader`：解析 YAML frontmatter + Markdown body | 给定 SKILL.md 文件，能解析出完整 Skill 对象 |
| 1.4 | **filters/default.py** | `DefaultSkillFilter`：9 级短路检查（OS/bins/env/config） | 能根据当前环境正确筛选出合格的 Skills |
| 1.5 | **runtime/bash.py** | `BashRuntime`：subprocess 执行 + 超时 + 输出收集 | 能执行 bash 命令并返回 `ExecutionResult` |
| 1.6 | **engine.py** | `SkillsEngine`：组装管线 load→filter→format→snapshot | 调用 `get_snapshot()` 能返回完整的 SkillSnapshot |
| 1.7 | **CLI** | `skills list` / `skills show <name>` / `skills exec <cmd>` 基本命令 | 命令行可列出、查看、执行 Skills |

#### Phase 1 示例代码

```python
# Phase 1 完成后可以这样使用:
from agent_skills_engine import SkillsEngine, SkillsConfig

config = SkillsConfig(skill_dirs=[Path("./skills")])
engine = SkillsEngine(config)

# 加载并过滤技能
snapshot = engine.get_snapshot()
print(f"合格技能: {snapshot.skill_names}")   # ['github', 'python']
print(f"Prompt:\n{snapshot.prompt}")          # <skills>...</skills>

# 执行命令
result = await engine.execute("gh issue list --limit 5")
print(result.output)
```

#### Phase 1 收获

- 理解 Skill 定义格式和数据模型
- 掌握管线（Pipeline）模式：Load → Filter → Format
- 掌握短路过滤和环境检测
- 掌握 async subprocess 执行

---

### Phase 2: Agent 运行时 — ReAct Loop + Event System

> **目标**：实现完整的 Agent 循环（LLM 调用 → 工具执行 → 多轮迭代），加入事件系统
> **核心文件**：agent.py, events.py, context.py, adapters/, cache.py

```
Phase 2 架构:

                    ┌─────────────────────────────────────────┐
                    │            AgentRunner                    │
                    │                                         │
用户输入 ──→        │  ┌───────────┐    ┌────────────────┐    │
                    │  │ EventBus  │    │ContextManager  │    │
                    │  │ (12 事件) │    │ (Token 预算)   │    │
                    │  └───────────┘    └────────────────┘    │
                    │                                         │
                    │  Agent Loop:                            │
                    │  ┌─────────────────────────────────┐   │
                    │  │ 1. build_system_prompt()         │   │
                    │  │ 2. _call_llm() [OpenAI/Anthropic]│  │
                    │  │ 3. _execute_tool() [BashRuntime] │   │
                    │  │ 4. 检查 tool_calls → 循环/退出   │   │
                    │  └─────────────────────────────────┘   │
                    │                                         │
                    │  SkillsEngine (Phase 1)                 │
                    └─────────────────────────────────────────┘
```

#### Phase 2 实现清单

| 步骤 | 模块 | 要实现的内容 | 验证标准 |
|------|------|------------|---------|
| 2.1 | **events.py** | `EventBus` 类 + 12 个事件数据类；支持 sync/async handler，priority 排序 | 事件注册 → 触发 → handler 按优先级执行 |
| 2.2 | **context.py** | `ContextManager` + `TokenBudgetCompactor` + `SlidingWindowCompactor` | 消息超过阈值时能自动压缩 |
| 2.3 | **adapters/base.py** | `LLMAdapter` 抽象类 + `Message`, `AgentResponse`, `ToolDefinition` | 定义统一的 LLM 接口 |
| 2.4 | **adapters/openai (内置)** | 通过 openai SDK 实现 `chat()` 和 `chat_stream()`；返回 tool_calls | 能调用 OpenAI 兼容 API 并解析 tool_calls |
| 2.5 | **adapters/anthropic.py** | `AnthropicAdapter`：thinking budget, cache_control, streaming events | 能调用 Claude API 并处理 thinking blocks |
| 2.6 | **adapters/transform.py** | `transform_messages()`：tool_call ID 归一化、thinking block 转换、孤立 tool_call 修复 | 消息在 OpenAI ↔ Anthropic 间无损转换 |
| 2.7 | **cache.py** | `get_cache_control_anthropic()`, `get_cache_config_openai()` | Prompt caching 正确配置 |
| 2.8 | **agent.py** | `AgentRunner`：ReAct loop, `chat()`, `chat_stream()`, `chat_stream_events()`, abort/steer/follow_up | 能多轮工具调用自动执行并返回最终结果 |
| 2.9 | **context_files.py** | `load_context_files()`：向上遍历发现 AGENTS.md/CLAUDE.md | 能自动发现并注入项目上下文 |

#### Phase 2 示例代码

```python
# Phase 2 完成后可以这样使用:
from agent_skills_engine import create_agent, BEFORE_TOOL_CALL

agent = await create_agent(
    skill_dirs=[Path("./skills")],
    system_prompt="You are a helpful assistant.",
    model="claude-sonnet-4-20250514",
)

# 注册安全钩子
@agent.events.on(BEFORE_TOOL_CALL, priority=0)
async def guard(event):
    if "rm -rf" in event.args.get("command", ""):
        return ToolCallEventResult(block=True, reason="Blocked")

# 同步对话
response = await agent.chat("帮我查看最近的 GitHub Issues")
print(response.content)

# 流式对话
async for event in agent.chat_stream_events("重构 auth 模块"):
    if event.type == "text_delta":
        print(event.content, end="")
    elif event.type == "tool_call_start":
        print(f"\n[调用工具: {event.tool_name}]")

# 中止长时间操作
agent.abort()
```

#### Phase 2 收获

- 理解 ReAct Loop（observe→think→act→learn→repeat）
- 掌握 EventBus 事件驱动架构（发布/订阅 + 拦截/修改）
- 掌握 LLM streaming + function calling 的完整流程
- 掌握跨 LLM 提供商的消息格式适配
- 理解上下文窗口管理和压缩策略

---

### Phase 3: 完整生态 — Memory + Session + Extension + TUI

> **目标**：接入持久化记忆、会话管理、插件系统、终端界面，形成完整可用的 Agent 产品
> **核心文件**：memory/, session/, packages/, tools/, tui/, web/, modes/

```
Phase 3 架构:

┌──────────────────────────────────────────────────────────────────────┐
│                        完整产品形态                                    │
│                                                                      │
│  ┌──────────────────────┐                                           │
│  │  用户交互层           │                                           │
│  │  ┌─────┐ ┌────┐ ┌───┐│                                           │
│  │  │ TUI │ │Web │ │RPC││                                           │
│  │  └──┬──┘ └─┬──┘ └─┬─┘│                                           │
│  └─────┼──────┼──────┼──┘                                           │
│        └──────┼──────┘                                               │
│               │                                                      │
│  ┌────────────▼──────────────────────────────────────────────────┐  │
│  │           AgentRunner (Phase 2)                                │  │
│  │                                                                │  │
│  │  ┌─────────────────────────────────────────────────────┐     │  │
│  │  │ New in Phase 3:                                      │     │  │
│  │  │                                                      │     │  │
│  │  │  Memory System          Session Manager              │     │  │
│  │  │  ├─ OpenVikingClient    ├─ JSONL Store              │     │  │
│  │  │  ├─ 4 Memory Tools      ├─ Tree Structure           │     │  │
│  │  │  └─ 3 Lifecycle Hooks   └─ Fork / Navigate          │     │  │
│  │  │                                                      │     │  │
│  │  │  Extension System        Package Manager             │     │  │
│  │  │  ├─ ToolInfo Registry    ├─ local/pypi/git Sources  │     │  │
│  │  │  ├─ CommandInfo          ├─ pyproject.toml Manifest  │     │  │
│  │  │  └─ ExtensionAPI         └─ Auto-discovery           │     │  │
│  │  │                                                      │     │  │
│  │  │  TUI Framework           Theme System                │     │  │
│  │  │  ├─ Diff Renderer        ├─ 53 Color Keys           │     │  │
│  │  │  ├─ Input/Editor         ├─ Variable Resolution      │     │  │
│  │  │  ├─ Autocomplete         └─ JSON + Discovery         │     │  │
│  │  │  └─ Keybindings                                      │     │  │
│  │  └──────────────────────────────────────────────────────┘     │  │
│  │                                                                │  │
│  │  SkillsEngine (Phase 1)                                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

#### Phase 3 实现清单

| 步骤 | 模块 | 要实现的内容 | 验证标准 |
|------|------|------------|---------|
| **3.1 Session** | | | |
| 3.1a | session/models.py | 8 种条目类型 + `SessionContext` | 能表示完整对话状态 |
| 3.1b | session/store.py | JSONL 读写 + `list_sessions()` | 对话可持久化到文件 |
| 3.1c | session/tree.py | `build_tree()`, `get_branches()`, `walk_to_root()` | 对话树可正确构建和遍历 |
| 3.1d | session/manager.py | `SessionManager`：append, build_context, fork, navigate | 支持分支和回溯 |
| **3.2 Memory** | | | |
| 3.2a | memory/client.py | `OpenVikingClient`：HTTP 封装 (create_session, add_message, find, commit) | 能与 OpenViking 服务器通信 |
| 3.2b | memory/tools.py | 4 个 Memory Tools (recall/save/explore/knowledge) | LLM 能通过 function calling 操作记忆 |
| 3.2c | memory/hooks.py | 3 个生命周期钩子（增量同步 + 自动 commit） | Agent 结束后记忆自动保存 |
| 3.2d | memory/extension.py | `setup_memory()` 一键接入函数 | 一行代码接入 Memory 系统 |
| **3.3 Extensions** | | | |
| 3.3a | extensions/ | `ExtensionManager`, `ExtensionAPI`, `ToolInfo`, `CommandInfo` | Extension 可注册工具和命令 |
| 3.3b | tools/registry.py | `ToolRegistry` + `BaseTool` + 7 个内置工具 | Read/Write/Edit/Bash/Find/Grep/Ls 可用 |
| **3.4 Packages** | | | |
| 3.4a | packages/ | `PackageManager`：auto-discover + pyproject.toml manifest | 能从多来源解析 Skills/Extensions |
| **3.5 Modes** | | | |
| 3.5a | modes/json_mode.py | JSONL 输出模式（单次调用） | 程序化集成可用 |
| 3.5b | modes/rpc_mode.py | stdin/stdout JSON-RPC 协议 | 外部程序可通过 RPC 控制 Agent |
| 3.5c | modes/interactive.py | Rich-based 交互模式 | 终端交互可用 |
| **3.6 TUI** | | | |
| 3.6a | tui/renderer.py | 差分渲染引擎 | 只重绘变化行，无闪烁 |
| 3.6b | tui/input_widget.py | 单行输入 + readline 快捷键 | 支持 Ctrl+A/E/K/U/W 等 |
| 3.6c | tui/editor_widget.py | 多行编辑器 + 自动换行 | 支持 Tab/PgUp/PgDn/Ctrl+Enter |
| 3.6d | tui/autocomplete.py | @文件 + /命令 自动补全 | Tab 补全可用 |
| 3.6e | tui/theme/ | 53 颜色键 + JSON 变量解析 + 发现 | 自定义主题可加载 |
| 3.6f | tui/keybindings.py | 快捷键管理 + ~/.skillengine/keybindings.json | 快捷键可自定义 |
| **3.7 Web** | | | |
| 3.7a | web/server.py | Starlette + SSE + WebSocket | Web UI 可用 |
| 3.7b | web/storage.py | SQLite 持久化 | Web 会话可保存 |

#### Phase 3 示例代码

```python
# Phase 3 完成后的完整产品形态:

from agent_skills_engine import create_agent, MemoryConfig, setup_memory

# 1. 创建 Agent
agent = await create_agent(
    skill_dirs=[Path("~/.skillengine/skills"), Path("./skills")],
    system_prompt="You are Alex's personal coding assistant.",
    model="claude-sonnet-4-20250514",
    watch_skills=True,        # 热重载
    cache_retention="long",   # 长缓存
)

# 2. 接入 Memory
await setup_memory(agent, MemoryConfig(auto_session=True, auto_commit=True))

# 3. 接入 Session 持久化
from agent_skills_engine.session import SessionManager
agent.session = SessionManager(session_dir=Path("~/.skillengine/sessions"))

# 4. 运行交互模式
from agent_skills_engine.modes import InteractiveMode
mode = InteractiveMode(agent)
await mode.run()  # 进入 TUI 交互界面

# 或者运行 Web UI
from agent_skills_engine.web import run_server
await run_server(agent, port=8080)

# 或者 CLI 单次调用
from agent_skills_engine.modes import JsonMode
mode = JsonMode(agent)
await mode.run("帮我部署最新版本到 staging")
```

#### Phase 3 收获

- 掌握跨会话记忆的透明集成模式（EventBus hooks）
- 理解 JSONL append-only 存储 + 对话树分支
- 掌握 Extension/Plugin 系统设计
- 理解包发现和解析机制
- 掌握终端 UI 差分渲染和输入处理
- 理解 SSE/WebSocket 实时通信

---

### 三阶段总结对照

```
Phase 1: Skills Engine Core          Phase 2: Agent Runtime            Phase 3: 完整生态
─────────────────────                ─────────────────────              ─────────────────────
models.py                            events.py                         memory/
config.py                            context.py                        session/
loaders/                             agent.py                          packages/
filters/                             adapters/                         tools/
runtime/bash.py                      cache.py                          tui/
engine.py                            context_files.py                  web/
cli.py (基本命令)                    cli.py (chat 命令)                modes/
                                                                       themes/

交付物:                              交付物:                           交付物:
✅ skills list/show/exec             ✅ 完整 ReAct Loop                ✅ 持久化记忆
✅ Prompt 生成                       ✅ 多轮工具调用                   ✅ 会话分支
✅ Bash 执行                         ✅ 流式响应                       ✅ 插件系统
✅ Skill 过滤                        ✅ 事件钩子                       ✅ 包管理
                                     ✅ 跨模型适配                    ✅ TUI + Web UI
                                     ✅ 上下文压缩                    ✅ 主题 + 快捷键

复杂度: ★★☆☆☆                      复杂度: ★★★★☆                   复杂度: ★★★★★
耗时: 1-2 周                        耗时: 2-3 周                     耗时: 3-4 周
代码量: ~1500 行                    代码量: ~3000 行                 代码量: ~5000 行
```

**每个 Phase 都是完整可用的**：
- Phase 1 完成后：可以通过 CLI 管理 Skills，手动拼接 Prompt 调用 LLM
- Phase 2 完成后：可以通过代码实现自动化的 Agent 对话（已是一个完整的 coding agent）
- Phase 3 完成后：可以作为日常使用的个人 Agent 产品（TUI/Web/记忆/会话）
