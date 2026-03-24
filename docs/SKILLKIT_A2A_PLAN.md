# SkillEngine A2A 集成方案

> Skill 是资产，Runtime 是可替换的。Orchestrator 的智能程度 = 它对下游 Agent 的理解深度。

## 现状评估

### 已有能力（agent-skills-engine）

| 模块 | 状态 | 说明 |
|---|---|---|
| Skill 模型 | 成熟 | `Skill`, `SkillMetadata`, `SkillAction`, `SkillInvocationPolicy` |
| Loader | 成熟 | `MarkdownSkillLoader` 解析 SKILL.md + YAML frontmatter |
| Filter | 成熟 | 环境/OS/bins 过滤，短路判定 |
| Engine | 成熟 | Pipeline 编排：Load → Filter → Prompt → Execute |
| Agent | 成熟 | `AgentRunner` + on-demand skill loading + fork isolation |
| EventBus | 已实现 | `before_tool_call`, `after_tool_result`, `turn_start/end` |
| Scheduler | 已实现 | INTERVAL / ONCE / EVENT / CRON 触发 |
| Transports | 已实现 | SSE, WebSocket |
| Adapters | 已实现 | OpenAI, Anthropic |
| Extensions | 已实现 | Plugin API |

### 缺失能力（本方案要补）

| 能力 | 优先级 | 说明 |
|---|---|---|
| Agent Card 生成 | P0 | 从 Skill 自动生成 A2A 标准 Agent Card |
| Agent Registry | P0 | 注册、发现、索引所有可用 Agent |
| 语义路由 | P1 | Orchestrator 基于意图 + 能力匹配做路由 |
| A2A Server | P1 | 暴露 `/.well-known/agent.json` + `/tasks` |
| A2A Client | P2 | 调用外部 A2A Agent |
| 绩效追踪 | P2 | Agent 执行历史、成功率、延迟 |

---

## 核心设计

### 设计原则

1. **SKILL.md 是唯一真相源** — Agent Card、路由元数据、系统提示词全部从 SKILL.md 派生
2. **零额外标注** — 现有 frontmatter 字段足以生成 Agent Card，不强制新增字段
3. **渐进增强** — `a2a:` frontmatter 块为可选扩展，无此块的 Skill 仍正常工作
4. **传输透明** — Skill 不感知自己是被本地调用还是 A2A 远程调用

### SKILL.md Frontmatter 扩展

```yaml
---
name: twitter-analyze
description: 推文分类和价值评估
allowed-tools: [Read, Bash, Grep]
model: claude-haiku-4-5-20250514

# 现有字段（已足够生成基础 Agent Card）
metadata:
  version: "1.0.0"
  author: sawzhang
  tags: [twitter, analysis, content]

# 新增 A2A 扩展（可选）
a2a:
  expose: true                    # 是否暴露为 A2A 服务端
  input_schema:                   # 输入格式约束
    type: object
    properties:
      tweets:
        type: array
        items: { type: string }
    required: [tweets]
  output_schema:                  # 输出格式约束
    type: object
    properties:
      analysis: { type: string }
      categories: { type: array }
  stateful: false                 # 是否支持多轮
  max_duration: 120               # 预估最大执行秒数
  cost_hint: low                  # low / medium / high
---
```

**向后兼容**：没有 `a2a:` 块的 Skill 默认 `expose: false`，行为不变。

---

## 模块设计

### 1. AgentCard — 从 Skill 自动生成

```
新文件：src/skillengine/a2a/agent_card.py
```

```python
@dataclass
class AgentCard:
    """A2A 标准 Agent Card，从 Skill 自动派生。"""

    name: str
    description: str
    version: str
    url: str | None = None            # A2A endpoint URL

    # 能力描述
    skills: list[AgentCardSkill] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])

    # 运行特征
    capabilities: AgentCapabilities = field(default_factory=lambda: AgentCapabilities())

    # 元数据
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    cost_hint: str | None = None      # low / medium / high

    @classmethod
    def from_skill(cls, skill: Skill, base_url: str | None = None) -> "AgentCard":
        """从 Skill 实例生成 Agent Card。

        映射关系：
        - skill.name → card.name
        - skill.description → card.skills[0].description
        - skill.metadata.version → card.version
        - skill.metadata.tags → card.tags
        - skill.metadata.author → card.author
        - skill.a2a.input_schema → card.skills[0].input_schema
        - skill.a2a.output_schema → card.skills[0].output_schema
        """
        a2a = skill.frontmatter.get("a2a", {})

        return cls(
            name=skill.name,
            description=skill.description,
            version=skill.metadata.version or "1.0.0",
            url=f"{base_url}/agents/{skill.name}" if base_url else None,
            skills=[
                AgentCardSkill(
                    name=skill.name,
                    description=skill.description,
                    input_schema=a2a.get("input_schema"),
                    output_schema=a2a.get("output_schema"),
                )
            ],
            capabilities=AgentCapabilities(
                streaming=False,
                multi_turn=a2a.get("stateful", False),
            ),
            author=skill.metadata.author,
            tags=skill.metadata.tags,
            cost_hint=a2a.get("cost_hint"),
        )

    def to_dict(self) -> dict:
        """序列化为 A2A /.well-known/agent.json 格式。"""
        ...

    def to_embedding_text(self) -> str:
        """生成用于语义索引的文本表示。

        包含 name + description + tags + skills，
        供 Orchestrator 做 embedding 相似度匹配。
        """
        parts = [self.name, self.description]
        parts.extend(self.tags)
        for s in self.skills:
            parts.append(s.description)
        return " | ".join(parts)


@dataclass
class AgentCardSkill:
    name: str
    description: str
    input_schema: dict | None = None
    output_schema: dict | None = None


@dataclass
class AgentCapabilities:
    streaming: bool = False
    multi_turn: bool = False
    push_notifications: bool = False
```

**关键设计**：`to_embedding_text()` 方法为语义路由提供索引文本，Orchestrator 可以对所有 Agent Card 做 embedding 后用向量检索匹配意图。

### 2. AgentRegistry — 注册、发现、索引

```
新文件：src/skillengine/a2a/registry.py
```

```python
class AgentRegistry:
    """统一的 Agent 注册表。

    管理两类 Agent：
    1. 本地 Agent（从 Skill 加载，进程内执行）
    2. 远程 Agent（通过 A2A 协议发现，网络调用）
    """

    def __init__(self):
        self._local: dict[str, RegisteredAgent] = {}
        self._remote: dict[str, RegisteredAgent] = {}
        self._embeddings: dict[str, list[float]] | None = None

    # ---- 注册 ----

    def register_skill(self, skill: Skill, base_url: str | None = None) -> None:
        """从 Skill 注册本地 Agent。"""
        card = AgentCard.from_skill(skill, base_url)
        self._local[skill.name] = RegisteredAgent(
            card=card,
            skill=skill,
            source=AgentSource.LOCAL,
        )
        self._invalidate_embeddings()

    def register_remote(self, agent_url: str) -> None:
        """通过 A2A 协议发现远程 Agent。

        GET {agent_url}/.well-known/agent.json → AgentCard
        """
        card = self._fetch_agent_card(agent_url)
        self._remote[card.name] = RegisteredAgent(
            card=card,
            endpoint=agent_url,
            source=AgentSource.REMOTE,
        )
        self._invalidate_embeddings()

    # ---- 查询 ----

    def get(self, name: str) -> RegisteredAgent | None:
        return self._local.get(name) or self._remote.get(name)

    def all_cards(self) -> list[AgentCard]:
        """返回所有已注册 Agent 的 Card（用于注入 Orchestrator system prompt）。"""
        agents = list(self._local.values()) + list(self._remote.values())
        return [a.card for a in agents]

    def cards_summary(self, budget: int = 4000) -> str:
        """生成 Agent Card 摘要文本，控制在 token 预算内。

        用于注入 Orchestrator 的 system prompt，
        让 LLM 在每次对话时"看到"所有可用 Agent 的能力。
        """
        lines = []
        for card in self.all_cards():
            entry = f"- **{card.name}**: {card.description}"
            if card.tags:
                entry += f" [{', '.join(card.tags)}]"
            lines.append(entry)
            if sum(len(l) for l in lines) > budget:
                break
        return "\n".join(lines)

    # ---- 语义路由 ----

    def match(self, query: str, top_k: int = 3) -> list[RegisteredAgent]:
        """语义匹配：用户意图 → 最相关的 Agent 列表。

        Phase 1: 关键词 + 描述匹配（无需 embedding 模型）
        Phase 2: embedding 向量相似度（需要 embedding 模型）
        """
        # Phase 1 实现：基于 LLM 的路由（利用 cards_summary）
        # Orchestrator LLM 读取所有 Card 摘要后自行判断
        # 这里返回所有候选，让 LLM 决策
        return self._keyword_match(query, top_k)

    def _keyword_match(self, query: str, top_k: int) -> list[RegisteredAgent]:
        """简单关键词匹配，作为语义路由的 Phase 1 实现。"""
        scored = []
        query_lower = query.lower()
        for agent in list(self._local.values()) + list(self._remote.values()):
            score = 0
            text = agent.card.to_embedding_text().lower()
            for word in query_lower.split():
                if word in text:
                    score += 1
            if score > 0:
                scored.append((score, agent))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:top_k]]


@dataclass
class RegisteredAgent:
    card: AgentCard
    skill: Skill | None = None        # 本地 Agent 才有
    endpoint: str | None = None       # 远程 Agent 才有
    source: AgentSource = AgentSource.LOCAL
    stats: AgentStats = field(default_factory=lambda: AgentStats())


class AgentSource(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass
class AgentStats:
    """Agent 绩效数据（动态理解层）。"""
    total_calls: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_called: float | None = None
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_calls if self.total_calls > 0 else 0.0
```

### 3. A2A Transport — Server + Client

```
新文件：src/skillengine/a2a/server.py
新文件：src/skillengine/a2a/client.py
```

**Server**（让 SkillEngine Agent 可被外部 A2A 调用）：

```python
class A2AServer:
    """将 SkillEngine 的 Skill 暴露为 A2A 端点。

    端点：
    - GET  /.well-known/agent.json  → Agent Card
    - POST /tasks                   → 创建任务
    - GET  /tasks/{id}              → 查询任务状态
    - POST /tasks/{id}/cancel       → 取消任务
    """

    def __init__(self, engine: SkillsEngine, registry: AgentRegistry):
        self.engine = engine
        self.registry = registry

    def create_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/.well-known/agent.json")
        async def agent_card():
            """返回聚合 Agent Card（包含所有暴露的 Skill）。"""
            cards = [
                a.card.to_dict()
                for a in self.registry._local.values()
                if a.skill and a.skill.frontmatter.get("a2a", {}).get("expose", False)
            ]
            return {"agents": cards}

        @app.post("/tasks")
        async def create_task(request: A2ATaskRequest):
            """接收 A2A 任务 → 路由到对应 Skill → 执行 → 返回结果。"""
            agent = self.registry.get(request.skill_name)
            if not agent or not agent.skill:
                raise HTTPException(404, f"Agent '{request.skill_name}' not found")

            # 复用现有 AgentRunner 执行
            runner = AgentRunner(config=self._config_for_skill(agent.skill))
            result = await runner.chat(request.input_text)

            # 更新绩效统计
            agent.stats.total_calls += 1
            agent.stats.success_count += 1
            agent.stats.last_called = time.time()

            return A2ATaskResponse(
                task_id=request.task_id,
                status="completed",
                output=result.text,
            )

        return app
```

**Client**（调用外部 A2A Agent）：

```python
class A2AClient:
    """调用外部 A2A Agent。

    集成到 AgentRunner 的 tool 体系中，
    让 Orchestrator 像调用本地 Skill 一样调用远程 Agent。
    """

    async def discover(self, agent_url: str) -> AgentCard:
        """GET /.well-known/agent.json 获取远程 Agent Card。"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{agent_url}/.well-known/agent.json")
            return AgentCard.from_dict(resp.json())

    async def send_task(
        self,
        agent_url: str,
        skill_name: str,
        input_text: str,
        timeout: float = 120.0,
    ) -> A2ATaskResponse:
        """POST /tasks 发送任务到远程 Agent。"""
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{agent_url}/tasks",
                json={
                    "skill_name": skill_name,
                    "input_text": input_text,
                },
            )
            return A2ATaskResponse.from_dict(resp.json())


def create_a2a_tool(client: A2AClient, registry: AgentRegistry) -> dict:
    """生成一个 LLM tool 定义，让 Orchestrator 可以调用远程 Agent。

    这个 tool 注入 Orchestrator 的 tool 列表，
    LLM 调用时自动路由到对应的远程 Agent。
    """
    return {
        "name": "call_remote_agent",
        "description": "调用远程 Agent 执行任务。可用 Agent：\n"
                       + registry.cards_summary(budget=2000),
        "parameters": {
            "agent_name": {"type": "string", "description": "目标 Agent 名称"},
            "task": {"type": "string", "description": "任务描述"},
        },
    }
```

### 4. Orchestrator 集成 — 理解下游 Agent

```
修改文件：src/skillengine/agent.py
```

在现有 `AgentRunner` 中集成 Registry，实现两层理解：

```python
class AgentRunner:
    def __init__(self, config: AgentConfig):
        ...
        # 新增：Agent Registry
        self.registry = AgentRegistry()

    async def setup(self):
        """启动时加载所有 Skill 并注册到 Registry。"""
        # 加载本地 Skills
        snapshot = self.engine.load()
        for skill in snapshot.skills:
            self.registry.register_skill(skill, base_url=self.config.a2a_base_url)

        # 发现远程 Agents（如果配置了）
        for url in self.config.a2a_remote_agents:
            try:
                self.registry.register_remote(url)
            except Exception as e:
                logger.warning(f"Failed to discover agent at {url}: {e}")

        # 将 Agent Card 摘要注入 system prompt（静态理解层）
        self._inject_agent_awareness()

    def _inject_agent_awareness(self):
        """将所有已注册 Agent 的能力摘要注入 system prompt。

        这是「Orchestrator 理解下游 Agent」的核心实现：
        LLM 在每次对话时都能"看到"所有可用 Agent 的能力描述。
        """
        summary = self.registry.cards_summary(budget=4000)
        if summary:
            awareness_block = (
                "\n\n## Available Agents\n"
                "以下 Agent 可通过 `skill` 工具或 `call_remote_agent` 工具调用：\n"
                f"{summary}\n"
                "根据用户意图选择最合适的 Agent。"
                "如果任务涉及多个 Agent 的能力，优先选择能力覆盖更广的那个。\n"
            )
            self.config.system_prompt += awareness_block
```

### 5. 与 Claude Agent SDK 的桥接

```
新文件：src/skillengine/a2a/claude_sdk_bridge.py
```

```python
class ClaudeSDKBridge:
    """将 SkillEngine Skill 桥接为 Claude Agent SDK 的执行模式。

    实现「一套 Skill 资产，两种运行模式」：
    - 模式 1：SkillEngine 原生 AgentRunner（自有 agent loop）
    - 模式 2：Claude Agent SDK query()（借用 Claude 的 agent loop）
    """

    def __init__(self, engine: SkillsEngine):
        self.engine = engine

    async def run_skill_via_sdk(
        self,
        skill_name: str,
        input_text: str,
        **sdk_options,
    ) -> str:
        """通过 Claude Agent SDK 执行一个 Skill。

        1. 加载 SKILL.md 作为 system prompt
        2. allowed-tools 映射为 SDK 的 allowed_tools
        3. model 映射为 SDK 的 model
        4. 调用 claude_agent_sdk.query() 执行
        """
        from claude_agent_sdk import query, ClaudeAgentOptions

        skill = self.engine.get_skill(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")

        options = ClaudeAgentOptions(
            system_prompt=skill.content,
            allowed_tools=skill.allowed_tools or None,
            model=skill.model or None,
            permission_mode="acceptEdits",
            max_turns=sdk_options.get("max_turns", 20),
            **sdk_options,
        )

        result_text = []
        async for message in query(
            prompt=input_text,
            options=options,
        ):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text.append(block.text)

        return "\n".join(result_text)

    def create_sdk_agent_from_skill(self, skill_name: str) -> dict:
        """将 Skill 导出为 Claude Agent SDK 的 AgentDefinition。

        可直接注入 ClaudeAgentOptions.agents 字段，
        作为 Orchestrator 的子代理使用。
        """
        skill = self.engine.get_skill(skill_name)
        return {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.content,
            "tools": skill.allowed_tools,
            "model": skill.model or "haiku",
        }
```

---

## 架构全景

```
                    ┌─────────────────────────────┐
                    │      SKILL.md 资产层          │
                    │  (唯一真相源, Markdown+YAML)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      SkillEngine Engine         │
                    │  Loader → Filter → Engine    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌─────────▼─────────┐
    │  AgentRunner       │ │  Claude SDK    │ │  A2A Server       │
    │  (SkillEngine 原生)   │ │  Bridge        │ │  (HTTP 暴露)      │
    │                    │ │                │ │                    │
    │  EventBus          │ │  query()       │ │  /.well-known/    │
    │  Scheduler         │ │  stream()      │ │  /tasks           │
    │  Memory            │ │                │ │                    │
    └─────────┬─────────┘ └───────┬────────┘ └─────────┬─────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Agent Registry          │
                    │                              │
                    │  ┌──────────┐ ┌──────────┐  │
                    │  │ Local    │ │ Remote   │  │
                    │  │ (Skill)  │ │ (A2A)    │  │
                    │  └────┬─────┘ └────┬─────┘  │
                    │       └─────┬──────┘        │
                    │        AgentCard             │
                    │    + Stats + Routing         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Orchestrator            │
                    │                              │
                    │  静态理解: Card → SysPrompt   │
                    │  动态理解: Stats → 绩效路由   │
                    │  语义路由: Intent → Agent     │
                    └─────────────────────────────┘
```

---

## 文件清单

```
src/skillengine/a2a/
├── __init__.py
├── agent_card.py        # AgentCard, AgentCardSkill, AgentCapabilities
├── registry.py          # AgentRegistry, RegisteredAgent, AgentStats
├── server.py            # A2AServer (FastAPI app)
├── client.py            # A2AClient + create_a2a_tool
├── claude_sdk_bridge.py # ClaudeSDKBridge
└── models.py            # A2ATaskRequest, A2ATaskResponse

# 修改
src/skillengine/agent.py    # AgentRunner 集成 Registry + awareness injection
src/skillengine/models.py   # Skill 新增 a2a frontmatter 解析
src/skillengine/loaders/    # MarkdownSkillLoader 解析 a2a: 块
src/skillengine/config.py   # AgentConfig 新增 a2a_base_url, a2a_remote_agents

# 测试
tests/test_agent_card.py
tests/test_registry.py
tests/test_a2a_server.py
tests/test_claude_sdk_bridge.py
```

---

## 实施路径

### Phase 1：Agent Card + Registry（1 周）

**目标**：让 Orchestrator "看见"所有 Agent 的能力

1. 实现 `AgentCard.from_skill()` — 从现有 Skill 自动生成
2. 实现 `AgentRegistry` — 注册 + 查询 + `cards_summary()`
3. 修改 `AgentRunner.setup()` — 自动注册所有 Skill，注入 system prompt
4. 验证：Orchestrator 能根据用户意图匹配正确的 Skill

**验证标准**：
```python
engine = SkillsEngine(dirs=["./skills"])
runner = AgentRunner(config)
await runner.setup()

# system prompt 中包含所有 Agent 的能力摘要
assert "twitter-analyze" in runner.config.system_prompt
assert "mcp-review" in runner.config.system_prompt

# 意图匹配
matches = runner.registry.match("帮我看看最近有没有适合露营的天气")
assert matches[0].card.name == "outdoor-planner"  # 而不是 "weather"
```

### Phase 2：A2A Server + Claude SDK Bridge（2 周）

**目标**：Skill 可被外部调用 + 可通过 Claude SDK 运行

1. 实现 `A2AServer` — `/.well-known/agent.json` + `/tasks`
2. 实现 `ClaudeSDKBridge.run_skill_via_sdk()` — SKILL.md → SDK query
3. Loader 支持解析 `a2a:` frontmatter 块
4. 验证：外部系统通过 HTTP 调用 SkillEngine Agent

**验证标准**：
```bash
# 启动 A2A Server
skillengine serve --port 8080

# 外部发现
curl http://localhost:8080/.well-known/agent.json
# → 返回所有 expose: true 的 Agent Card

# 外部调用
curl -X POST http://localhost:8080/tasks \
  -d '{"skill_name": "read-tweet", "input_text": "https://x.com/..."}'
# → 返回推文内容
```

### Phase 3：A2A Client + 绩效路由（1 月）

**目标**：调用外部 Agent + 基于绩效的智能路由

1. 实现 `A2AClient` — 发现 + 调用远程 Agent
2. 实现 `AgentStats` — 记录每次调用的成功率、延迟
3. Registry.match() 加入绩效权重：能力匹配 × 成功率 × 速度
4. 验证：Orchestrator 能动态发现并调用外部 A2A Agent

---

## 与现有 Roadmap 的关系

本方案与 `ROADMAP.md` 的对应：

| Roadmap 项 | 与 A2A 方案关系 |
|---|---|
| P0 Event System | **已完成** — A2A Server 可 emit task 事件 |
| P0 Structured Stream | A2A 响应可走 SSE 流式 |
| P1 Model Registry | Agent Card 的 `cost_hint` + `model` 可对接 |
| P1 Context Management | 跨 Agent handoff 时需要上下文传递 |
| P2 Steering & Abort | A2A `/tasks/{id}/cancel` 对应 abort |
| P3 Dynamic Provider | A2A Client 本质就是一个动态 "provider" |

**不冲突，互相增强。**
