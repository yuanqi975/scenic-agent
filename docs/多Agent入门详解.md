# 多 Agent（Multi-Agent）入门详解（零基础新手向）

> 专为你这个「九寨沟景区智能服务 Agent」项目定制。你项目的技术栈是 **Python + FastAPI + PostgreSQL(pgvector) + Redis + arq**。
> 这是一份渐进式教程：先讲清楚概念，再讲架构模式，最后**手把手告诉你怎么把你现在的代码改成多 Agent**。
> 配套阅读：`docs/核心功能详解.md`（讲你现在系统怎么跑的）、`docs/ORM入门教程.md`（讲数据库）。

---

## 第 0 章：先给你一句话结论

**你现在这个项目，其实已经"长得像"多 Agent 了，但骨子里还是一个 Agent。**

证据就在你的代码里（`backend/app/main.py` 第 311 行）：

```python
save_run(cid, intent, {"qa": "qa_agent", "recommendation": "recommendation_agent",
                       "feedback": "feedback_agent", "realtime": "qa_agent"}[intent], ...)
```

你数据库 `agent_runs` 表里已经存了 `agent_name`，有 `qa_agent`、`recommendation_agent`、`feedback_agent` 三个名字。
听起来像三个 Agent 在协作，**但实际上**它们只是同一个 `answer_sync()` 函数里 `if/elif/else` 的三个分支：

```python
if intent == "qa":              # ← 这里
    ...
elif intent == "recommendation": # ← 和这里
    ...
else:                            # ← 和这里
    ...
```

这叫 **"命名上的多 Agent，实现上的单 Agent"**。

> 💡 这不是批评，反而是**教科书级别正确的起点**。业界的共识是：**先用最简单的方式把系统跑通，等到真的遇到瓶颈再加 Agent。**
> 这份文档的目标，就是告诉你：**什么时候该加、加什么、怎么加、加了会踩什么坑。**

---

## 第 1 章：先搞清楚 "Agent" 到底是什么

### 1.1 一个公式

新手最容易把 Agent 想玄乎。其实它就是一个公式：

```
Agent = 目标(Goal) + 工具(Tools) + 记忆(Memory) + 循环(Loop)
```

对照你项目里的 `retrieve()` 函数（第 238 行），它**不是** Agent，它是一个**工具**：

```python
def retrieve(message: str, top_k: int = 5) -> list[dict[str, Any]]:
    """RAG retrieval: pgvector first, then PostgreSQL FTS, then a Chinese-safe ILIKE fallback."""
```

它输入字符串、输出列表，中间不"思考"、不"决策"、不"循环"。这就是个纯函数。

那什么算 Agent？看 `classify()`（第 200 行）：

```python
def classify(message: str) -> str:
    if any(word in message for word in ("路线", "怎么玩", "老人", "儿童", "孩子", "半天", "一天")):
        return "recommendation"
    if any(word in message for word in ("反馈", "投诉", "建议", "损坏")):
        return "feedback"
    if any(word in message for word in ("天气", "客流", "关闭", "开放状态")):
        return "realtime"
    return "qa"
```

**这个更接近 Agent 的雏形**——它在做"判断"，决定后面走哪条路。你自己在文档里也写了对它的评价：

> 这就是一个小型的"关键词路由 Agent"

所以：

| 组件 | 你项目里的东西 | 是 Agent 吗 | 为什么 |
|------|----------------|-------------|--------|
| `retrieve()` | RAG 三层漏斗检索 | ❌ 工具 | 纯输入输出，无决策 |
| `build_recommendation()` | 贪心算法排路线 | ❌ 工具 | 纯确定性算法 |
| `embed_query()` | 调 Embedding API | ❌ 工具 | 纯 IO |
| `classify()` | 关键词路由 | ⚠️ 半个 | 有决策，但规则写死 |
| `answer_sync()` | 完整问答流程 | ✅ 最接近 | 决策 + 调工具 + 记忆 + 落审计 |

**记住这条线**：能"自己做决定下一步干什么"的，才是 Agent；只能"被叫去做一件固定事"的，是工具。

---

### 1.2 最重要的一课：Workflow（工作流）≠ Agent（智能体）

这是新手**最需要**先分清的一组概念。业界（Anthropic、OpenAI 的官方指南都在反复强调）把它总结成一句话：

> **能用工作流解决的，就别用 Agent。**

| | Workflow 工作流 | Agent 智能体 |
|---|---|---|
| **路径** | 提前写死的（代码里 if/else、DAG 图） | 运行时由模型自己决定 |
| **可预测性** | 高，跑 100 次结果一样 | 低，每次可能走不同的路 |
| **成本** | 低（能精确算） | 高且不稳定（可能绕远路） |
| **调试** | 简单，看断点 | 难，要看 trace |
| **适合** | 流程固定、步骤明确的业务 | 开放式、步骤不确定的任务 |
| **你项目里的例子** | `answer_sync()` 整个流程 | （目前还没有真正的） |

**你现在整套系统 100% 是 Workflow。这是对的。**

为什么？因为"景区客服问答"这个业务的路径本来就是固定的：

```
收问题 → 判意图 → 检索 → 生成 → 存库 → 返回引用
```

这条链**永远**是这样。你用 Agent 去做这件事，只会让它变得更贵、更慢、更不可控，**不会更准**。

那什么时候才真的需要 Agent？当你遇到**"下一步干什么，事先没法用 if/else 写清楚"**的任务时。比如：

- "帮我规划一个 3 天行程，要考虑我妈腿不好、孩子要看猴子、预算 500 块" → 需要**多步推理 + 反复权衡 + 调多个数据源**
- "分析最近 100 条游客反馈，找出最该优先修的 3 个设施问题" → 需要**探索式地翻数据**
- "游客问的问题我知识库里没有，去网上查、核对、再决定要不要入库" → 需要**自主决定查什么**

**判断口诀**：
> 如果你能画出流程图，且流程图上的每个菱形判断都能用代码写出来 → **用 Workflow**。
> 如果流程图上有你不知道怎么写的菱形 → **那才需要 Agent**。

---

### 1.3 单 Agent 的天花板

那既然 Workflow 够用，为什么还要学多 Agent？

因为当业务变复杂，**单 Agent 会撞到四面墙**：

**墙 1：上下文窗口 / 注意力稀释**
你把 200 个景点、50 个设施、300 条知识文档、10 个工具全塞给一个 Agent 的 prompt，它反而会变笨——关键信息被淹没，这叫 **"lost in the middle"**。
> 你的 `retrieve()` 其实就是在解决这个问题（只取 top_k=5），只不过解决在"检索层"而不是"Agent层"。

**墙 2：工具太多，选错概率飙升**
工具从 5 个涨到 30 个，模型选错工具的概率会明显上升。

**墙 3：权限无法隔离**
一个能查票务、能改库存、能发退款的"全能 Agent"非常危险。如果拆开，"票务 Agent"只能读不能写，"反馈 Agent"只能写不能读别的，安全性立刻提升。

**墙 4：没法并行**
游客问"推荐路线 + 顺便告诉我今天天气 + 有没有轮椅租借"，单 Agent 只能一件事一件事做。多 Agent 可以三个同时跑。

**这四条，就是多 Agent 存在的全部理由。** 如果一条都不占，**别用**。

---

## 第 2 章：什么是多 Agent

### 2.1 定义

> **多 Agent 系统 = 多个各自有角色、工具、记忆的 Agent，通过某种编排方式协作完成一个任务。**

关键词是**"编排"（Orchestration）**——把几个 Agent 组织起来干活的那套机制。这也是多 Agent 里最难、最值钱的部分。

### 2.2 四个真实动机（对号入座）

| 动机 | 说明 | 你项目的潜在场景 |
|------|------|------------------|
| **专业分工** | 每个 Agent 只干一件事，prompt 更短更专注 | 路线 Agent 只懂排路线，票务 Agent 只懂票 |
| **上下文隔离** | 各干各的，互不污染 | 检索 Agent 看 500 篇文档，答题 Agent 只看到 5 条 |
| **并行提速** | 独立子任务同时跑 | 天气 + 路线 + 设施查询同时发 |
| **权限/风险隔离** | 危险操作只给一个 Agent | 只有"运营 Agent"能写数据库 |

### 2.3 ⚠️ 什么时候**不该**用多 Agent（新手必读）

这一段比前面都重要。新手 90% 的翻车都在这里。

**❌ 不要因为"听起来高级"就用。** 多 Agent 的代价是真实的：

- **成本**：一次游客提问，原本 1 次 LLM 调用（约 1 秒、1 分钱）；拆成 4 个 Agent = 4~8 次调用（3~6 秒、5~10 分钱）。
- **延迟**：你现在 `/api/v1/chat/stream` 的体验是"秒回"。多 Agent 串行后，游客要多等好几秒。
- **失败面**：1 个 Agent 有 95% 成功率，4 个串联 = 0.95⁴ ≈ 81%。**可靠性会相乘性下降！**
- **调试难度**：出错时你不知道是哪个 Agent 的锅。

**明确的反模式清单：**

| 反模式 | 症状 | 正确做法 |
|--------|------|----------|
| **为拆而拆** | 每个 Agent 只有一句话 prompt，没有独立工具和记忆 | 合并成一个 |
| **纯线性还拆** | A→B→C 顺序执行，没有任何判断 | 这就是 Workflow，写成函数就行 |
| **Agent 数量当 KPI** | "我们有 12 个 Agent！" | 数量不是指标，**任务成功率**才是 |
| **没有评测就上多 Agent** | 改完不知道是变好了还是变坏了 | 先建评测集（见 §11.2） |

> 🎯 **给你项目的具体建议**：先按本文件第 10 章**只做 Step 1 和 Step 2**。这两步能拿到 80% 的收益（分工清晰 + 可观测），而复杂度和成本只增加一点点。Step 3/4 是"等你业务真的变复杂了再说"。

---

## 第 3 章：多 Agent 的解剖学

一个 Agent 由 6 个部件组成。我们用你项目的真实代码逐个对照。

### 3.1 角色 / 人设（Role / Persona）

就是一段系统提示词，告诉它"你是谁、你负责什么、你不能干什么"。

你项目里现在有这段（第 272 行），但它是**写死的、只有一种角色**：

```python
prompt = "你是九寨沟景区官方客服，只能依据给定资料回答。资料：\n" + ... + f"\n问题：{message}"
```

多 Agent 化后，会变成**每个 Agent 一份**：

```python
ROLES = {
    "router":   "你是意图识别器。只输出 JSON，不要解释。",
    "route":    "你是路线规划师。你只负责根据景点时长和人群偏好排路线，不回答其他问题。",
    "ticket":   "你是票务专员。只回答门票、优惠、退改签问题。资料里没有就说不确定。",
    "facility": "你是设施向导。只回答卫生间、轮椅、餐饮、停车场问题。",
    "critic":   "你是质检员。检查答案是否忠于资料，找出无依据的断言。",
}
```

**新手要点**：角色 prompt 里**"不要做什么"比"要做什么"更重要**。这叫"负向约束"，能显著减少 Agent 越界。

### 3.2 工具（Tools）

工具 = Agent 能调用的函数。你项目里现成的工具：

| 你的函数 | 能当什么工具 | 参数 | 返回 |
|----------|--------------|------|------|
| `retrieve(message, top_k)` | `knowledge_search` | message, top_k | 知识片段列表 |
| `build_recommendation(req)` | `plan_route` | 时长/人群/偏好/天气 | 景点列表 |
| `payload_rows("attractions")` | `list_attractions` | limit | 景点列表 |
| `exact_answer(message)` | `attraction_lookup` | message | 精确匹配答案 |
| `payload_rows("facilities")` | `list_facilities` | limit | 设施列表 |

> 💡 **工具设计三原则**（新手最容易忽略）：
> 1. **描述要像写给新人看的说明书**——模型只靠描述决定用不用它。
> 2. **参数越少越好**——3 个以上参数的工具体积就太大了。
> 3. **返回要短**——返回 50KB JSON 会把上下文撑爆。你 `payload_rows` 的 `limit` 参数就是在防这个。

### 3.3 记忆（Memory）

分两种，别搞混：

| 类型 | 存什么 | 存哪儿 | 你项目里的对应 |
|------|--------|--------|----------------|
| **短期记忆** | 当前这次对话的历史 | 内存 / 消息列表 | `conversation_messages` 表 |
| **长期记忆** | 跨会话沉淀的知识 | 向量库 / 文档库 | `documents` + `document_chunks` |

**关键点**：多 Agent 系统里，**记忆往往是共享的，上下文是隔离的**。

举例：路线 Agent 和票务 Agent 都读同一个 `conversation_messages`（共享记忆），但它们各自的 prompt 里只装自己需要的那几条（隔离上下文）。

> ⚠️ 你项目现在有个**真实的小问题**值得注意：`answer_sync()` 里生成回答时**根本没读历史消息**——它只用了 `request.message` 和检索结果。所以游客问"那它几点关门？"（指代上一轮的"它"），系统是接不住的。
> 这正好是引入多 Agent 之前**更该先修**的一个点（详见 §10.3 Step 0）。

### 3.4 状态（State）

多 Agent 协作时，"大家共同操作的那份数据"叫状态。它记录：现在做到哪一步了、已经产出了什么、谁该接着做。

- 在 **LangGraph** 里，状态是一个显式的 `TypedDict`，每个节点读它、改它。
- 在**你项目**里，天然的共享状态就是 **JSON**——因为你的表全是 `payload jsonb`（详见 `docs/核心功能详解.md` 第 4.4 节）。

```python
# 你可以直接用一个 dict 当"黑板"
state = {
    "trace_id": uuid.uuid4().hex,
    "conversation_id": cid,
    "user_message": request.message,
    "intent": None,
    "retrieved": [],
    "answer": None,
    "steps": [],        # 每个 Agent 干完都往这里追加，用于审计
    "critique": None,
}
```

### 3.5 消息（Message）

Agent 之间怎么说话？工业界事实标准是 OpenAI 的三种角色格式：

```python
{"role": "system",    "content": "你是路线规划师..."}
{"role": "user",      "content": "帮我排半天的路线"}
{"role": "assistant", "content": "我建议先去..."}
```

多 Agent 里会多出两种约定俗成的角色：

- `tool` / `function`：工具返回结果
- `agent_name`（有些框架用）：标明这条消息是哪个 Agent 说的

> **你项目将来如果上多 Agent，`conversation_messages` 表建议加一列 `agent_name`**，不然审计时分不清哪句话是谁说的。

### 3.6 编排器（Orchestrator）

**这是多 Agent 的心脏。** 它决定：
- 任务怎么分给各个 Agent
- 按什么顺序
- 谁的结果给谁
- 什么时候停

**新手心态**：编排器**最好自己写**（用普通 Python 代码），而不是交给 LLM 去"自由发挥"。原因见第 7 章。

---

## 第 4 章：六种经典拓扑（架构模式）

下面六种是业界公认的模式。每种我都给你配了图、适用场景、和**你项目的映射建议**。

### 4.1 流水线 Pipeline（顺序接力）

```
游客问题 → [意图Agent] → [检索Agent] → [生成Agent] → [质检Agent] → 回答
```

- **特点**：最简单，每步的输出是下一步的输入。
- **适用**：步骤固定、无分支。
- **优点**：好懂、好调、可预测。
- **缺点**：**这其实是 Workflow，不是真正的多 Agent！** 如果中间没有"判断下一步去哪"，别叫它多 Agent。
- **你的用法**：可以作为**起步形态**——把现在的 `answer_sync()` 一个函数体拆成 4 个独立函数，每个函数配自己的 prompt。

> ⚠️ 常见误解：很多人把"把一个大函数拆成 4 个小函数"叫做"上了多 Agent"。如果拆完还是固定顺序调用，那**架构上什么都没变**，只是代码好看了。真正的分水岭是 **"谁来决定下一步"**。

### 4.2 主管模式 Supervisor（中心化调度）⭐ 新手首选

```
                    ┌──────────────┐
                    │  Supervisor  │  ← 主管：只负责"决定谁去干"
                    │   (主管)      │     自己不干活
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 路线Agent │ │ 票务Agent │ │ 设施Agent │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┴────────────┘
                           ▼
                   结果回到 Supervisor
                   （决定：还要叫人吗？还是结束？）
```

- **特点**：一个"主管"Agent 读任务 → 选一个下属 → 下属干完汇报 → 主管再决定下一步或结束。
- **适用**：**绝大多数业务场景**。任务有分支但整体可控。
- **优点**：
  - 控制流集中，**容易调试**（你只要盯住主管的决策日志）
  - 好加"最多循环 N 次"的保护
  - 好做权限隔离
- **缺点**：主管是单点，主管的 prompt 决定一切；延迟是串行的。
- **参考实现**：LangGraph 的 `langgraph-supervisor` 包就是专门做这个的。

**这是我最推荐你采用的模式。** 对你项目的映射：

```python
SUPERVISOR_SYSTEM = """
你是九寨沟景区客服调度主管。根据游客问题，决定交给哪个专员处理。
可选专员：route_expert(路线) / ticket_expert(票务) / facility_expert(设施) / knowledge_expert(知识问答) / feedback_expert(反馈受理)
如果问题已经解决，回复 FINISH。
只输出 JSON: {"next": "专员名或FINISH", "reason": "一句话理由", "task": "交给它的具体指令"}
"""
```

然后主管就是个循环（伪代码）：

```python
async def run_supervisor(state: dict) -> dict:
    for step in range(MAX_STEPS):          # ← 硬性上限，防死循环
        decision = await call_llm(SUPERVISOR_SYSTEM, state)   # 主管决策
        if decision["next"] == "FINISH":
            break
        worker = WORKERS[decision["next"]]  # 找到对应专员
        result = await worker(state, decision["task"])        # 专员干活
        state["steps"].append({"agent": decision["next"], "task": decision["task"], "result": result})
        save_step(state["trace_id"], step, decision["next"], result)  # ← 审计！
    return state
```

> 🔑 **注意那个 `MAX_STEPS`**。这是多 Agent 系统的安全带。**没有上限的 Agent 循环 = 半夜烧你钱的机器。**

### 4.3 移交 Handoff / Swarm（去中心化）

```
        ┌──────────┐
   ┌───►│ 前台Agent │
   │    └────┬─────┘
   │         │ "这个我不懂，转给票务"
   │         ▼
   │    ┌──────────┐
   │    │ 票务Agent │
   │    └────┬─────┘
   │         │ "这属于退款，转给财务"
   │         ▼
   │    ┌──────────┐
   └────┤ 财务Agent │  （也可能转回来 → 死循环风险！）
        └──────────┘
```

- **特点**：没有中心主管，**每个 Agent 自己决定把控制权交给谁**（这叫 handoff）。
- **适用**：客服分诊、Agent 数量多且边界模糊。
- **优点**：灵活，像真人转接电话。
- **缺点**：**容易乒乓死循环**（A 转给 B，B 又转回 A）；全局不可见。
- **参考实现**：OpenAI 的 Agents SDK（前身叫 Swarm）把 handoff 作为一等公民；LangGraph 有 `langgraph-swarm` 包。
- **给新手的忠告**：**先不要用这个模式。** 等你把 Supervisor 玩熟了再考虑。

### 4.4 层级模式 Hierarchical（主管的主管）

```
                 ┌─────────────┐
                 │  总主管      │
                 └──┬───────┬──┘
            ┌───────┘       └───────┐
            ▼                       ▼
     ┌────────────┐          ┌────────────┐
     │ 客服组主管  │          │ 运营组主管  │
     └──┬──────┬──┘          └──┬──────┬──┘
        ▼      ▼                ▼      ▼
     [问答]  [票务]          [报表]  [审核]
```

- **特点**：Supervisor 套 Supervisor。适合组织级、上百个 Agent 的场景。
- **优点**：可扩展到很大规模，职责分层清晰。
- **缺点**：**延迟按层数叠加**，成本也叠加；调试像剥洋葱。
- **给新手的忠告**：你的项目在很长一段时间里**根本用不到**。

### 4.5 反思 / 辩论 Debate & Reflection

```
    生成Agent ──输出──┐
                     ▼
              ┌─────────────┐
              │  Critic(质检)│  "第2句资料里没有依据"
              └──────┬──────┘
                     │ 反馈
                     ▼
              生成Agent 重写 ──► 再质检 ──► 通过 / 超过轮数上限则放弃
```

- **特点**：一个 Agent 产出，另一个 Agent 挑错，循环几轮。
- **适用**：**对准确率要求高**的场景——正好命中你的景区客服（答错门票价格是事故！）。
- **优点**：显著降低幻觉。
- **缺点**：成本翻倍、延迟翻倍。轮数必须封顶（**通常 1~2 轮就够，超过就没收益了**）。
- **你的项目的绝配用法**：

```python
CRITIC_SYSTEM = """
你是质检员。给定【资料】和【答案】，检查答案是否每一句都能在资料里找到依据。
输出 JSON: {"ok": true/false, "unsupported": ["无依据的句子"], "fixed": "修正后的答案"}
禁止自己补充资料里没有的信息。
"""
```

> 🎯 **对你项目的具体价值**：你 `generate_answer()` 第 272 行的 prompt 里已经写了"只能依据给定资料回答"——但**模型经常不听**。加一个 Critic 是"把软约束变成硬检查"，这是性价比最高的一个多 Agent 升级。

### 4.6 黑板模式 Blackboard（共享状态协作）

```
   ┌───────────────────────────────────────┐
   │         共享黑板 (shared state)         │
   │  { 意图, 检索结果, 答案草稿, 质检意见 }   │
   └───▲────────▲────────▲────────▲─────────┘
       │        │        │        │
   [Agent A] [Agent B] [Agent C] [Agent D]
   （谁觉得该自己出手，就去黑板上写一笔）
```

- **特点**：Agent 之间不直接说话，通过共享数据结构间接通信。
- **适用**：Agent 数量不定、任务不确定。
- **优点**：解耦彻底，加 Agent 不用改别人。
- **缺点**：**谁该出手？什么时候结束？** 这两个问题很难答，容易变成"大家都不动"或"大家一起动"。
- **你的项目的映射**：你的 `payload jsonb` 设计天然适合做黑板。第 3.4 节那个 `state` dict 就是个小黑板。

### 4.7 六种模式总对比

| 模式 | 控制流 | 难度 | 调试 | 成本 | 死循环风险 | 新手推荐度 |
|------|--------|------|------|------|-----------|-----------|
| 流水线 Pipeline | 固定顺序 | ⭐ | 易 | 低 | 无 | ⭐⭐⭐⭐⭐（但别叫它多 Agent） |
| **主管 Supervisor** | 中心调度 | ⭐⭐ | 易 | 中 | 低（有上限） | ⭐⭐⭐⭐⭐ **首选** |
| 移交 Swarm/Handoff | 去中心 | ⭐⭐⭐⭐ | 难 | 中 | **高** | ⭐ |
| 层级 Hierarchical | 多层调度 | ⭐⭐⭐⭐ | 难 | 高 | 中 | ⭐⭐ |
| 反思 Debate | 循环校验 | ⭐⭐ | 中 | 高 | 中 | ⭐⭐⭐⭐ **性价比最高** |
| 黑板 Blackboard | 事件驱动 | ⭐⭐⭐⭐⭐ | 极难 | 不定 | 中 | ⭐ |

---

## 第 5 章：两种"让 Agent 调另一个 Agent"的方式（最容易搞混）

这是多 Agent 里**概念上最绕**的一点，但业内已有清晰的区分（OpenAI Agents SDK 的官方文档就是这么分的）。

### 5.1 Handoff（移交 / 转接）

```
游客 ──► A Agent ──「我不擅长，你来」──► B Agent ──► 游客
                  （A 交出控制权，A 退场）
```

- 控制权**转移**了。
- 游客接下来是在跟 **B** 对话。
- 适合：**客户服务转接**——"退款问题？我帮您转财务专员"。

### 5.2 Agents-as-Tools（把 Agent 当工具调用）

```
游客 ──► 主管 Agent ──调用──► [B Agent 当工具] ──返回结果──► 主管 Agent ──► 游客
                  （B 干完活就退场，主管始终在场）
```

- 控制权**没转移**，主管一直是主线。
- B 的产出只是一个"返回值"，主管可以继续加工、再叫别人、或者直接回答。
- 适合：**一个大脑统筹多个专家**。

### 5.3 对比表（直接抄下来）

| 维度 | Handoff 移交 | Agents-as-Tools 当工具用 |
|------|-------------|------------------------|
| 控制权 | 转给别人，自己退场 | 始终在自己手里 |
| 谁最后对游客说话 | 接手的那位 | 主管 |
| 上下文 | 通常跟着转过去 | 只把结果摘要带回来 |
| 适合 | 分诊、转接、路由 | 统筹、汇总、编排 |
| 调试难度 | 难（链路长） | 易（层层返回） |
| **你的项目该用哪个** | 游客明确说"我要转人工" | ✅ **默认用这个** |

> 🎯 **结论**：你的 Supervisor 模式下，**下属 Agent 就应该是"工具"**，而不是接管对话。这样主管能汇总多方结果（路线+票务+天气）后给一个完整答案——这正是景区客服最理想的样子。

---

## 第 6 章：通信与状态怎么传

### 6.1 两种通信风格

| | 消息传递 Message Passing | 共享状态 Shared State |
|---|---|---|
| 怎么工作 | Agent 之间互相发消息 | 大家读写同一块数据 |
| 代表 | AutoGen、OpenAI Agents SDK | LangGraph、黑板模式 |
| 优点 | 直观，像群聊 | 结构清晰，好追踪 |
| 缺点 | 消息会越滚越多，上下文爆炸 | 需要设计好数据结构 |
| 你的项目 | 已有 `conversation_messages` | 已有 `payload jsonb` / `documents` |

### 6.2 ⚠️ 上下文爆炸：多 Agent 最大的隐形杀手

想象 4 个 Agent 来回对话 10 轮，每轮消息 500 字：

```
每轮 token ≈ 4 个Agent × 500字 ≈ 2000 token
10 轮 = 20000 token
每次调用都要把全部历史塞进 prompt → 成本按 O(n²) 增长
```

**这就是为什么很多多 Agent 项目一上线账单就爆。**

### 6.3 三个实战解法

**解法 1：摘要传递（最常用）**
不要让下游 Agent 看全部历史，只给它一个"交接摘要"。

```python
def handoff_packet(state: dict, limit: int = 1500) -> str:
    """给下一个 Agent 一个精简包裹，而不是全部历史。"""
    return f"""【游客问题】{state['user_message']}
【当前进度】{state['summary']}
【相关资料】{chr(10).join(item['content'][:300] for item in state['retrieved'][:3])}
【你的任务】{state['task']}"""
```

**解法 2：引用而非复制**
把长文档存数据库，Agent 之间只传 `document_id`。你项目的 `citations` 已经是这个思路了：

```python
citations.append({"source_type": "attraction", "source_id": item.get("attraction_id"), ...})
```

**解法 3：硬性截断**
每个 Agent 的输出限制长度，超了就要求它总结。

```python
MAX_AGENT_OUTPUT_CHARS = 800
```

---

## 第 7 章：多 Agent 的坑（这一章请反复读）

### 7.1 错误级联（Error Cascade）

**问题**：A 错了 → B 基于 A 的错误继续 → C 再放大。

有研究专门建模过这件事（见参考文献中 "From Spark to Fire"）。在 Supervisor 结构里，主管的一个错误路由会让后面的全白干。

**对策**：
- 每个 Agent 输出**结构化 JSON + 置信度**，低置信度直接转人工/兜底。
- 关键步骤加**校验**（如你已有 `embedding dimension must be 1536` 这种维度校验，思路完全正确）。
- 别让 Agent"自由发挥"地改数据。

### 7.2 无限循环 / 乒乓（Ping-Pong）

**问题**：A→B→A→B…… 永远结束不了。在 Handoff 模式里尤其常见。

**对策**（三件套，一个都不能少）：
```python
MAX_STEPS = 6              # ① 步数上限
MAX_SECONDS = 20           # ② 时间上限
MAX_TOKENS = 50000         # ③ token 上限
```
超限就**降级到你现在那套规则回答**——你已经有了这个兜底能力（`LLM_MODE=fallback`），太合适了。

### 7.3 成本爆炸

**算一笔账**（按你项目的真实场景）：

| 场景 | LLM 调用次数 | 估算延迟 | 相对成本 |
|------|-------------|---------|---------|
| 现在的 `answer_sync` | 1 次 | ~1.5s | 1× |
| + LLM 路由 Agent | 2 次 | ~2.5s | 2× |
| + 检索 Agent + 生成 Agent | 3 次 | ~3.5s | 3× |
| + Critic 质检 | 4 次 | ~5s | 4× |
| Supervisor + 3 个专员 | 5~8 次 | ~8s | 6~8× |

**对策**：
- **能不调 LLM 就不调**：你现在的缓存 `cache_get/cache_set`（第 92 行）已经能省掉大量重复检索。**建议给意图识别也加缓存**——游客的问法重复率极高。
- **分级**：简单问题走规则，复杂问题才走多 Agent。
- **用小模型做路由**：路由/分类任务用便宜的小模型就够，别用最贵的模型做"选择题"。

### 7.4 责任不清 & 不可观测

**问题**：答案错了，你完全不知道是哪个 Agent 的锅。

**对策**：**trace（调用链追踪）**。好消息是——**你的 `agent_runs` 表已经是半个 trace 了！** 详见第 8 章。

### 7.5 幻觉互相强化（最阴险的坑）

**问题**：A 说"门票 80 元"，B 看到后说"确认门票 80 元"，C 说"三方一致确认"——**其实全是编的**。

这叫 **sycophancy（谄媚）/ 回音室效应**。多 Agent **不会自动提高准确率**，如果设计不当反而会**给幻觉加盖公章**。

**对策**：
- **必须有一个 Agent 只看原始资料，不看其他 Agent 的结论。** Critic 的输入应该是 `(原始资料, 答案)`，**而不是** `(其他 Agent 的答案)`。
- 让 Agent 输出**引用来源**（你项目已经在做 `citations` 了，坚持下去！）。
- 关键事实（票价、开放时间）**永远从数据库直读**，不让 LLM 转述。你现在的 `exact_answer()` 就是这个思路，非常正确。

### 7.6 一个真实存在的小 bug（顺便教你读自己代码）

看你的 `review_candidate()` 第 480~484 行：

```python
# Queueing is best effort: task row allows an operator or worker cron to retry.
try:
    loop = asyncio.get_running_loop()
    loop.create_task(enqueue_embedding_task(task_id, document_id))
except RuntimeError:
    pass
```

这个端点是 `def review_candidate(...)`（同步函数，不是 `async def`）。FastAPI 会把同步端点丢到**线程池**里执行，而线程池的工作线程里**没有正在运行的事件循环**，所以 `asyncio.get_running_loop()` 会抛 `RuntimeError`，被 `except` 吞掉。

→ 结果是：**任务被写进了 `async_tasks` 表，但从来没被推进 Redis 队列。**

代码注释里说"靠运维或 cron 重试"，如果你确实没有这个 cron，那么管理员点"采纳"之后，`embedding` 永远不会被生成——知识只能靠 FTS/ILIKE 被检索到（这也解释了为什么 `docs/核心功能详解.md` 里特别强调"文档立即写入、只有语义检索要等 worker"）。

**验证方法**（不是改代码，先确认）：
```sql
SELECT status, count(*) FROM async_tasks WHERE task_type='embed_document' GROUP BY status;
```
如果一堆 `pending` 且从没变成 `completed`，就说明确实没入队。

**修法**（两种，任选）：
1. 把端点改成 `async def review_candidate(...)`（FastAPI 就会在事件循环里跑，`get_running_loop()` 正常）。但它里面全是同步 SQL，会阻塞事件循环，所以更推荐第 2 种。
2. 保持同步函数，改成"入库即触发"的可靠投递：让 worker 轮询 `async_tasks` 表取 `pending` 任务（**数据库即队列**——这正是你文档里总结的第 3 个设计思想"数据库即任务队列的源真相"）。

> 📌 这个例子想说明的事：**多 Agent 系统会放大这类问题**。单 Agent 时，一个后台任务没入队只是"某个功能慢一点"；多 Agent 时，一个 Agent 静默失败会导致整条链给出错误答案，而且**没人知道**。所以第 8 章的可观测性不是可选项。

---

## 第 8 章：可观测性——你已经有基础了

### 8.1 你的 `agent_runs` 表就是雏形

```sql
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id text primary key,
    conversation_id text,
    park_id text not null,
    intent text not null,                      -- 意图
    agent_name text not null,                  -- 哪个 Agent
    retrieved_document_ids jsonb,              -- 命中的文档
    cache_hit boolean,                         -- 是否命中缓存
    latency_ms integer,                        -- 耗时
    status text not null,                      -- 成功/失败
    error text,                                -- 错误信息
    created_at timestamptz
)
```

**这张表的设计已经包含了 trace 的核心思想**（谁、干了什么、多久、成没成）。点赞 👍

### 8.2 多 Agent 需要补什么

多 Agent 的关键是**父子关系**——一次游客提问会触发多个 Agent，它们属于同一条链。

| 要补的字段 | 作用 | 举例 |
|-----------|------|------|
| `trace_id` | 串起一次完整请求的所有 Agent | `tr_ab12...` |
| `parent_run_id` | 谁调用的我（形成树） | 主管的 run_id |
| `step_index` | 第几步 | 1, 2, 3 |
| `agent_role` | 角色（主管/专员/质检） | `supervisor` |
| `tool_calls` | 调了哪些工具 | `["knowledge_search"]` |
| `tokens_in` / `tokens_out` | token 消耗 | 1200 / 300 |
| `cost_usd` | 花了多少钱 | 0.0021 |

**迁移脚本**（贴合你项目的轻量风格，放在 `backend/alembic/versions/0002_agent_trace.py`）：

```sql
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS trace_id text;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS parent_run_id text;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS step_index integer DEFAULT 0;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS agent_role text;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tool_calls jsonb NOT NULL DEFAULT '[]';
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tokens_in integer DEFAULT 0;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tokens_out integer DEFAULT 0;
CREATE INDEX IF NOT EXISTS agent_runs_trace_idx ON agent_runs(park_id, trace_id, step_index);
```

### 8.3 改造 `save_run()`

你现在的 `save_run()`（第 282 行）签名是这样的：

```python
def save_run(conversation_id: str, intent: str, agent: str, docs: list[str],
             latency: int, cache_hit: bool = False, status: str = "success", error: str | None = None) -> None:
```

多 Agent 化后，建议改成传一个 `step` 对象，避免参数列表爆炸：

```python
def save_step(step: dict[str, Any]) -> None:
    """写一条 Agent 调用记录。step 里带 trace_id / parent_run_id / step_index 等。"""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO agent_runs(run_id, trace_id, parent_run_id, step_index, conversation_id,
                                   park_id, intent, agent_name, agent_role, retrieved_document_ids,
                                   tool_calls, tokens_in, tokens_out, cache_hit, latency_ms, status, error)
            VALUES (:run_id, :trace_id, :parent_run_id, :step_index, :conversation_id,
                    :park_id, :intent, :agent_name, :agent_role, CAST(:docs AS jsonb),
                    CAST(:tool_calls AS jsonb), :tokens_in, :tokens_out, :cache_hit, :latency_ms, :status, :error)
        """), {
            "run_id": uuid.uuid4().hex,
            "trace_id": step.get("trace_id"),
            "parent_run_id": step.get("parent_run_id"),
            "step_index": step.get("step_index", 0),
            "conversation_id": step.get("conversation_id"),
            "park_id": PARK_ID,
            "intent": step.get("intent", "unknown"),
            "agent_name": step.get("agent_name", "unknown"),
            "agent_role": step.get("agent_role"),
            "docs": json.dumps(step.get("retrieved_document_ids", [])),
            "tool_calls": json.dumps(step.get("tool_calls", [])),
            "tokens_in": step.get("tokens_in", 0),
            "tokens_out": step.get("tokens_out", 0),
            "cache_hit": step.get("cache_hit", False),
            "latency_ms": step.get("latency_ms"),
            "status": step.get("status", "success"),
            "error": step.get("error"),
        })
```

### 8.4 要盯的 5 个指标

| 指标 | 怎么看 | 健康值（参考） |
|------|--------|---------------|
| **端到端 P95 延迟** | 你已用 locust 压测 | < 5s |
| **单次请求 Agent 调用次数** | `count(*) group by trace_id` | < 8 |
| **单次请求成本** | `sum(cost_usd) group by trace_id` | 设个预算上限 |
| **质检不通过率** | Critic 返回 `ok=false` 的比例 | 关注趋势，突增=有 bug |
| **兜底率** | `status='fallback'` 的比例 | 越低越好，突增=模型/API 有问题 |

> 💡 你项目的 `agent_runs` 已经有管理端接口 `/api/v1/admin/agent-runs`（第 489 行），前端加个"按 trace_id 展开"的树形视图就能用了。

---

## 第 9 章：框架与协议地图（2025 年底现状）

### 9.1 两个协议，别搞混

| 协议 | 全称 | 解决什么 | 类比 |
|------|------|---------|------|
| **MCP** | Model Context Protocol | **Agent ↔ 工具/数据** 怎么接 | USB 接口：让 Agent 插上各种工具 |
| **A2A** | Agent2Agent Protocol | **Agent ↔ Agent** 怎么互相发现和调用 | 电话总机：让不同厂商的 Agent 互相通话 |

- MCP 由 Anthropic 提出，现已成为连接工具与数据源的事实标准。
- A2A 由 Google 提出，**2025 年 6 月捐赠给 Linux Foundation**（见参考文献），解决的是"跨厂商 Agent 互操作"。

> 🎯 **给你的判断**：你现在**完全不需要** A2A（那是给"跨公司/跨系统 Agent 互通"用的）。MCP 可以留意——如果你将来想让景区系统对接地图服务、天气服务、票务系统，用 MCP 会比每次手写 `httpx.post` 更规范。但**现阶段你自己写 HTTP 调用完全没问题**。

### 9.2 主流框架对比

| 框架 | 出品 | 核心抽象 | 强项 | 适合谁 |
|------|------|---------|------|--------|
| **LangGraph** | LangChain | 图（节点+边+状态） | 精细控制流程、可视化、生态成熟 | 想做**生产级**编排 |
| **OpenAI Agents SDK** | OpenAI | Agent + Handoff + Guardrail | 极简、handoff 一等公民、有 tracing | 快速上手、OpenAI 生态 |
| **Microsoft Agent Framework** | 微软 | Agent + Workflow | 企业级、Azure 集成、有工作流引擎 | .NET/企业环境 |
| **CrewAI** | CrewAI | Crew（角色团队）+ Flow | 概念直觉（像组队）、上手快 | 快速搭原型 |
| **Claude Agent SDK** | Anthropic | Agent + Subagent + Skill | 长任务、工具使用强 | Claude 生态 |
| ~~AutoGen~~ | 微软 | 群聊（GroupChat） | 多 Agent 对话研究的鼻祖 | ⚠️ **已被并入 Microsoft Agent Framework** |

> ⚠️ **重要更新**：微软在 2025 年 10 月发布了 **Microsoft Agent Framework**，把 **AutoGen 和 Semantic Kernel 合并**了（见参考文献）。如果你在网上看到 AutoGen 教程，注意它已经不是微软的主推方向了。

### 9.3 🤔 新手到底该不该上框架？

**我的建议：你现在的项目，先不要上框架。**

理由：

1. **你已经有了一套完整、能跑、能审计的架构**（FastAPI + 裸 SQL + arq）。引入 LangGraph 意味着要重写编排层、引入新的状态管理和依赖，收益不明确。
2. **框架会掩盖原理**。新手最该理解的是"状态怎么传、循环怎么停、错误怎么兜"，这些用 50 行纯 Python 写一遍，比调框架学到的多 10 倍。
3. **你的代码风格本来就是"轻量优先"**——`docs/核心功能详解.md` 最后一句你自己写的：
   > 有 `@app.route` 层没有真正的 Model 类，这是刻意为之的轻量设计，先用 `text()` 裸 SQL。等你熟悉后再慢慢引入 ORM 也不迟。

   **同样的道理完全适用于多 Agent 框架。**

**什么时候该上框架？**
- 你的编排逻辑超过 300 行，自己维护 start→end 的流程开始吃力
- 你需要"人类审批中断后恢复执行"（human-in-the-loop）
- 你需要流程图可视化给非技术人员看
- 你需要持久化执行（进程崩了能从中间恢复）

到那时候，**首推 LangGraph**（因为它的"图 + 显式状态"模型最接近你现在的思维，而且能渐进式引入）。

---

## 第 10 章：把你这个景区项目改造成多 Agent（完整落地方案）

### 10.1 现状诊断

| 现状 | 位置 | 问题 |
|------|------|------|
| `classify()` 用关键词 | 第 200 行 | 不智能："这地方老年人方便吗" 会被判成 `qa` 而不是 `recommendation` |
| 生成回答**没读历史** | 第 269-279 行 | 多轮指代接不住 |
| 一个函数干 4 件事 | `answer_sync` 第 293 行 | 难测、难扩展 |
| `agent_name` 是假的 | 第 311 行 | 审计上分不清真实执行单元 |
| 后台任务静默失败 | `review_candidate` 第 480 行 | 见 §7.6 |

### 10.2 目标架构

```
游客提问
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│  Router Agent（意图路由 — LLM + 关键词兜底）                │
│  输出: {intent, confidence, entities}                     │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Supervisor（调度主管 — 决定叫谁、要不要质检）              │
└──┬────────────┬────────────┬────────────┬────────────────┘
   │            │            │            │
   ▼            ▼            ▼            ▼
┌───────┐  ┌────────┐  ┌─────────┐  ┌──────────┐
│知识专员│  │路线专员 │  │票务专员  │  │设施专员   │   ← Agents-as-Tools
│retrieve│  │rec贪心 │  │documents│  │facilities│
└───┬───┘  └───┬────┘  └────┬────┘  └────┬─────┘
    └──────────┴────────────┴────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Critic 质检    │  ← 只对照原始资料，不看别人结论
            └────────┬───────┘
                     │ 通过 / 重写一次 / 降级
                     ▼
              回答 + citations
                     │
                     ▼
        agent_runs 落 trace（每个 Agent 一条记录）
```

### 10.3 分五步走（每步都能独立上线、独立回滚）

> 🔑 **核心原则：绝不要一次性重写。** 每一步都保持系统可运行，用开关控制（环境变量）。

#### Step 0：先修单 Agent 的硬伤（不打多 Agent 旗号）

**这一步收益最大，且零风险。**

**0-a. 让回答用上历史消息**

```python
def recent_history(conversation_id: str, limit: int = 6) -> list[dict[str, str]]:
    """读最近几轮对话，用于多轮指代（'它几点关门'）。"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT role, content FROM conversation_messages
            WHERE conversation_id=:cid ORDER BY id DESC LIMIT :limit
        """), {"cid": conversation_id, "limit": limit}).all()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
```

然后把 `generate_answer()` 的 messages 从单条变成多条：

```python
async def generate_answer(message: str, context: list[dict[str, Any]],
                          history: list[dict[str, str]] | None = None) -> str:
    if LLM_MODE == "api" and LLM_BASE_URL and LLM_API_KEY:
        async with _llm_sem:
            system = ("你是九寨沟景区官方客服，只能依据给定资料回答。"
                      "资料里没有的信息就说不知道，不要编造。\n资料：\n"
                      + "\n".join(item["content"] for item in context))
            messages = [{"role": "system", "content": system}]
            messages += (history or [])
            messages.append({"role": "user", "content": message})
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                    json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2},
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
    # ↓ 兜底逻辑保持不变（降级优先，你项目的核心思想）
    if context:
        return "根据景区资料：\n" + "\n".join(item["content"] for item in context[:3]) + "\n以上信息来自当前景区知识库，请以现场公告为准。"
    return "暂未在九寨沟景区官方知识库中查询到足够信息，建议咨询游客服务中心。"
```

**0-b. 修掉 §7.6 的后台任务静默失败**（改成 worker 轮询 `async_tasks`，或把端点改异步）。

**验收**：问"那它几点关门？"，能正确理解"它"。

---

#### Step 1：把关键词路由换成 LLM 路由（Router Agent）

保留关键词版做兜底——这是你项目的核心哲学。

```python
ROUTER_SYSTEM = """你是九寨沟景区客服的意图识别器。
把游客问题归类为以下之一：
- qa: 询问景点、门票、开放时间等知识
- recommendation: 要路线、行程、游览建议
- realtime: 问天气、客流、实时开放状态
- feedback: 投诉、建议、报修
- ticket: 退改签、优惠、发票等票务事务
只输出 JSON：{"intent": "...", "confidence": 0.0-1.0, "reason": "不超过20字"}
不要输出任何其他内容。"""

INTENTS = {"qa", "recommendation", "realtime", "feedback", "ticket"}


async def classify_llm(message: str) -> dict[str, Any]:
    """LLM 路由，失败或低置信度时降级到关键词 classify()。"""
    fallback = {"intent": classify(message), "confidence": 0.0, "reason": "keyword-fallback"}
    if not (LLM_MODE == "api" and LLM_BASE_URL and LLM_API_KEY):
        return fallback
    try:
        async with _llm_sem:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": ROUTER_SYSTEM},
                            {"role": "user", "content": message},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                data = json.loads(response.json()["choices"][0]["message"]["content"])
        if data.get("intent") not in INTENTS:
            return fallback
        return {"intent": data["intent"], "confidence": float(data.get("confidence", 0)), "reason": data.get("reason", "")}
    except Exception:
        return fallback   # ← 任何异常都不让主流程挂掉


def classify_with_cache(message: str) -> dict[str, Any]:
    """意图识别加缓存：游客问法重复率很高，这一层能省大量调用。"""
    key = f"intent:{PARK_ID}:{hashlib.sha256(message.strip().lower().encode()).hexdigest()}"
    cached = cache_get(key)
    if cached:
        return cached
    result = asyncio.run(classify_llm(message))
    cache_set(key, result)
    return result
```

**注意三个细节**（都是多 Agent 的通用经验）：
1. `temperature: 0` —— 路由任务要确定性。
2. `response_format: json_object` —— 强制结构化输出。
3. **低置信度就降级** —— 别让模型瞎猜。

**验收**：拿 30 条真实问题测，对比关键词版和 LLM 版的准确率（见 §11.2）。

---

#### Step 2：拆出 Retriever Agent 和 Answer Agent

把 `answer_sync()` 里混在一起的事情分成两个有明确边界的单元。

```python
async def retriever_agent(state: dict[str, Any]) -> dict[str, Any]:
    """检索专员：只负责找资料，不负责说话。"""
    started = time.perf_counter()
    hits = retrieve(state["user_message"], top_k=5)
    state["retrieved"] = hits
    state["steps"].append({
        "trace_id": state["trace_id"], "step_index": len(state["steps"]),
        "agent_name": "retriever_agent", "agent_role": "worker",
        "tool_calls": ["knowledge_search"],
        "retrieved_document_ids": [str(h.get("document_id")) for h in hits],
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "status": "success",
    })
    return state


async def answer_agent(state: dict[str, Any]) -> dict[str, Any]:
    """答题专员：只负责基于资料组织语言。"""
    started = time.perf_counter()
    history = recent_history(state["conversation_id"])
    state["answer"] = await generate_answer(state["user_message"], state["retrieved"], history)
    state["steps"].append({
        "trace_id": state["trace_id"], "step_index": len(state["steps"]),
        "agent_name": "answer_agent", "agent_role": "worker",
        "tool_calls": [],
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "status": "success",
    })
    return state
```

**为什么分这两个有价值？**
- **上下文隔离**：检索 Agent 面对的是**全部 500 篇文档**；答题 Agent 只看到 **5 条**。这正是 §1.3 说的"注意力稀释"的解法。
- **可独立优化**：换 embedding 模型只动 retriever；换文风只动 answer。
- **可独立评测**：检索召回率 ≠ 答案质量，分开才能分别度量。

---

#### Step 3：引入 Supervisor + 领域专员

先写两个通用小助手（后面所有 Agent 都用它俩，**不超过 30 行**）：

```python
async def call_llm_json(system: str, user_content: str, temperature: float = 0) -> dict[str, Any]:
    """调用 LLM 并强制返回 JSON。任何失败都抛异常，由调用方决定怎么降级。"""
    async with _llm_sem:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])


def build_supervisor_input(state: dict[str, Any]) -> str:
    """给主管看的进度摘要——注意只传摘要，不传全部历史（防止上下文爆炸）。"""
    done = "\n".join(
        f"- {s['agent_name']}: {(s.get('result') or '')[:200]}" for s in state.get("steps", [])
    ) or "（还没有专员被派出）"
    return f"""【游客问题】{state['user_message']}
【已完成】{done}
【下一步该做什么？】"""
```

然后才是主管本体：

```python
SUPERVISOR_SYSTEM = """你是九寨沟景区客服调度主管。
可用专员：
- knowledge_expert: 查景区知识（景点、开放时间、门票、规定）
- route_expert: 排游览路线
- facility_expert: 查设施（卫生间、轮椅、餐饮、停车）
- ticket_expert: 票务事务（退改签、优惠、发票）
- realtime_expert: 实时信息（天气、客流、开放状态）
规则：
1. 一次只派一个专员。看完结果再决定下一步。
2. 信息够了就输出 FINISH。
3. 最多派 3 次，超过就必须 FINISH。
只输出 JSON：{"next": "专员名或FINISH", "task": "给该专员的具体指令", "reason": "一句话"}"""


async def supervisor(state: dict[str, Any], max_steps: int = 3) -> dict[str, Any]:
    """中心化调度：主管不干活，只决定顺序。"""
    for step in range(max_steps):
        decision = await call_llm_json(SUPERVISOR_SYSTEM, build_supervisor_input(state))
        if decision.get("next") == "FINISH":
            break
        worker = WORKERS.get(decision["next"])
        if worker is None:
            break
        state["task"] = decision.get("task", state["user_message"])
        state = await worker(state)
    return state
```

配套工具（把现有函数包一层）：

```python
WORKERS = {
    "knowledge_expert": retriever_agent,       # 复用 Step 2 的
    "route_expert": route_agent,               # 包 build_recommendation
    "facility_expert": facility_agent,         # 包 payload_rows("facilities")
    "ticket_expert": ticket_agent,             # 包 retrieve，限定 source_type
    "realtime_expert": realtime_agent,         # 查实时数据
}
```

**⚠️ 务必注意**：`max_steps=3` 是硬编码的安全带，**不要**让主管自己数步数。

---

#### Step 4：加 Critic 质检（性价比最高的一步）

```python
CRITIC_SYSTEM = """你是景区客服质检员。
给你【资料】和【答案】，逐句检查答案是否能在资料中找到依据。
输出 JSON：
{"ok": true/false,
 "unsupported": ["无依据的句子"],
 "fixed": "如果ok为false，给出只使用资料信息的修正答案；否则为空字符串"}
绝对禁止补充资料里没有的信息。"""


async def critic_agent(state: dict[str, Any]) -> dict[str, Any]:
    """质检员：只看原始资料和答案，不看其他 Agent 的中间结论（防幻觉互相强化）。"""
    started = time.perf_counter()
    payload = "【资料】\n" + "\n".join(h["content"] for h in state["retrieved"]) + \
              "\n\n【答案】\n" + (state.get("answer") or "")
    try:
        result = await call_llm_json(CRITIC_SYSTEM, payload)
    except Exception as exc:
        state.setdefault("steps", []).append({
            "trace_id": state["trace_id"], "step_index": len(state["steps"]),
            "agent_name": "critic_agent", "agent_role": "critic",
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "status": "error", "error": str(exc),
        })
        return state          # ← 质检失败不影响主流程（降级优先）
    if not result.get("ok") and result.get("fixed"):
        state["answer"] = result["fixed"]
        state["critic_rewrote"] = True
    state["steps"].append({...})
    return state
```

**只跑一轮。** 两轮以上收益极低而成本线性上涨。

---

#### 改造后的 `answer_sync`

```python
async def answer_async(request: ChatRequest) -> dict[str, Any]:
    state: dict[str, Any] = {
        "trace_id": uuid.uuid4().hex,
        "conversation_id": create_conversation(request.conversation_id),
        "user_message": request.message,
        "retrieved": [], "answer": None, "steps": [],
    }
    route = classify_with_cache(request.message)
    state["intent"] = route["intent"]

    if route["confidence"] < 0.4:
        # 低置信度：走确定性的老路径，不让 Agent 瞎猜
        state = await retriever_agent(state)
        state = await answer_agent(state)
    else:
        state = await supervisor(state)
        if state.get("answer") is None:
            state = await answer_agent(state)
        state = await critic_agent(state)

    persist_and_log(state)
    return {"conversation_id": state["conversation_id"], "message": state["answer"],
            "citations": build_citations(state), "intent": state["intent"],
            "trace_id": state["trace_id"]}
```

> 🔑 注意那个 `confidence < 0.4` 的分支——**这就是"不是所有请求都走多 Agent"的分级策略**，是成本控制的关键。

### 10.4 成本与时延预算（改造前先算清楚）

| 步骤 | 新增 LLM 调用 | 累计延迟 | 累计成本 | 单独上线？ |
|------|--------------|---------|---------|-----------|
| 现状 | 1 | ~1.5s | 1× | — |
| Step 0 修历史 | +0 | ~1.8s | 1.2× | ✅ 立刻做 |
| Step 1 LLM 路由 | +1 | ~2.5s | 2× | ✅ |
| Step 2 拆分 | +0 | ~2.5s | 2× | ✅ |
| Step 3 Supervisor | +1~3 | ~5s | 4~5× | ⚠️ 视业务复杂度 |
| Step 4 Critic | +1 | ~6.5s | 5~6× | ✅ 值得 |

**如果 P95 延迟超过了业务能接受的上限（客服场景建议 < 5s）**，考虑：
- 把 Step 1 路由和 Step 4 质检**并行**（质检可以基于"规则预检"先跑）
- 用流式输出掩盖延迟（**你的 SSE 已经做了！** 第 363 行）
- 上更小更快的模型做路由/质检

### 10.5 前端要不要改？

**基本不用。** 你的 `/api/v1/chat/stream` 接口契约（`status` / `citations` / `token` / `result` 四种事件）可以完全保持不变。

但你**可以增强**——多 Agent 最大的体验红利是"让用户看到思考过程"：

```python
yield f"event: status\ndata: {json.dumps('正在识别人工意图', ensure_ascii=False)}\n\n"
yield f"event: status\ndata: {json.dumps('正在咨询路线专员', ensure_ascii=False)}\n\n"
yield f"event: status\ndata: {json.dumps('正在质检答案准确性', ensure_ascii=False)}\n\n"
```

前端 `frontend/src/api.ts` 的 `streamChat` 已经能分发 `status` 事件了——**改后端就能用，零前端成本**。这一条能极大提升"AI 在思考"的感知，是多 Agent 最划算的副产品。

### 10.6 验收标准

改造完必须能回答这些问题，否则不算成功：

- [ ] 30 条评测问题里，**准确率不低于改造前**（这是底线）
- [ ] P95 延迟 **< 5s**
- [ ] 单次请求成本 **< 设定预算**
- [ ] `agent_runs` 里能按 `trace_id` 还原出完整调用链
- [ ] 关掉 `LLM_API_KEY`（`LLM_MODE=fallback`）后，**系统仍然可用**（守住你项目的降级哲学）
- [ ] 死循环测试：构造一个"永远无法回答"的问题，确认 `max_steps` 能兜住

---

## 第 11 章：学习路径与练习

### 11.1 推荐学习顺序

```
① 读本文件第 1~3 章，能讲清 Agent / Workflow / 工具 / 记忆 的区别
       ↓
② 读本文件第 4 章，能画出 Supervisor 和 Handoff 两种拓扑
       ↓
③ 做练习 1、2（不写代码，只设计）
       ↓
④ 做练习 3（落地 Step 0 + Step 1）
       ↓
⑤ 做练习 4（落地 Step 2 + Step 4，这一步收益最大）
       ↓
⑥ 有需要再做练习 5（Supervisor）
       ↓
⑦ 最后才考虑：要不要上 LangGraph
```

### 11.2 五个递进练习

**练习 1（纯设计，1 小时）**
不看代码，拿纸笔画出你项目的"现状流程图"，然后标注：**哪些菱形判断是代码写死的？哪些是你不知道怎么用代码写的？**
后者才是真正需要 Agent 的地方。**大概率你会发现：一个都没有。** 这个认知本身就很宝贵。

**练习 2（读代码，1 小时）**
找出 `backend/app/main.py` 里所有**能当工具用**的函数（提示：输入简单参数、输出结构化数据、无副作用）。列成表格。
参考答案见 §3.2。

**练习 3（编码，半天）**
落地 Step 0-a（多轮历史）+ Step 1（LLM 路由）。
⚠️ 前提：你需要一个可用的 LLM API。没有就把 `LLM_MODE` 保持 `fallback`，把 `classify_llm` 写成"永远走关键词"的桩，先把**代码结构**搭出来。

**练习 4（编码，1~2 天）⭐ 最有价值**
落地 Step 2（拆 retriever / answer）+ Step 4（Critic）。
**同时建一个 30 题的评测集**：

```python
# tests/eval_questions.py
EVAL_SET = [
    {"q": "九寨沟景区门票多少钱？", "must_contain": ["元"], "source": "ticket"},
    {"q": "带老人玩半天怎么安排？", "must_contain": ["小时", "老人"], "source": "route"},
    {"q": "下雨天有什么推荐？", "must_contain": [], "source": "qa"},
    # ... 凑够 30 条，覆盖 qa/route/ticket/facility/realtime/feedback
]
```

然后写个脚本跑准确率对比（改造前 vs 改造后）。**没有评测集的多 Agent 改造 = 盲改。**

**练习 5（编码，3~5 天）**
落地 Step 3（Supervisor），并加一条**只走规则、不走 Agent** 的快路径，对比两条路径的准确率、延迟、成本。
**如果快路径的表现差不多——恭喜你，你验证了"多 Agent 不是万能的"，这比写出一堆 Agent 更有价值。**

### 11.3 延伸阅读

**概念与最佳实践（强烈推荐，都是官方/权威）**
- Anthropic《Building effective agents》—— 讲 Workflow vs Agent 最清楚的一篇
- OpenAI Agents SDK 编排指南 —— handoff vs agents-as-tools 的权威定义
- LangGraph 多 Agent 文档 —— supervisor / swarm / hierarchical 的参考实现

**协议**
- A2A 协议（Google 捐赠给 Linux Foundation）
- MCP 协议规范

**学术论文（有兴趣再看）**
- "From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration" —— 讲错误级联
- "A Comparative Study of MCP and A2A for Inter-Agent Coordination" —— 讲两个协议的分工

---

## 附录 A：术语对照表

| 英文 | 中文 | 一句话解释 |
|------|------|-----------|
| Agent | 智能体 | 能自己决定下一步干什么的 AI 程序 |
| Multi-Agent System | 多智能体系统 | 多个 Agent 协作 |
| Orchestration | 编排 | 组织多个 Agent 干活的机制 |
| Supervisor | 主管 | 中心化调度者，自己不干活 |
| Handoff | 移交 | Agent 把控制权转给别人 |
| Agents-as-Tools | 子 Agent 当工具 | 主管调专员，控制权不转移 |
| Swarm | 蜂群 | 去中心化、互相移交的模式 |
| Blackboard | 黑板 | 通过共享数据间接通信 |
| Tool / Function Calling | 工具调用 | Agent 调用外部函数 |
| Memory | 记忆 | 短期=对话历史；长期=知识库 |
| State | 状态 | 协作中共享的那份数据 |
| Trace | 调用链 | 一次请求触发的所有步骤记录 |
| Span | 跨度 | trace 里的一段（一次 Agent 调用） |
| Context Window | 上下文窗口 | 模型一次能看的最大文本量 |
| Error Cascade | 错误级联 | 上游错误被下游放大 |
| Hallucination | 幻觉 | 模型编造无依据的内容 |
| Human-in-the-loop | 人在环中 | 关键步骤要人审批 |
| Graceful Degradation | 优雅降级 | 主方案失败自动走兜底 |
| MCP | 模型上下文协议 | 管 Agent↔工具 |
| A2A | Agent 互操作协议 | 管 Agent↔Agent |

## 附录 B：常见误解速查

| 误解 | 事实 |
|------|------|
| "Agent 越多越厉害" | 数量不是指标，**任务成功率**才是 |
| "多 Agent 一定比单 Agent 准" | 不一定，设计不好会**互相强化幻觉** |
| "把函数拆小就叫多 Agent" | 顺序固定的话那还是 Workflow，**分水岭是"谁决定下一步"** |
| "上了框架才算多 Agent" | 框架只是脚手架，**50 行纯 Python 也能是标准多 Agent** |
| "多 Agent 能解决所有问题" | 它解决的是**上下文/权限/并行/专业分工**四个问题，别的不解决 |
| "用最贵的模型效果最好" | 路由/质检这类"选择题"用便宜模型就够，**分级用模型**才省钱 |

---

## 最后：三句话总结

1. **你现在做的是对的**——先用 Workflow 跑通，别急着上多 Agent。
2. **要改造就按 Step 0→1→2→4 走**，每一步都保持可运行、可回滚、可度量。Step 3（Supervisor）留到业务真变复杂时。
3. **多 Agent 的本质不是"多几个 AI"，而是"把一个大问题切成几个小问题，然后决定谁来切、谁来做、怎么收尾"**——**其中"怎么收尾"（编排、上限、兜底、审计）比"多几个 AI"重要一百倍。**

> 📖 配套：`docs/核心功能详解.md` 讲你系统现在怎么跑；`docs/ORM入门教程.md` 讲数据库；本文件讲多 Agent 怎么落地。
