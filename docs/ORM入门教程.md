
# ORM 数据库入门教程（零基础新手向）

> 专为你这个「景区 Agent」项目定制。你项目用的是 **Python + SQLAlchemy + PostgreSQL**。
> 这是一份渐进式教程：先讲故事，再讲原理，最后结合**你自己的代码**。

---

## 第 1 章：先搞懂"数据库"到底是什么

### 1.1 数据库就是"超强 Excel 表格"

你肯定用过 Excel。让我们把数据库想象成**很多张 Excel 表**：

| 表名 | 作用 | 类比 Excel |
|------|------|-----------|
| `users` | 存用户 | 「用户」工作表 |
| `conversations` | 存对话 | 「对话」工作表 |
| `attractions` | 存景点 | 「景点」工作表 |

每张表都有：
- **列（Column）**：就是字段，比如 `email`、`content`，像表格的表头
- **行（Row）**：一条记录，比如一个用户，像表格的每一行
- **主键（Primary Key）**：每一行的"唯一身份证号"，比如 `conversation_id`

### 1.2 要在数据库里干的事，总结起来就四件事

伟大的 CRUD 四兄弟，90% 的数据库操作都是它们：

| 简称 | 英文 | 中文 | 举例 |
|------|------|------|------|
| **C** | Create | 增 | 新建一条对话记录 |
| **R** | Read | 查 | 查询某个景点的信息 |
| **U** | Update | 改 | 更新对话的标题 |
| **D** | Delete | 删 | 删除一条反馈 |

**记住一个心法**：任何数据库操作，无非是这四个动作的某种组合。

---

## 第 2 章：为什么需要 ORM？

### 2.1 先看"没有 ORM"的世界（原生 SQL）

你的项目里大量使用原生 SQL（现在先不用看懂，感受一下）：

```python
# 这是你项目 backend/app/main.py 里的真实代码
row = conn.execute(
    text("SELECT task_id,task_type,status,payload,created_at,updated_at "
         "FROM async_tasks WHERE task_id=:id AND park_id=:park_id"),
    {"id": task_id, "park_id": PARK_ID}
).first()
```

这段代码的问题：
1. **要会 SQL 语法**（`SELECT`、`WHERE`、`JOIN`…）
2. **SQL 是字符串**，拼错了只在运行时报错，提前发现不了
3. **字段名全靠手打**，打字错了不提示
4. **数据库表结构变了，代码全要跟着改**

### 2.2 再看"有了 ORM"的世界

ORM 让你用"操作 Python 对象"的方式，代替"写 SQL 字符串"：

```python
# 加了 ORM 之后，同样的查询变成这样
task = db.query(AsyncTask).filter(AsyncTask.task_id == task_id).first()
```

### 2.3 一句话总结 ORM 是什么

> **ORM 是"对象"和"数据库表"之间的一座翻译桥**：
> 你把数据库看成 Python 对象，ORM 在背后自动帮你翻译成 SQL 语句执行。

```
你写（Python 对象）          ORM 自动翻译           数据库（表）
─────────────────     ────────────────     ─────────────────
Task(task_id=...)  ──►  INSERT INTO        ──►  async_tasks 表
                              async_tasks ...
```

---

## 第 3 章：核心概念逐个击破

### 3.1 表 ↔ 类

**在 ORM 里，"一张表" 对应 "一个 Python 类"**。

拿项目里的 `conversations`（对话表）举例。表结构大概是这样：

| conversation_id | park_id | title | created_at |
|-----------------|---------|-------|------------|
| abc123 | jiuzhaigou_scenic_area | 第一次咨询 | 2026-09-10 ... |

用 ORM 定义成类就是这样：

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime

class Base(DeclarativeBase):
    """所有模型的基类"""

class Conversation(Base):
    __tablename__ = "conversations"          # <<< 对应数据库表名

    conversation_id: Mapped[str] = mapped_column(primary_key=True)
    park_id: Mapped[str]
    title: Mapped[str | None]                # | None 表示该列可空
    created_at: Mapped[datetime]             # 对应 timestamptz 类型
```

**关键对应关系：**
| 数据库概念 | ORM 概念 |
|-----------|---------|
| 表（table） | 类（`class Conversation`） |
| 行（row） | 对象实例（`c1 = Conversation(...)`） |
| 列（column） | 类的字段（`title`） |
| 主键 | `primary_key=True` |

### 3.2 增（Create）

**数据库操作：**
```sql
INSERT INTO conversations (conversation_id, park_id, title)
VALUES ('abc123', 'jiuzhaigou_scenic_area', '第一次咨询');
```

**ORM 操作：**
```python
new_c = Conversation(conversation_id="abc123",
                     park_id="jiuzhaigou_scenic_area",
                     title="第一次咨询")
db.add(new_c)     # 加入"待办"区
db.commit()       # 真正写进数据库，相当于点了 Excel 的"保存"
```

### 3.3 查（Read）

**数据库操作：**
```sql
SELECT * FROM conversations WHERE park_id = 'jiuzhaigou_scenic_area';
```

**ORM 操作：**
```python
results = db.query(Conversation).filter(
    Conversation.park_id == "jiuzhaigou_scenic_area"
).all()
```

### 3.4 改（Update）

**数据库操作：**
```sql
UPDATE conversations SET title = '新标题' WHERE conversation_id = 'abc123';
```

**ORM 操作：**
```python
c = db.query(Conversation).filter(
    Conversation.conversation_id == "abc123"
).first()
c.title = "新标题"     # 直接改对象属性
db.commit()           # 保存
```

### 3.5 删（Delete）

**数据库操作：**
```sql
DELETE FROM conversations WHERE conversation_id = 'abc123';
```

**ORM 操作：**
```python
c = db.query(Conversation).filter(
    Conversation.conversation_id == "abc123"
).first()
db.delete(c)
db.commit()
```

---

## 第 4 章：重头戏——表与表之间的关系

这是 ORM 最强大的地方。你项目里这些表有天然的关联：

```
conversations（对话）
   └── conversation_messages（对话里的消息）
       conversation_id 指向 conversations.conversation_id

parks（景区）
   └── attractions（景点）    → 属于某个景区
   └── facilities（设施）     → 属于某个景区
   └── routes（路线）         → 属于某个景区
```

这种"一条对话有多条消息"就叫 **一对多（one-to-many）** 关系。

### 4.1 在 ORM 里怎么表示？

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str | None]

    # 关键：定义"关系"——一条对话关联多条消息
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation"
    )

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id: Mapped[int] = mapped_column(primary_key=True)

    # 外键：这个字段存的是所属对话的 id
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id")
    )
    content: Mapped[str]

    conversation: Mapped[Conversation] = relationship(
        back_populates="messages"
    )
```

**用起来就是"点语法"**：

```python
# 拿到一个对话
conv = db.query(Conversation).filter(
    Conversation.conversation_id == "abc123"
).first()

# 直接拿到它所有的消息！ORM 自动帮你查了另一张表
for msg in conv.messages:
    print(msg.content)
```

你看，**完全不需要写 JOIN 连接语句**，这就是关系型数据库 + ORM 的威力。

---

## 第 5 章：你项目里真实的 ORM 情况（重要！）

我仔细看了你的项目，要告诉你一个**关键事实**：

> 你的 `backend/app/main.py` 里，**并没有使用 ORM 的模型类**。
> 它用的是 SQLAlchemy 的**原生 SQL 写法**（`text()`），也就是第 2.1 节那种方式。

看这段真实代码（`main.py` 里面）：

```python
engine = create_engine(DATABASE_URL, ...)   # 建立数据库连接（SQLAlchemy）
...
with engine.connect() as conn:              # 打开连接
    rows = conn.execute(
        text("SELECT ... FROM async_tasks WHERE task_id=:id"),
        {...}
    ).all()
```

你目前用的 `engine.connect()` + `text()`，属于 SQLAlchemy 的 **SQL 表达式层 / 原生 SQL 层**，`requirements.txt` 里虽有 `SQLAlchemy>=2.0` 但项目并未定义模型类。

**这不是错误**——很多项目为了精细控制 SQL 会这样用。但对新手来说，直接基于模型类的 ORM 写法通常更易读、更不易出错。

### 两种写法的对比

| 维度 | 原生 SQL（你现在） | ORM 模型（可升级方向） |
|------|------------------|----------------------|
| 可读性 | 较低，全是字符串 | 较高，对象化 |
| 类型检查 | 无，靠手拼字段 | 有，IDE 有提示 |
| 易错性 | 拼错字段只运行时报错 | 拼错直接报错提示 |
| 学习难度 | 需要学 SQL | 学对象即可 |
| 灵活控制 SQL | 非常灵活 | 复杂查询稍受限 |

---

## 第 6 章：数据库迁移（Alembic）——为什么也需要了解

ORM 管的是"日常增删改查"，但数据库**表结构本身**怎么创建和修改？答案是**迁移工具**。

你项目里已经有 Alembic（`alembic.ini`、`backend/alembic/`）。它的作用：

- 记录"数据库结构是什么样"的**版本历史**
- 当结构要变（比如加一列），生成一个迁移脚本统一执行
- 把开发、测试、生产环境的表结构保持同步

你的迁移脚本在 `backend/alembic/versions/0001_runtime_tables.py`，里面用 `op.execute(...)` 创建了 `admin_users`、`conversations`、`agent_runs` 等表。

**日常接触的表情：**
```bash
alembic upgrade head      # 把数据库升级到最新表结构
alembic revision -m "add new table"   # 生成新的迁移脚本
```

---

## 第 7 章：给新手的学习路径建议

### 学习顺序
1. ✅ 先掌握 CRUD 四兄弟（第 3 章）——这是地基
2. ✅ 再掌握一对多关系（第 4 章）——这是 ORM 的灵魂
3. ✅ 尝试把项目里某个 `text()` 查询改写为 ORM 模型写法
4. ✅ 最后了解 `relationship` 的高级用法和查询优化（懒加载 / 急加载）

### 推荐动手练习（用你自己的项目）
由于你的项目已装好 SQLAlchemy，可以直接开个 Python 终端感受：

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent")
# 这只是连接到你的数据库，下一步用 ORM 模型去查询
```

### 不要太焦虑
- **不需要先精通 SQL 再学 ORM**，两个可以互相促进
- **不需要背语法**，用多了自然记住，IDE 也会自动提示
- **从"抄 + 改"开始**：拿项目里现成查询，改成 ORM 形式，反复练

---

## 总结：一张图记住全部分子

```
你的 Python 代码                      SQLAlchemy ORM                  PostgreSQL 数据库
────────────────────      ─────────────────────────────     ─────────────────────────
class Conversation   ──►  (自动映射表结构)              ──►   conversations 表
db.add(obj)          ──►  INSERT INTO                    ──►  新增一行
db.query(...).all()  ──►  SELECT ...                     ──►  返回多行
obj.title = "x"      ──►  UPDATE ...                     ──►  更新该行
db.delete(obj)       ──►  DELETE ...                     ──►  删除该行
conv.messages        ──►  SELECT 关联表 + JOIN           ──►  返回关联数据
```

> 需要 SQLAlchemy 一会儿，但天气渐明：**ORM 是让你用"操作对象"替代"写 SQL"，让你更专注于业务逻辑本身。**

如果你愿意，我可以接着帮你做这三件事之一：
1. 把项目里某个 `text()` 查询**改写成 ORM 模型写法**给你示范
2. 写一个**可以在你电脑上运行的实操练习**（基于你的数据库）
3. 带你逐行精读项目里某一段数据库代码

你选哪个？
