# 景区智能服务与运营 Agent

面向景区的智能问答、路线推荐、实时天气和运营管理系统。

- 后端：FastAPI、PostgreSQL + pgvector、Redis、ARQ Worker
- 检索：PostgreSQL 结构化检索，可选 Milvus 混合检索
- 前端：Vue 3 + Vite
- 部署：Docker Compose，适合本地开发和阿里云 ECS

## 产品界面导览

下面按用户在导航栏中的使用顺序展示五个核心页面。每张截图对应一个可直接操作的功能模块，帮助读者快速理解项目从咨询、浏览到规划出行的完整体验。

| 顺序 | 页面 | 核心价值 |
| --- | --- | --- |
| 01 | 智能咨询 | 用景区知识、公告和实时信息回答游客问题 |
| 02 | 景点浏览 | 浏览景点卡片并查看开放状态、游览时长和详情 |
| 03 | 景区地图 | 查看三沟路线、观光车主线、步行栈道和换乘中心 |
| 04 | 最新公告 | 集中展示开放、预约、承载量等官方通知 |
| 05 | 路线推荐 | 根据游玩时长、同行人群和偏好生成个性化路线 |

### 01 · 智能咨询

![智能咨询页面](docs/screenshots/01-smart-consultation.png)

智能咨询是项目的主入口。游客可以直接询问景点、门票预约、观光车换乘和游览路线，系统会结合景区知识库与实时信息进行回答，并在右侧展示回答来源、天气和反馈入口。

### 02 · 景点浏览

![景点浏览页面](docs/screenshots/02-attractions.png)

景点浏览以卡片方式呈现景区内容，支持按分类查看景点图片、简介、开放状态、预计游览时长和门票信息。点击卡片后可打开详情弹窗，查看景点亮点、游览提示和信息来源，适合出行前快速筛选目的地。

### 03 · 景区地图

![景区地图页面](docs/screenshots/03-scenic-map.png)

景区地图提供高分辨率游览路线总览，标注观光车主线、步行栈道、三沟换乘中心和高海拔景点。页面支持地图缩放，并配有路线图例和现场提示，帮助游客理解景点之间的空间关系和游览顺序。

### 04 · 最新公告

![最新公告页面](docs/screenshots/04-notices.png)

最新公告用于集中发布景区官方信息，包括恢复开放、预约规则、游客承载量、门票价格和入园时间等内容。公告卡片标注生效日期和来源，游客可在出行前快速核对最新政策。

### 05 · 路线推荐

![路线推荐页面](docs/screenshots/05-route-recommendation.png)

路线推荐根据游玩时长、同行人群和偏好生成可执行的游览方案，并展示推荐景点的顺序、所在沟谷、预计用时和难度。系统还会结合天气、换乘中心和现场提示，帮助游客减少折返，安排更合理的行程。

## 一、运行前准备

推荐使用 Docker Compose。请安装 Docker Desktop（Windows、macOS）或 Docker Engine + Docker Compose（Linux）。如需导入数据或运行测试，还需要 Python 3.11 及以上版本和 Git。

检查安装：

~~~bash
docker --version
docker compose version
git --version
python --version
~~~

获取代码：

~~~bash
git clone https://github.com/yuanqi975/scenic-agent.git
cd scenic-agent
~~~

## 二、配置环境变量

复制模板：

~~~bash
cp .env.example .env
~~~

Windows PowerShell：

~~~powershell
Copy-Item .env.example .env
~~~

编辑 .env，至少修改以下配置：

~~~env
POSTGRES_PASSWORD=请改成数据库强密码
DATABASE_URL=postgresql+psycopg://postgres:请改成数据库强密码@postgres:5432/scenic_agent
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=请改成管理员强密码
JWT_SECRET=请填写至少32位随机字符串
~~~

首次体验可以使用不依赖真实模型的降级模式：

~~~env
LLM_MODE=fallback
RAG_BACKEND=postgres
~~~

使用真实模型和 Milvus 时，填写：

~~~env
LLM_MODE=agent
LLM_BASE_URL=https://你的模型服务/v1
LLM_API_KEY=你的模型密钥
LLM_MODEL=你的模型名称
EMBEDDING_BASE_URL=https://你的向量服务/v1
EMBEDDING_API_KEY=你的向量服务密钥
RAG_BACKEND=milvus
~~~

.env 含有密钥，只能保存在本地或服务器，不能提交到 GitHub。.env.example 只保留配置说明。

## 三、使用 Docker Compose 启动（推荐）

在项目根目录执行：

~~~bash
docker compose up -d --build
docker compose ps
~~~

查看后端日志：

~~~bash
docker compose logs -f backend
~~~

访问地址：

- 前端：http://localhost:5173
- 后端健康检查：http://localhost:8000/api/v1/health

停止服务但保留数据：

~~~bash
docker compose stop
~~~

停止并删除容器（仍保留数据卷）：

~~~bash
docker compose down
~~~

不要随意执行 docker compose down -v，否则会删除 PostgreSQL 和 Milvus 数据卷。

## 四、初始化景区数据

Docker 首次创建 PostgreSQL 容器时会执行 sql/init.sql 创建表结构。要让问答和推荐接口返回完整景区数据，还需要导入数据。

### 1. 创建 Python 虚拟环境

Windows PowerShell：

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
~~~

Linux 或 macOS：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

### 2. 执行迁移和导入

Docker 将 PostgreSQL 映射到本机 5432、Redis 映射到本机 6380。Windows PowerShell：

~~~powershell
$env:PYTHONPATH = "backend"
$env:DATABASE_URL = "postgresql+psycopg://postgres:你的数据库密码@localhost:5432/scenic_agent"
$env:REDIS_URL = "redis://localhost:6380/0"

alembic -c alembic.ini upgrade head
python scripts/generate_jiuzhaigou_data.py
python scripts/import_to_postgres.py --dsn $env:DATABASE_URL --chunk-size 800 --chunk-overlap 120
~~~

Linux 或 macOS：

~~~bash
export PYTHONPATH=backend
export DATABASE_URL='postgresql+psycopg://postgres:你的数据库密码@localhost:5432/scenic_agent'
export REDIS_URL='redis://localhost:6380/0'

alembic -c alembic.ini upgrade head
python scripts/generate_jiuzhaigou_data.py
python scripts/import_to_postgres.py --dsn "$DATABASE_URL" --chunk-size 800 --chunk-overlap 120
~~~

导入后重启：

~~~bash
docker compose restart backend worker
~~~

### 3. 建立 Milvus 索引（可选）

配置 embedding 服务后执行：

~~~bash
python scripts/index_milvus.py --uri http://127.0.0.1:19530 --collection scenic_knowledge
~~~

没有 embedding 服务时使用 RAG_BACKEND=postgres。

## 五、不使用 Docker 的本地开发

需要自行安装并启动 PostgreSQL（需 pgvector）和 Redis。

~~~powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "backend"
$env:DATABASE_URL = "postgresql+psycopg://postgres:数据库密码@localhost:5432/scenic_agent"
$env:REDIS_URL = "redis://localhost:6379/0"
alembic -c alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000
~~~

另开终端启动前端：

~~~powershell
cd frontend
npm install
npm run dev
~~~

访问 http://localhost:5173。

## 六、部署到阿里云 ECS

推荐 Ubuntu ECS，只开放安全组端口 22、80、443；不要把数据库、Redis、Milvus 和后端端口开放到公网。

~~~bash
sudo apt update
sudo apt install -y git docker.io docker-compose-plugin nginx certbot python3-certbot-nginx
sudo systemctl enable --now docker
sudo mkdir -p /opt
sudo git clone https://github.com/yuanqi975/jiuzhaigou-scenic-agent.git /opt/jiuzhaigou-agent
cd /opt/jiuzhaigou-agent
cp .env.example .env
chmod 600 .env
~~~

编辑服务器上的 .env：

~~~env
ENVIRONMENT=production
CORS_ORIGINS=https://你的域名
LLM_MODE=agent
JWT_SECRET=随机生成的至少32位字符串
ADMIN_PASSWORD=管理员强密码
~~~

生产环境不能使用默认 JWT 密钥、默认管理员密码或 LLM_MODE=fallback。配置完成后：

~~~bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:8000/api/v1/health
~~~

域名解析到 ECS 后，可用 Certbot 配置 HTTPS：

~~~bash
sudo certbot --nginx -d 你的域名
~~~

## 七、接口和测试

常用接口：

~~~text
GET /api/v1/health
GET /api/v1/metrics
~~~

运行测试：

~~~bash
python -m pytest -q
python -m pytest backend/tests -v
~~~

前端测试：

~~~bash
cd frontend
npm install
npm run test
~~~

## 八、常见问题

### 页面打不开

~~~bash
docker compose ps
docker compose logs --tail=100 backend frontend
~~~

### 后端显示数据库不可用

检查 PostgreSQL 状态，并确认 .env 中 POSTGRES_PASSWORD 与 DATABASE_URL 使用了同一个密码：

~~~bash
docker compose logs --tail=100 postgres
~~~

### SSH 无法连接 ECS

安全组的 SSH 访问来源填写当前电脑的公网 IP，并加 /32，例如：

~~~text
你的公网IP/32
~~~

不要填写 192.168.x.x 局域网地址，也不要长期使用 0.0.0.0/0。


