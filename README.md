# 九寨沟景区智能服务与运营 Agent

面向九寨沟景区的智能问答、路线推荐、实时天气和运营管理系统。

- 后端：FastAPI、PostgreSQL + pgvector、Redis、ARQ Worker
- 检索：PostgreSQL 结构化检索，可选 Milvus 混合检索
- 前端：Vue 3 + Vite
- 部署：Docker Compose，适合本地开发和阿里云 ECS

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
git clone https://github.com/yuanqi975/jiuzhaigou-scenic-agent.git
cd jiuzhaigou-scenic-agent
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

## 九、提交代码到 GitHub

.env 已被 .gitignore 忽略。修改代码后，在项目根目录执行：

~~~bash
git status
git add .
git commit -m "说明本次修改内容"
git push
~~~

第一次关联远程仓库时：

~~~bash
git remote add origin https://github.com/yuanqi975/jiuzhaigou-scenic-agent.git
git branch -M main
git push -u origin main
~~~

当前电脑使用 SSH 443 端口连接 GitHub；网络无法访问 HTTPS 时使用：

~~~bash
git remote set-url origin ssh://git@ssh.github.com:443/yuanqi975/jiuzhaigou-scenic-agent.git
git push -u origin main
~~~

推送前检查：

~~~bash
git status
git ls-files .env
~~~

第二条命令没有输出，才表示 .env 没有被 Git 跟踪。若密钥曾经提交到仓库，必须立即在服务商后台撤销并重新生成。

