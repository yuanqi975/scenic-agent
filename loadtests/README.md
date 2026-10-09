# Locust 压测

`ScenicVisitor` 覆盖浏览、推荐、票务缓存问答、景点结构化问答和 SSE。
`ModelVisitor` 单独调用复杂问题，要求返回实际 Agent 链路；HTTP 200 的降级、超时和空回答也计为失败。
SSE 响应时间覆盖完整读取，并分别统计 HTTP 状态、终止事件与业务状态。

## 生产限流防护

```powershell
docker compose up -d --build --no-deps backend
locust -f loadtests/locustfile.py ScenicVisitor --headless --host http://localhost:8000 -u 50 -r 5 -t 2m --stop-timeout 65 --only-summary --csv reports/locust/protection --csv-full-history --html reports/locust/protection.html
```

默认每 IP 总额度 600 次/分钟；普通问答 60 次/分钟、推荐 30 次/分钟、SSE 60 次/分钟，接口分别计数。
已验证的 JWT 用户使用独立身份额度，同时受总 IP 保护。匿名游客按 IP 计数，来自同一 IP 的 50 个虚拟用户会触发防护。
不接受任意客户端身份或转发 IP 头作为绕过限流的依据。429 带有 `Retry-After`。

## 容量测试

仅本地复测使用额外配置，将各项额度临时提高至 6000 次/分钟：

```powershell
docker compose -f docker-compose.yml -f loadtests/compose.throughput.yml up -d --no-deps backend
locust -f loadtests/locustfile.py ScenicVisitor --headless --host http://localhost:8000 -u 50 -r 5 -t 2m --stop-timeout 65 --only-summary --csv reports/locust/capacity --csv-full-history --html reports/locust/capacity.html
locust -f loadtests/locustfile.py ModelVisitor --headless --host http://localhost:8000 -u 10 -r 2 -t 2m --stop-timeout 65 --only-summary --csv reports/locust/agent --csv-full-history --html reports/locust/agent.html
docker compose up -d --no-deps backend
```

测试完成恢复默认限流。不要在部署中启用 `compose.throughput.yml`。
额外输出的 `_final.json` 保存停止后的最终计数、成功请求耗时、业务状态、缓存命中数和 SSE 完整结束数；周期 CSV 可能少记停止前最后一秒的请求。

## 并发与缓存

- `WEB_CONCURRENCY=2`：两个 Uvicorn 进程。
- `LLM_MAX_CONCURRENCY=2`：每进程两个模型调用，全服务默认最多四个；增加进程前应核对供应商总额度。
- `LLM_TIMEOUT_SECONDS=30`：模型排队和调用共用时限。
- 供应商 401/402/403 触发默认 60 秒共享熔断，429/5xx 短时熔断；健康接口返回实际熔断原因。熔断只能减少无效请求，不能恢复供应商余额。
- 景点列表、确定性路线与公开直接答案缓存 15 秒；每次问答仍创建自己的会话、轨迹和持久化记录。
- `DB_POOL_SIZE=10`、`DB_MAX_OVERFLOW=20`、`DB_POOL_TIMEOUT_SECONDS=5`：每个后端进程的连接池。
- `SSE_MAX_CONCURRENCY=100`：每进程最多 100 个 SSE；连接超载返回 503，超时发送带业务错误的终止事件，断开时取消任务并释放容量。

## 验收口径

容量测试要求 HTTP 429 < 1%、HTTP 5xx = 0%、SSE 完整结束率 > 99%、景点查询 P95 < 100 ms，并核对业务失败与降级率。
模型生成的耗时单独报告，只统计成功生成请求时才有业务性能意义。少量重复问题的两分钟测试不代表极限吞吐或长时间稳定性。
