# 县域政策 DID 评估后端

面板数据版本化 + 双重差分（DID）估计 + 结果管理的后端服务。
FastAPI（Python 3.12）+ PostgreSQL 16，回归/聚类标准误/估计量全部手写，
只允许 NumPy，不依赖任何计量或统计建模库。

## 要解决的问题

补贴先在一部分县试点、随后交错铺开。经典双向固定效应（TWFE）回归

```
y_it = α_i + λ_t + τ·D_it + ε_it
```

在处理时点交错、效应又随时间变化时，会把**已经接受处理的县**在后续时期也当作
对照县使用；各干净的 2×2 比较被 OLS 以可能为负的权重压成同一个 τ，估计量
偏离真值，**甚至符号相反**。本服务用 Callaway & Sant'Anna (2021) 风格的
堆叠干净 2×2 估计量替代 TWFE，TWFE 仅作为对照结果同时返回并明确标注风险。

## 估计方法（`app/did.py`）

按处理队列 g（县首次处理的时期）与目标时期 t 构造干净的 2×2：

| 组成 | 规则 |
|---|---|
| 处理组 | 队列恰为 g、在 g−1 与 t 两期都有观测的县 |
| 对照组 | 在 g−1 与 t 两期都未处理的县（两种口径，见下） |
| 基期 | 处理前一期 g−1（CS 长差分，处理后各期共用同一基期）；队列若恰在首个观测期开始（经典 2×2 两期情形），退化为 t−1 |
| 估计 | 全部 2×2 堆叠成一个「案例截距 + 案例×处理组交互（+ ΔX 控制变量）」的 OLS，交互系数即 ATT(g,t)，**没有任何已处理县进入对照** |
| 标准误 | 按县聚类的 CR1 三明治估计量 |

汇总口径：

* **总体 ATT**：各 ATT(g,t) 按其处理组县数等权加总；
* **动态效应**：按相对时期 e = t − g 以处理组县数等权加总（事件研究，e=−1 为基准）；
* **平行趋势检验**：对 e ≤ −2 的处理前安慰剂 ATT(g,t) 同样按相对时期加总，
  做联合 Wald 检验（卡方分布 CDF 手写，见 `chi2_sf`）。

**选择这个口径的理由**：ATT(g,t) 是透明的干净组间-期前-期后比较，不做任何
跨队列的禁忌比较；按处理组县数加权的总体 ATT 就是"处理样本上的平均处理效应"，
没有负权重。**代价**：识别依赖条件平行趋势；某 (g,t) 在选定对照口径下没有
对照县时该案例被跳过（结果的 `skipped_cases_no_control` 会列出）；左删失队列
（首个观测期就已处理）只能用紧邻一期做基期。

### 两种对照口径与它们何时给出明显不同的结果

提交分析时选 `control_strategy`：

* `never_treated`（默认）：对照只取**从未处理**的县。需要数据里有这样的县；
  平行趋势假设只针对处理县与从未处理县。
* `not_yet_treated`：对照取在 g−1 与 t 两期**尚未处理**的县，包括后来才处理的
  队列。样本利用更充分，但额外要求**无预期效应**（anticipation）假设；末期才
  铺开的队列在该口径下往往找不到对照。

两者回答的是不同的反事实参数，下列情形会明显分叉：

1. **晚期处理县本身有不同的处理前趋势**。`not_yet_treated` 拿它们当 g=2 队列
   的反事实，差分就把这种趋势差当成了处理效应；`never_treated` 不受影响
   （测试 `test_control_strategies_can_differ` 即构造了这种数据）。
2. **存在队列构成的选择效应**：先试点的县通常是"更有潜力的县"，后来铺开的县
   系统性不同，用后者做前者的对照隐含"它们的反事实趋势相同"。
3. **处理效应有预期反应**：县在正式处理前就因预期政策改变行为，只违反
   `not_yet_treated` 所需的无预期假设。

如果有可信的从未处理组，优先用 `never_treated`；它的识别假设最容易讲清楚。

### TWFE 的偏误（实测）

`scripts/simulate_staggered.py` 构造 6 期、队列 g=1/3、效应 τ_e=e 线性增长的
模拟数据，真实总体 ATT = 1.625：

```
真实总体 ATT                 = 1.6250
堆叠 DID（never_treated）ATT = 1.5795  (SE=0.0873)
经典 TWFE 回归系数           = 0.4352  (SE=0.0888)
```

在没有从未处理县、效应增长更快的设定里，TWFE 可以反号而堆叠 DID 仍为正
（`test_staggered_sign_reversal_demo`）。

## 数据模型与版本

* 上传完整面板 → 创建数据集与 v1；
* 之后只提交**差异**：`upserts`（按 unit+period 覆盖/新增）与 `deletes`
  （按 unit+period 删除），服务合并父版本完整内容后物化出 v2/v3…；
* **每个版本都保存完整内容**（COPY 批量写入），同时保存差异原文；旧版本永不
  覆盖，按版本号可取回全部行；
* 结果绑定 `(数据版本, 分析设定)`：同版本同设定重复提交直接返回已有结果
  （幂等）；新版本上旧结果不受影响；全部落 PostgreSQL，服务重启后都在。

每行：`unit`（县编号）、`period`（年份整数 / `"YYYY"` / `"YYYYQn"` 季度）、
`outcome`（数值）、`treated`（布尔，开始后不得撤回）、`covariates`（数值列）。

校验失败统一返回 `422` 与 `{error: {code, message, details[]}}`，details 指出
字段与原因：同县同期两行、处理后撤回未处理、时期无法解析或年/季混用、
结果变量非数值/NaN/无穷、协变量非数值、删除不存在的记录等。

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/datasets` | 完整上传，建数据集+v1 |
| GET | `/api/datasets` | 列出数据集 |
| GET | `/api/datasets/{id}` | 数据集与全部版本号 |
| POST | `/api/datasets/{id}/versions` | 差异提交，生成新版本 |
| GET | `/api/datasets/{id}/versions/{v}?data=true` | 取回该版本完整内容 |
| POST | `/api/versions/{vid}/analyses` | 提交分析（幂等，重跑返回缓存） |
| GET | `/api/results/{rid}` | 取某次估计结果（含动态效应/平行趋势/TWFE） |
| GET | `/api/results?version_id=` | 列出结果 |
| GET | `/health` | 健康检查 |

上传示例：

```bash
curl -X POST localhost:8000/api/datasets -H 'content-type: application/json' -d '{
  "name": "补贴试点",
  "rows": [
    {"unit": "320123", "period": 2018, "outcome": 10.2, "treated": false, "covariates": {"pop": 45.1}},
    {"unit": "320123", "period": 2019, "outcome": 11.7, "treated": true,  "covariates": {"pop": 45.3}}
  ]
}'
```

差异修订：

```bash
curl -X POST localhost:8000/api/datasets/1/versions -H 'content-type: application/json' -d '{
  "upserts": [{"unit": "320123", "period": 2019, "outcome": 11.9, "treated": true,
               "covariates": {"pop": 45.3}}],
  "deletes": [{"unit": "320999", "period": 2018}],
  "change_note": "修订录错的2019年值并删除误录县"
}'
```

提交分析：

```bash
curl -X POST localhost:8000/api/versions/1/analyses -H 'content-type: application/json' -d '{
  "name": "main",
  "control_strategy": "never_treated",
  "covariates": ["pop"],
  "also_twfe": true
}'
```

## 运行

```bash
docker compose up --build        # 仅对外暴露 8000 端口（HTTP），Postgres 不暴露
# 演示模拟（宿主机需 numpy）：
python scripts/simulate_staggered.py
```

测试（纯估计量/校验测试无需数据库；端到端 API 测试在 compose 内跑）：

```bash
docker compose --profile test run --rm tests
# 或本地（需可访问的 Postgres）：
RUN_DB_TESTS=1 DATABASE_URL=postgresql://did:did@localhost:5432/did_panel pytest -q
```

## 目录

```
app/did.py      估计引擎：堆叠 DID、平行趋势 Wald、TWFE 对照、聚类标准误
app/panel.py    面板校验、时期解析、差异合并
app/db.py       PostgreSQL schema 与数据访问（版本/差异/结果不可变）
app/main.py     FastAPI 路由
tests/          估计量、校验、端到端 API 测试
scripts/        交错处理模拟演示
```
