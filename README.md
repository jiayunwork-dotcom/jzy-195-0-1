# 县域政策面板评估服务

管理“单位（县）× 时期（年/季度）”面板数据的**不可变版本**，并通过 HTTP 接口提交
双重差分（DID）分析。估计目标为**处理组平均处理效应（ATT）**，提供：

- 总 ATT（处理后各单元的组别-规模加权汇总）；
- 按“距处理开始的相对时期 e”展开的**动态效应**（e=0 为处理开始当期）；
- 处理前各期动态效应联合为零的 **Wald 平行趋势检验**；
- 同一数据上的**双向固定效应（TWFE）诊断**，用于对照展示其在交错处理+动态效应
  下的偏误。

标准误按**单位聚类**（CR1）。所有回归、聚类标准误与估计量都在 `app/did.py`
中自行实现，只依赖 NumPy 做矩阵运算，不调用任何计量/统计建模库。

---

## 1. 快速开始

```bash
docker compose up --build
# 服务在 http://localhost:8000 ，文档 http://localhost:8000/docs
# 对外只发布 HTTP 8000；PostgreSQL 仅在 compose 内网可见，不发布 5432。
```

健康检查：`GET /health`。

### 最小流程

```bash
# 1) 建数据集
DS=$(curl -s -X POST localhost:8000/datasets -H 'Content-Type: application/json' \
  -d '{"name":"农机购置补贴试点"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2) 上传 v1（整表）
curl -s -X POST localhost:8000/datasets/$DS/versions -H 'Content-Type: application/json' \
  -d '{"rows":[
    {"unit":"c1","period":"2020","outcome":10,"treated":0},
    {"unit":"c1","period":"2021","outcome":15,"treated":1},
    {"unit":"c2","period":"2020","outcome":8, "treated":0},
    {"unit":"c2","period":"2021","outcome":10,"treated":0}]}'

# 3) 提交估计（默认只用从未处理单位作对照）
curl -s -X POST localhost:8000/datasets/$DS/estimates -H 'Content-Type: application/json' \
  -d '{"version":"1","spec":{"control_group":"never"}}'
# -> att.estimate = 3.0
```

### 差异修订

新版本通过差异（upsert/delete）从最新版本物化生成；旧版本永不改变：

```bash
curl -X POST localhost:8000/datasets/$DS/versions/diff -H 'Content-Type: application/json' -d '{
  "changes":[
    {"op":"upsert","row":{"unit":"c1","period":"2021","outcome":15.2,"treated":1}},
    {"op":"upsert","row":{"unit":"c3","period":"2020","outcome":7,"treated":0,"covariates":{"x":1.1}}},
    {"op":"delete","unit":"c9","period":"2019"}
  ]}'
```

差异在落库前后都会做**整表校验**（差异应用到父版本全量内容上），不合规直接 422，
不会留下半截版本。

---

## 2. 数据模型与校验

每行：`unit`、`period`、`outcome`、`treated`(0/1 或布尔)、可选 `covariates`
（字符串名→数值的对象，各行协变量键集合必须一致）。

时期支持：

- 年度：`"2015"`；
- 季度：`"2015Q1"` / `"2015q1"` / `"2015-Q1"`（内部统一成 `2015Q1`）。

年度与季度不能混用；所有时期必须能解析为可排序序列。

下列情况返回 **422** 并指明字段/原因：

| code | 含义 |
| --- | --- |
| `duplicate_unit_period` | 同一单位同一时期出现两行 |
| `treatment_reversed` | 处理开始后又回到未处理（treated 必须是 0…0,1…1） |
| `period_unparseable` / `period_frequency_mixed` | 时期无法排序 / 年季混用 |
| `outcome_not_numeric`（或 `covariate:<name>_not_numeric`） | 非数值、NaN、无穷 |
| `treated_not_binary` | treated 不是 0/1 |
| `missing_field` / `empty_dataset` / `unit_invalid` / `period_invalid` | 行结构问题 |
| `no_treated_units` | 上传合法，但没有任何单位接受处理，估计阶段拒绝 |
| `no_never_treated_controls` | 数据里没有从未处理单位；可改用 `not_yet` |
| `treatment_at_first_period` | 单位首个观测期就已处理，没有处理前基线 |
| `covariate_keys_inconsistent` | 不同行协变量键集合不一致 |

非平衡面板是允许的：缺失观测在构造每个 2x2 单元时按可用性处理；估计明细的
`group_time_att` 中给出每单元实际使用的处理组/对照组单位数，被跳过的单元在
`skipped` 中说明原因。

---

## 3. 方法论：估计口径、理由与代价

### 3.1 为什么不用一个 TWFE 回归了事

处理在县之间**交错开始、一旦开始不再撤回**，且真实效应通常随“处理后时间”变化。
此时经典回归

```
y_it = α_i + γ_t + τ · D_it + ε_it
```

只设一个恒定的 τ。在交错设计下，已经处理的单位在某些“处理组 vs 处理组”的 2x2
比较里会被当成对照；当效应随时间变化，这些“禁止的比较”的权重可能为负，使 τ 偏离
各群组真实效应的加权平均，**甚至符号相反**（Goodman-Bacon, 2021；de Chaisemartin &
D'Haultfœuille, 2020；Callaway & Sant'Anna, 2021）。本服务因此把 TWFE 仅作为
`twfe` 诊断字段返回，主结果使用下面的估计量。

### 3.2 主估计量：组别-时期 ATT（Callaway–Sant'Anna 式 2x2 DID，自写）

定义 `g` 为某组单位的处理开始时期、`t` 为结果期。对每个处理组 g 与每个时期 t，
取该组单位从“最近的、全组共同可观测的处理前时期 b(g)”到 t 的结果变化，减去
对照组同期变化：

```
ATT(g,t) = ( E[Y_t - Y_b | G=g] ) − ( E[Y_t - Y_b | 对照组] )
```

实现上对每个 (g,t) 堆叠一阶差分样本，拟合

```
ΔY_i = α_{g,t} + β_{g,t} · 1{G_i=g} (+ Γ_{g,t}·ΔX_i) + e_i
```

`β_{g,t}` 即 ATT(g,t)。所有 (g,t) 单元横向拼成块状设计矩阵做联合 OLS，
残差在**单位层面**求和得到聚类协方差：

```
V = c · (X′X)⁻¹ · [ Σ_i X_i′e_i e_i′X_i ] · (X′X)⁻¹
c = G/(G−1) · (N−1)/(N−K)            # CR1（Stata 默认）；N≤K 时只用 G/(G−1)
```

已处理单位**永远不会**进入任何单元的对照组，因此从构造上消除了 TWFE 的禁止比较。

处理前的单元（t<g）同样计算，作为安慰剂效应（动态展开的 e<0 部分），并对所有
e<0 的系数做联合 Wald 检验（H0：全为 0），统计量服从卡方，自由度等于处理前
相对时期个数；p 值用正则化不完全伽马函数自算，不依赖统计库。

### 3.3 对照组的两种口径

`spec.control_group`：

- **`"never"`（默认）**：对照只用**从未处理**单位。
  - 理由：在“平行趋势 + 无预期效应”的常规 DID 假设下，它是最干净的参照；
    后处理县无论是否有提前反应，都不会污染估计。
  - 代价：若数据里最终所有单位都被处理，则无法使用（接口返回
    `no_never_treated_controls`）；从未处理县较少时功效较低，且要求处理组与
    从未处理组之间趋势可比。
- **`"not_yet"`**：对照还纳入“在 t 与参照期 b 都**尚未处理**”的单位
  （严格要求其处理开始期晚于 t）。
  - 好处：对照更多、功效更高；即使没有从未处理单位也能估。
  - 代价：需要更强的假设——不仅“已处理 vs 未处理”之间要满足平行趋势，
    **不同时间开始处理的组之间**也要满足平行趋势（无预期效应、无提前反应）。
    后处理的县如果在自己处理前就因预期/抢跑/消息而改变了结果，它们当对照时会把
    这种事前跳变误记为“共同趋势”，从而污染较早处理组的估计。

**什么时候两种口径给出明显不同的结果？** 当“后来处理的县”在其处理开始之前存在
异质性的结果变化（预期效应、数据问题、与处理时点相关的事前冲击）时：用
`never` 不受这些县影响；用 `not_yet` 会把它们当对照而被带偏。测试
`test_never_vs_not_yet_diverge_when_later_cohort_reacts_early` 构造了一个极端例子：
晚处理组在自己处理前一期有 +4 的跳变，早处理组在该期的 ATT(g,t) 在 `never`
口径下是真值 5，在 `not_yet` 口径下被拉到 3。实务建议：同时报告两种口径，
差异大说明“晚处理组平行趋势”假设可疑，应结合事前趋势图/事件研究判断。

### 3.4 各单元效应怎么加权汇总

- **总 ATT**：每个“处理组 g 的处理后单元 ATT(g,t)”按其处理组规模 `n_g`
  （处理开始期该组观测到的县数）赋权，对全部 post 单元归一化。等价于
  “对所有处理县的所有处理后期求平均”，早处理县不会因为处理后期数更多而被
  重复计权（这是直接平均各期 ATT 的一个隐蔽陷阱）。
- **动态效应（事件研究）**：在每个相对时期 e=t−g 内，对贡献该 e 的各组按
  `n_g` 加权；再报告 e=0,1,2,…（e=0 即处理开始当期）以及 e<0 的安慰剂系数。
  注意 e 越大，能贡献的队列越少（只有更早处理的组），尾部估计的代表性随之下降。
- **各队列 ATT**（`cohort_att`）：组内对其处理后各单元等权平均，便于检查
  “早处理县 vs 晚处理县”的效应差异。

### 3.5 协变量调整（可选，默认关闭）

`spec.adjust_covariates=true` 时，在每个 2x2 单元内对一阶差分后的协变量
ΔX 加入线性控制（各单元独立斜率）。这是**线性回归调整**，要求条件平行趋势以
线性函数形式成立；它不是双重稳健/IPW 估计量，协变量缺失按行剔除、差分后零变异
的协变量列会在该单元内自动删除以免共线。默认关闭是为了让主结果只依赖最弱、最
透明的假设；是否调整应在分析方案中预先声明。

### 3.6 适用边界

- 需要每个处理组至少有一个可观测的处理前时期（否则该组无法估计，会在 `skipped`
  中标记；若没有任何可估单元返回 422）。
- 平行趋势检验只能检查“已观察到的处理前时期”，不能外推到处理开始前一期之后。
- 事件研究远端（很大或很小的 e）由少数队列识别，应连同 `n_treated/n_control`
  一起解读。
- 聚类数很少时 CR1 小样本修正不稳定；接口要求至少 2 个聚类。

---

## 4. 版本与结果的语义

- 每个数据集的版本号从 1 起单调递增；新版本可整表上传（`created_via=full`）
  或从最新版本做差异（`created_via=diff`），父版本指针与差异清单
  （`version_changes`）都保留。
- 版本内容**不可变**；每个版本在 `panel_rows` 中是一份独立的物化全量拷贝
  （写放大换取读取与估计的简单可靠）。
- 估计结果绑定 `(dataset_id, version_id, spec_hash)`，其中
  `spec_hash = sha256(canonical_json({engine_version, spec}))`。
  同一版本同一设定重复提交：直接返回已有结果（HTTP 200，`cached=true`）。
- 新版本发布不影响挂在旧版本上的结果；结果表保存当时的 `engine_version`、
  设定与完整结果 JSON。
- 所有状态都在 PostgreSQL（compose 卷 `pgdata`），服务重启后版本与结果都在。

## 5. HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/datasets` | 建数据集 `{name}` |
| GET | `/datasets/{id}/versions` | 列出版本 |
| POST | `/datasets/{id}/versions` | 整表上传生成新版本 |
| POST | `/datasets/{id}/versions/diff` | 差异生成新版本 |
| GET | `/datasets/{id}/versions/{latest\|版本号}` | 取回该版本完整内容 |
| POST | `/datasets/{id}/estimates` | 提交（或取回幂等的）估计 |
| GET | `/datasets/{id}/estimates` | 列出估计 |
| GET | `/datasets/{id}/estimates/{eid}` | 取单个估计完整结果 |
| GET | `/health` | 健康检查 |

估计请求：`{"version":"latest"|"1"|"2"|..., "spec":{...}}`；422 错误体形如
`{"error":{"code":"...","message":"...","field":"..."}}`。

## 6. 测试

```bash
# 引擎测试（不需要数据库）
pytest tests/test_did_engine.py

# 全量（含 API/持久化），需要 PostgreSQL 16：
#   docker compose run --rm \
#     -e TEST_DATABASE_URL=postgresql://panel:panel@db:5432/panelapp_test \
#     api sh -c "pip install -r requirements-dev.txt && pytest"
```

`tests/test_did_engine.py` 覆盖题目要求的全部成立条件：两组两期效应恰为 3；
交错+动态效应模拟中估计量还原设定真值而 TWFE 显著向下偏；结果整体加常数效应/SE
不变、乘正数 k 两者同比例缩放；打乱单位编号与行序不变；以及全部 422 报错情形。
`tests/test_api.py` 覆盖整表/差异版本生命周期、旧版本结果不受新版本影响、
同设定幂等、差异版本与等量全量上传估计完全一致、错误经由 HTTP 正确暴露。

> 注：仓库 `scripts/` 下的 `numpy.py`/`pytest.py` 只是在无网络的受限构建环境中
> 用标准库验证引擎的一次性垫片，**不属于服务镜像**（已写入 `.dockerignore`）。
