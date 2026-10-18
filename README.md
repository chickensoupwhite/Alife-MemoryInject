# Mao.MemoryInject 4.8.2 · 开发副本说明（B 方案：只读检索层）

> 4.8.2（2026-10-18）· 在 4.8.0 基础上修 17 处逻辑问题（含对 4.8.1 三处的复查重做）

## 4.8.2 变更摘要

**对 4.8.1 三处的复查（结论：仍没真正解决，已重改）**

| 4.8.1 的修法 | 复查结论 | 4.8.2 改成 |
| --- | --- | --- |
| ③ 排序 | 排序发生在**选材之后**（`fresh.Sort` 在循环外），只改 prompt 拼接顺序；枚举顺序由目录/文件名决定，老档照样先吃光预算 | 先收集全部候选 → 按 mtime 新→旧排序 → 再按预算选材 |
| 超预算 `continue` | 单文件 > `LlmOrganizeMaxChars` 时**永远**排不上（每轮预算同值） | 恢复**截断** `Substring(0, budget)` + warning |
| ① 水位 key | 仍是 `relPath|mtime.Ticks`；`TrySaveState` 按**字典序**砍到 500 条 → 靠前的路径水位丢失 → 每轮重喂 | 水位改 `relPath + 内容哈希(SHA1[..12])`；不再截断；旧水位自动迁移 |

**新增两条 P 级**

- **P0**：侧车 `compact_scan` text 超预算就 `break`，只压前 N 条却把**整份来源**标记已压缩 → 尾巴既没进事实卡、又从索引消失。改为**整份放不进预算就不产出候选，绝不半压**。
- **P1**：`lastCompact` 触发即更新 → 压缩一直失败时 organize 仍跳过待压缩文件 → 文件被饿死。加 `lastCompactOk`。

**其他 12 处**

- 越界：审计报告写进官方 `Memory/` → 改写到 `MemoryFacts/`
- 丢数据：`db_build` 只清 `files` 表，迁移条目绕过 exclude → 按来源差集清理
- 丢数据：记忆目录不存在时会清空库 → 直接返回
- 回归：DB 召回丢图谱兜底（零命中不再顺关系链）→ 补回
- 配置失效：`MaxResults` 对侧车召回不生效（`top_k` 写死 5）→ 加 `topK`
- 引导错误：关 `ToolFirst` 时提示仍让模型"去目录翻" → 改调 `<memfetch>`
- 资源泄漏：切库不关旧连接 → `_db_for` 先 `close()`
- 崩溃：`db_build` 不更新 `_IDX.path` → 内部按旧路径开库（`closed database`）→ 建完 `_set_index`
- 缓存：build/reset 后向量、索引、图不失效 → 全清
- 并发：`chunks` 边清边读 → 先 `ToList()` 快照
- UI：5 秒定时器与慢查询重叠堆积 → 加 `refreshing` 标志

**验证**：七套测试 × 两轮全绿；语义编译 0 error（737 引用）。

---



> 来源：线上 `Storage\Plugins\Mao.MemoryInject`（4.6.5）的复制。
> 本轮按主人选定的 **B 方案**重构：**官方管"记住"，我们管"查得到、说得清"**。
> 日期：2026-10-18 · dsh

---

## 一、架构边界（B 方案，最重要）

| 角色 | 官方 `Alife.Function.Memory` | 本插件 `Mao.MemoryInject` |
| --- | --- | --- |
| 存储 | ✅ owner：`Character\<名>\Memory\`（含 `L1..L8` 存档、`History.json`、`duckdb`） | ❌ **只读**，一个字节都不写 |
| 历史压缩 | ✅ owner：`ChatHistoryAdd` → `MemoryManager` → `EditChatHistoryAsync`（L1..L8） | ❌ 不做（见"④ 为什么没做"） |
| 提示注入 | ❌ 已清空 `RecallHintKeywords`（原文备份 `.bak-20261018-hints`） | ✅ 唯一 owner：书签/工具提示 |
| 我们自己的写入 | — | ✅ 全部落在 `Character\<名>\MemoryFacts\`：`index.json`、`事实_*.md`、`审计报告_*.md`、`compacted_sources.txt`、`记忆索引视图.html` |
| 召回 | 它自己的 `MemoryService` 工具 | `memrecall / memwhen / memfetch / memprofile / memloops / memstatus / memlinks` |

**结论**：两套插件可以共存——只要守住"官方目录只读、历史不改、提示只留一套"这三条。

### ① 自动压缩改成"标记排除"（关键修正）

旧实现会 `File.Move(官方存档 → archive/)`——那会**踩坏官方插件**（它按 `L{n}` 路径读存档）。现在：

1. 模型把流水提炼成事实卡 → 写到 **`MemoryFacts/事实_YYYYMMDD.md`**；
2. 把来源相对路径记进 **`MemoryFacts/compacted_sources.txt`**；
3. 重建索引时把这个来源**排除**（不建索引）；
4. **官方文件原样保留**，官方插件照常能读。

效果一样（索引瘦身、冷读变快），但不动别人的数据。

### ④ 终局自动版：**不做**

技术上可行（`ChatBot.ChatHistoryAdd` + `ChatBot.EditChatHistoryAsync` 都在，官方就用它们），但：

- 官方**已经在做**同件事：`Threshold=100 / BatchSize=70 / Probability=0.4 / MaxCompressionLevel=8`，`ChatHistoryAdd(Assistant)` 后自动压缩；
- 两边同时改历史 → 时序不可控、**缓存双份击穿**；
- 收益低（官方压缩质量不差，`CompressPrompt` 是带人设的提炼）。

---

## 二、功能清单

### 工具（11 个，全部 `Task<string>` 返回；可选回显到聊天框）

| 工具 | 作用 |
| --- | --- |
| `memrecall` | 按人/事/物查书签 |
| `memwhen` | 中文时间词（上周三/三天前）→ 那段时间的记忆（⑨ 时间下钻）|
| `memfetch` | 按出处/关键词取原文 |
| `memprofile` | 按人聚画像卡 |
| `memloops` | 扫牵挂与欠账 |
| `memstatus` | 索引条数、覆盖日期、文件大小 |
| `memlinks` | 图谱邻居：某词/人/物常和什么一起出现（⑤）|
| `memcompact` | 手动压一次流水（force，不受自动开关限制）|
| `memaudit` | 手动审一次事实卡（force）|
| `memsettle` | 把长文本折成卡片（工具版）|
| `memview` | 生成 HTML 索引视图 |

> 工具返回 `Task<string>`：当轮就能给模型；`EchoToolResult`（默认开）再 Poke 一份到聊天框（400 字截断）——**注意这会双份进上下文**，想省 token 可关。

### ①~⑥

| # | 状态 | 实现 |
| --- | --- | --- |
| ① 自动压缩 | ✅（默认关） | `compactscan` 挑"条目≥30 且≥7 天"的流水 → 模型提炼 → ③ 合并写 `MemoryFacts/事实_*.md` → 记 `compacted_sources.txt` → 重建索引（排除这些来源）。官方文件不动 |
| ② 审计 | ✅ | `auditscan` 读 `MemoryFacts/事实_*.md`，找重复/可能重复/可能矛盾 → `审计报告_YYYYMMDD.md`（只报告，`max_facts=400` 防 O(n²)）|
| ③ 写入合并 | ✅ | `factmerge`：≥0.85 判重并刷新日期（新句更详细则合并保留）、≥阈值合并、其余追加；模型整理与压缩都走这条；`SemaphoreSlim` 串行化 |
| ⑤ 完整图谱 | ✅ | 规则抽实体（《》「」、专名后缀、拉丁词）入图；无向边双向可查；`memlinks` 暴露；索引重建/换角色时图谱缓存失效 |
| ⑥ 可视化内嵌 | ✅ | 配置页状态行 5 秒自动刷新 + 搜索浏览 + 五个按钮（浏览/重建/HTML/压缩/审计）|

---

## 三、配置项

**原有**：`Enable`、`UseSemantic`、`PythonExe`、`MemorySubPath`、`MaxResults`、`MinScore`、`MinMargin`、`MinScoreKeyword`、`SkipRecentDays`、`MinQueryChars`、`SyncWaitMs`、`TimeoutStreakLimit`、`CooldownSeconds`、`WaterHighChars`、`WaterLowChars`、`TriggerCooldownSeconds`、`HalfLifeDays`、`MaintainIntervalHours`、`UseGraph`、`BlockMaxChars`、`AutoOrganizeMinutes`、`LlmOrganizeEnable`、`LlmOrganizeMinutes`、`LlmOrganizeMaxChars`、`HintTemplate`。

**新增**：

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `EchoToolResult` | true | 工具结果回显到聊天框（400 字截断）|
| `MergeThreshold` | 0.5 | 事实合并阈值 |
| `AutoCompactEnable` | **false** | 自动压缩开关 |
| `AutoCompactMinutes` | 360 | 检查间隔 |
| `AutoCompactMinEntries` | 30 | 至少多少条才压 |
| `AutoCompactMinAgeDays` | 7 | 至少多旧才压 |
| `AutoCompactMaxChars` | 20000 | 单次喂模型上限 |
| `AuditEnable` | true | 审计开关 |
| `AuditMinutes` | 120 | 审计间隔 |
| `AuditLimit` | 30 | 单份报告最多几条 |
| `ToolFirst` | true | 提示里强调先调工具 |
| `WhenLimit` | 8 | `memwhen` 默认条数 |
| `ExcludeSources` | 空 | 手动排除的来源：每行一个相对路径或目录前缀（如 `L1/`）；只影响我们的索引，官方文件不动 |
| `MigrateOldIndex` | true | 首次启动是否迁移官方目录里的旧 `index.json`；关掉=直接按现有文件重建新库 |

> `AutoCompactKeepRaw` 已删除：B 方案不再移动/删除文件，这个开关没有意义了。

---

## 四、部署

1. 备份线上插件目录；
2. 覆盖：`MemoryInjectModule.cs`、`MemoryInjectConfig.cs`、`EmbedSidecar.cs`、`MemoryInjectUI.cs`、`sidecar_embed.py`、`manifest.json`；
3. **完整重启 Alife**（记忆插件禁 reload）；
4. 首次启动会：
   - 自动创建 `Character\<名>\MemoryFacts\`；
   - 把旧的 `Memory/index.json`、`llm_organize_state.txt` **复制**一份过来（省一次全量重建；旧文件保留不删）；
5. 官方 `RecallHintKeywords` 已在配置里清空（4 个文件，备份后缀 `.bak-20261018-hints`）；要恢复就还原备份。

## 五、验证清单

1. 角色调 `memrecall`：**当轮回复里就有书签**，聊天框也有回显；
2. `memsettle` 返回卡片（旧版必失败已修）；
3. `memwhen time="上周三"` 返回那天条目；`memstatus` 返回条数/范围；
4. 配置页：状态行 5 秒刷新、搜索能查、压缩/审计有结果提示；
5. 开 `AutoCompactEnable` 跑一轮后：
   - 日志出现 `已压缩 …`；
   - `MemoryFacts/compacted_sources.txt` 出现来源；
   - **官方 `Memory/L*/` 文件数量不变**（用 `Get-ChildItem -Recurse` 核对）；
   - 索引条目下降，`facts/` 来源出现在索引里；
6. `审计报告_YYYYMMDD.md` 生成。

## 六、已知未整改（下一步可选）

| 项 | 说明 |
| --- | --- |
| 本地 chunks 检索栈 | 与侧车索引检索重叠，只在侧车没就绪时兜底。**无法统一到 DB**：Alife 运行环境不带 SQLite 的 .NET 库（`resources\bin` 与插件树都没有 `Microsoft.Data.Sqlite/System.Data.SQLite`），C# 读不了 `memory.db` |
| `SkipRecentDays` 语义不一致 | 只影响本地 chunks，不影响侧车索引 |
| `memview` 与 ⑥ 内嵌浏览重复 | 保留 HTML 作"导出"用途 |
| 提示文案三处重复 | `HintTemplate` / XmlHandler / `ToolFirst`（无害）|

## 七、自检记录

- `py_compile` ✅
- 单元测试（18 项）✅
- **端到端协议测试**（真起子进程，14 个 op + 错误路径）✅
- **B 契约测试**：`build(extra=facts, exclude=官方文件)` → 排除的来源不进索引、**官方文件仍在**、facts 以 `facts/` 前缀入索引；`compact_scan` 也跳过已压缩来源 ✅
- Roslyn 语法检查 4 个 `.cs` ✅
- 语义编译仍只能在 Alife 启动时验证（插件 DLL 直引 `System.Private.CoreLib`，外部 `dotnet build` 报 CS0012）

---

## 八、存储层升级：SQLite（2026-10-18 下午）

之前是**单文件 JSON**：查询要把 36k 条全量载入内存再逐条打分 → 实测 **2.3 秒**，远超 200ms 的同步窗口。

现在换成 **SQLite（Python 自带，零依赖）**，仍然是"我们自己的一份派生库"，官方 `Memory/` 只读：

### 表结构

| 表 | 作用 |
| --- | --- |
| `entries` | 条目主键 `id`（source#块号）、anchor、时间/日/时段/小时、head/summary/body、人事物标签、生命周期字段（access_count / last_access / retired） |
| `files(path, mtime)` | **增量**依据：只重扫 mtime 变过的文件 |
| `vec(id, data)` | 条目向量的 float32 blob 缓存，避免重复嵌入 |
| `nodes(name, n)` / `edges(a,b,w)` | 互链：标签 + 正文实体共现 |
| `entries.*` 上的索引 | 时间、来源、关键词 LIKE 预筛 |

### 增量与迁移

- 建库时按 `files.mtime` 判断：只重扫变过的文件；文件消失则清掉它的条目。
- 首次运行若发现旧 `index.json`，**自动导入**一次（导入后不再依赖 JSON）。
- JSON 仍用于 `memview` 导出与人工查看，但**查询不再走 JSON**。

### 实测（真实索引：22.6 MB / 36427 条）

| 指标 | JSON | SQLite |
| --- | --- | --- |
| 召回（关键词路，同一台机器） | **2.323 s** | **0.037 s**（≈63 倍）|
| 二次查询 | — | 0.040 s |
| 全量载入 | 0.170~0.249 s（还要再打分） | 不适用（SQL 按需） |
| 一次性导入 | — | 1.13 s |
| 库体积 | 22.6 MB | 35.6 MB（索引换速度） |
| 增量重建（新增 1 个文件） | 全量重写 | 0.62 s |

### 兼容与降级

- 旧的 JSON 代码路径**全部保留**：`_load_index()` 优先走库，库不可用时自动退回 JSON。
- 生命周期（强化/消退）在库模式下直接 `UPDATE`，不再回写 JSON（回写会让 JSON 又长回来）。
- `compact_scan` / `audit_scan` / `profile` / `loops` / `bytime` / `maintain` / `view` 仍按原逻辑，只是数据来源变成库。

---

## 九、两轮排查记录（2026-10-18）

### 第一轮：功能 / 逻辑

| 检查 | 结果 |
| --- | --- |
| 互链类型：同人（主人-俾斯麦）、同事（健康-技术）、同物（血压计-ComfyUI）、实体（《记忆插件》-人/事/物） | ✅ 四类都正确 |
| `memlinks` 返回 `type` 字段、文本带类型（`名字(权重·类型)`） | ✅ |
| **旧库升级**：没有 `type` 列的库打开时自动 `ALTER` + 重建互链 | ✅ 类型已回填 |
| **增量删除**：删掉记忆文件后，条目与边一起清掉 | ✅ |
| **JSON 迁移**：空库 + 旧 `index.json` → 自动导入，并补建互链 | ✅ |
| 空库时的 `stats / links / recall` | ✅ 不抛异常 |
| 排除后官方文件仍在（B 契约） | ✅ |

> 首轮有 3 项"失败"，核对后都是**测试数据写错**（把不同条目的标签当成同一条的），代码无问题；修正用例后全绿。

### 第二轮：韧性 / 回归 / 性能

| 检查 | 结果 |
| --- | --- |
| **坏库降级**：把 `memory.db` 写成垃圾 → 打开失败、返回 None、上层退回 JSON 读取 | ✅ 不崩 |
| 建库目录不存在 | ✅ 不抛异常（返回 0/None）|
| Unicode / emoji（`心情😀`）建库、召回、浏览 | ✅ |
| 规模 2000 条：建库 0.09s / 召回 0.006s / 浏览 0.001s | ✅ |
| 重复 build 幂等（不翻倍） | ✅ |
| 全量回归：`test_db` / `test_db2` / `test_db3` / `test_sidecar` / `test_sidecar_e2e` | ✅ 全绿 |
| C# Roslyn 语法 4 文件 | ✅ |
| 静态一致性：C# 与侧车 op/字段名、无 `ArchiveRaw`/`KeepRaw`/`toolArgs` 残留 | ✅ |

### 真实索引基准（22.6 MB / 36427 条，第三次复测）

| 指标 | JSON | SQLite |
| --- | --- | --- |
| 召回 | 2.316 s | **0.039 s**（≈59×）|
| 二次查询 | — | 0.042 s |
| 一次性导入 | — | 1.47 s |
| 库体积 | — | 36.1 MB |
| 增量重建（新增 1 文件） | 全量重写 | 1.0 s（含互链重建）|

### `SkipRecentDays` 的正确语义（2026-10-18 修）

- **它是什么**：自动书签的"别提示我太新的记忆"开关，默认 1 天。
- **旧问题**：按**文件 mtime** 判断且只作用于本地 chunks 栈——同一文件里夹着的旧账会因文件被改过而整体跳过；主路径（索引召回）还根本不看它。
- **现在的行为**：
  1. 按**条目日期**（来自文件名，官方 L1 存档如 `1-20261005220027-….txt` 天然带日期）过滤，`date >= today - N` 的条目不参与**自动书签**；
  2. **手动工具**（`memrecall` / `memwhen`）永远不跳过——你主动查的东西不该被藏起来；
  3. 没有日期的条目按"不跳过"处理（宁可多给）。
- 实测：旧账（2026-09-01）+ 今天的新账 → `skip_recent_days=1` 时只返回旧账。

---

## 十、两轮复测（2026-10-18 晚，补完边类型后）

| 轮次 | py_compile | test_db | test_db2 | test_db3 | test_sidecar | test_e2e | C# 语法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 第 1 轮 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 第 2 轮 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

覆盖点（`test_db2` 已并入回归）：
- 七类边：同人 / 同事 / 同物 / 人-事 / 人-物 / 事-物 / 实体
- `memlinks` 的 `type` 过滤（只看"人-事"时不再带出"同人"边——修过 UNION 条件只作用于后半段的 bug）
- 旧库升级、增量删除、JSON 迁移、空库、排除契约
- 坏库降级、缺目录、Unicode/emoji、2000 条规模、幂等

### 模型整理 vs 自动压缩：怎么分工（2026-10-18 定）

| | 模型整理 `LlmOrganizeAsync` | 自动压缩 `AutoCompactAsync` |
| --- | --- | --- |
| 触发 | 每 `LlmOrganizeMinutes`（默认 120） | 每 `AutoCompactMinutes`（默认 360） |
| 对象 | **新**落盘、未整理过的文件 | **条目多且够旧**的流水（`AutoCompactMinEntries` / `MinAgeDays`） |
| 动作 | 模型提炼 → 事实卡（③ 合并写入） | 同上，但额外把来源记入 `compacted_sources.txt`（不再进索引） |

**协调规则**：当 `AutoCompactEnable=true` 时，organize 会先问侧车"哪些文件待压缩"（`compactscan`），**跳过它们**，避免同一份文件被两次喂模型。压缩关着时 organize 照旧处理全部文件，不会出现"没人管"的空档。

---

## 十一、三个小功能（2026-10-18 补）

### 1. `ExcludeSources`（手动排除）

配置里每行一个相对路径或**目录前缀**：

```
L1/
L2/2026-08
memory_20260701_旧账.md
```

- 只影响**我们的索引**：这些来源不建索引、不被压缩候选选中；
- 官方文件一个字节都不动，需要时把行删掉、重建索引就回来；
- 与自动压缩的 `compacted_sources.txt` 合并生效（两者取并集）。

### 2. 「重建干净库」按钮

管理页新增按钮 → 侧车 `reset`（关连接、删 `memory.db`）→ 立即按当前配置全量重建。

- **不会**删事实卡（`事实_*.md`）、不会清 `compacted_sources.txt`；
- 官方 `Memory/` 依旧只读；
- 用途：换了排除规则、或怀疑库脏了，一键重来。

### 3. `MigrateOldIndex`（迁移开关，默认开）

- 开：首次启动把官方目录里的旧 `index.json`（23 MB 那份）复制到 `MemoryFacts/` 导入 —— 省一次全量重建；
- 关：不迁移，直接从现有记忆文件建**全新的小库**（嫌旧索引太大时用这个）。

### 配套：想把俾斯麦的库变小

1. 关 `MigrateOldIndex`（或不装、先把旧 `Memory/index.json` 改名）；
2. 加 `ExcludeSources`：`L1/`（L1 占 85% 条目）；
3. 或开 `AutoCompactEnable` 让它们被压成事实卡后再排除；
4. 点「重建干净库」。

### 实测

| 检查 | 结果 |
| --- | --- |
| `exclude=["L1/"]` → L1 不进索引、官方文件仍在、库里无 L1 来源 | ✅ |
| 排除后 `compactscan` 候选为 0 | ✅ |
| `reset` → `memory.db` 被删、返回"索引库已清空" | ✅ |
| reset 后重建 → 数据回来 | ✅ |

---

## 十二、编译事故与语义编译校验器（2026-10-18 晚）

### 事故

装上线上后 Alife 报 3 个编译错误：

| 错误 | 原因 | 修法 |
| --- | --- | --- |
| `CS1061 IEnumerable<string> 未包含 ToArray` ×2（EmbedSidecar 363/364） | 我加了 `extra?.ToArray()` 但 `EmbedSidecar.cs` 没 `using System.Linq;` | 补 using |
| `CS0120 对象引用对于非静态方法 EnumerateMemoryFiles 是必需的`（MemoryInjectModule 655） | `NewestMemoryStamp` 还是 `static`，却调用了我从 static 改成实例的 `EnumerateMemoryFiles` | 改为实例方法 |

> 根因：此前只做了 **Roslyn 语法检查**（`CSharpSyntaxTree.ParseText`），它不查类型/引用，所以这两类错漏掉了。

### 新增：语义编译校验器

`D:\dsh\work\compilecheck\`（dotnet console）：

- 引用集 = 运行时 **TPA** + `Microsoft.AspNetCore.App` / `Microsoft.NETCore.App` 共享框架 + `D:\alife\resources\bin\*.dll` + `CompiledPlugins\*.dll`（只收托管程序集）；
- 用 `CSharpCompilation` 真编译 4 个源码文件，输出 `文件(行,列): 错误`；
- 用法：

```powershell
dotnet run --project D:\dsh\work\compilecheck\compilecheck.csproj -v q -- `
  <四个 .cs 路径>
```

**当前结果**：dev 副本与线上目录均 **0 error**（737 个引用）。

### 以后改 C# 的标准动作

1. 语法检查（syntaxcheck）→ 2. **语义编译（compilecheck）**→ 3. Python 测试 → 4. 才允许重载/重启。

---

## 十三、模型整理/压缩逻辑修复（2026-10-18，接真央三处修复之后）

真央修了「水位 key 对不上 / 自产出被扫 / 预算从旧档扣」三处，复查后**①③ 仍未真正解决**，另发现压缩侧一个丢数据问题。已全部修掉：

| # | 问题 | 修法 |
| --- | --- | --- |
| P0-1 | **排序在选材之后**：`fresh.Sort` 只改 prompt 拼接顺序，预算早在枚举时按"目录/文件名"扣光，新档照样被跳过 | 先收集全部候选 → 按 mtime **从新到旧**排序 → 再按预算选材 |
| P0-2 | **超预算整份跳过**（`txt.Length > budget → continue`）：单文件大于 `LlmOrganizeMaxChars` 时永远排不上 | 恢复**截断**（`Substring(0, budget)`），并记 warning |
| P0-3 | **水位仍是 mtime 键**：touch/复制/还原就整份重喂；`TrySaveState` 还按**字典序**砍到 500 条，超过后路径靠前的文件水位丢失 → 每轮重新整理 | 水位改 `relPath + 内容哈希(SHA1[..12])`；不再按字典序截断（上限 20000 仅防病态）；旧 `relPath|数字` 水位自动迁移（只认"水位写盘后 mtime 没再变"的文件）|
| P1-4 | **compact 饿死文件**：`lastCompact` 触发即更新，压缩失败时 organize 仍一直跳过待压缩文件 | 新增 `lastCompactOk`（清单空或至少压成一份才更新）；只有它新鲜（2×间隔内）才跳过 |
| P0-5 | **压缩丢尾巴**（侧车 `compact_scan`）：text 超 `max_chars` 时 `break`，只压前 N 条却把**整份来源**记成已压缩 → 尾巴既没进事实卡、又从索引消失 | 整份放不进预算就**不产出候选**；放宽预算后自然能压 |

### 同类问题排查（逐个位置）

| 位置 | 结论 |
| --- | --- |
| `EnsureChunks`（本地兜底） | 无预算/无水位，按 `EnumerateMemoryFiles` 排除规则走 —— 无同类问题 |
| `compact_scan` | 已修（见 P0-5）；`_is_excluded` + `facts/` 前缀已排除自产出 |
| `audit_scan` | 只扫 `事实_*.md`（不扫审计报告自身）；只审**最近 400 条**——防 O(n²) 的有意设计，已注明 |
| `MarkCompacted` / 排除清单 | 追加写，无截断；排除清单 30s 热加载 |
| `TrySaveState` | 已修（P0-3）|
| `db_build` 增量 | 按 `files.mtime`；mtime 不变而内容变属极端情况，可接受 |
| `render_view` / `settle` | 单次遍历，无预算/水位 |

### 残留风险（已记录，未改）

- organize 与 compact 仍可能对同一文件各跑一次（`pendingCompact` 是开头快照）：事实卡会合并去重，只浪费一次模型调用；
- 超过 `LlmOrganizeMaxChars` 的文件会被截断整理（会记 warning），完整内容仍保留在官方文件里，可调大上限。

---

## 十四、其他功能排查与修复（2026-10-18 晚，第二轮复查）

### 越界 / 丢数据

| # | 问题 | 修法 |
| --- | --- | --- |
| 1 | **审计报告写进官方 `Memory/`**（`AuditAsync` + `Audit()` 两处）——违反"官方目录只读" | 改写到 `MemoryFacts/` |
| 2 | `db_build` 清理只遍历 `files` 表 → **从旧 index.json 迁移进来的条目没有 files 行，被排除/删除的来源清不掉**（exclude 形同虚设） | 改成"按来源差集"清理（`seen` 之外一律删），并加测试 |
| 3 | `db_build` 在记忆目录不存在时会把库清空 | 目录不存在直接返回，不动库 |

### 功能缺失 / 回归

| # | 问题 | 修法 |
| --- | --- | --- |
| 4 | **DB 召回丢了图谱兜底**：旧 JSON 路径在零命中时 `graph_expand` 顺关系链捞，DB 版直接返回空 | 补回（扩词后再 LIKE 预筛 + 用扩词打分）|
| 5 | `MaxResults` 对侧车召回不生效（payload 里 `top_k` 写死 5） | `RecallSync/RecallAsync` 加 `topK`，ChatSend 传 `Configuration.MaxResults` |
| 6 | 关掉 `ToolFirst` 时提示仍让模型"去目录翻" | 改成"调 `<memfetch source=...>`" |

### 连接 / 缓存 / 并发

| # | 问题 | 修法 |
| --- | --- | --- |
| 7 | **换库不关旧连接**（多角色切换）→ 句柄泄漏 + 文件锁（测试里 `WinError 32` 暴露） | `_db_for` 切换前先 `close()` |
| 8 | **`db_build` 不更新 `_IDX["path"]`** → 内部 `_load_index()` 按旧路径开库，把当前连接关掉 → `Cannot operate on a closed database` | 建完库 `_set_index(outpath)` |
| 9 | `_db_ready` 不同步索引路径 → 图谱兜底时触发同一问题 | 入口统一 `_set_index(idx)` |
| 10 | build / reset 后 `_VEC_CACHE`、`_IDX`、`_GRAPH` 不失效（旧向量/旧图） | 全部清掉 |
| 11 | `chunks` 边清边读（`QueryKeyword` / `QueryByVector` / `ComputeVectors`）→ 可能抛异常或漏结果 | 先 `ToList()` 快照 |
| 12 | 配置页 5 秒定时器可能与慢查询重叠、堆积请求 | 加 `refreshing` 标志 |

### 已核对无问题

- 9 个工具的参数名与侧车 dispatch **全部匹配**（fetch / bywhen / profile / loops / links / browse / stats / settle）；
- `memsettle` 写入 `MemoryFacts/settled/`，`memfetch source=s…` 能按 id 取回卡片；
- browse/stats 返回字段与 C# 解析一致；
- 侧车退出清 pending、Dispose 杀进程；`warmDone` 不会永久卡住首条消息；
- 每个请求都会 `_set_index`，profile/loops/view 不会串到别人的索引。

### 残留（记录未改）

- organize 与 compact 仍可能对同一文件各跑一次（事实卡合并去重，只浪费一次模型调用）；
- `audit_scan` 只审最近 400 条（防 O(n²) 的有意设计）；
- 自动书签会追加进用户消息历史（前缀缓存友好，重复触发会累积）。
