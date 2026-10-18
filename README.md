# Mao.MemoryInject · 记忆检索层

> ⚠️ **本插件仍在迭代中，可能存在 bug。** 使用中如遇问题（索引为空、召回不准、报错等），欢迎在仓库 Issue 反馈。

## v4.8.0 (2026-10-18) —— B 方案重构：官方管"记住"，我们管"查得到、说得清"

**主存储换成 SQLite**（Python 自带，零依赖）。此前单文件 `index.json`（3.6 万条 / 22.6 MB）查询要全量载入再逐条打分，**实测召回 2.3 秒**，远超同步窗口 200ms。换库后：

```
召回   2.316 s → 0.039 s（≈59 倍）
二次   0.042 s
导入   1.47 s（一次性）      库 36.1 MB
增量重建（新增 1 文件） 1.0 s
```

库打不开（损坏/被占）→ **自动退回原 JSON 路径**，不抛异常。

### 三条硬边界

| | 官方 `Alife.Function.Memory` | 本插件 |
| --- | --- | --- |
| 存储 | ✅ owner：`Character\<名>\Memory\`（L1..L8、History.json、duckdb） | ❌ **只读**，一个字节都不写 |
| 历史压缩 | ✅ owner：`ChatHistoryAdd` → `MemoryManager` → `EditChatHistoryAsync` | ❌ 不做 |
| 提示注入 | 已清空 `RecallHintKeywords` | ✅ 唯一 owner：书签/工具提示 |
| 自己的写入 | — | ✅ 全落 `Character\<名>\MemoryFacts\` |

**自动压缩改成"标记排除"**：旧实现会 `File.Move(官方存档 → archive/)`，那会踩坏官方插件（它按 `L{n}` 路径读存档）。现在只把来源相对路径记进 `compacted_sources.txt`，重建索引时排除它——**官方文件原样保留**。

### 11 个工具

`memrecall`（按人事查书签）｜`memwhen`（中文时间词→那段时间，⑨时间下钻）｜`memfetch`（取原文）｜`memprofile`（按人聚画像）｜`memloops`（钓牵挂欠账）｜`memstatus`（索引体量）｜`memlinks`（图谱邻居，⑤带类型）｜`memcompact`（手动压流水）｜`memaudit`（审计重复矛盾）｜`memsettle`（长文本折卡片）｜`memview`（HTML 视图，⑥）

### 顺手修的原版 bug（12 条）

1. `memsettle` 取 `text` 字段但侧车回 `card` → **旧版必然"折叠失败"**
2. 四个工具返回 `Task`(void) → 模型当轮拿不到结果
3. `Fetch/Profile/Loops` 共用 `toolArgs` → 并发**串参**
4. `hasMemory` 只在 `OnAwake` 判一次 → 记忆目录后建就**永久不检索**
5. `EnsureChunks` 只比目录 mtime → 改文件内容片段不刷新
6. `maintain` 把 index 塞进 args 被覆盖
7. 压缩/整理并发写同一事实卡 → 丢更新
8. 压缩后没重建索引 → 旧条目残留
9. 近似重复但新句更详细时只刷日期 → 新信息丢失
10. 图谱缓存不失效 → 重建/换角色后用旧图
11. 审计 O(n²) 无上限 → 事实上千条卡死侧车
12. `browse` 关键词逐条 `json.dumps` → 3 万条时明显卡

### 行为变更

- `SkipRecentDays` 改成按**条目日期**（来自文件名）过滤，且**只作用于自动书签**；`memrecall`/`memwhen` 永不跳过。
- 模型整理与自动压缩已协调（压缩开着时整理会跳过待压缩文件）。
- 新增配置：`EchoToolResult`、`MergeThreshold`、`AutoCompactEnable/Minutes/MinEntries/MinAgeDays/MaxChars`、`AuditEnable/Minutes/Limit`、`ToolFirst`、`WhenLimit`、`ExcludeSources`、`MigrateOldIndex`。
- **`AutoCompactEnable` 默认关**（要调模型写事实卡，建议先跑一轮确认官方 `Memory/` 文件数不变再开）。
- `ExcludeSources`：每行一个相对路径或目录前缀（如 `L1/`），**只影响我们的索引，官方文件不动**。
- 首次启动 `MigrateOldIndex`（默认开）会把旧 `index.json` 复制到 `MemoryFacts/` 导入。

---

## 它是什么

给 Alife 角色装一层"嗅旧账"的鼻子：主人说话时，在后台把这句话跟角色的记忆档案比对一遍，
命中相关旧账就在消息末尾轻轻提示一句——**只告诉你去翻哪几份，不把内容塞进上下文**。

> 书签式注入（告诉你"哪份、哪件事、涉及谁"）＋ 工具式取书（想看全文再自己调工具捞）。

## 和官方 Alife.Function.Memory 的区别

- 官方记忆：命中"还记得""以前""上次"等**关键词**才提示，不认语义。
- 本插件：**不看关键词，每句话都做语义比对**，意思像就够了；还能按人/事/物翻旧账。
- 两条可并存，也可只留一条。

## 检索

关键词（字段权重 × IDF）＋ 语义（向量余弦）双路，用 **RRF 融合**排序；时间维可按天/时段/小时下钻；中文时间可解析（"上周三下午"→具体日期+时段）。

**生命周期**：命中即计一次访问，访问越多排序越上浮；半衰期衰减，越久没碰权重越低（强化过的不会被时间压死）。

**图谱**：书名号/引号/专名/拉丁词抽实体入图，边带类型（同人/同事/同物/人-事/人-物/事-物/实体/共现），`memlinks` 支持类型过滤。

## 行为原则

- **只读不写**，绝不改动记忆档案（除自己那份 `MemoryFacts/`）。
- **命中才提示**，没有相关旧账时一句话都不加。
- **add-only**：只往消息末尾"追加"临时书签，绝不改写原消息、不往历史里写（防击穿前缀缓存）。
- 任何异常一律**放过原消息**，绝不影响正常对话。
- 语义检索有**毫秒级硬上限**；侧车没就绪或超时立刻退回关键词匹配，绝不拖慢主线程。
- **双水位线迟滞缓冲**：攒够一定字数才查一次，查完回落，两次之间设最小间隔，避免每句都触发。

## 依赖

- `Alife.Function.FunctionCaller`
- 本机 Python 及 `torch`、`transformers`，并已下载 `BAAI/bge-small-zh-v1.5` 向量模型（与官方记忆插件同款）。未就绪时自动退回关键词模式，不报错。

## 安装与更新

1. 备份线上插件目录；
2. 覆盖文件：`MemoryInjectModule.cs`、`MemoryInjectConfig.cs`、`EmbedSidecar.cs`、`MemoryInjectUI.cs`、`sidecar_embed.py`、`manifest.json`；
3. **完整重启 Alife**（本插件禁走 reload——重载会掐死侧车驱动）。

首次启动会自动建 `MemoryFacts/`；旧 `Memory/` 只读。

## 参考与致谢

- **memcore**（misaka-coder，Apache-2.0）—— 借鉴 **RRF 排名融合**、**指数半衰期衰减**、**时间锚点**。
- **livingmemory / astrbot_plugin_livingmemory**（lxfight-s，AGPL-3.0）—— 借鉴 **add-only 注入原则**、**记忆生命周期三段式**、**图谱路**。
- **Alife.Function.Memory**（Alife 官方）—— 复用同款向量模型与侧车交互范式。

特别感谢以上作者把思路公开出来。**思想不受版权约束，借鉴的是思路；若有代码层面的相似之处，以各自许可证为准。** 本插件采用 MIT 许可。

## 作者

chickensoup
