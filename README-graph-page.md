# MemoryInject 混合关系图谱（开发状态）

## 目标与交互
独立于 Alife UI 的浏览器详情页，提供 Obsidian 原生变量主题/玻璃拟态切换、缩放和平移、缩小时隐藏标签、悬停邻接高亮、左键详情、右键展开/折叠与复制节点信息。实体节点大小反映直接关联数，文案应称“直接关联数”。

## 关系语义
- 实体共现关系：实线，反映记忆索引共现统计，不表示父子或因果。
- 来源文件与记忆条目的包含层级：虚线 `kind=hierarchy`。
- 实体提及条目关系：虚线 `kind=hierarchy`。
层级数据从索引 `entries.source` 与 `people/topics/things` 标签构建，不是完整目录树。当前没有向文件做写操作；来源路径被视为不可信数据，不使用 `file://` 直接打开。

## 重要状态
本项目仍是开发中，不要部署到线上插件或真实角色数据。HTTP 集成测试可用隔离 SQLite fixture 执行，已经覆盖 loopback 的服务启动、展开查询、并发请求、端口冲突和关闭清理；这不替代 Alife 真机或真实库性能测试。sidecar 的图谱查询以 read-only SQLite 打开既有 memory.db，不创建/迁移库；若 DB 缺失或 schema 不兼容，返回不可用。见 `HANDOFF.md`。

## 静态/单元检查
- `test_graphdata_temp.py`：18 项 in-memory SQLite 测试实体标签精确匹配、来源层级、source/entry 展开、节点限制和不可信 expand 输入；不要连接真实索引。
- `test_graph_http.py`：17 项 HTTP 集成测试，使用临时 SQLite 和 loopback 高位端口，包括固定 index、只读查询、不建库、只读目录、并发、异常和关闭清理。
- 运行 Python：`python -m unittest -v test_graphdata_temp.py test_graph_http.py`、`python -m py_compile sidecar_embed.py GraphServer.py test_graphdata_temp.py test_graph_http.py`。
- 从 HTML 抽取 `<script>` 到临时 JS 文件后运行 `node --check`。
- C# 用 Roslyn syntaxcheck 检查；插件目录没有 `.csproj/.sln`，需要 Alife 构建环境实际编译。
