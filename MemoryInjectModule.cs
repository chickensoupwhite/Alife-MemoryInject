
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ComponentModel;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Alife.Foundation;
using Alife.Framework;
using Alife.Function.FunctionCaller;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Mao.MemoryRecall;

/// <summary>
/// Mao.MemoryInject —— 记忆回忆提示模块（后台检索·命中才提示）。
///
/// 铁律（2026-10-05 修）：
///   1. 同步钩子 ChatSend 里**绝不无上限等待外部进程**。语义检索只允许
///      <see cref="Configuration"/> 里 SyncWaitMs 毫秒的有上限等待（侧车常驻后单次约 10ms）；
///      超时立刻退回纯内存关键词，绝不把 UI 线程拖死。
///   2. 连续超时达到阈值就进冷静期，暂停语义检索一段时间，避免每句都卡。
///   3. 侧车启动/批量向量化一律后台；批量期间 Busy 置位，同步路径直接放过。
///   4. 任何异常一律放过原消息，并记日志。
/// </summary>
[Module("记忆书签",
    "主人说话时在后台检索记忆目录，命中旧账只在消息末尾提示一句（不塞内容、不占上下文）；另做流水压缩、事实审计、写入合并与内嵌索引浏览。检索只读，压缩/审计只写事实卡与报告，异常一律放过。",
    defaultCategory: "Mao",
    editorUI: typeof(MemoryInjectUI))]
public class MemoryInjectModule(
    ILogger<MemoryInjectModule> logger,
    XmlFunctionCaller functionCaller,
    Interactor<MemoryInjectModule> interactor,
    ChatBot chatBot) :
    ChatBehaviour,
    IConfigurable<MemoryInjectConfig>
{
    public MemoryInjectConfig Configuration { get; set; } = null!;

    string memoryRoot = "";          // 主记忆目录（只读，兼容旧代码）
    readonly List<(string root, string prefix)> memoryRoots = new();
    string roleRoot = "";            // 角色根目录
    string factsRoot = "";           // 我们自己的事实层（写这里，绝不写官方目录）
    string compactStatePath = "";    // 已压缩来源清单（只影响我们的索引，不动官方文件）
    readonly HashSet<string> excludes = new(StringComparer.OrdinalIgnoreCase);
    readonly object excludesLock = new();
    DateTime lastExcludesLoad = DateTime.MinValue;
    string scriptPath = "";
    string indexPath = "";
    DateTime lastBuildCheck = DateTime.MinValue;
    readonly SemaphoreSlim rebuildLock = new(1, 1);
    volatile bool warmDone = false;   // 索引/模型预热是否完成（未完成时首次查询给长窗口）
    DateTime lastScanStamp = DateTime.MinValue;
    volatile bool hasMemory;   // 本角色有没有记忆文件：没有就别启侧车/别检索，免得串到别人的索引
    DateTime lastMemCheck = DateTime.MinValue;   // 上次重判 hasMemory 的时间（目录可能后建/后补文件）
    DateTime lastStampCalc = DateTime.MinValue;  // 上次计算"最新文件时间"的时间（节流用）

    readonly List<Chunk> chunks = [];
    readonly SemaphoreSlim scanLock = new(1, 1);
    readonly EmbedSidecar sidecar = new();

    volatile bool vectorsReady;
    volatile bool computing;
    int timeoutStreak;
    DateTime cooldownUntil = DateTime.MinValue;

    // 双水位线：攒够有效字数才查一次，查完回落；两次之间有最小间隔
    int waterLevel;
    DateTime lastTrigger = DateTime.MinValue;

    public string SidecarStatus => sidecar.Status;

    sealed class Chunk
    {
        public string Source = "";
        public string Text = "";
        public float[]? Vector;
    }

    // ===== 生命周期 =====

    DateTime lastMaintain = DateTime.MinValue;
    DateTime lastLlmOrganize = DateTime.MinValue;   // 上次整理时间；启动冷却由 startedAt/StartupDelay 单独控制
    DateTime startedAt = DateTime.Now;
    bool llmStartupLogged;
    DateTime lastCompactOk = DateTime.MinValue;     // 上次压缩"健康"完成（扫到清单且没积压/有产出）
    DateTime lastCompact = DateTime.MinValue;        // 上次自动压缩（流水 -> 事实卡）
    DateTime lastAudit = DateTime.MinValue;          // 上次事实审计
    volatile bool compacting;                        // 压缩/审计互斥，避免并发重复干活
    volatile bool auditing;
    volatile bool organizing;                      // 手动/自动整理互斥，避免同一批重复喂模型
    readonly SemaphoreSlim factLock = new(1, 1);      // 事实卡写入串行化（整理与压缩会并发）
    string llmStatePath = "";                        // 记录"已整理到哪个文件/时间"的水位

    bool migrationsApplied;

    /// <summary>配置可能晚于 OnAwake 注入：首次拿到 Configuration 时再跑迁移（幂等）。</summary>
    void EnsureMigrations()
    {
        if (migrationsApplied || Configuration == null) return;
        migrationsApplied = true;
        ApplyDefaultMigrations();
    }

    // 升级只迁移仍处于旧默认值的配置，保留用户自定义的排除规则。
    void ApplyDefaultMigrations()
    {
        if (Configuration == null) return;
        int ver = Configuration.DefaultsMigrationVersion;
        if (ver < 1)
        {
            if (string.IsNullOrWhiteSpace(Configuration.ExcludeSources))
                Configuration.ExcludeSources = "L1/";
            Configuration.DefaultsMigrationVersion = 1;
            ver = 1;
            logger.LogInformation("记忆书签：已应用默认规则迁移 v1（原排除规则为空时加入 L1/；用户自定义规则保留）");
        }
        if (ver < 2)
        {
            // v2（vnext）：语义检索默认关闭——旧配置里可能存着 true，这里统一关一次，避免召回被弱相关结果污染。
            Configuration.UseSemantic = false;
            Configuration.DefaultsMigrationVersion = 2;
            logger.LogInformation("记忆书签：已应用默认规则迁移 v2（语义召回默认关闭；需要时可在配置里重新打开）");
        }
    }

    // 判断目录里有没有记忆文件（.md/.txt，递归；跳过 settled/backup/.bak）
    static bool HasMemoryFiles(string dir)
    {
        try
        {
            if (!Directory.Exists(dir)) return false;
            foreach (string f in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
            {
                string name = Path.GetFileName(f).ToLowerInvariant();
                if (name.EndsWith(".bak")) continue;
                string full = f.Replace('\\', '/').ToLowerInvariant();
                if (full.Contains("/settled/") || full.Contains("/backup/") || full.Contains("/archive/")) continue;
                if (name.EndsWith(".md") || name.EndsWith(".txt")) return true;
            }
        }
        catch { }
        return false;
    }

    // 解析记忆目录：配置路径存在就用它；否则按候选顺序自动探测，兼容不同角色的目录结构
    string ResolveMemoryRoot(string roleRoot, string configuredRoot)
    {
        memoryRoots.Clear();
        var candidates = new List<string> { configuredRoot };
        string raw = Configuration?.MemorySourcePaths ?? "";
        foreach (string part in raw.Split(new[] { '\n', '\r', ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string rel = part.Trim().Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            if (rel.Length == 0 || Path.IsPathRooted(rel))
            {
                if (rel.Length > 0) logger.LogWarning("记忆书签：忽略角色目录外的记忆来源配置：{Path}", part.Trim());
                continue;
            }
            string candidate = Path.GetFullPath(Path.Combine(roleRoot, rel));
            if (candidate.StartsWith(Path.GetFullPath(roleRoot).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                candidates.Add(candidate);
            else
                logger.LogWarning("记忆书签：忽略越过角色根目录的来源配置：{Path}", part.Trim());
        }
        candidates.Add(Path.Combine(roleRoot, "Memory"));
        candidates.Add(Path.Combine(roleRoot, "Storage/Memory"));
        int sourceNo = 0;
        foreach (string path in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (Directory.Exists(path) && HasMemoryFiles(path))
            {
                string prefix = sourceNo++ == 0 ? "" : "source" + sourceNo + "/";
                memoryRoots.Add((path, prefix));
            }
        }
        if (memoryRoots.Count == 0) memoryRoots.Add((configuredRoot, ""));
        return memoryRoots[0].root;
    }

    protected override Task OnAwake()
    {
        try
        {
            roleRoot = Path.Combine(AlifePath.StorageFolderPath, Character.StorageKey);
            EnsureMigrations();
            string sub = (Configuration?.MemorySubPath ?? "Storage/Memory")
                .Replace('/', Path.DirectorySeparatorChar)
                .Replace('\\', Path.DirectorySeparatorChar);
            memoryRoot = ResolveMemoryRoot(roleRoot, Path.Combine(roleRoot, sub));
            // 不写死目录名：先试本插件目录、再退回旧名，谁在就用谁，避免改名后静默找不到脚本
            {
                string pluginsRoot = Path.Combine(AlifePath.StorageFolderPath, "Plugins");
                scriptPath = "";
                foreach (string dir in new[] { "Mao.MemoryRecall", "Mao.MemoryInject" })
                {
                    string cand = Path.Combine(pluginsRoot, dir, "sidecar_embed.py");
                    if (File.Exists(cand)) { scriptPath = cand; break; }
                }
                if (string.IsNullOrEmpty(scriptPath))
                    scriptPath = Path.Combine(pluginsRoot, "Mao.MemoryRecall", "sidecar_embed.py");

                // ★B 方案：我们只写自己的目录 MemoryFacts/，官方 Memory/ 只读
                factsRoot = Path.Combine(roleRoot, "MemoryFacts");
                try { Directory.CreateDirectory(factsRoot); } catch { }
                compactStatePath = Path.Combine(factsRoot, "compacted_sources.txt");
                indexPath = Path.Combine(factsRoot, "index.json");
                llmStatePath = Path.Combine(factsRoot, "llm_organize_state.txt");
                // 首次升级：把旧的索引/水位从官方目录搬过来（可在配置里关掉）
                try
                {
                    if (Configuration == null || Configuration.MigrateOldIndex)
                    {
                        string oldIdx = Path.Combine(memoryRoot, "index.json");
                        if (!File.Exists(indexPath) && File.Exists(oldIdx)) File.Copy(oldIdx, indexPath);
                        string oldState = Path.Combine(memoryRoot, "llm_organize_state.txt");
                        if (!File.Exists(llmStatePath) && File.Exists(oldState)) File.Copy(oldState, llmStatePath);
                    }
                }
                catch (Exception) { }
            }
        }
        catch (Exception e)
        {
            logger.LogError(e, "记忆回忆提示：路径初始化失败");
        }

        // 清掉历史换库留下的 swap / 旧备份（每份几十~几百 MB，不清会把盘吃满）
        try { PruneBackups(); } catch { }
        // 一次性迁移：清掉事实卡里的 XML/代码垃圾（自动，带备份；历史污染走手动 memcleanhistory）
        try { MigrateFactCards(); } catch { }

        // 本角色有没有记忆文件：没有就别启侧车、别检索（否则会串到别人的索引）
        try { hasMemory = memoryRoots.Any(x => HasMemoryFiles(x.root)); } catch { hasMemory = false; }

        try { ChatBot.ChatSend += OnChatSend; }
        catch (Exception e) { logger.LogError(e, "记忆回忆提示：挂载 ChatSend 失败"); }

        // 主动调用的工具（取原文/画像卡/钓欠账/按人事物查书签），隐式注入
        try
        {
            XmlHandler handler = new(this)
            {
                Description = "记忆检索工具：翻自己落盘的旧账。想确认往事、查来龙去脉、按时间找、盘点欠账时，**只能**用这里的工具，" +
                              "禁止用 <Python>、读文件、扫目录、sqlite3 去查记忆（既慢又容易读到过期数据）。" +
                              "memrecall 按人事查书签；memwhen 按时间翻；memfetch 取原文；memprofile 聚画像；memloops 钓欠账；" +
                              "memstatus 看索引；memlinks 顺关系；memdebug 解释召回；memcompact 压流水；memaudit 审矛盾；memcleanhistory 清历史合成消息；" +
                              "memorganize 空闲时小批整理（系统报点/没人说话时，可主动整理一小批未整理的流水）。",
                Explanation =
                    "这些函数：检索类是只读的；压缩/审计只写事实卡与报告，不动原文。" +
                    "★铁律：查记忆只用下面这些工具；禁止用 <Python>、glob、读文件、`sqlite3` 去翻记忆目录或索引库——" +
                    "那属于本工具的职责，绕过工具既慢又容易拿到过期/未整理的数据。" +
                    "<memrecall query=\"关键词\"/> 查书签；" +
                    "<memwhen time=\"上周三\" keyword=\"可选\" limit=\"8\"/> 按时间翻旧账；" +
                    "<memfetch source=\"memory_xxx.md\" 或 keyword=\"关键词\" limit=\"3\"/> 取原文；" +
                    "<memprofile who=\"主人/小鲸鱼/俾斯麦/妈\" limit=\"30\"/> 按人聚画像卡；" +
                    "<memloops keyword=\"可选\" limit=\"15\"/> 扫牵挂与欠账；" +
                    "<memstatus/> 看索引多少条、覆盖到哪天；" +
                    "<memcompact/> 手动压一次流水；" +
                    "<memorganize max_files=\"5\"/> 手动整理一小批流水（别超 20，免得占满上下文；系统报点、没人说话的闲时可主动调它，一次一两段即可）；" +
                    "<memaudit/> 手动审一次事实卡；" +
                    "<memlinks query=\"俾斯麦\"/> 看某个词/人/物常和什么一起出现。",
            };
            // Explicit：每轮把工具说明/参数表注入系统提示。Implicit 只露一行，模型常不知道有这些工具。
            functionCaller.RegisterHandler(handler, DocumentMode.Explicit, DestroyCancellationToken);
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆回忆提示：XmlHandler 注册失败（角色未启用 FunctionCaller 时属预期）");
        }

        // 扫描一次，把记忆目录切成片段（纯文件 IO，不碰网络/进程）
        try { EnsureChunks(force: true); }
        catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：首次扫描失败"); }

        // 侧车只在开关打开时启动；重活一律丢后台，绝不在 OnAwake 里等
        // ★B：侧车提供索引召回（主路径），不该被"语义开关"卡住；UseSemantic 只控制向量那一路
        if (Configuration != null && Configuration.Enable && hasMemory)
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    bool ok = await sidecar.StartAsync(scriptPath, Configuration.PythonExe, loadModel: Configuration.UseSemantic).ConfigureAwait(false);
                    logger.LogInformation("记忆回忆提示：向量侧车 {Ok}，{Status}", ok, sidecar.Status);
                    if (ok)
                    {
                        // 先确保索引在：缺了或旧了，就地重建（后台，不占前台）
                        await EnsureIndexAsync(force: false).ConfigureAwait(false);
                        // 预热：把索引读进内存、建好词表/DF，并吃掉模型首次编码开销。
                        // 大库首次冷读要 2s+，会超出同步等待窗口；预热放到启动后台，不占真查询。
                        try
                        {
                            await sidecar.AskAsync("warm", new Dictionary<string, object?>(), 120000, indexPath).ConfigureAwait(false);
                            warmDone = true;
                        }
                        catch { }
                        // ★只有开启语义才需要本地片段的向量；关着还算会导致侧车惰性加载模型、
                        //   白编码几千条片段（启动慢 + 吃显存），而且这些向量根本不会被用到。
                        if (Configuration.UseSemantic)
                            await ComputeVectorsAsync().ConfigureAwait(false);
                    }
                }
                catch (Exception e)
                {
                    logger.LogWarning(e, "记忆回忆提示：向量侧车异常（不影响主流程）");
                }
            });
        }

        // ⑨后台调度：定期自动整理索引（新记忆落盘 → 自动并进索引，不用手动重建）+ 每日维护（衰减/清理）
        _ = Task.Run(async () =>
        {
            while (true)
            {
                // 自动整理：记忆目录里有比索引新的文件就自动重建（EnsureIndexAsync 内部已判断"旧了才建"，并有节流）
                try
                {
                    if (sidecar.IsReady && warmDone)
                        await EnsureIndexAsync(force: false).ConfigureAwait(false);
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：自动整理索引失败（不影响主流程）"); }

                try
                {
                    if (sidecar.IsReady && (DateTime.Now - lastMaintain).TotalHours >= Math.Max(1, Configuration.MaintainIntervalHours))
                    {
                        sidecar.AskSync("maintain", new Dictionary<string, object?>(), 8000, indexPath);
                        lastMaintain = DateTime.Now;
                        logger.LogInformation("记忆回忆提示：每日维护已执行（衰减/清理）");
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：每日维护失败（不影响主流程）"); }

                // 模型整理：新落盘的流水交给模型提炼成事实卡（默认两小时一次，精度比机械切块高）
                try
                {
                    int startupDelay = Math.Max(0, Configuration?.LlmOrganizeStartupDelayMinutes ?? 120);
                    int interval = Math.Max(5, Configuration?.LlmOrganizeMinutes ?? 120);
                    bool startupCooldownDone = (DateTime.Now - startedAt).TotalMinutes >= startupDelay;
                    bool intervalDone = (DateTime.Now - lastLlmOrganize).TotalMinutes >= interval;
                    if (!startupCooldownDone)
                    {
                        if (!llmStartupLogged)
                        {
                            llmStartupLogged = true;
                            logger.LogInformation("记忆书签：模型整理启动冷却中，{Minutes} 分钟后才允许自动整理", startupDelay);
                        }
                    }
                    else if (intervalDone)
                    {
                        lastLlmOrganize = DateTime.Now;
                        await LlmOrganizeAsync().ConfigureAwait(false);
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：模型整理失败（不影响主流程）"); }

                // ① 自动压缩：流水占索引大头时压成事实卡（默认六小时一次；开关默认关，确认稳了再开）
                try
                {
                    if (Configuration != null && Configuration.AutoCompactEnable
                        && (DateTime.Now - lastCompact).TotalMinutes >= Math.Max(30, Configuration.AutoCompactMinutes))
                    {
                        await AutoCompactAsync().ConfigureAwait(false);
                        lastCompact = DateTime.Now;
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：自动压缩失败（不影响主流程）"); }

                // ② 审计：事实卡查重/查可能矛盾，只写报告（默认两小时一次）
                try
                {
                    if (Configuration != null && Configuration.AuditEnable
                        && (DateTime.Now - lastAudit).TotalMinutes >= Math.Max(10, Configuration.AuditMinutes))
                    {
                        await AuditAsync().ConfigureAwait(false);
                        lastAudit = DateTime.Now;
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：审计失败（不影响主流程）"); }

                try { await Task.Delay(TimeSpan.FromMinutes(Math.Max(2, Configuration.AutoOrganizeMinutes)), DestroyCancellationToken); }
                catch { break; }
            }
        });

        logger.LogInformation("记忆回忆提示已挂载，记忆目录：{Root}", memoryRoot);
        return Task.CompletedTask;
    }

    /// <summary>⑧终局结算：把一段长输出折成卡片（历史里只留卡片+id），正文挪到 settled/ 目录，需要时用 memfetch id 取回。</summary>
    [XmlFunction(FunctionMode.OneShot, "memsettle")]
    [Description("把一段很长的输出折成卡片：历史里只留卡片和 id，正文挪到 settled/，需要时再用 memfetch 取回。")]
    public async Task<string> SettleOutput(
        [Description("要折叠的长文本")] string text,
        [Description("可选：卡片 id，不填自动生成")] string card_id = "")
    {
        try
        {
            var args = new Dictionary<string, object?> { ["text"] = text, ["card_id"] = card_id };
            // ★settle 回的是 card 字段，不是 text —— 用 AskAsync 会永远取不到（旧版就是这里一直"折叠失败"）
            string? card = await sidecar.AskFieldAsync("settle", args, "card", 8000, indexPath).ConfigureAwait(false);
            return R(string.IsNullOrWhiteSpace(card) ? "折叠失败（侧车没返回卡片）。" : card!);
        }
        catch (Exception e) { return "折叠失败：" + e.Message; }
    }

    /// <summary>⑩生成记忆索引视图 HTML 并打开，一键查看全部条目。</summary>
    [XmlFunction(FunctionMode.OneShot, "memview")]
    [Description("生成记忆索引的网页视图并打开，方便人看（不用于查单个问题）。")]
    public async Task<string> ViewIndex()
    {
        try
        {
            string html = Path.Combine(factsRoot, "记忆索引视图.html");
            await sidecar.AskAsync("view", new Dictionary<string, object?> { ["out"] = html }, 8000, indexPath).ConfigureAwait(false);
            if (File.Exists(html))
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(html) { UseShellExecute = true });
            return R(File.Exists(html) ? "记忆索引视图已生成并打开：" + html : "视图没生成成功（侧车可能没就绪）。");
        }
        catch (Exception e) { return "生成视图失败：" + e.Message; }
    }

    /// <summary>供管理页调用：生成并打开记忆索引视图，返回结果文字。</summary>
    public async Task<string> ViewIndexAsync()
    {
        try
        {
            string html = Path.Combine(factsRoot, "记忆索引视图.html");
            await sidecar.AskAsync("view", new Dictionary<string, object?> { ["out"] = html }, 8000, indexPath).ConfigureAwait(false);
            if (File.Exists(html))
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(html) { UseShellExecute = true });
            return File.Exists(html) ? "视图已生成并打开：" + html : "视图没生成成功（侧车可能没就绪）。";
        }
        catch (Exception e) { return "生成视图失败：" + e.Message; }
    }

    /// <summary>构建前预览：只统计来源和排除规则，不写索引。</summary>
    public string PreviewBuild()
    {
        try
        {
            var rows = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            int total = 0;
            foreach (var item in EnumerateAllMemoryFiles())
            {
                total++;
                string key = item.file.StartsWith(roleRoot, StringComparison.OrdinalIgnoreCase) ? item.file.Substring(roleRoot.Length).TrimStart('\\', '/') : item.file;
                key = key.Split(new[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "?";
                rows[key] = rows.TryGetValue(key, out int n) ? n + 1 : 1;
            }
            string excluded = string.Join(", ", ExcludeSnapshot());
            return "本次预计扫描 " + total + " 个文件；来源：" + string.Join(", ", rows.Select(x => x.Key + "=" + x.Value)) +
                   "；排除：" + (excluded.Length == 0 ? "无" : excluded) + "；索引只写入 MemoryFacts，官方来源只读。";
        }
        catch (Exception e) { return "预览失败：" + e.Message; }
    }

    void WriteManifest(string result)
    {
        try
        {
            string path = Path.Combine(factsRoot, "index_manifest.json");
            var manifest = new { version = "next", builtAt = DateTime.Now, blockMax = Configuration?.BlockMaxChars ?? 200,
                exclude = ExcludeSnapshot(), roots = memoryRoots.Select(x => x.root).ToArray(), result };
            File.WriteAllText(path, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：写入索引构建指纹失败"); }
    }

    /// <summary>供管理页调用：重建索引；失败时保留原库，成功后备份上一个版本。</summary>
    public async Task<string> RebuildIndexAsync() => await RebuildIndexCoreAsync(forceClean: false).ConfigureAwait(false);

    /// <summary>兼容原有按钮语义：按当前规则完整扫描；新库成功之前不删除旧库。</summary>
    public async Task<string> RebuildCleanAsync() => await RebuildIndexCoreAsync(forceClean: true).ConfigureAwait(false);

    async Task<string> RebuildIndexCoreAsync(bool forceClean)
    {
        if (!await rebuildLock.WaitAsync(0).ConfigureAwait(false)) return "已有索引重建在运行，请等待完成。";
        string dbPath = Path.Combine(factsRoot, "memory.db");
        string indexBackup = "";
        string dbBackup = "";
        string jsonBackup = "";
        string dbSwap = "";
        string tempDir = Path.Combine(factsRoot, ".rebuild-" + Guid.NewGuid().ToString("N"));
        string tempIndex = Path.Combine(tempDir, "index.json");
        string tempDb = Path.Combine(tempDir, "memory.db");
        try
        {
            Directory.CreateDirectory(factsRoot);
            if (string.IsNullOrWhiteSpace(memoryRoot) || !Directory.Exists(memoryRoot))
                return "重建取消：记忆根目录不存在，旧索引保持不变。";
            if (!sidecar.IsReady) return "重建失败：侧车未就绪，旧索引保持不变。";
            EnsureExcludes();
            Directory.CreateDirectory(tempDir);

            // 先释放当前库句柄，Windows 下才能安全替换数据库文件；不删除旧库。
            var closeOld = await sidecar.AskJsonAsync("close", new Dictionary<string, object?>(), 15000, indexPath).ConfigureAwait(false);
            if (closeOld == null) return "重建失败：无法释放旧索引连接，旧索引保持不变。";
            if (forceClean)
            {
                var reset = await sidecar.AskJsonAsync("reset", new Dictionary<string, object?>(), 15000, tempIndex).ConfigureAwait(false);
                if (reset == null || !reset.Value.TryGetProperty("ok", out JsonElement resetOk) || !resetOk.GetBoolean())
                    return "重建失败：无法准备隔离的新库，旧索引保持不变。";
            }

            string? result = await sidecar.AskAsync("build",
                new Dictionary<string, object?>
                {
                    ["memdir"] = memoryRoot,
                    ["out"] = tempIndex,
                    ["block_max"] = (Configuration?.BlockMaxChars ?? 200),
                    ["extra"] = BuildExtraRoots(),
                    ["exclude"] = ExcludeSnapshot(),
                }, 180000, tempIndex).ConfigureAwait(false);
            if (string.IsNullOrWhiteSpace(result)) return "重建失败或超时：旧索引保持不变。";
            if (!File.Exists(tempDb)) return "重建失败：新数据库未生成，旧索引保持不变。";
            var closeNew = await sidecar.AskJsonAsync("close", new Dictionary<string, object?>(), 15000, tempIndex).ConfigureAwait(false);
            if (closeNew == null) return "重建失败：无法释放新索引连接，旧索引保持不变。";

            string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            if (File.Exists(indexPath))
            {
                indexBackup = indexPath + ".bak-" + stamp;
                File.Copy(indexPath, indexBackup, false);
            }
            if (File.Exists(dbPath))
            {
                dbBackup = dbPath + ".bak-" + stamp;
                File.Copy(dbPath, dbBackup, false);
            }
            if (File.Exists(indexPath))
            {
                jsonBackup = indexPath + ".swap-" + Guid.NewGuid().ToString("N");
                File.Move(indexPath, jsonBackup);
            }
            try
            {
                if (File.Exists(dbPath))
                {
                    dbSwap = dbPath + ".swap-" + Guid.NewGuid().ToString("N");
                    File.Move(dbPath, dbSwap);
                }
                File.Move(tempDb, dbPath);
                if (File.Exists(tempIndex)) File.Delete(tempIndex);
                if (File.Exists(jsonBackup)) File.Delete(jsonBackup);
            }
            catch
            {
                if (!File.Exists(dbPath) && dbSwap.Length > 0 && File.Exists(dbSwap)) File.Move(dbSwap, dbPath);
                if (!File.Exists(indexPath) && File.Exists(jsonBackup)) File.Move(jsonBackup, indexPath);
                throw;
            }
            await sidecar.AskAsync("stats", new Dictionary<string, object?>(), 5000, indexPath).ConfigureAwait(false);
            WriteManifest(result.Trim());
            lastBuildCheck = DateTime.MinValue;
            return "安全重建完成：" + result.Trim() + "；原索引备份：" + (dbBackup.Length > 0 ? dbBackup : (indexBackup.Length > 0 ? indexBackup : "无旧库"));
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆书签：安全索引重建失败");
            return "重建失败，旧索引已保留" + (dbBackup.Length > 0 ? "（备份：" + dbBackup + "）" : "") + "：" + e.Message;
        }
        finally
        {
            try { if (File.Exists(tempIndex)) File.Delete(tempIndex); } catch { }
            try { if (File.Exists(tempDb)) File.Delete(tempDb); } catch { }
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true); } catch { }
            // ★换库成功后 swap 文件会留着（每份几十~几百 MB），必须清掉
            try { if (dbSwap.Length > 0 && File.Exists(dbSwap)) File.Delete(dbSwap); } catch { }
            try { if (jsonBackup.Length > 0 && File.Exists(jsonBackup)) File.Delete(jsonBackup); } catch { }
            PruneBackups();
            rebuildLock.Release();
        }
    }

    /// <summary>只保留最近 2 份备份，避免 MemoryFacts 被 .bak 撑爆（每份几十~几百 MB）。</summary>
    void PruneBackups()
    {
        try
        {
            foreach (string pat in new[] { "memory.db.bak-*", "index.json.bak-*" })
            {
                var files = Directory.GetFiles(factsRoot, pat).OrderByDescending(f => f, StringComparer.Ordinal).ToList();
                foreach (string old in files.Skip(2))
                    try { File.Delete(old); } catch { }
            }
            // 换库失败/中断留下的 swap 副本（每份几十~几百 MB），启动和重建后都清掉
            foreach (string sw in Directory.GetFiles(factsRoot, "*.swap-*"))
                try { File.Delete(sw); } catch { }
        }
        catch { }
    }

    protected override Task OnDestroy()
    {
        try { ChatBot.ChatSend -= OnChatSend; } catch { }
        _ = Task.Run(async () => { try { await sidecar.DisposeAsync(); } catch { } });
        return Task.CompletedTask;
    }

    // ===== 主流程 =====

    /// <summary>
    /// 剔除系统尾巴与系统标签（只作用于召回查询，不改动原消息/历史）。
    /// 全面策略：
    ///  1) 行首元数据（当前时间/消息来源/系统时间/来源/角色/场景）连同 [标签] 一起去掉；
    ///  2) [系统]/【系统提示】这类独立标签去掉；
    ///  3) 整行都是括号的段落：含短语，或系统标记词 ≥2 → 整行删；
    ///  4) 行内括号段：含短语，或系统标记词 ≥3 → 删（"（笑）"这类保留）；
    ///  5) 裸短语（≥4 字）抹掉；
    ///  6) 去零宽字符、折叠空白、清掉纯标点行。
    /// </summary>
    string StripExcludedPhrases(string message)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(message)) return message;
            string raw = Configuration?.QueryExcludePhrases;
            List<string> phrases = string.IsNullOrWhiteSpace(raw)
                ? new List<string>()
                : raw.Split(new[] { ',', '，', ';', '；' }, StringSplitOptions.RemoveEmptyEntries)
                     .Select(x => x.Trim()).Where(x => x.Length >= 2).ToList();
            string[] markers = { "请", "注意", "保持", "不要", "禁止", "使用", "遵守", "规范", "示例", "旁白", "表情", "emoji", "报点", "系统", "输出", "格式" };

            string result = message;
            // 0) 零宽/不可见字符
            result = Regex.Replace(result, @"[\u200B-\u200D\uFEFF]", "");
            // 0b) HTML 注释（系统提示常以注释形式注入）
            result = Regex.Replace(result, @"<!--[\s\S]*?-->", " ");
            // 1) 行首元数据标签（可带 [标签]）
            result = Regex.Replace(result, @"(?im)^[ \t]*(?:当前时间|消息来源|系统时间|来源|角色|场景)[ \t]*[:：][ \t]*(?:\[[^\]]*\])?[ \t]*", "");
            // 2) [系统]/【系统提示】这类独立标签
            result = Regex.Replace(result, @"[\[【][ \t]*(?:系统|system)(?:提示|消息|通知)?[ \t]*[\]】]", " ");

            bool HasPhrase(string text)
            {
                foreach (string p in phrases) if (text.Contains(p, StringComparison.OrdinalIgnoreCase)) return true;
                return false;
            }
            int CountMarkers(string text)
            {
                int n = 0;
                foreach (string mk in markers) if (text.Contains(mk, StringComparison.OrdinalIgnoreCase)) n++;
                return n;
            }
            string Normalize(string text) => Regex.Replace(text, @"[【】「」《》〈〉\[\]<>]", "");

            // 3) 整行括号段：含短语，或系统标记词 ≥2 → 删整行
            result = Regex.Replace(result, @"(?m)^[ \t]*[（(][^\n]*[）)][ \t]*$", mm =>
            {
                string norm = Normalize(mm.Value);
                return (HasPhrase(norm) || CountMarkers(norm) >= 2) ? " " : mm.Value;
            });
            // 4) 行内括号段：含短语，或系统标记 ≥3 → 删（保留"（笑）""（请帮我查一下）"这类用户补充）
            result = Regex.Replace(result, @"[（(][^（()）\n]*[)）]", mm =>
            {
                string norm = Normalize(mm.Value);
                return (HasPhrase(norm) || CountMarkers(norm) >= 3) ? " " : mm.Value;
            });
            // 5) 去掉装饰括号后再匹配裸短语：这样"注意遵守【对话输出规范】"也能整段命中
            result = Regex.Replace(result, @"[【】「」《》〈〉\[\]<>]", "");
            foreach (string p in phrases.Where(x => x.Length >= 4))
                result = result.Replace(p, " ", StringComparison.OrdinalIgnoreCase);
            // 6) 收尾：清纯标点行、折叠所有空白（含换行）
            result = Regex.Replace(result, @"(?m)^[ \t]*[，,。.；;：:、\-—]+[ \t]*$", "");
            result = Regex.Replace(result, @"\s+", " ");
            return result.Trim();
        }
        catch
        {
            return message;   // 任何异常都放过原消息，绝不因为过滤把话弄丢
        }
    }


    /// <summary>明显不是事实的行：XML 注释/标签、代码围栏、Python 代码。</summary>
    static bool IsJunkFactLine(string t)
    {
        if (Regex.IsMatch(t, @"^[-*・]?\s*<!?-?-?.*-->")) return true;
        if (Regex.IsMatch(t, @"^[-*・]?\s*</?[A-Za-z][^>]*/?>")) return true;
        if (Regex.IsMatch(t, @"^[-*・]?\s*\x60\x60\x60")) return true;
        if (Regex.IsMatch(t, @"^[-*・]?\s*(import |from |for |if |def |print\(|os\.|re\.|base\s*=|p\s*=|t\s*=|return |class )")) return true;
        return false;
    }

    /// <summary>
    /// 4.8.6 一次性迁移：清掉所有事实卡里的 XML/注释/代码行，只删明显垃圾，散文与自由格式保留。
    /// 每个文件先备份 .bak-migrate-4.8.6；写标记文件，只跑一次。
    /// </summary>
    void MigrateFactCards()
    {
        try
        {
            string marker = Path.Combine(factsRoot, ".migrated-facts-4.8.6");
            if (File.Exists(marker)) return;
            int files = 0, removed = 0;
            var roots = new List<string>();
            foreach (var item in memoryRoots) roots.Add(item.root);
            roots.Add(factsRoot);
            foreach (string root in roots.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (Directory.Exists(root) == false) continue;
                foreach (string f in Directory.EnumerateFiles(root, "事实_*.md", SearchOption.AllDirectories))
                {
                    if (f.Contains(".bak", StringComparison.OrdinalIgnoreCase)) continue;
                    try
                    {
                        var lines = File.ReadAllLines(f);
                        var kept = new List<string>();
                        int dropped = 0;
                        bool inCode = false;
                        foreach (string ln in lines)
                        {
                            string t = ln.Trim();
                            if (t.StartsWith("\x60\x60\x60")) { inCode = !inCode; dropped++; continue; }
                            if (inCode) { dropped++; continue; }
                            if (t.Length > 0 && IsJunkFactLine(t)) { dropped++; continue; }
                            kept.Add(ln);
                        }
                        if (dropped == 0) continue;
                        File.Copy(f, f + ".bak-migrate-4.8.6", overwrite: true);
                        File.WriteAllLines(f, kept);
                        files++;
                        removed += dropped;
                    }
                    catch { }
                }
            }
            Directory.CreateDirectory(factsRoot);
            File.WriteAllText(marker, DateTime.Now.ToString("o") + "\nfiles=" + files + " removed=" + removed);
            logger.LogInformation("记忆书签：事实卡迁移完成，清理 {f} 个文件 / {n} 行垃圾", files, removed);
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：事实卡迁移失败（下次启动重试）"); }
    }

    static bool IsSyntheticHistory(ChatMessageContent m)
    {
        string c = m?.Content ?? "";
        return c.Contains("[消息来源(记忆整理)]", StringComparison.Ordinal)
            || c.Contains("[消息来源(记忆压缩)]", StringComparison.Ordinal);
    }

    /// <summary>手动清理历史里由整理/压缩写入的合成提示词（默认只统计；apply=true 才删，删前备份 History.json）。</summary>
    async Task<(int count, int chars, string backup)> CleanHistoryAsync(bool apply)
    {
        int count = 0, chars = 0;
        foreach (ChatMessageContent m in ChatBot.ChatHistory)
        {
            if (IsSyntheticHistory(m) == false) continue;
            count++;
            chars += (m.Content ?? "").Length;
        }
        if (apply == false || count == 0) return (count, chars, "");

        string backup = "";
        try
        {
            string hist = Path.Combine(roleRoot, "Memory", "History.json");
            if (File.Exists(hist))
            {
                backup = hist + ".bak-cleanhistory-" + DateTime.Now.ToString("yyyyMMddHHmmss");
                File.Copy(hist, backup, true);
            }
        }
        catch { }

        int removed = 0;
        await ChatBot.EditChatHistoryAsync(thread =>
        {
            var h = thread.ChatHistory;
            for (int i = h.Count - 1; i >= 0; i--)
            {
                if (IsSyntheticHistory(h[i]) == false) continue;
                // 先删它的配对回包（紧随其后、assistant）
                if (i + 1 < h.Count && h[i + 1].Role == AuthorRole.Assistant) h.RemoveAt(i + 1);
                h.RemoveAt(i);
                removed++;
            }
            return Task.CompletedTask;
        }, "记忆书签：清理历史合成消息").ConfigureAwait(false);
        return (removed, chars, backup);
    }

    string OnChatSend(string message)
    {
        try
        {
            EnsureMigrations();
            if (Configuration == null || Configuration.Enable == false) return message;
            if (string.IsNullOrWhiteSpace(message)) return message;
            if (message.Contains(ChatBot.PokeMessageTag, StringComparison.Ordinal)) return message;
            // 本角色没有记忆文件：不检索，绝不串到别人的索引。
            // ★但目录可能是后建/后补的：60 秒重判一次，别因为启动时没有就永久判死。
            if (hasMemory == false && (DateTime.Now - lastMemCheck).TotalSeconds >= 60)
            {
                lastMemCheck = DateTime.Now;
                try { hasMemory = memoryRoots.Any(x => HasMemoryFiles(x.root)); } catch { }
            }
            if (hasMemory == false) return message;

            // 先剔掉 [来源][...] 这类方括号标签，再数有效字数
            string stripped = Regex.Replace(message, @"\[[^\]]*\]", " ");
            stripped = StripExcludedPhrases(stripped);
            string bare = new string(stripped.Where(char.IsLetterOrDigit).ToArray());
            if (bare.Length < Configuration.MinQueryChars) return message;

            // 双水位线：先攒水，不到高水位或还在最小间隔内，就不查
            waterLevel += bare.Length;
            int high = Math.Max(1, Configuration.WaterHighChars);
            double gap = Math.Max(0, Configuration.TriggerCooldownSeconds);
            if (waterLevel < high) return message;
            if ((DateTime.Now - lastTrigger).TotalSeconds < gap) return message;
            // 到点：回落水位、记下本次触发（无论查没查到，都算数，避免每句重试）
            waterLevel = Math.Max(0, Configuration.WaterLowChars);
            lastTrigger = DateTime.Now;

            // 第一优先：按 人/事/物 查 index.json，拿到可读书签就直接用
            // （有上限等待，超时/异常一律放过，退回下面的原提示）
            int waitMs = Math.Max(Configuration.SyncWaitMs, 200);
            if (warmDone == false) waitMs = Math.Max(waitMs, 5000);   // 预热未完成：给大库冷读留出时间，别当没命中
            // ★SkipRecentDays 只作用于"自动书签"（按条目日期跳过最近 N 天）；手动工具传 0，永不跳过
            string? bookmarks = sidecar.RecallSync(stripped, waitMs, indexPath, Configuration.HalfLifeDays, Configuration.UseGraph, Configuration.SkipRecentDays, Configuration.MaxResults, Configuration.UseSemantic, Configuration.SemanticMinScore);
            if (string.IsNullOrWhiteSpace(bookmarks) == false)
            {
                // ★只追加：书签拼在本轮用户消息末尾，用户原话一字不动。
                //   框架 ChatSend 的返回值即本轮新消息内容（会进历史），前缀不变，前缀缓存照常命中。
                string tail = Configuration.ToolFirst
                    ? "\n[要用全文：调 <memfetch source=\"出处文件名\"/> 或 <memwhen time=\"上周三\"/>，不要扫盘、不要凭印象编。]"
                    : "\n[要用全文：调 <memfetch source=\"出处文件名\"/>，别凭印象编。]";
                string block =
                    "[记忆书签·临时·仅本轮参考（按人/事/物查 index.json）]\n" + bookmarks + tail;
                logger.LogInformation("记忆回忆提示：按人事物命中，已附书签");
                return message + "\n" + block;
            }

            EnsureChunks();
            if (chunks.Count == 0) return message;

            List<Chunk> hits = [];

            // 语义：有上限等待（默认 150ms），超时/冷却期立刻退回关键词
            if (Configuration.UseSemantic && vectorsReady && sidecar.IsReady && sidecar.Busy == false
                && DateTime.Now >= cooldownUntil)
            {
                float[]? q = sidecar.EmbedSync(stripped, Configuration.SyncWaitMs);
                if (q != null)
                {
                    timeoutStreak = 0;
                    hits = QueryByVector(q);
                }
                else
                {
                    // 没拿到向量：不计数为超时（可能只是 Busy），只在连续失败时冷静
                    if (Interlocked.Increment(ref timeoutStreak) >= Math.Max(1, Configuration.TimeoutStreakLimit))
                    {
                        cooldownUntil = DateTime.Now.AddSeconds(Math.Max(5, Configuration.CooldownSeconds));
                        timeoutStreak = 0;
                        logger.LogWarning("记忆回忆提示：语义连续超时，冷静 {S} 秒", Configuration.CooldownSeconds);
                    }
                }
            }

            if (hits.Count == 0) hits = QueryKeyword(stripped);
            if (hits.Count == 0) return message;

            var files = hits.Select(h => Path.GetFileNameWithoutExtension(h.Source))
                            .Distinct().Take(Configuration.MaxResults);
            string list = string.Join("、", files);

            string hint = (Configuration.HintTemplate ?? "")
                .Replace("{n}", hits.Count.ToString())
                .Replace("{files}", list)
                .Replace("{path}", memoryRoot);

            logger.LogInformation("记忆回忆提示：命中 {N} 段（{Files}）", hits.Count, list);
            return message + "\n" + hint;
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆回忆提示失败，已放过原消息");
            return message;
        }
    }

    // ===== 主动调用的工具（XmlFunction） =====
    // ★必须返回 Task<string>：返回值当轮就能给模型；interactor.Poke 是挂进聊天框，模型要到下一轮才看得到。
    //   （旧版四个工具全是 Task + Poke —— 调了等于白调，这是"工具调用"最要命的 bug。）
    // ★参数不要共用字段：并发调用会串参，一律用局部字典。

    /// <summary>按人/事/物查书签。</summary>
    [XmlFunction(FunctionMode.OneShot, "memrecall")]
    [Description("按人/事/物查记忆书签。想确认往事、查来龙去脉时第一个用它。")]
    public async Task<string> Recall(
        [Description("要查的内容，比如某人、某件事、某个物件")] string query)
    {
        // ★区分两种情况：真的没命中 vs 侧车没响应（超时/报错）。旧版一律说"没翻到"，误导排查。
        JsonElement? r = await sidecar.AskJsonAsync("recall", new Dictionary<string, object?>
        {
            ["query"] = query ?? "",
            ["top_k"] = Math.Clamp(Configuration.MaxResults, 1, 20),
            ["half_life"] = Configuration.HalfLifeDays,
            ["use_graph"] = Configuration.UseGraph,
            ["use_semantic"] = Configuration.UseSemantic,
            ["semantic_min_score"] = Configuration.SemanticMinScore,
        }, 8000, indexPath).ConfigureAwait(false);
        string? bm = (r != null && r.Value.TryGetProperty("bookmarks", out JsonElement bEl)) ? (bEl.GetString() ?? "") : null;
        string msg = bm == null
            ? $"召回没响应（侧车未就绪或超时），不是没有记忆。稍后再试，或用 memstatus 看侧车状态。"
            : string.IsNullOrWhiteSpace(bm)
                ? $"记忆里没翻到跟「{query}」相关的旧账。可以换 memwhen 按时间找，或换个关键词。"
                : bm!;
        Echo(msg);
        return msg;
    }

    /// <summary>取原文：按出处文件名或关键字，把记忆正文捞出来。</summary>
    [XmlFunction(FunctionMode.OneShot, "memfetch")]
    [Description("按出处文件名或关键字取记忆原文。memrecall 的结果里带文件名，想细看就把它传进来。")]
    public async Task<string> Fetch(
        [Description("出处文件名，如 memory_20261012_夜.md；按文件名取时填这个")] string? source = null,
        [Description("关键字；不按文件名时用它搜正文")] string? keyword = null,
        [Description("最多取几段，默认3")] int limit = 3)
    {
        var args = new Dictionary<string, object?>
        {
            ["source"] = source ?? "",
            ["keyword"] = keyword ?? "",
            ["limit"] = limit,
        };
        string? txt = await sidecar.AskAsync("fetch", args, 5000, indexPath).ConfigureAwait(false);
        return R(string.IsNullOrWhiteSpace(txt) ? "没取到原文（换个文件名或关键字再试）。" : txt!);
    }

    /// <summary>按时间翻旧账：中文时间词直接传，日/时段/小时都能定位。</summary>
    [XmlFunction(FunctionMode.OneShot, "memwhen")]
    [Description("按时间翻旧账。传一句中文时间词（上周三/三天前/昨晚/上个月），直接返回那段时间记得的事，不用自己算日期。")]
    public async Task<string> When(
        [Description("时间说法，如 上周三 / 三天前 / 昨晚 / 上个月")] string time,
        [Description("可选：再加个关键词收窄")] string? keyword = null,
        [Description("最多几条，默认8")] int limit = 8)
    {
        try
        {
            int k = limit <= 0 ? Math.Max(1, Configuration.WhenLimit) : limit;
            var args = new Dictionary<string, object?>
            {
                ["text"] = time ?? "",
                ["keyword"] = keyword ?? "",
                ["top_k"] = k,
                ["use_graph"] = Configuration.UseGraph,
                ["use_semantic"] = Configuration.UseSemantic,
            };
            string? bm = await sidecar.AskFieldAsync("bywhen", args, "bookmarks", 8000, indexPath).ConfigureAwait(false);
            return R(string.IsNullOrWhiteSpace(bm) ? $"「{time}」那段时间没翻到记忆（或者说不清是哪段）。换个说法试试，比如「上周三」「三天前」。" : bm!);
        }
        catch (Exception e) { return "按时间查失败：" + e.Message; }
    }

    [XmlFunction(FunctionMode.OneShot, "memdebug")]
    [Description("解释一次记忆召回：显示候选数量、关键词/语义模式、阈值、最近过滤与最终结果，排查为什么召回为空。")]
    public async Task<string> DebugRecall(string query, int top = 5)
    {
        try
        {
            string? text = await sidecar.AskAsync("debug", new Dictionary<string, object?>
            {
                ["query"] = query ?? "", ["top_k"] = top, ["skip_recent_days"] = 0,
                ["use_graph"] = Configuration.UseGraph,
                ["use_semantic"] = Configuration.UseSemantic,
                ["semantic_min_score"] = Configuration.SemanticMinScore
            }, 8000, indexPath).ConfigureAwait(false);
            return R(string.IsNullOrWhiteSpace(text) ? "调试查询无结果（侧车可能未就绪）。" : text!);
        }
        catch (Exception e) { return "召回调试失败：" + e.Message; }
    }

    /// <summary>索引体量/覆盖范围及诊断信息。</summary>
    [XmlFunction(FunctionMode.OneShot, "memstatus")]
    [Description("诊断记忆根目录、索引新旧、排除规则与侧车状态。怀疑没扫到记忆或召回为空时使用。")]
    public async Task<string> Status()
    {
        try
        {
            string? txt = await sidecar.AskAsync("stats", new Dictionary<string, object?>(), 5000, indexPath).ConfigureAwait(false);
            string dbPath = Path.Combine(factsRoot, "memory.db");
            string store = File.Exists(dbPath) ? dbPath : indexPath;
            string storeInfo = File.Exists(store)
                ? "索引文件 " + Math.Round(new FileInfo(store).Length / 1024.0 / 1024.0, 1) + " MB，更新于 " + File.GetLastWriteTime(store).ToString("yyyy-MM-dd HH:mm")
                : "索引文件不存在";
            string newest = "未知";
            int fileCount = 0, l1Count = 0, otherCount = 0;
            var sourceSummary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            DateTime newestTime = DateTime.MinValue;
            string[] excluded = ExcludeSnapshot();
            foreach (var item in EnumerateAllMemoryFiles())
            {
                fileCount++;
                string rel = item.source;
                string sourceName = item.file.StartsWith(roleRoot, StringComparison.OrdinalIgnoreCase) ? item.file.Substring(roleRoot.Length).TrimStart('\\', '/') : item.file;
                sourceName = sourceName.Split(new[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "?";
                sourceSummary[sourceName] = sourceSummary.TryGetValue(sourceName, out int prior) ? prior + 1 : 1;
                if (rel.StartsWith("L1/", StringComparison.OrdinalIgnoreCase) || rel.IndexOf("/L1/", StringComparison.OrdinalIgnoreCase) >= 0) l1Count++;
                else otherCount++;
                try { if (File.GetLastWriteTime(item.file) > newestTime) newestTime = File.GetLastWriteTime(item.file); } catch { }
            }
            if (newestTime > DateTime.MinValue) newest = newestTime.ToString("yyyy-MM-dd HH:mm");
            DateTime storeTime = File.Exists(store) ? File.GetLastWriteTime(store) : DateTime.MinValue;
            string freshness = newestTime == DateTime.MinValue ? "无源文件"
                : !File.Exists(store) ? "待建索引"
                : newestTime > storeTime ? "索引可能落后（有更新源文件）" : "索引时间不早于源文件";
            bool excludesL1 = excluded.Any(x => x.Equals("L1/", StringComparison.OrdinalIgnoreCase));
            string rule = string.Join(", ", excluded);
            if (rule.Length == 0) rule = "（无）";
            return R((string.IsNullOrWhiteSpace(txt) ? "索引统计暂不可用（侧车可能未就绪）。" : txt!) +
                "｜根目录 " + memoryRoot + "｜来源文件 " + fileCount + "（L1 " + l1Count + "，其他 " + otherCount + "）" +
                "｜来源分布 " + string.Join(", ", sourceSummary.Select(x => x.Key + ":" + x.Value)) +
                "｜排除 " + rule + "｜L1 " + (excludesL1 ? "已排除" : "未排除") +
                "｜最新源文件 " + newest + "｜" + freshness + "｜" + storeInfo + "｜侧车 " + sidecar.Status);
        }
        catch (Exception e) { return "看索引状态失败：" + e.Message; }
    }

    /// <summary>⑤ 图谱邻居：某个词/人/物跟谁常一起出现，顺着关系找线索。</summary>
    [XmlFunction(FunctionMode.OneShot, "memlinks")]
    [Description("看图谱：某个词/人/物跟哪些东西常一起出现（比如「俾斯麦」常和哪些事一起出现）。想顺关系找线索时用。")]
    public async Task<string> Links(
        [Description("节点名或一句话，如 俾斯麦 / 血压计 / 记忆插件")] string query,
        [Description("最多列几个邻居，默认6")] int top = 6,
        [Description("只看某类关系，可选：同人/同事/同物/人-事/人-物/事-物/实体")] string type = "")
    {
        try
        {
            var args = new Dictionary<string, object?> { ["query"] = query ?? "", ["top"] = top, ["type"] = type ?? "" };
            string? txt = await sidecar.AskFieldAsync("links", args, "text", 6000, indexPath).ConfigureAwait(false);
            string msg = string.IsNullOrWhiteSpace(txt)
                ? $"图谱里没找到跟「{query}」相关的节点（试试更短的词，比如人名或物件名）。"
                : txt!;
            return R(msg);
        }
        catch (Exception e) { return "查图谱失败：" + e.Message; }
    }

    /// <summary>压缩候选预览，不调用模型、不写文件。</summary>
    [XmlFunction(FunctionMode.OneShot, "memcompactpreview")]
    [Description("预览哪些记忆流水达到自动压缩门槛；只读，不调用模型，不写事实卡。")]
    public async Task<string> CompactPreview()
    {
        try
        {
            JsonElement? scan = await sidecar.AskJsonAsync("compactscan", new Dictionary<string, object?>
            {
                ["memdir"] = memoryRoot, ["min_entries"] = Configuration.AutoCompactMinEntries,
                ["min_age_days"] = Configuration.AutoCompactMinAgeDays, ["max_chars"] = 0,   // probe：只数候选，不抓正文
                ["top_k"] = 100, ["exclude"] = ExcludeSnapshot()
            }, 15000, indexPath).ConfigureAwait(false);
            if (scan == null || !scan.Value.TryGetProperty("candidates", out JsonElement items)) return R("压缩候选暂时读不到（侧车可能未就绪）。");
            int n = items.ValueKind == JsonValueKind.Array ? items.GetArrayLength() : 0;
            return R("当前有 " + n + " 个压缩候选；门槛为至少 " + Configuration.AutoCompactMinEntries + " 条且至少 " + Configuration.AutoCompactMinAgeDays + " 天。确认后再调用 memcompact。" );
        }
        catch (Exception e) { return "预览压缩失败：" + e.Message; }
    }

    /// <summary>手动清理历史里的合成提示词（默认只统计）。</summary>
    [XmlFunction(FunctionMode.OneShot, "memcleanhistory")]
    [Description("清理聊天历史里由本插件整理/压缩写入的合成提示词（默认只统计不删；apply=true 才真删，删前自动备份 History.json）。")]
    public async Task<string> CleanHistory([Description("true=真正删除；默认 false 只统计")] bool apply = false)
    {
        try
        {
            var (n, chars, bak) = await CleanHistoryAsync(apply).ConfigureAwait(false);
            if (n == 0) return R("历史里没有本插件写入的合成消息，干净。");
            if (apply == false)
                return R($"历史里有 {n} 条合成消息（{chars} 字），尚未删除。确认后可调 memcleanhistory apply=true。");
            return R($"已清理 {n} 条合成消息（{chars} 字）。" + (string.IsNullOrEmpty(bak) ? "" : "备份：" + bak));
        }
        catch (Exception e) { return "清理历史失败：" + e.Message; }
    }

    /// <summary>手动整理一批记忆流水（空闲时用；批次别大，模型上下文有限）。</summary>
    [XmlFunction(FunctionMode.OneShot, "memorganize")]
    [Description("手动整理一小批记忆流水（从新到旧），提炼成事实卡。空闲或觉得事实卡没跟上时用。默认只整理 5 个源文件（模型上下文有限，别一次喂太多）。")]
    public async Task<string> Organize(
        [Description("这一批最多整理几个源文件，默认 5，最大 20（每个文件还会按字数预算截断）")] int max_files = 5)
    {
        try
        {
            if (organizing) return R("正在整理中（自动或上一次手动还没跑完），稍后再试。");
            int cap = Math.Clamp(max_files <= 0 ? 5 : max_files, 1, 20);
            int chars = Math.Clamp(Configuration.LlmOrganizeManualMaxChars, 2000, Configuration.LlmOrganizeMaxChars);
            var (n, fact) = await LlmOrganizeAsync(cap, force: true, maxChars: chars).ConfigureAwait(false);
            if (n == 0) return R("没有需要整理的流水（都已整理过，或模型/侧车没就绪）。");
            string tail = string.IsNullOrWhiteSpace(fact)
                ? "模型认为这批没长期价值，已推进水位。"
                : "事实卡 -> " + Path.GetFileName(fact);
            return R($"已整理 {n} 份流水：{tail}（批次刻意小，避免占满上下文；需要继续就再调一次）");
        }
        catch (Exception e) { return "整理失败：" + e.Message; }
    }

    /// <summary>① 手动触发一次压缩。</summary>
    [XmlFunction(FunctionMode.OneShot, "memcompact")]
    [Description("把条目多且够旧的记忆流水压成事实卡（只把来源记为「已压缩」、不再进索引，官方文件原样不动）。")]
    public async Task<string> Compact()
    {
        int n = await AutoCompactAsync(force: true).ConfigureAwait(false);
        string msg = n > 0
            ? ("已压缩 " + n + " 份流水，事实已并入事实卡；这些来源不再进我们的索引（官方文件未动）。")
            : "没有需要压缩的流水（条目数/天数门槛未到，或侧车没就绪）。";
        return R(msg);
    }

    /// <summary>② 手动触发一次审计。</summary>
    [XmlFunction(FunctionMode.OneShot, "memaudit")]
    [Description("扫事实卡里的重复或可能矛盾，写一份审计报告（只报告，不自动改）。")]
    public async Task<string> Audit()
    {
        int n = await AuditAsync(force: true).ConfigureAwait(false);
        string report = Path.Combine(factsRoot, "审计报告_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
        string msg = n > 0 ? ("审计新增 " + n + " 条问题，报告：" + report) : "没发现新的重复或矛盾。";
        return R(msg);
    }

    /// <summary>按人聚画像卡：某人的事按时间排成一页。</summary>
    [XmlFunction(FunctionMode.OneShot, "memprofile")]
    [Description("把某个人的事按时间聚成一页画像卡。")]
    public async Task<string> Profile(
        [Description("谁，如 主人/小鲸鱼/俾斯麦/妈")] string who,
        [Description("最多列几条，默认30")] int limit = 30)
    {
        var args = new Dictionary<string, object?> { ["who"] = who, ["limit"] = limit };
        string? txt = await sidecar.AskAsync("profile", args, 5000, indexPath).ConfigureAwait(false);
        return R(string.IsNullOrWhiteSpace(txt) ? $"没翻到跟「{who}」相关的记忆。" : txt!);
    }

    /// <summary>钓牵挂与欠账：扫记忆里的待办、还欠、没动笔这类字。</summary>
    [XmlFunction(FunctionMode.OneShot, "memloops")]
    [Description("扫记忆里的牵挂与欠账（待办、还欠、没动笔这类）。")]
    public async Task<string> Loops(
        [Description("可选关键字，如 运动/作文；不填则扫全部牵挂词")] string? keyword = null,
        [Description("最多列几条，默认15")] int limit = 15)
    {
        var args = new Dictionary<string, object?> { ["keyword"] = keyword ?? "", ["limit"] = limit };
        string? txt = await sidecar.AskAsync("loops", args, 5000, indexPath).ConfigureAwait(false);
        return R(string.IsNullOrWhiteSpace(txt) ? "没扫到牵挂或欠账。" : txt!);
    }

    // ===== 扫描 =====

    /// <summary>
    /// 目录内最新写入时间（含目录自身）。
    /// 只看目录 mtime 会漏掉"改已有文件、不改目录"的情况，导致改过正文后片段不刷新。
    /// </summary>
    DateTime NewestMemoryStamp(string root)
    {
        DateTime newest = Directory.Exists(root) ? Directory.GetLastWriteTime(root) : DateTime.MinValue;
        foreach (var item in EnumerateAllMemoryFiles())
        {
            try { DateTime t = File.GetLastWriteTime(item.file); if (t > newest) newest = t; } catch { }
        }
        return newest;
    }

    void EnsureChunks(bool force = false)
    {
        if (Directory.Exists(memoryRoot) == false) return;

        DateTime stamp;
        try
        {
            // 重算有 IO 成本：10 秒内复用上次结果
            if (lastStampCalc != DateTime.MinValue && (DateTime.Now - lastStampCalc).TotalSeconds < 10)
                stamp = lastScanStamp;
            else
            {
                stamp = NewestMemoryStamp(memoryRoot);
                lastStampCalc = DateTime.Now;
            }
        }
        catch { return; }

        if (force == false && stamp == lastScanStamp && chunks.Count > 0) return;

        if (scanLock.Wait(0) == false) return;   // 不等待，另一次扫描在跑就先用旧的
        try
        {
            if (force == false && stamp == lastScanStamp && chunks.Count > 0) return;
            lastScanStamp = stamp;
            chunks.Clear();
            vectorsReady = false;

            DateTime cutoff = DateTime.Now.AddDays(-Math.Max(0, Configuration.SkipRecentDays));
            foreach (var item in EnumerateAllMemoryFiles())
            {
                string file = item.file;
                try
                {
                    if (File.GetLastWriteTime(file) >= cutoff) continue;
                    string text = File.ReadAllText(file);
                    foreach (string seg in SplitSegments(text))
                    {
                        if (seg.Length < 6) continue;
                        chunks.Add(new Chunk { Source = file, Text = seg });
                    }
                }
                catch { }
            }
            logger.LogInformation("记忆回忆提示：扫描到 {N} 段（{F} 个文件）", chunks.Count,
                chunks.Select(c => c.Source).Distinct().Count());
        }
        finally { scanLock.Release(); }
    }

    static IEnumerable<string> SplitSegments(string text)
    {
        foreach (string raw in text.Split('\n'))
        {
            string s = raw.Trim().TrimStart('-', '*', '#', '>', ' ').Replace("**", "").Trim();
            if (s.Length == 0) continue;
            if (s.StartsWith('|') || s.StartsWith("---") || s.StartsWith('`')) continue;
            yield return s;
        }
    }

    // ===== 向量 =====

    static IEnumerable<string> EnumerateFilesSafe(string root)
    {
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            string dir = pending.Pop();
            IEnumerable<string> files;
            try { files = Directory.EnumerateFiles(dir, "*", SearchOption.TopDirectoryOnly).ToArray(); }
            catch { continue; }
            foreach (string file in files)
            {
                string ext = Path.GetExtension(file);
                if (ext.Equals(".md", StringComparison.OrdinalIgnoreCase) || ext.Equals(".txt", StringComparison.OrdinalIgnoreCase))
                    yield return file;
            }
            IEnumerable<string> dirs;
            try { dirs = Directory.EnumerateDirectories(dir, "*", SearchOption.TopDirectoryOnly).ToArray(); }
            catch { continue; }
            foreach (string child in dirs)
            {
                string name = Path.GetFileName(child);
                if (name.StartsWith(".", StringComparison.Ordinal) ||
                    name.Equals("settled", StringComparison.OrdinalIgnoreCase) ||
                    name.Equals("backup", StringComparison.OrdinalIgnoreCase) ||
                    name.Equals("archive", StringComparison.OrdinalIgnoreCase) ||
                    name.Equals("MemoryFacts", StringComparison.OrdinalIgnoreCase)) continue;
                pending.Push(child);
            }
        }
    }

    /// <summary>枚举记忆目录下的记忆文件（兼容 .md 与 .txt；跳过已压缩来源——它们只是不进我们的索引）。</summary>
    IEnumerable<string> EnumerateMemoryFiles(string root)
    {
        EnsureExcludes();
        string[] exSnapshot = ExcludeSnapshot();
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) yield break;
        foreach (string f in EnumerateFilesSafe(root))
        {
            string rel = f.Substring(root.Length).Replace('\\', '/').TrimStart('/');
            if (rel.Contains(".bak", StringComparison.OrdinalIgnoreCase)) continue;
            if (IsExcludedRel(rel, exSnapshot)) continue;
            yield return f;
        }
    }

    string[] BuildExtraRoots()
    {
        var roots = memoryRoots.Skip(1).Select(x => x.root).Where(Directory.Exists).ToList();
        roots.Add(factsRoot);
        return roots.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    IEnumerable<(string file, string source)> EnumerateAllMemoryFiles()
    {
        foreach (var item in memoryRoots.ToArray())
            foreach (string f in EnumerateMemoryFiles(item.root))
                yield return (f, item.prefix + f.Substring(item.root.Length).Replace('\\', '/').TrimStart('/'));
    }

    /// <summary>
    /// 确保索引可用：index.json 不存在、或比记忆目录最新文件还旧，就请侧车重建。
    /// 重建慢，一律后台；失败只记日志，绝不外抛。
    /// </summary>
    async Task EnsureIndexAsync(bool force)
    {
        if (sidecar.IsReady == false) return;
        if (string.IsNullOrWhiteSpace(indexPath)) return;
        if (Directory.Exists(memoryRoot) == false) return;

        // 节流：非强制时，两分钟内只查一次
        if (force == false && (DateTime.Now - lastBuildCheck).TotalSeconds < 120) return;
        lastBuildCheck = DateTime.Now;

        try
        {
            // ★SQLite 模式下索引实体是 memory.db，index.json 可能不存在
            string dbPath = Path.Combine(factsRoot, "memory.db");
            bool hasStore = File.Exists(indexPath) || File.Exists(dbPath);
            bool need = force || hasStore == false;
            if (need == false)
            {
                DateTime idxTime = File.Exists(dbPath) ? File.GetLastWriteTime(dbPath) : File.GetLastWriteTime(indexPath);
                DateTime newest = idxTime;
                foreach (var item in EnumerateAllMemoryFiles())
                {
                    try { DateTime t = File.GetLastWriteTime(item.file); if (t > newest) newest = t; } catch { }
                }
                need = newest > idxTime;   // 有比索引还新的记忆 → 重建
            }
            string manifestPath = Path.Combine(factsRoot, "index_manifest.json");
            if (need == false && !File.Exists(manifestPath)) need = true;
            if (need == false) return;

            EnsureExcludes();
            var r = await sidecar.BuildAsync(memoryRoot, indexPath, blockMax: Configuration.BlockMaxChars,
                extra: BuildExtraRoots(), exclude: ExcludeSnapshot()).ConfigureAwait(false);
            if (r != null)
            {
                WriteManifest("auto: " + r.Value.count + " entries");
                logger.LogInformation("记忆书签：索引已重建 {N} 条 -> {Out}", r.Value.count, r.Value.outPath);
            }
            else logger.LogWarning("记忆书签：索引重建未成功（不影响主流程）");
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆书签：索引重建异常（已忽略）");
        }
    }

    /// <summary>
    /// 模型整理：把新落盘的记忆流水交给模型，提炼成"事实卡"追加落盘（原文一律不动）。
    /// 走后台模型调用（ChatAsync，breakLast:false），不进主对话上下文。
    /// 水位记在 llm_organize_state.txt：记"已整理过的 文件|mtime"，只处理没整理过的新文件。
    /// </summary>
    async Task<(int files, string fact)> LlmOrganizeAsync(int maxFiles = 0, bool force = false, int maxChars = 0)
    {
        if (organizing) return (0, "");
        organizing = true;
        try { return await LlmOrganizeCoreAsync(maxFiles, force, maxChars).ConfigureAwait(false); }
        finally { organizing = false; }
    }

    async Task<(int files, string fact)> LlmOrganizeCoreAsync(int maxFiles, bool force, int maxChars = 0)
    {
        if (Configuration == null) return (0, "");
        if (force == false && Configuration.LlmOrganizeEnable == false) return (0, "");
        // ★自动整理绝不在模型回合进行中插 ChatAsync：ChatAsync 会写 ChatHistory，
        //   与 UI 渲染并发时会抛 "Collection was modified"。手动工具(force)不受此限。
        if (force == false && functionCaller.IsIdle == false) return (0, "");
        if (string.IsNullOrWhiteSpace(memoryRoot) || Directory.Exists(memoryRoot) == false) return (0, "");

        // 读水位：已整理过的 文件相对路径|mtime 集合
        var done = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var legacyDone = new HashSet<string>(StringComparer.OrdinalIgnoreCase);   // 旧 mtime 水位对应的路径
        DateTime stateTime = DateTime.MinValue;                                  // 水位文件写盘时间
        try { if (File.Exists(llmStatePath)) stateTime = File.GetLastWriteTime(llmStatePath); } catch { }
        try
        {
            if (File.Exists(llmStatePath))
                foreach (string ln in File.ReadAllLines(llmStatePath))
                {
                    string one = ln.Trim();
                    if (one.Length == 0) continue;
                    // 旧格式 relPath|mtimeTicks：只记路径，避免升级后整库重喂；不写回新水位
                    int bar = one.IndexOf('|');
                    if (bar > 0 && one.Substring(bar + 1).Length > 0 && one.Substring(bar + 1).All(char.IsDigit))
                    {
                        legacyDone.Add(one.Substring(0, bar));
                        continue;
                    }
                    done.Add(one);
                }
        }
        catch { }

        // ★与 ① 协调：只有压缩"最近健康"时才把待压缩文件留给它，
        //   否则压缩一直失败（侧车没就绪/模型报错）会把文件饿死、永远不产事实卡。
        bool compactHealthy = Configuration.AutoCompactEnable
            && (DateTime.Now - lastCompactOk).TotalMinutes < Math.Max(30, Configuration.AutoCompactMinutes) * 2;
        var pendingCompact = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (compactHealthy)
        {
            try
            {
                var scanArgs = new Dictionary<string, object?>
                {
                    ["memdir"] = memoryRoot,
                    ["min_entries"] = Configuration.AutoCompactMinEntries,
                    ["min_age_days"] = Configuration.AutoCompactMinAgeDays,
                    ["max_chars"] = 0,   // probe：只问"哪些文件待压缩"，不抓正文
                    ["top_k"] = 500,
                    ["exclude"] = ExcludeSnapshot(),
                };
                JsonElement? scan = await sidecar.AskJsonAsync("compactscan", scanArgs, 15000, indexPath).ConfigureAwait(false);
                if (scan != null && scan.Value.TryGetProperty("candidates", out JsonElement cands2) && cands2.ValueKind == JsonValueKind.Array)
                    foreach (JsonElement ce in cands2.EnumerateArray())
                    {
                        string s2 = ce.TryGetProperty("source", out JsonElement sEl) ? (sEl.GetString() ?? "") : "";
                        if (s2.Length > 0) pendingCompact.Add(s2);
                    }
                if (pendingCompact.Count > 0)
                    logger.LogInformation("记忆书签：整理跳过 {n} 个待压缩文件（留给压缩）", pendingCompact.Count);
            }
            catch (Exception e) { logger.LogWarning(e, "记忆书签：查询待压缩清单失败（本轮不跳过）"); }
        }

        // 找出没整理过的记忆文件（跳过事实卡本身、索引、状态文件）
        // ★关键顺序：先收集全部候选 → 按修改时间从新到旧排序 → 再按预算选材。
        //   旧实现边枚举边扣预算、最后才排序，枚举顺序（目录/文件名）会让老档先吃光预算，新档永远轮不到。
        var fresh = new List<(string path, string key, string text, DateTime mt)>();
        int budget = Math.Max(2000, maxChars > 0 ? maxChars : Configuration.LlmOrganizeMaxChars);
        try
        {
            var cands = new List<(string path, string baseKey, string rel, DateTime mt, string text, int offset)>();
            foreach (var item in EnumerateAllMemoryFiles())
            {
                string f = item.file;
                string name = Path.GetFileName(f);
                if (name.StartsWith("事实_") || name.StartsWith("审计报告_")
                    || name == "index.json" || name == "llm_organize_state.txt"
                    || name == "compacted_sources.txt" || name.EndsWith("记忆索引视图.html")) continue;
                string relPath = item.source;
                if (compactHealthy && pendingCompact.Contains(relPath)) continue;   // 留给压缩
                DateTime mt;
                try { mt = File.GetLastWriteTime(f); } catch { continue; }
                string txt;
                try { txt = File.ReadAllText(f); } catch { continue; }
                // ★水位 = relPath|内容哈希|已整理到第几个字符。
                //   内容变了哈希就变，自然从头来；内容没变只推进 offset，绝不重复喂同一段。
                //   这样既能"每批只喂很小一段"（省上下文），也不会像截断那样丢掉文件尾巴。
                string baseKey = relPath + "|" + ContentKey(txt);
                if (done.Contains(baseKey)) continue;                 // 旧格式：整份已处理
                int offset = 0;
                foreach (string k in done)
                {
                    if (k.StartsWith(baseKey + "|", StringComparison.OrdinalIgnoreCase) == false) continue;
                    if (int.TryParse(k.Substring(baseKey.Length + 1), out int off2) && off2 > offset) offset = off2;
                }
                if (legacyDone.Contains(relPath))
                {
                    // 旧水位：只有文件在水位写盘之后没再改过，才算"已处理"；
                    // 改过就按新内容整理，避免升级后漏掉最后一次改动。
                    if (mt <= stateTime.AddSeconds(5)) { done.Add(baseKey + "|" + txt.Length); continue; }
                }
                if (string.IsNullOrWhiteSpace(txt)) { done.Add(baseKey + "|" + txt.Length); continue; }
                if (offset >= txt.Length) continue;
                cands.Add((f, baseKey, relPath, mt, txt, offset));
            }
            // 新→旧：新流水优先
            cands.Sort((x, y) => y.mt.CompareTo(x.mt));
            foreach (var c in cands)
            {
                if (budget <= 0) break;
                int take = Math.Min(c.text.Length - c.offset, budget);
                if (take <= 0) continue;
                string seg = c.text.Substring(c.offset, take);
                string key = c.baseKey + "|" + (c.offset + take);
                if (done.Contains(key)) continue;
                int fileCap = maxFiles > 0 ? Math.Clamp(maxFiles, 1, 200) : Math.Max(1, Configuration.LlmOrganizeMaxFilesPerRun);
                if (fresh.Count >= fileCap) break;
                fresh.Add((c.path, key, seg, c.mt));
                budget -= take;
            }
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：模型整理扫描失败"); return (0, ""); }

        if (fresh.Count == 0) return (0, "");

        string body = string.Join("\n\n", fresh.Select(x => "【" + Path.GetFileName(x.path) + "】\n" + x.text));
        string prompt =
            "[消息来源(记忆整理)]下面是刚落盘的新记忆流水。请把它提炼成若干条「事实卡」，每条一行，格式：\n" +
            "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
            "要求：只记事实，别抒情、别复述细节；只提炼有长期价值的信息（决定、进度、健康状况、约定、重要结论）；" +
            "没价值的不记。只输出事实行：不要调用任何工具（Python/Expression/Speak 等），不要输出代码、XML 或解释，不要寒暄。\n\n" + body;

        // ★隔离调用：不写角色 ChatHistory（旧写法会把 1 万字提示词永久留在上下文里）
        string cards = await AskModelIsolatedAsync(prompt).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(cards))
        {
            // 模型认为没价值：仍记水位，免得反复重试同一个文件
            foreach (var x in fresh) done.Add(x.key);
            TrySaveState(done);
            return (fresh.Count, "");
        }

        // ★只留事实行：模型回包常夹带 XML/代码（真央那份事实卡就被写进了 <Python>…）
        string cleanCards = SanitizeFactCards(cards, out int droppedCards);
        if (droppedCards > 0)
            logger.LogWarning("记忆书签：整理回包丢弃 {n} 行非事实内容", droppedCards);
        if (string.IsNullOrWhiteSpace(cleanCards))
        {
            // 全是垃圾：不算失败，推进水位，别让同一批反复重喂
            foreach (var x in fresh) done.Add(x.key);
            TrySaveState(done);
            return (fresh.Count, "");
        }
        // ③ 写入合并：不再直接 AppendAllText —— 先与已有事实合并（重复丢弃、相近合并）
        string factFile = await AppendFactsAsync(cleanCards).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(factFile))
        {
            logger.LogWarning("记忆书签：模型整理写盘失败（水位不推进，稍后重试）");
            return (0, "");
        }
        logger.LogInformation("记忆书签：模型整理落盘 {n} 个文件 -> {f}（已合并去重）", fresh.Count, factFile);

        foreach (var x in fresh) done.Add(x.key);
        TrySaveState(done);
        return (fresh.Count, factFile);
    }

    /// <summary>内容指纹（文件内容变了才变），水位用它，避免 mtime 抖动导致反复重喂。</summary>
    static string ContentKey(string text)
    {
        try
        {
            byte[] h = System.Security.Cryptography.SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(text ?? ""));
            return Convert.ToHexString(h).Substring(0, 12);
        }
        catch { return (text ?? "").Length.ToString(); }
    }

    void TrySaveState(HashSet<string> done)
    {
        try
        {
            // ★不要按字典序截断：旧实现只留 500 条且按路径排序，超过后会把"路径靠前"的文件
            //   水位丢掉 → 它们每轮都被当成新文件重新整理（反复重喂）。
            var all = done.ToList();
            if (all.Count > 20000) all = all.Take(20000).ToList();   // 只是防病态膨胀
            File.WriteAllLines(llmStatePath, all);
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：模型整理水位写盘失败"); }
    }

    // ===== ① 自动压缩 / ② 审计 / ③ 写入合并 / ⑥ 内嵌浏览 =====

    /// <summary>载入"已压缩来源"清单（30 秒节流）。只是不建索引，官方文件原样保留。</summary>
    void EnsureExcludes()
    {
        try
        {
            if ((DateTime.Now - lastExcludesLoad).TotalSeconds < 30) return;
            lastExcludesLoad = DateTime.Now;
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (File.Exists(compactStatePath))
                foreach (string ln in File.ReadAllLines(compactStatePath))
                    if (string.IsNullOrWhiteSpace(ln) == false) set.Add(ln.Trim());
            lock (excludesLock) { excludes.Clear(); foreach (string x in set) excludes.Add(x); }
        }
        catch { }
    }

    /// <summary>排除清单 = 已压缩来源 + 配置里的手动排除（支持 L1/ 这样的目录前缀）。</summary>
    string[] ExcludeSnapshot()
    {
        var list = new List<string>();
        lock (excludesLock) list.AddRange(excludes);
        try
        {
            string raw = Configuration?.ExcludeSources ?? "";
            foreach (string part in raw.Split(new[] { '\n', '\r', ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                string one = part.Trim().Replace('\\', '/').TrimStart('/');
                if (one.Length > 0) list.Add(one);
            }
        }
        catch { }
        return list.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    /// <summary>按精确路径或目录前缀判断是否排除。</summary>
    static bool IsExcludedRel(string rel, string[] list)
    {
        if (string.IsNullOrEmpty(rel)) return false;
        string r = rel.Replace('\\', '/').Trim('/');
        if (r.Length == 0) return false;
        string hay = "/" + r + "/";
        foreach (string raw in list)
        {
            if (string.IsNullOrEmpty(raw)) continue;
            string one = raw.Replace('\\', '/').Trim('/');
            if (one.Length == 0) continue;
            // 目录前缀在任意层级生效：填 L1/ 也能排掉 facts/L1/、Memory/L1/
            if (raw.Replace('\\', '/').EndsWith("/"))
            {
                if (hay.Contains("/" + one + "/", StringComparison.OrdinalIgnoreCase)) return true;
            }
            else if (hay.Contains("/" + one + "/", StringComparison.OrdinalIgnoreCase)) return true;
        }
        return false;
    }

    /// <summary>把某个来源记为"已压缩"：只影响我们的索引，绝不移动/删除官方文件。</summary>
    void MarkCompacted(string source)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(source)) return;
            lock (excludesLock)
            {
                if (excludes.Add(source) == false) return;
            }
            Directory.CreateDirectory(factsRoot);
            File.AppendAllText(compactStatePath, source + "\n");
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：记录压缩标记失败"); }
    }

    /// <summary>工具结果回显：Poke 会挂进聊天框（人工可见）；模型同轮已拿到返回值，故可关。</summary>
    void Echo(string text)
    {
        if (Configuration == null || Configuration.EchoToolResult == false) return;
        try
        {
            string s = (text ?? "").Trim();
            if (s.Length == 0) return;
            if (s.Length > 400) s = s.Substring(0, 400) + "…";
            interactor.Poke(s);
        }
        catch { }
    }

    /// <summary>回显 + 返回：工具结果既给模型（返回值）也给聊天框（Poke，可关）。</summary>
    string R(string text)
    {
        Echo(text ?? "");
        return text ?? "";
    }

    /// <summary>③ 写事实卡：先与已有事实合并（重复丢弃 / 相近合并 / 其余追加），避免事实卡膨胀。</summary>
    /// <summary>
    /// 隔离模型调用：用独立的 ChatHistoryAgentThread，**不写角色 ChatHistory**。
    /// 原来的 chatBot.ChatAsync 会把上万字的整理/压缩提示词永久塞进上下文，
    /// 把 128k 窗口顶爆，进而触发官方压缩、UI 渲染与改历史打架。
    /// </summary>
    async Task<string> AskModelIsolatedAsync(string prompt)
    {
        try
        {
            ChatHistoryAgentThread thread = new();
            thread.ChatHistory.AddUserMessage(prompt ?? "");
            string? resp = await chatBot.LanguageModel.ChatStreamingAsync(thread).ConfigureAwait(false);
            return (resp ?? "").Trim();
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆书签：隔离模型调用失败");
            return "";
        }
    }

    /// <summary>事实行格式：日期 | 谁 | 事 | 事实（至少 4 段）。</summary>
    static bool IsFactLine(string body)
    {
        if (string.IsNullOrWhiteSpace(body)) return false;
        string s = body.TrimStart('-', '*', '・', ' ', '\t').Trim();
        // 兼容：2026-10-20 / 10/20 / 2026年10月20日（小鲸鱼的事实卡用短日期）
        if (Regex.IsMatch(s, @"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)\s*\|") == false) return false;
        return s.Count(ch => ch == '|') >= 3;
    }

    /// <summary>
    /// 只保留"事实行"：模型的原始回包可能夹带工具 XML、Python 代码、解释文本，
    /// 这些绝不能当事实写进事实卡（真央那份就被写进了 &lt;Python&gt; / &lt;Expression/&gt;）。
    /// </summary>
    static string SanitizeFactCards(string raw, out int dropped)
    {
        dropped = 0;
        if (string.IsNullOrWhiteSpace(raw)) return "";
        string text = raw;
        text = Regex.Replace(text, @"\x60\x60\x60[\s\S]*?\x60\x60\x60", "\n");   // 成对代码围栏
        text = Regex.Replace(text, @"\x60\x60\x60[\s\S]*$", "\n");               // 未闭合围栏
        text = Regex.Replace(text, @"<(Python|Speak|Expression|Motion|Poke|Function|Tool)[^>]*>[\s\S]*?</\1\s*>", "\n", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"<[^>\n]{1,100}/>", "\n");                   // 自闭合标签
        var kept = new List<string>();
        foreach (string lnRaw in text.Split('\n'))
        {
            string ln = lnRaw.Trim();
            if (ln.Length == 0) continue;
            string body = ln.TrimStart('-', '*', '・', ' ', '\t').Trim();
            if (IsFactLine(body))
            {
                if (body.Length > 500) body = body.Substring(0, 500);
                kept.Add("- " + body);
            }
            else dropped++;
        }
        return string.Join("\n", kept);
    }

    /// <summary>清洗已有事实卡：保留 # 标题和合法事实行，清掉历史遗留的代码/XML 垃圾。</summary>
    static string SanitizeExistingFacts(string existing)
    {
        if (string.IsNullOrWhiteSpace(existing)) return "";
        var kept = new List<string>();
        foreach (string lnRaw in existing.Split('\n'))
        {
            string ln = lnRaw.TrimEnd();
            string t = ln.Trim();
            if (t.Length == 0) { kept.Add(""); continue; }
            if (t.StartsWith("#")) { kept.Add(ln); continue; }
            string body = t.TrimStart('-', '*', '・', ' ', '\t').Trim();
            if (IsFactLine(body)) kept.Add(ln);
        }
        return string.Join("\n", kept).Trim();
    }

    async Task<string> AppendFactsAsync(string cards)
    {
        await factLock.WaitAsync().ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(factsRoot);
            string factFile = Path.Combine(factsRoot, "事实_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
            // ★先只留事实行：模型回包里的 XML/代码/解释一律丢掉
            string clean = SanitizeFactCards(cards, out int dropped);
            if (dropped > 0)
                logger.LogWarning("记忆书签：事实卡丢弃 {n} 行非事实内容（XML/代码/自由文本）", dropped);
            if (string.IsNullOrWhiteSpace(clean)) return "";
            string existing = File.Exists(factFile) ? SanitizeExistingFacts(File.ReadAllText(factFile)) : "";
            var lines = clean.Split('\n').Select(x => x.Trim()).Where(x => x.Length > 0).ToList();
            if (lines.Count == 0) return "";
            var args = new Dictionary<string, object?>
            {
                ["lines"] = lines,
                ["existing"] = existing,
                ["threshold"] = Configuration?.MergeThreshold ?? 0.5,
            };
            string? merged = await sidecar.AskFieldAsync("factmerge", args, "text", 8000, indexPath).ConfigureAwait(false);
            string body;
            if (string.IsNullOrWhiteSpace(merged))
                body = (string.IsNullOrEmpty(existing) ? "" : existing.TrimEnd() + "\n") + clean;
            else
                body = merged!;
            if (body.TrimStart().StartsWith("#") == false)
                body = "# 事实卡 · " + DateTime.Now.ToString("yyyy-MM-dd") + "\n\n" + body;
            File.WriteAllText(factFile, body.TrimEnd() + "\n");
            return factFile;
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆书签：事实卡写入失败");
            return "";
        }
        finally { factLock.Release(); }
    }

    /// <summary>① 自动压缩：条目多且够旧的流水交给模型压成事实卡，原文归档（不丢）。</summary>
    async Task<int> AutoCompactAsync(bool force = false)
    {
        if (Configuration == null) return 0;
        if (force == false && Configuration.AutoCompactEnable == false) return 0;
        if (force == false && functionCaller.IsIdle == false) return 0;   // 同上：自动压缩不在回合中插模型调用
        if (sidecar.IsReady == false) return 0;
        if (string.IsNullOrWhiteSpace(memoryRoot) || Directory.Exists(memoryRoot) == false) return 0;
        if (compacting) return 0;
        compacting = true;
        int done = 0;
        try
        {
            var args = new Dictionary<string, object?>
            {
                ["memdir"] = memoryRoot,
                ["min_entries"] = Configuration.AutoCompactMinEntries,
                ["min_age_days"] = Configuration.AutoCompactMinAgeDays,
                ["max_chars"] = Configuration.AutoCompactMaxChars,
                ["top_k"] = 3,
                ["exclude"] = ExcludeSnapshot(),
            };
            JsonElement? root = await sidecar.AskJsonAsync("compactscan", args, 20000, indexPath).ConfigureAwait(false);
            if (root == null) return 0;
            if (root.Value.TryGetProperty("candidates", out JsonElement cands) == false || cands.ValueKind != JsonValueKind.Array)
                return 0;

            bool anyCand = false;
            foreach (JsonElement c in cands.EnumerateArray())
            {
                anyCand = true;
                string source = c.TryGetProperty("source", out JsonElement se) ? (se.GetString() ?? "") : "";
                string text = c.TryGetProperty("text", out JsonElement te) ? (te.GetString() ?? "") : "";
                if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(text)) continue;

                string prompt =
                    "[消息来源(记忆压缩)]下面的流水来自同一份记忆文件。请压成事实卡，每条一行，格式：\n" +
                    "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
                    "要求：只留有长期价值的信息（决定、进度、健康状况、约定、重要结论）；去掉寒暄、重复和过程细节；没价值就不写。" +
                    "只输出事实行：不要调用任何工具（Python/Expression/Speak 等），不要输出代码、XML 或解释。\n\n" + text;

                string cards = await AskModelIsolatedAsync(prompt).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(cards)) { logger.LogInformation("记忆书签：压缩无产出 {src}", source); continue; }
                string cleanCards = SanitizeFactCards(cards, out int droppedCards);
                if (droppedCards > 0)
                    logger.LogWarning("记忆书签：压缩回包丢弃 {n} 行非事实内容 {src}", droppedCards, source);
                if (string.IsNullOrWhiteSpace(cleanCards)) { logger.LogInformation("记忆书签：压缩无有效事实 {src}", source); continue; }

                string factFile = await AppendFactsAsync(cleanCards).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(factFile)) continue;
                // ★B：只记"已压缩"，官方文件一个字节都不动（旧版会 Move 到 archive/，那会踩坏官方插件）
                MarkCompacted(source);
                done++;
                logger.LogInformation("记忆书签：已压缩 {src} -> {f}", source, factFile);
            }
            // 没积压（清单空）或至少压成一份 → 认为压缩链路健康，organize 才敢把文件留给它
            if (anyCand == false || done > 0) lastCompactOk = DateTime.Now;
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：自动压缩失败"); }
        finally { compacting = false; }
        // 压缩会把原文挪进 archive/：立刻重建索引，别让旧条目继续留在索引里
        if (done > 0)
        {
            try { await EnsureIndexAsync(force: true).ConfigureAwait(false); }
            catch (Exception e) { logger.LogWarning(e, "记忆书签：压缩后重建索引失败"); }
        }
        return done;
    }

    /// <summary>② 审计：扫事实卡里的重复/可能矛盾，写报告（只报告，不自动改）。</summary>
    async Task<int> AuditAsync(bool force = false)
    {
        if (Configuration == null) return 0;
        if (force == false && Configuration.AuditEnable == false) return 0;
        if (sidecar.IsReady == false) return 0;
        if (string.IsNullOrWhiteSpace(memoryRoot) || Directory.Exists(memoryRoot) == false) return 0;
        if (auditing) return 0;
        auditing = true;
        try
        {
            var args = new Dictionary<string, object?>
            {
                ["factsdir"] = factsRoot,
                ["limit"] = Configuration.AuditLimit,
                ["dup"] = 0.85,
                ["near"] = 0.5,
            };
            JsonElement? root = await sidecar.AskJsonAsync("auditscan", args, 20000, indexPath).ConfigureAwait(false);
            if (root == null) return 0;
            if (root.Value.TryGetProperty("issues", out JsonElement issues) == false || issues.ValueKind != JsonValueKind.Array)
                return 0;

            var rows = new List<string>();
            foreach (JsonElement it in issues.EnumerateArray())
            {
                string kind = it.TryGetProperty("kind", out JsonElement ke) ? (ke.GetString() ?? "") : "";
                string lineA = it.TryGetProperty("a", out JsonElement ae) ? (ae.GetString() ?? "") : "";
                string lineB = it.TryGetProperty("b", out JsonElement be) ? (be.GetString() ?? "") : "";
                string fa = it.TryGetProperty("file_a", out JsonElement fe) ? (fe.GetString() ?? "") : "";
                string fb = it.TryGetProperty("file_b", out JsonElement fbe) ? (fbe.GetString() ?? "") : "";
                double score = it.TryGetProperty("score", out JsonElement sce) ? sce.GetDouble() : 0;
                rows.Add("- [" + kind + " " + score.ToString("0.00") + "] " + lineA + "（" + fa + "） / " + lineB + "（" + fb + "）");
            }
            if (rows.Count == 0) return 0;

            // ★报告写在 factsRoot：官方 Memory/ 只读，一个字节都不写
            Directory.CreateDirectory(factsRoot);
            string report = Path.Combine(factsRoot, "审计报告_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
            var known = new HashSet<string>(StringComparer.Ordinal);
            if (File.Exists(report))
                foreach (string ln in File.ReadAllLines(report))
                    if (ln.Trim().Length > 0) known.Add(ln.Trim());
            var fresh = rows.Where(r => known.Contains(r) == false).ToList();
            if (fresh.Count == 0) return 0;

            var sb = new StringBuilder();
            if (File.Exists(report) == false)
                sb.Append("# 记忆审计报告 · ").Append(DateTime.Now.ToString("yyyy-MM-dd")).Append("\n\n");
            foreach (string r in fresh) sb.Append(r).Append('\n');
            File.AppendAllText(report, sb.ToString());
            logger.LogInformation("记忆书签：审计新增 {n} 条 -> {f}", fresh.Count, report);
            return fresh.Count;
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：审计失败"); return 0; }
        finally { auditing = false; }
    }

    /// <summary>⑥ 内嵌浏览：管理页查条目（关键词/人/事/物/时间 + 分页）。</summary>
    public async Task<string> BrowseAsync(string query, string who, string topic, string thing, string dateFrom, string dateTo, int offset, int limit)
    {
        try
        {
            var args = new Dictionary<string, object?>
            {
                ["query"] = query ?? "",
                ["who"] = who ?? "",
                ["topic"] = topic ?? "",
                ["thing"] = thing ?? "",
                ["date_from"] = dateFrom ?? "",
                ["date_to"] = dateTo ?? "",
                ["offset"] = offset,
                ["limit"] = limit,
            };
            JsonElement? root = await sidecar.AskJsonAsync("browse", args, 8000, indexPath).ConfigureAwait(false);
            if (root == null) return "浏览失败（侧车没就绪）。";
            string txt = root.Value.TryGetProperty("text", out JsonElement te) ? (te.GetString() ?? "") : "";
            int total = (root.Value.TryGetProperty("total", out JsonElement tt) && tt.ValueKind == JsonValueKind.Number) ? tt.GetInt32() : 0;
            if (string.IsNullOrWhiteSpace(txt)) return "没有匹配的条目（共 " + total + " 条）。";
            return "共 " + total + " 条，从第 " + (offset + 1) + " 条起：\n\n" + txt;
        }
        catch (Exception e) { return "浏览失败：" + e.Message; }
    }

    /// <summary>⑥ 管理页状态行：索引体量 + 侧车状态。</summary>
    public async Task<string> StatsAsync()
    {
        try
        {
            string? txt = await sidecar.AskAsync("stats", new Dictionary<string, object?>(), 4000, indexPath).ConfigureAwait(false);
            string dbPathS = Path.Combine(factsRoot, "memory.db");
            string store = File.Exists(dbPathS) ? dbPathS : indexPath;
            string size = File.Exists(store) ? Math.Round(new FileInfo(store).Length / 1024.0 / 1024.0, 1) + " MB" : "无索引";
            string upd = File.Exists(store) ? File.GetLastWriteTime(store).ToString("MM-dd HH:mm") : "-";
            return (string.IsNullOrWhiteSpace(txt) ? "索引读不出来" : txt!) + "｜" + size + "｜更新 " + upd + "｜侧车 " + sidecar.Status;
        }
        catch (Exception e) { return "状态读取失败：" + e.Message; }
    }

    async Task ComputeVectorsAsync()
    {
        if (computing) return;
        computing = true;
        sidecar.Busy = true;
        try
        {
            var list = chunks.ToList();   // ★快照：算向量期间 EnsureChunks 可能重建
            if (list.Count == 0) return;
            if (sidecar.IsReady == false) return;
            for (int i = 0; i < list.Count; i += 32)
            {
                var batch = list.Skip(i).Take(32).Where(c => c.Vector == null).ToList();
                if (batch.Count == 0) continue;
                float[][]? vecs = await sidecar.EmbedAsync(batch.Select(c => c.Text).ToArray()).ConfigureAwait(false);
                if (vecs == null) return;
                for (int j = 0; j < batch.Count && j < vecs.Length; j++)
                    batch[j].Vector = vecs[j];
            }
            vectorsReady = list.All(c => c.Vector != null);
            logger.LogInformation("记忆回忆提示：向量就绪 {Ok}（{N} 段）", vectorsReady, list.Count);
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆回忆提示：向量计算失败");
        }
        finally
        {
            sidecar.Busy = false;
            computing = false;
        }
    }

    List<Chunk> QueryByVector(float[] q)
    {
        var ready = chunks.ToList().Where(c => c.Vector != null).ToList();   // 快照，防并发 Clear
        if (ready.Count == 0) return [];

        var scored = ready.Select(c => (c, s: Cosine(q, c.Vector!))).ToList();
        double mean = scored.Average(x => x.s);

        var hits = scored
            .Where(x => x.s >= Configuration.MinScore && (x.s - mean) >= Configuration.MinMargin)
            .OrderByDescending(x => x.s)
            .Take(Configuration.MaxResults)
            .Select(x => x.c)
            .ToList();
        logger.LogInformation("记忆回忆提示：语义命中 {N}（均值 {M:F3}）", hits.Count, mean);
        return hits;
    }

    static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        if (na < 1e-9 || nb < 1e-9) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    // ===== 关键词兜底（纯内存） =====

    List<Chunk> QueryKeyword(string query)
    {
        var qgrams = Grams(query);
        if (qgrams.Count == 0) return [];

        var scored = new List<(Chunk c, double s)>();
        var snapshot = chunks.ToList();   // ★后台扫描会 chunks.Clear()，边遍历边改会炸/漏
        foreach (Chunk c in snapshot)
        {
            var grams = Grams(c.Text);
            if (grams.Count == 0) continue;
            int hit = qgrams.Count(g => grams.Contains(g));
            double score = (double)hit / qgrams.Count;
            if (score > 0) scored.Add((c, score));
        }

        return scored
            .Where(x => x.s >= Configuration.MinScoreKeyword)
            .OrderByDescending(x => x.s)
            .Take(Configuration.MaxResults)
            .Select(x => x.c)
            .ToList();
    }

    static HashSet<string> Grams(string text)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        string s = new string(text.Where(ch => char.IsLetterOrDigit(ch)).ToArray());
        for (int i = 0; i < s.Length; i++)
        {
            if (i + 1 < s.Length) set.Add(s.Substring(i, 2));
            if (i + 2 < s.Length) set.Add(s.Substring(i, 3));
        }
        return set;
    }
}
