
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ComponentModel;
using System.Diagnostics;
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
/// Mao.MemoryInject —— Retromem 溯忆模块（后台检索·命中才提示）。
///
/// 铁律（2026-10-05 修）：
///   1. 同步钩子 ChatSend 里**绝不无上限等待外部进程**。语义检索只允许
///      <see cref="Configuration"/> 里 SyncWaitMs 毫秒的有上限等待（侧车常驻后单次约 10ms）；
///      超时立刻退回纯内存关键词，绝不把 UI 线程拖死。
///   2. 连续超时达到阈值就进冷静期，暂停语义检索一段时间，避免每句都卡。
///   3. 侧车启动/批量向量化一律后台；批量期间 Busy 置位，同步路径直接放过。
///   4. 任何异常一律放过原消息，并记日志。
/// </summary>
[Module("Retromem 溯忆",
    "以官方记忆为只读整理来源，独立管理事实卡，提供记忆提炼、多路召回、本地精排、关系图谱、人物画像与安全审计的可视化记忆工作台。",
    defaultCategory: "Mao",
    editorUI: typeof(MemoryInjectUI))]
public class MemoryInjectModule(
    ILogger<MemoryInjectModule> logger,
    XmlFunctionCaller functionCaller,
    Interactor<MemoryInjectModule> interactor,
    ChatBot chatBot,
    ConfigurationSystem configurationSystem) :
    ChatBehaviour,
    IConfigurable<MemoryInjectConfig>
{
    public MemoryInjectConfig Configuration { get; set; } = null!;

    string memoryRoot = "";          // 主记忆目录（只读，兼容旧代码）
    readonly List<(string root, string prefix)> memoryRoots = new();
    string roleRoot = "";            // 角色根目录
    string factsRoot = "";           // 我们自己的事实层（写这里，绝不写官方目录）
    string compactStatePath = "";    // 已压缩来源清单（只影响我们的索引，不动官方文件）
    string pendingProposalPath = ""; // 已提交、待人工接受的压缩来源
    readonly HashSet<string> excludes = new(StringComparer.OrdinalIgnoreCase);
    // 自动压缩已提交但尚未人工接受的来源。它们仍保留在官方索引中，
    // 这里只是避免每轮重复提交相同提案；接受后才转入 excludes。
    readonly HashSet<string> pendingProposalSources = new(StringComparer.OrdinalIgnoreCase);
    readonly object excludesLock = new();
    DateTime lastExcludesLoad = DateTime.MinValue;
    string scriptPath = "";
    string indexPath = "";
    System.Diagnostics.Process? graphServer;
    string graphIndexArg = "";
    const int GraphPort = 18763;
    DateTime lastBuildCheck = DateTime.MinValue;
    readonly SemaphoreSlim rebuildLock = new(1, 1);
    readonly SemaphoreSlim graphLifecycleLock = new(1, 1);
    volatile bool warmDone = false;   // 索引/模型预热是否完成（未完成时首次查询给长窗口）
    DateTime lastScanStamp = DateTime.MinValue;
    volatile bool hasMemory;   // 本角色有没有记忆文件：没有就别启侧车/别检索，免得串到别人的索引
    DateTime lastMemCheck = DateTime.MinValue;   // 上次重判 hasMemory 的时间（目录可能后建/后补文件）
    DateTime lastStampCalc = DateTime.MinValue;  // 上次计算"最新文件时间"的时间（节流用）

    readonly List<Chunk> chunks = [];
    readonly SemaphoreSlim scanLock = new(1, 1);
    readonly EmbedSidecar sidecar = new();
    readonly object callStatsGapLock = new();

    volatile bool vectorsReady;
    volatile bool computing;
    int timeoutStreak;
    DateTime cooldownUntil = DateTime.MinValue;

    // 双水位线：攒够有效字数才查一次，查完回落；两次之间有最小间隔
    int waterLevel;
    DateTime lastTrigger = DateTime.MinValue;

    string CallStatsGapPath => Path.Combine(factsRoot, "call_stats_gap.txt");

    string? ReadCallStatsGap()
    {
        lock (callStatsGapLock)
        {
            try { return File.Exists(CallStatsGapPath) ? File.ReadAllText(CallStatsGapPath).Trim() : null; }
            catch { return null; }
        }
    }

    void MarkCallStatsGap()
    {
        lock (callStatsGapLock)
        {
            try
            {
                Directory.CreateDirectory(factsRoot);
                if (!File.Exists(CallStatsGapPath))
                    File.WriteAllText(CallStatsGapPath, DateTime.UtcNow.ToString("O"));
            }
            catch (Exception e) { logger.LogWarning(e, "无法保存调用统计缺口标记"); }
        }
    }

    void ClearCallStatsGap(string? expected)
    {
        if (string.IsNullOrWhiteSpace(expected)) return;
        lock (callStatsGapLock)
        {
            try
            {
                if (File.Exists(CallStatsGapPath)
                    && string.Equals(File.ReadAllText(CallStatsGapPath).Trim(), expected, StringComparison.Ordinal))
                    File.Delete(CallStatsGapPath);
            }
            catch (Exception e) { logger.LogWarning(e, "无法清除已持久化的调用统计缺口标记"); }
        }
    }

    bool RecordCallsSyncWithGap(string callId, string channel, JsonElement facts, int windowGen = 0)
    {
        string? gap = ReadCallStatsGap();
        bool ok = sidecar.RecordCallsSync(callId, channel, facts, indexPath, gapStart: gap, windowGen: windowGen);
        if (ok) ClearCallStatsGap(gap); else MarkCallStatsGap();
        return ok;
    }

    async Task<bool> RecordCallsAsyncWithGap(string callId, string channel, JsonElement facts)
    {
        string? gap = ReadCallStatsGap();
        bool ok = await sidecar.RecordCallsAsync(callId, channel, facts, 1200, indexPath, gap).ConfigureAwait(false);
        if (ok) ClearCallStatsGap(gap); else MarkCallStatsGap();
        return ok;
    }

    public string SidecarStatus => sidecar.Status;
    public string GraphPageUrl => $"http://127.0.0.1:{GraphPort}/";

    sealed class Chunk
    {
        public string Source = "";
        public string Text = "";
        public float[]? Vector;
    }

    // ===== 生命周期 =====

    DateTime lastMaintain = DateTime.MinValue;
    DateTime lastVecWarm = DateTime.MinValue;        // 上次触发语义缓存预热（分批，只补缺口）
    DateTime lastLlmOrganize = DateTime.MinValue;   // 上次成功自动整理；跳过或失败不消耗成功间隔
    DateTime startedAt = DateTime.Now;
    bool llmStartupLogged;
    DateTime lastCompactOk = DateTime.MinValue;     // 上次压缩"健康"完成（扫到清单且没积压/有产出）
    DateTime lastCompact = DateTime.MinValue;        // 上次自动压缩（流水 -> 事实卡）
    DateTime lastAudit = DateTime.MinValue;          // 上次事实审计
    volatile bool compacting;                        // 压缩/审计互斥，避免并发重复干活
    volatile bool auditing;
    volatile bool organizing;                      // 手动/自动整理互斥，避免同一批重复喂模型
    readonly SemaphoreSlim organizeLock = new(1, 1);
    readonly SemaphoreSlim factLock = new(1, 1);      // 事实卡写入串行化（整理与压缩会并发）
    string llmStatePath = "";                        // 记录"已整理到哪个文件/时间"的水位
    readonly object sourceProgressLock = new();
    string sourceProgressCache = "", sourceProgressScope = "";
    DateTime sourceProgressCheckedAt = DateTime.MinValue;
    Dictionary<string, object?> sourceProgressPayload = new();
    Timer? sourceProgressTimer;
    int sourceProgressRefreshing;

    MemoryInjectConfig? migratedConfiguration;
    bool migrationSavePending;
    DateTime lastMigrationSaveAt = DateTime.MinValue;

    /// <summary>配置可能晚于 OnAwake 注入：首次拿到 Configuration 时再跑迁移（幂等）。</summary>
    void EnsureMigrations()
    {
        if (Configuration == null || ReferenceEquals(migratedConfiguration, Configuration)) return;
        int before = Configuration.DefaultsMigrationVersion;
        ApplyDefaultMigrations();
        migrationSavePending |= Configuration.DefaultsMigrationVersion != before;
        if (migrationSavePending)
        {
            string root = Character?.StorageKey ?? "";
            if (root.Length == 0)
            {
                // 拿不到角色键时不写全局配置，免得把一个角色的配置写到所有人身上
                logger.LogDebug("Retromem 溯忆：角色键还没就绪，升级配置稍后再存");
                return;
            }
            // 保存失败（磁盘/权限/路径）不要每条消息都重试一遍：10 分钟一次，日志也只报一次
            if ((DateTime.UtcNow - lastMigrationSaveAt).TotalMinutes < 10) return;
            lastMigrationSaveAt = DateTime.UtcNow;
            try
            {
                // ★第三参是"存储根目录下的相对 root"，不是绝对路径：传 roleRoot 会拼成
                //   D:\alife\storage\D:\alife\storage\Character\xxx\Configuration 而失败。
                //   框架自己的角色键就是 Character.StorageKey（路由插件保存配置也用它）。
                configurationSystem.SetConfiguration(typeof(MemoryInjectModule), Configuration, root);
                migrationSavePending = false;
            }
            catch (Exception e)
            {
                logger.LogWarning(e, "Retromem 溯忆：升级配置保存失败，10 分钟后重试");
                return;
            }
        }
        migratedConfiguration = Configuration;
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
            logger.LogInformation("Retromem 溯忆：已应用默认规则迁移 v1（原排除规则为空时加入 L1/；用户自定义规则保留）");
        }
        if (ver < 2)
        {
            // v2：仅推进迁移标记，保留当前语义选择；新配置按默认开启，不再强制关闭。
            Configuration.DefaultsMigrationVersion = 2;
            logger.LogInformation("Retromem 溯忆：已应用默认规则迁移 v2（保留当前语义重排设置）");
        }
        if (ver < 3)
        {
            // v3：官方记忆默认只读 L1/L100。旧默认 ExcludeSources="L1/" 会把白名单里的 L1
            // 反过来挡住，所以只移除这一个精确 token；用户写的 L1/某文件 等自定义排除保留。
            string raw = Configuration.ExcludeSources ?? "";
            var kept = new List<string>();
            foreach (string part in raw.Split(new[] { '\n', '\r', ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                string one = part.Trim().Replace('\\', '/').Trim('/');
                if (string.Equals(one, "L1", StringComparison.OrdinalIgnoreCase)) continue;
                if (one.Length > 0) kept.Add(part.Trim());
            }
            Configuration.ExcludeSources = string.Join("\n", kept);
            if (string.IsNullOrWhiteSpace(Configuration.SourceLayers))
                Configuration.SourceLayers = "L1,L100";
            Configuration.DefaultsMigrationVersion = 3;
            logger.LogInformation("Retromem 溯忆：已应用默认规则迁移 v3（官方来源默认只扫描 L1/L100；旧默认排除 L1/ 已移除）");
        }
        if (ver < 4)
        {
            // This release intentionally enables previously-off features once.
            Configuration.UseSemantic = true;
            Configuration.AllowSoftMaintenance = true;
            Configuration.EchoToolResult = true;
            Configuration.DefaultsMigrationVersion = 4;
            logger.LogInformation("Retromem：本次升级已开启语义重排、维护软删、工具回显；自动压缩保留原值，之后可自行关闭。");
        }
        if (ver < 5)
        {
            // 本次升级覆盖 v4 之后保存的关闭值；迁移标记落盘后尊重后续手动选择。
            Configuration.UseSemantic = true;
            Configuration.DefaultsMigrationVersion = 5;
            logger.LogInformation("Retromem 溯忆：本次升级已开启语义检索；升级配置将保存，之后可自行关闭。");
        }
    }

    /// <summary>官方记忆允许扫描的层级（空 = 不过滤）。只约束官方来源，插件自己的 MemoryFacts 不受限。</summary>
    string[] LayerSnapshot()
    {
        string raw = Configuration?.SourceLayers ?? "";
        var list = new List<string>();
        foreach (string part in raw.Split(new[] { '\n', '\r', ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string one = part.Trim();
            if (one.Length > 0) list.Add(one);
        }
        return list.Count == 0 ? Array.Empty<string>() : list.ToArray();
    }

    /// <summary>是不是 L1 / L100 这种层级目录名。按整段判断，绝不把 L1 当成 L100 的前缀。</summary>
    static bool IsLayerSegment(string segment)
    {
        if (string.IsNullOrEmpty(segment) || segment.Length < 2) return false;
        if (segment[0] != 'L' && segment[0] != 'l') return false;
        for (int i = 1; i < segment.Length; i++)
            if (!char.IsDigit(segment[i])) return false;
        return true;
    }

    /// <summary>该相对路径是否落在允许的官方层级里。层级目录可能出现在第一段（L5/x），
    /// 也可能在 Memory/L5/x 里，所以逐段找；不在层级目录下的文件不受约束。</summary>
    static bool InLayerRange(string rel, string[] layers)
    {
        if (layers.Length == 0) return true;
        string clean = (rel ?? "").Replace('\\', '/').Trim('/');
        if (clean.Length == 0) return true;
        string[] parts = clean.Split('/');
        for (int i = 0; i < parts.Length - 1; i++)
        {
            if (!IsLayerSegment(parts[i])) continue;
            foreach (string name in layers)
                if (string.Equals(name.Trim(), parts[i], StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }
        return true;
    }

    /// <summary>来源范围的签名：层级白名单 + 排除清单。范围一变就要重建一次，
    /// 否则升级后旧库还按旧范围（只读 L2–L6）继续用，配置等于没生效。
    /// scope=facts-v1：索引口径改为"只认事实卡"，旧库（含官方流水）必须重建一次。</summary>
    string ScopeSignature()
    {
        EnsureExcludes();   // 排除清单含"已压缩来源"，没加载就会和写签名时的值不一致 → 反复重建
        var layers = LayerSnapshot().OrderBy(x => x, StringComparer.OrdinalIgnoreCase);
        var exclude = ExcludeSnapshot().OrderBy(x => x, StringComparer.OrdinalIgnoreCase);
        return "scope=facts-v1|layers=" + string.Join(",", layers) + "|exclude=" + string.Join(",", exclude);
    }

    string ScopeStampPath()
    {
        return string.IsNullOrWhiteSpace(factsRoot) ? "" : Path.Combine(factsRoot, "scope_signature.txt");
    }

    /// <summary>已有库、但范围签名变了（升级或用户改了层级/排除）：需要重建一次。</summary>
    bool ScopeChanged()
    {
        try
        {
            string stamp = ScopeStampPath();
            if (stamp.Length == 0) return false;
            bool hasStore = File.Exists(indexPath) || File.Exists(Path.Combine(factsRoot, "memory.db"));
            if (!hasStore) return false;          // 还没建过库：走正常首次建，不算"范围变了"
            if (!File.Exists(stamp)) return true; // 5.1 之前的库没有签名
            return File.ReadAllText(stamp).Trim() != ScopeSignature();
        }
        catch { return false; }
    }

    void WriteScopeStamp()
    {
        try
        {
            string stamp = ScopeStampPath();
            if (stamp.Length > 0) File.WriteAllText(stamp, ScopeSignature());
        }
        catch { }
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
                if (rel.Length > 0) logger.LogWarning("Retromem 溯忆：忽略角色目录外的记忆来源配置：{Path}", part.Trim());
                continue;
            }
            string candidate = Path.GetFullPath(Path.Combine(roleRoot, rel));
            if (candidate.StartsWith(Path.GetFullPath(roleRoot).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                candidates.Add(candidate);
            else
                logger.LogWarning("Retromem 溯忆：忽略越过角色根目录的来源配置：{Path}", part.Trim());
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
                pendingProposalPath = Path.Combine(factsRoot, "pending_proposal_sources.txt");
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
            logger.LogError(e, "Retromem 溯忆：路径初始化失败");
        }

        // 清掉历史换库留下的 swap / 旧备份（每份几十~几百 MB，不清会把盘吃满）
        try { PruneBackups(); } catch { }
        // 一次性迁移：清掉事实卡里的 XML/代码垃圾（自动，带备份；历史污染走手动 memcleanhistory）
        try { MigrateFactCards(); } catch { }

        // 本角色有没有记忆文件：没有就别启侧车、别检索（否则会串到别人的索引）
        try { hasMemory = memoryRoots.Any(x => HasMemoryFiles(x.root)) || HasMemoryFiles(factsRoot); } catch { hasMemory = false; }

        try { ChatBot.ChatSend += OnChatSend; }
        catch (Exception e) { logger.LogError(e, "Retromem 溯忆：挂载 ChatSend 失败"); }

        // 主动调用的工具（取原文/画像卡/钓欠账/按人事物检索事实卡），隐式注入
        try
        {
            XmlHandler handler = new(this)
            {
                Description = "Retromem 溯忆记忆工具：翻自己落盘的旧账。想确认往事、查来龙去脉、按时间找、盘点欠账时，**只能**用这里的工具，" +
                              "禁止用 <Python>、读文件、扫目录、sqlite3 去查记忆（既慢又容易读到过期数据）。" +
                              "memrecall 按人事检索事实卡；memwhen 按时间翻；memfetch 取原文；memprofile 聚画像；memloops 钓欠账；" +
                              "memstatus 看索引；memlinks 顺关系；memdebug 解释召回；memcompact 压流水；memaudit 审矛盾；memcleanhistory 只统计历史合成消息（不删除）；" +
                              "memorganize 空闲时小批整理（系统报点/没人说话时，可主动整理一小批未整理的流水）。" +
                              "mem_impression 按人物名取印象摘要；",
                Explanation =
                    "这些函数：检索类是只读的；压缩/审计只写事实卡与报告，不动原文。" +
                    "★铁律：查记忆只用下面这些工具；禁止用 <Python>、glob、读文件、`sqlite3` 去翻记忆目录或索引库——" +
                    "那属于本工具的职责，绕过工具既慢又容易拿到过期/未整理的数据。" +
                    "<memrecall query=\"关键词\"/> 检索事实卡；" +
                    "<memwhen time=\"上周三\" keyword=\"可选\" limit=\"8\"/> 按时间翻旧账；" +
                    "<memfetch source=\"memory_xxx.md\" 或 keyword=\"关键词\" limit=\"3\"/> 取原文；" +
                    "<memprofile who=\"主人/小鲸鱼/俾斯麦/妈\" limit=\"30\"/> 按人聚画像卡；" +
                    "<memloops keyword=\"可选\" limit=\"15\"/> 扫牵挂与欠账；" +
                    "<memstatus/> 看索引多少条、覆盖到哪天；" +
                    "<memcompact/> 手动压一次流水；" +
                    "<memorganize max_files=\"5\"/> 手动整理一小批流水（别超 20，免得占满上下文；系统报点、没人说话的闲时可主动调它，一次一两段即可）；" +
                    "<memaudit/> 手动审一次事实卡；" +
                    "<memlinks query=\"俾斯麦\"/> 看某个词/人/物常和什么一起出现；" +
                     "<mem_impression name=\"沐禾\"/> 取某人的印象摘要。",
            };
            // Explicit：每轮把工具说明/参数表注入系统提示。Implicit 只露一行，模型常不知道有这些工具。
            functionCaller.RegisterHandler(handler, DocumentMode.Explicit, DestroyCancellationToken);
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：XmlHandler 注册失败（角色未启用 FunctionCaller 时属预期）");
        }

        // 扫描一次，把记忆目录切成片段（纯文件 IO，不碰网络/进程）
        try { EnsureChunks(force: true); }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：首次扫描失败"); }

        // 侧车只在开关打开时启动；重活一律丢后台，绝不在 OnAwake 里等
        // ★B：侧车提供索引召回（主路径），不该被"语义开关"卡住；UseSemantic 只控制向量那一路
        if (Configuration != null && Configuration.Enable && hasMemory)
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    bool ok = await sidecar.StartAsync(scriptPath, Configuration.PythonExe, loadModel: Configuration.UseSemantic).ConfigureAwait(false);
                    if (ok)
                    {
                        // 先确保索引在：缺了或旧了，就地重建（后台，不占前台）
                        await EnsureIndexAsync(force: false).ConfigureAwait(false);
                        // 预热：把索引读进内存、建好词表/DF，并吃掉模型首次编码开销。
                        // 大库首次冷读要 2s+，会超出同步等待窗口；预热放到启动后台，不占真查询。
                        try
                        {
                            await sidecar.AskAsync("warm", new Dictionary<string, object?> { ["use_reranker"] = Configuration.UseReranker, ["rerank_model_path"] = Configuration.RerankModelPath }, 120000, indexPath).ConfigureAwait(false);
                            warmDone = true;
                        }
                        catch { }
                        // ★只有开启语义才需要本地片段的向量；关着还算会导致侧车惰性加载模型、
                        //   白编码几千条片段（启动慢 + 吃显存），而且这些向量根本不会被用到。
                        if (Configuration.UseSemantic)
                            await sidecar.AskAsync("vecwarm", new Dictionary<string, object?> { ["batch"] = 32, ["limit"] = 0 }, 8000, indexPath).ConfigureAwait(false);
                    }
                }
                catch (Exception e)
                {
                    logger.LogWarning(e, "Retromem 溯忆：向量侧车异常（不影响主流程）");
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
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：自动整理索引失败（不影响主流程）"); }

                try
                {
                    if (sidecar.IsReady && (DateTime.Now - lastMaintain).TotalHours >= Math.Max(1, Configuration.MaintainIntervalHours))
                    {
                        sidecar.AskSync("maintain", new Dictionary<string, object?>
                        {
                            ["allow_soft"] = Configuration.AllowSoftMaintenance
                        }, 8000, indexPath);
                        lastMaintain = DateTime.Now;
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：每日维护失败（不影响主流程）"); }

                // 语义缓存：开启语义才预热，分批补缺口，不阻塞召回（侧车内部后台线程）
                try
                {
                    if (sidecar.IsReady && Configuration.UseSemantic
                        && (DateTime.Now - lastVecWarm).TotalMinutes >= 30)
                    {
                        sidecar.AskSync("vecwarm", new Dictionary<string, object?>
                        {
                            ["batch"] = 64, ["limit"] = 2000
                        }, 8000, indexPath);
                        lastVecWarm = DateTime.Now;
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：语义缓存预热失败（不影响召回）"); }

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
                            logger.LogInformation("Retromem 溯忆：模型整理启动冷却中，{Minutes} 分钟后才允许自动整理", startupDelay);
                        }
                    }
                    else if (intervalDone)
                    {
                        var result = await LlmOrganizeAsync().ConfigureAwait(false);
                        if (result.files > 0)
                            lastLlmOrganize = DateTime.Now;
                        else if (!string.IsNullOrWhiteSpace(result.fact))
                            logger.LogInformation("Retromem 溯忆：自动整理本轮未推进：{Reason}；下一轮检查重试", result.fact);
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：模型整理失败（不影响主流程）"); }

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
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：自动压缩失败（不影响主流程）"); }

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
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：审计失败（不影响主流程）"); }

                try { await Task.Delay(TimeSpan.FromMinutes(Math.Max(2, Configuration.AutoOrganizeMinutes)), DestroyCancellationToken); }
                catch { break; }
            }
        });

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
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：写入索引构建指纹失败"); }
    }

    /// <summary>供管理页调用：重建索引；失败时保留原库，成功后备份上一个版本。</summary>
    public async Task<string> RebuildIndexAsync() => await RebuildIndexCoreAsync(forceClean: false).ConfigureAwait(false);

    /// <summary>兼容原有按钮语义：按当前规则完整扫描；新库成功之前不删除旧库。</summary>
    public async Task<string> RebuildCleanAsync() => await RebuildIndexCoreAsync(forceClean: true).ConfigureAwait(false);

    async Task<string> RebuildIndexCoreAsync(bool forceClean)
    {
        if (!await rebuildLock.WaitAsync(0).ConfigureAwait(false)) return "已有索引重建在运行，请等待完成。";
        bool graphLifecycleLockHeld = false;
        bool graphWasRunning = false;
        bool graphStoppedForRebuild = false;
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
            await graphLifecycleLock.WaitAsync().ConfigureAwait(false);
            graphLifecycleLockHeld = true;
            graphWasRunning = graphServer != null && !graphServer.HasExited;
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
                    // ★状态目录必须是插件实际目录：审计与人工操作的状态文件在这里，
                    //   按临时目录去找会拿到空状态，重建后软删卡全部复活、编辑全部回退。
                    ["state_dir"] = factsRoot,
                    ["block_max"] = (Configuration?.BlockMaxChars ?? 200),
                    ["extra"] = BuildExtraRoots(),
                    ["exclude"] = ExcludeSnapshot(),
                    ["layers"] = LayerSnapshot(),
                    ["facts_only"] = true,   // 索引只认事实卡：重建也不能把官方流水再灌回来
                }, 180000, tempIndex).ConfigureAwait(false);
            if (string.IsNullOrWhiteSpace(result)) return "重建失败或超时：旧索引保持不变。";
            if (!File.Exists(tempDb)) return "重建失败：新数据库未生成，旧索引保持不变。";
            var closeNew = await sidecar.AskJsonAsync("close", new Dictionary<string, object?>(), 15000, tempIndex).ConfigureAwait(false);
            if (closeNew == null) return "重建失败：无法释放新索引连接，旧索引保持不变。";

            // GraphServer 的独立侧车会缓存只读 SQLite 句柄。先优雅停服并
            // 等待进程树退出，不能确认释放时绝不尝试移动旧库。
            if (!await StopGraphServerAsync().ConfigureAwait(false))
                return "重建失败：图谱服务仍占用索引文件，旧索引保持不变；请关闭图谱后重试。";
            graphStoppedForRebuild = graphWasRunning;

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
            WriteScopeStamp();
            lastBuildCheck = DateTime.MinValue;
            return "安全重建完成：" + result.Trim() + "；原索引备份：" + (dbBackup.Length > 0 ? dbBackup : (indexBackup.Length > 0 ? indexBackup : "无旧库"));
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：安全索引重建失败");
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
            if (graphLifecycleLockHeld) graphLifecycleLock.Release();
            rebuildLock.Release();
            if (graphStoppedForRebuild)
            {
                try
                {
                    var restart = await StartGraphServerOnlyAsync().ConfigureAwait(false);
                    if (!restart.success) logger.LogWarning("Retromem 溯忆：索引重建后图谱服务未能自动重启：{Message}", restart.message);
                }
                catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：索引重建后图谱服务自动重启失败"); }
            }
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

    protected override async Task OnDestroy()
    {
        sourceProgressTimer?.Dispose();
        sourceProgressTimer = null;
        try { ChatBot.ChatSend -= OnChatSend; } catch { }
        await graphLifecycleLock.WaitAsync().ConfigureAwait(false);
        try
        {
            await StopGraphServerAsync().ConfigureAwait(false);
            try { await sidecar.DisposeAsync().ConfigureAwait(false); } catch { }
        }
        finally { graphLifecycleLock.Release(); }
    }

    static string GraphIndexHash(string path)
    {
        try
        {
            string full = Path.GetFullPath(path ?? "").ToLowerInvariant();
            byte[] hash = System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(full));
            return Convert.ToHexString(hash).Substring(0, 16).ToLowerInvariant();
        }
        catch { return ""; }
    }

    async Task<(bool ok, bool foreign)> ProbeGraphHealthAsync(string expectedHash)
    {
        try
        {
            using var http = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromMilliseconds(800) };
            using var resp = await http.GetAsync("http://127.0.0.1:" + GraphPort + "/health").ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode) return (false, false);
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync().ConfigureAwait(false));
            JsonElement root = doc.RootElement;
            string service = root.TryGetProperty("service", out var s) ? (s.GetString() ?? "") : "";
            string hash = root.TryGetProperty("index_hash", out var h) ? (h.GetString() ?? "") : "";
            if (service != "memory-graph") return (false, true);
            return (hash == expectedHash, hash != expectedHash);
        }
        catch { return (false, false); }
    }

    async Task<bool> ProbeGraphPageAsync()
    {
        try
        {
            using var http = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromMilliseconds(800) };
            using var resp = await http.GetAsync("http://127.0.0.1:" + GraphPort + "/").ConfigureAwait(false);
            return resp.IsSuccessStatusCode && resp.Content.Headers.ContentType?.MediaType == "text/html";
        }
        catch { return false; }
    }

    async Task<bool> StopGraphServerAsync()
    {
        var p = graphServer;
        graphServer = null;
        if (p == null) return true;
        bool stopped = false;
        try
        {
            if (!p.HasExited)
            {
                // GraphServer watches parent stdin EOF and closes its read-only
                // sidecar from its own shutdown path.
                try { p.StandardInput.Close(); } catch { }
                try { await p.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false); }
                catch (TimeoutException)
                {
                    if (!p.HasExited)
                    {
                        try { p.Kill(entireProcessTree: true); } catch { }
                        await p.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false);
                    }
                }
            }
            stopped = p.HasExited;
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：等待图谱服务退出失败");
            try { stopped = p.HasExited; } catch { stopped = false; }
        }
        if (stopped)
        {
            graphIndexArg = "";
            try { p.Dispose(); } catch { }
        }
        else graphServer = p;
        return stopped;
    }

    public async Task<string> OpenGraphPageAsync()
    {
        await graphLifecycleLock.WaitAsync().ConfigureAwait(false);
        try
        {
            if (string.IsNullOrWhiteSpace(scriptPath)) return "侧车脚本未就绪，请稍后重试。";
            string serverScript = Path.Combine(Path.GetDirectoryName(scriptPath) ?? "", "GraphServer.py");
            if (!File.Exists(serverScript)) return "关系图谱服务脚本不存在：" + serverScript;
            if (string.IsNullOrWhiteSpace(indexPath)) return "索引路径未就绪，请先构建索引。";
            if (!File.Exists(indexPath))
            {
                string dbPath = Path.Combine(factsRoot, "memory.db");
                if (!File.Exists(dbPath)) return "图谱索引不存在，请先构建索引。";
            }
            string expected = GraphIndexHash(indexPath);
            await SourceStatusAsync().ConfigureAwait(false);
            sourceProgressTimer ??= new Timer(async _ =>
            {
                if (Interlocked.Exchange(ref sourceProgressRefreshing, 1) != 0) return;
                try { await SourceStatusAsync().ConfigureAwait(false); }
                finally { Interlocked.Exchange(ref sourceProgressRefreshing, 0); }
            }, null, 30000, 30000);
            if (graphServer != null && !graphServer.HasExited && graphIndexArg != indexPath)
            {
                if (!await StopGraphServerAsync().ConfigureAwait(false))
                    return "旧的关系图谱服务尚未退出，请稍后重试。";
            }
            graphIndexArg = indexPath;

            var (healthy, foreign) = await ProbeGraphHealthAsync(expected).ConfigureAwait(false);
            if (foreign)
                return $"本机端口 {GraphPort} 已被另一个服务占用，请关闭旧的图谱服务或 Alife 后重试。";

            string logPath = Path.Combine(factsRoot, "graphserver.log");
            if (!healthy && (graphServer == null || graphServer.HasExited))
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = string.IsNullOrWhiteSpace(Configuration?.PythonExe) ? "python" : Configuration.PythonExe,
                    Arguments = "\"" + serverScript + "\"",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(serverScript) ?? "",
                    RedirectStandardError = true,
                    RedirectStandardOutput = true,
                    RedirectStandardInput = true
                };
                psi.Environment["ALIFE_MEMORY_GRAPH_PORT"] = GraphPort.ToString();
                psi.Environment["ALIFE_MEMORY_GRAPH_INDEX"] = indexPath;
                psi.Environment["ALIFE_MEMORY_GRAPH_PARENT_STDIN"] = "1";
                psi.Environment["ALIFE_MEMORY_GRAPH_NO_MODEL"] = "1";
                psi.Environment["PYTHONIOENCODING"] = "utf-8";
                var p = System.Diagnostics.Process.Start(psi);
                if (p == null) return "无法启动关系图谱服务。";
                graphServer = p;
                DataReceivedEventHandler logLine = (_, ev) =>
                {
                    if (string.IsNullOrEmpty(ev.Data)) return;
                    try { File.AppendAllText(logPath, DateTime.Now.ToString("HH:mm:ss ") + ev.Data + Environment.NewLine); } catch { }
                };
                p.ErrorDataReceived += logLine;
                p.OutputDataReceived += logLine;
                p.BeginErrorReadLine();
                p.BeginOutputReadLine();
            }

            bool pageReady = healthy && await ProbeGraphPageAsync().ConfigureAwait(false);
            bool ready = healthy && pageReady;
            for (int i = 0; i < 40 && !ready; i++)
            {
                await Task.Delay(150).ConfigureAwait(false);
                if (graphServer != null && graphServer.HasExited) break;
                var probe = await ProbeGraphHealthAsync(expected).ConfigureAwait(false);
                if (probe.foreign)
                    return $"本机端口 {GraphPort} 已被另一个服务占用，请关闭旧的图谱服务或 Alife 后重试。";
                ready = probe.ok && await ProbeGraphPageAsync().ConfigureAwait(false);
            }
            if (!ready)
                return $"关系图谱服务启动失败（端口 {GraphPort}）。查看日志：{logPath}";

            string url = GraphPageUrl;
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(url) { UseShellExecute = true });
            return "已打开关系图谱详情页：" + url;
        }
        catch (Exception e) { return "打开关系图谱详情页失败：" + e.Message; }
        finally { graphLifecycleLock.Release(); }
    }

    /// <summary>
    /// 启动关系图谱服务但不打开浏览器（供索引重建后恢复图谱服务使用）。
    /// </summary>
    public async Task<(bool success, string message, string url)> StartGraphServerOnlyAsync()
    {
        await graphLifecycleLock.WaitAsync().ConfigureAwait(false);
        try
        {
            if (string.IsNullOrWhiteSpace(roleRoot)) roleRoot = Path.GetDirectoryName(memoryRoot) ?? "";
            string serverScript = Path.Combine(Path.GetDirectoryName(scriptPath) ?? "", "GraphServer.py");
            string indexPath = this.indexPath;

            if (!File.Exists(serverScript))
                return (false, "GraphServer.py 不存在，请检查插件安装。", "");

            if (!File.Exists(indexPath))
            {
                string dbPath = Path.Combine(factsRoot, "memory.db");
                if (!File.Exists(dbPath))
                    return (false, "图谱索引不存在，请先构建索引。", "");
            }

            string expected = GraphIndexHash(indexPath);
            if (graphServer != null && !graphServer.HasExited && graphIndexArg != indexPath)
            {
                if (!await StopGraphServerAsync().ConfigureAwait(false))
                    return (false, "旧的关系图谱服务尚未退出，请稍后重试。", "");
            }
            graphIndexArg = indexPath;

            var (healthy, foreign) = await ProbeGraphHealthAsync(expected).ConfigureAwait(false);
            if (foreign)
                return (false, $"端口 {GraphPort} 已被占用，请关闭旧服务后重试。", "");

            string logPath = Path.Combine(factsRoot, "graphserver.log");
            if (!healthy && (graphServer == null || graphServer.HasExited))
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = string.IsNullOrWhiteSpace(Configuration?.PythonExe) ? "python" : Configuration.PythonExe,
                    Arguments = "\"" + serverScript + "\"",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(serverScript) ?? "",
                    RedirectStandardError = true,
                    RedirectStandardOutput = true,
                    RedirectStandardInput = true
                };
                psi.Environment["ALIFE_MEMORY_GRAPH_PORT"] = GraphPort.ToString();
                psi.Environment["ALIFE_MEMORY_GRAPH_INDEX"] = indexPath;
                psi.Environment["ALIFE_MEMORY_GRAPH_PARENT_STDIN"] = "1";
                psi.Environment["ALIFE_MEMORY_GRAPH_NO_MODEL"] = "1";
                psi.Environment["PYTHONIOENCODING"] = "utf-8";
                var p = System.Diagnostics.Process.Start(psi);
                if (p == null)
                    return (false, "无法启动关系图谱服务。", "");
                graphServer = p;
                DataReceivedEventHandler logLine = (_, ev) =>
                {
                    if (string.IsNullOrEmpty(ev.Data)) return;
                    try { File.AppendAllText(logPath, DateTime.Now.ToString("HH:mm:ss ") + ev.Data + Environment.NewLine); } catch { }
                };
                p.ErrorDataReceived += logLine;
                p.OutputDataReceived += logLine;
                p.BeginErrorReadLine();
                p.BeginOutputReadLine();
            }

            bool pageReady = healthy && await ProbeGraphPageAsync().ConfigureAwait(false);
            bool ready = healthy && pageReady;
            for (int i = 0; i < 40 && !ready; i++)
            {
                await Task.Delay(150).ConfigureAwait(false);
                if (graphServer != null && graphServer.HasExited) break;
                var probe = await ProbeGraphHealthAsync(expected).ConfigureAwait(false);
                if (probe.foreign)
                    return (false, $"端口 {GraphPort} 已被占用。", "");
                ready = probe.ok && await ProbeGraphPageAsync().ConfigureAwait(false);
            }

            if (!ready)
                return (false, $"服务启动超时（端口 {GraphPort}）。日志：{logPath}", "");

            return (true, "图谱服务已启动", GraphPageUrl);
        }
        catch (Exception e)
        {
            return (false, "启动失败：" + e.Message, "");
        }
        finally { graphLifecycleLock.Release(); }
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
            // ★只迁移插件自有事实卡目录：官方 Memory/ 与 Storage/Memory 只读，历史不改
            var roots = new List<string> { factsRoot };
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
                    catch (Exception e)
                    {
                        logger.LogWarning(e, "Retromem 溯忆：事实卡迁移跳过 {File}", f);
                    }
                }
            }
            Directory.CreateDirectory(factsRoot);
            File.WriteAllText(marker, DateTime.Now.ToString("o") + "\nfiles=" + files + " removed=" + removed);
            logger.LogInformation("Retromem 溯忆：事实卡迁移完成，清理 {f} 个文件 / {n} 行垃圾", files, removed);
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：事实卡迁移失败（下次启动重试）"); }
    }

    static bool IsSyntheticHistory(ChatMessageContent m)
    {
        string c = m?.Content ?? "";
        return c.Contains("[消息来源(记忆整理)]", StringComparison.Ordinal)
            || c.Contains("[消息来源(记忆压缩)]", StringComparison.Ordinal);
    }

    /// <summary>手动清理历史里由整理/压缩写入的合成提示词（默认只统计；apply=true 才删，删前备份 History.json）。</summary>
    /// <summary>只统计历史里由整理/压缩写入的合成提示词；本版本不提供删除（历史不改）。</summary>
    Task<(int count, int chars, string backup)> CleanHistoryAsync(bool apply)
    {
        int count = 0, chars = 0;
        foreach (ChatMessageContent m in ChatBot.ChatHistory)
        {
            if (IsSyntheticHistory(m) == false) continue;
            count++;
            chars += (m.Content ?? "").Length;
        }
        // ★删除分支已按硬边界移除：不调用 EditChatHistoryAsync、不写 Memory/History.json
        return Task.FromResult((count, chars, ""));
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
                try { hasMemory = memoryRoots.Any(x => HasMemoryFiles(x.root)) || HasMemoryFiles(factsRoot); } catch { }
            }
            if (hasMemory == false) return message;

            // 先剔掉 [来源][...] 这类方括号标签，再数有效字数
            string stripped = Regex.Replace(message, @"\[[^\]]*\]", " ");
            stripped = StripExcludedPhrases(stripped);
            string bare = new string(stripped.Where(char.IsLetterOrDigit).ToArray());
            if (bare.Length < Configuration.MinQueryChars) return message;

            // 双水位线：先攒水，不到高水位或还在最小间隔内，就不查
            // ★问句不等水位：一句「我前天血压多少」只有 7 个字，按 60 字水位永远攒不到，
            //   等于单句提问从不触发自动召回（实测）。问句直接进检索，仍受最小间隔约束。
            bool questionLike = Regex.IsMatch(stripped, "[?？]")
                || Regex.IsMatch(stripped, "(吗|呢|多少|几个|几天|几点|哪|谁|什么|怎么|为什么|是否|有没有|记不记得)");
            waterLevel += bare.Length;
            int high = Math.Max(1, Configuration.WaterHighChars);
            double gap = Math.Max(0, Configuration.TriggerCooldownSeconds);
            if (waterLevel < high && questionLike == false) return message;
            if ((DateTime.Now - lastTrigger).TotalSeconds < gap) return message;
            // 到点：回落水位、记下本次触发（无论查没查到，都算数，避免每句重试）
            waterLevel = Math.Max(0, Configuration.WaterLowChars);
            lastTrigger = DateTime.Now;

            // 第一优先：按 人/事/物 查 index.json，拿到可读书签就直接用
            // （有上限等待，超时/异常一律放过，退回下面的原提示）
            int waitMs = Math.Max(Configuration.SyncWaitMs, 200);
            if (warmDone == false) waitMs = Math.Max(waitMs, 5000);   // 预热未完成：给大库冷读留出时间，别当没命中
            // ★SkipRecentDays 只作用于"自动记忆提示"（按条目日期跳过最近 N 天）；手动工具传 0，永不跳过
            if (Configuration.UseReranker) waitMs = Math.Max(waitMs, Math.Clamp(Configuration.RerankWaitMs, 20, 5000));
            RecallResult? recallResult = sidecar.RecallDetailedSync(stripped, waitMs, indexPath, Configuration.HalfLifeDays, Configuration.UseGraph, Configuration.SkipRecentDays, Configuration.MaxResults, Configuration.UseSemantic, Configuration.SemanticMinScore, recallGate: Configuration.RecallGateEnable, recallFilter: Configuration.RecallFilterEnable, useReranker: Configuration.UseReranker, rerankModelPath: Configuration.RerankModelPath, rerankMinScore: Configuration.RerankMinScore, vectorMinScore: Configuration.VectorMinScore);
            string? bookmarks = recallResult?.Bookmarks;
            if (string.IsNullOrWhiteSpace(bookmarks) == false)
            {
                // ★只追加：书签拼在本轮用户消息末尾，用户原话一字不动。
                //   框架 ChatSend 的返回值即本轮新消息内容（会进历史），前缀不变，前缀缓存照常命中。
                string tail = Configuration.ToolFirst
                    ? "\n[要用全文：调 <memfetch source=\"出处文件名\"/> 或 <memwhen time=\"上周三\"/>，不要扫盘、不要凭印象编。]"
                    : "\n[要用全文：调 <memfetch source=\"出处文件名\"/>，别凭印象编。]";
                string block =
                    "[Retromem 溯忆·临时记忆参考（按人/事/物检索有效事实卡）]\n" + bookmarks + tail;
                if (recallResult != null && recallResult.Facts.ValueKind == JsonValueKind.Array
                    && RecordCallsSyncWithGap(Guid.NewGuid().ToString("N"), "automatic", recallResult.Facts, recallResult.WindowGen) == false)
                    logger.LogWarning("记忆调用统计写入失败；本轮书签仍正常附加");
                return message + "\n" + block;
            }

            // No verified facts means no automatic hint. Legacy file-vector fallback
            // must not bypass subject/time/evidence filters, including on backend failure.
            return message;

        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆失败，已放过原消息");
            return message;
        }
    }

    // ===== 主动调用的工具（XmlFunction） =====
    // ★必须返回 Task<string>：返回值当轮就能给模型；interactor.Poke 是挂进聊天框，模型要到下一轮才看得到。
    //   （旧版四个工具全是 Task + Poke —— 调了等于白调，这是"工具调用"最要命的 bug。）
    // ★参数不要共用字段：并发调用会串参，一律用局部字典。

    /// <summary>按人/事/物检索事实卡。</summary>
    [XmlFunction(FunctionMode.OneShot, "memrecall")]
    [Description("通过 Retromem 溯忆按人/事/物检索事实卡。想确认往事、查来龙去脉时第一个用它。")]
    public async Task<string> Recall(
        [Description("要查的内容，比如某人、某件事、某个物件")] string query)
    {
        // ★区分两种情况：真的没命中 vs 侧车没响应（超时/报错）。旧版一律说"没翻到"，误导排查。
        JsonElement? r = await sidecar.AskJsonAsync("recall", new Dictionary<string, object?>
        {
            ["query"] = query ?? "",
            ["entry"] = "memrecall",
            ["call_channel"] = "memrecall",
            ["top_k"] = Math.Clamp(Configuration.MaxResults, 1, 20),
            ["half_life"] = Configuration.HalfLifeDays,
            ["use_graph"] = Configuration.UseGraph,
            ["use_semantic"] = Configuration.UseSemantic,
                ["use_reranker"] = Configuration.UseReranker,
                ["rerank_model_path"] = Configuration.RerankModelPath,
                ["rerank_min_score"] = Configuration.RerankMinScore,
                ["vector_min_score"] = Configuration.VectorMinScore,
            ["semantic_min_score"] = Configuration.SemanticMinScore,
        }, 8000, indexPath).ConfigureAwait(false);
        string? bm = (r != null && r.Value.TryGetProperty("bookmarks", out JsonElement bEl)) ? (bEl.GetString() ?? "") : null;
        string msg = bm == null
            ? $"召回没响应（侧车未就绪或超时），不是没有记忆。稍后再试，或用 memstatus 看侧车状态。"
            : string.IsNullOrWhiteSpace(bm)
                ? $"记忆里没翻到跟「{query}」相关的旧账。可以换 memwhen 按时间找，或换个关键词。"
                : bm!;
        if (string.IsNullOrWhiteSpace(bm) == false && r != null
            && r.Value.TryGetProperty("facts", out JsonElement facts) && facts.ValueKind == JsonValueKind.Array
            && await RecordCallsAsyncWithGap(Guid.NewGuid().ToString("N"), "memrecall", facts).ConfigureAwait(false) == false)
            logger.LogWarning("memrecall 统计写入失败；工具结果仍正常返回");
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

    /// <summary>取人物印象：优先用缓存；缓存缺失或过期时用隔离线程的模型依据事实卡生成，再回写缓存。</summary>
    [XmlFunction(FunctionMode.OneShot, "mem_impression")]
    [Description("按人物名取该人物的印象摘要（由其有效事实卡派生）。不确定人名时不要猜。")]
    public async Task<string> Impression(
        [Description("人物名，如 沐禾")] string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return R("请给出人物名。");
        var req = new Dictionary<string, object?> { ["name"] = name.Trim() };
        JsonElement? cached = await sidecar.AskJsonAsync("impression", req, 8000, indexPath).ConfigureAwait(false);
        if (cached == null) return R("印象服务不可用。");

        JsonElement root = cached.Value;
        if (root.TryGetProperty("cached", out JsonElement cachedFlag) && cachedFlag.ValueKind == JsonValueKind.True)
        {
            string text = root.TryGetProperty("impression", out JsonElement boxed) ? (boxed.GetString() ?? "") : "";
            string when = root.TryGetProperty("generated_at", out JsonElement stamp) ? (stamp.GetString() ?? "") : "";
            return R(string.IsNullOrWhiteSpace(text) ? $"{name} 暂无印象。" : $"{name} 的印象（生成于 {when}）：\n{text}");
        }

        if (!root.TryGetProperty("needs_generation", out JsonElement needs) || needs.ValueKind != JsonValueKind.True)
        {
            string reason = root.TryGetProperty("reason", out JsonElement why) ? (why.GetString() ?? "") : "暂无有效事实卡";
            return R($"{name}：{reason}");
        }

        List<string> lines = new();
        List<string> factIds = new();
        List<string> factVersions = new();
        if (root.TryGetProperty("facts", out JsonElement facts) && facts.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement fact in facts.EnumerateArray())
            {
                string id = fact.TryGetProperty("id", out JsonElement idBox) ? (idBox.GetString() ?? "") : "";
                string body = fact.TryGetProperty("text", out JsonElement textBox) ? (textBox.GetString() ?? "") : "";
                string date = fact.TryGetProperty("date", out JsonElement dateBox) ? (dateBox.GetString() ?? "") : "";
                if (id.Length > 0) factIds.Add(id);
                factVersions.Add(fact.TryGetProperty("version", out JsonElement verEl) ? (verEl.GetString() ?? "") : "");
                if (body.Length > 0) lines.Add($"- {date} {body}");
            }
        }
        if (lines.Count == 0) return R($"{name} 暂无有效事实卡，无法生成印象。");

        string prompt = $"以下是关于「{name}」的事实卡，请据此写一段 100-200 字的人物印象，" +
                        "只描述这些事实能支撑的内容，不要补充事实卡里没有的信息：\n" + string.Join("\n", lines);
        string generated;
        try
        {
            generated = (await AskModelIsolatedAsync(prompt).ConfigureAwait(false) ?? "").Trim();
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：印象生成失败 person={Name}", name);
            return R($"生成 {name} 的印象失败：{e.Message}");
        }
        if (generated.Length == 0) return R($"生成 {name} 的印象失败：模型未返回内容（未写入缓存）。");

        var save = new Dictionary<string, object?>
        {
            ["name"] = name.Trim(),
            ["impression"] = generated,
            ["fact_ids"] = factIds,
            ["fact_versions"] = factVersions,
            ["model_source"] = "AskModelIsolatedAsync",
        };
        JsonElement? saved = await sidecar.AskJsonAsync("impressionsave", save, 8000, indexPath).ConfigureAwait(false);
        bool stored = saved != null && saved.Value.TryGetProperty("ok", out JsonElement okEl)
                      && okEl.ValueKind == JsonValueKind.True;
        string saveError = "";
        if (!stored && saved != null && saved.Value.TryGetProperty("error", out JsonElement errEl))
            saveError = errEl.GetString() ?? "";
        string tail = stored ? "" : $"（缓存未写入：{(saveError.Length > 0 ? saveError : "保存失败，下次会重新生成")}）";
        return R($"{name} 的印象（本次生成）：\n{generated}{tail}");
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
                ["use_reranker"] = Configuration.UseReranker,
                ["rerank_model_path"] = Configuration.RerankModelPath,
                ["rerank_min_score"] = Configuration.RerankMinScore,
                ["vector_min_score"] = Configuration.VectorMinScore,
            };
            JsonElement? result = await sidecar.AskJsonAsync("bywhen", args, 8000, indexPath).ConfigureAwait(false);
            string? bm = result != null && result.Value.TryGetProperty("bookmarks", out JsonElement bmElement)
                ? bmElement.GetString()
                : null;
            if (string.IsNullOrWhiteSpace(bm) == false && result != null
                && result.Value.TryGetProperty("facts", out JsonElement facts) && facts.ValueKind == JsonValueKind.Array
                && await RecordCallsAsyncWithGap(Guid.NewGuid().ToString("N"), "memwhen", facts).ConfigureAwait(false) == false)
                logger.LogWarning("memwhen 统计写入失败；工具结果仍正常返回");
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
                ["use_reranker"] = Configuration.UseReranker,
                ["rerank_model_path"] = Configuration.RerankModelPath,
                ["rerank_min_score"] = Configuration.RerankMinScore,
                ["vector_min_score"] = Configuration.VectorMinScore,
                ["semantic_min_score"] = Configuration.SemanticMinScore
            }, 8000, indexPath).ConfigureAwait(false);
            return R(string.IsNullOrWhiteSpace(text) ? "调试查询无结果（侧车可能未就绪）。" : text!);
        }
        catch (Exception e) { return "召回调试失败：" + e.Message; }
    }

    /// <summary>读取 JSON 里的整数字段（数字或字符串都认），取不到按 0。</summary>
    private static int JsonInt(JsonElement element, string name)
    {
        if (element.TryGetProperty(name, out JsonElement value))
        {
            if (value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int number)) return number;
            if (value.ValueKind == JsonValueKind.String && int.TryParse(value.GetString(), out int parsed)) return parsed;
        }
        return 0;
    }

    /// <summary>索引体量/覆盖范围及诊断信息。</summary>
    [XmlFunction(FunctionMode.OneShot, "memstatus")]
    [Description("诊断记忆根目录、索引新旧、排除规则与侧车状态。怀疑没扫到记忆或召回为空时使用。")]
    public async Task<string> Status()
    {
        try
        {
            JsonElement? stats = await sidecar.AskJsonAsync("stats", new Dictionary<string, object?>(), 5000, indexPath).ConfigureAwait(false);
            string? txt = null;
            string upkeep = "";
            string semantic = "";
            if (stats != null && stats.Value.ValueKind == JsonValueKind.Object)
            {
                if (stats.Value.TryGetProperty("text", out JsonElement textEl) && textEl.ValueKind == JsonValueKind.String)
                    txt = textEl.GetString();
                if (stats.Value.TryGetProperty("maintenance", out JsonElement upkeepEl)
                    && upkeepEl.ValueKind == JsonValueKind.Object)
                {
                    if (upkeepEl.TryGetProperty("last", out JsonElement lastEl) && lastEl.ValueKind == JsonValueKind.Object)
                    {
                        string when = lastEl.TryGetProperty("created_at", out JsonElement createdEl)
                            ? (createdEl.GetString() ?? "") : "";
                        bool ok = lastEl.TryGetProperty("ok", out JsonElement okEl) && okEl.ValueKind == JsonValueKind.True;
                        bool softOn = lastEl.TryGetProperty("allow_soft", out JsonElement softEl) && softEl.ValueKind == JsonValueKind.True;
                        upkeep = "｜上次维护 " + (when.Length >= 19 ? when.Substring(0, 19) : when) + (ok ? " 成功" : " 失败") +
                                 "（扫描 " + JsonInt(lastEl, "scanned") + "，软删 " + JsonInt(lastEl, "soft_deleted") +
                                 "，物理清 " + JsonInt(lastEl, "purged") + "，软删" + (softOn ? "开启" : "关闭") + "）";
                    }
                    else upkeep = "｜还没有维护记录";
                }
                if (stats.Value.TryGetProperty("semantic", out JsonElement semEl) && semEl.ValueKind == JsonValueKind.Object)
                {
                    bool ready = semEl.TryGetProperty("ready", out JsonElement readyEl) && readyEl.ValueKind == JsonValueKind.True;
                    bool warmed = semEl.TryGetProperty("warmed", out JsonElement warmEl) && warmEl.ValueKind == JsonValueKind.True;
                    bool running = semEl.TryGetProperty("running", out JsonElement runEl) && runEl.ValueKind == JsonValueKind.True;
                    semantic = "｜语义 " + (ready ? (warmed ? "就绪" : "已加载未预热") : "未加载") +
                               (running ? "·预热中 " + JsonInt(semEl, "done") + "/" + JsonInt(semEl, "total") : "") +
                               "·待算 " + JsonInt(semEl, "total") + " 条·已缓存 " + JsonInt(semEl, "cached") + " 条";
                    if (semEl.TryGetProperty("reranker", out JsonElement rankEl))
                    {
                        semantic += rankEl.TryGetProperty("ready", out JsonElement rankReady) && rankReady.ValueKind == JsonValueKind.True ? "·精排就绪" : "·精排未就绪（关键词降级）";
                        if (rankEl.TryGetProperty("error", out JsonElement rankError) && !string.IsNullOrWhiteSpace(rankError.GetString()))
                            semantic += "·" + rankError.GetString();
                    }
                }
            }
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
                "｜最新源文件 " + newest + "｜" + freshness + "｜" + storeInfo + upkeep + semantic + "｜侧车 " + sidecar.Status +
                "\n" + await SourceStatusAsync().ConfigureAwait(false));
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
    [Description("统计聊天历史里由本插件整理/压缩写入的合成提示词条数。本版本只统计、不删除（历史不改）。")]
    public async Task<string> CleanHistory()
    {
        try
        {
            var (n, chars, _) = await CleanHistoryAsync(false).ConfigureAwait(false);
            if (n == 0) return R("历史里没有本插件写入的合成消息，干净。");
            return R($"历史里有 {n} 条合成消息（{chars} 字）。本版本只统计、不删除；如需清理请手工处理。");
        }
        catch (Exception e) { return R("统计历史失败：" + e.Message); }
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
            // 侧车加载 BGE 通常要几十秒；这期间提案一定失败，先说明白，别白烧一次模型调用
            if (sidecar.IsReady == false)
                return R("侧车还在加载模型（通常 1 分钟内），这次没有开始整理，也没消耗模型调用；稍后再调一次。");
            int cap = Math.Clamp(max_files <= 0 ? 5 : max_files, 1, 20);
            int chars = Math.Clamp(Configuration.LlmOrganizeManualMaxChars, 2000, Configuration.LlmOrganizeMaxChars);
            var (n, fact) = await LlmOrganizeAsync(cap, force: true, maxChars: chars).ConfigureAwait(false);
            if (n == 0) return R(string.IsNullOrWhiteSpace(fact) ? "本轮未完成整理；水位未推进，请查看日志。" : fact);
            string tail = string.IsNullOrWhiteSpace(fact)
                ? "模型认为这批没长期价值，已推进水位。"
                : fact;
            return R($"已整理 {n} 份流水：{tail}（批次刻意小，避免占满上下文；需要继续就再调一次）");
        }
        catch (Exception e) { return "整理失败：" + e.Message; }
    }

    /// <summary>① 手动触发一次压缩。</summary>
    [XmlFunction(FunctionMode.OneShot, "memcompact")]
    [Description("整理较旧流水：新事实自动入库，完全重复跳过，近似或冲突留审；全部处理完成后标记来源，官方文件原样不动。")]
    public async Task<string> Compact()
    {
        int n = await AutoCompactAsync(force: true).ConfigureAwait(false);
        string msg = n > 0
            ? ("已处理 " + n + " 份流水：新事实自动入库，重复跳过，近似或冲突留审；全部处理完成后标记来源（官方文件未动）。")
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
        string[] layers = LayerSnapshot();
        // 层级白名单只约束官方来源：插件自有 MemoryFacts 里没有 L1/L5 这种目录
        bool official = true;
        try
        {
            if (!string.IsNullOrWhiteSpace(factsRoot))
            {
                official = !string.Equals(
                    Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar),
                    Path.GetFullPath(factsRoot).TrimEnd(Path.DirectorySeparatorChar),
                    StringComparison.OrdinalIgnoreCase);
            }
        }
        catch (Exception) { }
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) yield break;
        foreach (string f in EnumerateFilesSafe(root))
        {
            string rel = f.Substring(root.Length).Replace('\\', '/').TrimStart('/');
            if (rel.Contains(".bak", StringComparison.OrdinalIgnoreCase)) continue;
            if (IsExcludedRel(rel, exSnapshot)) continue;
            if (official && !InLayerRange(rel, layers)) continue;
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
        // 索引只认事实卡：官方目录不在也要能建（角色只剩卡时不能失联）
        if (Directory.Exists(memoryRoot) == false && Directory.Exists(factsRoot) == false) return;

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
                // 事实卡也要算"比库新"：卡片追加后不重建，新记忆不会进索引
                if (Directory.Exists(factsRoot))
                {
                    foreach (string factFile in EnumerateMemoryFiles(factsRoot))
                    {
                        try { DateTime t = File.GetLastWriteTime(factFile); if (t > newest) newest = t; } catch { }
                    }
                }
                need = newest > idxTime;   // 有比索引还新的记忆 → 重建
            }
            string manifestPath = Path.Combine(factsRoot, "index_manifest.json");
            if (need == false && !File.Exists(manifestPath)) need = true;
            // 来源范围（层级白名单/排除清单）变了也要重建一次：升级到 5.1 的旧库
            // 还按旧范围（只读 L2–L6）在用，不重建等于配置没生效。
            if (need == false && ScopeChanged()) need = true;
            if (need == false) return;

            EnsureExcludes();
            var r = await sidecar.BuildAsync(memoryRoot, indexPath, blockMax: Configuration.BlockMaxChars,
                extra: BuildExtraRoots(), exclude: ExcludeSnapshot(), layers: LayerSnapshot(),
                factsOnly: true).ConfigureAwait(false);
            if (r != null)
            {
                WriteManifest("auto: " + r.Value.count + " entries");
                WriteScopeStamp();
            }
            else logger.LogWarning("Retromem 溯忆：索引重建未成功（不影响主流程）");
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：索引重建异常（已忽略）");
        }
    }

    /// <summary>
    /// 模型整理：把新落盘的记忆流水交给模型，提炼成"事实卡"追加落盘（原文一律不动）。
    /// 走后台模型调用（ChatAsync，breakLast:false），不进主对话上下文。
    /// 水位记在 llm_organize_state.txt：记"已整理过的 文件|mtime"，只处理没整理过的新文件。
    /// </summary>
    async Task<(int files, string fact)> LlmOrganizeAsync(int maxFiles = 0, bool force = false, int maxChars = 0)
    {
        if (!await organizeLock.WaitAsync(0).ConfigureAwait(false)) return (0, "正在整理中，请稍后再试。");
        organizing = true;
        try { return await LlmOrganizeCoreAsync(maxFiles, force, maxChars).ConfigureAwait(false); }
        finally { organizing = false; organizeLock.Release(); }
    }

    async Task<(int files, string fact)> LlmOrganizeCoreAsync(int maxFiles, bool force, int maxChars = 0)
    {
        if (Configuration == null) return (0, "");
        if (force == false && Configuration.LlmOrganizeEnable == false) return (0, "");
        // ★自动整理绝不在模型回合进行中插 ChatAsync：ChatAsync 会写 ChatHistory，
        //   与 UI 渲染并发时会抛 "Collection was modified"。手动工具(force)不受此限。
        if (force == false && functionCaller.IsIdle == false) return (0, "模型正在处理对话，本轮暂缓自动整理。");
        if (string.IsNullOrWhiteSpace(memoryRoot) || Directory.Exists(memoryRoot) == false) return (0, "整理来源目录不可用，未开始模型调用。");
        // 侧车未就绪时提案一定失败：别先把 20 秒的模型调用烧掉（自动路径同理）
        if (sidecar.IsReady == false) return (0, "侧车尚未就绪，未开始模型调用；下一轮检查重试。");

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
            }
            catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：查询待压缩清单失败（本轮不跳过）"); }
        }

        // 找出没整理过的记忆文件（跳过事实卡本身、索引、状态文件）
        // ★关键顺序：先收集全部候选 → 按修改时间从新到旧排序 → 再按预算选材。
        //   旧实现边枚举边扣预算、最后才排序，枚举顺序（目录/文件名）会让老档先吃光预算，新档永远轮不到。
        var readProgress = LoadOrganizeProgress();
        var fresh = new List<(string path, string source, string key, string text, int end, string prefixHash, DateTime mt)>();
        int budget = Math.Max(2000, maxChars > 0 ? maxChars : Configuration.LlmOrganizeMaxChars);
        try
        {
            var cands = new List<(string path, string baseKey, string rel, DateTime mt, string text, int offset)>();
            foreach (var item in EnumerateAllMemoryFiles())
            {
                string f = item.file;
                if (!IsOrganizeSourceFile(f)) continue;
                string relPath = item.source;
                if (compactHealthy && pendingCompact.Contains(relPath)) continue;   // 留给压缩
                DateTime mt;
                try { mt = File.GetLastWriteTime(f); } catch { continue; }
                string txt;
                try { txt = File.ReadAllText(f); } catch { continue; }
                // 原水位保留兼容；新增前缀校验允许追加后的文件从已读位置续读。
                // 前缀改写才重新整理，失败不推进，未读完的尾部由下一批继续。
                string baseKey = relPath + "|" + ContentKey(txt);
                int offset = OrganizedOffset(relPath, txt, mt, done, legacyDone, stateTime, readProgress);
                if (string.IsNullOrWhiteSpace(txt)) { done.Add(baseKey + "|" + txt.Length); continue; }
                if (offset >= txt.Length) { done.Add(baseKey + "|" + txt.Length); continue; }
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
                fresh.Add((c.path, c.rel, key, seg, c.offset + take,
                           ContentKey(c.text.Substring(0, c.offset + take)), c.mt));
                budget -= take;
            }
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：模型整理扫描失败"); return (0, "读取整理来源失败，未开始模型调用；请查看日志。"); }

        if (fresh.Count == 0) return (0, "没有新增或改动的待整理流水，本轮未调用模型。");

        string body = string.Join("\n\n", fresh.Select(x => "【" + Path.GetFileName(x.path) + "】\n" + x.text));
        string prompt =
            "[消息来源(记忆整理)]下面是刚落盘的新记忆流水。请把它提炼成若干条「事实卡」，每条一行，格式：\n" +
            "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
            "要求：只记事实，别抒情、别复述细节；只提炼有长期价值的信息（决定、进度、健康状况、约定、重要结论）；" +
            "确无长期价值时仅输出 [NO_FACTS]。否则只输出事实行：不要调用任何工具（Python/Expression/Speak 等），不要输出代码、XML 或解释，不要寒暄。\n\n" + body;

        // ★隔离调用：不写角色 ChatHistory（旧写法会把 1 万字提示词永久留在上下文里）
        string cards = await AskModelIsolatedAsync(prompt, throwOnFailure: true).ConfigureAwait(false);
        if (cards.Trim() == "[NO_FACTS]")
        {
            // Only an explicit successful no-facts response advances this watermark.
            SaveOrganizeCompletion(done, fresh.Select(x => (x.source, x.key, x.end, x.prefixHash)));
            return (fresh.Count, "");
        }

        // ★只留事实行：模型回包常夹带 XML/代码（真央那份事实卡就被写进了 <Python>…）
        string cleanCards = SanitizeFactCards(cards, out int droppedCards);
        if (droppedCards > 0)
            logger.LogWarning("Retromem 溯忆：整理回包丢弃 {n} 行非事实内容", droppedCards);
        if (string.IsNullOrWhiteSpace(cleanCards))
        {
            logger.LogWarning("Retromem 溯忆：整理回包没有合法事实行，保留本批输入等待重试");
            return (0, "模型有回包，但没有合法事实行；水位未推进，请查看日志后重试。");
        }
        // ③ 写入合并：不再直接 AppendAllText —— 先与已有事实合并（重复丢弃、相近合并）
        var inputRanges = new List<object?>();
        foreach (var item in fresh)
        {
            int end = 0;
            int bar = item.key.LastIndexOf('|');
            if (bar >= 0) int.TryParse(item.key.Substring(bar + 1), out end);
            int start = Math.Max(0, end - item.text.Length);
            string rel = Path.GetRelativePath(memoryRoot, item.path).Replace('\\', '/');
            inputRanges.Add(BuildInputRange(rel, item.text, start, "input_range_unverified"));
        }
        string factFile = await AppendFactsAsync(cleanCards, inputRanges).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(factFile))
        {
            logger.LogWarning("Retromem 溯忆：事实提案入队失败（水位不推进，稍后重试）");
            return (0, "模型回包已收到，但事实提案未成功入队；水位未推进，请查看日志。");
        }

        SaveOrganizeCompletion(done, fresh.Select(x => (x.source, x.key, x.end, x.prefixHash)));
        return (fresh.Count, factFile);
    }

    /// <summary>各来源的已读取前缀与可证明的成功读取时间。</summary>
    sealed class OrganizeReadProgress
    {
        public int Offset { get; set; }
        public string PrefixHash { get; set; } = "";
        public DateTime ReadAtUtc { get; set; }
    }

    string OrganizeProgressPath => Path.Combine(factsRoot, "organize_read_progress.json");

    Dictionary<string, OrganizeReadProgress> LoadOrganizeProgress()
    {
        if (!File.Exists(OrganizeProgressPath)) return new(StringComparer.OrdinalIgnoreCase);
        var records = JsonSerializer.Deserialize<Dictionary<string, OrganizeReadProgress>>(File.ReadAllText(OrganizeProgressPath));
        return new(records ?? throw new InvalidDataException("整理进度文件不可读"), StringComparer.OrdinalIgnoreCase);
    }

    static bool IsOrganizeSourceFile(string file)
    {
        string name = Path.GetFileName(file);
        return !(name.StartsWith("事实_") || name.StartsWith("审计报告_")
            || name == "index.json" || name == "llm_organize_state.txt"
            || name == "compacted_sources.txt" || name.EndsWith("记忆索引视图.html"));
    }

    static int OrganizedOffset(string source, string text, DateTime modified,
        HashSet<string> done, HashSet<string> legacy, DateTime stateTime,
        Dictionary<string, OrganizeReadProgress> progress)
    {
        string baseKey = source + "|" + ContentKey(text);
        int offset = 0;
        if (done.Contains(baseKey)) return text.Length;
        if (legacy.Contains(source) && modified <= stateTime.AddSeconds(5)) return text.Length;
        if (progress.TryGetValue(source, out var saved) && saved != null
            && saved.Offset > 0 && saved.Offset <= text.Length
            && ContentKey(text.Substring(0, saved.Offset)) == saved.PrefixHash)
            offset = saved.Offset;
        foreach (string key in done)
        {
            if (!key.StartsWith(source + "|", StringComparison.OrdinalIgnoreCase)) continue;
            string[] parts = key.Substring(source.Length + 1).Split('|');
            if (parts.Length != 2 || !int.TryParse(parts[1], out int end)
                || end <= offset || end > text.Length) continue;
            // 同版本延续原水位；升级前已读完的文件，可用旧全量哈希验证现有前缀。
            if (key.StartsWith(baseKey + "|", StringComparison.OrdinalIgnoreCase)
                || ContentKey(text.Substring(0, end)) == parts[0]) offset = end;
        }
        return offset;
    }

    void SaveOrganizeCompletion(HashSet<string> done,
        IEnumerable<(string source, string key, int end, string prefixHash)> reads)
    {
        var progress = LoadOrganizeProgress();
        DateTime now = DateTime.UtcNow;
        foreach (var read in reads)
        {
            done.Add(read.key);
            progress[read.source] = new() { Offset = read.end, PrefixHash = read.prefixHash, ReadAtUtc = now };
        }
        // 只在模型明确无事实、或事实已成功自动入库/留审后推进；失败重试不提前跳过。
        Directory.CreateDirectory(factsRoot);
        string path = OrganizeProgressPath, temp = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try { File.WriteAllText(temp, JsonSerializer.Serialize(progress)); File.Move(temp, path, true); }
        finally { if (File.Exists(temp)) File.Delete(temp); }
        TrySaveState(done);
        lock (sourceProgressLock) sourceProgressCheckedAt = DateTime.MinValue;
    }

    sealed class OrganizeLayerProgress
    {
        public int Total, Pending, Partial, Unreadable, Processed;
        public DateTime LastReadUtc;
    }

    static string SourceLayer(string source)
        => source.Replace('\\', '/').Split('/').FirstOrDefault(IsLayerSegment)?.ToUpperInvariant() ?? "未分层";

    public Task<string> SourceStatusAsync() => Task.Run(() =>
    {
        try
        {
            lock (sourceProgressLock)
            {
                string scope = ScopeSignature();
                if (scope != sourceProgressScope || (DateTime.UtcNow - sourceProgressCheckedAt).TotalSeconds >= 30)
                {
                    var done = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    var legacy = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    DateTime stateTime = File.Exists(llmStatePath) ? File.GetLastWriteTime(llmStatePath) : DateTime.MinValue;
                    if (File.Exists(llmStatePath)) foreach (string raw in File.ReadAllLines(llmStatePath))
                    {
                        string one = raw.Trim(); int bar = one.IndexOf('|');
                        if (bar > 0 && one.Substring(bar + 1).Length > 0 && one.Substring(bar + 1).All(char.IsDigit))
                            legacy.Add(one.Substring(0, bar));
                        else if (one.Length > 0) done.Add(one);
                    }
                    var reads = LoadOrganizeProgress();
                    string[] layers = LayerSnapshot();
                    var rows = new Dictionary<string, OrganizeLayerProgress>(StringComparer.OrdinalIgnoreCase);
                    var pendingFiles = new List<object>();
                    foreach (string layer in new[] { "L1", "L100" }.Concat(layers)) rows.TryAdd(layer.ToUpperInvariant(), new());
                    foreach (var item in EnumerateAllMemoryFiles())
                    {
                        if (!IsOrganizeSourceFile(item.file)) continue;
                        string layer = SourceLayer(item.source);
                        if (!rows.TryGetValue(layer, out var row)) rows[layer] = row = new();
                        row.Total++;
                        try
                        {
                            string text = File.ReadAllText(item.file);
                            int offset = OrganizedOffset(item.source, text, File.GetLastWriteTime(item.file), done, legacy, stateTime, reads);
                            if (!string.IsNullOrWhiteSpace(text) && offset < text.Length)
                            {
                                row.Pending++;
                                pendingFiles.Add(new { source = item.source, layer,
                                    status = offset > 0 ? "partial" : "pending", offset, total_chars = text.Length });
                            }
                            if (offset > 0) { row.Processed++; if (offset < text.Length) row.Partial++; }
                        }
                        catch (IOException) { row.Unreadable++; pendingFiles.Add(new { source = item.source, layer, status = "unreadable", offset = 0, total_chars = 0 }); }
                        catch (UnauthorizedAccessException) { row.Unreadable++; pendingFiles.Add(new { source = item.source, layer, status = "unreadable", offset = 0, total_chars = 0 }); }
                    }
                    foreach (var read in reads)
                        if (read.Value != null && rows.TryGetValue(SourceLayer(read.Key), out var row)
                            && read.Value.ReadAtUtc > row.LastReadUtc) row.LastReadUtc = read.Value.ReadAtUtc;
                    var lines = new List<string>();
                    var layerData = new List<object>();
                    foreach (var pair in rows)
                    {
                        bool enabled = pair.Key == "未分层" || layers.Length == 0 || layers.Contains(pair.Key, StringComparer.OrdinalIgnoreCase);
                        var row = pair.Value;
                        layerData.Add(new { layer = pair.Key, enabled, total = row.Total,
                            pending = row.Pending, partial = row.Partial, unreadable = row.Unreadable,
                            last_read_at = row.LastReadUtc == DateTime.MinValue ? (DateTime?)null : row.LastReadUtc,
                            legacy_time_unknown = row.LastReadUtc == DateTime.MinValue && row.Processed > 0 });
                        string last = row.LastReadUtc == DateTime.MinValue
                            ? (row.Processed > 0 ? "未知（旧版未记录时间）" : "暂无成功读取记录")
                            : row.LastReadUtc.ToLocalTime().ToString("MM-dd HH:mm:ss");
                        lines.Add(pair.Key + (enabled
                            ? " 已启用｜待整理 " + row.Pending + " / " + row.Total + " 份" + (row.Partial > 0 ? "（含部分完成 " + row.Partial + "）" : "")
                            : " 已停用｜待整理 —") + "｜最近成功读取 " + last +
                            (row.Unreadable > 0 ? "｜读取失败 " + row.Unreadable + " 份" : ""));
                    }
                    sourceProgressCache = string.Join("\n", lines);
                    sourceProgressScope = scope;
                    sourceProgressCheckedAt = DateTime.UtcNow;
                    sourceProgressPayload = new() { ["updated_at"] = sourceProgressCheckedAt,
                        ["layers"] = layerData, ["files"] = pendingFiles };
                }
                string run = organizing ? "正在整理" : !Configuration.Enable ? "插件已停用" : Configuration.LlmOrganizeEnable ? "自动整理已开启" : "自动整理已关闭，可手动整理";
                if (Configuration.Enable && Configuration.LlmOrganizeEnable && !organizing)
                {
                    double startup = Configuration.LlmOrganizeStartupDelayMinutes - (DateTime.Now - startedAt).TotalMinutes;
                    if (startup > 0) run += " · 启动等待约 " + Math.Ceiling(startup) + " 分钟";
                }
                sourceProgressPayload["state"] = run;
                Directory.CreateDirectory(factsRoot);
                string snapshot = Path.Combine(factsRoot, "organize_status.json");
                string temp = snapshot + ".tmp-" + Guid.NewGuid().ToString("N");
                try { File.WriteAllText(temp, JsonSerializer.Serialize(sourceProgressPayload)); File.Move(temp, snapshot, true); }
                finally { if (File.Exists(temp)) File.Delete(temp); }
                return run + "\n" + sourceProgressCache + "\n每 30 秒统计；待整理按文件计，含未读完的部分。成功读取含无事实或留审结果。";
            }
        }
        catch (Exception e) { return "来源进度读取失败：" + e.Message; }
    });

    /// <summary>内容指纹；仅修改时间变化不会触发重复整理。</summary>
    static string ContentKey(string text)
    {
        try
        {
            byte[] h = System.Security.Cryptography.SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(text ?? ""));
            return Convert.ToHexString(h).Substring(0, 12);
        }
        catch { return (text ?? "").Length.ToString(); }
    }

    Dictionary<string, object?> BuildInputRange(string source, string text, int offset, string mapping)
    {
        string sample = (text ?? "").Trim();
        if (sample.Length > 240) sample = sample.Substring(0, 240);
        return new Dictionary<string, object?>
        {
            ["source"] = (source ?? "").Replace('\\', '/'),
            ["offset"] = Math.Max(0, offset),
            ["length"] = (text ?? "").Length,
            ["sha1"] = ContentKey(text ?? ""),
            ["excerpt"] = sample,
            ["mapping"] = mapping,
        };
    }

    void TrySaveState(HashSet<string> done)
    {
        string temp = llmStatePath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            // ★不要按字典序截断：旧实现只留 500 条且按路径排序，超过后会把"路径靠前"的文件
            //   水位丢掉 → 它们每轮都被当成新文件重新整理（反复重喂）。
            var all = done.ToList();
            if (all.Count > 20000) all = all.Take(20000).ToList();   // 只是防病态膨胀
            File.WriteAllLines(temp, all);
            File.Move(temp, llmStatePath, true);
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：模型整理水位写盘失败"); }
        finally { if (File.Exists(temp)) File.Delete(temp); }
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
            var pending = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (File.Exists(pendingProposalPath))
                foreach (string ln in File.ReadAllLines(pendingProposalPath))
                    if (string.IsNullOrWhiteSpace(ln) == false) pending.Add(ln.Trim());
            lock (excludesLock)
            {
                excludes.Clear(); foreach (string x in set) excludes.Add(x);
                pendingProposalSources.Clear(); foreach (string x in pending) pendingProposalSources.Add(x);
            }
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
            bool added;
            lock (excludesLock)
            {
                added = excludes.Add(source);
                pendingProposalSources.Remove(source);
            }
            if (!added)
            {
                try
                {
                    string[] pending;
                    lock (excludesLock) pending = pendingProposalSources.ToArray();
                    File.WriteAllLines(pendingProposalPath, pending);
                }
                catch (Exception e) { logger.LogDebug(e, "Retromem 溯忆：清理待审来源清单失败"); }
                return;
            }
            Directory.CreateDirectory(factsRoot);
            File.AppendAllText(compactStatePath, source + "\n");
            try
            {
                string[] pending;
                lock (excludesLock) pending = pendingProposalSources.ToArray();
                File.WriteAllLines(pendingProposalPath, pending);
            }
            catch (Exception e) { logger.LogDebug(e, "Retromem 溯忆：更新待审来源清单失败"); }
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：记录压缩标记失败"); }
    }

    /// <summary>撤回提案审计时恢复来源候选；只改插件自己的标记文件，不改官方记忆文件。</summary>
    void UnmarkCompacted(string source)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(source)) return;
            string[] compacted;
            string[] pending;
            lock (excludesLock)
            {
                excludes.Remove(source);
                pendingProposalSources.Add(source);
                compacted = excludes.ToArray();
                pending = pendingProposalSources.ToArray();
            }
            Directory.CreateDirectory(factsRoot);
            File.WriteAllLines(compactStatePath, compacted);
            File.WriteAllLines(pendingProposalPath, pending);
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：撤回提案后恢复来源失败"); }
    }

    async Task RefreshAcceptedProposalSourcesAsync()
    {
        if (sidecar.IsReady == false || string.IsNullOrWhiteSpace(indexPath)) return;
        string[] pending;
        lock (excludesLock) pending = pendingProposalSources.ToArray();
        try
        {
            JsonElement? root = await sidecar.AskJsonAsync("proposalstatus",
                new Dictionary<string, object?> { ["sources"] = pending, ["factsdir"] = factsRoot },
                3000, indexPath, preserveError: true).ConfigureAwait(false);
            if (root == null) return;
            if (root.Value.TryGetProperty("error", out JsonElement statusError))
            {
                logger.LogWarning("Retromem 溯忆：提案状态查询被拒绝：{Reason}", statusError.GetString() ?? "");
                return;
            }
            if (root.Value.TryGetProperty("restored_sources", out JsonElement restored)
                && restored.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement item in restored.EnumerateArray())
                {
                    string source = item.GetString() ?? "";
                    if (source.Length > 0) UnmarkCompacted(source);
                }
            }
            if (root.Value.TryGetProperty("accepted_sources", out JsonElement accepted) == false
                || accepted.ValueKind != JsonValueKind.Array) return;
            foreach (JsonElement item in accepted.EnumerateArray())
            {
                string source = item.GetString() ?? "";
                if (source.Length == 0) continue;
                lock (excludesLock) pendingProposalSources.Remove(source);
                MarkCompacted(source);
            }
        }
        catch (Exception e) { logger.LogDebug(e, "Retromem 溯忆：查询待审提案状态失败"); }
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
    async Task<string> AskModelIsolatedAsync(string prompt, bool throwOnFailure = false)
    {
        try
        {
            ChatHistoryAgentThread thread = new();
            thread.ChatHistory.AddUserMessage(prompt ?? "");
            // ★ChatStreamingAsync 是流式接口：正文可能只从 onToken 回调出来，
            //   返回值在部分模型/通道下是空串（实测印象生成、事实卡整理全被判成
            //   "模型未返回内容"）。这里同时累积回调内容，再回退到返回值。
            StringBuilder chunks = new();
            string? resp = await chatBot.LanguageModel.ChatStreamingAsync(
                thread,
                chunk => { if (string.IsNullOrEmpty(chunk) == false) chunks.Append(chunk); },
                null, null, null, CancellationToken.None).ConfigureAwait(false);
            string text = (resp ?? "").Trim();
            if (text.Length == 0) text = chunks.ToString().Trim();
            if (text.Length == 0)
                logger.LogWarning("Retromem 溯忆：隔离模型调用返回空（prompt {Chars} 字，流式累计 {Streamed} 字）",
                    (prompt ?? "").Length, chunks.Length);
            if (throwOnFailure && text.Length == 0)
                throw new InvalidOperationException("记忆整理模型返回空响应，未推进水位。");
            return text;
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：隔离模型调用失败");
            if (throwOnFailure) throw;
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

    async Task<string> AppendFactsAsync(string cards, IEnumerable<object?>? provenance = null, string? markSource = null)
    {
        // ★侧车刚启动时还在加载 BGE：AskJsonAsync 会立刻返回 null（不是超时，
        //   日志却写成"超时"，18:32 那次就是这么误报的）。提案前先等它就绪，最多 90 秒。
        for (int i = 0; i < 900 && sidecar.IsReady == false; i++)
            await Task.Delay(100).ConfigureAwait(false);
        if (sidecar.IsReady == false)
        {
            logger.LogWarning("Retromem 溯忆：侧车尚未就绪，事实提案留到下一轮（水位不推进）");
            return "";
        }
        await factLock.WaitAsync().ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(factsRoot);
            string factFile = Path.Combine(factsRoot, "事实_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
            // ★先只留事实行：模型回包里的 XML/代码/解释一律丢掉
            string clean = SanitizeFactCards(cards, out int dropped);
            if (dropped > 0)
                logger.LogWarning("Retromem 溯忆：事实卡丢弃 {n} 行非事实内容（XML/代码/自由文本）", dropped);
            if (string.IsNullOrWhiteSpace(clean)) return "";
            var lines = clean.Split('\n').Select(x => x.Trim()).Where(x => x.Length > 0).ToList();
            if (lines.Count == 0) return "";
            // 统一侧车处理：新事实自动写入，完全重复跳过，其余保留审计候选。
            // provenance 是模型输入范围，不能声称与每条事实逐条精确对应。
            var args = new Dictionary<string, object?>
            {
                ["lines"] = lines,
                ["factsdir"] = factsRoot,
                ["auto_accept"] = true,
                ["target"] = Path.GetFileName(factFile),
                ["provenance"] = provenance?.ToArray() ?? Array.Empty<object?>(),
                ["mark_source"] = markSource ?? "",
            };
            JsonElement? proposal = await sidecar.AskJsonAsync("factproposal", args, 20000, indexPath, preserveError: true).ConfigureAwait(false);
            if (proposal == null)
            {
                // 侧车可能正忙于召回/向量预热，排队超过 20 秒：等一下重试一次
                await Task.Delay(1500).ConfigureAwait(false);
                proposal = await sidecar.AskJsonAsync("factproposal", args, 20000, indexPath, preserveError: true).ConfigureAwait(false);
            }
            if (proposal == null)
            {
                logger.LogWarning("Retromem 溯忆：事实提案入队超时（侧车无响应，水位不推进）");
                return "";
            }
            if (proposal.Value.TryGetProperty("error", out JsonElement errorEl))
            {
                logger.LogWarning("Retromem 溯忆：事实提案被拒绝：{Reason}", errorEl.GetString() ?? "");
                return "";
            }
            if (proposal.Value.TryGetProperty("accepted", out JsonElement accepted)
                && accepted.ValueKind == JsonValueKind.True && accepted.GetBoolean())
            {
                if (!string.IsNullOrWhiteSpace(markSource)) MarkCompacted(markSource);
                return "该来源此前已处理，本次未重复写入。";
            }
            bool automaticResult = proposal.Value.TryGetProperty("processed_count", out JsonElement processed)
                && processed.ValueKind == JsonValueKind.Number && processed.GetInt32() > 0;
            int pendingCount = automaticResult ? JsonInt(proposal.Value, "pending_count") : 0;
            if (proposal.Value.TryGetProperty("proposal_ids", out JsonElement ids) == false
                || ids.ValueKind != JsonValueKind.Array || (!automaticResult && ids.GetArrayLength() == 0))
                return "";
            if (!string.IsNullOrWhiteSpace(markSource))
            {
                bool sourceAccepted = proposal.Value.TryGetProperty("source_accepted", out JsonElement completed)
                    && completed.ValueKind == JsonValueKind.True;
                if (automaticResult && pendingCount == 0 && sourceAccepted)
                    MarkCompacted(markSource);
                else if (!automaticResult || pendingCount > 0)
                    lock (excludesLock) pendingProposalSources.Add(markSource);
                try
                {
                    Directory.CreateDirectory(factsRoot);
                    string[] pending;
                    lock (excludesLock) pending = pendingProposalSources.ToArray();
                    File.WriteAllLines(pendingProposalPath, pending);
                }
                catch (Exception e) { logger.LogDebug(e, "Retromem 溯忆：保存待审来源清单失败"); }
            }
            if (automaticResult)
                return "自动入库 " + JsonInt(proposal.Value, "auto_written") + " 条，重复或已处理跳过 " +
                       (JsonInt(proposal.Value, "auto_skipped") + JsonInt(proposal.Value, "already_processed")) +
                       " 条，待审 " + pendingCount + " 条（仅近似、冲突等候选需人工确认）。";
            return "待审提案：" + ids.GetArrayLength();
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：事实提案提交异常（水位不推进）");
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
        EnsureExcludes();
        compacting = true;
        int done = 0;
        try
        {
            // 上一轮提案可能已经在管理页被人工接受；只有此时才把来源
            // 写入 compacted_sources.txt，从而让官方流水退出压缩候选。
            await RefreshAcceptedProposalSourcesAsync().ConfigureAwait(false);
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
                lock (excludesLock)
                    if (pendingProposalSources.Contains(source)) continue;

                string prompt =
                    "[消息来源(记忆压缩)]下面的流水来自同一份记忆文件。请压成事实卡，每条一行，格式：\n" +
                    "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
                    "要求：只留有长期价值的信息（决定、进度、健康状况、约定、重要结论）；去掉寒暄、重复和过程细节；没价值就不写。" +
                    "只输出事实行：不要调用任何工具（Python/Expression/Speak 等），不要输出代码、XML 或解释。\n\n" + text;

                string cards = await AskModelIsolatedAsync(prompt).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(cards)) { continue; }
                string cleanCards = SanitizeFactCards(cards, out int droppedCards);
                if (droppedCards > 0)
                    logger.LogWarning("Retromem 溯忆：压缩回包丢弃 {n} 行非事实内容 {src}", droppedCards, source);
                if (string.IsNullOrWhiteSpace(cleanCards)) { continue; }

                var inputRanges = new List<object?>();
                if (c.TryGetProperty("provenance", out JsonElement provenance)
                    && provenance.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement item in provenance.EnumerateArray())
                        inputRanges.Add(item.Clone());
                }
                if (inputRanges.Count == 0)
                    inputRanges.Add(BuildInputRange(source, text, 0, "aggregated_index_input_unverified"));
                string factFile = await AppendFactsAsync(cleanCards, inputRanges, source).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(factFile)) continue;
                // 提案尚未人工接受：此处绝不能标记来源已压缩。
                // RefreshAcceptedProposalSourcesAsync 在后续轮次确认 accept 后再标记。
                done++;
                logger.LogInformation("Retromem 溯忆：流水整理结果 {src} -> {f}", source, factFile);
            }
            // 没积压（清单空）或至少压成一份 → 认为压缩链路健康，organize 才敢把文件留给它
            if (anyCand == false || done > 0) lastCompactOk = DateTime.Now;
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：自动压缩失败"); }
        finally { compacting = false; }
        // 提案提交后刷新侧车索引；来源在人工接受前仍保留在官方索引中。
        if (done > 0)
        {
            try { await EnsureIndexAsync(force: true).ConfigureAwait(false); }
            catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：压缩后重建索引失败"); }
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
            logger.LogInformation("Retromem 溯忆：审计新增 {n} 条 -> {f}", fresh.Count, report);
            return fresh.Count;
        }
        catch (Exception e) { logger.LogWarning(e, "Retromem 溯忆：审计失败"); return 0; }
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
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Retromem 溯忆：向量计算失败");
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
