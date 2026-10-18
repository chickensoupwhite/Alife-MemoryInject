
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

    string memoryRoot = "";          // 官方记忆目录（只读）
    string roleRoot = "";            // 角色根目录
    string factsRoot = "";           // 我们自己的事实层（写这里，绝不写官方目录）
    string compactStatePath = "";    // 已压缩来源清单（只影响我们的索引，不动官方文件）
    readonly HashSet<string> excludes = new(StringComparer.OrdinalIgnoreCase);
    readonly object excludesLock = new();
    DateTime lastExcludesLoad = DateTime.MinValue;
    string scriptPath = "";
    string indexPath = "";
    DateTime lastBuildCheck = DateTime.MinValue;
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
    DateTime lastLlmOrganize = DateTime.MinValue;   // 上次模型整理时间（提炼事实卡）
    DateTime lastCompact = DateTime.MinValue;        // 上次自动压缩（流水 -> 事实卡）
    DateTime lastAudit = DateTime.MinValue;          // 上次事实审计
    volatile bool compacting;                        // 压缩/审计互斥，避免并发重复干活
    volatile bool auditing;
    readonly SemaphoreSlim factLock = new(1, 1);      // 事实卡写入串行化（整理与压缩会并发）
    string llmStatePath = "";                        // 记录"已整理到哪个文件/时间"的水位

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
        if (Directory.Exists(configuredRoot) && HasMemoryFiles(configuredRoot)) return configuredRoot;
        string[] cands = { "Memory", "Storage/Memory", "Storage", "" };
        foreach (string c in cands)
        {
            string norm = c.Replace('/', Path.DirectorySeparatorChar);
            string cand = string.IsNullOrEmpty(norm) ? roleRoot : Path.Combine(roleRoot, norm);
            if (HasMemoryFiles(cand)) return cand;
        }
        return configuredRoot;
    }

    protected override Task OnAwake()
    {
        try
        {
            roleRoot = Path.Combine(AlifePath.StorageFolderPath, Character.StorageKey);
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

        // 本角色有没有记忆文件：没有就别启侧车、别检索（否则会串到别人的索引）
        try { hasMemory = HasMemoryFiles(memoryRoot); } catch { hasMemory = false; }

        try { ChatBot.ChatSend += OnChatSend; }
        catch (Exception e) { logger.LogError(e, "记忆回忆提示：挂载 ChatSend 失败"); }

        // 主动调用的工具（取原文/画像卡/钓欠账/按人事物查书签），隐式注入
        try
        {
            XmlHandler handler = new(this)
            {
                Description = "记忆检索工具：翻自己落盘的旧账。想确认往事、查来龙去脉、按时间找、盘点欠账时，先调这里的工具，" +
                              "不要用文件系统去扫记忆目录。九把镰刀：memrecall 按人事查书签；memwhen 按时间（上周三/三天前/昨晚）翻；" +
                              "memfetch 取原文；memprofile 按人聚画像；memloops 钓牵挂欠账；memstatus 看索引体量；" +
                              "memcompact 压流水；memaudit 查重复矛盾；memlinks 顺关系看邻居。",
                Explanation =
                    "八个函数：六个检索是只读的；压缩/审计只写事实卡与报告，不动原文。" +
                    "★铁律：查记忆只用下面这些工具，禁止用 glob/读文件的方式翻记忆目录——只有工具返回的内容你才看得到。" +
                    "<memrecall query=\"关键词\"/> 查书签；" +
                    "<memwhen time=\"上周三\" keyword=\"可选\" limit=\"8\"/> 按时间翻旧账；" +
                    "<memfetch source=\"memory_xxx.md\" 或 keyword=\"关键词\" limit=\"3\"/> 取原文；" +
                    "<memprofile who=\"主人/小鲸鱼/俾斯麦/妈\" limit=\"30\"/> 按人聚画像卡；" +
                    "<memloops keyword=\"可选\" limit=\"15\"/> 扫牵挂与欠账；" +
                    "<memstatus/> 看索引多少条、覆盖到哪天；" +
                    "<memcompact/> 手动压一次流水；" +
                    "<memaudit/> 手动审一次事实卡；" +
                    "<memlinks query=\"俾斯麦\"/> 看某个词/人/物常和什么一起出现。",
            };
            functionCaller.RegisterHandler(handler, DocumentMode.Implicit, DestroyCancellationToken);
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
                    bool ok = await sidecar.StartAsync(scriptPath, Configuration.PythonExe).ConfigureAwait(false);
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
                    if ((DateTime.Now - lastLlmOrganize).TotalMinutes >= Math.Max(5, Configuration.LlmOrganizeMinutes))
                    {
                        await LlmOrganizeAsync().ConfigureAwait(false);
                        lastLlmOrganize = DateTime.Now;
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

    /// <summary>供管理页调用：重建本角色的记忆索引，返回结果文字。</summary>
    public async Task<string> RebuildIndexAsync()
    {
        try
        {
            EnsureExcludes();
            string? r = await sidecar.AskAsync("build",
                new Dictionary<string, object?>
                {
                    ["memdir"] = memoryRoot,
                    ["out"] = indexPath,
                    ["block_max"] = (Configuration?.BlockMaxChars ?? 200),
                    ["extra"] = new[] { factsRoot },
                    ["exclude"] = ExcludeSnapshot(),
                },
                180000, indexPath).ConfigureAwait(false);
            return "索引重建完成：" + (r ?? "").Trim();
        }
        catch (Exception e) { return "重建索引失败：" + e.Message; }
    }

    /// <summary>管理页：清空索引库后按当前配置全量重建（官方文件、事实卡都不动）。</summary>
    public async Task<string> RebuildCleanAsync()
    {
        try
        {
            await sidecar.AskAsync("reset", new Dictionary<string, object?>(), 15000, indexPath).ConfigureAwait(false);
            string dbp = Path.Combine(factsRoot, "memory.db");
            try { if (File.Exists(dbp)) File.Delete(dbp); } catch { }
            try { if (File.Exists(indexPath)) File.Delete(indexPath); } catch { }
            return "干净库已重建：" + await RebuildIndexAsync().ConfigureAwait(false);
        }
        catch (Exception e) { return "重建干净库失败：" + e.Message; }
    }

    protected override Task OnDestroy()
    {
        try { ChatBot.ChatSend -= OnChatSend; } catch { }
        _ = Task.Run(async () => { try { await sidecar.DisposeAsync(); } catch { } });
        return Task.CompletedTask;
    }

    // ===== 主流程 =====

    string OnChatSend(string message)
    {
        try
        {
            if (Configuration == null || Configuration.Enable == false) return message;
            if (string.IsNullOrWhiteSpace(message)) return message;
            if (message.Contains(ChatBot.PokeMessageTag, StringComparison.Ordinal)) return message;
            // 本角色没有记忆文件：不检索，绝不串到别人的索引。
            // ★但目录可能是后建/后补的：60 秒重判一次，别因为启动时没有就永久判死。
            if (hasMemory == false && (DateTime.Now - lastMemCheck).TotalSeconds >= 60)
            {
                lastMemCheck = DateTime.Now;
                try { hasMemory = HasMemoryFiles(memoryRoot); } catch { }
            }
            if (hasMemory == false) return message;

            // 先剔掉 [来源][...] 这类方括号标签，再数有效字数
            string stripped = Regex.Replace(message, @"\[[^\]]*\]", " ");
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
            string? bookmarks = sidecar.RecallSync(stripped, waitMs, indexPath, Configuration.HalfLifeDays, Configuration.UseGraph, Configuration.SkipRecentDays);
            if (string.IsNullOrWhiteSpace(bookmarks) == false)
            {
                // ★add-only 原则：只在本条消息末尾"追加"临时书签，绝不改写原 message、
                //   不往历史里写。参考 livingmemory 的 extra_user_content 注入（防击穿前缀缓存）。
                string tail = Configuration.ToolFirst
                    ? "\n[要用全文：调 <memfetch source=\"出处文件名\"/> 或 <memwhen time=\"上周三\"/>，不要扫盘、不要凭印象编。]"
                    : "\n[要用全文：去 " + memoryRoot + " 翻对应那份，别凭印象编。]";
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
        string? bm = await sidecar.RecallAsync(query, 5000, indexPath, Configuration.HalfLifeDays, Configuration.UseGraph).ConfigureAwait(false);
        string msg = string.IsNullOrWhiteSpace(bm)
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
            };
            string? bm = await sidecar.AskFieldAsync("bywhen", args, "bookmarks", 8000, indexPath).ConfigureAwait(false);
            return R(string.IsNullOrWhiteSpace(bm) ? $"「{time}」那段时间没翻到记忆（或者说不清是哪段）。换个说法试试，比如「上周三」「三天前」。" : bm!);
        }
        catch (Exception e) { return "按时间查失败：" + e.Message; }
    }

    /// <summary>索引体量/覆盖范围。</summary>
    [XmlFunction(FunctionMode.OneShot, "memstatus")]
    [Description("看一眼记忆索引的体量与覆盖时间范围。不确定有没有记忆、或怀疑索引没更新时用。")]
    public async Task<string> Status()
    {
        try
        {
            string? txt = await sidecar.AskAsync("stats", new Dictionary<string, object?>(), 5000, indexPath).ConfigureAwait(false);
            string file = File.Exists(indexPath)
                ? "；索引文件 " + Math.Round(new FileInfo(indexPath).Length / 1024.0 / 1024.0, 1) + " MB，更新于 " +
                  File.GetLastWriteTime(indexPath).ToString("MM-dd HH:mm")
                : "；索引文件还不存在";
            return R((string.IsNullOrWhiteSpace(txt) ? "索引读不出来（侧车可能没就绪）。" : txt!) + file + "；记忆目录 " + memoryRoot);
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
        string report = Path.Combine(memoryRoot, "审计报告_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
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
        DateTime newest = Directory.GetLastWriteTime(root);
        foreach (string f in EnumerateMemoryFiles(root))
        {
            try { DateTime t = File.GetLastWriteTime(f); if (t > newest) newest = t; } catch { }
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
            foreach (string file in EnumerateMemoryFiles(memoryRoot))
            {
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

    /// <summary>枚举记忆目录下的记忆文件（兼容 .md 与 .txt；跳过已压缩来源——它们只是不进我们的索引）。</summary>
    IEnumerable<string> EnumerateMemoryFiles(string root)
    {
        EnsureExcludes();
        // 快照一次：HashSet 在后台线程会被改写，边遍历边判断会炸
        string[] exSnapshot = ExcludeSnapshot();
        string[] skip = { "settled", "backup", "archive", "MemoryFacts" };
        foreach (string pat in new[] { "*.md", "*.txt" })
            foreach (string f in Directory.EnumerateFiles(root, pat, SearchOption.AllDirectories))
            {
                string rel = f.Substring(root.Length).Replace('\\', '/');
                if (rel.Contains(".bak", StringComparison.OrdinalIgnoreCase)) continue;
                if (IsExcludedRel(rel, exSnapshot)) continue;   // 已压缩/手动排除：只跳过索引
                bool skipIt = false;
                foreach (string d in skip)
                    if (rel.Contains("/" + d + "/")) { skipIt = true; break; }
                if (!skipIt) yield return f;
            }
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
                foreach (string f in EnumerateMemoryFiles(memoryRoot))
                {
                    try { DateTime t = File.GetLastWriteTime(f); if (t > newest) newest = t; } catch { }
                }
                need = newest > idxTime;   // 有比索引还新的记忆 → 重建
            }
            if (need == false) return;

            EnsureExcludes();
            var r = await sidecar.BuildAsync(memoryRoot, indexPath, blockMax: Configuration.BlockMaxChars,
                extra: new[] { factsRoot }, exclude: ExcludeSnapshot()).ConfigureAwait(false);
            if (r != null) logger.LogInformation("记忆书签：索引已重建 {N} 条 -> {Out}", r.Value.count, r.Value.outPath);
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
    async Task LlmOrganizeAsync()
    {
        if (Configuration == null || Configuration.LlmOrganizeEnable == false) return;
        if (string.IsNullOrWhiteSpace(memoryRoot) || Directory.Exists(memoryRoot) == false) return;

        // 读水位：已整理过的 文件相对路径|mtime 集合
        var done = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            if (File.Exists(llmStatePath))
                foreach (string ln in File.ReadAllLines(llmStatePath))
                    if (string.IsNullOrWhiteSpace(ln) == false) done.Add(ln.Trim());
        }
        catch { }

        // ★与 ① 协调：压缩开着时，"条目多且够旧"的文件留给 compact，organize 不重复喂模型
        var pendingCompact = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (Configuration.AutoCompactEnable)
        {
            try
            {
                var scanArgs = new Dictionary<string, object?>
                {
                    ["memdir"] = memoryRoot,
                    ["min_entries"] = Configuration.AutoCompactMinEntries,
                    ["min_age_days"] = Configuration.AutoCompactMinAgeDays,
                    ["max_chars"] = 1,
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
        var fresh = new List<(string path, string key, string text)>();
        int budget = Math.Max(2000, Configuration.LlmOrganizeMaxChars);
        try
        {
            foreach (string f in EnumerateMemoryFiles(memoryRoot))
            {
                string name = Path.GetFileName(f);
                if (name.StartsWith("事实_") || name == "index.json" || name.EndsWith("记忆索引视图.html")) continue;
                string relPath = f.Substring(memoryRoot.Length).Replace('\\', '/').TrimStart('/');
                if (pendingCompact.Count > 0 && pendingCompact.Contains(relPath)) continue;   // 留给压缩
                DateTime mt;
                try { mt = File.GetLastWriteTime(f); } catch { continue; }
                string key = relPath + "|" + mt.Ticks;
                if (done.Contains(key)) continue;
                string txt;
                try { txt = File.ReadAllText(f); } catch { continue; }
                if (string.IsNullOrWhiteSpace(txt)) { done.Add(key); continue; }
                if (txt.Length > budget) txt = txt.Substring(0, budget);
                fresh.Add((f, key, txt));
                budget -= txt.Length;
                if (budget <= 0) break;
            }
        }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：模型整理扫描失败"); return; }

        if (fresh.Count == 0) return;

        string body = string.Join("\n\n", fresh.Select(x => "【" + Path.GetFileName(x.path) + "】\n" + x.text));
        string prompt =
            "[消息来源(记忆整理)]下面是刚落盘的新记忆流水。请把它提炼成若干条「事实卡」，每条一行，格式：\n" +
            "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
            "要求：只记事实，别抒情、别复述细节；只提炼有长期价值的信息（决定、进度、健康状况、约定、重要结论）；" +
            "没价值的不记。直接输出卡片行，不要解释、不要寒暄。\n\n" + body;

        ChatResult result;
        try { result = await chatBot.ChatAsync(prompt, breakLast: false).ConfigureAwait(false); }
        catch (Exception e) { logger.LogWarning(e, "记忆书签：模型整理调用失败"); return; }

        if (result.Exception != null) { logger.LogWarning("记忆书签：模型整理出错 {m}", result.Exception.Message); return; }
        string cards = (result.AIMessage ?? "").Trim();
        if (string.IsNullOrWhiteSpace(cards))
        {
            // 模型认为没价值：仍记水位，免得反复重试同一个文件
            foreach (var x in fresh) done.Add(x.key);
            TrySaveState(done);
            return;
        }

        // ③ 写入合并：不再直接 AppendAllText —— 先与已有事实合并（重复丢弃、相近合并）
        string factFile = await AppendFactsAsync(cards).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(factFile))
        {
            logger.LogWarning("记忆书签：模型整理写盘失败（水位不推进，稍后重试）");
            return;
        }
        logger.LogInformation("记忆书签：模型整理落盘 {n} 个文件 -> {f}（已合并去重）", fresh.Count, factFile);

        foreach (var x in fresh) done.Add(x.key);
        TrySaveState(done);
    }

    void TrySaveState(HashSet<string> done)
    {
        try
        {
            // 只留最近 500 条水位，别无限长
            var tail = done.Count > 500 ? done.Skip(done.Count - 500).ToList() : done.ToList();
            File.WriteAllLines(llmStatePath, tail);
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
        string r = rel.Replace('\\', '/').TrimStart('/');
        foreach (string one in list)
        {
            if (string.IsNullOrEmpty(one)) continue;
            if (one.EndsWith("/"))
            {
                if (r.StartsWith(one, StringComparison.OrdinalIgnoreCase)) return true;
            }
            else if (string.Equals(r, one, StringComparison.OrdinalIgnoreCase)) return true;
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
    async Task<string> AppendFactsAsync(string cards)
    {
        await factLock.WaitAsync().ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(factsRoot);
            string factFile = Path.Combine(factsRoot, "事实_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
            string existing = File.Exists(factFile) ? File.ReadAllText(factFile) : "";
            var lines = (cards ?? "").Split('\n').Select(x => x.Trim()).Where(x => x.Length > 0).ToList();
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
                body = (string.IsNullOrEmpty(existing) ? "" : existing.TrimEnd() + "\n") + cards;
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

            foreach (JsonElement c in cands.EnumerateArray())
            {
                string source = c.TryGetProperty("source", out JsonElement se) ? (se.GetString() ?? "") : "";
                string text = c.TryGetProperty("text", out JsonElement te) ? (te.GetString() ?? "") : "";
                if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(text)) continue;

                string prompt =
                    "[消息来源(记忆压缩)]下面的流水来自同一份记忆文件。请压成事实卡，每条一行，格式：\n" +
                    "- 日期 | 涉及谁 | 属哪件事 | 一句话事实\n" +
                    "要求：只留有长期价值的信息（决定、进度、健康状况、约定、重要结论）；去掉寒暄、重复和过程细节；没价值就不写。" +
                    "直接输出卡片行，不要解释。\n\n" + text;

                ChatResult r;
                try { r = await chatBot.ChatAsync(prompt, breakLast: false).ConfigureAwait(false); }
                catch (Exception e) { logger.LogWarning(e, "记忆书签：压缩调用失败 {src}", source); continue; }
                if (r.Exception != null) { logger.LogWarning("记忆书签：压缩出错 {m}", r.Exception.Message); continue; }
                string cards = (r.AIMessage ?? "").Trim();
                if (string.IsNullOrWhiteSpace(cards)) { logger.LogInformation("记忆书签：压缩无产出 {src}", source); continue; }

                string factFile = await AppendFactsAsync(cards).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(factFile)) continue;
                // ★B：只记"已压缩"，官方文件一个字节都不动（旧版会 Move 到 archive/，那会踩坏官方插件）
                MarkCompacted(source);
                done++;
                logger.LogInformation("记忆书签：已压缩 {src} -> {f}", source, factFile);
            }
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

            string report = Path.Combine(memoryRoot, "审计报告_" + DateTime.Now.ToString("yyyyMMdd") + ".md");
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
            if (chunks.Count == 0) return;
            if (sidecar.IsReady == false) return;
            for (int i = 0; i < chunks.Count; i += 32)
            {
                var batch = chunks.Skip(i).Take(32).Where(c => c.Vector == null).ToList();
                if (batch.Count == 0) continue;
                float[][]? vecs = await sidecar.EmbedAsync(batch.Select(c => c.Text).ToArray()).ConfigureAwait(false);
                if (vecs == null) return;
                for (int j = 0; j < batch.Count && j < vecs.Length; j++)
                    batch[j].Vector = vecs[j];
            }
            vectorsReady = chunks.All(c => c.Vector != null);
            logger.LogInformation("记忆回忆提示：向量就绪 {Ok}（{N} 段）", vectorsReady, chunks.Count);
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
        var ready = chunks.Where(c => c.Vector != null).ToList();
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
        foreach (Chunk c in chunks)
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
