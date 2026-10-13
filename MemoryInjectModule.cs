
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ComponentModel;
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
    "主人说话时在后台检索记忆目录，命中相关旧账时只在消息末尾提示一句「要不要看」，不塞内容、不占上下文。只读不写，异常放过。",
    defaultCategory: "Mao",
    editorUI: typeof(MemoryInjectUI))]
public class MemoryInjectModule(
    ILogger<MemoryInjectModule> logger,
    XmlFunctionCaller functionCaller,
    Interactor<MemoryInjectModule> interactor) :
    ChatBehaviour,
    IConfigurable<MemoryInjectConfig>
{
    public MemoryInjectConfig Configuration { get; set; } = null!;

    string memoryRoot = "";
    string scriptPath = "";
    string indexPath = "";
    DateTime lastBuildCheck = DateTime.MinValue;
    DateTime lastScanStamp = DateTime.MinValue;

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

    protected override Task OnAwake()
    {
        try
        {
            string sub = (Configuration?.MemorySubPath ?? "Storage/Memory")
                .Replace('/', Path.DirectorySeparatorChar)
                .Replace('\\', Path.DirectorySeparatorChar);
            memoryRoot = Path.Combine(AlifePath.StorageFolderPath, Character.StorageKey, sub);
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

                // 每角色一份索引：放记忆目录里，多角色各存各的，不打架
                indexPath = Path.Combine(memoryRoot, "index.json");
            }
        }
        catch (Exception e)
        {
            logger.LogError(e, "记忆回忆提示：路径初始化失败");
        }

        try { ChatBot.ChatSend += OnChatSend; }
        catch (Exception e) { logger.LogError(e, "记忆回忆提示：挂载 ChatSend 失败"); }

        // 主动调用的工具（取原文/画像卡/钓欠账/按人事物查书签），隐式注入
        try
        {
            XmlHandler handler = new(this)
            {
                Description = "记忆检索工具：按人/事/物翻自己落盘的旧账。想确认往事、查某人某事的来龙去脉、或盘点还欠着什么时用。" +
                              "四把镰刀：memrecall 按人事物查书签；memfetch 取原文（给出处文件名或关键字）；memprofile 按人聚画像卡；memloops 钓牵挂与欠账。",
                Explanation =
                    "四个只读函数，全部从本角色的记忆目录按 index.json 查，不改文件、不占上下文：" +
                    "<memrecall query=\"关键词\"/> 查书签；" +
                    "<memfetch source=\"memory_xxx.md\" 或 keyword=\"关键词\" limit=\"3\"/> 取记忆原文；" +
                    "<memprofile who=\"主人/小鲸鱼/俾斯麦/妈\" limit=\"30\"/> 按人聚画像卡；" +
                    "<memloops keyword=\"可选\" limit=\"15\"/> 扫描牵挂与欠账。",
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
        if (Configuration != null && Configuration.Enable && Configuration.UseSemantic)
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
                        await ComputeVectorsAsync().ConfigureAwait(false);
                    }
                }
                catch (Exception e)
                {
                    logger.LogWarning(e, "记忆回忆提示：向量侧车异常（不影响主流程）");
                }
            });
        }

        // ⑨每日维护调度：后台每小时看一眼，距上次超过20小时就跑一次 maintain（带错过天数补偿）
        _ = Task.Run(async () =>
        {
            while (true)
            {
                try
                {
                    if (sidecar.IsReady && (DateTime.Now - lastMaintain).TotalHours >= Math.Max(1, Configuration.MaintainIntervalHours))
                    {
                        sidecar.AskSync("maintain", new Dictionary<string, object?> { ["index"] = indexPath }, 8000);
                        lastMaintain = DateTime.Now;
                        logger.LogInformation("记忆回忆提示：每日维护已执行（衰减/清理）");
                    }
                }
                catch (Exception e) { logger.LogWarning(e, "记忆回忆提示：每日维护失败（不影响主流程）"); }
                try { await Task.Delay(TimeSpan.FromHours(1), DestroyCancellationToken); }
                catch { break; }
            }
        });

        logger.LogInformation("记忆回忆提示已挂载，记忆目录：{Root}", memoryRoot);
        return Task.CompletedTask;
    }

    /// <summary>⑧终局结算：把一段长输出折成卡片（历史里只留卡片+id），正文挪到 settled/ 目录，需要时用 memfetch id 取回。</summary>
    [XmlFunction(FunctionMode.OneShot, "memsettle")]
    public Task SettleOutput(
        [Description("要折叠的长文本")] string text,
        [Description("可选：卡片 id，不填自动生成")] string card_id = "")
    {
        try
        {
            var args = new Dictionary<string, object?> { ["index"] = indexPath, ["text"] = text, ["card_id"] = card_id };
            string? raw = sidecar.AskSync("settle", args, 6000);
            string card = raw ?? "";
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(raw ?? "{}");
                if (doc.RootElement.TryGetProperty("card", out var c)) card = c.GetString() ?? card;
            }
            catch { }
            interactor.Poke(string.IsNullOrWhiteSpace(card) ? "折叠失败。" : card);
        }
        catch (Exception e) { interactor.Poke("折叠失败：" + e.Message); }
        return Task.CompletedTask;
    }

    /// <summary>⑩生成记忆索引视图 HTML 并打开，一键查看全部条目。</summary>
    [XmlFunction(FunctionMode.OneShot, "memview")]
    public Task ViewIndex()
    {
        try
        {
            string html = Path.Combine(memoryRoot, "记忆索引视图.html");
            sidecar.AskSync("view", new Dictionary<string, object?> { ["index"] = indexPath, ["out"] = html }, 6000);
            if (File.Exists(html))
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(html) { UseShellExecute = true });
            interactor.Poke("记忆索引视图已生成并打开。");
        }
        catch (Exception e) { interactor.Poke("生成视图失败：" + e.Message); }
        return Task.CompletedTask;
    }

    /// <summary>供管理页调用：生成并打开记忆索引视图，返回结果文字。</summary>
    public Task<string> ViewIndexAsync()
    {
        try
        {
            string html = Path.Combine(memoryRoot, "记忆索引视图.html");
            sidecar.AskSync("view", new Dictionary<string, object?> { ["index"] = indexPath, ["out"] = html }, 6000);
            if (File.Exists(html))
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(html) { UseShellExecute = true });
            return Task.FromResult("视图已生成并打开：" + html);
        }
        catch (Exception e) { return Task.FromResult("生成视图失败：" + e.Message); }
    }

    /// <summary>供管理页调用：重建本角色的记忆索引，返回结果文字。</summary>
    public async Task<string> RebuildIndexAsync()
    {
        try
        {
            string? r = await sidecar.AskAsync("build",
                new Dictionary<string, object?> { ["memdir"] = memoryRoot, ["out"] = indexPath, ["block_max"] = Configuration.BlockMaxChars },
                180000, indexPath).ConfigureAwait(false);
            return "索引重建完成：" + (r ?? "").Trim();
        }
        catch (Exception e) { return "重建索引失败：" + e.Message; }
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
            string? bookmarks = sidecar.RecallSync(stripped, Math.Max(Configuration.SyncWaitMs, 200), indexPath, Configuration.HalfLifeDays, Configuration.UseGraph);
            if (string.IsNullOrWhiteSpace(bookmarks) == false)
            {
                // ★add-only 原则：只在本条消息末尾"追加"临时书签，绝不改写原 message、
                //   不往历史里写。参考 livingmemory 的 extra_user_content 注入（防击穿前缀缓存）。
                string block =
                    "[记忆书签·临时·仅本轮参考（按人/事/物查 index.json）]\n" + bookmarks +
                    "\n[要用全文：去 " + memoryRoot + " 翻对应那份，别凭印象编。]";
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

    readonly Dictionary<string, object?> toolArgs = [];

    /// <summary>按人/事/物查书签。等一句话就能看旧账。</summary>
    [XmlFunction(FunctionMode.OneShot, "memrecall")]
    public async Task Recall(
        [Description("要查的内容，比如某人、某件事、某个物件")] string query)
    {
        string? bm = await sidecar.RecallAsync(query, 5000, indexPath, Configuration.HalfLifeDays, Configuration.UseGraph).ConfigureAwait(false);
        interactor.Poke(string.IsNullOrWhiteSpace(bm)
            ? $"记忆里没翻到跟「{query}」相关的旧账。"
            : $"按「{query}」翻到：\n{bm}");
    }

    /// <summary>取原文：按出处文件名或关键字，把记忆正文捞出来。</summary>
    [XmlFunction(FunctionMode.OneShot, "memfetch")]
    public async Task Fetch(
        [Description("出处文件名，如 memory_20261012_夜.md；按文件名取时填这个")] string? source = null,
        [Description("关键字；不按文件名时用它搜正文")] string? keyword = null,
        [Description("最多取几段，默认3")] int limit = 3)
    {
        toolArgs.Clear();
        toolArgs["source"] = source ?? "";
        toolArgs["keyword"] = keyword ?? "";
        toolArgs["limit"] = limit;
        string? txt = await sidecar.AskAsync("fetch", toolArgs, 5000, indexPath).ConfigureAwait(false);
        interactor.Poke(string.IsNullOrWhiteSpace(txt) ? "没取到原文（换个文件名或关键字再试）。" : txt);
    }

    /// <summary>按人聚画像卡：某人的事按时间排成一页。</summary>
    [XmlFunction(FunctionMode.OneShot, "memprofile")]
    public async Task Profile(
        [Description("谁，如 主人/小鲸鱼/俾斯麦/妈")] string who,
        [Description("最多列几条，默认30")] int limit = 30)
    {
        toolArgs.Clear();
        toolArgs["who"] = who;
        toolArgs["limit"] = limit;
        string? txt = await sidecar.AskAsync("profile", toolArgs, 5000, indexPath).ConfigureAwait(false);
        interactor.Poke(string.IsNullOrWhiteSpace(txt) ? $"没翻到跟「{who}」相关的记忆。" : txt);
    }

    /// <summary>钓牵挂与欠账：扫记忆里的待办、还欠、没动笔这类字。</summary>
    [XmlFunction(FunctionMode.OneShot, "memloops")]
    public async Task Loops(
        [Description("可选关键字，如 运动/作文；不填则扫全部牵挂词")] string? keyword = null,
        [Description("最多列几条，默认15")] int limit = 15)
    {
        toolArgs.Clear();
        toolArgs["keyword"] = keyword ?? "";
        toolArgs["limit"] = limit;
        string? txt = await sidecar.AskAsync("loops", toolArgs, 5000, indexPath).ConfigureAwait(false);
        interactor.Poke(string.IsNullOrWhiteSpace(txt) ? "没扫到牵挂或欠账。" : txt);
    }

    // ===== 扫描 =====

    void EnsureChunks(bool force = false)
    {
        if (Directory.Exists(memoryRoot) == false) return;

        DateTime stamp;
        try { stamp = Directory.GetLastWriteTime(memoryRoot); }
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

    /// <summary>枚举记忆目录下的记忆文件（兼容 .md 与 .txt 两种后缀）。</summary>
    static IEnumerable<string> EnumerateMemoryFiles(string root)
    {
        foreach (string pat in new[] { "*.md", "*.txt" })
            foreach (string f in Directory.EnumerateFiles(root, pat, SearchOption.AllDirectories))
                yield return f;
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
            bool need = force || File.Exists(indexPath) == false;
            if (need == false)
            {
                DateTime idxTime = File.GetLastWriteTime(indexPath);
                DateTime newest = idxTime;
                foreach (string f in EnumerateMemoryFiles(memoryRoot))
                {
                    try { DateTime t = File.GetLastWriteTime(f); if (t > newest) newest = t; } catch { }
                }
                need = newest > idxTime;   // 有比索引还新的记忆 → 重建
            }
            if (need == false) return;

            var r = await sidecar.BuildAsync(memoryRoot, indexPath, blockMax: Configuration.BlockMaxChars).ConfigureAwait(false);
            if (r != null) logger.LogInformation("记忆书签：索引已重建 {N} 条 -> {Out}", r.Value.count, r.Value.outPath);
            else logger.LogWarning("记忆书签：索引重建未成功（不影响主流程）");
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "记忆书签：索引重建异常（已忽略）");
        }
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
