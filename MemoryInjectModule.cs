
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Alife.Foundation;
using Alife.Framework;
using Microsoft.Extensions.Logging;

namespace Mao.MemoryInject;

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
[Module("记忆回忆提示",
    "主人说话时在后台检索记忆目录，命中相关旧账时只在消息末尾提示一句「要不要看」，不塞内容、不占上下文。只读不写，异常放过。",
    defaultCategory: "Mao")]
public class MemoryInjectModule(
    ILogger<MemoryInjectModule> logger) :
    ChatBehaviour,
    IConfigurable<MemoryInjectConfig>
{
    public MemoryInjectConfig Configuration { get; set; } = null!;

    string memoryRoot = "";
    string scriptPath = "";
    DateTime lastScanStamp = DateTime.MinValue;

    readonly List<Chunk> chunks = [];
    readonly SemaphoreSlim scanLock = new(1, 1);
    readonly EmbedSidecar sidecar = new();

    volatile bool vectorsReady;
    volatile bool computing;
    int timeoutStreak;
    DateTime cooldownUntil = DateTime.MinValue;

    public string SidecarStatus => sidecar.Status;

    sealed class Chunk
    {
        public string Source = "";
        public string Text = "";
        public float[]? Vector;
    }

    // ===== 生命周期 =====

    protected override Task OnAwake()
    {
        try
        {
            string sub = (Configuration?.MemorySubPath ?? "Storage/Memory")
                .Replace('/', Path.DirectorySeparatorChar)
                .Replace('\\', Path.DirectorySeparatorChar);
            memoryRoot = Path.Combine(AlifePath.StorageFolderPath, Character.StorageKey, sub);
            scriptPath = Path.Combine(AlifePath.StorageFolderPath, "Plugins", "Mao.MemoryInject", "sidecar_embed.py");
        }
        catch (Exception e)
        {
            logger.LogError(e, "记忆回忆提示：路径初始化失败");
        }

        try { ChatBot.ChatSend += OnChatSend; }
        catch (Exception e) { logger.LogError(e, "记忆回忆提示：挂载 ChatSend 失败"); }

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
                    if (ok) await ComputeVectorsAsync().ConfigureAwait(false);
                }
                catch (Exception e)
                {
                    logger.LogWarning(e, "记忆回忆提示：向量侧车异常（不影响主流程）");
                }
            });
        }

        logger.LogInformation("记忆回忆提示已挂载，记忆目录：{Root}", memoryRoot);
        return Task.CompletedTask;
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
            foreach (string file in Directory.EnumerateFiles(memoryRoot, "*.md", SearchOption.AllDirectories))
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
