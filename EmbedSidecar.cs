
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Mao.MemoryRecall;

/// <summary>
/// 嵌入计算子进程管理：拉起常驻 Python 服务，通过 stdin/stdout 交换 JSON。
/// 所有 await 均 ConfigureAwait(false)。
/// </summary>
public sealed class EmbedSidecar : IAsyncDisposable
{
    Process? process;
    StreamWriter? stdin;
    StreamReader? stdout;
    CancellationTokenSource? readCts;
    Task? readTask;

    readonly SemaphoreSlim writeLock = new(1, 1);
    readonly ConcurrentDictionary<int, TaskCompletionSource<JsonElement>> pending = new();
    int nextId;

    public bool IsReady { get; private set; }
    public string Status { get; private set; } = "未启动";
    public string Device { get; private set; } = "";

    /// <summary>侧车是否正在跑批量向量化。批量时同步路径直接放过，绝不抢。</summary>
    public volatile bool Busy;

    readonly TaskCompletionSource<bool> readyTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>启动子进程并等待 ready。返回是否成功。</summary>
    public async Task<bool> StartAsync(string scriptPath, string? pythonExe, int waitMs = 120000, bool loadModel = true)
    {
        try
        {
            if (File.Exists(scriptPath) == false)
            {
                Status = "找不到脚本：" + scriptPath;
                return false;
            }

            ProcessStartInfo psi = new()
            {
                FileName = string.IsNullOrWhiteSpace(pythonExe) ? "python" : pythonExe,
                Arguments = loadModel ? $"\"{scriptPath}\"" : $"\"{scriptPath}\" --no-model",
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            psi.Environment["PYTHONIOENCODING"] = "utf-8";
            psi.Environment["PYTHONUTF8"] = "1";

            process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            process.Start();
            stdin = process.StandardInput;
            stdout = process.StandardOutput;

            _ = Task.Run(async () =>
            {
                try
                {
                    var err = process.StandardError;
                    while (await err.ReadLineAsync().ConfigureAwait(false) != null) { }
                }
                catch { }
            });

            readCts = new CancellationTokenSource();
            readTask = Task.Run(() => ReadLoopAsync(readCts.Token));

            Task done = await Task.WhenAny(readyTcs.Task, Task.Delay(waitMs)).ConfigureAwait(false);
            if (done == readyTcs.Task && readyTcs.Task.Result)
            {
                IsReady = true;
                return true;
            }
            Status = "等待就绪超时";
            return false;
        }
        catch (Exception e)
        {
            Status = "启动失败：" + e.Message;
            return false;
        }
    }

    async Task ReadLoopAsync(CancellationToken ct)
    {
        try
        {
            var reader = stdout!;
            while (ct.IsCancellationRequested == false)
            {
                string? line = await reader.ReadLineAsync().ConfigureAwait(false);
                if (line == null) break;
                if (string.IsNullOrWhiteSpace(line)) continue;
                if (line[0] != '{') continue;

                JsonElement root;
                try
                {
                    using JsonDocument doc = JsonDocument.Parse(line);
                    root = doc.RootElement.Clone();
                }
                catch { continue; }

                if (root.TryGetProperty("fn", out JsonElement fnEl) &&
                    fnEl.GetString() == "ready")
                {
                    Device = root.TryGetProperty("device", out JsonElement dv) ? dv.GetString() ?? "" : "";
                    Status = "就绪 (" + Device + ")";
                    readyTcs.TrySetResult(true);
                    continue;
                }

                if (root.TryGetProperty("id", out JsonElement idEl) && idEl.TryGetInt32(out int id))
                {
                    if (pending.TryRemove(id, out TaskCompletionSource<JsonElement>? tcs))
                        tcs.TrySetResult(root);
                }
            }
        }
        catch { }
        finally
        {
            Status = "已退出";
            foreach (KeyValuePair<int, TaskCompletionSource<JsonElement>> kv in pending)
                kv.Value.TrySetException(new Exception("sidecar 已退出"));
            pending.Clear();
            readyTcs.TrySetResult(false);
        }
    }

    /// <summary>批量向量化。失败或超时返回 null。</summary>
    public async Task<float[][]?> EmbedAsync(string[] texts, int timeoutMs = 20000)
    {
        if (IsReady == false || process == null || process.HasExited) return null;
        if (texts.Length == 0) return [];

        int id = Interlocked.Increment(ref nextId);
        TaskCompletionSource<JsonElement> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = tcs;

        try
        {
            string payload = JsonSerializer.Serialize(new { id, fn = "embed", texts });
            await writeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                await stdin!.WriteLineAsync(payload.AsMemory()).ConfigureAwait(false);
                await stdin.FlushAsync().ConfigureAwait(false);
            }
            finally { writeLock.Release(); }

            Task done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            if (done != tcs.Task)
            {
                pending.TryRemove(id, out _);
                Status = "请求超时";
                return null;
            }

            JsonElement root = await tcs.Task.ConfigureAwait(false);
            if (root.TryGetProperty("ok", out JsonElement okEl) == false || okEl.GetBoolean() == false)
                return null;
            if (root.TryGetProperty("vectors", out JsonElement vecEl) == false)
                return null;

            List<float[]> list = [];
            foreach (JsonElement v in vecEl.EnumerateArray())
            {
                float[] arr = new float[v.GetArrayLength()];
                int i = 0;
                foreach (JsonElement f in v.EnumerateArray())
                    arr[i++] = f.GetSingle();
                list.Add(arr);
            }
            return [.. list];
        }
        catch
        {
            pending.TryRemove(id, out _);
            return null;
        }
    }

    /// <summary>
    /// 有上限的同步等待：最多阻塞 waitMs 毫秒拿一个查询向量。
    /// 侧车常驻后单次约 10ms，正常无感；超时立刻返回 null，绝不死等。
    /// 内部走 Task.Run 隔离，不会把调用方（同步钩子）拖死在 UI 线程上。
    /// </summary>
    public float[]? EmbedSync(string text, int waitMs)
    {
        if (IsReady == false || Busy || process == null || process.HasExited) return null;
        int budget = Math.Max(waitMs, 20);
        Task<float[][]?> t = Task.Run(() => EmbedAsync([text], budget * 4));
        try
        {
            if (t.Wait(budget) == false) return null;
            float[][]? v = t.Result;
            return (v != null && v.Length > 0) ? v[0] : null;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 有上限的同步召回：最多阻塞 waitMs 毫秒拿书签文本。
    /// 侧车常驻后单次约十几毫秒；超时或未就绪立刻返回 null，绝不死等、绝不外抛。
    /// 走 Task.Run 隔离，不会把同步钩子所在的 UI 线程拖死。
    /// </summary>
    public string? RecallSync(string query, int waitMs, string? indexPath = null, double? halfLife = null, bool? useGraph = null, int skipRecentDays = 0, int topK = 5, bool useSemantic = false, double semanticMinScore = 0.58)
    {
        if (IsReady == false || Busy || process == null || process.HasExited) return null;
        int budget = Math.Max(waitMs, 20);
        Task<string?> t = Task.Run(() => RecallAsync(query, budget * 4, indexPath, halfLife, useGraph, skipRecentDays, topK, useSemantic, semanticMinScore));
        try
        {
            if (t.Wait(budget) == false) return null;
            return t.Result;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>异步召回：发 recall 请求给侧车、拿书签文本；失败返回 null。</summary>
    public async Task<string?> RecallAsync(string query, int timeoutMs = 800, string? indexPath = null, double? halfLife = null, bool? useGraph = null, int skipRecentDays = 0, int topK = 5, bool useSemantic = false, double semanticMinScore = 0.58)
    {
        if (IsReady == false || process == null || process.HasExited) return null;
        int id = Interlocked.Increment(ref nextId);
        TaskCompletionSource<JsonElement> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = tcs;

        try
        {
            string payload = JsonSerializer.Serialize(new { id, fn = "recall", query, top_k = Math.Clamp(topK, 1, 20), index = indexPath ?? "", half_life = halfLife, use_graph = useGraph, skip_recent_days = skipRecentDays, use_semantic = useSemantic, semantic_min_score = semanticMinScore });
            await writeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                await stdin!.WriteLineAsync(payload.AsMemory()).ConfigureAwait(false);
                await stdin.FlushAsync().ConfigureAwait(false);
            }
            finally { writeLock.Release(); }

            Task done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            if (done != tcs.Task)
            {
                pending.TryRemove(id, out _);
                return null;
            }

            JsonElement root = await tcs.Task.ConfigureAwait(false);
            if (root.TryGetProperty("ok", out JsonElement okEl) == false || okEl.GetBoolean() == false)
                return null;
            if (root.TryGetProperty("bookmarks", out JsonElement bmEl) == false)
                return null;
            string? s = bmEl.GetString();
            return string.IsNullOrWhiteSpace(s) ? null : s;
        }
        catch
        {
            pending.TryRemove(id, out _);
            return null;
        }
    }

    /// <summary>
    /// 带时间维的召回：把 date_from/date_to/period/hour_from/hour_to 直接透传给侧车 recall。
    /// 用于「上周三」「三天前」「昨晚」这类按时间翻旧账；返回书签文本，失败返回 null。
    /// </summary>
    public async Task<string?> RecallExAsync(string query, int timeoutMs = 3000, string? indexPath = null,
        double? halfLife = null, bool? useGraph = null, int topK = 5,
        string? dateFrom = null, string? dateTo = null, string? period = null,
        int? hourFrom = null, int? hourTo = null)
    {
        if (IsReady == false || process == null || process.HasExited) return null;
        int id = Interlocked.Increment(ref nextId);
        TaskCompletionSource<JsonElement> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = tcs;

        try
        {
            string payload = JsonSerializer.Serialize(new
            {
                id,
                fn = "recall",
                query,
                top_k = Math.Max(1, topK),
                index = indexPath ?? "",
                half_life = halfLife,
                use_graph = useGraph,
                date_from = dateFrom ?? "",
                date_to = dateTo ?? "",
                period = period ?? "",
                hour_from = hourFrom,
                hour_to = hourTo,
            });
            await writeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                await stdin!.WriteLineAsync(payload.AsMemory()).ConfigureAwait(false);
                await stdin.FlushAsync().ConfigureAwait(false);
            }
            finally { writeLock.Release(); }

            Task done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            if (done != tcs.Task)
            {
                pending.TryRemove(id, out _);
                return null;
            }

            JsonElement root = await tcs.Task.ConfigureAwait(false);
            if (root.TryGetProperty("ok", out JsonElement okEl) == false || okEl.GetBoolean() == false)
                return null;
            if (root.TryGetProperty("bookmarks", out JsonElement bmEl) == false)
                return null;
            string? s = bmEl.GetString();
            return string.IsNullOrWhiteSpace(s) ? null : s;
        }
        catch
        {
            pending.TryRemove(id, out _);
            return null;
        }
    }

    /// <summary>
    /// 后台重建索引：请侧车扫描记忆目录、重写 index.json，并清掉自己的索引缓存。
    /// 重建慢，调用方一律放后台；超时/失败返回 null，绝不外抛。
    /// </summary>
    public async Task<(int count, string outPath)?> BuildAsync(string memdir, string? outPath = null, int timeoutMs = 60000, int? blockMax = null,
        IEnumerable<string>? extra = null, IEnumerable<string>? exclude = null)
    {
        if (IsReady == false || process == null || process.HasExited) return null;
        if (string.IsNullOrWhiteSpace(memdir)) return null;

        int id = Interlocked.Increment(ref nextId);
        TaskCompletionSource<JsonElement> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = tcs;

        try
        {
            string payload = JsonSerializer.Serialize(new
            {
                id, fn = "build", memdir, @out = outPath ?? "", block_max = blockMax,
                extra = extra?.ToArray() ?? Array.Empty<string>(),
                exclude = exclude?.ToArray() ?? Array.Empty<string>(),
            });
            await writeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                await stdin!.WriteLineAsync(payload.AsMemory()).ConfigureAwait(false);
                await stdin.FlushAsync().ConfigureAwait(false);
            }
            finally { writeLock.Release(); }

            Task done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            if (done != tcs.Task)
            {
                pending.TryRemove(id, out _);
                Status = "建索引超时";
                return null;
            }

            JsonElement root = await tcs.Task.ConfigureAwait(false);
            if (root.TryGetProperty("ok", out JsonElement okEl) == false || okEl.GetBoolean() == false)
                return null;
            int c = root.TryGetProperty("count", out JsonElement cEl) ? cEl.GetInt32() : 0;
            string o = root.TryGetProperty("out", out JsonElement oEl) ? oEl.GetString() ?? "" : "";
            return (c, o);
        }
        catch
        {
            pending.TryRemove(id, out _);
            return null;
        }
    }

    /// <summary>
    /// 通用查询口：给 fetch / profile / loops 这类只读工具用。
    /// 侧车返回的 text 字段原样带回；超时/失败返回 null，绝不死等、绝不外抛。
    /// </summary>
    public string? AskSync(string fn, Dictionary<string, object?> args, int waitMs, string? indexPath = null)
    {
        if (IsReady == false || process == null || process.HasExited) return null;
        int budget = Math.Max(waitMs, 50);
        Task<string?> t = Task.Run(() => AskAsync(fn, args, budget * 4, indexPath));
        try
        {
            if (t.Wait(budget) == false) return null;
            return t.Result;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>异步版通用查询：发 fn + args 给侧车，取回 text 字段。</summary>
    public Task<string?> AskAsync(string fn, Dictionary<string, object?> args, int timeoutMs = 3000, string? indexPath = null)
        => AskFieldAsync(fn, args, "text", timeoutMs, indexPath);

    /// <summary>
    /// 异步取整个回包（compactscan / auditscan / browse 这类结构化结果要用整包）。
    /// 超时或 ok=false 返回 null。
    /// </summary>
    public async Task<JsonElement?> AskJsonAsync(string fn, Dictionary<string, object?> args, int timeoutMs = 5000, string? indexPath = null)
    {
        if (IsReady == false || process == null || process.HasExited) return null;

        int id = Interlocked.Increment(ref nextId);
        TaskCompletionSource<JsonElement> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        pending[id] = tcs;

        try
        {
            var payload = new Dictionary<string, object?>(args) { ["id"] = id, ["fn"] = fn, ["index"] = indexPath ?? "" };
            string json = JsonSerializer.Serialize(payload);

            await writeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                await stdin!.WriteLineAsync(json.AsMemory()).ConfigureAwait(false);
                await stdin.FlushAsync().ConfigureAwait(false);
            }
            finally { writeLock.Release(); }

            Task done = await Task.WhenAny(tcs.Task, Task.Delay(timeoutMs)).ConfigureAwait(false);
            if (done != tcs.Task)
            {
                pending.TryRemove(id, out _);
                return null;
            }

            JsonElement root = await tcs.Task.ConfigureAwait(false);
            if (root.TryGetProperty("ok", out JsonElement okEl) == false || okEl.GetBoolean() == false)
                return null;
            return root.Clone();
        }
        catch
        {
            pending.TryRemove(id, out _);
            return null;
        }
    }

    /// <summary>
    /// 异步版通用查询（指定字段）：bytime 返回 bookmarks、stats 返回 count/first/last，
    /// 不能一律取 text。超时/失败返回 null，绝不死等、绝不外抛。
    /// </summary>
    public async Task<string?> AskFieldAsync(string fn, Dictionary<string, object?> args, string field, int timeoutMs = 3000, string? indexPath = null)
    {
        JsonElement? root = await AskJsonAsync(fn, args, timeoutMs, indexPath).ConfigureAwait(false);
        if (root == null) return null;
        if (root.Value.TryGetProperty(field, out JsonElement el) == false) return null;
        // 兼容数值字段（stats 的 count）：非字符串时用原始文本
        string? s = el.ValueKind == JsonValueKind.String ? el.GetString() : el.ToString();
        return string.IsNullOrWhiteSpace(s) ? null : s;
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            readCts?.Cancel();
            if (stdin != null)
            {
                try
                {
                    await stdin.WriteLineAsync("{\"fn\":\"__shutdown__\"}").ConfigureAwait(false);
                    await stdin.FlushAsync().ConfigureAwait(false);
                }
                catch { }
            }
            if (process != null && process.HasExited == false)
            {
                if (process.WaitForExit(3000) == false)
                {
                    try { process.Kill(); } catch { }
                }
            }
            stdin?.Dispose();
            stdout?.Dispose();
            process?.Dispose();
        }
        catch { }
        finally
        {
            IsReady = false;
        }
    }
}
