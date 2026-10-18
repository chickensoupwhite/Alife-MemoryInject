using System;
using System.Threading;
using System.Threading.Tasks;
using Alife.Framework;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Rendering;
using Microsoft.AspNetCore.Components.Web;

namespace Mao.MemoryRecall;

/// <summary>
/// Mao.MemoryInject —— 插件配置页（手写 C# 组件，非 razor）。
/// 顶部：状态行（自动刷新）+ 索引浏览 + 操作按钮；
/// 下方必须渲染框架自动生成的配置表单（DefaultUI），否则配置项不会显示。
/// </summary>
public class MemoryInjectUI : ModuleUIBase<MemoryInjectModule, MemoryInjectConfig>, IDisposable
{
    const string BtnStyle = "border:0;border-radius:8px;background:#7c3aed;color:#fff;padding:8px 18px;font-size:13px;cursor:pointer";
    const string GhostStyle = "border:1px solid #cfc7ea;border-radius:8px;background:#fff;color:#5b4b8a;padding:7px 14px;font-size:13px;cursor:pointer";

    bool busy;
    string status = "";
    string stats = "状态读取中…";
    string query = "";
    string view = "";
    Timer? refreshTimer;

    protected override void OnInitialized()
    {
        base.OnInitialized();
        // ⑥ 自动刷新：每 5 秒拉一次状态（只读，很轻）
        refreshTimer = new Timer(_ => _ = InvokeAsync(async () =>
        {
            try
            {
                if (Module == null) return;
                stats = await Module.StatsAsync();
                StateHasChanged();
            }
            catch { }
        }), null, 800, 5000);
    }

    public void Dispose()
    {
        refreshTimer?.Dispose();
        refreshTimer = null;
    }

    protected override void BuildRenderTree(RenderTreeBuilder b)
    {
        base.BuildRenderTree(b);
        if (Configuration == null) { b.AddContent(0, "配置未加载"); return; }
        int i = 0;

        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "padding:14px 4px;display:flex;flex-direction:column;gap:14px");

        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "font-size:13.5px;color:#3a4252;line-height:1.75");
        b.AddContent(i++, "记忆书签：后台按人/事/物检索记忆目录，命中旧账只在消息末尾追加一句提示，不塞内容、不占上下文。下面可以直接浏览索引（每 5 秒自动刷新状态）。");
        b.CloseElement();

        // ① 状态行（自动刷新）
        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "font-size:12.5px;color:#5a6270;background:#f6f3ff;border:1px solid #e6e2f5;border-radius:9px;padding:8px 12px;line-height:1.6");
        b.AddContent(i++, stats);
        b.CloseElement();

        // ② 搜索框
        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "display:flex;gap:8px;flex-wrap:wrap;align-items:center");
        b.OpenElement(i++, "input");
        b.AddAttribute(i++, "type", "text");
        b.AddAttribute(i++, "placeholder", "关键词（留空 = 最近全部）");
        b.AddAttribute(i++, "value", query);
        b.AddAttribute(i++, "style", "flex:1;min-width:220px;padding:8px 12px;border:1px solid #ddd6f3;border-radius:8px;font-size:13px");
        b.AddAttribute(i++, "oninput", EventCallback.Factory.Create<ChangeEventArgs>(this, e => query = e.Value?.ToString() ?? ""));
        b.CloseElement();
        b.CloseElement();

        // ③ 按钮区
        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "display:flex;gap:10px;flex-wrap:wrap");

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", BtnStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this, Browse));
        b.AddContent(i++, busy ? "处理中…" : "浏览索引");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", GhostStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.RebuildIndexAsync())));
        b.AddContent(i++, "重建索引");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", GhostStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.ViewIndexAsync())));
        b.AddContent(i++, "生成HTML视图");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", GhostStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.RebuildCleanAsync())));
        b.AddContent(i++, "重建干净库");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", GhostStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.Compact())));
        b.AddContent(i++, "压缩一次");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", GhostStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.Audit())));
        b.AddContent(i++, "审计一次");
        b.CloseElement();

        b.CloseElement();

        // ④ 浏览结果
        if (!string.IsNullOrEmpty(view))
        {
            b.OpenElement(i++, "div");
            b.AddAttribute(i++, "style", "font-size:12.5px;color:#3f4654;line-height:1.7;padding:10px 12px;border:1px solid #e6e2f5;border-radius:9px;background:#fcfbff;white-space:pre-wrap;max-height:360px;overflow:auto");
            b.AddContent(i++, view);
            b.CloseElement();
        }

        // ⑤ 操作结果
        if (!string.IsNullOrEmpty(status))
        {
            b.OpenElement(i++, "div");
            b.AddAttribute(i++, "style", "font-size:12.5px;color:#69707d;line-height:1.6;padding:8px 12px;border:1px solid #e6e2f5;border-radius:9px;background:#faf8ff;white-space:pre-wrap");
            b.AddContent(i++, status);
            b.CloseElement();
        }

        b.AddContent(i++, DefaultUI);

        b.CloseElement();
    }

    async Task Browse()
    {
        if (busy) return;
        busy = true; StateHasChanged();
        try
        {
            view = Module == null
                ? "模块未就绪"
                : await Module.BrowseAsync(query, "", "", "", "", "", 0, 50);
        }
        catch (Exception e) { view = "浏览出错：" + e.Message; }
        finally { busy = false; StateHasChanged(); }
    }

    async Task RunAsync(Func<Task<string>> action)
    {
        if (busy) return;
        busy = true; status = ""; StateHasChanged();
        try { status = await action(); }
        catch (Exception e) { status = "出错：" + e.Message; }
        finally { busy = false; StateHasChanged(); }
    }
}
