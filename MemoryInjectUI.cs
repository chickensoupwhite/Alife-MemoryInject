using System;
using System.Threading.Tasks;
using Alife.Framework;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Rendering;
using Microsoft.AspNetCore.Components.Web;

namespace Mao.MemoryRecall;

/// <summary>
/// Mao.MemoryInject —— 插件配置页（手写 C# 组件，非 razor）。
/// 顶部：说明 + 两个操作按钮（重建索引 / 打开记忆视图）；
/// 下方必须渲染框架自动生成的配置表单（DefaultUI），否则配置项不会显示。
/// </summary>
public class MemoryInjectUI : ModuleUIBase<MemoryInjectModule, MemoryInjectConfig>
{
    const string BtnStyle = "border:0;border-radius:8px;background:#7c3aed;color:#fff;padding:8px 18px;font-size:13px;cursor:pointer";

    bool busy;
    string status = "";

    protected override void BuildRenderTree(RenderTreeBuilder b)
    {
        base.BuildRenderTree(b);
        if (Configuration == null) { b.AddContent(0, "配置未加载"); return; }
        int i = 0;

        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "padding:14px 4px;display:flex;flex-direction:column;gap:14px");

        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "font-size:13.5px;color:#3a4252;line-height:1.75");
        b.AddContent(i++, "记忆书签：主人说话时后台按人/事/物检索记忆目录，命中相关旧账只在消息末尾追加一句提示，不塞内容、不占上下文。只读不写，异常放过。");
        b.CloseElement();

        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "style", "display:flex;gap:10px;flex-wrap:wrap");

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", BtnStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.RebuildIndexAsync())));
        b.AddContent(i++, busy ? "处理中…" : "重建索引");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "style", BtnStyle);
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.ViewIndexAsync())));
        b.AddContent(i++, busy ? "处理中…" : "打开记忆视图");
        b.CloseElement();

        b.CloseElement();

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

    async Task RunAsync(Func<Task<string>> action)
    {
        if (busy) return;
        busy = true; status = ""; StateHasChanged();
        try { status = await action(); }
        catch (Exception e) { status = "出错：" + e.Message; }
        finally { busy = false; StateHasChanged(); }
    }
}
