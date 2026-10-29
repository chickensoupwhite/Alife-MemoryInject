using System;
using System.Threading;
using System.Threading.Tasks;
using Alife.Framework;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Rendering;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.JSInterop;

namespace Mao.MemoryRecall;

/// <summary>
/// Mao.MemoryInject —— 插件配置页（手写 C# 组件，非 razor）。
/// 顶部：状态行（自动刷新）+ 索引浏览 + 操作按钮；
/// 下方必须渲染框架自动生成的配置表单（DefaultUI），否则配置项不会显示。
/// </summary>
public class MemoryInjectUI : ModuleUIBase<MemoryInjectModule, MemoryInjectConfig>, IDisposable
{
    [Inject]
    public IJSRuntime JavaScript { get; set; } = default!;

    const string settingsStyles = """
.mi-settings{--mi-bg:#f5f4f7;--mi-card:#fff;--mi-text:#242126;--mi-muted:#746c80;--mi-line:rgba(48,35,62,.13);--mi-accent:#7c3aed;--mi-on-accent:#fff;--mi-tint:#f2ecfc;--mi-shadow:0 10px 30px rgba(35,25,45,.06);position:relative;max-width:1120px;margin:12px auto;padding:24px;border:1px solid var(--mi-line);border-radius:22px;background:var(--mi-bg);color:var(--mi-text);font-family:"Segoe UI Variable","Segoe UI","Microsoft YaHei",sans-serif;font-size:13px;line-height:1.65}
.mi-settings--light{--mi-bg:#f5f1e9;--mi-card:#fffdf8;--mi-text:#382d42;--mi-muted:#7a6e83;--mi-accent:#8462be;--mi-tint:#f0eaf9}
.mi-settings--ocean{--mi-bg:#eaf4f7;--mi-card:#f9fdff;--mi-text:#214451;--mi-muted:#607f8c;--mi-accent:#2784a1;--mi-tint:#e4f2f6}
.mi-settings--forest{--mi-bg:#edf4ea;--mi-card:#fbfdf9;--mi-text:#30452a;--mi-muted:#6b7e62;--mi-accent:#548149;--mi-tint:#e9f2e5}
.mi-settings--dark{color-scheme:dark;--mi-bg:#191725;--mi-card:#242131;--mi-text:#eeeaf5;--mi-muted:#b0a5c2;--mi-line:rgba(201,184,230,.15);--mi-accent:#bba1f0;--mi-on-accent:#241d36;--mi-tint:#30283f;--mi-shadow:0 10px 30px rgba(0,0,0,.16)}
.mi-settings--glass{--mi-bg:#dfebf8;--mi-card:rgba(255,255,255,.36);--mi-text:#253b58;--mi-muted:#526a86;--mi-line:rgba(255,255,255,.7);--mi-accent:#4969ce;--mi-tint:rgba(241,248,255,.38);--mi-shadow:0 12px 30px rgba(49,75,126,.1),inset 0 1px 0 rgba(255,255,255,.8);background:radial-gradient(ellipse at 0% 0%,rgba(130,161,235,.65),transparent 55%),radial-gradient(ellipse at 100% 35%,rgba(175,214,224,.7),transparent 60%),linear-gradient(135deg,#e6eafa,#d8e7f4)}
.mi-settings{color-scheme:light}
.mi-settings--dark{color-scheme:dark}
.mi-settings *{box-sizing:border-box}
.mi-settings .mi-heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.mi-settings .mi-brand{display:grid;place-items:center;width:42px;height:42px;border-radius:14px;color:var(--mi-accent);font-size:23px;background:var(--mi-tint);border:1px solid var(--mi-line);font-weight:600}
.mi-settings .mi-heading-copy{flex:1;min-width:180px}
.mi-settings .mi-eyebrow{font-size:9px;letter-spacing:.2em;font-weight:700;color:var(--mi-muted)}
.mi-settings h2{font-size:21px;letter-spacing:.015em;margin:0;color:var(--mi-text);line-height:1.5}
.mi-settings h3{font-size:14px;margin:0 0 5px;color:var(--mi-text);font-weight:650}
.mi-settings .mi-subtitle,.mi-settings .mi-note{color:var(--mi-muted);font-size:12px}
.mi-settings .mi-appearance{display:flex;align-items:center;gap:8px;color:var(--mi-muted);font-size:12px}
.mi-settings .mi-card,.mi-settings .mi-config{padding:19px;margin-bottom:15px;background:var(--mi-card);border:1px solid var(--mi-line);border-radius:15px;box-shadow:var(--mi-shadow)}
.mi-settings .mi-status-heading{display:flex;align-items:center;gap:8px;margin-bottom:9px;font-size:11px;color:var(--mi-muted)}
.mi-settings .mi-indicator{width:7px;height:7px;border-radius:50%;background:#389d85;box-shadow:0 0 0 4px rgba(56,157,133,.09)}
.mi-settings .mi-indicator-off{background:#98a1b1;box-shadow:0 0 0 4px rgba(120,135,160,.09)}
.mi-settings .mi-refresh{margin-left:auto;font-variant-numeric:tabular-nums}
.mi-settings .mi-stats{font-size:12px;line-height:1.9;color:var(--mi-text);overflow-wrap:anywhere}
.mi-settings .mi-toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-top:13px}
.mi-settings input.mi-query{flex:1;min-width:180px}
.mi-settings button:where(.mi-primary,.mi-soft,.mi-secondary),.mi-settings input.mi-query,.mi-settings select.mi-theme-select{font:inherit;border:1px solid var(--mi-line);border-radius:9px;background:var(--mi-card);color:var(--mi-text);padding:8px 12px;min-height:36px;transition:background-color .18s,box-shadow .18s,border-color .18s,transform .18s}
.mi-settings button:where(.mi-primary,.mi-soft,.mi-secondary){cursor:pointer;white-space:nowrap;font-size:12px}
.mi-settings button:where(.mi-primary,.mi-soft,.mi-secondary):hover:enabled{border-color:var(--mi-accent);transform:translateY(-1px);box-shadow:0 4px 12px color-mix(in srgb,var(--mi-accent) 12%,transparent)}
.mi-settings button:where(.mi-primary,.mi-soft,.mi-secondary):active:enabled{transform:translateY(0);box-shadow:none}
.mi-settings button:where(.mi-primary,.mi-soft,.mi-secondary):disabled{opacity:.5;cursor:wait}
.mi-settings .mi-primary{background:var(--mi-accent);border-color:transparent;color:var(--mi-on-accent);font-weight:600;padding-inline:16px}
.mi-settings .mi-soft{background:var(--mi-tint);color:var(--mi-accent)}
.mi-settings :is(button,input,select,summary):focus-visible{outline:2px solid var(--mi-accent);outline-offset:3px}
.mi-settings select.mi-theme-select{font-size:12px;cursor:pointer;padding:7px 9px;max-width:150px}
.mi-settings select.mi-theme-select option{background:var(--mi-bg);color:var(--mi-text)}
.mi-settings .mi-maintenance{margin-top:18px;padding-top:14px;border-top:1px solid var(--mi-line)}
.mi-settings .mi-maintenance summary{cursor:pointer;font-size:12px;color:var(--mi-muted);user-select:none}
.mi-settings .mi-maintenance[open] summary{color:var(--mi-accent)}
.mi-settings .mi-maintenance .mi-toolbar{gap:8px}
.mi-settings .mi-output{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.8;max-height:340px;overflow:auto;animation:mi-enter .18s ease-out}
.mi-settings .mi-feedback{border-left:3px solid var(--mi-accent)}
.mi-settings .mi-config{margin-bottom:0;padding:20px}
.mi-settings .mi-config-heading{margin-bottom:20px;padding-bottom:15px;border-bottom:1px solid var(--mi-line)}
.mi-settings .mi-config :is(.ant-form-item-label>label,label){color:var(--mi-text)}
.mi-settings .mi-config :is(.ant-form-item-explain,.ant-form-item-extra,.ant-typography-secondary){color:var(--mi-muted);font-size:12px;line-height:1.75}
.mi-settings .mi-config .ant-form-item{padding-bottom:16px;margin-bottom:16px;border-bottom:1px solid var(--mi-line)}
.mi-settings .mi-config :is(.ant-input,.ant-input-number,.ant-select-selector,.ant-input-affix-wrapper){border-color:var(--mi-line);border-radius:8px;background:var(--mi-card);color:var(--mi-text)}
.mi-settings .mi-config :is(.ant-input-number-input,.ant-select-selection-item){color:var(--mi-text)}
.mi-settings .mi-config .ant-select-arrow{color:var(--mi-muted)}
.mi-settings .mi-config :is(.ant-switch-checked,.ant-checkbox-checked .ant-checkbox-inner){background:var(--mi-accent);border-color:var(--mi-accent)}
.mi-settings .mi-config :is(.ant-collapse,.ant-collapse-content){background:transparent;border-color:var(--mi-line)}
.mi-settings .mi-config :is(.ant-collapse-header,.ant-typography){color:var(--mi-text)}
.mi-settings--glass .mi-card,.mi-settings--glass .mi-config{backdrop-filter:blur(24px) saturate(140%);-webkit-backdrop-filter:blur(24px) saturate(140%)}
.mi-settings .mi-working::before{content:"";display:inline-block;width:10px;height:10px;margin-right:7px;border:1.5px solid currentColor;border-right-color:transparent;border-radius:50%;animation:mi-spin .8s linear infinite;vertical-align:-1px}
@keyframes mi-enter{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@keyframes mi-spin{to{transform:rotate(360deg)}}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){.mi-settings--glass .mi-card,.mi-settings--glass .mi-config{background:#eef4fc}}
@media(prefers-reduced-motion:reduce){.mi-settings *{animation:none!important;transition:none!important}}
@media(max-width:640px){.mi-settings{padding:14px;margin:8px 0;border-radius:16px}.mi-settings .mi-card,.mi-settings .mi-config{padding:15px}.mi-settings .mi-appearance{width:100%;justify-content:flex-end}.mi-settings .mi-toolbar input.mi-query{flex-basis:100%}}
""";

    static readonly string[] themes = ["native", "light", "ocean", "forest", "dark", "glass"];
    static readonly string[] themeLabels = ["原生 · 黑白紫", "暖白 · 鸢尾", "海洋 · 青蓝", "森林 · 松绿", "暗夜 · 紫灰", "玻璃 · 晨雾"];

    bool busy;
    string status = "";
    string stats = "状态读取中…";
    string query = "";
    string view = "";
    string theme = "glass";
    Timer? refreshTimer;
    volatile bool refreshing;   // 防 5 秒定时器与慢查询重叠

    protected override void OnInitialized()
    {
        base.OnInitialized();
        // ⑥ 自动刷新：每 5 秒拉一次状态（只读，很轻）
        refreshTimer = new Timer(_ => _ = InvokeAsync(async () =>
        {
            if (refreshing) return;   // 上一次还没回就跳过，别堆请求
            refreshing = true;
            try
            {
                if (Module == null) return;
                stats = await Module.StatsAsync();
                StateHasChanged();
            }
            catch { }
            finally { refreshing = false; }
        }), null, 800, 5000);
    }

    public void Dispose()
    {
        refreshTimer?.Dispose();
        refreshTimer = null;
    }

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        await base.OnAfterRenderAsync(firstRender);
        if (firstRender == false) return;
        try
        {
            string? saved = await JavaScript.InvokeAsync<string?>("localStorage.getItem", "memoryinject-settings-theme");
            if (saved != null && Array.IndexOf(themes, saved) >= 0)
            {
                theme = saved;
                StateHasChanged();
            }
        }
        catch (JSException)
        {
            status = "当前环境无法读取外观偏好，已使用默认主题。";
            StateHasChanged();
        }
    }

    async Task changeTheme(ChangeEventArgs args)
    {
        string selected = args.Value?.ToString() ?? "glass";
        theme = Array.IndexOf(themes, selected) >= 0 ? selected : "glass";
        try { await JavaScript.InvokeVoidAsync("localStorage.setItem", "memoryinject-settings-theme", theme); }
        catch (JSException) { status = "外观已应用，但当前环境未能保存偏好。"; }
    }

    protected override void BuildRenderTree(RenderTreeBuilder b)
    {
        base.BuildRenderTree(b);
        if (Configuration == null) { b.AddContent(0, "配置未加载"); return; }
        int i = 0;

        b.OpenElement(i++, "style"); b.AddMarkupContent(i++, settingsStyles); b.CloseElement();
        b.OpenElement(i++, "section");
        b.AddAttribute(i++, "class", "mi-settings mi-settings--" + theme);
        b.AddAttribute(i++, "aria-label", "MemoryInject 插件设置");

        b.OpenElement(i++, "header"); b.AddAttribute(i++, "class", "mi-heading");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-brand"); b.AddAttribute(i++, "aria-hidden", "true"); b.AddContent(i++, "◇"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-heading-copy");
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-eyebrow"); b.AddContent(i++, "MEMORY INJECT"); b.CloseElement();
        b.OpenElement(i++, "h2"); b.AddContent(i++, "记忆管理"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-subtitle"); b.AddContent(i++, "事实卡、关系与召回，在同一处管理。"); b.CloseElement();
        b.CloseElement();
        b.OpenElement(i++, "label"); b.AddAttribute(i++, "class", "mi-appearance");
        b.AddContent(i++, "外观");
        b.OpenElement(i++, "select"); b.AddAttribute(i++, "class", "mi-theme-select"); b.AddAttribute(i++, "aria-label", "插件设置外观主题");
        b.AddAttribute(i++, "value", theme);
        b.AddAttribute(i++, "onchange", EventCallback.Factory.Create<ChangeEventArgs>(this, changeTheme));
        for (int themeIndex = 0; themeIndex < themes.Length; themeIndex++)
        {
            b.OpenElement(i++, "option"); b.AddAttribute(i++, "value", themes[themeIndex]); b.AddContent(i++, themeLabels[themeIndex]); b.CloseElement();
        }
        b.CloseElement(); b.CloseElement(); b.CloseElement();

        b.OpenElement(i++, "section"); b.AddAttribute(i++, "class", "mi-card");
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-status-heading");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", Configuration.Enable ? "mi-indicator" : "mi-indicator mi-indicator-off"); b.AddAttribute(i++, "aria-hidden", "true"); b.CloseElement();
        b.AddContent(i++, Configuration.Enable ? "插件已启用" : "插件已停用");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-refresh"); b.AddContent(i++, "状态每 5 秒刷新"); b.CloseElement();
        b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-stats"); b.AddContent(i++, stats); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-toolbar");
        b.OpenElement(i++, "button"); b.AddAttribute(i++, "type", "button"); b.AddAttribute(i++, "class", busy ? "mi-primary mi-working" : "mi-primary"); b.AddAttribute(i++, "disabled", busy); b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this, OpenGraphPage)); b.AddContent(i++, "打开管理页面 ↗"); b.CloseElement();
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-note"); b.AddContent(i++, "图谱 · 事实卡 · 画像 · 审计"); b.CloseElement();
        b.CloseElement(); b.CloseElement();

        b.OpenElement(i++, "section"); b.AddAttribute(i++, "class", "mi-card");
        b.OpenElement(i++, "h3"); b.AddContent(i++, "索引浏览"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-note"); b.AddContent(i++, "快速查找已索引的记忆，留空查看最近条目。"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-toolbar");
        b.OpenElement(i++, "input");
        b.AddAttribute(i++, "type", "text");
        b.AddAttribute(i++, "placeholder", "关键词（留空 = 最近全部）");
        b.AddAttribute(i++, "value", query);
        b.AddAttribute(i++, "class", "mi-query");
        b.AddAttribute(i++, "aria-label", "索引关键词");
        b.AddAttribute(i++, "oninput", EventCallback.Factory.Create<ChangeEventArgs>(this, e => query = e.Value?.ToString() ?? ""));
        b.CloseElement();
        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", busy ? "mi-soft mi-working" : "mi-soft");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this, Browse));
        b.AddContent(i++, busy ? "处理中…" : "浏览索引");
        b.CloseElement();
        b.CloseElement();

        b.OpenElement(i++, "details"); b.AddAttribute(i++, "class", "mi-maintenance");
        b.OpenElement(i++, "summary"); b.AddContent(i++, "索引与维护 · 预览、重建、压缩、审计"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-toolbar");

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-secondary");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Task.FromResult(Module.PreviewBuild()))));
        b.AddContent(i++, "预览本次索引");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-secondary");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.RebuildIndexAsync())));
        b.AddContent(i++, "重建索引");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-secondary");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.RebuildCleanAsync())));
        b.AddContent(i++, "安全重建干净库");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-secondary");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.Compact())));
        b.AddContent(i++, "压缩一次");
        b.CloseElement();

        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-secondary");
        b.AddAttribute(i++, "disabled", busy);
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => RunAsync(() => Module == null ? Task.FromResult("模块未就绪") : Module.Audit())));
        b.AddContent(i++, "审计一次");
        b.CloseElement();

        b.CloseElement();
        b.CloseElement();
        b.CloseElement();

        // ④ 浏览结果
        if (!string.IsNullOrEmpty(view))
        {
            b.OpenElement(i++, "div");
            b.AddAttribute(i++, "class", "mi-card mi-output");
            b.AddContent(i++, view);
            b.CloseElement();
        }

        // ⑤ 操作结果
        if (!string.IsNullOrEmpty(status))
        {
            b.OpenElement(i++, "div");
            b.AddAttribute(i++, "class", "mi-card mi-output mi-feedback");
            b.AddAttribute(i++, "role", "status");
            b.AddContent(i++, status);
            b.CloseElement();
        }

        b.OpenElement(i++, "section"); b.AddAttribute(i++, "class", "mi-config");
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-config-heading");
        b.OpenElement(i++, "h3"); b.AddContent(i++, "插件配置"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-note"); b.AddContent(i++, "调整召回策略、索引与模型行为。"); b.CloseElement();
        b.CloseElement();
        b.AddContent(i++, DefaultUI);
        b.CloseElement();

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

    async Task OpenGraphPage()
    {
        if (Module == null) { status = "模块未就绪"; StateHasChanged(); return; }
        busy = true; StateHasChanged();
        try { status = await Module.OpenGraphPageAsync(); }
        catch (Exception e) { status = "打开详情页失败：" + e.Message; }
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
