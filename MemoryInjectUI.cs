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
.mi-settings.mi-settings--miami{color-scheme:dark;--mi-bg:#201a32;--mi-card:rgba(43,33,57,.93);--mi-text:#fff1e8;--mi-muted:#cfb6ca;--mi-line:rgba(255,159,192,.22);--mi-accent:#ff9abd;--mi-on-accent:#352039;--mi-tint:#463047;--mi-shadow:0 12px 32px rgba(20,10,35,.22),inset 0 1px 0 rgba(255,207,191,.08);background:radial-gradient(ellipse at 95% 0%,rgba(255,126,165,.3),transparent 48%),radial-gradient(ellipse at 0% 90%,rgba(56,196,188,.2),transparent 45%),linear-gradient(145deg,#38213e,#201a32 65%,#163b40)}
.mi-settings--miami .mi-brand{color:#83e3d8;background:linear-gradient(145deg,#6b335e,#25484e);box-shadow:0 0 22px rgba(255,128,179,.18)}
.mi-settings .mi-miami-motto{display:none}
.mi-settings--miami .mi-miami-motto{display:flex;align-items:center;justify-content:center;gap:18px;margin:0 0 22px;padding:21px 12px;border-block:1px solid rgba(255,170,183,.26);border-radius:12px;background:radial-gradient(ellipse at 50% 130%,rgba(255,155,105,.23),transparent 70%),linear-gradient(100deg,rgba(255,145,189,.06),rgba(255,184,145,.12),rgba(100,225,212,.07));color:#ffd6ba;text-align:center;font-family:"Segoe Script","Brush Script MT",cursive;font-size:clamp(19px,3vw,29px);line-height:1.7;text-shadow:0 2px 18px rgba(255,143,174,.3)}
.mi-settings--miami .mi-miami-motto::before,.mi-settings--miami .mi-miami-motto::after{content:"❧";color:#83e3d8;font:28px Georgia,serif}
.mi-settings--miami .mi-miami-motto::after{transform:rotate(180deg)}
.mi-settings--miami .mi-config-heading h3{color:#83e3d8}
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

/* Miami symbols and translucent panels */
.mi-settings.mi-settings--miami{--mi-card:rgba(83,61,94,.32);--mi-line:rgba(255,220,230,.24);--mi-tint:rgba(84,45,76,.38);--mi-shadow:0 12px 32px rgba(16,10,30,.2),inset 0 1px 0 rgba(255,235,229,.16);background-color:#42344f;background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%221600%22%20height%3D%221000%22%20viewBox%3D%220%200%201600%201000%22%20fill%3D%22none%22%3E%0A%20%20%3Cdefs%3E%0A%20%20%20%20%3ClinearGradient%20id%3D%22sun%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%0A%20%20%20%20%20%20%3Cstop%20stop-color%3D%22%23ffe88a%22%2F%3E%3Cstop%20offset%3D%22.28%22%20stop-color%3D%22%23ffae45%22%2F%3E%3Cstop%20offset%3D%22.52%22%20stop-color%3D%22%23ff6a5c%22%2F%3E%3Cstop%20offset%3D%22.78%22%20stop-color%3D%22%23ff347f%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23e23eff%22%2F%3E%0A%20%20%20%20%3C%2FlinearGradient%3E%0A%20%20%20%20%3CradialGradient%20id%3D%22halo%22%3E%0A%20%20%20%20%20%20%3Cstop%20stop-color%3D%22%23ff6b52%22%20stop-opacity%3D%22.5%22%2F%3E%3Cstop%20offset%3D%22.55%22%20stop-color%3D%22%23ff388e%22%20stop-opacity%3D%22.25%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23ab39ff%22%20stop-opacity%3D%220%22%2F%3E%0A%20%20%20%20%3C%2FradialGradient%3E%0A%20%20%20%20%3Cmask%20id%3D%22bands%22%3E%0A%20%20%20%20%20%20%3Ccircle%20cx%3D%22800%22%20cy%3D%22500%22%20r%3D%22220%22%20fill%3D%22white%22%2F%3E%0A%20%20%20%20%20%20%3Cpath%20d%3D%22M550%20538H1050M550%20575H1050M550%20619H1050M550%20671H1050%22%20stroke%3D%22black%22%20stroke-width%3D%2215%22%2F%3E%0A%20%20%20%20%3C%2Fmask%3E%0A%20%20%3C%2Fdefs%3E%0A%20%20%3C%21--%20Just%20a%20sunset%20disc%2C%20palm%20gestures%2C%20and%20a%20coupe%20outline%3B%20no%20scenic%20detail.%20--%3E%0A%20%20%3Ccircle%20cx%3D%22800%22%20cy%3D%22500%22%20r%3D%22355%22%20fill%3D%22url%28%23halo%29%22%2F%3E%0A%20%20%3Ccircle%20cx%3D%22800%22%20cy%3D%22500%22%20r%3D%22220%22%20fill%3D%22url%28%23sun%29%22%20mask%3D%22url%28%23bands%29%22%20opacity%3D%22.95%22%2F%3E%0A%20%20%3Cg%20stroke%3D%22%23ff8bb7%22%20stroke-width%3D%223%22%20stroke-linecap%3D%22round%22%20opacity%3D%22.4%22%3E%0A%20%20%20%20%3Cpath%20d%3D%22M470%20744H1130M605%20764H995M710%20784H890%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Cg%20stroke%3D%22%2364ffe0%22%20stroke-width%3D%225%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%20opacity%3D%22.52%22%3E%0A%20%20%20%20%3Cpath%20d%3D%22M180%20770Q219%20576%20177%20370M177%20370Q114%20295%2048%20344M177%20370Q81%20352%2053%20418M177%20370Q127%20382%20103%20453M177%20370Q204%20282%20272%20302M177%20370Q261%20326%20314%20376M177%20370Q264%20379%20275%20437%22%2F%3E%0A%20%20%20%20%3Cpath%20d%3D%22M1438%20710Q1400%20580%201440%20474M1440%20474Q1379%20410%201330%20451M1440%20474Q1375%20466%201355%20513M1440%20474Q1463%20404%201510%20428M1440%20474Q1505%20447%201552%20489M1440%20474Q1507%20493%201502%20540%22%20stroke-width%3D%224%22%20opacity%3D%22.65%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Cg%20stroke%3D%22%23ff9fdb%22%20stroke-width%3D%224%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%20opacity%3D%22.6%22%3E%0A%20%20%20%20%3Cpath%20d%3D%22M962%20811L997%20779L1128%20763L1180%20720H1294L1347%20763L1437%20781L1450%20823H1413M1018%20823H962V811M1090%20823H1341M1152%20763H1323L1286%20734H1188Z%22%2F%3E%0A%20%20%20%20%3Ccircle%20cx%3D%221054%22%20cy%3D%22820%22%20r%3D%2234%22%2F%3E%3Ccircle%20cx%3D%221377%22%20cy%3D%22820%22%20r%3D%2234%22%2F%3E%0A%20%20%20%20%3Cpath%20d%3D%22M976%20797H1007M1407%20797H1434M1125%20785H1321%22%20stroke%3D%22%2364ffe0%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%20%20%3Cg%20stroke%3D%22%23ffd4aa%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20opacity%3D%22.38%22%3E%0A%20%20%20%20%3Cpath%20d%3D%22M487%20254V278M475%20266H499M864%20157V173M856%20165H872M683%20652V668M675%20660H691%22%2F%3E%0A%20%20%3C%2Fg%3E%0A%3C%2Fsvg%3E%0A"),radial-gradient(ellipse at 85% 12%,rgba(211,79,125,.27),transparent 55%),linear-gradient(145deg,#604368,#423650 70%,#365b60);background-size:100% auto,100% 100%,100% 100%;background-position:center 70px,center,center;background-repeat:no-repeat}
.mi-settings--miami :is(.mi-card,.mi-config){-webkit-backdrop-filter:blur(4px) saturate(125%);backdrop-filter:blur(4px) saturate(125%);box-shadow:var(--mi-shadow)}
.mi-settings--miami :is(.mi-heading,.mi-stats,.mi-config-heading,label){text-shadow:0 1px 5px rgba(15,10,28,.55)}
.mi-settings--miami .mi-miami-motto{background:rgba(49,28,50,.17);-webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px)}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){.mi-settings--miami :is(.mi-card,.mi-config){background:rgba(28,23,43,.78)}}

.mi-settings .mi-working::before{content:"";display:inline-block;width:10px;height:10px;margin-right:7px;border:1.5px solid currentColor;border-right-color:transparent;border-radius:50%;animation:mi-spin .8s linear infinite;vertical-align:-1px}
@keyframes mi-enter{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@keyframes mi-spin{to{transform:rotate(360deg)}}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){.mi-settings--glass .mi-card,.mi-settings--glass .mi-config{background:#eef4fc}}
/* RetroMem cassette settings */
.mi-settings--miami .mi-card,.mi-settings--miami .mi-config{border-radius:5px 18px 5px 5px;border-color:rgba(255,163,203,.3);box-shadow:inset 0 1px 0 rgba(255,223,209,.2),inset 0 -1px 0 rgba(131,227,216,.12),0 12px 28px rgba(12,8,25,.22)}
.mi-settings--miami .mi-status-heading,.mi-settings--miami .mi-eyebrow,.mi-settings--miami .mi-stats{font-family:Consolas,"Cascadia Mono",monospace;letter-spacing:.04em}
.mi-settings--miami .mi-status-heading{padding-bottom:10px;border-bottom:1px dashed rgba(131,227,216,.25)}
.mi-settings--miami .mi-card{background-image:repeating-linear-gradient(0deg,transparent 0 4px,rgba(255,255,255,.025) 4px 5px)}
.mi-settings--miami .mi-config-heading{border-bottom-style:dashed}
.mi-settings--miami .mi-maintenance{border-top:1px dashed rgba(131,227,216,.3)}
.mi-settings--miami .mi-maintenance summary{font-family:Consolas,"Cascadia Mono",monospace;letter-spacing:.025em}
.mi-settings--miami button:where(.mi-primary,.mi-secondary,.mi-soft){border-radius:5px;box-shadow:inset 0 1px 0 rgba(255,255,255,.15),0 3px 0 rgba(9,7,20,.28)}
.mi-settings--miami button.mi-primary{background:#a83d70;color:#fff1e8!important;border-color:rgba(255,154,189,.65)}
.mi-settings--miami button:where(.mi-primary,.mi-secondary,.mi-soft):active:enabled{transform:translateY(2px);box-shadow:inset 0 1px 5px rgba(9,7,20,.25)}
.mi-settings--miami .mi-config .ant-switch{border:1px solid rgba(255,207,191,.35);box-shadow:inset 0 2px 4px rgba(10,8,22,.4)}
.mi-settings--miami .mi-working::after{content:"";display:inline-block;width:10px;height:10px;margin-left:7px;border:1.5px dashed currentColor;border-radius:50%;vertical-align:-1px;animation:mi-spin 1.2s linear infinite}
.mi-settings--miami .mi-feedback{border-left-color:#83e3d8;animation:mi-enter .2s ease-out}

/* Miami: override framework light-form text inside the dark settings panel. */
.mi-settings--miami .mi-config{--ant-text-color:var(--mi-text);--ant-text-color-secondary:var(--mi-muted);--ant-heading-color:var(--mi-text);--ant-disabled-color:#b9a8bc}
.mi-settings--miami .mi-config :is(.ant-form,.ant-form-item,.ant-form-item-label>label,.ant-form-item-control-input-content,.ant-checkbox-wrapper,.ant-radio-wrapper,.ant-select-selection-item,.ant-select-selection-search-input,.ant-input-number-input,.ant-input,.ant-input-group-addon,.ant-collapse-header-text,.ant-typography){color:var(--mi-text)!important}
.mi-settings--miami .mi-config :is(.ant-form-item-label label,.ant-checkbox-wrapper,.ant-radio-wrapper,.ant-typography,.ant-collapse-header-text) :is(span,strong,em){color:inherit!important}
.mi-settings--miami .mi-config :is(.ant-form-item-extra,.ant-form-item-explain,.ant-typography-secondary,.ant-input-number-handler,.ant-select-arrow){color:var(--mi-muted)!important}
.mi-settings--miami .mi-config :is(.ant-form-item-extra,.ant-form-item-explain,.ant-typography-secondary) :is(span,p,div){color:inherit!important}
/* Framework config descriptions also use an inline light-theme color outside Ant extra/help. */
.mi-settings--miami .mi-config :is([style*="color: rgba(0,0,0,0.45)"],[style*="color:rgba(0,0,0,0.45)"],[style*="color: rgba(0, 0, 0, 0.45)"],[style*="color:rgba(0, 0, 0, 0.45)"]){color:var(--mi-muted)!important}
.mi-settings--miami .mi-config :is(input,textarea)::placeholder{color:var(--mi-muted);opacity:1}
.mi-settings--miami .mi-config select,.mi-settings--miami .mi-config select option{color:var(--mi-text);background:#201a32}

/* Keep the framework form mounted once; collapse only tuning rows. */
.mi-settings .mi-advanced-toggle{display:flex;align-items:center;gap:9px;margin:16px 0;padding:10px 14px;border:1px solid var(--mi-line);border-radius:10px;background:var(--mi-tint);color:var(--mi-text);font:inherit;cursor:pointer;text-align:left}
.mi-settings .mi-advanced-toggle:hover{border-color:var(--mi-accent)}
.mi-settings .mi-advanced-toggle::before{content:"▸";color:var(--mi-accent);transition:transform .18s}
.mi-settings .mi-advanced-toggle[aria-expanded="true"]::before{transform:rotate(90deg)}
.mi-settings .mi-config-fields--basic .ant-form-item:not(:has(.ant-switch,.ant-checkbox,input[type="checkbox"])){display:none}
.mi-settings .mi-config-fields--expanded .ant-form-item:not(:has(.ant-switch,.ant-checkbox,input[type="checkbox"])){border-left:2px solid var(--mi-line);padding-left:14px}

/* RetroMem minimal settings baseline */
.mi-settings:not(.mi-settings--miami){padding:28px;border-radius:22px}
.mi-settings:not(.mi-settings--miami) .mi-heading{margin-bottom:25px;gap:14px}
.mi-settings:not(.mi-settings--miami) .mi-brand{width:44px;height:44px;border-radius:13px;box-shadow:inset 0 1px 0 color-mix(in srgb,var(--mi-card) 80%,transparent)}
.mi-settings:not(.mi-settings--miami) h2{font-size:24px;letter-spacing:-.025em;font-weight:650}
.mi-settings:not(.mi-settings--miami) .mi-eyebrow{font-size:10px;letter-spacing:.18em;margin-bottom:2px}
.mi-settings:not(.mi-settings--miami) :is(.mi-card,.mi-config){padding:22px;border-radius:17px;border-color:var(--mi-line);box-shadow:var(--mi-shadow)}
.mi-settings:not(.mi-settings--miami) .mi-status-heading{padding-bottom:12px;margin-bottom:13px;border-bottom:1px solid var(--mi-line);font-size:11px}
.mi-settings:not(.mi-settings--miami) .mi-stats{padding:14px 16px;border:1px solid var(--mi-line);border-radius:11px;background:color-mix(in srgb,var(--mi-tint) 45%,transparent);font-size:12px;line-height:1.9}
.mi-settings:not(.mi-settings--miami) .mi-toolbar{margin-top:16px;gap:10px}
.mi-settings:not(.mi-settings--miami) h3{font-size:15px;font-weight:650;margin-bottom:6px}
.mi-settings:not(.mi-settings--miami) :is(button.mi-primary,button.mi-soft,button.mi-secondary,input.mi-query,select.mi-theme-select){border-radius:9px;min-height:38px}
.mi-settings:not(.mi-settings--miami) .mi-primary{box-shadow:0 3px 10px color-mix(in srgb,var(--mi-accent) 15%,transparent)}
.mi-settings:not(.mi-settings--miami) .mi-secondary{background:transparent}
.mi-settings:not(.mi-settings--miami) .mi-maintenance{margin-top:22px;padding-top:17px}
.mi-settings:not(.mi-settings--miami) .mi-maintenance summary{font-size:12px;line-height:1.9}
.mi-settings:not(.mi-settings--miami) .mi-config-heading{padding-bottom:18px;margin-bottom:24px}
.mi-settings:not(.mi-settings--miami) .mi-config .ant-form-item{padding-bottom:20px;margin-bottom:20px}
.mi-settings:not(.mi-settings--miami) .mi-feedback{border-left-width:3px;background:color-mix(in srgb,var(--mi-accent) 3%,var(--mi-card))}
.mi-settings.mi-settings--glass .mi-stats{background:rgba(255,255,255,.2);border-color:rgba(255,255,255,.6)}
@media(max-width:640px){.mi-settings:not(.mi-settings--miami){padding:15px;margin:8px 0;border-radius:17px}.mi-settings:not(.mi-settings--miami) :is(.mi-card,.mi-config){padding:17px}.mi-settings:not(.mi-settings--miami) h2{font-size:21px}}

@media(prefers-reduced-motion:reduce){.mi-settings *{animation:none!important;transition:none!important}}
@media(max-width:640px){.mi-settings{padding:14px;margin:8px 0;border-radius:16px}.mi-settings .mi-card,.mi-settings .mi-config{padding:15px}.mi-settings .mi-appearance{width:100%;justify-content:flex-end}.mi-settings .mi-toolbar input.mi-query{flex-basis:100%}}
""";

    static readonly string[] themes = ["native", "light", "ocean", "forest", "dark", "glass", "miami"];
    static readonly string[] themeLabels = ["原生 · 黑白紫", "暖白 · 鸢尾", "海洋 · 青蓝", "森林 · 松绿", "暗夜 · 紫灰", "玻璃 · 晨雾", "迈阿密 · 落日"];

    bool advancedExpanded;
    bool busy;
    string status = "";
    string stats = "状态读取中…";
    string query = "";
    string view = "";
    string theme = "miami";
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
        string selected = args.Value?.ToString() ?? "miami";
        theme = Array.IndexOf(themes, selected) >= 0 ? selected : "miami";
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
        b.AddAttribute(i++, "aria-label", "Retromem 溯忆插件设置");

        b.OpenElement(i++, "header"); b.AddAttribute(i++, "class", "mi-heading");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-brand"); b.AddAttribute(i++, "aria-hidden", "true"); b.AddContent(i++, "◇"); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-heading-copy");
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-eyebrow"); b.AddContent(i++, "Retromem 溯忆"); b.CloseElement();
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

        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-miami-motto");
        b.AddAttribute(i++, "lang", "en"); b.AddContent(i++, "Like old time passing by."); b.CloseElement();

        b.OpenElement(i++, "section"); b.AddAttribute(i++, "class", "mi-card");
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-status-heading");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", Configuration.Enable ? "mi-indicator" : "mi-indicator mi-indicator-off"); b.AddAttribute(i++, "aria-hidden", "true"); b.CloseElement();
        b.AddContent(i++, Configuration.Enable ? "插件已启用" : "插件已停用");
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-refresh"); b.AddContent(i++, "状态每 5 秒刷新"); b.CloseElement();
        b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-stats"); b.AddContent(i++, stats); b.CloseElement();
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-toolbar");
        b.OpenElement(i++, "button"); b.AddAttribute(i++, "type", "button"); b.AddAttribute(i++, "class", busy ? "mi-primary mi-working" : "mi-primary"); b.AddAttribute(i++, "disabled", busy); b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this, OpenGraphPage)); b.AddContent(i++, "打开管理页面 ↗"); b.CloseElement();
        b.OpenElement(i++, "span"); b.AddAttribute(i++, "class", "mi-note"); b.AddContent(i++, ThemeCopy("图谱 · 事实卡 · 画像 · 审计", "Neon Graph / Echo Audit")); b.CloseElement();
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
        b.OpenElement(i++, "summary"); b.AddContent(i++, ThemeCopy("索引与维护 · 预览、重建、压缩、审计", "System Deck & Rollback")); b.CloseElement();
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
        b.AddContent(i++, ThemeCopy("审计一次", "Echo Audit"));
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
        b.OpenElement(i++, "div"); b.AddAttribute(i++, "class", "mi-note"); b.AddContent(i++, "本次升级会开启语义检索并保存配置；之后可自行关闭。自动压缩保留选择。"); b.CloseElement();
        b.CloseElement();
        b.OpenElement(i++, "p"); b.AddAttribute(i++, "class", "mi-note");
        b.AddContent(i++, "日常只需选择功能开关。阈值、间隔、目录和处理预算保持默认即可，需要时再展开调节。");
        b.CloseElement();
        b.OpenElement(i++, "button");
        b.AddAttribute(i++, "type", "button");
        b.AddAttribute(i++, "class", "mi-advanced-toggle");
        b.AddAttribute(i++, "aria-expanded", advancedExpanded ? "true" : "false");
        b.AddAttribute(i++, "aria-controls", "mi-config-fields");
        b.AddAttribute(i++, "onclick", EventCallback.Factory.Create<MouseEventArgs>(this,
            () => { advancedExpanded = !advancedExpanded; }));
        b.AddContent(i++, advancedExpanded ? "收起高级调节" : "展开高级调节 · 阈值、间隔与目录");
        b.CloseElement();
        b.OpenElement(i++, "div");
        b.AddAttribute(i++, "id", "mi-config-fields");
        b.AddAttribute(i++, "class", "mi-config-fields " + (advancedExpanded ? "mi-config-fields--expanded" : "mi-config-fields--basic"));
        b.AddContent(i++, DefaultUI);
        b.CloseElement();
        b.CloseElement();

        b.CloseElement();
    }

    string ThemeCopy(string standard, string retro) => theme == "miami" ? retro + " · " + standard : standard;

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
