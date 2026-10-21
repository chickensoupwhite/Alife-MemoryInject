
using System.Collections.Generic;
using System.ComponentModel;

namespace Mao.MemoryRecall;

/// <summary>
/// Mao.MemoryInject —— 记忆回忆提示配置。
/// 平时照旧在后台检索记忆目录；命中相关旧账时不塞内容，只追加一句给 AI 自己的提示。
/// </summary>
public class MemoryInjectConfig
{
    [DisplayName("启用")]
    [Description("是否在后台检索记忆，并在命中时追加一句提示。")]
    public bool Enable { get; set; } = true;

    [DisplayName("使用语义检索")]
    [Description("默认关闭。开启后仅对关键词已命中的候选做语义重排；不会用向量单独制造命中。模型没就绪时退回关键词匹配。")]
    public bool UseSemantic { get; set; } = false;

    [DisplayName("Python 解释器")]
    [Description("留空使用 PATH 中的 python。需已安装 torch 与 transformers。建议写绝对路径。")]
    public string PythonExe { get; set; } = "";

    [DisplayName("记忆目录（相对角色目录）")]
    [Description("记忆文件所在位置，相对角色目录。默认 Storage/Memory。")]
    public string MemorySubPath { get; set; } = "Storage/Memory";

    [DisplayName("附加记忆目录（每行一个）")]
    [Description("相对角色目录的附加来源；留空时自动尝试 Memory 与 Storage/Memory。来源只读，最终统一进入本角色索引。")]
    public string MemorySourcePaths { get; set; } = "";

    [DisplayName("最多提示几条")]
    [Description("提示里最多列出几个命中的记忆文件。建议 3。")]
    public int MaxResults { get; set; } = 3;

    [DisplayName("绝对相关度下限")]
    [Description("本地片段向量检索的余弦下限。建议 0.55。（侧车语义重排用的是下面单独的一项）")]
    public double MinScore { get; set; } = 0.55;

    [DisplayName("侧车语义重排阈值")]
    [Description("开启语义时，只有相似度达到这个值的候选才会得到最高 +25% 的排序加权；不会单独制造命中。建议 0.58。")]
    public double SemanticMinScore { get; set; } = 0.58;

    [DisplayName("相对相关度阈值（比平均高多少）")]
    [Description("语义模式：命中的片段要比本轮平均相似度高出这么多才算数。越大越严。建议 0.22。")]
    public double MinMargin { get; set; } = 0.22;

    [DisplayName("关键词模式分数阈值")]
    [Description("仅字面匹配降级时使用的阈值，建议 0.10~0.20。")]
    public double MinScoreKeyword { get; set; } = 0.15;

    [DisplayName("忽略最近几天内的文件")]
    [Description("只作用于自动书签：按条目的日期跳过最近 N 天（不是文件修改时间）；手动调 memrecall/memwhen 永不跳过。0 = 不忽略。默认 1。")]
    public int SkipRecentDays { get; set; } = 1;

    [DisplayName("排除的系统短语")]
    [Description("消息里含这些短语的圆括号段（如\"回复时请保持发言简洁…\"）整段剔除，不参与检索；裸出现的短语也一并抹掉。逗号分隔。")]
    public string QueryExcludePhrases { get; set; } = "回复时请保持发言简洁,注意遵守对话输出规范,对话输出规范,对话输出示例,通用工具要求,报点提示,禁用旁白,积极使用系统提供的图片,keep your response concise,keep responses concise,do not use emoji,follow the output format";

    [DisplayName("最短检索字符数")]
    [Description("消息去掉系统标签后不足这么多字就不检索（如「行」「嗯」这类应答）。建议 3。")]
    public int MinQueryChars { get; set; } = 3;

    [DisplayName("语义等待上限（毫秒）")]
    [Description("侧车就绪且空闲时，最多阻塞这么多毫秒等一次语义检索（常驻后单次约 10ms）。超时立刻退回关键词，绝不多等。默认 150。")]
    public int SyncWaitMs { get; set; } = 150;

    [DisplayName("连续超时后冷静几秒")]
    [Description("连续这么多次超时后，暂停语义检索一段时间，避免每句都卡。默认 5 次 / 60 秒。")]
    public int TimeoutStreakLimit { get; set; } = 5;

    [DisplayName("冷静时长（秒）")]
    [Description("触发冷静后暂停语义检索的秒数。默认 60。")]
    public int CooldownSeconds { get; set; } = 60;

    [DisplayName("触发高水位（字）")]
    [Description("攒够这么多有效字数才做一次检索；不到就不查，避免每句都触发、时而有时而无。建议 60。")]
    public int WaterHighChars { get; set; } = 60;

    [DisplayName("回落低水位（字）")]
    [Description("触发一次检索后，水位回落到这个值再重新攒。建议 0。")]
    public int WaterLowChars { get; set; } = 0;

    [DisplayName("两次触发最小间隔（秒）")]
    [Description("两次检索提示之间至少隔这么多秒，防止连发。建议 20。")]
    public int TriggerCooldownSeconds { get; set; } = 20;

    [DisplayName("衰减半衰期（天）")]
    [Description("记忆权重的半衰期：这么久没被命中过，它的排序权重就减半（命中会强化、把计时归零）。越大越念旧。建议 30。")]
    public int HalfLifeDays { get; set; } = 30;

    [DisplayName("每日维护间隔（小时）")]
    [Description("后台每隔这么久自动整理一次索引（衰减/清理）。建议 20（约一天一次）。")]
    public int MaintainIntervalHours { get; set; } = 20;

    [DisplayName("启用图谱兜底")]
    [Description("关键词、语义都没命中时，顺着人/事/物的关系链再扩一层找。关了更省、召回更保守。默认开。")]
    public bool UseGraph { get; set; } = true;

    [DisplayName("块粒度上限（字）")]
    [Description("一条记忆切成多大块：切得细召回准、切得粗索引小。改动后需重建索引。建议 200。")]
    public int BlockMaxChars { get; set; } = 200;

    [DisplayName("自动整理间隔（分钟）")]
    [Description("后台每隔这么久检查一次记忆目录：有了新记忆文件就自动并进索引，不用手动点重建。建议 10。")]
    public int AutoOrganizeMinutes { get; set; } = 10;

    [DisplayName("排除的记忆来源（每行一个）")]
    [Description("相对路径或目录前缀，例如 L1/ 排掉整个 L1；L1/xxx.txt 排掉单个文件。默认排除 L1/ 以避免首次索引过大。官方文件不会被删除，只是不进我们的索引。")]
    public string ExcludeSources { get; set; } = "L1/";

    /// <summary>默认规则迁移版本；升级时仅在用户仍使用旧默认空值时自动应用推荐排除规则。</summary>
    public int DefaultsMigrationVersion { get; set; } = 0;

    [DisplayName("首次迁移旧索引")]
    [Description("开：首次启动把官方目录里的旧 index.json 复制过来导入；关：直接按现有记忆文件重建新库。默认开。")]
    public bool MigrateOldIndex { get; set; } = true;

    [DisplayName("启用模型整理")]
    [Description("让模型定期读新落盘的记忆流水，提炼成事实卡落盘（原文不动）。关了就只走原来的文件级整理。")]
    public bool LlmOrganizeEnable { get; set; } = true;

    [DisplayName("模型整理间隔（分钟）")]
    [Description("后台每隔这么久让模型整理一次新记忆。建议 120（两小时）。")]
    public int LlmOrganizeMinutes { get; set; } = 120;

    [DisplayName("模型整理单次上限（字）")]
    [Description("单次喂给模型的新记忆最多多少字，超了截断。建议 30000。")]
    public int LlmOrganizeMaxChars { get; set; } = 30000;

    [DisplayName("手动整理单批上限（字）")]
    [Description("memorganize 一次最多喂多少字（默认 12000，比自动整理小，避免占满上下文）。")]
    public int LlmOrganizeManualMaxChars { get; set; } = 12000;

    [DisplayName("启动后首次整理延迟（分钟）")]
    [Description("重启后不要立即调用模型；等待这么久再做第一次自动整理。手动 memcompact / memorganize 不受影响。建议 120。")]
    public int LlmOrganizeStartupDelayMinutes { get; set; } = 120;

    [DisplayName("模型整理单次最多文件数")]
    [Description("一次模型整理最多处理多少个源文件，防止积压时一口气吃掉大量上下文。建议 5。")]
    public int LlmOrganizeMaxFilesPerRun { get; set; } = 5;

    [DisplayName("提示里强调先调工具")]
    [Description("打开后，命中提示会明确要求模型先用 memrecall / memfetch / memwhen 查，而不是自己去翻磁盘。建议开。")]
    public bool ToolFirst { get; set; } = true;

    [DisplayName("按时间查默认条数")]
    [Description("memwhen 不指定条数时默认返回几条。建议 8。")]
    public int WhenLimit { get; set; } = 8;

    [DisplayName("工具结果回显到聊天框")]
    [Description("打开后，工具查到的内容也会 Poke 进聊天框（截断到 400 字）。模型同轮已能拿到返回值，这主要是给人看的。建议开。")]
    public bool EchoToolResult { get; set; } = false;

    [DisplayName("事实合并阈值")]
    [Description("新事实与已有事实（同主体同类别）相似度达到这里就合并，低于则新增。0.5 左右比较合适。")]
    public double MergeThreshold { get; set; } = 0.5;

    [DisplayName("启用自动压缩")]
    [Description("把条目多、够旧的流水文件交给模型压成事实卡；只标记为「已压缩」（不进索引），官方文件不动。默认关。")]
    public bool AutoCompactEnable { get; set; } = false;

    [DisplayName("自动压缩间隔（分钟）")]
    [Description("后台每隔这么久检查一次可压缩的流水。建议 360（六小时）。")]
    public int AutoCompactMinutes { get; set; } = 360;

    [DisplayName("压缩门槛：至少多少条")]
    [Description("一个流水文件至少被切成这么多条才值得压。建议 30。")]
    public int AutoCompactMinEntries { get; set; } = 30;

    [DisplayName("压缩门槛：至少多旧（天）")]
    [Description("最近这么多天内的流水不压（还在用）。建议 7。")]
    public int AutoCompactMinAgeDays { get; set; } = 7;

    [DisplayName("压缩单次上限（字）")]
    [Description("一次喂给模型最多多少字。建议 20000。")]
    public int AutoCompactMaxChars { get; set; } = 20000;


    [DisplayName("启用事实审计")]
    [Description("后台定期扫事实卡里的重复/可能矛盾，写审计报告（只报告，不自动改）。建议开。")]
    public bool AuditEnable { get; set; } = true;

    [DisplayName("审计间隔（分钟）")]
    [Description("后台每隔这么久审一次。建议 120（两小时）。")]
    public int AuditMinutes { get; set; } = 120;

    [DisplayName("审计单次最多列几条")]
    [Description("一份报告最多列多少条问题。建议 30。")]
    public int AuditLimit { get; set; } = 30;

    [DisplayName("提示文本")]
    [Description("命中后追加到消息末尾的提示。{n} 命中条数、{files} 命中文件、{path} 记忆目录。")]
    public string HintTemplate { get; set; } =
        "[回忆提示：Memory 里可能有 {n} 条旧账跟这句相关（{files}）。要查往事就调记忆工具（别用 python/别扫盘）：" +
        "<memrecall query=\"关键词\"/> 查书签、<memfetch source=\"出处文件名\"/> 取原文、" +
        "<memwhen time=\"上周三\"/> 按时间翻；工具都查不到就直说没查到，不要编。]";
}
