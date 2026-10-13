
using System.Collections.Generic;
using System.ComponentModel;

namespace Mao.MemoryInject;

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
    [Description("用本地 bge-small 向量模型按意思检索；关闭或模型没就绪时退回字面关键词匹配。")]
    public bool UseSemantic { get; set; } = true;

    [DisplayName("Python 解释器")]
    [Description("留空使用 PATH 中的 python。需已安装 torch 与 transformers。建议写绝对路径。")]
    public string PythonExe { get; set; } = "";

    [DisplayName("记忆目录（相对角色目录）")]
    [Description("记忆文件所在位置，相对角色目录。默认 Storage/Memory。")]
    public string MemorySubPath { get; set; } = "Storage/Memory";

    [DisplayName("最多提示几条")]
    [Description("提示里最多列出几个命中的记忆文件。建议 3。")]
    public int MaxResults { get; set; } = 3;

    [DisplayName("绝对相关度下限")]
    [Description("语义模式余弦相似度下限，低于它不算命中。建议 0.55。")]
    public double MinScore { get; set; } = 0.55;

    [DisplayName("相对相关度阈值（比平均高多少）")]
    [Description("语义模式：命中的片段要比本轮平均相似度高出这么多才算数。越大越严。建议 0.22。")]
    public double MinMargin { get; set; } = 0.22;

    [DisplayName("关键词模式分数阈值")]
    [Description("仅字面匹配降级时使用的阈值，建议 0.10~0.20。")]
    public double MinScoreKeyword { get; set; } = 0.15;

    [DisplayName("忽略最近几天内的文件")]
    [Description("最近 N 天内的记忆文件不参与检索（那些内容本来就在眼前）。0 = 不忽略。默认 1。")]
    public int SkipRecentDays { get; set; } = 1;

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

    [DisplayName("提示文本")]
    [Description("命中后追加到消息末尾的提示。{n} 替换成命中条数，{files} 替换成命中的文件，{path} 替换成记忆目录。")]
    public string HintTemplate { get; set; } =
        "[回忆提示：Memory 里可能有 {n} 条旧账跟这句相关（{files}）。若这句话牵涉往事，先去 {path} 翻一翻再答，别凭印象编。]";
}
