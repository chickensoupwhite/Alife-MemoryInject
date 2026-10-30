"""人物画像的组织：证据归属、时间顺序、主题归纳与可追溯摘要。

问题来源（Retromem 5.1 · PROFILE-01/02/03）：

* PROFILE-01：`memprofile` 只把"近 N 条"按时间铺成流水，读者看不到"这个人是什么样"；
* PROFILE-02：按 `time` 字符串升序取尾部，显示时「昨天」排在「2 天前」后面；
* PROFILE-03：只要 `people` 拼成的字符串里子串命中就算个人事实，共同主体
  （主人/真央）和别人的行为被混进个人画像。

这里只做**有事实依据的归纳**，不调模型、不生成性格标签：
摘要里的每一条都能指回具体事实卡 id／出处，证据不足就明说；
配置、模型、插件这类事务记录单独计数，不折算成人格特点。
"""
import re

import textsim
from state import PEOPLE

# 主题只看"这个人怎样"，不看"那天发生了什么"。顺序即优先级：一句话同时命中
# 多个主题时归到最靠前的那个，避免同一句被重复计入多个主题。
THEMES = (
    ("preference", "偏好与习惯",
     ("喜欢", "爱吃", "爱喝", "偏好", "习惯", "口味", "讨厌", "不吃", "常去", "每天", "总是")),
    ("expression", "表达与互动",
     ("说话", "聊", "语气", "称呼", "叫她", "叫他", "笑", "撒娇", "哄", "夸", "回我", "问我")),
    ("attention", "持续关注",
     ("在意", "担心", "关注", "惦记", "想", "计划", "准备", "复习", "学", "工作", "上班")),
    ("relationship", "关系与约定",
     ("约定", "答应", "一起", "陪", "承诺", "约好", "见面", "关系", "陪着")),
)
_THEME_TITLES = {key: title for key, title, _ in THEMES}

# 事务性记录：渠道、模型、配置、插件、索引这些是"系统状态"，不是人物特征。
_TRANSACTION_WORDS = ("配置", "设置", "模型", "渠道", "接口", "端口", "插件", "索引", "版本",
                      "重启", "安装", "部署", "日志", "协议", "参数", "工作流", "显存", "内存")
_TRANSACTION_TOPICS = ("技术", "配置", "设置")


def aliases_of(name):
    """已确认别名：同一人物在事实卡里可能写成"妈""妈妈""母亲"。"""
    target = str(name or "").strip().casefold()
    for key, aliases in PEOPLE.items():
        names = [key] + list(aliases)
        if target in [item.casefold() for item in names]:
            return names
    return [str(name or "").strip()]


def _norm(value):
    return str(value or "").strip().casefold()


def _body(entry):
    return str(entry.get("text") or entry.get("body") or "")


def _blob(entry):
    return " ".join([str(entry.get("head") or ""), str(entry.get("summary") or ""),
                     _body(entry)[:600], " ".join(entry.get("topics") or [])])


def _people_of(entry):
    return textsim.classified_people(entry)


def _matches(entry, names):
    wanted = {_norm(item) for item in names}
    return [person for person in _people_of(entry) if _norm(person) in wanted]


def when_key(entry):
    """绝对时间排序键：新→旧，缺日期但有时刻的用时刻里的日期，两者都没有排最后。

    显示用的「昨天／2 天前」由日期现算，不参与排序——否则跨月、缺日期时会乱。
    """
    date = str(entry.get("date") or "").strip()
    time = str(entry.get("time") or "").strip()
    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date) or \
        re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", time)
    if match:
        return "%04d-%02d-%02d" % (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return "0000-00-00"


def when_text(entry):
    date = when_key(entry)
    time = str(entry.get("time") or "").strip()
    clock = re.search(r"(\d{1,2}):(\d{2})", time)
    if date == "0000-00-00":
        return "日期未知"
    if clock:
        return "%s %02d:%02d" % (date, int(clock.group(1)), int(clock.group(2)))
    return date


def _event_key(entry):
    """完整时间排序键：日期 → 时刻 → id，三者都参与，截断前后口径一致（R06）。"""
    return (when_key(entry), str(entry.get("time") or ""), str(entry.get("id") or ""))


def is_transaction(entry):
    """事务记录：系统状态，不当人物特征用。"""
    topics = {str(x) for x in (entry.get("topics") or [])}
    if topics & set(_TRANSACTION_TOPICS):
        return True
    blob = _blob(entry)
    return any(word in blob for word in _TRANSACTION_WORDS)


def theme_of(entry):
    """给一条证据选主题；事务记录与认不出的内容归 other，不进摘要。"""
    if is_transaction(entry):
        return "transaction"
    blob = _blob(entry)
    for key, _title, words in THEMES:
        if any(word in blob for word in words):
            return key
    return "other"


def collect(entries, name, aliases=None):
    """把候选条目分成 直接事实 / 共同事件 / 仅正文提及。

    * 直接事实：标签里只有这一个确认人物，事实的主语就是他/她；
    * 共同事件：标签里还有别人（含共同主体拆出的参与者），只能当关系证据；
    * 仅提及：标签里没有这个人、只在正文出现——不当作个人事实，只报数量。
    """
    names = list(dict.fromkeys(aliases_of(name) + list(aliases or [])))
    direct, shared, mention = [], [], []
    for entry in entries or []:
        persons = _people_of(entry)
        hit = _matches(entry, names)
        if hit:
            others = [person for person in persons if _norm(person) not in {_norm(item) for item in names}]
            if others:
                shared.append((entry, others))
            else:
                direct.append(entry)
        elif any(_norm(alias) in _norm(_blob(entry)) for alias in names if alias):
            mention.append(entry)
    direct.sort(key=_event_key, reverse=True)
    shared.sort(key=lambda pair: _event_key(pair[0]), reverse=True)
    mention.sort(key=_event_key, reverse=True)
    return {"direct": direct, "shared": shared, "mention": mention}


def _evidence(entry, extra=""):
    head = str(entry.get("head") or "").lstrip("#").strip() or "（无标题）"
    summary = str(entry.get("summary") or "").strip()
    text = str(entry.get("text") or "").strip()
    if summary and summary != head:
        detail = summary
    else:
        detail = text[:80]
    item = {
        "id": str(entry.get("id") or ""),
        "source": str(entry.get("source") or ""),
        "date": when_key(entry),
        "when": when_text(entry),
        "head": head,
        "detail": detail[:120],
        "shared_with": list(extra or []),
        # 证据归属：共同事件只能当关系证据，不能冒充目标人物的个人特征（R07）
        "basis": "shared" if extra else "direct",
        "theme": theme_of(entry),
    }
    return item


def build(entries, name, limit=40, aliases=None):
    """生成画像结构：计数、主题、摘要行与证据列表（全部可追溯）。"""
    groups = collect(entries, name, aliases=aliases)
    direct, shared, mention = groups["direct"], groups["shared"], groups["mention"]
    total = len(direct) + len(shared)

    # ★主题只归纳"直接事实"。共同事件（含共同主体拆出的参与者）另列关系证据，
    #   不能因为一句"主人喜欢红茶，真央只是帮忙记录"就写成真央的偏好（R07）。
    buckets = {}
    for entry in direct:
        buckets.setdefault(theme_of(entry), []).append(entry)

    themes = []
    for key, title, _words in THEMES:
        rows = buckets.get(key) or []
        if not rows:
            continue
        rows.sort(key=_event_key, reverse=True)
        themes.append({"key": key, "title": title, "count": len(rows), "basis": "direct",
                       "evidence": [_evidence(entry) for entry in rows[:6]]})
    for key, title in (("transaction", "事务记录"), ("other", "其他记录")):
        rows = buckets.get(key) or []
        if rows:
            rows.sort(key=_event_key, reverse=True)
            themes.append({"key": key, "title": title, "count": len(rows), "basis": "direct",
                           "evidence": [_evidence(entry) for entry in rows[:6]]})
    if shared:
        themes.append({
            "key": "shared", "title": "与他人共同（关系证据）", "count": len(shared),
            "basis": "shared",
            "evidence": [_evidence(entry, others) for entry, others in shared[:6]],
        })

    summary = summarize(name, direct, shared, themes, total, len(mention))

    # 同一份完整时间键排序，再截断；主题样例与近期证据口径一致（R06）
    merged = [(entry, "") for entry in direct] + [(entry, others) for entry, others in shared]
    merged.sort(key=lambda pair: _event_key(pair[0]), reverse=True)
    evidence = [_evidence(entry, extra) for entry, extra in merged]
    cap = max(1, int(limit or 40))
    return {
        "name": str(name or ""),
        "total": total,
        "direct_count": len(direct),
        "shared_count": len(shared),
        "mention_count": len(mention),
        "themes": themes,
        "summary": summary,
        "evidence": evidence[:cap],
        "evidence_truncated": len(evidence) > cap,
    }


def summarize(name, direct, shared, themes, total, mention_count):
    """摘要行：只把有证据的主题写成句子，事务记录单独说明，不编造性格。"""
    lines = []
    if not total:
        return ["没有可用的个人事实（标签里没有该人物），以下只能是正文提及，不能当画像依据。"] \
            if mention_count else ["没有找到与该人物直接相关的事实卡。"]
    lines.append("共 %d 条事实：直接 %d 条、与他人共同 %d 条%s。" % (
        total, len(direct), len(shared),
        ("；另有 %d 条仅正文提及，未计入画像" % mention_count) if mention_count else ""))
    for theme in themes:
        if theme["key"] in ("transaction", "other", "shared"):
            continue
        sample = theme["evidence"][:2]
        points = "；".join(item["detail"] or item["head"] for item in sample)
        basis = "、".join(item["id"] for item in sample[:2] if item["id"]) or "（出处见证据列表）"
        lines.append("%s（%d 条，直接事实）：%s。依据 %s" % (theme["title"], theme["count"], points, basis))
    shared_theme = next((theme for theme in themes if theme["key"] == "shared"), None)
    if shared_theme:
        others = []
        for item in shared_theme["evidence"]:
            for person in item.get("shared_with") or []:
                if person not in others:
                    others.append(person)
        lines.append("另有 %d 条与他人共同的事件，只作关系证据，不计入个人特征%s。" % (
            shared_theme["count"], ("（涉及 " + "、".join(others[:4]) + "）") if others else ""))
    transactions = next((theme for theme in themes if theme["key"] == "transaction"), None)
    if transactions:
        lines.append("另有 %d 条属配置／模型／插件等事务记录，不计入人物特征。" % transactions["count"])
    if not any(theme["key"] not in ("transaction", "other", "shared") for theme in themes):
        lines.append("证据以事务、共同事件与零散记录为主，暂不足以归纳稳定的人物特点。")
    return lines


def render(name, entries, limit=40, note="", aliases=None):
    """纯文本画像卡：摘要在前、证据在后，按绝对时间新→旧。"""
    card = build(entries, name, limit, aliases=aliases)
    out = ["# 画像卡 · " + str(name or ""), ""]
    if note:
        # 兜底路径必须把降级范围说清楚，不能把旧状态当成有效画像（R09）
        out.extend(["> " + str(note), ""])
    for line in card["summary"]:
        out.append("- " + line)
    out.append("")
    if card["evidence"]:
        out.append("## 近期证据（新→旧，最多 %d 条）" % len(card["evidence"]))
    for item in card["evidence"]:
        label = item["when"]
        if item["shared_with"]:
            label += " · 共同：" + "、".join(item["shared_with"])
        out.append("- [%s] %s：%s（%s）" % (label, item["head"], item["detail"],
                                          item["source"] or item["id"] or "-"))
    if card["evidence_truncated"]:
        out.append("（只列了最近的 %d 条，共 %d 条）" % (len(card["evidence"]), card["total"]))
    return "\n".join(out)
