"""文本切分、标签、时间解析、相似度与实体提取。纯函数层，只依赖共享状态。"""

import os
import re
import datetime
import hashlib

# >>> WP1 导入（自动生成，勿手改）
from state import (
    DECAY_FLOOR,
    DECAY_HALF_LIFE_DAYS,
    PEOPLE,
    _CONFIRMED,
    PERIOD_HOUR,
    THINGS,
    TOPICS,
    W_HEAD,
    W_SUMMARY,
    W_TAG,
    W_TEXT,
    _CFG,
    _ENTITY_PATTERNS,
    _ENTITY_STOP,
    _PERSON_STOP,
    _warn,
)
# <<< WP1 导入

def _field_text(e, field):
    v = e.get(field) or ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def score_entry(e, terms):
    """按权重累加打分；返回 (总分, 命中词数)。"""
    tags = " ".join([_field_text(e, "people"), _field_text(e, "topics"), _field_text(e, "things")]).lower()
    head = _field_text(e, "head").lower()
    summary = _field_text(e, "summary").lower()
    text = _field_text(e, "text").lower()
    total = 0
    hit = 0
    for t in terms:
        if t in tags:
            s = W_TAG
        elif t in head:
            s = W_HEAD
        elif t in summary:
            s = W_SUMMARY
        elif t in text:
            s = W_TEXT
        else:
            s = 0
        if s:
            total += s
            hit += 1
    return total, hit


def rel_time(date_str, today=None):
    """相对时间：今天/昨天/N天前/上周/N周前/N个月前/N年前。读取时按当下重算。"""
    try:
        d = datetime.date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return str(date_str)
    today = today or datetime.date.today()
    n = (today - d).days
    if n < 0:
        return str(date_str)
    if n == 0:
        return "今天"
    if n == 1:
        return "昨天"
    if n < 7:
        return "%d天前" % n
    if n < 14:
        return "上周"
    if n < 60:
        return "%d周前" % (n // 7)
    if n < 365:
        return "%d个月前" % (n // 30)
    return "%d年前" % (n // 365)


def parse_time_hint(text):
    """把中文时间词解析成 (date_from, date_to, period)。解析不出就返回 (None,None,None)。
    只做保守解析：识别不出宁可不过滤，避免误杀。"""
    if not text:
        return None, None, None
    t = str(text)
    today = datetime.date.today()
    df = dt = None
    period = None

    for kw, p in (("凌晨", "凌晨"), ("早上", "早上"), ("早晨", "早上"), ("上午", "上午"),
                  ("中午", "中午"), ("午后", "下午"), ("下午", "下午"),
                  ("傍晚", "傍晚"), ("晚上", "晚上"), ("夜里", "夜"), ("深夜", "深夜"), ("晚", "晚")):
        if kw in t:
            period = p
            break

    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", t)
    if m:
        df = dt = "%04d-%02d-%02d" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", t)
        if m:
            df = dt = "%04d-%02d-%02d" % (today.year, int(m.group(1)), int(m.group(2)))
        elif "前天" in t:
            df = dt = (today - datetime.timedelta(days=2)).isoformat()
        elif "昨天" in t or "昨晚" in t:
            df = dt = (today - datetime.timedelta(days=1)).isoformat()
        elif "今天" in t or "今日" in t or "今晚" in t or "刚才" in t:
            df = dt = today.isoformat()
        else:
            m = re.search(r"(\d+)\s*天前", t)
            if m:
                df = dt = (today - datetime.timedelta(days=int(m.group(1)))).isoformat()
            elif "上周" in t:
                df = (today - datetime.timedelta(days=today.weekday() + 7)).isoformat()
                dt = (today - datetime.timedelta(days=today.weekday() + 1)).isoformat()
            elif "这周" in t or "本周" in t:
                df = (today - datetime.timedelta(days=today.weekday())).isoformat()
                dt = today.isoformat()
            elif "上个月" in t:
                df = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1).isoformat()
                dt = (today.replace(day=1) - datetime.timedelta(days=1)).isoformat()
    return df, dt, period


def _decay_factor(entry):
    """按事实日期衰减；调用统计和最后访问时间不得改变召回排序。"""
    base = entry.get("date") or ""
    try:
        d = datetime.date.fromisoformat(str(base)[:10])
    except ValueError:
        return 1.0
    age = (datetime.date.today() - d).days
    if age <= 0:
        return 1.0
    import math
    f = 0.5 ** (age / float(_CFG.get("half_life") or DECAY_HALF_LIFE_DAYS))
    return max(DECAY_FLOOR, min(1.0, f))


def _entry_weight(entry):
    """Keep date decay while excluding call statistics from ranking."""
    decay = _decay_factor(entry)
    return 0.5 + 0.5 * decay


def _looks_important(e):
    """像牵挂/欠账/约定/健康这类，永远不软删。"""
    blob = (_field_text(e, "head") + _field_text(e, "summary") + _field_text(e, "text"))
    for k in ("牵挂", "欠账", "约定", "待办", "血压", "禁欲", "早睡", "欠", "别忘"):
        if k in blob:
            return True
    return False


def _split_fine(head, body):
    """把一个大标题块切细：列表项/空行/句末切，单块上限 200 字。
    让一句话只跟真正出现它的那一小块挂钩，治'大块混进不相关词'的噪声。"""
    out = []
    units = []
    buf = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s:
            if buf:
                units.append("\n".join(buf)); buf = []
            continue
        if re.match(r"^([-*+]|\d+[.)])\s", s):
            if buf:
                units.append("\n".join(buf)); buf = []
            units.append(s)
        else:
            buf.append(s)
    if buf:
        units.append("\n".join(buf))
    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(u) <= _CFG["block_max"]:
            out.append((head, u))
            continue
        parts = re.split(r"(?<=[。！？!?；;])", u)
        b = ""
        for pt in parts:
            if b and len(b) + len(pt) > _CFG["block_max"]:
                out.append((head, b)); b = pt
            else:
                b += pt
        if b.strip():
            out.append((head, b))
    return out


def _extract_hour(text):
    """从正文找 'HH:MM' 或 'HH点'，返回小时(0-23)，找不到返回 None。"""
    m = re.search(r"(\d{1,2})[:：](\d{2})", text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    m = re.search(r"(\d{1,2})\s*点", text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    return None


def _split_blocks(text):
    blocks = []
    head = ""
    buf = []
    for ln in text.split("\n"):
        if re.match(r"^#{1,6}\s", ln.strip()):
            if buf and "".join(buf).strip():
                blocks.extend(_split_fine(head, "\n".join(buf)))
            head = ln.strip()
            buf = []
        else:
            buf.append(ln)
    if buf and "".join(buf).strip():
        blocks.extend(_split_fine(head, "\n".join(buf)))
    return blocks


def _guess_time(fname, fpath):
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", fname)
    if m:
        d = "%s-%s-%s" % m.groups()
    else:
        m2 = re.search(r"(\d{8})", fname)
        if m2:
            s2 = m2.group(1)
            d = "%s-%s-%s" % (s2[:4], s2[4:6], s2[6:8])
        else:
            d = datetime.date.fromtimestamp(os.path.getmtime(fpath)).isoformat()
    period = ""
    for pp in PERIOD_HOUR:
        if pp in fname:
            period = pp
            break
    hh = PERIOD_HOUR.get(period, 12)
    return {"date": d, "period": period or "日",
            "time": "%sT%02d:00:00" % (d, hh), "hour": hh}


def _tag(text):
    r = {"people": [], "topics": [], "things": []}
    for name, kws in PEOPLE.items():
        if any(k in text for k in kws):
            r["people"].append(name)
    for name, kws in TOPICS.items():
        if any(k in text for k in kws):
            r["topics"].append(name)
    for name, kws in THINGS.items():
        if any(k in text for k in kws):
            r["things"].append(name)
    return r


def _first_sent(body):
    for ln in body.split("\n"):
        t = ln.strip().lstrip("#-*> ").strip()
        if len(t) >= 6:
            return t[:46]
    # 没有任何一行够长（标题行、短句列表）：退化成压平后的开头，避免 head 为空
    flat = re.sub(r"\s+", " ", str(body or "")).strip().lstrip("#-*> ").strip()
    return flat[:46]


# ---------- 标签（people / topics / things）的切分与归一 ----------

# 历史数据里 people 用过顿号拼接（如 主人|真央、主人），竖线才是当前约定
_TAG_SEP_RE = re.compile(r"[|、]")


def split_tag_list(value):
    """把 "a|b、c" 或 ["a|b", "c"] 统一切成去空、去重的标签列表。"""
    raw = []
    if isinstance(value, (list, tuple)):
        for item in value:
            raw.extend(_TAG_SEP_RE.split(str(item or "")))
    else:
        raw.extend(_TAG_SEP_RE.split(str(value or "")))
    out, seen = [], set()
    for name in raw:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


_ASCII_IDENT_RE = re.compile(r"[_./#\-\d]")

# ---------- 主体分类：人物 / 共同主体 / 软件 / 事件 / 待分类 ----------
#
# 实测污染（Retromem 5.1 · PROFILE-04）：`people` 列里混进了无上下文代词（他、我、
# 双方）、共同主体（主人/真央、他/真央）、软件与平台标识（ComfyUI、CodeBuddy、
# MemoryInject、Alife.Function.Memory）以及事件／健康主题（健康、打卡）。
# 旧规则"含中文就算人""首字母大写就算人"会把它们统统放行，所以这里按证据分层判定。
#
# 关键约束：**分类不等于删除**。被判成软件／事件的标签分流到 things/topics，
# 判不准的进 pending（待分类），原文仍存在库里可追溯；只有无上下文代词被丢弃。
# 项目里被当作人物的 AI 角色（Astra、Mao、真央、白姬汤）先于"软件同名"判定，
# 不会因为和插件／模型重名就被剔掉。

_ROLE_PEOPLE = set(PEOPLE) | {alias for aliases in PEOPLE.values() for alias in aliases}
_ROLE_PEOPLE |= {"真央", "Astra", "Mao", "白姬汤"}

_SOFTWARE_TAGS = set(THINGS) | {alias for aliases in THINGS.values() for alias in aliases}
_SOFTWARE_TAGS |= {
    "MemoryInject", "Retromem", "Mao.MemoryInject", "Alife", "Alife.Function.Memory",
    "DeepSeek", "OpenAI", "GPT", "Claude", "Qwen", "Llama", "Ollama", "Python", "SQLite",
    "Unity", "Windows", "ComfyUI", "CodeBuddy", "Obsidian", "QQ",
    "插件", "系统", "软件", "客户端", "服务", "模型", "工作流", "端口", "索引", "数据库",
}

_EVENT_TAGS = set(TOPICS) | {
    "打卡", "健康", "审计", "召回", "配置", "设置", "版本", "地址", "住处", "称呼", "偏好",
    "血压", "低压", "高压", "睡眠", "饮食", "体重", "运动", "吃药", "头痛", "头晕",
}

_SUBJECT_PRONOUNS = set(_PERSON_STOP) | {"双方", "两者", "二人", "两人", "对方"}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 中文标签默认**不确认**为人物：`记忆库`、`工具调用`、`定时任务`、`习惯机制` 都曾被
# 当成人物（5.1.1 复核 R05）。"含中文"或"像姓名"都不是身份证据；判不准的进
# people_pending，由用户在画像页点「确认是人物」写入 people_confirm.json。
# 已知人物来自 state.PEOPLE（含别名）、内置角色表和用户确认清单。


def _subject_core(name):
    """单个标签（不含 A/B 组合）的判定，返回 (kind, reason)。"""
    s = str(name or "").strip()
    if not s:
        return "empty", "空标签"
    if s in _SUBJECT_PRONOUNS:
        return "pronoun", "无上下文代词"
    if s in _ROLE_PEOPLE:
        return "person", "本项目人物或别名"
    if s in _CONFIRMED["rejected"]:
        return "unknown", "用户已标记为非人物"
    # ★结构证据先于"用户确认"：5.1.2 曾把整串共同主体或路径写进确认清单，
    #   旧文件不能让 `主人/沐禾`、`MemoryInject/4.8.7` 继续当人物（S05 兼容）。
    if s in _SOFTWARE_TAGS:
        return "object", "软件／平台／模型标识"
    if s in _EVENT_TAGS:
        return "event", "事件或主题"
    if re.search(r"[：:\\]", s) or _ASCII_IDENT_RE.search(s):
        return "unknown", "标识符／路径，非人名"
    if s in _CONFIRMED["names"]:
        return "person", "用户已确认"
    if _CJK_RE.search(s):
        if len(s) > 12 or re.search(r"[的了是在都也就]", s):
            return "unknown", "像句子片段，留待分类"
        return "unknown", "无法确认是否人名，留待分类"
    if re.match(r"^[A-Z][A-Za-z]{1,}$", s):
        if s.isupper():
            return "unknown", "英文缩写，留待分类"
        return "person", "首字母大写英文名"
    return "unknown", "无法判定，留待分类"


def classify_subject(name):
    """把一个主体标签判成 person / shared / object / event / pronoun / unknown。

    `shared` 是「共同主体」：`主人/真央` 拆成参与者 [主人, 真央]，不再生成一个叫
    「主人/真央」的人物；`他/真央` 里的代词被丢掉，只留 真央。
    斜杠只有在两侧都像人（或一侧是代词）时才当分隔符——`MemoryInject/4.8.7`、
    `C:/models/x` 这类路径／版本号不会被拆成伪人物。
    """
    s = str(name or "").strip()
    if not s:
        return {"name": s, "kind": "empty", "reason": "空标签", "participants": [], "dropped": []}
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if len(parts) >= 2:
            cores = [(part,) + _subject_core(part) for part in parts]
            kinds = [item[1] for item in cores]
            if any(kind == "person" for kind in kinds) and all(kind in ("person", "pronoun") for kind in kinds):
                return {
                    "name": s, "kind": "shared",
                    "participants": [item[0] for item in cores if item[1] == "person"],
                    "dropped": [item[0] for item in cores if item[1] != "person"],
                    "reason": "共同主体（按参与者拆分）",
                }
    kind, reason = _subject_core(s)
    participants = [s] if kind in ("person", "object", "event") else []
    return {"name": s, "kind": kind, "reason": reason,
            "participants": participants, "dropped": []}



def can_reject_subject(name):
    """路径/版本组合可以排除；可能的共同人物保留逐人确认入口。"""
    text = str(name or "").strip()
    if "/" not in text:
        return True
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if len(parts) < 2:
        return True
    for part in parts:
        info = classify_subject(part)
        if info["kind"] in ("object", "event", "empty"):
            return True
        if info["kind"] == "unknown" and any(
                word in info["reason"] for word in ("标识符", "句子片段")):
            return True
    return False


def _is_person_tag(name):
    """这个标签能不能当"人"用：只认能确认的人物，共同主体与待分类都不算。"""
    return classify_subject(name)["kind"] == "person"


def classified_people(entry):
    """条目里可确认的人物（含共同主体拆出的参与者），去重且保持出现顺序。"""
    out, seen = [], set()
    for name in split_tag_list(entry.get("people")):
        info = classify_subject(name)
        if info["kind"] not in ("person", "shared"):
            continue
        for person in info["participants"]:
            key = person.casefold()
            if key not in seen:
                seen.add(key)
                out.append(person)
    return out


def classify_tags(entry):
    """三类标签过一遍主体分类，返回干净的 people/topics/things 与分流记录。

    * 人物与共同主体拆出的参与者进 people；
    * 判成软件/平台/模型的进 things，判成事件/主题的进 topics（信息不丢）；
    * 代词直接丢；判不准的进 pending，等人工分类，不冒充人物。

    ★输入里已有的 `shared` / `pending`（库里 5.1 的两列）必须保留并重新合并：
      历史整理只凭当前 people 重算，会把上一轮存下的待分类／共同主体覆盖成空
      （5.1.1 复核 R03）。重复执行必须幂等。
    """
    people, shared, pending = [], [], []
    topics = split_tag_list(entry.get("topics"))
    things = split_tag_list(entry.get("things"))
    seen = {name.casefold() for name in topics + things}
    pending_seen = set()

    def add(bucket, name):
        key = str(name or "").strip().casefold()
        if not key or key in seen:
            return
        seen.add(key)
        bucket.append(name)

    def route(raw, allow_shared=True):
        info = classify_subject(raw)
        kind = info["kind"]
        if kind in ("person", "shared"):
            for person in info["participants"]:
                add(people, person)
            if kind == "shared" and allow_shared and raw not in shared:
                shared.append(raw)
        elif kind == "object":
            add(things, raw)
        elif kind == "event":
            add(topics, raw)
        elif kind in ("pronoun", "empty"):
            return
        else:
            key = str(raw or "").strip().casefold()
            if key and key not in seen and key not in pending_seen:
                pending_seen.add(key)
                pending.append(raw)

    for raw in split_tag_list(entry.get("people")):
        route(raw)
    # 已存的共同主体：拆成参与者，同时保留原文一列（可追溯）
    for raw in split_tag_list(entry.get("shared")):
        route(raw)
    # 已存的待分类：仍然判不准就留在待分类，确认后自动升为人物
    for raw in split_tag_list(entry.get("pending")):
        route(raw)
    return {"people": people, "topics": topics, "things": things,
            "shared": shared, "pending": pending}


def normalize_tags(entry):
    """把一条条目的三类标签规整成干净列表（不改变正文，只影响图与检索标签）。"""
    return classify_tags(entry)


def hay_of(entry, tags=None):
    """检索用的 hay：标题 + 摘要 + 标签（小写），**不再复制正文**。

    实测（Retromem 5.1 · DB-01）`hay` 把整段正文又存了一遍，真央库多占约 8 MB。
    召回的关键词预筛改成 `hay LIKE ? OR body LIKE ?`，检索契约不变（正文仍可命中，
    LIKE 对 ASCII 依旧大小写不敏感），但库里的正文只保留 body 一份。
    """
    tags = tags if tags is not None else classify_tags(entry)
    names = (tags.get("people") or []) + (tags.get("topics") or []) + (tags.get("things") or [])
    # 共同主体与待分类单独成列时，分类结果里没有它们，要显式带上（否则重算 hay 会漏词）
    names = names + list(tags.get("shared") or []) + list(entry.get("shared") or [])
    names = names + list(tags.get("pending") or []) + list(entry.get("pending") or [])
    names = list(dict.fromkeys(str(x) for x in names if x))
    return " ".join([str(entry.get("head") or ""), str(entry.get("summary") or ""),
                     " ".join(names)]).lower()


def entry_nodes(entry):
    """一条记忆参与图谱的节点集合。

    页面图（`dbside._db_rebuild_links`）与对话图（`graph_indexer.build_graph`）
    都调这里，避免两套规则各算各的。
    """
    tags = normalize_tags(entry)
    text = "%s %s %s" % (entry.get("head") or "", entry.get("summary") or "",
                         str(entry.get("text") or "")[:400])
    nodes, seen = [], set()
    for name in tags["people"] + tags["topics"] + tags["things"] + extract_entities(text)[:6]:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        nodes.append(name)
    return nodes


def _tokens(text):
    """粗粒度 token（中文按 2-gram、英文按词），只用于相似度比较。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text or ""))
    out = set()
    for w in s.split():
        if len(w) <= 2:
            out.add(w)
        else:
            for i in range(len(w) - 1):
                out.add(w[i:i + 2])
    return out


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return (inter / float(uni)) if uni else 0.0


def _sim(a, b):
    """事实相似度：Jaccard 与"包含度"取大。
    短句（如「血压偏高」vs「血压还是偏高些」）Jaccard 会被稀释，包含度能抓住。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    j = inter / float(len(a | b))
    c = inter / float(min(len(a), len(b)))
    return max(j, c * 0.9)


def _fact_files(memdir):
    out = []
    try:
        for name in os.listdir(memdir):
            if name.startswith("事实_") and name.endswith(".md"):
                out.append(os.path.join(memdir, name))
    except OSError as e:
        # 目录不存在属正常（新角色还没有事实卡），其余失败要能定位
        if not isinstance(e, FileNotFoundError):
            _warn("list fact files: %s" % memdir, e)
    out.sort()
    return out


def _parse_fact(line):
    """解析事实卡行：- 日期 | 涉及谁 | 属哪件事 | 一句话事实"""
    s = str(line or "").strip()
    if s.startswith("-"):
        s = s[1:].strip()
    if "|" not in s:
        return None
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 4:
        return None
    return {"date": parts[0], "who": parts[1], "topic": parts[2], "fact": "|".join(parts[3:]).strip()}


# ---------- 现态（会过期）与日期次序 ----------
#
# 记忆里只有一小部分是"当前值"：渠道、配置、健康数值、住处这类，新的说法会
# 顶掉旧的。其余绝大多数是"某天发生过什么"，两条都要留着，召回才答得出时间线。
# 所以这里只做最小的可解释判定，宁可漏判，也不把一次性事件当成状态。

STATE_WORDS = ("现在", "目前", "当前", "如今", "最新", "改为", "改成", "改用",
               "换成", "仍在", "现阶段", "当下", "今后", "以后")
# 只收「一条属性值」的主题。健康／习惯这类是逐日记录，每次都带自己的日期与数值，
# 旧的一条不是被新的一条取代（实测真央 97 行健康卡，进白名单会造 14 条误判），
# 所以它们只能靠现态词进入，不能靠主题。
_STATE_TOPICS = ("配置", "设置", "渠道", "模型", "地址", "住处", "称呼", "偏好", "版本")


def looks_like_state(topic, fact):
    """这条事实是"会过期的当前值"，还是"发生过的事"。

    只用表面信号：带现态词，或主题属于属性类（配置/健康/渠道……）。
    事件描述没有这些特征，因此不会被"以新为主"影响。
    """
    text = str(fact or "")
    if any(word in text for word in STATE_WORDS):
        return True
    topic = str(topic or "")
    return any(key in topic for key in _STATE_TOPICS)


def fact_date_key(value, fallback_year=None):
    """事实卡日期 → 可比较的 (年, 月, 日)；认不出返回 None（不猜）。

    真库 2034 行里 305 行写成 10/28（缺年份），其余是 2026-10-28。
    缺年份时用调用方给的 fallback_year，没给就按今年。
    """
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.fullmatch(r"(\d{1,2})\s*/\s*(\d{1,2})", text)
    if match:
        year = int(fallback_year) if fallback_year else datetime.date.today().year
        return (year, int(match.group(1)), int(match.group(2)))
    return None


def fact_time(value, fallback_date=""):
    """事实卡里的日期 → 条目的 (date, time)。

    事实卡正文自带日期，比文件 mtime 更接近"这件事发生在哪天"：同一天写的十张卡
    不该都被记成"建库那天"（画像排序、去重的同/跨日判断都依赖它）。
    卡里没有钟点，统一记 12:00，避免重建时间不同导致顺序漂移。
    """
    year = None
    match = re.match(r"(\d{4})", str(fallback_date or ""))
    if match:
        year = int(match.group(1))
    key = fact_date_key(value, year)
    if not key:
        return "", ""
    day = "%04d-%02d-%02d" % key
    return day, day + "T12:00:00"


def _audit_card_id(source, body, ordinal):
    version = hashlib.sha256(str(body).strip().encode("utf-8")).hexdigest()
    raw = "%s\0%d" % (source, int(ordinal))
    return "fc:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32], version


_ENTITY_BAD_CHARS = re.compile(
    r"[\s\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u2026\u2014"
    r"\uff08\uff09\u201c\u201d\u2018\u2019\u300a\u300b\u300c\u300d\u300e\u300f]")
_ENTITY_MARKUP = re.compile(r"[#*`>~\[\]{}|<>+=]")


def _looks_like_entity(value):
    """实体候选的最低门槛：拒绝句子碎片与 markdown 痕迹，保留技术专名。

    实测依据（真央库 26742 个节点）：含中文标点的 6817 个、长度 >12 的 8043 个，
    典型垃圾是「#只是随 - **★我犯了严重的错**：他根本没说过」。这些全部来自
    引号兜底匹配吞掉了整段正文，而不是真的专名。
    """
    s = str(value or "").strip().strip(" 　,，。.;；:：、")
    if len(s) < 2 or len(s) > 24:
        return False
    if _ENTITY_BAD_CHARS.search(s) or _ENTITY_MARKUP.search(s):
        return False
    if s.isdigit():
        return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", s):
        return False          # 纯数字/编号（10/10、9-19）不是实体
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    if cjk and len(s) > 12:      # 中文专名不该超过 12 字；超过基本是整句
        return False
    return True


def extract_entities(text):
    """从正文抽实体（书名号/引号/专名后缀/拉丁词），不用模型，避免吃显存。"""
    t = str(text or "")
    out = []
    for rx in _ENTITY_PATTERNS:
        for m in rx.finditer(t):
            out.append((m.group(1) if m.groups() else m.group(0)).strip())
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_.\-]{2,20}", t):
        out.append(m.group(0))
    seen, res = set(), []
    for x in out:
        x = x.strip(" 　,，。.;；:：")
        if not _looks_like_entity(x) or x in _ENTITY_STOP:
            continue
        key = x.casefold()
        if key in seen:
            continue
        seen.add(key)
        res.append(x)
    return res
