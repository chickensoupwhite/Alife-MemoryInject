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
    return ""


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


def _audit_card_id(source, body, ordinal):
    version = hashlib.sha256(str(body).strip().encode("utf-8")).hexdigest()
    raw = "%s\0%d" % (source, int(ordinal))
    return "fc:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32], version


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
        if not x or len(x) < 2 or x in _ENTITY_STOP or x in seen:
            continue
        seen.add(x)
        res.append(x)
    return res
