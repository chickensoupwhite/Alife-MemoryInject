"""自动召回的入口门控、话题断层与出口重复压制。

落点按 D10：全部在侧车，`db_recall` 是唯一召回入口，因此覆盖自动注入与主动工具。
主动工具（memrecall / memwhen / memfetch）在这里直接放行，不进入门控与压制分支，
且该放行不依赖 C# 是否正确传递入口标识——入口缺失时按自动召回处理之外，
调用方仍可用 `entry` 显式声明；两者都不影响主动工具的既有能力。

指纹 = 稳定事实卡 ID + 正文版本。标签、重要度、统计变化都不改变指纹；
正文改动会改变版本，从而允许修正后的内容再次进入召回评估。
"""
import time

import textsim

# 会话窗口：session(= 角色索引路径) -> {"tokens": set, "prints": set, "stamp": float}
WINDOW = {}

TOPIC_JACCARD = 0.25      # 与上次查询词集的重合度低于此值，视为换话题
WINDOW_TTL_SECONDS = 3600  # 会话窗口闲置上限，避免多角色长期驻留占内存

SMALLTALK = {
    "你好", "您好", "在吗", "在么", "在不在", "谢谢", "多谢", "晚安", "早安", "早上好",
    "晚上好", "哈哈", "呵呵", "嗯", "哦", "好", "行", "收到", "知道了", "没事", "算了",
}

ACTIVE_TOOLS = ("memrecall", "memwhen", "memfetch")


def switch(req, name):
    """读取开关；缺省关闭，保证未接线时行为与 4.8.7 一致。"""
    value = req.get(name)
    if value is None:
        return False
    return bool(value)


def entry_of(req):
    return str(req.get("entry") or "automatic").strip().lower()


def is_active_tool(req):
    return entry_of(req) in ACTIVE_TOOLS


def session_of(req):
    return str(req.get("index") or "") or "-"


def _expire(now):
    stale = [key for key, item in WINDOW.items()
             if now - float(item.get("stamp") or 0) > WINDOW_TTL_SECONDS]
    for key in stale:
        WINDOW.pop(key, None)


def gate_allows(query):
    """入口门控：判断这句话是否值得自动召回。

    只看可解释的表面特征，不做语义猜测：空串、过长无实义、纯寒暄都不触发。
    真正的相关性仍由出口侧的关键词与打分决定。
    """
    q = (query or "").strip()
    if len(q) < 2:
        return False, "query too short"
    if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in q):
        return False, "query has no content"
    plain = q.strip("，。！？、,.!?~～ \t")
    if plain in SMALLTALK:
        return False, "smalltalk"
    return True, ""


def fingerprint(fact):
    """同一张卡的同一版正文才是同一指纹。"""
    return "%s@%s" % (str(fact.get("id") or ""),
                      str(fact.get("version") or fact.get("text_hash") or ""))


def prepare(session, tokens, topic_filter):
    """取会话窗口；开启话题断层时，换话题先清窗（顺序不可反）。"""
    now = time.time()
    _expire(now)
    current = set(tokens or ())
    window = WINDOW.setdefault(session, {"tokens": set(), "prints": set(), "gen": 0})
    window["gen"] = int(window.get("gen") or 0) + 1
    window["stamp"] = now
    if topic_filter:
        previous = window.get("tokens") or set()
        same = textsim._jaccard(previous, current) >= TOPIC_JACCARD if previous and current else True
        if not same:
            window["prints"] = set()
    # 新一轮召回开始：上一次的返回集合立即作废，迟到回执不能借交集通过校验
    window["returned"] = set()
    window["tokens"] = current
    return window


def suppress(facts, window):
    """出口压制：本窗口内已注入过的同一指纹不再重复注入。"""
    prints = window.get("prints") or set()
    kept, dropped = [], []
    for fact in facts or []:
        if fingerprint(fact) in prints:
            dropped.append(fact)
        else:
            kept.append(fact)
    return kept, dropped


def note_injected(session, facts, window_gen=None):
    """成功交付后登记指纹。只接受召回时那一代的回执，迟到回执不得污染新话题。"""
    if not facts:
        return 0
    now = time.time()
    window = WINDOW.setdefault(session, {"tokens": set(), "prints": set(), "gen": 0})
    if window_gen is not None and int(window_gen) != int(window.get("gen") or 0):
        return 0
    allowed = window.get("returned")
    if allowed is not None:
        facts = [fact for fact in facts if fingerprint(fact) in allowed]
    window["stamp"] = now
    prints = window.setdefault("prints", set())
    added = 0
    for fact in facts:
        mark = fingerprint(fact)
        if mark not in prints:
            prints.add(mark)
            added += 1
    return added


def remember_returned(session, facts):
    """记录本次召回真正返回的指纹；成功回执只认这份交集。

    这样即使宿主延迟提交上一话题的回执，它携带的指纹也不在当前交集里，
    不会写入新话题窗口。
    """
    window = WINDOW.setdefault(session, {"tokens": set(), "prints": set(), "gen": 0})
    window["returned"] = {fingerprint(fact) for fact in (facts or [])}


def clear(session=None):
    if session is None:
        WINDOW.clear()
    else:
        WINDOW.pop(session, None)
