"""索引加载（数据库优先）与查询词表、IDF。"""

import os
import json


# >>> WP1 导入（自动生成，勿手改）
import dbside
from dbside import _row_entry
from state import CJK_RUN_RE, WORD_RE, _IDX, _cache_put, _find_index, _warn
from textsim import _field_text
# <<< WP1 导入

def _load_index():
    """加载索引（带缓存）。显式路径给了但文件不在 → 返回空表，绝不回退到别人的索引。"""
    p = _IDX["path"]
    if p is None:
        p = _find_index()
        _IDX["path"] = p
    # ★数据库优先：建过库就以库为准（JSON 只用于首次迁移/导出）
    #   不能要求连接"已经打开"：侧车刚启动时第一个 op 可能是画像/压缩预览/时间查询，
    #   那时还没人碰过库；这里要自己会开，否则会把索引误读成空。
    try:
        if p and _IDX["data"] is None:
            _dbp = os.path.join(os.path.dirname(p), "memory.db")
            if os.path.isfile(_dbp):
                _c = dbside._db_for(p)
                if _c is not None:
                    rows = [_row_entry(_c, r) for r in _c.execute("SELECT * FROM entries")]
                    _IDX["data"] = rows
                    return rows
    except Exception as e:
        # 库优先策略的回退点：失败要能定位，否则只会表现为「索引是空的」
        _warn("load index from database: %s" % p, e)
    if _IDX["data"] is None:
        if p and os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    _IDX["data"] = json.load(f)
            except Exception as e:
                _warn("read index json: %s" % p, e)
                _IDX["data"] = []
            _cache_put(_IDX["path"], "data", _IDX["data"])
        else:
            return []   # 显式路径缺失：空表，不缓存，等 build 出来再读
    return _IDX["data"]


def _vocab():
    """从索引收集全部标签词（人/事/物），长词优先，供最长匹配切词。"""
    if _IDX["vocab"] is None:
        words = set()
        for e in _load_index():
            for field in ("people", "topics", "things"):
                for w in e.get(field) or []:
                    if w:
                        words.add(w)
        # 长的排前面：保证"血压计"不会被切成"血压"
        _IDX["vocab"] = sorted(words, key=lambda w: -len(w))
        _cache_put(_IDX["path"], "vocab", _IDX["vocab"])
    return _IDX["vocab"]


def tokenize(query):
    """把一句中文切成查询词。标签词最长匹配 + 汉字 2 字滑窗 + 英文串。"""
    if not query:
        return []
    q = query.lower()
    terms = []
    used = []

    def covered(pos):
        return any(a <= pos < b for a, b in used)

    # 1) 标签词最长匹配
    for w in _vocab():
        lw = w.lower()
        start = 0
        while True:
            p = q.find(lw, start)
            if p < 0:
                break
            start = p + 1
            if covered(p):
                continue
            terms.append(lw)
            used.append((p, p + len(lw)))
    # 2) 汉字串 2 字滑窗
    for m in CJK_RUN_RE.finditer(q):
        run = m.group()
        base = m.start()
        if len(run) == 1:
            if not covered(base):
                terms.append(run)
            continue
        # 短串整体成词：提高精度（"帕尔曼"整体命中 > 拆成"帕尔"+"尔曼"）。
        # ★只对 4 字以内的实体名这么做；5 字以上基本是句子（"我前天体温多少"），
        #   整句成一个词就再也切不出"体温"，召回直接空转，所以整句不占位、照常滑窗。
        if len(run) <= 4 and not covered(base):
            terms.append(run)
            used.append((base, base + len(run)))
        for i in range(len(run) - 1):
            pos = base + i
            if covered(pos) and covered(pos + 1):
                continue
            terms.append(run[i:i + 2])
    # 3) 英文/数字串
    for m in WORD_RE.finditer(q):
        if not covered(m.start()):
            terms.append(m.group().lower())

    seen = set()
    out = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _df():
    """每个词在多少条里出现过（文档频率），用来给常见词降权。"""
    if _IDX.get("df") is not None:
        return _IDX["df"]
    df = {}
    for e in _load_index():
        tags = " ".join([_field_text(e, "people"), _field_text(e, "topics"),
                         _field_text(e, "things"), _field_text(e, "head"),
                         _field_text(e, "summary"), _field_text(e, "text")]).lower()
        for t in set(tokenize(tags)):
            df[t] = df.get(t, 0) + 1
    _IDX["df"] = df
    _cache_put(_IDX["path"], "df", df)
    return df


def _idf(term, n):
    """常见词权重小（"真央"满篇都是就没区分度），稀罕词权重大。"""
    import math
    df = _df().get(term, 0)
    return math.log((n + 1.0) / (df + 1.0)) + 1.0
