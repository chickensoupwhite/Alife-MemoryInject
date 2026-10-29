"""侧车共享可变状态、静态表与纯状态变更。所有对象只被就地修改，不做重绑定。"""

import os
import re
import sys

# >>> WP1 导入（自动生成，勿手改）
# <<< WP1 导入

_HERE = os.path.dirname(os.path.abspath(__file__))


W_TAG = 3      # people / topics / things 标签命中


W_HEAD = 3     # 块标题命中


W_SUMMARY = 2  # 首句摘要命中


W_TEXT = 1     # 正文命中


CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


WORD_RE = re.compile(r"[A-Za-z0-9]+")


_IDX = {"path": None, "data": None, "vocab": None, "df": None}


# 多角色共用同一个侧车进程：按"索引路径"分桶缓存，切角色时不必反复冷读。
_CACHE = {}         # path -> {"data":..., "vocab":..., "df":...}


_CACHE_ORDER = []   # LRU 顺序（最近用的排后）


_CACHE_MAX = 3      # 最多缓存几份完整索引（三四个角色够用，不无限吃内存）


def _cache_put(path, key, val):
    if not path:
        return
    c = _CACHE.setdefault(path, {"data": None, "vocab": None, "df": None})
    c[key] = val
    if path in _CACHE_ORDER:
        _CACHE_ORDER.remove(path)
    _CACHE_ORDER.append(path)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)


def _find_index():
    """找 index.json：先看脚本同目录，再看上级目录。"""
    for c in (os.path.join(_HERE, "index.json"),
              os.path.join(os.path.dirname(_HERE), "index.json")):
        if os.path.isfile(c):
            return c
    return None


def _set_index(path):
    """切换当前使用的索引文件（多角色各自一份，不打架）。
    命中分桶缓存就直接用热的，避免多角色来回切时反复冷读。
    ★显式路径即使文件暂时不在也要记下来：绝不回退到别人的索引。"""
    if path and _IDX.get("path") != path:
        _GRAPH["built"] = False      # ★换索引=换图：不清就会拿上一个角色的图去扩词
        _IDX["path"] = path
        c = _CACHE.get(path)
        if c is not None:
            _IDX["data"] = c.get("data")
            _IDX["vocab"] = c.get("vocab")
            _IDX["df"] = c.get("df")
            if path in _CACHE_ORDER:
                _CACHE_ORDER.remove(path)
            _CACHE_ORDER.append(path)
        else:
            _IDX["data"] = None
            _IDX["vocab"] = None
            _IDX["df"] = None


_LIFE = {"dirty": False, "last_write": 0.0}


_CFG = {"half_life": 30, "block_max": 200}   # 运行时可配（由插件配置传入）


DECAY_HALF_LIFE_DAYS = 30      # 半衰期：30 天没碰，权重减半（默认值，可被 _CFG 覆盖）


DECAY_FLOOR = 0.05             # 衰减下限，别归零（怕误杀重要旧账）


SOFT_DELETE_DAYS = 60          # 从没访问过且老过这么久 → 才考虑软删（门槛要高，防误杀）


HARD_DELETE_DAYS = 30          # 软删后再过这么久 → 物理清


_GRAPH = {"nodes": {}, "edges": {}, "built": False}


PEOPLE = {
    "主人": ["主人", "570167521", "白姬汤"],
    "小鲸鱼": ["小鲸鱼", "鲸鱼娘"],
    "俾斯麦": ["俾斯麦"],
    "妈": ["妈妈", "他妈", "母亲", "咱妈"],
    "麦乐迪": ["麦乐迪"],
}


TOPICS = {
    "健康": ["血压", "低压", "高压", "睡眠", "早睡", "饮食", "头痛", "头晕", "耳鸣", "减肥", "禁欲", "体重", "运动", "吃药", "晕厥"],
    "学习": ["综应", "职测", "资料分析", "案例分析", "作文", "背诵", "考点", "速算", "逻辑", "常识", "辨析", "讲义", "真题", "笔记", "班会", "研学"],
    "技术": ["插件", "脚本", "ComfyUI", "模型", "配置", "重启", "代码", "索引", "端口", "内存", "显存", "工作流", "regedit", "注册表"],
    "约定": ["约定", "陪跑", "记账", "提醒", "打卡"],
    "关系": ["晚安", "摸头", "工具人", "名字", "重要", "真央", "立绘"],
    "生活": ["吃饭", "天气", "衣服", "Steam", "游戏", "群聊", "早餐", "午餐", "晚餐"],
}


THINGS = {
    "血压计": ["血压计", "血压"],
    "粉笔": ["粉笔"],
    "ComfyUI": ["ComfyUI", "comfyui"],
    "Obsidian": ["Obsidian", "obsidian"],
    "QQ": ["QQ", "qchat", "QuickChat"],
    "网易云": ["网易云"],
    "MC": ["MC", "我的世界", "minecraft"],
    "街霸": ["街霸"],
    "桌宠": ["桌宠"],
    "屏幕": ["屏幕"],
    "CodeBuddy": ["CodeBuddy", "codebuddy", "buddy"],
    "记忆插件": ["MemoryInject", "记忆插件", "RecallHint"],
}


PERIOD_HOUR = {"凌晨": 3, "早上": 8, "早晨": 8, "上午": 10, "中午": 12, "午后": 14,
               "下午": 15, "傍晚": 18, "晚": 20, "晚上": 21, "夜": 22, "深夜": 23}


_MODEL = {"ready": False, "warmed": False, "tok": None, "mdl": None, "device": "unavailable",
          "torch": None, "F": None, "err": ""}


_VEC_CACHE = {}


_NEG = ("没", "未", "不", "取消", "放弃", "作废")


_ENTITY_PATTERNS = [
    re.compile(r"《([^》]{1,20})》"),
    re.compile(r"「([^」]{1,20})」"),
    re.compile(r"『([^』]{1,20})』"),
    re.compile(r"“([^”]{1,30})”"),
    re.compile(r"‘([^’]{1,30})’"),
    re.compile(r'"([^"\r\n]{1,30})"'),
    re.compile(r"([\u4e00-\u9fff]{2,6}(?:插件|项目|模型|客户端|服务|工具|游戏|软件))"),
]


_ENTITY_STOP = set(["什么", "这个", "那个", "我们", "他们", "自己", "现在", "今天", "昨天", "明天"])


_DB = {"ok": False, "path": "", "conn": None}


_GRAPH_CACHE = {"path": "", "conn": None}


_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(
  id TEXT PRIMARY KEY, anchor TEXT, source TEXT,
  time TEXT, date TEXT, period TEXT, hour INTEGER,
  head TEXT, summary TEXT, body TEXT, hay TEXT, text_hash TEXT,
  people TEXT, topics TEXT, things TEXT,
  access_count INTEGER DEFAULT 0, last_access TEXT,
  retired INTEGER DEFAULT 0, retired_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_entries_time ON entries(time);
CREATE INDEX IF NOT EXISTS ix_entries_source ON entries(source);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL);
CREATE TABLE IF NOT EXISTS vec(id TEXT PRIMARY KEY, data BLOB);
CREATE TABLE IF NOT EXISTS nodes(name TEXT PRIMARY KEY, n INTEGER);
CREATE TABLE IF NOT EXISTS edges(a TEXT, b TEXT, w INTEGER, type TEXT, PRIMARY KEY(a, b));
CREATE TABLE IF NOT EXISTS call_events(
  logical_id TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  fact_version TEXT NOT NULL,
  channel TEXT NOT NULL,
  called_at TEXT NOT NULL,
  PRIMARY KEY(logical_id, fact_id)
);
CREATE INDEX IF NOT EXISTS ix_call_events_fact_time ON call_events(fact_id, called_at);
CREATE TABLE IF NOT EXISTS call_stats_meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS call_stats_gaps(
  started_at TEXT PRIMARY KEY,
  ended_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS person_impressions(
  name TEXT PRIMARY KEY,
  impression_text TEXT NOT NULL,
  fact_ids TEXT NOT NULL,
  fact_versions TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  is_stale INTEGER NOT NULL DEFAULT 0 CHECK(is_stale IN (0,1)),
  model_source TEXT
);
CREATE INDEX IF NOT EXISTS ix_person_impressions_stale ON person_impressions(is_stale);
"""


_PROFILE_KIND_COLUMNS = {
    "person": "people",
    "event": "topics",
    "object": "things",
}


_WARNED = set()


def _warn(context, error):
    """记录一次可诊断的降级失败。

    外部边界（文件、SQLite、模型、HTTP）失败时仍然降级继续，但不允许静默：
    输出异常类型、异常本身与发生位置到 stderr。同一「位置＋异常类型」只报一次，
    因为读取类操作在每次召回都会执行，重复播报会把真正有用的日志淹掉。
    """
    key = (context, type(error).__name__)
    if key in _WARNED:
        return
    _WARNED.add(key)
    sys.stderr.write("[memoryinject] %s failed: %s: %s\n"
                     % (context, type(error).__name__, error))
    sys.stderr.flush()
