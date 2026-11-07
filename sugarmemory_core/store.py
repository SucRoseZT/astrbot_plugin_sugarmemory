"""存储层：只负责存取。约束在存储层强制执行：
- L2 表 CHECK 约束 actionable=0、layer 固定 'L2'；
- L3 表 CHECK 约束 actionable=0、layer 固定 'L3'；
- write_l2 必须携带 allow 裁决，否则抛 PermissionError；
- 不提供绕过门控写 L2 的公开方法；
- L2 与 L3 查询接口完全隔离，绝不 union。
"""
import hashlib
import math
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter

from .social import DEFAULT_STAGES, DEFAULT_STAGE_DESCS

# 短期记忆的「一条」= 一次对话，不是一条消息。
# 与面板 pages/记忆中枢 的 CONV_GAP_MS 保持一致：30 分钟内、且没切会话算同一段。
CONV_GAP_SECONDS = 30 * 60

# 多值分类：同一个人名下允许多条（不同的事/爱好/经历）。与 summarize_core 的 _MULTI_VALUED 保持一致（7 类）。
_MULTI_VALUED_PREDICATES = {"爱好技能", "生活习惯", "经历", "打算", "评价认为", "自我相关", "其他"}


def is_multi_valued(predicate: str) -> bool:
    return (predicate or "").strip() in _MULTI_VALUED_PREDICATES


def bot_of_session(session) -> str:
    """从 unified_msg_origin 取机器人命名空间：第一段就是平台实例 id。

    一个平台实例 = 一个机器人账号（一个 QQ 号一条连接），所以
    `111:FriendMessage:100010001` 属于机器人 `111`。多账号共用一份数据库时，
    记忆必须按这个维度隔离：否则 A 机器人给用户发的定时提醒，会被 B 机器人
    当成「自己说过的话」写进记忆包。

    没有冒号（手动补录、`import:` 导入）的会话不归属任何机器人，返回 ''，
    这类记录按「共享」处理。
    """
    s = str(session or "").strip()
    if not s or ":" not in s:
        return ""
    head = s.split(":", 1)[0].strip()
    if not head or head == "import":
        return ""
    return head[:32]


# ---------- 遗忘曲线：保留度 ----------
#
# 这里刻意不用「被提了多少次」当遗忘依据。次数只会越攒越多，两年前很火、
# 现在没人提的条目永远最热；而且静态事实（病史、联系方式、家人）本来就不
# 需要每轮被提起，按次数淘汰第一批死的就是它们。真正该看的是「隔了多久
# 没碰到」——这是遗忘曲线的自变量，次数只作为稳定度的奖励项进分母。

# 记忆强度的时间常数（天）。S = BASE × 确认次数加成 × 置信度加成。
# 标定：新事实 S≈35 天，90 天不碰降到 7%，半年基本沉底；
# 反复确认过 5 次的 S≈140 天，一年不碰还剩一半。
RETENTION_BASE_DAYS = 30.0
# 低到这个程度才允许自动归档（约等于 3.5 个时间常数没被任何方式碰过）
RETENTION_ARCHIVE_BELOW = 0.02
# 归档前至少静默这么久。防止刚补录的一批老事实当天就被收走。
RETENTION_ARCHIVE_MIN_DAYS = 90

# 错峰沉底：同一个人几百条事实往往是同一周攒下来的，置信度也都在 0.9 上下，
# 于是 S 几乎一样、过线时间几乎一样 —— 遗忘变成「半年后一次性清仓」。
# 给每条一个 0~N 天的确定性偏移（只由 id+subject 决定，改配置不会重排全库），
# 让它到点时不是齐步走，而是两三个月里陆续沉。
# 这只是观感开关：改的是「哪天沉」，不是「该不该沉」，所以默认关。
# 下面这个数只作配置项的默认值参考，实际跨度由 store.stagger_days 决定。
RETENTION_STAGGER_DAYS = 75.0

# 「人的定义」：身份、疾病、年龄、家庭、联系方式。它们本来就不该每轮被提起，
# 按热度淘汰等于把人的底档删了。这一类只降权，永不自动归档。
PERMANENT_PREDICATES = {"自我相关", "其他"}


def _last_touch(row) -> float:
    """最后一次「碰到」这条事实的时刻：被注入、被重新确认、或事实本身发生。

    刻意不含 stored_at（入库时间）。补录半年前的聊天时，created_at 是半年前，
    stored_at 是今天；如果拿 stored_at 算 Δt，这批老事实会全部显示成刚聊过，
    遗忘曲线直接失效。入库时间只用来算「观察窗口」和防止刚导入就被归档。
    """
    return max(
        float(row.get("last_hit_at") or 0),
        float(row.get("updated_at") or 0),
        float(row.get("created_at") or 0),
    )


def retention_score(row, now: float | None = None) -> float:
    """这条事实现在还记得多牢，0~1。

    R = exp(-Δt / S)
      Δt = 距最后一次「碰到它」多少天（被注入、被重新确认、入库，取最近那个）
      S  = 记忆强度 = 基础寿命 × (1 + 0.5×确认次数) × (0.6 + 0.8×置信度)

    确认次数封顶 10 次：说了十遍的事再强调也不会更牢，只会让 S 无限膨胀，
    最后谁都沉不下去。
    """
    now = now or time.time()
    last = _last_touch(row)
    if last <= 0:
        return 1.0
    dt = max(0.0, (now - last) / 86400.0)
    stab = max(0, min(int(row.get("stability") or 0), 10))
    conf = max(0.0, min(1.0, float(row.get("confidence") or 0.6)))
    s = RETENTION_BASE_DAYS * (1.0 + 0.5 * stab) * (0.6 + 0.8 * conf)
    if s <= 0:
        return 0.0
    return math.exp(-dt / s)


def retention_quiet_days(row, now: float | None = None) -> float:
    """距最后一次碰到它，静默了几天。归档判定的第二个条件。"""
    now = now or time.time()
    last = _last_touch(row)
    if last <= 0:
        return 0.0
    return max(0.0, (now - last) / 86400.0)


def retention_stagger_days(row, span: float | None = None) -> float:
    """这条事实自己的一次性相位偏移，0~span 天。

    只由 (id, subject) 决定：同一条永远得到同一个偏移，改了偏移天数也不会
    让全库相位重排；不同条目均匀散开，避免同批入库的条目同一天集体过线。
    没有 id 的行（测试里手搓的 dict）给 0，保持可预测。
    span 不传就是 0（不错峰）——错峰是可选项，不能因为某个调用点忘了传配置
    就悄悄生效，那样面板倒计时和清理会差出几十天。
    """
    span = 0.0 if span is None else float(span)
    if span <= 0:
        return 0.0
    try:
        fid = int(row.get("id") or 0)
    except (TypeError, ValueError):
        fid = 0
    seed = f"{fid}|{str(row.get('subject') or '')}"
    frac = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:4], "big") / 0xFFFFFFFF
    return frac * span


def retention_ttl_days(row) -> float:
    """从最后一次碰到到「保留度掉到归档线」需要多少天（不含静默闸门/入库闸门）。"""
    stab = max(0, min(int(row.get("stability") or 0), 10))
    conf = max(0.0, min(1.0, float(row.get("confidence") or 0.6)))
    s = RETENTION_BASE_DAYS * (1.0 + 0.5 * stab) * (0.6 + 0.8 * conf)
    if s <= 0:
        return 0.0
    return s * math.log(1.0 / RETENTION_ARCHIVE_BELOW)


def retention_archive_after(row, stagger: float | None = None) -> float:
    """真正允许沉底前要静默多少天：曲线寿命 + 本条自己的相位偏移。

    相位只认行里塞好的 `_stagger`（store 读库时按自己的配置算）。裸行一律按 0 算：
    错峰是「库」的策略，不该由几个模块级常量偷偷决定，否则关掉开关后
    面板上的倒计时还会带着偏移。
    """
    if stagger is None:
        stagger = float(row.get("_stagger") or 0.0) if "_stagger" in row else 0.0
    return retention_ttl_days(row) + max(0.0, float(stagger))


def retention_days_left(row, now: float | None = None) -> float:
    """还要多久掉到归档线（天）。已经过线返回 0，永久/图钉返回 -1。

    面板上的「约 N 天后沉底」直接用这个，别在前端重算公式——
    参数一改两头不同步，显示的倒计时就成了假的。
    """
    if int(row.get("pinned") or 0):
        return -1.0
    if str(row.get("predicate") or "").strip() in PERMANENT_PREDICATES:
        return -1.0
    now = now or time.time()
    quiet = retention_quiet_days(row, now)
    if retention_score(row, now) <= RETENTION_ARCHIVE_BELOW:
        left = 0.0
    else:
        left = retention_archive_after(row) - quiet
    # 静默闸门和入库闸门跟曲线是「同时满足」的关系，倒计时取最晚的那个。
    # 以前 R 一过线就直接 return 0，把两个闸门跳过了——补录半年前的聊天时，
    # 面板会写「约 0 天后沉底」，其实还得等满 90 天观察期，等于给了个假倒计时。
    left = max(left, RETENTION_ARCHIVE_MIN_DAYS - quiet)
    stored = float(row.get("stored_at") or 0)
    if stored:
        left = max(left, RETENTION_ARCHIVE_MIN_DAYS - (now - stored) / 86400.0)
    return max(0.0, left)
# 阶段总结不单独成段，挂到时间最近的对话上（15 分钟内允许尾部对齐，1 天内有效）。
SUMMARY_ATTACH_AFTER = 300
SUMMARY_ATTACH_WINDOW = 86400

# 人名里那些看不见的装饰：零宽字符、方向控制符、BOM
# 再加上韩文填充符（U+115F/U+1160/U+3164/U+FFA0）——QQ 昵称里常塞这个占位，
# 「李四喵」和「李四ᅠ⁧喵ᅠ」其实是同一个人，肉眼根本看不出区别。
_ZW_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u115f\u1160\u3164\uffa0]")
# 名字里的备注括号：「张三（摸鱼）」「张三(老板)」「张三【群友】」
_BRACKET_RE = re.compile(r"[\(（\[\【\{｛][^\)）\]\】\}｝]{0,24}[\)）\]\】\}｝]")


def alias_key(name: str) -> str:
    """人名归一键：同一个人被写成几种样子时，这个键相同。

    只做「无损」归并——去零宽字符、全半角统一、去掉备注括号、压空白、忽略大小写。
    不做包含匹配（「路人甲」和「路人甲妈妈」是两个人），那种交给面板手动合并。
    """
    s = _ZW_RE.sub("", str(name or ""))
    s = unicodedata.normalize("NFKC", s)
    stripped = _BRACKET_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", stripped or s).strip()
    return s.casefold()


def pinyin_key(name: str) -> str:
    """读音归一键：「丙哥」和「用户丙」得到同一个键 用户丙。

    比 alias_key 松，所以只用来**提示**、不用来自动合并：同音不同人
    （王伟 / 王薇）在中文里太常见，自动合会把人并错。pypinyin 没装就返回
    空串，整套降级成「只认字面」的老行为，不会报错。
    """
    s = alias_key(name)
    if not s:
        return ""
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return ""
    try:
        parts = lazy_pinyin(s, errors=lambda x: list(x))
    except Exception:
        return ""
    key = re.sub(r"[^a-z0-9]", "", "".join(parts).casefold())
    # 纯西里尔字母之类读不出拼音的名字会得到空串，别把它当成「所有无名氏相等」
    return key



def count_conversations(rows: list[dict]) -> int:
    """把 L3 行切成「一次对话」，返回段数。规则与面板完全一致。"""
    def _key(r):
        return (float(r.get("created_at") or 0), int(r.get("id") or 0))

    msgs = sorted([r for r in rows if r.get("event_type") != "session_summary"], key=_key)
    sums = sorted([r for r in rows if r.get("event_type") == "session_summary"], key=_key)
    convs: list[dict] = []
    for r in msgs:
        ts = float(r.get("created_at") or 0)
        sid = r.get("session_id") or ""
        if not convs:
            convs.append({"end": ts, "sid": sid})
            continue
        cur = convs[-1]
        switched = bool(cur["sid"] and sid and cur["sid"] != sid)
        if ts - cur["end"] > CONV_GAP_SECONDS or switched:
            convs.append({"end": ts, "sid": sid})
        else:
            cur["end"] = max(cur["end"], ts)
            if not cur["sid"]:
                cur["sid"] = sid
    for r in sums:
        ts = float(r.get("created_at") or 0)
        target = None
        for c in convs:
            if c["end"] <= ts + SUMMARY_ATTACH_AFTER and ts - c["end"] <= SUMMARY_ATTACH_WINDOW:
                target = c
        if target:
            target["end"] = max(target["end"], ts)
        else:
            convs.append({"end": ts, "sid": r.get("session_id") or ""})
    return len(convs)


SCHEMA = """
CREATE TABLE IF NOT EXISTS l2_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  layer TEXT NOT NULL DEFAULT 'L2' CHECK (layer = 'L2'),
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  value TEXT NOT NULL,
  speaker TEXT NOT NULL,
  speech_act TEXT NOT NULL DEFAULT 'statement',
  source_type TEXT NOT NULL,
  source_msg_id TEXT NOT NULL DEFAULT '',
  source_quote TEXT NOT NULL DEFAULT '',
  source_summary_id INTEGER NOT NULL DEFAULT 0,
  actionable INTEGER NOT NULL DEFAULT 0 CHECK (actionable = 0),
  confidence REAL NOT NULL DEFAULT 0.6,
  status TEXT NOT NULL DEFAULT 'active',
  stability INTEGER NOT NULL DEFAULT 0,
  expires_at REAL NOT NULL DEFAULT 0,
  replaced_by INTEGER NOT NULL DEFAULT 0,
  gate_reason TEXT NOT NULL DEFAULT '',
  fingerprint TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  ask_alias TEXT NOT NULL DEFAULT '',
  hit_count INTEGER NOT NULL DEFAULT 0,
  last_hit_at REAL NOT NULL DEFAULT 0,
  stored_at REAL NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_l2_subj ON l2_facts(subject, status);
CREATE INDEX IF NOT EXISTS idx_l2_fp ON l2_facts(fingerprint);

CREATE TABLE IF NOT EXISTS l3_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  layer TEXT NOT NULL DEFAULT 'L3' CHECK (layer = 'L3'),
  event_type TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  summary_id INTEGER NOT NULL DEFAULT 0,
  content TEXT NOT NULL,
  source_msg_id TEXT NOT NULL DEFAULT '',
  actionable INTEGER NOT NULL DEFAULT 0 CHECK (actionable = 0),
  expires_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  bot_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_l3_type ON l3_events(event_type, created_at);

CREATE TABLE IF NOT EXISTS gate_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate TEXT NOT NULL,
  decision TEXT NOT NULL,
  reason TEXT NOT NULL,
  source_type TEXT NOT NULL DEFAULT '',
  speech_act TEXT NOT NULL DEFAULT '',
  target_layer TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS affinity (
  subject TEXT PRIMARY KEY,
  xp REAL NOT NULL DEFAULT 0,
  gain_total REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS affinity_stages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  threshold REAL NOT NULL,
  sort INTEGER NOT NULL DEFAULT 0,
  desc TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS social_relations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  relation TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.6,
  source_type TEXT NOT NULL DEFAULT 'summary',
  session_id TEXT NOT NULL DEFAULT '',
  summary_id INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(source, target)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON social_relations(source);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON social_relations(target);

CREATE TABLE IF NOT EXISTS relation_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT '',
  quote TEXT NOT NULL DEFAULT '',
  source_type TEXT NOT NULL DEFAULT 'summary',
  session_id TEXT NOT NULL DEFAULT '',
  summary_id INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evid_pair ON relation_evidence(source, target, created_at);

CREATE TABLE IF NOT EXISTS token_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  module TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'actual',
  provider_id TEXT NOT NULL DEFAULT '',
  subject TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  input INTEGER NOT NULL DEFAULT 0,
  cached INTEGER NOT NULL DEFAULT 0,
  output INTEGER NOT NULL DEFAULT 0,
  total INTEGER NOT NULL DEFAULT 0,
  -- 2026-11-06 可观测性：把「这一轮用的是哪份前缀、是否刚重建」跟真实
  -- cached_tokens 绑在一起，才能反推冻结策略的收益。arch_ver 区分新旧架构，
  -- 否则升级后旧动态包的数据和冻结包混在一张表里，A/B 无从谈起。
  arch_ver TEXT NOT NULL DEFAULT '',
  inject_fp TEXT NOT NULL DEFAULT '',
  rebuilt INTEGER NOT NULL DEFAULT 0,
  freeze_rounds INTEGER NOT NULL DEFAULT 0,
  cfg_ver TEXT NOT NULL DEFAULT '',
  -- 2026-11-06 归因列：重建到底是因为改了配置、L2 改写、节流到点，
  -- 还是「插件重载后内存缓存被清空」。最后一种（cold/reload）字节往往
  -- 一个都没变，provider 缓存理论上还在，整轮 0% 属于固定正常，不该算失败样本。
  rebuild_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tok_ts ON token_usage(ts);
CREATE INDEX IF NOT EXISTS idx_tok_mod ON token_usage(module, ts);

-- 幂等锁：工具写操作在时间窗内只落一次
CREATE TABLE IF NOT EXISTS tool_idem (
  key TEXT PRIMARY KEY,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_idem_ts ON tool_idem(ts);

-- 熔断状态：连续失败计数 + 冷却起点，落库所以重载不丢
CREATE TABLE IF NOT EXISTS circuit (
  name TEXT PRIMARY KEY,
  fails INTEGER NOT NULL DEFAULT 0,
  opened_at REAL NOT NULL DEFAULT 0,
  last_fail REAL NOT NULL DEFAULT 0,
  last_ok REAL NOT NULL DEFAULT 0
);

-- 见过的群：用来把「群」从人名列表里摘出去，群号/群名都可能被人当成"人"存进来
CREATE TABLE IF NOT EXISTS known_groups (
  gid TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0
);

-- 人物身份：一个人 = 一个 QQ + 一串称呼。person 是内部稳定键（第一次见到的
-- 称呼），QQ 才是身份锚——同一个 QQ 改了昵称、换了马甲，都还是同一个人。
-- display_name 是最新的 QQ 昵称，call_name 是「能被称呼的名」，两个都随 QQ 走。
CREATE TABLE IF NOT EXISTS people (
  person TEXT PRIMARY KEY,
  qq TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  display_name TEXT NOT NULL DEFAULT '',
  call_name TEXT NOT NULL DEFAULT '',
  call_locked INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

-- 称呼 → 人物。alias_key 是字面归一键（唯一），pinyin 是读音键（只用于提示）。
-- tier 分层：0=显示名 1=能被称呼的名 2=绰号外号 3=叫错的(可自动忘)
-- last_seen 是最近一次被提到的时间，只有 tier 3（和长期没人叫的 tier 2）会被清。
CREATE TABLE IF NOT EXISTS person_alias (
  alias_key TEXT PRIMARY KEY,
  person TEXT NOT NULL,
  raw TEXT NOT NULL DEFAULT '',
  pinyin TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'auto',
  tier INTEGER NOT NULL DEFAULT 2,
  hits INTEGER NOT NULL DEFAULT 0,
  last_seen REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_palias_person ON person_alias(person);
CREATE INDEX IF NOT EXISTS idx_palias_py ON person_alias(pinyin);

-- 补录提炼的累计处理量：只用来显示「累计已处理 N 条」，不参与进度边界。
-- 老版本的 last_id 是全局正序水位，留着当升级兼容读，新代码不再写它。
CREATE TABLE IF NOT EXISTS distill_cursor (
  subject TEXT PRIMARY KEY,
  last_id INTEGER NOT NULL DEFAULT 0,
  blocks INTEGER NOT NULL DEFAULT 0,
  items INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);

-- 补录提炼的按天水位：一天一条，记这一天已经提炼到哪条原话。
-- 为什么不复用 last_id：提炼从最近一天往前推，全局 id 是倒着走的，
-- 一个单调递增的游标根本表达不了「后面几天还没动」。
CREATE TABLE IF NOT EXISTS distill_day (
  subject TEXT NOT NULL,
  day TEXT NOT NULL,
  last_id INTEGER NOT NULL DEFAULT 0,
  items INTEGER NOT NULL DEFAULT 0,
  blocks INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (subject, day)
);
"""


def _fp(*parts: str) -> str:
    return hashlib.sha256("|".join(p.strip().lower() for p in parts).encode("utf-8")).hexdigest()[:32]


# 「同一件事、换了描述」在字面上最常见的三类差异，写指纹前先抹平：
#   1. 全角/半角数字与字母（５月 vs 5月）
#   2. 日期写法（5月20日 / 5/20 / 05-20 / 5.20）
#   3. 空白与标点（生日是 5 月 20 日 / 生日是5月20日）
# 这样 value 归一化后，「路人甲生日5月20日」和「路人甲生日 5/20」能落到同一个指纹。
_HD = "０１２３４５６７８９"
_HD_MAP = {ord(ch): str(i) for i, ch in enumerate(_HD)}
_AS_FW = {ord(ch): ch2 for ch, ch2 in zip("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
                                            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
                                            "abcdefghijklmnopqrstuvwxyz"
                                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
_DATE_RE = re.compile(
    r"(\d{1,4})\s*(?:年|/|\.|-)\s*(\d{1,2})\s*(?:月|/|\.|-)\s*(\d{1,2})\s*(?:日|号)?"
)
# 「5/20」「5.20」「5月20日」这种只有月和日、没有年的两段式日期（生日最常见）
_MD_RE = re.compile(
    r"(?<![\d])(\d{1,2})\s*(?:月|/|\.|-)\s*(\d{1,2})\s*(?:日|号)?(?![\d])"
)


def _norm_fingerprint(text: str) -> str:
    """把 value 拉平到「只剩语义骨架」的字串，专用于判重，不用于展示。"""
    s = str(text or "")
    s = s.translate(_HD_MAP).translate(_AS_FW)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # 日期统一成 YYYY-MM-DD（补零），「5/20」「5.20」「05-20」全归成同一种
    def _fix_date(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if len(y) <= 2:
            y = ("20" + y) if y and int(y) <= 30 else ("19" + y)
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    s = _DATE_RE.sub(_fix_date, s)
    # 两段式月/日 → 统一成 MM-DD，只保留「月日」这个语义（生日不关心哪一年）
    s = _MD_RE.sub(lambda m: f"{int(m.group(1)):02d}-{int(m.group(2)):02d}", s)
    # 去掉所有空白与标点，只留字母数字和 CJK
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s, flags=re.UNICODE)
    # 保守剥离纯语法虚词：只剥这几个无语义的系词/助词，让「生日是5月20日」
    # 和「生日5月20日」落到同一个指纹。绝不剥「有/在/喜欢/爱」这类携带语义的
    # 动词——那会把「有猫」和「猫」这种真差异抹平，误合并比漏合并更糟。
    for w in ("是", "的", "了", "就"):
        s = s.replace(w, "")
    return s


def _fp_norm(*parts: str) -> str:
    """归一化指纹：subject 与 predicate 走原样小写，value 走语义归一。"""
    normalized = []
    for i, p in enumerate(parts):
        if i >= 2:  # subject, predicate 之后的部分（value）做语义归一
            normalized.append(_norm_fingerprint(p))
        else:
            normalized.append(str(p).strip().lower())
    return hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()[:32]


# ---------- 「谁又叫什么」的识别 ----------
# 只认明说的几种句式，宁可漏也不瞎猜：把「就是」「也叫」「外号是」当成信号，
# 「A 是 B」这种普通的判断句不算（「他是朋友」里没有别名信息）。
_LABEL = r"[^\s，。,.!?？…、；;:：\"'“”‘’（）()\[\]【】]"
_BOUND = r"[\s，。,.!?？…、；;:：\"'“”‘’（）()\[\]【】]"
_ALIAS_PATTERNS = (
    # 用户丙 就是 丙哥 / 其实就是 / 原名是 / 本名是
    re.compile(
        rf"(?:^|(?<={_BOUND}))(?P<a>{_LABEL}{{1,20}}?)"
        rf"(?:就是|也就是|其实就是|其实是|本来是|原来是|原名是|本名是)"
        rf"(?P<b>{_LABEL}{{1,12}}?)(?={_BOUND}|$)"
    ),
    # 丙哥是 用户丙 的外号 / 绰号 / 小名 / 昵称 / 别名
    re.compile(
        rf"(?P<b>{_LABEL}{{1,20}}?)(?:(?:就)?是|也就是)(?P<a>{_LABEL}{{1,12}}?)"
        rf"的(?:外号|绰号|小名|昵称|别名|代号|英文名|真名|本名|另一个名字)"
    ),
    # 丙哥 也叫 / 又叫 / 又称 / 别名 / 外号叫 用户丙
    re.compile(
        rf"(?:^|(?<={_BOUND}))(?P<a>{_LABEL}{{1,20}}?)"
        rf"(?:也叫|又叫|也称|又称|也叫做|又叫做|别名是|别名|外号叫|外号是|绰号叫|绰号是)"
        rf"(?P<b>{_LABEL}{{1,12}}?)(?={_BOUND}|$)"
    ),
    # 用户丙 = 丙哥
    re.compile(
        rf"(?:^|(?<={_BOUND}))(?P<a>{_LABEL}{{1,20}})\s*[=＝]\s*(?P<b>{_LABEL}{{1,12}})"
        rf"(?={_BOUND}|$)"
    ),
)

# 这些词当不成人名：抓到了也只会把「他」登记成一个人的称呼
_ALIAS_STOP = {
    "", "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "这个", "那个",
    "这里", "那里", "什么", "啥", "谁", "自己", "别人", "大家", "有人", "一个人",
    "主人", "朋友", "好友", "恋人", "对象", "情人", "亲人", "家人", "熟人", "陌生人",
    "认识", "亲友", "关系", "记忆", "消息", "提醒", "定时", "任务", "名字", "外号",
    "绰号", "小名", "昵称", "别名", "真名", "本名", "代号", "英文名", "群主", "管理",
    "用户", "机器人", "ai", "bot", "鱼", "喵",
}

# 根本不能当人名的词：代词、泛称、系统占位。
# 这些一旦进了人物库，就会变成一个永远认不对的「人」，还会把真人的称呼抢走 ——
# 实机上「用户」就曾经被当成一个人登记，后来并进张三，成了他名下的一个假称呼。
_GENERIC_NAMES = {
    "", "我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "她们", "咱们",
    "自己", "对方", "这个人", "那个人", "某人", "有人", "大家", "各位",
    "用户", "群众", "群友", "网友", "陌生人", "路人", "朋友", "好友", "对象",
    "机器人", "助手", "系统", "管理员", "群主", "客服", "主人", "作者", "楼主",
    "user", "users", "someone", "somebody", "unknown", "anonymous", "anon",
    "ai", "bot", "assistant", "system", "nobody", "me", "you",
}
# 「用户12345」「user_abc」这种带尾巴的占位名同样不是人
_GENERIC_RE = re.compile(
    r"^(?:用户|匿名用户|用户名称|user|username|someone)[\s_\-]*[\w]{0,12}$", re.I)


def is_generic_name(name: str) -> bool:
    """这个词配不配当一个人的名字。

    认人靠名字、名字靠不住就靠 QQ。可泛称连名字都不是 —— 把它写进 people，
    下次同一个人说话就再也认不回来，只会多出一个叫「用户」的假人。
    """
    s = str(name or "").strip().strip(" 　[]【】()（）:：")
    if not s:
        return True
    k = s.casefold().strip()
    if k in _GENERIC_NAMES:
        return True
    return bool(_GENERIC_RE.match(k))


# 关系称谓：是「称呼」，不是「名字」。
#
# 泛称（用户/对方/大家）压根不是人，这里单独一套是因为它们**能**当称呼用：
# 「弟弟」「我妈」「女朋友」都是人，只是这个人已经以别的名字存在了。
# 一旦拿它当名字建一个新的人，结果就是一个永远认不回来、也永远清不掉的
# 假人——名下有 L2 就进 `_person_has_memory`，连称呼清理都绕过它。
# 实机上「XX 是我弟」就被总结出一个叫「弟弟」的新用户，用户看了气笑。
_KINSHIP_TERMS = frozenset({
    # 父母 / 长辈
    "爸", "爸爸", "老爸", "爹", "爹爹", "父亲", "妈", "妈妈", "老妈", "娘", "母亲",
    "父母", "爸妈", "家长", "爷爷", "阿公", "奶奶", "阿婆", "外公", "外婆",
    "姥爷", "姥姥", "祖父", "祖母", "干爹", "干妈", "义父", "义母", "岳父", "岳母",
    "公公", "婆婆",
    # 同辈
    "哥", "哥哥", "大哥", "二哥", "小哥", "兄长", "兄", "弟", "弟弟", "小弟", "老弟",
    "姐", "姐姐", "大姐", "小姐姐", "姊", "姊姊", "妹", "妹妹", "小妹", "老妹",
    "兄弟姐妹", "兄弟", "姐妹", "表哥", "表弟", "表姐", "表妹", "表兄",
    "堂哥", "堂弟", "堂姐", "堂妹", "嫂子", "姐夫", "妹夫", "弟妹", "弟媳",
    "小舅子", "小姨子", "大舅子", "大姨子",
    # 晚辈
    "儿子", "女儿", "孩子", "闺女", "娃", "宝宝", "侄子", "侄女", "外甥", "外甥女",
    "孙子", "孙女", "干儿子", "干女儿",
    # 亲戚
    "叔叔", "阿姨", "舅舅", "舅妈", "姑姑", "姑妈", "姑母", "姨妈", "姨",
    "姨父", "姨夫", "伯伯", "伯父", "婶婶", "婶子",
    # 伴侣（「老公」「老婆」也走这一档：它们是称呼，不是谁的名字）
    "老公", "老婆", "媳妇", "丈夫", "妻子", "夫人", "老伴", "男票", "女票",
    "男朋友", "女朋友", "男友", "女友", "未婚夫", "未婚妻", "前男友", "前女友",
    "初恋", "干哥", "干姐", "干弟", "干妹",
    # 日常关系 / 角色
    "同学", "同事", "室友", "同屋", "同桌", "校友", "学长", "学姐", "学弟", "学妹",
    "老师", "班主任", "导师", "教授", "老板", "上司", "领导", "下属", "客户", "甲方",
    "邻居", "房东", "网友", "群友", "基友", "闺蜜", "死党", "哥们", "姐们",
})
# 「我弟弟」「你的妈妈」这类带人称前缀的同样不是名字
_KINSHIP_PREFIX_RE = re.compile(r"^(?:我|你|您|他|她|俺|咱|本人|自己)?的?")


def is_kinship_alias(name: str) -> bool:
    """这个词是不是「关系称谓」而不是名字。

    只在**不许凭空造人**的地方用：`person_ensure` 没拿到 QQ、库里又没人
    认领这个名字时，称谓词一律不落地。已经在库里登记过的（用户自己在面板
    把「弟弟」绑给了某个人）照旧认得出来——那是明说的，不是猜的。
    """
    s = alias_key(name)
    if not s:
        return False
    s = _KINSHIP_PREFIX_RE.sub("", re.sub(r"\s+", "", s))
    return s in _KINSHIP_TERMS


def _is_qq_like(s) -> bool:
    """纯数字 5~12 位——QQ 号的样子，绝不是昵称。

    event_sender_name 拿不到昵称时会兜底返回 sender_id（QQ 号），这个值
    流进 person_ensure 就是灾难：tier=0 的「真实昵称」槽被一串号码占掉，
    面板上所有人的「QQ昵称」都变成数字（2026-11-06 实测：好几个人的
    显示名全是各自的 QQ 号）。号码该去的地方是别名表和
    people.qq 锚点，永远不该当名字写。
    """
    return bool(re.fullmatch(r"\d{5,12}", str(s or "").strip()))


def event_sender_name(event) -> str:
    """适配器给的名字：私聊是 QQ 昵称，**群聊是群名片**。"""
    name = ""
    try:
        name = event.get_sender_name() or ""
    except Exception:
        name = ""
    if not name:
        try:
            name = str(event.get_sender_id() or "")
        except Exception:
            name = ""
    return str(name or "").strip()


def speaker_key(store, event) -> str:
    """这条消息是谁说的 —— 拿去查记忆的主体。

    QQ 优先。群聊里 get_sender_name() 返回的是**群名片**（aiocqhttp 的
    sender.card），同一个人在不同群叫不同的名字，拿它去 person_resolve
    等于认不出：记忆包整个是空的，AI 在群里像是失忆。
    号是稳定的，先按号认，认不出才退回名字。
    """
    qq = ""
    try:
        qq = str(event.get_sender_id() or "")
    except Exception:
        qq = ""
    if qq:
        try:
            who = store.person_by_qq(qq)
            if who:
                return who
        except Exception:
            pass
    return event_sender_name(event)


# 句首的引语词：「她说 用户丙 就是滚木」里的「她说」不是名字的一部分。
# 只按固定词表剥，并且只在句首/标点之后剥，不然会把真名字削掉。
_LEAD_RE = re.compile(
    r"^(?:其实|对了|话说|还有|另外|顺便|然后|所以|而且|反正|好像|刚刚|刚才|"
    r"嗯|哦|诶|欸|唉|哎|喂|就是|"
    r"(?:我|你|他|她|它|ta|TA|有人|大家|别人|群里|上面|那个人)?"
    r"(?:跟你|跟我|跟他|跟她|跟大家)?"
    r"(?:说|讲|问|觉得|认为|提到|说过|说的是|听说|据说|告诉我|告诉我一下))"
    r"[\s,，、:：]*"
)


def _strip_lead(text: str) -> str:
    """剥掉每句开头的「她说」「其实」这类引语词，让后面的别名句式能对上。"""
    if not text:
        return ""
    out = []
    for seg in re.split(rf"(?<={_BOUND})", str(text)):
        for _ in range(3):
            m = _LEAD_RE.match(seg)
            if not m or m.end() == 0:
                break
            seg = seg[m.end():]
        out.append(seg)
    return "".join(out)


def _trim_alias(name: str) -> str:
    """把「丙哥我觉得他其实…」这种粘在后面的从句切掉。"""
    s = str(name or "").strip()
    # 名字里几乎不会出现「我/你/他/她」在第二个字之后，出现就当从句开始
    m = re.search(r"我|你|他|她|它", s[1:])
    if m:
        s = s[: m.start() + 1]
    return s.strip(" 　,，。.、;；:：!！?？\"'“”‘’()（）[]【】")


def _alias_ok(name: str) -> bool:
    s = _trim_alias(name)
    if not (2 <= len(s) <= 20):
        return False
    if s.casefold() in _ALIAS_STOP:
        return False
    # 「我对象」「你老婆」这种代词开头的不是名字，是称呼关系
    if s[0] in "我你他她它":
        return False
    if s.isdigit():
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", s))


def _merge_value(old: str, new: str, limit: int = 240) -> str:
    """单值属性换值时，新旧两个值怎么合。

    直接拿新值顶掉旧值会丢信息 —— 「喜欢喝美式」被「改喝拿铁」顶掉就没了。
    所以先看包含关系，再拼；拼到上限才真的覆盖。
    """
    old = (old or "").strip()
    new = (new or "").strip()
    if not old:
        return new
    if not new or new == old:
        return old
    if new in old:
        return old
    if old in new:
        return new
    joined = f"{old}；{new}"
    return joined if len(joined) <= limit else new


class SugarStore:
    def __init__(self, db_path: str):
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # 错峰沉底的相位跨度（天）。0 = 不错峰，全库按同一条曲线走。
        # 由插件按配置塞进来；store 自己被单测直接打开时保持 0，结果可预测。
        self.stagger_days = 0.0
        # 写入性能：SQLite 默认 journal=delete + synchronous=FULL，每条 INSERT
        # 都是一次全量 fsync，实测 80~190ms/条（Windows Defender 实时扫描叠加后
        # 更糟），而 record_message 是同步调用——群聊里每条用户消息都会把事件
        # 循环卡住这么久。WAL + NORMAL 实测 0.7ms/条（115 倍），代价仅是断电时
        # 可能丢最后几条记忆（不会损坏库），对闲聊记录完全可接受。
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # 可重入：有些方法（归并名字、合并关系）会在自己持锁时再调内部读方法，
        # 用普通 Lock 会当场死锁，而且卡住的是整个插件的事件循环
        self._lock = threading.RLock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)
            self._migrate()
            self._seed_stages()

    def _seed_stages(self):
        """好感度阶段表为空时灌默认档位（管理员改过就不再动）。"""
        row = self._db.execute("SELECT COUNT(*) c FROM affinity_stages").fetchone()
        if row and row["c"]:
            return
        for i, (name, th) in enumerate(DEFAULT_STAGES):
            self._db.execute(
                "INSERT INTO affinity_stages (name, threshold, sort, desc) VALUES (?,?,?,?)",
                (name, float(th), i, DEFAULT_STAGE_DESCS.get(name, "")),
            )

    def _migrate(self):
        """老库补列 + 回填归属/会话，并清掉历史内容里的 [MSG_ID:...] 尾巴。"""
        l3_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(l3_events)")}
        for col, ddl in (
            ("subject", "TEXT NOT NULL DEFAULT ''"),
            ("session_id", "TEXT NOT NULL DEFAULT ''"),
            ("summary_id", "INTEGER NOT NULL DEFAULT 0"),
            ("bot_id", "TEXT NOT NULL DEFAULT ''"),
        ):
            if col not in l3_cols:
                self._db.execute(f"ALTER TABLE l3_events ADD COLUMN {col} {ddl}")
        if "bot_id" not in l3_cols:
            # 多机器人隔离：升级前的记录没有 bot_id，按它自己的 session 前缀
            # （unified_msg_origin 第一段 = 平台实例 id）回填。一个平台实例就是
            # 一个机器人账号，所以这比「全部塞给当前机器人」准，也不会把
            # A 机器人的历史记忆判给 B。
            # 没有 session 的（手动补录/导入）留在 ''，继续当共享记录。
            self._db.execute(
                "UPDATE l3_events SET bot_id = substr(session_id, 1, instr(session_id, ':') - 1)"
                " WHERE bot_id = '' AND instr(session_id, ':') > 1"
            )
        l2_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(l2_facts)")}
        if "source_summary_id" not in l2_cols:
            self._db.execute(
                "ALTER TABLE l2_facts ADD COLUMN source_summary_id INTEGER NOT NULL DEFAULT 0"
            )
        if "ask_alias" not in l2_cols:
            # 问法别名：总结阶段让 LLM 顺手给每条事实生成几个「用户可能怎么问起
            # 它」的说法，检索时与 query 一并做 2-gram 匹配。改写问法（「多大了」
            # vs「20 岁」）字面零重叠，光靠 value 撞不上，靠这个列补召回。
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN ask_alias TEXT NOT NULL DEFAULT ''")
        if "hit_count" not in l2_cols:
            # 提及计数：这条事实被实际注入进 LLM 请求的次数（+最近一次时间）。
            # 面板拿它显示「提及率」，未来做冷热淘汰时当 tie-breaker 信号。
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0")
        if "last_hit_at" not in l2_cols:
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN last_hit_at REAL NOT NULL DEFAULT 0")
        if "stored_at" not in l2_cols:
            # 入库时间，和 created_at（事实发生时间）分开。
            # 补录会把 created_at 记成半年前的原话时间，可「被观察了多久」
            # 是从导进库那一刻算的——用 created_at 当分母会把提及率压成 0。
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN stored_at REAL NOT NULL DEFAULT 0")
            # 老数据没有真实入库时间，用 created_at 兜底（对实时总结写的事实
            # 来说两者本来就相等；对补录数据只能算近似，但不会算错方向）
            self._db.execute("UPDATE l2_facts SET stored_at=created_at WHERE stored_at=0")
        if "pinned" not in l2_cols:
            # 图钉：钉住的事实永不自动归档，也不参与低保留度下沉。
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        if "superseded_at" not in l2_cols:
            # 被新事实取代的时间。和「沉底」共用 status='archived'，但两者
            # 原因完全不同：沉底是「没人提了」，取代是「改口了」。
            # 面板要分开显示，否则用户分不清哪条是过期、哪条是被自己推翻。
            self._db.execute("ALTER TABLE l2_facts ADD COLUMN superseded_at REAL NOT NULL DEFAULT 0")

        # token_usage 可观测性列（2026-11-06）。老库补列时顺手把已经存在的
        # 行标成旧架构：它们全是 A 方案之前「动态包」跑出来的 baseline。
        # 只在首次加列那一次回填，之后每条新行都由调用方写明 arch_ver，
        # 不会被后面的 UPDATE 误标。
        tok_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(token_usage)")}
        for col, ddl in (
            ("arch_ver", "TEXT NOT NULL DEFAULT ''"),
            ("inject_fp", "TEXT NOT NULL DEFAULT ''"),
            ("rebuilt", "INTEGER NOT NULL DEFAULT 0"),
            ("freeze_rounds", "INTEGER NOT NULL DEFAULT 0"),
            ("cfg_ver", "TEXT NOT NULL DEFAULT ''"),
            ("rebuild_reason", "TEXT NOT NULL DEFAULT ''"),
        ):
            if col not in tok_cols:
                self._db.execute(f"ALTER TABLE token_usage ADD COLUMN {col} {ddl}")
        if "arch_ver" not in tok_cols:
            self._db.execute(
                "UPDATE token_usage SET arch_ver='dynamic' WHERE arch_ver=''")

        # 人物：QQ 是身份锚，称呼要分层。老库两个字段都没有，补上并回填。
        people_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(people)")}
        if "display_name" not in people_cols:
            self._db.execute("ALTER TABLE people ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
        if "call_name" not in people_cols:
            self._db.execute("ALTER TABLE people ADD COLUMN call_name TEXT NOT NULL DEFAULT ''")
        if "call_locked" not in people_cols:
            # 「能被称呼的名」是自动跟随昵称，还是被人手动指定过。
            # 锁住之后换昵称不动它——用户自己定的叫法不能被顶掉。
            self._db.execute(
                "ALTER TABLE people ADD COLUMN call_locked INTEGER NOT NULL DEFAULT 0")
        # 老数据没有 display/call 的概念，用 person 自身兜底：
        # 它就是当初第一次见到这个人的称呼，当显示名和称呼名都成立。
        self._db.execute(
            "UPDATE people SET display_name=person WHERE display_name=''")
        self._db.execute(
            "UPDATE people SET call_name=person WHERE call_name=''")
        # 阶段描述列（2026-11-06）：面板可以逐阶段自定义语气提示文案
        astage_cols = {r["name"] for r in self._db.execute(
            "PRAGMA table_info(affinity_stages)")}
        if "desc" not in astage_cols:
            self._db.execute(
                "ALTER TABLE affinity_stages ADD COLUMN desc TEXT NOT NULL DEFAULT ''")
            # 老库里已有的默认阶段顺手把描述补上，别让升级后全是空的
            for r in self._db.execute("SELECT id, name FROM affinity_stages").fetchall():
                d = DEFAULT_STAGE_DESCS.get(str(r["name"] or ""))
                if d:
                    self._db.execute("UPDATE affinity_stages SET desc=? WHERE id=?", (d, r["id"]))
        # 自愈（2026-11-06 实机）：老版本把「拿不到昵称时的 QQ 号兜底」当
        # tier=0 真昵称写进了 display_name/call_name，好几个老用户的
        # 显示名全成了各自的号码。号码不是名字：显示名直接清空（真昵称是
        # 什么我们不知道，不编）；称呼名用别名里最像称呼的非数字写法回填
        # （tier 小的优先=更正式的叫法，同档 hits 高的优先=用得多）。
        # 顺序必须在上面两句兜底之后，否则清空又被填回 person 键。
        for _r in self._db.execute(
                "SELECT person, display_name, call_name FROM people").fetchall():
            _disp = str(_r["display_name"] or "").strip()
            _call = str(_r["call_name"] or "").strip()
            if not _is_qq_like(_disp) and not _is_qq_like(_call):
                continue
            _nd = _disp if not _is_qq_like(_disp) else ""
            if not _is_qq_like(_call):
                _nc = _call
            else:
                # 收集该人所有非数字别名，按 (tier 升序, hits 降序) 挑最优
                _cands = [
                    (str(_a["raw"] or "").strip(), int(_a["tier"] or 2),
                     -int(_a["hits"] or 0))
                    for _a in self._db.execute(
                        "SELECT raw, tier, hits FROM person_alias WHERE person=?",
                        (_r["person"],)).fetchall()
                ]
                _cands = [c for c in _cands
                          if c[0] and not _is_qq_like(c[0]) and c[0] != _r["person"]]
                _nc = min(_cands, key=lambda c: c[1:])[0] if _cands else ""
            self._db.execute(
                "UPDATE people SET display_name=?, call_name=? WHERE person=?",
                (_nd, _nc, _r["person"]))
        palias_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(person_alias)")}
        if "tier" not in palias_cols:
            self._db.execute("ALTER TABLE person_alias ADD COLUMN tier INTEGER NOT NULL DEFAULT 2")
            # 回填：跟 person 同名的那个称呼是「能被称呼的名」，其余都当外号。
            # 没把 source='auto' 的降成 tier 3——分不清哪些是叫错的，
            # 宁可都留着让人工确认，也不要一升级就把别人的称呼清掉。
            # alias_key 是 Python 函数，SQL 里没有，只能逐个人算。
            for r in self._db.execute("SELECT person FROM people").fetchall():
                self._db.execute(
                    "UPDATE person_alias SET tier=1 WHERE alias_key=?",
                    (alias_key(r["person"]),))
        if "last_seen" not in palias_cols:
            self._db.execute("ALTER TABLE person_alias ADD COLUMN last_seen REAL NOT NULL DEFAULT 0")
            self._db.execute("UPDATE person_alias SET last_seen=updated_at WHERE last_seen=0")

        msg_id_pat = re.compile(r"\s*\[MSG_ID:[^\]]*\]\s*$")
        for r in self._db.execute("SELECT id, content FROM l3_events").fetchall():
            cleaned = msg_id_pat.sub("", r["content"] or "")
            if cleaned != (r["content"] or ""):
                self._db.execute("UPDATE l3_events SET content=? WHERE id=?", (cleaned, r["id"]))

        # 历史消息：AI 回复 / 降级候选挂到最近一条用户消息；总结挂到窗口内最活跃的人
        rows = self._db.execute(
            "SELECT id, event_type, subject, session_id, created_at FROM l3_events ORDER BY id"
        ).fetchall()
        users: list = []
        for r in rows:
            if r["event_type"] == "user_message":
                users.append(r)
                if len(users) > 300:
                    users.pop(0)
                continue
            if r["subject"]:
                continue
            if r["event_type"] in ("bot_reply", "demoted_candidate", "manual_note"):
                pick = None
                for u in reversed(users):
                    if u["created_at"] > r["created_at"]:
                        continue
                    if r["created_at"] - u["created_at"] <= 1800:
                        pick = u
                    break
                if pick:
                    self._db.execute(
                        "UPDATE l3_events SET subject=?, session_id=? WHERE id=?",
                        (pick["subject"], pick["session_id"] or "", r["id"]),
                    )
            elif r["event_type"] == "session_summary":
                window = [
                    u for u in users
                    if u["created_at"] <= r["created_at"]
                    and r["created_at"] - u["created_at"] <= 12 * 3600
                ][-90:]
                counter = Counter(u["subject"] for u in window if u["subject"])
                if counter:
                    subj = counter.most_common(1)[0][0]
                    sess = next(
                        (u["session_id"] for u in reversed(window)
                         if u["subject"] == subj and u["session_id"]),
                        "",
                    )
                    self._db.execute(
                        "UPDATE l3_events SET subject=?, session_id=? WHERE id=?",
                        (subj, sess, r["id"]),
                    )

        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_subj ON l3_events(subject, created_at)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_session ON l3_events(session_id, created_at)"
        )
        # 记忆包的查询是 WHERE subject/session_id=? ORDER BY id DESC LIMIT n。
        # 只有 (subject, created_at) 索引时，SQLite 拿到该主体全部行再临时排序取
        # 前 n 条——单人 1 万行时 build 实测 172ms（小库 4.7ms）。带 id 的索引让
        # 排序直接走索引倒序扫描、取够 LIMIT 就停，深库不再退化。
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_subj_id ON l3_events(subject, id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_sess_type_id"
            " ON l3_events(session_id, event_type, id)"
        )
        # recent_nick_by_session 专用：只收「有昵称的原话」，GROUP BY 会话取最新
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_nick ON l3_events(session_id, id)"
            " WHERE event_type='user_message' AND subject != ''"
        )
        # cleanup 分批扫描过期行用：expires_at 范围直取下一批，不重扫全表
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_exp ON l3_events(expires_at)"
        )
        # 多机器人隔离：记忆包/外发列表都是 WHERE bot_id=? AND event_type=?
        # ORDER BY id DESC，这条索引让多账号库也能直接倒序取够 LIMIT 就停。
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_bot ON l3_events(bot_id, event_type, id)"
        )

    def close(self):
        with self._lock:
            self._db.close()

    # ---------- L2 ----------

    def write_l2(self, *, subject: str, predicate: str, value: str, speaker: str,
                 speech_act: str = "statement", source_type: str = "user_message",
                 source_msg_id: str = "", source_quote: str = "",
                 source_summary_id: int = 0, confidence: float = 0.6,
                 gate_verdict: dict | None = None, gate_reason: str = "",
                 expires_at: float = 0.0, now: float | None = None,
                 single_valued: bool | None = None, source_ts: float = 0.0) -> dict:
        """受控写入 L2。必须携带门控 allow 裁决；存储层强制 actionable=0。

        single_valued 不传时按 predicate 自动判断：七个受控分类全是多值桶，
        同一个人可以有几十条「经历」。以前默认 True，谁忘了传就会把「跑过半马」
        和「昨天学会游泳」拼成一条——调用方稍不注意就中招，所以改成自动判断。

        source_ts：这条事实在对话里发生的时间。补录提炼会传原话时间——
        半年前的「我 19 岁」不能把现在的「20 岁」改回去，比现有记录还旧的一律跳过。
        """
        if not gate_verdict or gate_verdict.get("decision") != "allow":
            raise PermissionError(f"L2 写入被拒绝：缺少门控 allow 裁决（got {gate_verdict}）")
        if speaker == "bot" or source_type == "bot_utterance":
            raise PermissionError("L2 存储层硬约束：Bot 自述/来源不得写入 L2")
        if speech_act in ("quote", "roleplay", "hypothesis", "temp_state", "plan"):
            raise PermissionError(f"L2 存储层硬约束：{speech_act} 只能降级 L3，不得写入 L2")
        now = now or time.time()
        if single_valued is None:
            single_valued = not is_multi_valued(predicate)
        # 补录的原话有自己的时间。把 created_at 记成原话时间，而不是「导入的这一刻」：
        # 半年期补录不该让所有事实都显示成今天。updated_at 是「最后一次确认」，
        # 补录数据对应的确认发生在它被说出来的那一刻，不是导入的那一刻——
        # 否则半年前的事在遗忘曲线上跟今天刚说的一样新。
        at = float(source_ts) if source_ts else now
        fp = _fp_norm(subject, predicate, value)
        with self._lock, self._db:
            # 这里刻意不按 (subject, predicate) 判「旧话退让」。
            # 七个受控分类全是宽桶多值（爱好技能里装着「喜欢什么」和「会什么」），
            # 按桶判新旧等于随机删：补录「昨天喜欢篮球」会被「今天喜欢摇滚」挤掉，
            # 人的经历就少了一半。旧事实保留，靠 created_at 标注它是什么时候说的。
            row = self._db.execute(
                "SELECT id, stability, confidence, updated_at FROM l2_facts"
                " WHERE fingerprint=? AND status='active'",
                (fp,),
            ).fetchone()
            if row:
                # 值完全一样就是重复陈述，不是冲突：稳度+1 就行，不判新旧。
                # updated_at 取 MAX：补录半年前的旧话不该把「最后一次确认」
                # 顶到今天，否则这条事实在遗忘曲线上会被当成刚聊过。
                self._db.execute(
                    "UPDATE l2_facts SET stability=stability+1, confidence=MAX(confidence,?),"
                    " updated_at=MAX(updated_at,?) WHERE id=?",
                    (float(confidence), at, row["id"]),
                )
                if source_summary_id:
                    self._db.execute(
                        "UPDATE l2_facts SET source_summary_id=?"
                        " WHERE id=? AND source_summary_id=0",
                        (int(source_summary_id), row["id"]),
                    )
                return {"action": "merged", "id": row["id"]}
            # 跨谓词兜底：同一个人的两条事实，谓词被模型写成两种、但 value 语义相同
            # （「生日：5月20日」和「身份：1999年5月20日出生」），也会被当成两件事。
            # 归一化指纹在谓词层面会失配，所以这里再按「subject + value 归一化」查一次。
            val_fp = _norm_fingerprint(value)
            if val_fp:
                dup = self._db.execute(
                    "SELECT id, value FROM l2_facts WHERE status='active' AND subject=?"
                    " ORDER BY updated_at DESC LIMIT 200",
                    (subject,),
                ).fetchall()
                for d in dup:
                    if _norm_fingerprint(d["value"]) == val_fp:
                        # 语义重复：别新插一条，把这条标记为已见过（稳度+1）。
                        # value 不强行覆盖——旧的可能更完整，这里只认「重复」不认「换写」。
                        self._db.execute(
                            "UPDATE l2_facts SET stability=stability+1,"
                            " confidence=MAX(confidence,?), updated_at=MAX(updated_at,?) WHERE id=?",
                            (float(confidence), at, d["id"]),
                        )
                        return {"action": "deduped", "id": d["id"],
                                "predicate": predicate, "matched_value": d["value"]}
            # 单值属性：同一个 (subject, predicate) 只允许一条 active，新值就地覆盖。
            # 旧做法是插一条新的、把旧的标 replaced —— 结果是 active 里永远只剩最后一句，
            # 前面说过的东西全进了历史表，等于从记忆里消失。改成原地合并后，
            # 一条记录承载这个属性的全部已知内容，历史仍可从 L3 原文追溯。
            conflict = None
            if single_valued:
                conflict = self._db.execute(
                    "SELECT id, value, updated_at FROM l2_facts"
                    " WHERE subject=? AND predicate=? AND status='active'"
                    " ORDER BY updated_at DESC LIMIT 1",
                    (subject, predicate),
                ).fetchone()
            if conflict:
                # 补录提炼的旧原话退让：现有记录比这段对话还新，就不许它往回改。
                if source_ts and float(conflict["updated_at"] or 0) > float(source_ts):
                    return {"action": "stale", "id": conflict["id"]}
                # 新旧 value 语义相同（归一化后一致）就不再拼，直接当重复稳度+1，
                # 避免「5月20日；5/20」这种换描述的重复串。
                if _norm_fingerprint(conflict["value"]) == val_fp:
                    self._db.execute(
                        "UPDATE l2_facts SET stability=stability+1,"
                        " confidence=MAX(confidence,?), source_msg_id=?, source_quote=?,"
                        " source_summary_id=?, updated_at=MAX(updated_at,?) WHERE id=?",
                        (float(confidence), source_msg_id, source_quote,
                         int(source_summary_id or 0), at, conflict["id"]),
                    )
                    return {"action": "merged", "id": conflict["id"],
                            "predicate": predicate, "same_value": True}
                merged = _merge_value(conflict["value"], value)
                self._db.execute(
                    "UPDATE l2_facts SET value=?, fingerprint=?, stability=stability+1,"
                    " confidence=MAX(confidence,?), source_msg_id=?, source_quote=?,"
                    " source_summary_id=?, gate_reason=?, updated_at=MAX(updated_at,?) WHERE id=?",
                    (merged, _fp_norm(subject, predicate, merged), float(confidence),
                     source_msg_id, source_quote, int(source_summary_id or 0),
                     gate_reason or "用户显式陈述", at, conflict["id"]),
                )
                return {"action": "merged", "id": conflict["id"],
                        "predicate": predicate, "merged_value": merged}
            reason = gate_reason or "用户显式陈述"
            cur = self._db.execute(
                "INSERT INTO l2_facts (subject, predicate, value, speaker, speech_act, source_type,"
                " source_msg_id, source_quote, source_summary_id, actionable, confidence, status,"
                " stability, expires_at, replaced_by, gate_reason, fingerprint, created_at, updated_at,"
                " stored_at, pinned)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,?,'active',0,?,0,?,?,?,?,?,0)",
                (subject, predicate, value, speaker, speech_act, source_type,
                 source_msg_id, source_quote, int(source_summary_id or 0),
                 float(confidence), expires_at, reason, fp, at, at, now),
            )
            return {"action": "created", "id": int(cur.lastrowid), "predicate": predicate}

    def list_subjects(self) -> list[str]:
        """所有有 active L2 的人，给「这轮提到了谁」用。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT subject FROM l2_facts"
                " WHERE status='active' AND subject != ''"
            ).fetchall()
        return [str(r[0]) for r in rows if r[0]]

    def subjects_mentioned(self, text: str, exclude: str = "",
                           limit: int = 3) -> list[str]:
        """从这轮消息里认出被提到的其他人。

        A 问「用户丙 是不是说过他叫滚木」时，只注入 A 自己的记忆是没用的 ——
        答案在 B 的档案里。名字直接做子串匹配；单字名不参与，太容易误命中。
        别名同样算点名：库里登记过「别名B＝用户乙」，用户拿昵称点名时
        extra 档案通道必须照样生效（压测抓到的真 bug：此前只认本名，
        person_resolve 认得出、这里认不出，两条链路不一致）。
        """
        t = str(text or "").lower()
        if not t:
            return []
        ex = str(exclude or "").strip()
        names = [n for n in self.list_subjects() if n != ex and len(n) >= 2]
        hits = [n for n in names if n.lower() in t]
        remaining = [n for n in names if n not in hits]
        if remaining:
            with self._lock:
                marks = ",".join("?" * len(remaining))
                rows = self._db.execute(
                    f"SELECT person, alias_key FROM person_alias WHERE person IN ({marks})",
                    remaining,
                ).fetchall()
            alias_of: dict = {}
            for r in rows:
                alias_of.setdefault(r["person"], []).append(str(r["alias_key"] or ""))
            for n in remaining:
                keys = alias_of.get(n) or []
                if any(k and len(k) >= 2 and k in t for k in keys):
                    hits.append(n)
        hits.sort(key=len, reverse=True)
        return hits[: max(1, int(limit))]

    def query_active_l2(self, subject: str | None = None, limit: int = 60) -> list[dict]:
        """查询 active L2（带出处/置信度）。与 L3 查询完全隔离。"""
        now = time.time()
        with self._lock:
            if subject:
                rows = self._db.execute(
                    "SELECT * FROM l2_facts WHERE status='active' AND subject=?"
                    " AND (expires_at = 0 OR expires_at > ?)"
                    " ORDER BY confidence DESC, stability DESC, updated_at DESC LIMIT ?",
                    (subject, now, int(limit)),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM l2_facts WHERE status='active'"
                    " AND (expires_at = 0 OR expires_at > ?)"
                    " ORDER BY confidence DESC, stability DESC, updated_at DESC LIMIT ?",
                    (now, int(limit)),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_l2(self, fact_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM l2_facts WHERE id=?", (int(fact_id),)).fetchone()
        return dict(row) if row else None

    def bump_l2_hits(self, fact_ids, now: float | None = None) -> int:
        """提及计数：这批事实被真正注入进 LLM 请求时各 +1。

        只对 SELECT 到的 id 生效；空列表直接返回。失败由调用方兜底，
        计数丢一轮无所谓，不值得为它打断对话。
        """
        ids = [int(i) for i in set(fact_ids or ()) if i]
        if not ids:
            return 0
        now = now or time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                f"UPDATE l2_facts SET hit_count = hit_count + 1, last_hit_at = ?"
                f" WHERE id IN ({','.join('?' * len(ids))})",
                (now, *ids),
            )
            return cur.rowcount

    # ---------- 遗忘：保留度、图钉、归档 ----------

    def set_l2_pinned(self, fact_id: int, pinned: bool = True) -> bool:
        """钉住/松开。钉住的条目永不自动归档，也不参与低保留度下沉。"""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE l2_facts SET pinned=? WHERE id=?",
                (1 if pinned else 0, int(fact_id)),
            )
            return cur.rowcount > 0

    def archive_l2(self, fact_id: int) -> bool:
        """归档：从 active 挪进 archived。不删数据，面板能查、能恢复。"""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE l2_facts SET status='archived'"
                " WHERE id=? AND status='active'",
                (int(fact_id),),
            )
            return cur.rowcount > 0

    def unarchive_l2(self, fact_id: int) -> bool:
        """恢复归档。顺手把 last_hit_at 记成现在——用户亲手捞回来的东西，
        在遗忘曲线上就该当刚想起来处理，否则一恢复又立刻沉底。"""
        now = time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE l2_facts SET status='active', last_hit_at=?,"
                " replaced_by=0, superseded_at=0"
                " WHERE id=? AND status='archived'",
                (now, int(fact_id)),
            )
            return cur.rowcount > 0

    def supersede_l2(self, old_id: int, new_id: int = 0,
                     now: float | None = None) -> bool:
        """旧事实被新事实取代：只打标记 + 归档，数据一条不删。

        跟 cleanup_decay 的区别是原因不同——这里是「他改口了」，不是
        「没人提了」。面板分开显示，捞回来时两个标记一起清掉。
        """
        now = now or time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE l2_facts SET status='archived', replaced_by=?, superseded_at=?"
                " WHERE id=? AND status='active'",
                (int(new_id or 0), now, int(old_id)),
            )
            return cur.rowcount > 0

    def query_archived_l2(self, subject: str | None = None,
                          limit: int = 200) -> list[dict]:
        with self._lock:
            if subject:
                rows = self._db.execute(
                    "SELECT * FROM l2_facts WHERE status='archived' AND subject=?"
                    " ORDER BY updated_at DESC LIMIT ?",
                    (subject, int(limit)),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM l2_facts WHERE status='archived'"
                    " ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [dict(r) for r in rows]

    def l2_retention_map(self, subject: str | None = None,
                         now: float | None = None) -> dict[int, dict]:
        """给面板批量算保留度：{id: {r, quiet, left, pinned}}。

        统一走这里而不是让前端照着公式重算——参数一改两头就不同步，
        面板上的倒计时会变成假的。
        """
        now = now or time.time()
        out: dict[int, dict] = {}
        for r in self.query_active_l2(subject=subject, limit=2000):
            d = dict(r)
            # 相位按 store 的配置算好塞进行里：倒计时和清理必须用同一个偏移，
            # 否则面板说「还有 3 天」，清理那边还要再等两个月。
            d["_stagger"] = retention_stagger_days(d, self.stagger_days)
            out[int(d["id"])] = {
                "r": round(retention_score(d, now), 4),
                "quiet": round(retention_quiet_days(d, now), 1),
                "left": round(retention_days_left(d, now), 1),
                "pinned": int(d.get("pinned") or 0),
                "stagger": round(d["_stagger"], 1),
            }
        return out

    def cleanup_decay(self, now: float | None = None,
                      min_days: float = RETENTION_ARCHIVE_MIN_DAYS,
                      dry_run: bool = False) -> dict:
        """把沉到底的长期记忆归档（不是删除）。

        三个条件同时满足才动手：非图钉、不在永久分类里、静默超过
        「曲线寿命 + 本条相位偏移」（错峰默认 0~75 天）。
        默认 90 天起跳——宁可让垃圾多躺一阵，也不能把「三个月没聊到」
        的病史当成没人要的东西收走。
        """
        now = now or time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT id, subject, predicate, value, confidence, stability,"
                " last_hit_at, updated_at, created_at, stored_at, pinned"
                " FROM l2_facts WHERE status='active' AND pinned=0"
            ).fetchall()
        picked: list[dict] = []
        for r in rows:
            d = dict(r)
            if str(d.get("predicate") or "").strip() in PERMANENT_PREDICATES:
                continue
            # 错峰：同批入库的条目曲线寿命几乎一样，靠相位把它们摊到两三个月里
            # 陆续沉，而不是同一天集体过线。stagger_days=0 时退化成老行为。
            d["_stagger"] = retention_stagger_days(d, self.stagger_days)
            if retention_quiet_days(d, now) < max(
                    float(min_days), retention_archive_after(d)):
                continue
            # 入库也要够久。补录的历史数据 created_at 可能是半年前，
            # 保留度天然很低，但人家今天刚导进来，不能当天就收走。
            stored = float(d.get("stored_at") or 0)
            if stored and (now - stored) / 86400.0 < float(min_days):
                continue
            picked.append(d)
        if dry_run or not picked:
            return {"archived": 0, "subjects": [], "dry_run": bool(dry_run)}
        ids = [int(d["id"]) for d in picked]
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE l2_facts SET status='archived'"
                f" WHERE id IN ({','.join('?' * len(ids))})",
                tuple(ids),
            )
        return {
            "archived": len(ids),
            "subjects": sorted({str(d.get("subject") or "") for d in picked}),
            "dry_run": False,
        }

    def l2_need_alias(self, limit: int = 10) -> list[dict]:
        """还没有问法别名的 active L2，供 alias 生成任务批量取。

        只挑 value 够长（>=4 字）的：太短的事实（「16岁」这种）生成问法
        的信息量有限，而且容易在 LLM 输出里跟别的条目混淆。
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT id, subject, predicate, value FROM l2_facts"
                " WHERE status='active' AND ask_alias='' AND length(value) >= 4"
                " ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def l2_set_alias(self, fact_id: int, alias: str) -> None:
        """写问法别名。刻意不更新 updated_at：它随稳度/合并变化会导致注入块的
        时间锚字面量跳变，把 provider 前缀缓存整段打废（见 injector._abs_when）。
        """
        with self._lock, self._db:
            self._db.execute(
                "UPDATE l2_facts SET ask_alias=? WHERE id=?",
                (str(alias or "")[:400], int(fact_id)),
            )

    def l2_alias_stats(self) -> tuple[int, int]:
        """(有别名的 active 条数, active 总条数)，给面板/日志用。"""
        with self._lock:
            total = self._db.execute(
                "SELECT COUNT(*) c FROM l2_facts WHERE status='active'").fetchone()["c"]
            done = self._db.execute(
                "SELECT COUNT(*) c FROM l2_facts WHERE status='active' AND ask_alias<>''"
            ).fetchone()["c"]
        return int(done), int(total)

    def l2_chain(self, fact_id: int) -> list[dict]:
        """回溯一条 L2 的替代链（谁被它取代/它取代了谁）。"""
        chain, cur = [], self.get_l2(fact_id)
        seen = set()
        while cur and cur["id"] not in seen:
            seen.add(cur["id"])
            chain.append(cur)
            cur = self.get_l2(cur["replaced_by"]) if cur["replaced_by"] else None
        if chain:
            prev = self._db.execute(
                "SELECT * FROM l2_facts WHERE replaced_by=? ORDER BY id DESC LIMIT 1",
                (int(fact_id),),
            ).fetchall()
            chain.extend(dict(r) for r in prev if r["id"] not in seen)
        return chain

    def delete_l2(self, fact_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute(
                "DELETE FROM l2_facts WHERE id=?", (int(fact_id),)
            )
            return cur.rowcount > 0

    def update_l2(self, fact_id: int, *, predicate: str | None = None,
                  value: str | None = None, confidence: float | None = None,
                  now: float | None = None) -> bool:
        """人工修订一条 L2。只改内容，不改 subject/来源；指纹重算避免与旧条目误合并。"""
        cur = self.get_l2(fact_id)
        if not cur:
            return False
        pred = (predicate if predicate is not None else cur["predicate"]).strip()[:40] or cur["predicate"]
        val = (value if value is not None else cur["value"]).strip()[:200] or cur["value"]
        conf = float(cur["confidence"]) if confidence is None else max(0.0, min(1.0, float(confidence)))
        fp = _fp_norm(cur["subject"], pred, val)
        now = now or time.time()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE l2_facts SET predicate=?, value=?, confidence=?, fingerprint=?, updated_at=?"
                " WHERE id=?",
                (pred, val, conf, fp, now, int(fact_id)),
            )
        return True

    def add_l2_manual(self, *, subject: str, predicate: str, value: str,
                      confidence: float = 0.9, reason: str = "面板手动添加",
                      now: float | None = None) -> int:
        """管理面板专用写入通道。仍受存储层硬约束（actionable=0、非 Bot 来源），
        但不需要门控裁决——人工添加本身就是裁决，来源标为 manual 以便追溯。"""
        now = now or time.time()
        subject = (subject or "").strip()[:40]
        predicate = (predicate or "").strip()[:40]
        value = (value or "").strip()[:200]
        if not subject or not predicate or not value:
            raise ValueError("subject / predicate / value 均不能为空")
        fp = _fp_norm(subject, predicate, value)
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT id FROM l2_facts WHERE fingerprint=? AND status='active'", (fp,)
            ).fetchone()
            if row:
                self._db.execute(
                    "UPDATE l2_facts SET stability=stability+1, updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                return int(row["id"])
            # 与 write_l2 保持一致：多值分类（7 类）允许多条并存，每件事独立一条；
            # 只有单值分类才做「同一个 (subject, predicate) 只留一条」的就地合并。
            if not is_multi_valued(predicate):
                conflict = self._db.execute(
                    "SELECT id, value FROM l2_facts"
                    " WHERE subject=? AND predicate=? AND status='active'"
                    " ORDER BY updated_at DESC LIMIT 1",
                    (subject, predicate),
                ).fetchone()
                if conflict:
                    # 语义重复（归一化后 value 相同）不再拼串，直接稳度+1
                    if _norm_fingerprint(conflict["value"]) == _norm_fingerprint(value):
                        self._db.execute(
                            "UPDATE l2_facts SET stability=stability+1, updated_at=? WHERE id=?",
                            (now, conflict["id"]),
                        )
                        return int(conflict["id"])
                    merged = _merge_value(conflict["value"], value)
                    self._db.execute(
                        "UPDATE l2_facts SET value=?, fingerprint=?, stability=stability+1,"
                        " confidence=MAX(confidence,?), gate_reason=?, updated_at=? WHERE id=?",
                        (merged, _fp_norm(subject, predicate, merged),
                         max(0.0, min(1.0, float(confidence))), reason, now, conflict["id"]),
                    )
                    return int(conflict["id"])
            cur = self._db.execute(
                "INSERT INTO l2_facts (subject, predicate, value, speaker, speech_act, source_type,"
                " source_msg_id, source_quote, actionable, confidence, status, stability, expires_at,"
                " replaced_by, gate_reason, fingerprint, created_at, updated_at, stored_at)"
                " VALUES (?,?,?,'user','statement','manual','','',0,?,'active',0,0,0,?,?,?,?,?)",
                (subject, predicate, value, max(0.0, min(1.0, float(confidence))),
                 reason, fp, now, now, now),
            )
            return int(cur.lastrowid)

    def count_units(self, subject: str = "") -> dict:
        """单个用户的短期记忆「段数」与消息条数。"""
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT id, event_type, session_id, created_at FROM l3_events"
                " WHERE subject=? AND (expires_at = 0 OR expires_at > ?)",
                (subject or "", now),
            ).fetchall()
        data = [dict(r) for r in rows]
        return {"l3": count_conversations(data), "l3_msgs": len(data)}

    def subject_stats(self) -> list[dict]:
        """每个用户（subject）的长期条数、短期「段数」与最后活跃时间。

        短期不再按消息条数计：一次对话（含其中的 AI 回复与总结）算 1 条。
        """
        now = time.time()
        with self._lock:
            l2 = self._db.execute(
                "SELECT subject, COUNT(*) c, MAX(updated_at) t FROM l2_facts"
                " WHERE status='active' AND (expires_at = 0 OR expires_at > ?) GROUP BY subject",
                (now,),
            ).fetchall()
            l3 = self._db.execute(
                "SELECT id, subject, event_type, session_id, created_at FROM l3_events"
                " WHERE (expires_at = 0 OR expires_at > ?)",
                (now,),
            ).fetchall()
        agg: dict[str, dict] = {}
        for r in l2:
            s = r["subject"] or ""
            agg[s] = {"subject": s, "l2": r["c"], "l3": 0, "l3_msgs": 0, "sum": 0, "last_at": r["t"] or 0}
        grouped: dict[str, list[dict]] = {}
        last_at: dict[str, float] = {}
        for r in l3:
            row = dict(r)
            s = row["subject"] or ""
            grouped.setdefault(s, []).append(row)
            last_at[s] = max(last_at.get(s, 0.0), float(row["created_at"] or 0))
        for s, rows in grouped.items():
            cur = agg.setdefault(s, {"subject": s, "l2": 0, "l3": 0, "l3_msgs": 0, "sum": 0, "last_at": 0})
            cur["l3"] = count_conversations(rows)
            cur["l3_msgs"] = len(rows)
            cur["sum"] = sum(1 for r in rows if r["event_type"] == "session_summary")
            cur["last_at"] = max(float(cur["last_at"] or 0), last_at.get(s, 0.0))
        items = [v for v in agg.values() if v["l2"] or v["l3"]]
        items.sort(key=lambda x: (-(x["l2"] * 2 + x["l3"]), -float(x["last_at"] or 0)))
        return items

    # ---------- L3 ----------

    def write_l3(self, *, event_type: str, content: str, source_msg_id: str = "",
                 subject: str = "", session_id: str = "", summary_id: int = 0,
                 expires_at: float = 0.0, now: float | None = None,
                 bot_id: str = "") -> int:
        now = now or time.time()
        # 主体归一：同一个人的不同写法（用户乙 / 用户甲）在写入时就归到 canonical 名下，
        # 否则 person_alias 里明明有映射，实时事件流却用原始称呼落库，同一个人又裂成两个。
        # 归一认不出（群名、数字 id、空串）就保留原值，绝不把特殊 subject 也硬并成一个人。
        subject = (subject or "").strip()[:40]
        if subject:
            resolved = self.person_resolve(subject)
            if resolved:
                subject = resolved
        # bot_id 不给就按会话推：所有调用点（实时录制、总结、骨架、补录）
        # 都不用自己记着传，少一处漏传就少一条会串的记忆。
        bot = str(bot_id or "").strip()[:32] or bot_of_session(session_id)
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO l3_events (event_type, subject, session_id, summary_id, content,"
                " source_msg_id, actionable, expires_at, created_at, bot_id)"
                " VALUES (?,?,?,?,?,?,0,?,?,?)",
                (event_type, subject, (session_id or "")[:120], int(summary_id or 0),
                 content, source_msg_id, expires_at, now, bot),
            )
            return int(cur.lastrowid)

    def write_l3_bulk(self, rows: list[dict], *, expires_at: float = 0.0) -> int:
        """批量写 L3：单事务 executemany。

        逐条调 write_l3 是「一条一个事务」，几千条补录会慢到不可接受；
        这里合并成一个事务，并且只跑一遍主体归一（同名的只查一次库）。
        rows 每项：event_type/subject/content/session_id/created_at。
        """
        if not rows:
            return 0
        norm: dict[str, str] = {}
        prepared = []
        for r in rows:
            subject = (r.get("subject") or "").strip()[:40]
            if subject:
                if subject not in norm:
                    norm[subject] = self.person_resolve(subject) or subject
                subject = norm[subject]
            prepared.append((
                r.get("event_type") or "user_message",
                subject,
                (r.get("session_id") or "")[:120],
                0,
                (r.get("content") or "")[:300],
                r.get("source_msg_id") or "",
                expires_at,
                float(r.get("created_at") or time.time()),
                str(r.get("bot_id") or "").strip()[:32]
                or bot_of_session(r.get("session_id")),
            ))
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO l3_events (event_type, subject, session_id, summary_id, content,"
                " source_msg_id, actionable, expires_at, created_at, bot_id)"
                " VALUES (?,?,?,?,?,?,0,?,?,?)",
                prepared,
            )
        return len(prepared)

    def query_l3(self, *, subject: str | None = None, event_type: str | None = None,
                 limit: int = 200, bot_id: str | None = None) -> list[dict]:
        """按主体/类型查询 L3。subject=None 表示不限，subject="" 表示未归类。
        已到期的短期记录在这里就被过滤掉，不必等物理清理。

        bot_id=None 不过滤（面板/管理用）；传具体机器人时只返回它自己的记录，
        外加 bot_id='' 的共享记录（手动补录/导入没有归属，属于所有人）。
        多账号共用一份库时这是防止「A 的定时消息串进 B 的记忆包」的关键。
        """
        now = time.time()
        q = "SELECT * FROM l3_events WHERE (expires_at = 0 OR expires_at > ?)"
        args: list = [now]
        if subject is not None:
            q += " AND subject=?"
            args.append(subject)
        if event_type:
            q += " AND event_type=?"
            args.append(event_type)
        bot = str(bot_id or "").strip()
        if bot:
            q += " AND (bot_id=? OR bot_id='')"
            args.append(bot)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def update_l3(self, event_id: int, *, content: str) -> bool:
        content = (content or "").strip()
        if not content:
            return False
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE l3_events SET content=? WHERE id=?", (content[:500], int(event_id))
            )
            return cur.rowcount > 0

    def delete_l3(self, event_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM l3_events WHERE id=?", (int(event_id),))
            return cur.rowcount > 0

    def last_summary_at(self, session_id: str) -> float:
        """该会话最后一次成功的阶段总结时间戳；没有则 0。"""
        if not session_id:
            return 0.0
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(created_at) FROM l3_events"
                " WHERE event_type='session_summary' AND session_id=?",
                (session_id,),
            ).fetchone()
        try:
            return float(row[0] or 0.0)
        except (TypeError, ValueError, IndexError):
            return 0.0

    def count_user_messages_since(self, session_id: str, since: float = 0.0) -> int:
        """该会话自 since 之后攒了多少条用户消息（触发总结的计数，耐久、抗重启）。"""
        if not session_id:
            return 0
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM l3_events"
                " WHERE event_type='user_message' AND session_id=?"
                " AND created_at>=? AND (expires_at=0 OR expires_at>?)",
                (session_id, float(since or 0.0), time.time()),
            ).fetchone()
        try:
            return int(row[0] or 0)
        except (TypeError, ValueError, IndexError):
            return 0

    def active_session_ids(self) -> list[str]:
        """有未过期用户消息的会话 ID 列表（给面板显示总结进度用）。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT session_id FROM l3_events"
                " WHERE event_type='user_message' AND session_id<>''"
                " AND (expires_at=0 OR expires_at>?)",
                (time.time(),),
            ).fetchall()
        return [str(r[0]) for r in rows if r[0]]

    def sessions_mentioning(self, subject: str, limit: int = 200) -> set[str]:
        """跨会话归因：找出「消息内容里提到过某人」的会话 ID 集合。

        场景：用户丙 和 bot 在私聊里聊到用户甲，但字面上只说「她」「被子」，
        没有「用户甲」两个字。AI 拿到这段跨会话消息时无法归因「这是在说我」。
        解决：回看每个会话的历史消息，凡是内容里出现过 subject 名字的会话，
        就判定为「这个会话在聊这个人」，其近期消息都应优先注入并标注「可能关于 ta」。

        纯本地子串匹配，零 LLM。只扫未过期的 user_message + bot_reply，
        限条数防止历史库太大时拖慢注入。
        """
        name = str(subject or "").strip()
        if not name:
            return set()
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT session_id FROM l3_events NOT INDEXED"
                " WHERE event_type IN ('user_message','bot_reply')"
                " AND session_id<>'' AND instr(content, ?) > 0"
                " AND (expires_at=0 OR expires_at>?)"
                " ORDER BY id DESC LIMIT ?",
                (name, time.time(), int(limit)),
            ).fetchall()
        return {str(r[0]) for r in rows if r[0]}

    def query_recent_l3(self, limit: int = 40, event_type: str | None = None,
                        session_id: str | None = None,
                        subject: str | None = None,
                        bot_id: str | None = None) -> list[dict]:
        """取最近 L3 事件。session_id / subject / bot_id 为空表示不限。

        bot_id 传具体值时只取「这个机器人 + 无归属共享记录」，多账号共用一份
        数据库时不会把别的机器人发的话捞进来。
        """
        now = time.time()
        sql = "SELECT * FROM l3_events WHERE (expires_at = 0 OR expires_at > ?)"
        args: list = [now]
        if event_type:
            sql += " AND event_type=?"
            args.append(event_type)
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        if subject:
            sql += " AND subject=?"
            args.append(subject)
        bot = str(bot_id or "").strip()
        if bot:
            sql += " AND (bot_id=? OR bot_id='')"
            args.append(bot)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def bot_ids(self, limit: int = 20) -> list[dict]:
        """库里出现过的机器人（按记录数倒序）。面板给「这是哪个机器人说的」用。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT bot_id, COUNT(*) AS n, MAX(created_at) AS last_at"
                " FROM l3_events WHERE bot_id != '' GROUP BY bot_id"
                " ORDER BY n DESC LIMIT ?", (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 补录提炼（把导入的原话变成长期记忆） ----------
    # 分块单位是「自然天」，不是固定条数：一天的对话本身是完整的一段，
    # 按 60 条机械切会把同一天的话劈成两半，模型看不到前因。
    # 多天时从最近的一天往前推 —— 用户最想先拿到手的本来就是近几天的记忆。

    # 只认补录/导入进来的记录。实时聊天归各自的会话窗口，这里再提炼一遍等于重复烧钱。
    _DISTILL_WHERE = (
        " AND event_type IN ('user_message','bot_reply')"
        " AND (session_id='' OR session_id LIKE 'import:%')"
        " AND (expires_at=0 OR expires_at>?)"
    )
    # 按本地时区归天。没有时间的记录（created_at<=0）单拎一桶，排最后。
    _DISTILL_DAY = (
        "CASE WHEN created_at IS NULL OR created_at<=0 THEN '0000-00-00'"
        " ELSE date(created_at,'unixepoch','localtime') END"
    )

    def distill_cursor(self, subject: str) -> dict:
        """累计处理量。last_id 是旧版本的全局水位，只读，用来做升级兼容。"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM distill_cursor WHERE subject=?", (str(subject or ""),)
            ).fetchone()
        if row:
            return dict(row)
        return {"subject": str(subject or ""), "last_id": 0, "blocks": 0,
                "items": 0, "updated_at": 0.0}

    def distill_tally(self, subject: str, items: int = 0) -> None:
        """累计处理量 +1 块。刻意不碰 last_id：那个位置留给老版本水位。"""
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO distill_cursor (subject, last_id, blocks, items, updated_at)"
                " VALUES (?,0,1,?,?) ON CONFLICT(subject) DO UPDATE SET"
                " blocks=blocks+1, items=items+excluded.items,"
                " updated_at=excluded.updated_at",
                (str(subject or ""), int(items), now),
            )

    def distill_days(self, subject: str) -> list[dict]:
        """待提炼的天，最近的在前。每天带条数、id 范围和这一天已走到哪。"""
        now = time.time()
        subj = str(subject or "")
        with self._lock:
            rows = self._db.execute(
                f"SELECT {self._DISTILL_DAY} AS day,"
                " COUNT(*) AS n, MIN(id) AS lo, MAX(id) AS hi,"
                " MIN(created_at) AS t0, MAX(created_at) AS t1"
                f" FROM l3_events WHERE subject=?{self._DISTILL_WHERE}"
                " GROUP BY day ORDER BY day DESC",
                (subj, now),
            ).fetchall()
            marks = {r["day"]: dict(r) for r in self._db.execute(
                "SELECT * FROM distill_day WHERE subject=?", (subj,)).fetchall()}
        out = []
        for r in rows:
            d = dict(r)
            m = marks.get(d["day"]) or {}
            d["last_id"] = int(m.get("last_id") or 0)
            d["items"] = int(m.get("items") or 0)
            d["blocks"] = int(m.get("blocks") or 0)
            d["done"] = d["last_id"] >= int(d["hi"])
            out.append(d)
        return out

    def distill_chunk(self, subject: str, block: int = 120, max_chars: int = 5500,
                      legacy_after_id: int = 0) -> dict | None:
        """取下一段：最近还没提完的那天的开头一段。

        一天之内也可能切成多段（一天几百条的话），切点永远落在消息边界上，
        不会把一条话砍成两半。max_chars 是硬预算，超了宁可少塞几条。
        """
        days = self.distill_days(subject)
        if not days:
            return None
        subj = str(subject or "")
        for idx, d in enumerate(days):
            # 升级兼容：老版本的正序水位落在这天里，就当天内已经提过这么多
            eff = max(int(d["last_id"]), int(legacy_after_id or 0))
            if eff >= int(d["hi"]):
                continue
            with self._lock:
                rows = self._db.execute(
                    f"SELECT * FROM l3_events WHERE subject=?{self._DISTILL_WHERE}"
                    f" AND {self._DISTILL_DAY}=? AND id>? AND id<=?"
                    " ORDER BY id ASC LIMIT ?",
                    (subj, time.time(), d["day"], eff, int(d["hi"]), int(block)),
                ).fetchall()
            pick, used = [], 0
            for r in rows:
                ln = len(str(r["content"] or "")) + 1
                if pick and used + ln > max_chars:
                    break
                pick.append(dict(r))
                used += ln
            if not pick:
                continue
            return {
                "day": d["day"], "rows": pick,
                "day_n": int(d["n"]), "day_lo": int(d["lo"]), "day_hi": int(d["hi"]),
                "day_items": int(d["items"]), "day_last": int(d["last_id"]),
                "day_index": idx + 1, "total_days": len(days),
                "more_in_day": int(pick[-1]["id"]) < int(d["hi"]),
                "chars": used,
            }
        return None

    def distill_mark(self, subject: str, day: str, last_id: int, items: int) -> None:
        """记下这一天提炼到哪条。只进不退，重放不会把进度拽回去。"""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO distill_day (subject, day, last_id, items, blocks, updated_at)"
                " VALUES (?,?,?,?,1,?) ON CONFLICT(subject, day) DO UPDATE SET"
                " last_id=MAX(last_id, excluded.last_id),"
                " items=items+excluded.items, blocks=blocks+1,"
                " updated_at=excluded.updated_at",
                (str(subject or ""), str(day), int(last_id), int(items), time.time()),
            )

    def distill_progress(self, subject: str, legacy_after_id: int = 0) -> dict:
        """还剩多少条、多少天没提。天数给面板做进度，条数给费用预估。"""
        days = self.distill_days(subject)
        total = left = done_days = 0
        current = ""
        for d in days:
            eff = max(int(d["last_id"]), int(legacy_after_id or 0))
            total += int(d["n"])
            if eff >= int(d["hi"]):
                done_days += 1
                continue
            if not current:
                current = d["day"]
            with self._lock:
                row = self._db.execute(
                    f"SELECT COUNT(*) FROM l3_events WHERE subject=?{self._DISTILL_WHERE}"
                    f" AND {self._DISTILL_DAY}=? AND id>? AND id<=?",
                    (str(subject or ""), time.time(), d["day"], eff, int(d["hi"])),
                ).fetchone()
            try:
                left += int(row[0] or 0)
            except (TypeError, ValueError, IndexError):
                pass
        return {"total": total, "left": left, "days": len(days),
                "done_days": done_days, "days_left": len(days) - done_days,
                "current_day": current}

    def count_l3_for_distill(self, subject: str, after_id: int = 0) -> int:
        now = time.time()
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM l3_events WHERE subject=?"
                " AND event_type IN ('user_message','bot_reply')"
                " AND (session_id='' OR session_id LIKE 'import:%')"
                " AND id>? AND (expires_at=0 OR expires_at>?)",
                (str(subject or ""), int(after_id), now),
            ).fetchone()
        try:
            return int(row[0] or 0)
        except (TypeError, ValueError, IndexError):
            return 0

    def recent_nick_by_session(self, max_sessions: int = 400) -> dict[str, str]:
        """号码会话 → 对方最近用过的昵称。

        记忆块里全是「用户 123456003」这种冷号码，模型对不上「眼前这位是谁」
        也不敢把号码和别的会话里的人认成同一个（实测 AI 收到完整转述记录仍去
        调工具现查）。聊天记录本身就带答案：每条 user_message 都有 session_id
        （号码）和 subject（说话时的昵称），GROUP BY 取每会话最新一条的昵称即可。
        人改名会自然跟随（取最新），不用额外建表维护。

        部分索引 idx_l3_nick 只收 user_message 且 subject 非空的行，
        (session_id, id) 让 GROUP BY 直接在索引上取每组 MAX(id)，
        不用全表扫 + 临时排序（10 万行实测 97ms → ~15ms）。
        """
        sql = (
            "SELECT session_id, subject, MAX(id) AS mid FROM l3_events "
            "INDEXED BY idx_l3_nick "
            "WHERE event_type='user_message' AND subject != '' "
            "GROUP BY session_id ORDER BY mid DESC LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(sql, (int(max_sessions),)).fetchall()
        return {str(r["session_id"]): str(r["subject"]) for r in rows if r["session_id"]}

    def cleanup_expired(self, now: float | None = None,
                        skeleton_fn=None, batch: int = 5000) -> int:
        """物理清理过期的 L3/L2。

        skeleton_fn 非空时（L3 骨架化开启）：L3 过期行先按组交给回调生成
        骨架行（event_type='l3_skeleton'，自带更长的 expires_at）写回再删，
        实现「原话忘成关键词，关键词再随时间忘掉」。回调签名
        skeleton_fn(rows) -> list[dict]，由 summarize_core.make_skeleton 提供。
        L2 过期条目始终直接删（临时状态没有骨架价值）。

        分批处理：积压场景（首次接入老库 / 停摆后恢复）一次可能吞几万行，
        整个过程锁在单事务里会让所有写入卡死（实测 8.6 万行 19 秒）。按批
        加锁-提交-放锁，批次之间写入可以穿插，单次最长阻塞 ≈ 一批的耗时。
        极端情况下一个 (会话,人,天) 组可能被批边界劈成两个半组，各出一条
        骨架——只影响关键词重复度，不影响正确性。
        """
        now = now or time.time()
        total = 0
        while True:
            with self._lock, self._db:
                if skeleton_fn is not None:
                    # 只骨架化「原话」类行；骨架行自己到期就走 DELETE——
                    # 不排除的话骨架会被再压一层（骨架的骨架），无限套娃
                    dying = self._db.execute(
                        "SELECT * FROM l3_events WHERE expires_at > 0 AND expires_at <= ?"
                        " AND event_type != 'l3_skeleton' LIMIT ?",
                        (now, int(batch)),
                    ).fetchall()
                else:
                    dying = []
                if not dying:
                    # 没有待骨架化的行了：一次删掉剩余全部过期行（含到期骨架）
                    cur = self._db.execute(
                        "DELETE FROM l3_events WHERE expires_at > 0 AND expires_at <= ?",
                        (now,),
                    )
                    cur2 = self._db.execute(
                        "DELETE FROM l2_facts WHERE expires_at > 0 AND expires_at <= ?"
                        " AND status='active'",
                        (now,),
                    )
                    total += cur.rowcount + cur2.rowcount
                    break
                try:
                    skeletons = skeleton_fn([dict(r) for r in dying]) or []
                except Exception:
                    skeletons = []
                for s in skeletons:
                    # 骨架行必须带自己的过期时间，否则会永生
                    if not s.get("expires_at"):
                        continue
                    self._db.execute(
                        "INSERT INTO l3_events (event_type, subject, session_id,"
                        " summary_id, content, source_msg_id, actionable,"
                        " expires_at, created_at, bot_id)"
                        " VALUES ('l3_skeleton',?,?,0,?,'',0,?,?,?)",
                        (str(s.get("subject") or "")[:40],
                         str(s.get("session_id") or "")[:120],
                         str(s.get("content") or "")[:200],
                         float(s.get("expires_at")),
                         float(s.get("created_at") or now),
                         str(s.get("bot_id") or "").strip()[:32]
                         or bot_of_session(s.get("session_id"))),
                    )
                ids = [r["id"] for r in dying]
                cur = self._db.execute(
                    f"DELETE FROM l3_events WHERE id IN ({','.join('?' * len(ids))})",
                    ids,
                )
                total += cur.rowcount
        return total

    # ---------- 门控审计 ----------

    def write_audit(self, *, candidate: str, decision: str, reason: str,
                    source_type: str = "", speech_act: str = "", target_layer: str = "",
                    now: float | None = None) -> int:
        now = now or time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO gate_audit (candidate, decision, reason, source_type, speech_act,"
                " target_layer, created_at) VALUES (?,?,?,?,?,?,?)",
                (candidate, decision, reason, source_type, speech_act, target_layer, now),
            )
            return int(cur.lastrowid)

    def recent_audit(self, limit: int = 10, decisions: tuple = ("deny", "transform")) -> list[dict]:
        q = "SELECT * FROM gate_audit"
        args: list = []
        if decisions:
            q += " WHERE decision IN (%s)" % ",".join("?" for _ in decisions)
            args.extend(decisions)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    # ---------- 好感度（bot ↔ 人） ----------

    def affinity_stages(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, threshold, sort, desc FROM affinity_stages"
                " ORDER BY threshold, sort"
            ).fetchall()
        return [dict(r) for r in rows]

    def affinity_stages_save(self, stages: list[dict]) -> list[dict]:
        """全量替换阶段表。接受 [{name, threshold, desc}]，自动按阈值排序。"""
        clean: list[tuple[str, float, str]] = []
        for s in stages or []:
            name = str((s or {}).get("name") or "").strip()[:20]
            if not name:
                continue
            try:
                th = float((s or {}).get("threshold") or 0)
            except (TypeError, ValueError):
                th = 0.0
            desc = str((s or {}).get("desc") or "").strip()[:120]
            clean.append((name, max(0.0, th), desc))
        if not clean:
            return self.affinity_stages()
        clean.sort(key=lambda x: x[1])
        with self._lock, self._db:
            self._db.execute("DELETE FROM affinity_stages")
            for i, (name, th, desc) in enumerate(clean):
                self._db.execute(
                    "INSERT INTO affinity_stages (name, threshold, sort, desc)"
                    " VALUES (?,?,?,?)",
                    (name, th, i, desc),
                )
        return self.affinity_stages()

    def affinity_stages_reset(self) -> list[dict]:
        with self._lock, self._db:
            self._db.execute("DELETE FROM affinity_stages")
            for i, (name, th) in enumerate(DEFAULT_STAGES):
                self._db.execute(
                    "INSERT INTO affinity_stages (name, threshold, sort, desc)"
                    " VALUES (?,?,?,?)",
                    (name, float(th), i, DEFAULT_STAGE_DESCS.get(name, "")),
                )
        return self.affinity_stages()

    def person_call_name_clear(self, person: str) -> dict:
        """删掉「能被称呼的名」：call_name 清空并解锁，原写法降回外号档。

        跟 person_set_call_name（改名）互补——用户想的是「AI 别再这么叫了」，
        所以是清空而不是换一个；显示名不受影响，AI 的称呼退回显示名/键名。
        原写法不直接删：它仍然是这个人的一种叫法，留在别名表里认人用。
        """
        person = (person or "").strip()[:40]
        if not person:
            return {"ok": False, "reason": "invalid"}
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT call_name FROM people WHERE person=?", (person,)).fetchone()
            if not row:
                return {"ok": False, "reason": "not_found"}
            old = str(row["call_name"] or "").strip()
            self._db.execute(
                "UPDATE people SET call_name='', call_locked=0, updated_at=?"
                " WHERE person=?",
                (now, person))
            if old:
                k = alias_key(old)
                if k:
                    self._db.execute(
                        "UPDATE person_alias SET tier=2, updated_at=?"
                        " WHERE alias_key=? AND person=?",
                        (now, k, person))
        return {"ok": True, "person": person, "cleared": old}

    def affinity_get(self, subject: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM affinity WHERE subject=?", (subject or "",)
            ).fetchone()
        return dict(row) if row else None

    def affinity_all(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM affinity ORDER BY xp DESC, updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def affinity_add(self, subject: str, gain: float) -> dict:
        """累加好感度；不存在则建一条。返回累加后的行。"""
        subject = (subject or "").strip()[:40]
        if not subject:
            return {}
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO affinity (subject, xp, gain_total, updated_at) VALUES (?,?,?,?)"
                " ON CONFLICT(subject) DO UPDATE SET xp = xp + excluded.xp,"
                " gain_total = gain_total + excluded.gain_total, updated_at = excluded.updated_at",
                (subject, float(gain), float(gain), now),
            )
            row = self._db.execute(
                "SELECT * FROM affinity WHERE subject=?", (subject,)
            ).fetchone()
        return dict(row) if row else {}

    def affinity_set(self, subject: str, xp: float) -> dict:
        subject = (subject or "").strip()[:40]
        if not subject:
            return {}
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO affinity (subject, xp, gain_total, updated_at) VALUES (?,?,?,?)"
                " ON CONFLICT(subject) DO UPDATE SET xp = excluded.xp, updated_at = excluded.updated_at",
                (subject, max(0.0, float(xp)), max(0.0, float(xp)), now),
            )
            row = self._db.execute(
                "SELECT * FROM affinity WHERE subject=?", (subject,)
            ).fetchone()
        return dict(row) if row else {}

    def affinity_delete(self, subject: str) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM affinity WHERE subject=?", (subject or "",))
            return cur.rowcount > 0

    # ---------- 人物身份（QQ ↔ 称呼） ----------

    def _person_nodes(self) -> set[str]:
        """库里被当成「一个人」出现过的所有写法（含只有好感度/关系、还没登记身份的）。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT person AS n FROM people UNION SELECT subject FROM affinity"
                " UNION SELECT source FROM social_relations"
                " UNION SELECT target FROM social_relations"
            ).fetchall()
        return {str(r["n"] or "").strip() for r in rows if str(r["n"] or "").strip()}

    def person_all(self) -> list[dict]:
        """所有人物 + 各自的称呼列表。面板的「AI 与人」直接吃这个。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT person, qq, note, display_name, call_name, created_at, updated_at"
                " FROM people"
            ).fetchall()
            aliases = self._db.execute(
                "SELECT alias_key, person, raw, pinyin, source, tier, hits, last_seen,"
                " updated_at FROM person_alias ORDER BY tier ASC, hits DESC, updated_at ASC"
            ).fetchall()
        by: dict[str, list[dict]] = {}
        for a in aliases:
            by.setdefault(a["person"], []).append(dict(a))
        out = []
        for r in rows:
            d = dict(r)
            d["aliases"] = by.get(d["person"], [])
            out.append(d)
        return out

    def person_get(self, person: str) -> dict | None:
        person = (person or "").strip()[:40]
        if not person:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT person, qq, note, display_name, call_name, created_at, updated_at"
                " FROM people WHERE person=?",
                (person,),
            ).fetchone()
            if not row:
                return None
            aliases = self._db.execute(
                "SELECT alias_key, person, raw, pinyin, source, tier, hits, last_seen"
                " FROM person_alias WHERE person=? ORDER BY tier ASC, hits DESC, updated_at ASC",
                (person,),
            ).fetchall()
        d = dict(row)
        d["aliases"] = [dict(a) for a in aliases]
        return d

    def person_resolve(self, name: str) -> str:
        """一个称呼对应哪个人。只认字面（alias_key），认不出返回空串。

        读音相近的（丙哥 / 用户丙）在这里**故意不自动认**——同音不同人
        太常见，交给面板的「疑似同一人」去问一次。
        """
        key = alias_key(name)
        if not key:
            return ""
        with self._lock:
            row = self._db.execute(
                "SELECT person FROM person_alias WHERE alias_key=?", (key,)
            ).fetchone()
        return str(row["person"] or "") if row else ""

    def _person_resolve_any(self, name: str) -> tuple[str, str]:
        """整串认不出时，从左边一个个砍字再试。

        「用户说用户丙就是丙哥」抓出来的左半边是「用户说用户丙」，
        整串当然没人认识；砍到「用户丙」就认出来了。
        """
        name = (name or "").strip()
        if not name:
            return "", ""
        hit = self.person_resolve(name)
        if hit:
            return hit, name
        for i in range(1, min(9, len(name))):
            cand = name[i:]
            hit = self.person_resolve(cand)
            if hit:
                return hit, cand
        return "", ""

    def person_names(self, persons) -> dict:
        """person 键 → 「该怎么叫这个人」= call_name，没设过就用 QQ 昵称。

        两个名字分工不同，别混：
        - display_name 是QQ 昵称，稳定、用来认人（群里没称呼时就叫这个）；
        - call_name 是 AI 真正开口叫的那个（比如 QQ 叫 Moon，但一直叫「张三」）。
        AI 说话走这里；面板要显示"这人是谁"走 person_nicknames()。
        """
        names = [str(x or "").strip() for x in (persons or ())]
        names = [x for x in dict.fromkeys(names) if x]
        if not names:
            return {}
        out: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(names), 400):
                part = names[i:i + 400]
                marks = ",".join("?" * len(part))
                rows = self._db.execute(
                    f"SELECT person, display_name, call_name FROM people"
                    f" WHERE person IN ({marks})", tuple(part)
                ).fetchall()
                for r in rows:
                    shown = str(r["call_name"] or "").strip() \
                        or str(r["display_name"] or "").strip() or str(r["person"] or "")
                    out[str(r["person"] or "")] = shown
        return out

    def person_nicknames(self, persons) -> dict:
        """person 键 → QQ 昵称（display_name）。认人用，不是叫法。

        跟 person_names() 的区别：那个返回「怎么叫」，这个返回「这人是谁」。
        两个名字不一样时（QQ 叫 Moon、平时叫张三），注入里会并排写出来，
        模型才既认得出是谁、又叫得对。
        """
        names = [str(x or "").strip() for x in (persons or ())]
        names = [x for x in dict.fromkeys(names) if x]
        if not names:
            return {}
        out: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(names), 400):
                part = names[i:i + 400]
                marks = ",".join("?" * len(part))
                rows = self._db.execute(
                    f"SELECT person, display_name FROM people"
                    f" WHERE person IN ({marks})", tuple(part)
                ).fetchall()
                for r in rows:
                    nick = str(r["display_name"] or "").strip()
                    if nick:
                        out[str(r["person"] or "")] = nick
        return out

    def person_by_qq(self, qq: str) -> str:
        """这个 QQ 是谁。空 QQ 或者没绑过就是空串。

        QQ 是硬证据：昵称能改、马甲能换，号换不了。所以同一个人发消息，
        不管这次叫什么，都该落到同一个 person 上。
        """
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        if not qq:
            return ""
        with self._lock:
            row = self._db.execute(
                "SELECT person FROM people WHERE qq=? ORDER BY created_at LIMIT 1", (qq,)
            ).fetchone()
        return str(row["person"] or "") if row else ""

    def person_qq(self, person: str) -> str:
        """这个人绑的 QQ。没绑过就是空串。"""
        person = (person or "").strip()[:40]
        if not person:
            return ""
        with self._lock:
            row = self._db.execute(
                "SELECT qq FROM people WHERE person=?", (person,)).fetchone()
        return re.sub(r"\D", "", str(row["qq"] or ""))[:20] if row else ""

    def person_identity(self, persons, alias_cap: int = 10) -> dict:
        """person → 身份对照 {shown, nick, qq, aliases:[(称呼, tier)]}。

        注入块的「谁是谁」段和 alias_lookup 工具共用这一份。
        aliases 按 tier 升序（显示名 > 可称呼名 > 外号）、同档按 hits 降序，
        截前 alias_cap 条——称呼量本来就小（tier 上限 + 自动遗忘），
        这里再兜一层，防止个别刷屏号把对照行撑爆。
        """
        names = [str(x or "").strip() for x in (persons or ())]
        names = [x for x in dict.fromkeys(names) if x]
        if not names:
            return {}
        cap = max(1, int(alias_cap))
        out: dict = {}
        with self._lock:
            for i in range(0, len(names), 400):
                part = names[i:i + 400]
                marks = ",".join("?" * len(part))
                prows = self._db.execute(
                    f"SELECT person, qq, display_name, call_name FROM people"
                    f" WHERE person IN ({marks})", tuple(part)).fetchall()
                arows = self._db.execute(
                    f"SELECT person, raw, alias_key, tier, hits FROM person_alias"
                    f" WHERE person IN ({marks})"
                    f" ORDER BY tier ASC, hits DESC, last_seen DESC", tuple(part)).fetchall()
                by_p: dict = {}
                for r in arows:
                    nm = str(r["raw"] or r["alias_key"] or "").strip()
                    if nm:
                        by_p.setdefault(str(r["person"]), []).append((nm, int(r["tier"] or 2)))
                for r in prows:
                    shown = str(r["call_name"] or "").strip() \
                        or str(r["display_name"] or "").strip() \
                        or str(r["person"] or "")
                    p = str(r["person"])
                    out[p] = {
                        "shown": shown,
                        "nick": str(r["display_name"] or "").strip(),
                        "qq": re.sub(r"\D", "", str(r["qq"] or ""))[:20],
                        "aliases": by_p.get(p, [])[:cap],
                    }
            # 没进 people 表但有别名记录的（老库边角）：别名照样给，
            # 别因为少一行档案就把「滚木=用户乙」这条对照弄丢
            for n in names:
                if n not in out and n in by_p:
                    out[n] = {"shown": n, "nick": "", "qq": "",
                              "aliases": by_p[n][:cap]}
        return out

    def alias_owner(self, name: str) -> list[dict]:
        """这个称呼/外号指的是谁。精确命中（归一键/原样/pinyin）优先。

        返回 [{person, alias, tier, exact}]，exact=1 表示字面精确命中。
        认不出时给同前缀的近似候选（最多 5 个，exact=0），供工具提示
        「你是不是想说」。找不到任何东西返回空表。
        """
        key = alias_key((name or "").strip())
        if not key:
            return []
        out: list = []
        with self._lock:
            rows = self._db.execute(
                "SELECT person, raw, tier FROM person_alias WHERE alias_key=?",
                (key,)).fetchall()
            for r in rows:
                out.append({"person": str(r["person"]), "alias": str(r["raw"] or key),
                            "tier": int(r["tier"] or 2), "exact": 1})
            if not out:
                # 原样再试一次（raw 和归一键可能不同：带表情/符号的称呼）
                rows = self._db.execute(
                    "SELECT person, raw, tier FROM person_alias WHERE raw=? LIMIT 3",
                    (str(name or "").strip(),)).fetchall()
                for r in rows:
                    out.append({"person": str(r["person"]), "alias": str(r["raw"]),
                                "tier": int(r["tier"] or 2), "exact": 1})
            if not out:
                rows = self._db.execute(
                    "SELECT person, raw, tier FROM person_alias"
                    " WHERE alias_key LIKE ? OR raw LIKE ?"
                    " ORDER BY hits DESC LIMIT 5",
                    (f"%{key}%", f"%{str(name or '').strip()}%")).fetchall()
                for r in rows:
                    out.append({"person": str(r["person"]), "alias": str(r["raw"] or ""),
                                "tier": int(r["tier"] or 2), "exact": 0})
        return out


    def _qq_conflicts(self, person: str, qq: str) -> bool:
        """这个名字记在 person 名下，但说话的是另一个号 —— 不是同一个人。"""
        qq = re.sub(r"\D", "", str(qq or ""))
        if not qq:
            return False
        owner = self.person_qq(person)
        return bool(owner) and owner != qq

    def _fresh_person(self, name: str, qq: str = "") -> str:
        """同名不同人时，给后来者一个不撞的键。能直接用名字就用名字。"""
        base = (name or "").strip()[:32]
        if not base:
            return f"qq{qq}" if qq else ""
        with self._lock:
            row = self._db.execute(
                "SELECT qq FROM people WHERE person=?", (base,)).fetchone()
        if row is None:
            return base
        # 名字被占了，带号尾区分：张三#0791 和 张三#0601 是两个人
        return f"{base}#{qq[-4:]}" if qq else f"{base}#2"

    def person_ensure(self, name: str, qq: str = "", source: str = "auto",
                      tier: int | None = None) -> str:
        """确保这个称呼有主，返回它的规范名（person 键）。

        认人的顺序是「QQ → 名字 → 新建」：QQ 认得就直接用那个人，昵称改了
        也只是换了个称呼，不会再裂成一个新人。名字认得说明以前见过这个写法。
        都不认得才建新人。

        为什么要这个顺序（2026-11-04 实机）：张三把 QQ 昵称改成 Moon 之后，
        同一个 QQ 的话被写成两个 subject，L2 分成两套、关系图上是两个人。
        只靠名字认是修不了的——这俩名字字面上没有任何关系。

        tier：0=显示名 1=能被称呼的名 2=外号 3=叫错的。不传就按来源推：
        event→0（真实昵称）、quote→2（用户自己说的外号）、其余→3（可能是听错的）。
        """
        name = (name or "").strip()[:40]
        if not name:
            return ""
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        now = time.time()
        key = alias_key(name)
        if not key:
            return ""
        # 泛称不是人。「用户」「对方」这种一旦登记，就多一个永远认不对的假人。
        if is_generic_name(name):
            return ""
        # 认人顺序：QQ → 名字 → 新建。QQ 是硬证据，名字只是称呼。
        # 但如果这个名字已经记在**另一个 QQ** 名下，说明是同名不同人（或者历史上
        # 绑错了）—— 复用会把两个人的记忆搅成一份，只能另起一个。
        person = self.person_by_qq(qq) if qq else ""
        if not person:
            cand = self.person_resolve(name)
            if cand and self._qq_conflicts(cand, qq):
                person = self._fresh_person(name, qq)
            else:
                person = cand or name
                # 称谓词不是名字，除非有人认领过它。「弟弟」「我妈」没有号做锚、
                # 库里也没登记过，就只是 AI 从对话里抄出来的一个说法——拿它建人
                # 只会多一个永远认不回来的假人（实机：XX 是我弟 → 冒出「弟弟」）。
                if not cand and not qq and is_kinship_alias(name):
                    return ""
        if tier is None:
            if source == "event":
                tier = 0
            elif source in ("quote", "manual", "import", "merge", "bootstrap",
                            "ai_tool", "event"):
                tier = 2
            else:
                # summary / auto：AI 从对话里抄出来的写法。可能是听岔的，
                # 归「叫错的」那档，久了不提会被清掉。
                tier = 3
        # 号码不当「当前昵称」（tier 0 会覆盖 display_name，把真昵称顶成号码）。
        # 降成 tier 2 照样登记进别名表——按号认人、session 归属都还要用它。
        if tier == 0 and _is_qq_like(name):
            tier = 2
        # 建档时 display/call 也不许种下号码：留空，等真昵称来了再填。
        _name_seed = "" if _is_qq_like(name) else name
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO people (person, qq, note, display_name, call_name,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(person) DO NOTHING",
                (person, qq, "", _name_seed, _name_seed, now, now),
            )
            if qq:
                # 已有 QQ 就保留（先到先得），空着才补
                self._db.execute(
                    "UPDATE people SET qq=?, updated_at=? WHERE person=? AND qq=''",
                    (qq, now, person),
                )
            if tier == 0:
                # 真实昵称：它就是这个人现在叫什么。上一任显示名降成外号，
                # 否则同一个人会挂着一串 tier=0，面板上分不清哪个是当前名字。
                self._db.execute(
                    "UPDATE person_alias SET tier=2 WHERE person=? AND tier=0 AND alias_key<>?",
                    (person, key),
                )
                # call_name（大家怎么叫他）**不跟着昵称改**：昵称是他自己随手
                # 改的，常用叫法是别人认下来的。跟着改的话每换一次名，AI 的
                # 称呼也跟着变，等于每个人都只有「当前昵称」、没有常用叫法。
                # 面板要的是「本人昵称 + 一个常用叫法」并列，两者得各管各的。
                self._db.execute(
                    "UPDATE people SET display_name=?, updated_at=? WHERE person=?",
                    (name, now, person))
                self._db.execute(
                    "UPDATE people SET call_name=? WHERE person=?"
                    " AND (call_name IS NULL OR call_name='')",
                    (name, person))
            self._db.execute(
                "INSERT INTO person_alias (alias_key, person, raw, pinyin, source,"
                " tier, hits, last_seen, created_at, updated_at) VALUES (?,?,?,?,?,?,1,?,?,?)"
                " ON CONFLICT(alias_key) DO UPDATE SET person=excluded.person,"
                " hits=hits+1, last_seen=excluded.last_seen, updated_at=excluded.updated_at,"
                " pinyin=excluded.pinyin, raw=excluded.raw",
                (key, person, name, pinyin_key(name), source, int(tier), now, now, now),
            )
        return person

    def person_set_call_name(self, person: str, name: str) -> dict:
        """指定「能被称呼的名」。一个人只能有一个，设新的会把旧的降回外号。"""
        person = (person or "").strip()[:40]
        name = (name or "").strip()[:40]
        if not person or not name:
            return {"ok": False, "reason": "invalid"}
        if is_generic_name(name):
            return {"ok": False, "reason": "generic"}
        key = alias_key(name)
        if not key:
            return {"ok": False, "reason": "invalid"}
        owner = self.person_resolve(name)
        if owner and owner != person:
            return {"ok": False, "reason": "taken", "owner": owner}
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO people (person, qq, note, display_name, call_name,"
                " created_at, updated_at) VALUES (?,?,'','',?,?,?)"
                " ON CONFLICT(person) DO NOTHING", (person, "", name, now, now))
            self._db.execute(
                "UPDATE person_alias SET tier=2 WHERE person=? AND tier=1", (person,))
            self._db.execute(
                "INSERT INTO person_alias (alias_key, person, raw, pinyin, source,"
                " tier, hits, last_seen, created_at, updated_at) VALUES (?,?,?,?,?,1,0,?,?,?)"
                " ON CONFLICT(alias_key) DO UPDATE SET person=excluded.person,"
                " tier=1, raw=excluded.raw, updated_at=excluded.updated_at",
                (key, person, name, pinyin_key(name), "manual", now, now, now),
            )
            self._db.execute(
                "UPDATE people SET call_name=?, call_locked=1, updated_at=? WHERE person=?",
                (name, now, person))
        return {"ok": True, "person": person, "call_name": name}

    def person_touch(self, name: str, now: float | None = None) -> None:
        """这个称呼刚被提到过：刷新 last_seen / hits（遗忘要靠它）。"""
        key = alias_key(name)
        if not key:
            return
        now = now or time.time()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE person_alias SET hits=hits+1, last_seen=? WHERE alias_key=?",
                (now, key),
            )

    def _person_has_memory(self, person: str) -> bool:
        """这个人身上有没有值得保的东西（长期记忆 / 好感 / 关系边）。"""
        person = (person or "").strip()
        if not person:
            return False
        with self._lock:
            if self._db.execute(
                    "SELECT 1 FROM l2_facts WHERE subject=? LIMIT 1", (person,)).fetchone():
                return True
            if self._db.execute(
                    "SELECT 1 FROM affinity WHERE subject=? AND xp>0 LIMIT 1",
                    (person,)).fetchone():
                return True
            if self._db.execute(
                    "SELECT 1 FROM social_relations WHERE source=? OR target=? LIMIT 1",
                    (person, person)).fetchone():
                return True
        return False

    def alias_cleanup(self, guess_days: float = 90, nick_days: float = 365,
                      now: float | None = None, dry_run: bool = False) -> dict:
        """忘掉「叫错的名字」。

        只清 tier 3（叫错的）和长期没人叫的外号，顶层两种（显示名 / 能被
        称呼的名）永远不动——那是这个人的身份，不是装饰。

        - tier 3：guess_days 天没出现就删（默认 90）
        - tier 2：nick_days 天没出现、而且从头到尾只被叫过一次才删（默认 365）
        - tier 0/1：永不自动删
        只删称呼映射，人、记忆、关系边都不动。
        """
        now = now or time.time()
        guess_cut = now - max(1.0, float(guess_days)) * 86400
        nick_cut = now - max(1.0, float(nick_days)) * 86400
        with self._lock:
            rows = self._db.execute(
                "SELECT alias_key, person, raw, tier, hits, last_seen FROM person_alias"
                " WHERE tier IN (2, 3)"
            ).fetchall()
        picked = []
        for r in rows:
            tier = int(r["tier"] or 2)
            seen = float(r["last_seen"] or 0)
            if not seen:
                continue
            if tier == 3:
                matched = seen < guess_cut
            else:
                matched = seen < nick_cut and int(r["hits"] or 0) <= 1
            if not matched:
                continue
            # 名下有记忆/好感/关系的人不能删名字：删了之后 AI 再写同一个写法会
            # 认不出来、又建一个新的人，那些记忆就永久分家了。宁可留个错名。
            if self._person_has_memory(r["person"]):
                continue
            picked.append(dict(r))
        if dry_run or not picked:
            return {"removed": 0, "aliases": [p["raw"] for p in picked], "dry_run": bool(dry_run)}
        with self._lock, self._db:
            for p in picked:
                self._db.execute(
                    "DELETE FROM person_alias WHERE alias_key=?", (p["alias_key"],))
        return {"removed": len(picked), "aliases": [p["raw"] for p in picked],
                "dry_run": False}

    def person_bind(self, raw: str, person: str, qq: str = "",
                    source: str = "manual", force: bool = False) -> dict:
        """把一个称呼绑到某个人身上。默认遇到「已经属于别人」的称呼就拒绝，
        force=True 才是用户明确要改绑（面板里点确认时传）。"""
        raw = (raw or "").strip()[:40]
        person = (person or "").strip()[:40]
        if not raw or not person:
            return {"ok": False, "reason": "invalid"}
        if is_generic_name(raw):
            return {"ok": False, "reason": "generic"}
        key = alias_key(raw)
        if not key:
            return {"ok": False, "reason": "invalid"}
        owner = self.person_resolve(raw)
        if owner and owner != person and not force:
            return {"ok": False, "reason": "taken", "owner": owner}
        # 绑到「哪个人」由调用方说了算：只把已经认得的写法归到它原本那个人，
        # 不走 person_ensure 的 QQ/名字解析——否则「把 A 并进 B」会解析成
        # 「把 A 并进 A」，合并直接空转。
        canonical = self.person_resolve(person) or person
        now = time.time()
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO people (person, qq, note, display_name, call_name,"
                " created_at, updated_at) VALUES (?,?,'',?,?,?,?)"
                " ON CONFLICT(person) DO NOTHING",
                (canonical, qq, person, person, now, now),
            )
            self._db.execute(
                "INSERT INTO person_alias (alias_key, person, raw, pinyin, source,"
                " tier, hits, last_seen, created_at, updated_at) VALUES (?,?,?,?,?,2,1,?,?,?)"
                " ON CONFLICT(alias_key) DO UPDATE SET person=excluded.person,"
                " raw=excluded.raw, source=excluded.source, last_seen=excluded.last_seen,"
                " updated_at=excluded.updated_at",
                (key, canonical, raw, pinyin_key(raw), source, now, now, now),
            )
            self._db.execute("UPDATE people SET updated_at=? WHERE person=?", (now, canonical))
        return {"ok": True, "person": canonical, "alias": raw, "replaced": owner}

    def person_set_qq(self, person: str, qq: str) -> dict:
        person = (person or "").strip()[:40]
        if not person:
            return {"ok": False, "reason": "invalid"}
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO people (person, qq, note, created_at, updated_at)"
                " VALUES (?,?,'',?,?) ON CONFLICT(person) DO UPDATE SET"
                " qq=excluded.qq, updated_at=excluded.updated_at",
                (person, qq, now, now),
            )
        return {"ok": True, "person": person, "qq": qq}

    def person_alias_remove(self, alias_key_val: str, person: str = "") -> bool:
        key = (alias_key_val or "").strip()
        if not key:
            return False
        with self._lock, self._db:
            if person:
                cur = self._db.execute(
                    "DELETE FROM person_alias WHERE alias_key=? AND person=?",
                    (key, person or ""),
                )
            else:
                cur = self._db.execute(
                    "DELETE FROM person_alias WHERE alias_key=?", (key,)
                )
            return cur.rowcount > 0

    def person_merge(self, keep: str, drop: str, force: bool = False) -> dict:
        """把 drop 这个人整个并进 keep：称呼、好感度、关系边、记忆归属都搬过去。

        force=True 只给「用户亲手在面板上点的合并」用。自动流程绝不能传：
        两个号各绑着不同的 QQ 就不是同一个人，自动合并会把两个人的一生
        搅成一份，而且事后没法拆。
        """
        keep = (keep or "").strip()[:40]
        drop = (drop or "").strip()[:40]
        if not keep or not drop or keep == drop:
            return {"ok": False, "reason": "invalid"}
        # 两个人各自绑着不同的 QQ，就不可能是同一个人。合并前先挡一道：
        # 一次误点会把朋友的记忆、好感、关系全并到自己身上，而且他的号以后
        # 还会一直认成你 —— 越滚越大，事后很难拆。
        kqq, dqq = self.person_qq(keep), self.person_qq(drop)
        if kqq and dqq and kqq != dqq and not force:
            return {"ok": False, "reason": "qq_conflict",
                    "keep_qq": kqq, "drop_qq": dqq}
        kk, dk = alias_key(keep), alias_key(drop)
        # 先搬关系边（它自己带事务、也会顺带清重复证据）
        rel = self.relation_merge(keep=keep, drop=drop)
        moved = {"aliases": 0, "xp": 0.0, "l2": 0, "l3": 0, "edges": rel.get("edges", 0)}
        now = time.time()
        with self._lock, self._db:
            # QQ：keep 没有就继承 drop 的
            krow = self._db.execute("SELECT qq FROM people WHERE person=?", (keep,)).fetchone()
            drow = self._db.execute("SELECT qq FROM people WHERE person=?", (drop,)).fetchone()
            if krow is None:
                self._db.execute(
                    "INSERT INTO people (person, qq, note, created_at, updated_at)"
                    " VALUES (?,?,'',?,?)", (keep, "", now, now),
                )
                krow = {"qq": ""}
            if drow and not (krow["qq"] or "").strip():
                self._db.execute("UPDATE people SET qq=? WHERE person=?", (drow["qq"], keep))
            # 称呼：drop 名下的都改成 keep；撞键的（同一写法已经在 keep 名下）直接删
            for a in self._db.execute(
                "SELECT alias_key, person FROM person_alias WHERE person=?", (drop,)
            ).fetchall():
                if a["alias_key"] == kk:
                    self._db.execute(
                        "DELETE FROM person_alias WHERE alias_key=?", (a["alias_key"],))
                else:
                    self._db.execute(
                        "UPDATE person_alias SET person=?, updated_at=? WHERE alias_key=?",
                        (keep, now, a["alias_key"]),
                    )
            moved["aliases"] = self._db.execute(
                "SELECT COUNT(*) c FROM person_alias WHERE person=?", (keep,)
            ).fetchone()["c"]
            # 好感度：合并求和，drop 那条删掉
            kxp = self._db.execute("SELECT xp, gain_total FROM affinity WHERE subject=?", (keep,)).fetchone()
            dxp = self._db.execute("SELECT xp, gain_total FROM affinity WHERE subject=?", (drop,)).fetchone()
            if dxp:
                if kxp:
                    self._db.execute(
                        "UPDATE affinity SET xp=xp+?, gain_total=gain_total+?, updated_at=?"
                        " WHERE subject=?",
                        (dxp["xp"], dxp["gain_total"], now, keep),
                    )
                else:
                    self._db.execute(
                        "INSERT INTO affinity (subject, xp, gain_total, updated_at)"
                        " VALUES (?,?,?,?)",
                        (keep, dxp["xp"], dxp["gain_total"], now),
                    )
                self._db.execute("DELETE FROM affinity WHERE subject=?", (drop,))
                moved["xp"] = float(dxp["xp"] or 0)
            # 记忆归属：历史消息挂到同一个名字下，面板才看得到完整的一条人生
            moved["l2"] = int(self._db.execute(
                "UPDATE l2_facts SET subject=? WHERE subject=?", (keep, drop)).rowcount or 0)
            moved["l3"] = int(self._db.execute(
                "UPDATE l3_events SET subject=? WHERE subject=?", (keep, drop)).rowcount or 0)
            self._db.execute("DELETE FROM people WHERE person=?", (drop,))
            self._db.execute("UPDATE people SET updated_at=? WHERE person=?", (now, keep))
        # drop 这个名字本身变成 keep 的称呼（下次 AI 再写它，直接落到 keep 上）
        self.person_bind(drop, keep, source="merge", force=True)
        self._repair_person_names(keep)
        return {"ok": True, "keep": keep, "drop": drop, **moved}

    def person_delete(self, person: str) -> dict:
        """面板手动删除：这个人连同名下所有记忆一起清掉。

        与 person_merge 的「搬过去」相反，这是物理删除、不可恢复：
        称呼、好感、关系边、关系证据、长期事实、短期事件、档案本体全清。
        只给面板（人亲手点）用；自动流程永远不该调它。
        """
        person = (person or "").strip()[:40]
        if not person:
            return {"ok": False, "reason": "invalid"}
        names = [person]
        with self._lock, self._db:
            # 先确认这个人真的在。不先查就会先删掉同名的关系边再报「没找到」，
            # 面板上点了删一个已经不存在的人，反而把别人的边一起带走了。
            if not self._db.execute(
                    "SELECT 1 FROM people WHERE person=?", (person,)).fetchone():
                return {"ok": False, "reason": "not_found"}
            # 日志里的 subject 可能是外号而不是本名，先把登记过的写法全收来
            names += [r["raw"] for r in self._db.execute(
                "SELECT raw FROM person_alias WHERE person=?", (person,)
            ).fetchall()]
            # 泛称发送者会退回用「qq123456」当主体（record_message 里那条兜底），
            # 删人时得把这个变体一起带走，否则它会在「记忆」列表里单开一格。
            qq_row = self._db.execute(
                "SELECT qq FROM people WHERE person=?", (person,)).fetchone()
            if qq_row and str(qq_row["qq"] or "").strip():
                names.append(f"qq{str(qq_row['qq']).strip()}")
            uniq = list(dict.fromkeys(n for n in names if n))
            ph = ",".join("?" for _ in uniq)
            deleted = {"aliases": 0, "xp": 0.0, "l2": 0, "l3": 0,
                       "edges": 0, "evidence": 0, "tokens": 0}
            deleted["aliases"] = self._db.execute(
                "DELETE FROM person_alias WHERE person=?", (person,)).rowcount
            row = self._db.execute(
                "SELECT xp FROM affinity WHERE subject=?", (person,)).fetchone()
            if row:
                deleted["xp"] = float(row["xp"] or 0)
                self._db.execute("DELETE FROM affinity WHERE subject=?", (person,))
            deleted["l2"] = self._db.execute(
                f"DELETE FROM l2_facts WHERE subject IN ({ph})", uniq).rowcount
            deleted["l3"] = self._db.execute(
                f"DELETE FROM l3_events WHERE subject IN ({ph})", uniq).rowcount
            deleted["edges"] = self._db.execute(
                f"DELETE FROM social_relations"
                f" WHERE source IN ({ph}) OR target IN ({ph})", uniq + uniq).rowcount
            deleted["evidence"] = self._db.execute(
                f"DELETE FROM relation_evidence"
                f" WHERE source IN ({ph}) OR target IN ({ph})", uniq + uniq).rowcount
            # 用量记录不删（总账还要对得上），只把归属人抹成空：
            # 面板「用量」里按 subject 列名字，不抹的话删完人还会剩一条「1 · 入 xx / 出 xx」，
            # 看起来就像这个人没删干净。
            deleted["tokens"] = self._db.execute(
                f"UPDATE token_usage SET subject='' WHERE subject IN ({ph})",
                uniq).rowcount
            gone = self._db.execute(
                "DELETE FROM people WHERE person=?", (person,)).rowcount
            if not gone:
                return {"ok": False, "reason": "not_found"}
        return {"ok": True, "person": person, **deleted}

    def person_residue(self, person: str) -> dict:
        """删完之后这个人还剩多少条引用：只读自检，不动数据。

        面板上删完人还能看到名字时，第一个要回答的问题是「库里到底还有没有」。
        这里把所有可能挂名字的列都数一遍（含称呼写法、含 qq123 变体），
        返回 {total, detail:{表.列: 条数}}，total 为 0 就说明是界面没刷新。
        """
        person = (person or "").strip()[:40]
        if not person:
            return {"total": 0, "detail": {}}
        with self._lock:
            names = [person]
            try:
                names += [r["raw"] for r in self._db.execute(
                    "SELECT raw FROM person_alias WHERE person=?", (person,)
                ).fetchall()]
                row = self._db.execute(
                    "SELECT qq FROM people WHERE person=?", (person,)).fetchone()
                if row and str(row["qq"] or "").strip():
                    names.append(f"qq{str(row['qq']).strip()}")
            except Exception:
                pass
            uniq = list(dict.fromkeys(n for n in names if n))
            ph = ",".join("?" for _ in uniq)
            detail: dict[str, int] = {}
            checks = [
                ("people", "person", f"SELECT COUNT(*) c FROM people WHERE person IN ({ph})", uniq),
                ("person_alias", "person", f"SELECT COUNT(*) c FROM person_alias WHERE person IN ({ph})", uniq),
                ("affinity", "subject", f"SELECT COUNT(*) c FROM affinity WHERE subject IN ({ph})", uniq),
                ("l2_facts", "subject", f"SELECT COUNT(*) c FROM l2_facts WHERE subject IN ({ph})", uniq),
                ("l3_events", "subject", f"SELECT COUNT(*) c FROM l3_events WHERE subject IN ({ph})", uniq),
                ("social_relations", "端点", f"SELECT COUNT(*) c FROM social_relations"
                                             f" WHERE source IN ({ph}) OR target IN ({ph})", uniq + uniq),
                ("relation_evidence", "端点", f"SELECT COUNT(*) c FROM relation_evidence"
                                             f" WHERE source IN ({ph}) OR target IN ({ph})", uniq + uniq),
                ("token_usage", "subject", f"SELECT COUNT(*) c FROM token_usage WHERE subject IN ({ph})", uniq),
            ]
            for table, col, sql, args in checks:
                try:
                    n = int(self._db.execute(sql, args).fetchone()["c"] or 0)
                except Exception:
                    continue
                if n:
                    detail[f"{table}.{col}"] = n
        return {"total": sum(detail.values()), "detail": detail}

    def _repair_person_names(self, person: str) -> dict:
        """把一个人的两个顶层名字重新指对：显示名和能被称呼的名都得真实存在。

        合并、改绑、删称呼之后，people.call_name 可能指向一条已经不存在的
        称呼。那之后注入时会显示一个查不到源头的名字，面板也点不动它。
        """
        person = (person or "").strip()[:40]
        if not person:
            return {}
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT display_name, call_name FROM people WHERE person=?", (person,)
            ).fetchone()
            if not row:
                return {}
            aliases = [dict(a) for a in self._db.execute(
                "SELECT alias_key, raw, tier, hits FROM person_alias WHERE person=?", (person,)
            ).fetchall()]
            if not aliases:
                return {}
            by_key = {a["alias_key"]: a for a in aliases}
            self_key = alias_key(person)
            fixed = {}
            # 能被称呼的名：优先和 person 同名的那条，其次 hits 最高的
            call = str(row["call_name"] or "").strip()
            if not call or alias_key(call) not in by_key:
                pick = by_key.get(self_key) or max(aliases, key=lambda a: int(a["hits"] or 0))
                call = pick["raw"]
                fixed["call_name"] = call
            # 显示名：没有就退到 call
            disp = str(row["display_name"] or "").strip()
            if not disp or alias_key(disp) not in by_key:
                disp = call
                fixed["display_name"] = disp
            # 顶层只留一条：显示名那条是 tier=0，称呼名是 tier=1。
            # 两者是同一个名字时按显示名算（同一条不挂两个档）。
            for a in aliases:
                want = 0 if a["raw"] == disp else (1 if a["raw"] == call else None)
                if want is not None and int(a["tier"] or 0) != want:
                    self._db.execute(
                        "UPDATE person_alias SET tier=? WHERE alias_key=?",
                        (want, a["alias_key"]))
            self._db.execute(
                "UPDATE people SET display_name=?, call_name=?, updated_at=? WHERE person=?",
                (disp, call, now, person))
        return fixed

    def person_suggestions(self) -> list[dict]:
        """读音一样、写法不同 → 疑似同一个人，列出来让人点一下确认。

        这里只出建议，绝不自动合并：王伟 / 王薇 同音，合错了比不合更难查。
        """
        items = self.person_all()
        by: dict[str, list[dict]] = {}
        for p in items:
            keys = {a.get("pinyin") for a in p.get("aliases") or []}
            keys.add(pinyin_key(p["person"]))
            for k in keys:
                if k and len(k) >= 3:
                    by.setdefault(k, []).append(p)
        out = []
        for key, group in by.items():
            people = {p["person"]: p for p in group}
            if len(people) < 2:
                continue
            out.append({
                "pinyin": key,
                "people": [{
                    "person": p["person"],
                    "qq": p.get("qq") or "",
                    "aliases": [a.get("raw") for a in p.get("aliases") or []],
                } for p in people.values()],
            })
        out.sort(key=lambda x: -len(x["people"]))
        return out

    def person_learn_aliases(self, text: str) -> list[dict]:
        """从一句话里学「谁又叫什么」。只认明说的那几种说法，识别不出就不动。

        实机踩过的坑（2026-10-28）：用户说「用户丙 就是丙哥」，总结把
        丙哥当成新人物写进关系图，于是同一个人在图上裂成两个节点，
        好感度和关系也各记一半。这句原话本身就写着答案，所以直接学它。
        """
        text = str(text or "")[:600]
        if not text:
            return []
        probe = _strip_lead(text)
        found: list[tuple[str, str]] = []
        for pat in _ALIAS_PATTERNS:
            for m in pat.finditer(probe):
                a = _trim_alias(m.group("a"))
                b = _trim_alias(m.group("b"))
                if _alias_ok(a) and _alias_ok(b):
                    found.append((a, b))
        done: list[dict] = []
        for a, b in found:
            pa, pa_raw = self._person_resolve_any(a)
            pb, pb_raw = self._person_resolve_any(b)
            if pa and pb and pa != pb:
                # 两边都认得、却是两个人：这句话本身就是在说「他们是同一个」。
                # 留分量重的那个（有记忆、有边、说话多的），另一个并进去。
                keep, drop = ((pa, pb) if self._person_weight(pa) >= self._person_weight(pb)
                              else (pb, pa))
                r = self.person_bind(drop, keep, source="quote", force=True)
                if r.get("ok"):
                    done.append({"keep": keep, "alias": drop, "merged": True})
                    self._fold_one(keep, drop)
            elif pa and not pb:
                r = self.person_bind(b, pa, source="quote")
                if r.get("ok"):
                    done.append({"keep": pa, "alias": b})
                    self._fold_one(pa, b)
            elif pb and not pa:
                r = self.person_bind(a, pb, source="quote")
                if r.get("ok"):
                    done.append({"keep": pb, "alias": a})
                    self._fold_one(pb, a)
        return done

    def _fold_one(self, keep: str, raw: str) -> dict:
        """刚学到一个称呼，而这个写法正被当成独立的人存着 → 当场并掉。

        不然要等下次重载/点「重新识别」才合，中间这段时间关系图上还是两个人。
        """
        raw = (raw or "").strip()
        if not raw or alias_key(raw) == alias_key(keep):
            return {}
        if getattr(self, "_folding", False):
            return {}
        with self._lock:
            row = self._db.execute(
                "SELECT person FROM people WHERE person=?", (raw,)).fetchone()
        if not row:
            return {}
        self._folding = True
        try:
            return self.person_merge(keep=keep, drop=raw)
        finally:
            self._folding = False

    def _person_weight(self, person: str) -> float:
        """一个人「有多真」：好感度 + 关系条数 + 说话次数，用来决定合并时留哪个名字。"""
        person = (person or "").strip()
        if not person:
            return 0.0
        with self._lock:
            xp = self._db.execute(
                "SELECT xp FROM affinity WHERE subject=?", (person,)).fetchone()
            edges = self._db.execute(
                "SELECT COUNT(*) c FROM social_relations WHERE source=? OR target=?",
                (person, person)).fetchone()
            alias = self._db.execute(
                "SELECT COUNT(*) c, COALESCE(SUM(hits),0) h FROM person_alias WHERE person=?",
                (person,)).fetchone()
            msgs = self._db.execute(
                "SELECT COUNT(*) c FROM l3_events WHERE subject=?", (person,)).fetchone()
        return (float(xp["xp"] or 0) if xp else 0.0) + int(edges["c"] or 0) * 20 \
            + int(alias["h"] or 0) + int(msgs["c"] or 0) * 0.5

    def person_fold_aliases(self) -> list[dict]:
        """把「同一个人被拆成几个节点」的情况就地并回去。

        三条判据，从硬到软：
        1. 同一个字面键（只是零宽字符/全半角/备注括号不同）——一定是同一个人；
        2. 已经登记过的称呼（用户说「A 就是 B」学来的）——按登记合并；
        3. 同一个 QQ——这是身份锚，撞号的一定是同一个人。
        同音不同字（王伟/王薇）**不在这里合**，只在面板里当建议提出来。
        """
        done: list[dict] = []
        # 1) 字面同键
        by_key: dict[str, list[str]] = {}
        for n in self._person_nodes():
            k = alias_key(n)
            if k:
                by_key.setdefault(k, []).append(n)
        for names in by_key.values():
            if len(names) < 2:
                continue
            names.sort(key=lambda n: -self._person_weight(n))
            keep = names[0]
            for n in names[1:]:
                r = self.person_merge(keep=keep, drop=n)
                if r.get("ok"):
                    done.append({"keep": keep, "drop": n, "why": "同一写法"})
        # 2) 学到过的称呼
        nodes = self._person_nodes()
        for p in self.person_all():
            keep = p["person"]
            for a in p.get("aliases") or []:
                raw = str(a.get("raw") or "").strip()
                if not raw or alias_key(raw) == alias_key(keep):
                    continue
                if raw in nodes and self.person_resolve(raw) == keep:
                    r = self.person_merge(keep=keep, drop=raw)
                    if r.get("ok"):
                        done.append({"keep": keep, "drop": raw, "why": "学到的称呼"})
                        nodes.discard(raw)
        # 3) 同一个 QQ
        by_qq: dict[str, list[str]] = {}
        for p in self.person_all():
            qq = str(p.get("qq") or "").strip()
            if qq:
                by_qq.setdefault(qq, []).append(p["person"])
        for qq, names in by_qq.items():
            if len(names) < 2:
                continue
            names.sort(key=lambda n: -self._person_weight(n))
            keep = names[0]
            for n in names[1:]:
                r = self.person_merge(keep=keep, drop=n)
                if r.get("ok"):
                    done.append({"keep": keep, "drop": n, "why": f"同一个 QQ {qq}"})
        return done

    def person_learn_from_evidence(self, limit: int = 800) -> list[dict]:
        """把历史关系证据里「A 就是 B」这类原话扫一遍，补登记称呼。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT quote FROM relation_evidence WHERE quote<>''"
                " ORDER BY id DESC LIMIT ?", (int(limit),),
            ).fetchall()
        done: list[dict] = []
        for r in rows:
            try:
                done.extend(self.person_learn_aliases(r["quote"]))
            except Exception:
                continue
        return done

    @staticmethod
    def qq_from_session(session_id: str) -> str:
        """私聊会话串里带着对方的 QQ：`111:FriendMessage:123456003` → 123456003。

        群聊/临时会话一律返回空：群号不是 QQ，套到人身上比没有更糟。
        """
        sid = str(session_id or "")
        parts = sid.split(":")
        if len(parts) < 3 or "FriendMessage" not in parts[-2]:
            return ""
        qq = re.sub(r"\D", "", parts[-1])[:20]
        return qq if 5 <= len(qq) <= 12 else ""

    def realign_subjects(self) -> dict:
        """把历史数据里「其实是同一个人、却存成两个 subject」的记录改到规范名上。

        2026-11-04 实机：同一个人把昵称从「Moon」改成「张三」，L3/L2 各存了一套，
        后来人物库里并成一个人，但记忆的 subject 还是旧昵称——面板上看是两个人，
        注入时也分成两段。这里按「名字 → 人」的映射回填一次。

        只改能明确认到某个人的写法；认不出的原样留着，绝不猜。
        """
        moved = {"l2": 0, "l3": 0, "affinity": 0, "rels": 0}
        for table, key in (("l2_facts", "l2"), ("l3_events", "l3")):
            with self._lock:
                names = [str(r[0] or "") for r in self._db.execute(
                    f"SELECT DISTINCT subject FROM {table} WHERE subject<>''")]
            for name in names:
                target = self.person_resolve(name)
                if not target or target == name:
                    continue
                with self._lock, self._db:
                    cur = self._db.execute(
                        f"UPDATE {table} SET subject=? WHERE subject=?", (target, name))
                    moved[key] += int(cur.rowcount or 0)
        # 好感度是主键，撞了要合并求和，不能直接改名
        with self._lock:
            affs = [dict(r) for r in self._db.execute(
                "SELECT subject, xp, gain_total FROM affinity")]
        now = time.time()
        for r in affs:
            target = self.person_resolve(r["subject"])
            if not target or target == r["subject"]:
                continue
            with self._lock, self._db:
                self._db.execute(
                    "INSERT INTO affinity (subject, xp, gain_total, updated_at)"
                    " VALUES (?,?,?,?) ON CONFLICT(subject) DO UPDATE SET"
                    " xp=xp+excluded.xp, gain_total=gain_total+excluded.gain_total,"
                    " updated_at=excluded.updated_at",
                    (target, float(r["xp"] or 0), float(r["gain_total"] or 0), now))
                self._db.execute("DELETE FROM affinity WHERE subject=?", (r["subject"],))
                moved["affinity"] += 1
        # 关系边有 UNIQUE(source,target)：先删掉会因为改名而撞车的那条，再改
        with self._lock:
            rels = [dict(r) for r in self._db.execute(
                "SELECT id, source, target FROM social_relations")]
        for e in rels:
            ns = self.person_resolve(e["source"]) or e["source"]
            nt = self.person_resolve(e["target"]) or e["target"]
            if ns == e["source"] and nt == e["target"]:
                continue
            with self._lock, self._db:
                if ns == nt:
                    self._db.execute("DELETE FROM social_relations WHERE id=?", (e["id"],))
                else:
                    self._db.execute(
                        "DELETE FROM social_relations WHERE source=? AND target=? AND id<>?",
                        (ns, nt, e["id"]))
                    self._db.execute(
                        "UPDATE social_relations SET source=?, target=? WHERE id=?",
                        (ns, nt, e["id"]))
                moved["rels"] += 1
        return moved

    def person_bootstrap(self, learn: bool = True) -> dict:
        """把库里已经存在的人补登记成「人物」，QQ 能从会话串里推的顺手推出来。

        不这么做的话，`person_learn_aliases` 永远学不到东西：它只认
        「一边认得、另一边不认得」，而一开始谁都没登记过。
        """
        created = 0
        qq_bound = 0
        with self._lock:
            pairs = self._db.execute(
                "SELECT subject, session_id FROM l3_events"
                " WHERE subject<>'' AND session_id<>'' GROUP BY subject, session_id"
            ).fetchall()
        for r in pairs:
            name = str(r["subject"] or "").strip()
            if not name:
                continue
            qq = self.qq_from_session(r["session_id"])
            before = self.person_resolve(name)
            # person_ensure 会拒绝泛称/称谓词，返回空串——那就没建人，不能算数
            made = self.person_ensure(name, qq=qq, source="bootstrap")
            created += 1 if (made and not before) else 0
            if qq:
                cur = self.person_get(made or before) or {}
                qq_bound += 1 if (cur.get("qq") or "") == qq else 0
        # （2026-11-06 用户定稿）建档从严：新建人物必须有「对话 + 能拿到 QQ 号
        # 和名称」的证据。上面那段 L3 扫库满足这个条件（真说过话）；这里剩下
        # 的只有好感度或关系边、从没说过话的名字，一概**不再建档**——它们继续
        # 作为关系图节点/好感记录存在，但 people 表里不再多一个查无此人的假人。
        # 手动操作（面板加关系/绑 QQ）不受影响，那是用户亲口认的人。
        learned = self.person_learn_from_evidence() if learn else []
        folded = self.person_fold_aliases()
        # 人并完了还要把记忆的归属改过来，否则面板上「张三」和「Moon」还是
        # 两个人——people 表只剩一行，L2 里却各躺一套。
        realigned = self.realign_subjects()
        return {"created": created, "qq_bound": qq_bound,
                "learned": learned, "folded": folded, "realigned": realigned}

    # ---------- 人际关系（人 ↔ 人，AI 从对话里抓的） ----------

    def _rel_names(self) -> list[str]:
        """关系图里出现过的所有名字（原样，不归并）。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT source AS s FROM social_relations UNION "
                "SELECT target AS s FROM social_relations"
            ).fetchall()
        return [str(r["s"] or "") for r in rows if r["s"]]

    def canonical_name(self, name: str, extra: tuple[str, ...] = ()) -> str:
        """把名字映射到已有的写法上：同一个人已经用「张三」存过，就不再新建「张三（群友）」。

        认不出就原样返回（只做去空白/截断），所以新名字永远不会被瞎合并。
        `extra` 用来把默认要保留的名字排除在候选之外。
        """
        name = (name or "").strip()[:40]
        key = alias_key(name)
        if not key:
            return name
        counts: dict[str, int] = {}
        for n in self._rel_names():
            k = alias_key(n)
            if k and k == key and n not in extra:
                counts[n] = counts.get(n, 0) + 1
        if not counts:
            return name
        # 同一个键可能有多种写法（历史脏数据），挑关系最多的那个，其次挑短的
        return max(counts, key=lambda n: (counts[n], -len(n)))

    def relation_upsert(self, *, source: str, target: str, relation: str,
                        quote: str = "", confidence: float = 0.6,
                        source_type: str = "summary", session_id: str = "",
                        summary_id: int = 0) -> dict:
        """写一条「source 认为 target 是 relation」。同一条边只保留最新档位，
        每次写入都追加一条证据（同内容的证据自动去重）。

        写入前会先把两个人名归并到已有的写法上，避免「张三」和「张三（群友）」
        被当成两个人、各存一条一模一样的边。"""
        source = (source or "").strip()[:40]
        target = (target or "").strip()[:40]
        relation = (relation or "").strip()[:20]
        if not source or not target or not relation:
            return {"ok": False, "reason": "invalid"}
        quote = (quote or "").strip()[:200]
        # 这句话里可能就写着答案：「用户丙就是丙哥」。先学着，下面解析名字
        # 时两边才能落到同一个人身上，不然同一个人会在图上裂成两个节点。
        if quote:
            try:
                self.person_learn_aliases(quote)
            except Exception:
                pass
        # 先各自归并，再拿归并结果互相比对：归并后撞成同一个人说明这层关系无从记起
        raw_src, raw_dst = source, target
        # 称谓词不是名字。「弟弟」「我妈」被自动写进 source/target，库里又没人
        # 认领过它，就只是一句称呼——建这条边等于凭空造一个叫「弟弟」的人。
        # 手动写（面板/用户明说）例外：那是人自己认的关系，照记。
        if source_type != "manual":
            for _nm in (source, target):
                if is_kinship_alias(_nm) and not self.person_resolve(_nm):
                    return {"ok": False, "reason": "kinship", "name": _nm}
        # 身份优先：认得出是谁就落到那个人身上（丙哥 → 用户丙），
        # 认不出再退回字面归并（「张三」与「张三（群友）」）。
        cs = self.person_resolve(source) or self.canonical_name(source)
        ct = self.person_resolve(target) or self.canonical_name(target)
        if cs and ct:
            source, target = cs, ct
        if source == target:
            return {"ok": False, "reason": "same_person", "source": source, "target": target}
        # 新出现的人也要有身份记录，不然面板上他就是「没名字的路人」。
        # 但「建人」这个动作只留给手动操作（面板加关系、用户明说）：自动总结
        # 写出来的名字认不出是谁，说明它只是在猜，不该因此多一个用户。
        # 边照写（图上节点由边推出来，不需要 people 行），人一个不建。
        if source_type == "manual":
            try:
                self.person_ensure(source, source=source_type)
                self.person_ensure(target, source=source_type)
            except Exception:
                pass
        now = time.time()
        with self._lock, self._db:
            old = self._db.execute(
                "SELECT relation FROM social_relations WHERE source=? AND target=?",
                (source, target),
            ).fetchone()
            old_relation = old["relation"] if old else ""
            if old:
                self._db.execute(
                    "UPDATE social_relations SET relation=?, confidence=?, source_type=?,"
                    " session_id=?, summary_id=?, updated_at=? WHERE source=? AND target=?",
                    (relation, float(confidence), source_type, session_id,
                     int(summary_id), now, source, target),
                )
            else:
                self._db.execute(
                    "INSERT INTO social_relations (source, target, relation, confidence,"
                    " source_type, session_id, summary_id, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (source, target, relation, float(confidence), source_type,
                     session_id, int(summary_id), now, now),
                )
            evidence_added = 0
            if quote:
                dup = self._db.execute(
                    "SELECT id FROM relation_evidence WHERE source=? AND target=? AND quote=?",
                    (source, target, quote),
                ).fetchone()
                if not dup:
                    self._db.execute(
                        "INSERT INTO relation_evidence (source, target, relation, quote,"
                        " source_type, session_id, summary_id, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (source, target, relation, quote, source_type,
                         session_id, int(summary_id), now),
                    )
                    evidence_added = 1
        return {"ok": True, "changed": old_relation != relation,
                "old": old_relation, "relation": relation,
                "evidence_added": evidence_added,
                "source": source, "target": target,
                "aliased": (source, target) != (raw_src, raw_dst)}

    def relations_all(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM relation_evidence e"
                " WHERE e.source=r.source AND e.target=r.target) AS evidence_count"
                " FROM social_relations r ORDER BY r.updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def relation_evidence(self, source: str, target: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM relation_evidence WHERE source=? AND target=?"
                " ORDER BY id DESC LIMIT ?",
                (source or "", target or "", int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def relation_evidence_update(self, ev_id: int, quote: str) -> bool:
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE relation_evidence SET quote=? WHERE id=?",
                ((quote or "").strip()[:200], int(ev_id)),
            )
            return cur.rowcount > 0

    def relation_evidence_delete(self, ev_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM relation_evidence WHERE id=?", (int(ev_id),))
            return cur.rowcount > 0

    def relation_delete_edge(self, source: str, target: str, drop_evidence: bool = True) -> int:
        """删一条边。先按原样删；一条都没删到就按归并后的名字再试一次
        （面板和 AI 可能用了同一个人的两种写法），返回真实删掉的条数。"""
        src = (source or "").strip()
        dst = (target or "").strip()
        # 归并要在拿锁之前算：canonical_name 自己会去读库、也要拿这把锁
        alt = (self.canonical_name(src), self.canonical_name(dst))

        def _run(s: str, t: str) -> int:
            cur = self._db.execute(
                "DELETE FROM social_relations WHERE source=? AND target=?", (s, t)
            )
            n = cur.rowcount
            if drop_evidence:
                cur2 = self._db.execute(
                    "DELETE FROM relation_evidence WHERE source=? AND target=?", (s, t)
                )
                n += cur2.rowcount
            return n

        with self._lock, self._db:
            n = _run(src, dst)
            if n == 0 and alt != (src, dst):
                n = _run(alt[0], alt[1])
            return n

    def relation_alias_groups(self) -> list[dict]:
        """找「名字不同但看起来是同一个人」的组，给面板的合并功能用。"""
        counts: dict[str, int] = {}
        for e in self.relations_all():
            for n in (e["source"], e["target"]):
                if n:
                    counts[n] = counts.get(n, 0) + 1
        groups: dict[str, list[dict]] = {}
        for n, c in counts.items():
            k = alias_key(n)
            if k:
                groups.setdefault(k, []).append({"name": n, "edges": c})
        out = []
        for k, items in groups.items():
            if len(items) > 1:
                items.sort(key=lambda x: (-x["edges"], len(x["name"]), x["name"]))
                out.append({"key": k, "names": items})
        out.sort(key=lambda g: (-len(g["names"]), g["key"]))
        return out

    def relation_merge(self, keep: str, drop: str) -> dict:
        """把 drop 这个人并进 keep：所有以 drop 为端点的边和证据都改挂到 keep 上。

        两边都有同一方向的边时，保留 updated_at 更新的那条档位，旧的那条丢掉。
        drop→drop 这种自环直接删。返回改了几条边、几条证据。
        """
        keep = (keep or "").strip()[:40]
        drop = (drop or "").strip()[:40]
        if not keep or not drop or keep == drop:
            return {"ok": False, "reason": "invalid"}
        moved_edges = 0
        merged_edges = 0
        now = time.time()
        with self._lock, self._db:
            rows = self._db.execute(
                "SELECT id, source, target, relation, confidence, updated_at"
                " FROM social_relations WHERE source IN (?,?) OR target IN (?,?)",
                (keep, drop, keep, drop),
            ).fetchall()
            for r in rows:
                src = keep if r["source"] == drop else r["source"]
                tgt = keep if r["target"] == drop else r["target"]
                if src == tgt:
                    self._db.execute("DELETE FROM social_relations WHERE id=?", (r["id"],))
                    merged_edges += 1
                    continue
                ex = self._db.execute(
                    "SELECT id, relation, confidence, updated_at FROM social_relations"
                    " WHERE source=? AND target=? AND id<>?", (src, tgt, r["id"]),
                ).fetchone()
                if ex:
                    # 同一对同一方向两条：留时间新的那条（档位、置信度一起搬过去）
                    if float(r["updated_at"] or 0) > float(ex["updated_at"] or 0):
                        self._db.execute(
                            "UPDATE social_relations SET relation=?, confidence=?, updated_at=?"
                            " WHERE id=?",
                            (r["relation"], r["confidence"], r["updated_at"], ex["id"]),
                        )
                    self._db.execute("DELETE FROM social_relations WHERE id=?", (r["id"],))
                    moved_edges += 1
                    continue
                self._db.execute(
                    "UPDATE social_relations SET source=?, target=?, updated_at=? WHERE id=?",
                    (src, tgt, max(now, float(r["updated_at"] or 0)), r["id"]),
                )
                moved_edges += 1
            ev1 = self._db.execute(
                "UPDATE relation_evidence SET source=? WHERE source=?", (keep, drop))
            ev2 = self._db.execute(
                "UPDATE relation_evidence SET target=? WHERE target=?", (keep, drop))
            moved_ev = int(ev1.rowcount or 0) + int(ev2.rowcount or 0)
            # 自环证据（drop→drop 变成 keep→keep）没有意义，清掉
            self._db.execute(
                "DELETE FROM relation_evidence WHERE source=? AND target=?", (keep, keep))
            # 合并后可能出现「同一条边、同一句话」的重复证据，只留最早那条
            self._db.execute(
                "DELETE FROM relation_evidence WHERE id NOT IN ("
                " SELECT MIN(id) FROM relation_evidence GROUP BY source, target, quote)")
        return {"ok": True, "keep": keep, "drop": drop,
                "edges": moved_edges, "merged": merged_edges, "evidence": moved_ev}


    def relation_fold_aliases(self) -> list[dict]:
        """把「同一个人被写成几种样子」的边自动折到同一个名字上。

        启动时跑一遍，用来清历史脏数据：老版本不认别名，AI 写「用户乙（用户甲）」、
        面板写「用户乙」，库里就有两行、图上两个节点，看着像两条一样的单向关系。
        保留哪个写法由 relation_alias_groups 排序决定（边多的 → 名字短的 → 字典序）。
        """
        done: list[dict] = []
        for g in self.relation_alias_groups():
            names = [x["name"] for x in g["names"]]
            if len(names) < 2:
                continue
            keep = names[0]
            for n in names[1:]:
                try:
                    r = self.relation_merge(keep=keep, drop=n)
                except Exception:
                    continue
                if r.get("ok"):
                    done.append({"keep": r.get("keep", keep), "drop": n})
                    keep = r.get("keep", keep)
        return done

    # ---------- 群 / 人 区分 ----------

    def group_remember(self, gid: str, name: str = "") -> None:
        """记下这个群。群号一定有，群名可能拿不到（有些适配器不给）。"""
        gid = str(gid or "").strip()[:40]
        if not gid:
            return
        name = str(name or "").strip()[:60]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO known_groups (gid, name, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(gid) DO UPDATE SET"
                " name=CASE WHEN excluded.name<>'' THEN excluded.name"
                "           ELSE known_groups.name END,"
                " updated_at=excluded.updated_at",
                (gid, name, time.time()),
            )

    def group_keys(self) -> set[str]:
        """所有已知群号 + 群名（casefold 后），用来判断一个 subject 到底是人还是群。"""
        with self._lock:
            rows = self._db.execute("SELECT gid, name FROM known_groups").fetchall()
        keys: set[str] = set()
        for r in rows:
            for v in (r["gid"], r["name"]):
                v = str(v or "").strip()
                if v:
                    keys.add(v.casefold())
        return keys

    def group_backfill(self) -> int:
        """从历史会话里把群号捞出来：session_id 形如 `111:GroupMessage:12345`。

        老版本没登记过群，但 L3 里存着会话来源，能反推出哪些是群。
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT session_id FROM l3_events"
                " WHERE session_id LIKE '%GroupMessage%'"
            ).fetchall()
        n = 0
        for r in rows:
            parts = str(r["session_id"] or "").split(":")
            if len(parts) >= 3 and "GroupMessage" in parts[1]:
                gid = parts[-1].strip()
                if gid:
                    self.group_remember(gid)
                    n += 1
        return n

    def relation_subjects(self) -> list[str]:
        """关系图里出现过的所有人。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT source AS s FROM social_relations UNION "
                "SELECT target AS s FROM social_relations"
            ).fetchall()
        return sorted({str(r["s"] or "") for r in rows if r["s"]})

    # ---------- Token 用量 ----------

    def token_record(self, module: str, *, input: int = 0, cached: int = 0, output: int = 0,
                     source: str = "actual", provider_id: str = "", subject: str = "",
                     session_id: str = "", total: int | None = None,
                     ts: float | None = None, arch_ver: str = "", inject_fp: str = "",
                     rebuilt: int = 0, freeze_rounds: int = 0, cfg_ver: str = "",
                     rebuild_reason: str = "") -> int:
        """记一条用量。source=actual 是 provider 报的真实 usage，estimate 是我们估的。

        arch_ver/inject_fp/rebuilt/freeze_rounds/cfg_ver/rebuild_reason 只对
        module='chat' 有意义：它们描述这一轮请求挂在 system 里的是哪份冻结
        前缀、是不是刚重建过、为什么重建。
        注意 cached 是**整轮请求**的前缀缓存（框架 system + 历史 + 工具 schema），
        不等于「记忆块命中率」，只有把同 fp 连续轮与重建轮分开对比才有意义。
        """
        total = int(total if total is not None else (input + cached + output))
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO token_usage (ts, module, source, provider_id, subject, session_id,"
                " input, cached, output, total, arch_ver, inject_fp, rebuilt, freeze_rounds,"
                " cfg_ver, rebuild_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (float(ts or time.time()), str(module or "other")[:32], str(source or "actual")[:16],
                 str(provider_id or "")[:64], str(subject or "")[:40], str(session_id or "")[:120],
                 int(input or 0), int(cached or 0), int(output or 0), max(0, total),
                 str(arch_ver or "")[:16], str(inject_fp or "")[:32],
                 1 if rebuilt else 0, max(0, int(freeze_rounds or 0)),
                 str(cfg_ver or "")[:32], str(rebuild_reason or "")[:24]),
            )
            return int(cur.lastrowid)

    def token_last_inject_fp(self, session_id: str) -> str:
        """这个会话上一轮实际挂进 system 的冻结包 fp（跨进程重启也认）。

        冻结块本身只存在内存 _inject_cache 里，插件一重载就没了；要判断
        「这次重建是不是重载冷启动、字节其实没变」，只能从库里取上一次的 fp。
        只查 module='chat' AND source='actual' 的真实账单，estimate 行没有 fp。
        """
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT inject_fp FROM token_usage WHERE session_id=? AND module='chat'"
                    " AND source='actual' AND inject_fp<>'' ORDER BY id DESC LIMIT 1",
                    (str(session_id or ""),),
                ).fetchone()
            return str(row["inject_fp"] or "") if row else ""
        except Exception:
            return ""

    def token_totals(self, since: float = 0.0) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) calls, COALESCE(SUM(input),0) input, COALESCE(SUM(cached),0) cached,"
                " COALESCE(SUM(output),0) output, COALESCE(SUM(total),0) total"
                " FROM token_usage WHERE ts >= ?", (float(since),),
            ).fetchone()
        return {k: (row[k] if k == "calls" else int(row[k] or 0)) for k in
                ("calls", "input", "cached", "output", "total")}

    def token_by_module(self, since: float = 0.0) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT module, COUNT(*) calls, COALESCE(SUM(input),0) input,"
                " COALESCE(SUM(cached),0) cached, COALESCE(SUM(output),0) output,"
                " COALESCE(SUM(total),0) total FROM token_usage WHERE ts >= ?"
                " GROUP BY module ORDER BY total DESC", (float(since),),
            ).fetchall()
        return [{"module": r["module"], "calls": int(r["calls"]), "input": int(r["input"]),
                 "cached": int(r["cached"]), "output": int(r["output"]),
                 "total": int(r["total"])} for r in rows]

    def token_series(self, since: float, step: float, offset: float = 0.0) -> list[dict]:
        """按固定宽度分桶（step 秒），返回 [{bucket, module, total}]，空桶不返回。
        offset 用来把桶对齐到本地时间（传 -tz，东八区传 -28800）。"""
        step = max(60.0, float(step))
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, module, total FROM token_usage WHERE ts >= ?", (float(since),)
            ).fetchall()
        agg: dict[tuple[float, str], int] = {}
        for r in rows:
            b = int((float(r["ts"]) - offset) / step) * step + offset
            key = (float(b), str(r["module"]))
            agg[key] = agg.get(key, 0) + int(r["total"] or 0)
        return [{"bucket": b, "module": m, "total": t}
                for (b, m), t in sorted(agg.items())]

    def token_recent(self, limit: int = 40) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM token_usage ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def token_chat_rows(self, since: float = 0.0, limit: int = 20000) -> list[dict]:
        """缓存命中率页用：只取 provider 上报了真实 usage 的对话轮。

        estimate 行（注入块、关系快照、工具结果都是按文本估的）cached 恒为 0，
        混进来会把命中率算低；module='chat' AND source='actual' 才是真账单。
        一次取回在调用方做聚合，避免为了各种分组反复查库。
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM token_usage WHERE ts >= ? AND module='chat'"
                " AND source='actual' ORDER BY id DESC LIMIT ?",
                (float(since), max(1, int(limit))),
            ).fetchall()
        return [dict(r) for r in rows]

    def token_first_ts(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT MIN(ts) t FROM token_usage").fetchone()
        return float(row["t"] or 0) if row else 0.0

    def token_clear(self) -> int:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM token_usage")
            return cur.rowcount

    # ---------- 幂等锁 ----------

    def idem_hit(self, key: str, window: float) -> bool:
        """窗口内已经做过同一件事 → True（不刷新时间戳，避免被无限续期）。

        窗口外重新放行并刷新，所以用户「我改主意了」不会被永久吞掉。
        """
        now = time.time()
        window = max(1.0, float(window))
        with self._lock, self._db:
            row = self._db.execute("SELECT ts FROM tool_idem WHERE key=?", (key,)).fetchone()
            if row and now - float(row["ts"] or 0) < window:
                return True
            self._db.execute(
                "INSERT INTO tool_idem(key, ts) VALUES(?,?)"
                " ON CONFLICT(key) DO UPDATE SET ts=excluded.ts",
                (key, now),
            )
            self._db.execute("DELETE FROM tool_idem WHERE ts < ?", (now - 7 * 86400,))
        return False

    # ---------- 熔断 ----------

    def circuit_state(self, name: str) -> dict:
        with self._lock:
            r = self._db.execute("SELECT * FROM circuit WHERE name=?", (name,)).fetchone()
        if not r:
            return {"name": name, "fails": 0, "opened_at": 0.0,
                    "last_fail": 0.0, "last_ok": 0.0}
        return dict(r)

    def circuit_blocked(self, name: str, cooldown: float) -> tuple[bool, float]:
        """返回 (是否被挡, 剩余秒数)。冷却到期自动放行一次（半开探测）。"""
        st = self.circuit_state(name)
        opened = float(st.get("opened_at") or 0)
        if opened <= 0:
            return False, 0.0
        left = float(cooldown) - (time.time() - opened)
        if left > 0:
            return True, left
        return False, 0.0

    def circuit_fail(self, name: str, threshold: int, cooldown: float) -> dict:
        """记一次失败。达到阈值立刻进冷却（opened_at 置为当前时间）。"""
        now = time.time()
        threshold = max(1, int(threshold))
        with self._lock, self._db:
            r = self._db.execute("SELECT * FROM circuit WHERE name=?", (name,)).fetchone()
            fails = (int(r["fails"]) if r else 0) + 1
            opened = float(r["opened_at"]) if r else 0.0
            last_ok = float(r["last_ok"]) if r else 0.0
            if fails >= threshold:
                opened = now
            self._db.execute(
                "INSERT INTO circuit(name, fails, opened_at, last_fail, last_ok)"
                " VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET"
                " fails=excluded.fails, opened_at=excluded.opened_at,"
                " last_fail=excluded.last_fail",
                (name, fails, opened, now, last_ok),
            )
        return {"name": name, "fails": fails, "opened_at": opened,
                "threshold": threshold, "cooldown": float(cooldown)}

    def circuit_ok(self, name: str) -> None:
        """成功一次就清零并关闭熔断。"""
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO circuit(name, fails, opened_at, last_fail, last_ok)"
                " VALUES(?,0,0,0,?) ON CONFLICT(name) DO UPDATE SET"
                " fails=0, opened_at=0, last_ok=excluded.last_ok",
                (name, now),
            )

    # ---------- 导出 ----------

    def export_all(self) -> dict:
        with self._lock:
            out = {}
            for t in ("l2_facts", "l3_events", "gate_audit", "affinity",
                      "affinity_stages", "social_relations", "relation_evidence",
                      "token_usage"):
                rows = self._db.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
                out[t] = [dict(r) for r in rows]
        return out
