"""存储层：只负责存取。约束在存储层强制执行：
- L2 表 CHECK 约束 actionable=0、layer 固定 'L2'；
- L3 表 CHECK 约束 actionable=0、layer 固定 'L3'；
- write_l2 必须携带 allow 裁决，否则抛 PermissionError；
- 不提供绕过门控写 L2 的公开方法；
- L2 与 L3 查询接口完全隔离，绝不 union。
"""
import hashlib
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter

from .social import DEFAULT_STAGES

# 短期记忆的「一条」= 一次对话，不是一条消息。
# 与面板 pages/记忆中枢 的 CONV_GAP_MS 保持一致：30 分钟内、且没切会话算同一段。
CONV_GAP_SECONDS = 30 * 60

# 多值分类：同一个人名下允许多条（不同的事/爱好/经历）。与 summarize_core 的 _MULTI_VALUED 保持一致（7 类）。
_MULTI_VALUED_PREDICATES = {"爱好技能", "生活习惯", "经历", "打算", "评价认为", "自我相关", "其他"}


def is_multi_valued(predicate: str) -> bool:
    return (predicate or "").strip() in _MULTI_VALUED_PREDICATES
# 阶段总结不单独成段，挂到时间最近的对话上（15 分钟内允许尾部对齐，1 天内有效）。
SUMMARY_ATTACH_AFTER = 300
SUMMARY_ATTACH_WINDOW = 86400

# 人名里那些看不见的装饰：零宽字符、方向控制符、BOM
# 再加上韩文填充符（U+115F/U+1160/U+3164/U+FFA0）——QQ 昵称里常塞这个占位，
# 「小豆喵」和「小豆ᅠ⁧喵ᅠ」其实是同一个人，肉眼根本看不出区别。
_ZW_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u115f\u1160\u3164\uffa0]")
# 名字里的备注括号：「张三（摸鱼）」「张三(老板)」「张三【群友】」
_BRACKET_RE = re.compile(r"[\(（\[\【\{｛][^\)）\]\】\}｝]{0,24}[\)）\]\】\}｝]")


def alias_key(name: str) -> str:
    """人名归一键：同一个人被写成几种样子时，这个键相同。

    只做「无损」归并——去零宽字符、全半角统一、去掉备注括号、压空白、忽略大小写。
    不做包含匹配（「小明」和「小明妈妈」是两个人），那种交给面板手动合并。
    """
    s = _ZW_RE.sub("", str(name or ""))
    s = unicodedata.normalize("NFKC", s)
    stripped = _BRACKET_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", stripped or s).strip()
    return s.casefold()


def pinyin_key(name: str) -> str:
    """读音归一键：「徐老鸭」和「小红」得到同一个键 小红。

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
  updated_at REAL NOT NULL
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
  created_at REAL NOT NULL
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
  sort INTEGER NOT NULL DEFAULT 0
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
  total INTEGER NOT NULL DEFAULT 0
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

-- 人物身份：一个人 = 一个 QQ + 一串称呼。person 是显示用的规范名，
-- QQ 才是身份锚（同一个人叫「徐老鸭」还是「小红」都是他）。
CREATE TABLE IF NOT EXISTS people (
  person TEXT PRIMARY KEY,
  qq TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

-- 称呼 → 人物。alias_key 是字面归一键（唯一），pinyin 是读音键（只用于提示）。
CREATE TABLE IF NOT EXISTS person_alias (
  alias_key TEXT PRIMARY KEY,
  person TEXT NOT NULL,
  raw TEXT NOT NULL DEFAULT '',
  pinyin TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'auto',
  hits INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_palias_person ON person_alias(person);
CREATE INDEX IF NOT EXISTS idx_palias_py ON person_alias(pinyin);
"""


def _fp(*parts: str) -> str:
    return hashlib.sha256("|".join(p.strip().lower() for p in parts).encode("utf-8")).hexdigest()[:32]


# 「同一件事、换了描述」在字面上最常见的三类差异，写指纹前先抹平：
#   1. 全角/半角数字与字母（５月 vs 5月）
#   2. 日期写法（5月20日 / 5/20 / 05-20 / 5.20）
#   3. 空白与标点（生日是 5 月 20 日 / 生日是5月20日）
# 这样 value 归一化后，「小明生日5月20日」和「小明生日 5/20」能落到同一个指纹。
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
    # 小红 就是 徐老鸭 / 其实就是 / 原名是 / 本名是
    re.compile(
        rf"(?:^|(?<={_BOUND}))(?P<a>{_LABEL}{{1,20}}?)"
        rf"(?:就是|也就是|其实就是|其实是|本来是|原来是|原名是|本名是)"
        rf"(?P<b>{_LABEL}{{1,12}}?)(?={_BOUND}|$)"
    ),
    # 徐老鸭是 小红 的外号 / 绰号 / 小名 / 昵称 / 别名
    re.compile(
        rf"(?P<b>{_LABEL}{{1,20}}?)(?:(?:就)?是|也就是)(?P<a>{_LABEL}{{1,12}}?)"
        rf"的(?:外号|绰号|小名|昵称|别名|代号|英文名|真名|本名|另一个名字)"
    ),
    # 徐老鸭 也叫 / 又叫 / 又称 / 别名 / 外号叫 小红
    re.compile(
        rf"(?:^|(?<={_BOUND}))(?P<a>{_LABEL}{{1,20}}?)"
        rf"(?:也叫|又叫|也称|又称|也叫做|又叫做|别名是|别名|外号叫|外号是|绰号叫|绰号是)"
        rf"(?P<b>{_LABEL}{{1,12}}?)(?={_BOUND}|$)"
    ),
    # 小红 = 徐老鸭
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


# 句首的引语词：「她说 小红 就是滚木」里的「她说」不是名字的一部分。
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
    """把「徐老鸭我觉得他其实…」这种粘在后面的从句切掉。"""
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

    直接拿新值顶掉旧值会丢信息 —— 「修好了 ncm 点歌插件」被「改过菜单」顶掉就没了。
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
                "INSERT INTO affinity_stages (name, threshold, sort) VALUES (?,?,?)",
                (name, float(th), i),
            )

    def _migrate(self):
        """老库补列 + 回填归属/会话，并清掉历史内容里的 [MSG_ID:...] 尾巴。"""
        l3_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(l3_events)")}
        for col, ddl in (
            ("subject", "TEXT NOT NULL DEFAULT ''"),
            ("session_id", "TEXT NOT NULL DEFAULT ''"),
            ("summary_id", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if col not in l3_cols:
                self._db.execute(f"ALTER TABLE l3_events ADD COLUMN {col} {ddl}")
        l2_cols = {r["name"] for r in self._db.execute("PRAGMA table_info(l2_facts)")}
        if "source_summary_id" not in l2_cols:
            self._db.execute(
                "ALTER TABLE l2_facts ADD COLUMN source_summary_id INTEGER NOT NULL DEFAULT 0"
            )

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
                 single_valued: bool = True) -> dict:
        """受控写入 L2。必须携带门控 allow 裁决；存储层强制 actionable=0。

        single_valued=True（默认）时同一个 (subject, predicate) 只保留一条 active，
        新值就地合并进旧记录；predicate 为「其他」这类垃圾桶时传 False，允许多条。
        """
        if not gate_verdict or gate_verdict.get("decision") != "allow":
            raise PermissionError(f"L2 写入被拒绝：缺少门控 allow 裁决（got {gate_verdict}）")
        if speaker == "bot" or source_type == "bot_utterance":
            raise PermissionError("L2 存储层硬约束：Bot 自述/来源不得写入 L2")
        if speech_act in ("quote", "roleplay", "hypothesis", "temp_state", "plan"):
            raise PermissionError(f"L2 存储层硬约束：{speech_act} 只能降级 L3，不得写入 L2")
        now = now or time.time()
        fp = _fp_norm(subject, predicate, value)
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT id, stability, confidence FROM l2_facts WHERE fingerprint=? AND status='active'",
                (fp,),
            ).fetchone()
            if row:
                self._db.execute(
                    "UPDATE l2_facts SET stability=stability+1, confidence=MAX(confidence,?),"
                    " updated_at=? WHERE id=?",
                    (float(confidence), now, row["id"]),
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
                            " confidence=MAX(confidence,?), updated_at=? WHERE id=?",
                            (float(confidence), now, d["id"]),
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
                    "SELECT id, value FROM l2_facts"
                    " WHERE subject=? AND predicate=? AND status='active'"
                    " ORDER BY updated_at DESC LIMIT 1",
                    (subject, predicate),
                ).fetchone()
            if conflict:
                # 新旧 value 语义相同（归一化后一致）就不再拼，直接当重复稳度+1，
                # 避免「5月20日；5/20」这种换描述的重复串。
                if _norm_fingerprint(conflict["value"]) == val_fp:
                    self._db.execute(
                        "UPDATE l2_facts SET stability=stability+1,"
                        " confidence=MAX(confidence,?), source_msg_id=?, source_quote=?,"
                        " source_summary_id=?, updated_at=? WHERE id=?",
                        (float(confidence), source_msg_id, source_quote,
                         int(source_summary_id or 0), now, conflict["id"]),
                    )
                    return {"action": "merged", "id": conflict["id"],
                            "predicate": predicate, "same_value": True}
                merged = _merge_value(conflict["value"], value)
                self._db.execute(
                    "UPDATE l2_facts SET value=?, fingerprint=?, stability=stability+1,"
                    " confidence=MAX(confidence,?), source_msg_id=?, source_quote=?,"
                    " source_summary_id=?, gate_reason=?, updated_at=? WHERE id=?",
                    (merged, _fp_norm(subject, predicate, merged), float(confidence),
                     source_msg_id, source_quote, int(source_summary_id or 0),
                     gate_reason or "用户显式陈述", now, conflict["id"]),
                )
                return {"action": "merged", "id": conflict["id"],
                        "predicate": predicate, "merged_value": merged}
            reason = gate_reason or "用户显式陈述"
            cur = self._db.execute(
                "INSERT INTO l2_facts (subject, predicate, value, speaker, speech_act, source_type,"
                " source_msg_id, source_quote, source_summary_id, actionable, confidence, status,"
                " stability, expires_at, replaced_by, gate_reason, fingerprint, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,?,'active',0,?,0,?,?,?,?)",
                (subject, predicate, value, speaker, speech_act, source_type,
                 source_msg_id, source_quote, int(source_summary_id or 0),
                 float(confidence), expires_at, reason, fp, now, now),
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

        A 问「小红 是不是说过他叫滚木」时，只注入 A 自己的记忆是没用的 ——
        答案在 B 的档案里。名字直接做子串匹配；单字名不参与，太容易误命中。
        """
        t = str(text or "").lower()
        if not t:
            return []
        ex = str(exclude or "").strip()
        names = [n for n in self.list_subjects() if n != ex and len(n) >= 2]
        hits = [n for n in names if n.lower() in t]
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
                " replaced_by, gate_reason, fingerprint, created_at, updated_at)"
                " VALUES (?,?,?,'user','statement','manual','','',0,?,'active',0,0,0,?,?,?,?)",
                (subject, predicate, value, max(0.0, min(1.0, float(confidence))),
                 reason, fp, now, now),
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
                 expires_at: float = 0.0, now: float | None = None) -> int:
        now = now or time.time()
        # 主体归一：同一个人的不同写法（小刚 / 蔗糖）在写入时就归到 canonical 名下，
        # 否则 person_alias 里明明有映射，实时事件流却用原始称呼落库，同一个人又裂成两个。
        # 归一认不出（群名、数字 id、空串）就保留原值，绝不把特殊 subject 也硬并成一个人。
        subject = (subject or "").strip()[:40]
        if subject:
            resolved = self.person_resolve(subject)
            if resolved:
                subject = resolved
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO l3_events (event_type, subject, session_id, summary_id, content,"
                " source_msg_id, actionable, expires_at, created_at) VALUES (?,?,?,?,?,?,0,?,?)",
                (event_type, subject, (session_id or "")[:120], int(summary_id or 0),
                 content, source_msg_id, expires_at, now),
            )
            return int(cur.lastrowid)

    def query_l3(self, *, subject: str | None = None, event_type: str | None = None,
                 limit: int = 200) -> list[dict]:
        """按主体/类型查询 L3。subject=None 表示不限，subject="" 表示未归类。
        已到期的短期记录在这里就被过滤掉，不必等物理清理。"""
        now = time.time()
        q = "SELECT * FROM l3_events WHERE (expires_at = 0 OR expires_at > ?)"
        args: list = [now]
        if subject is not None:
            q += " AND subject=?"
            args.append(subject)
        if event_type:
            q += " AND event_type=?"
            args.append(event_type)
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

        场景：小红 和 bot 在私聊里聊到蔗糖，但字面上只说「她」「被子」，
        没有「蔗糖」两个字。AI 拿到这段跨会话消息时无法归因「这是在说我」。
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
                "SELECT DISTINCT session_id FROM l3_events"
                " WHERE event_type IN ('user_message','bot_reply')"
                " AND session_id<>'' AND instr(content, ?) > 0"
                " AND (expires_at=0 OR expires_at>?)"
                " ORDER BY id DESC LIMIT ?",
                (name, time.time(), int(limit)),
            ).fetchall()
        return {str(r[0]) for r in rows if r[0]}

    def query_recent_l3(self, limit: int = 40, event_type: str | None = None,
                        session_id: str | None = None,
                        subject: str | None = None) -> list[dict]:
        """取最近 L3 事件。session_id / subject 为空表示不限（面板/手动总结用）。"""
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
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def recent_nick_by_session(self, max_sessions: int = 400) -> dict[str, str]:
        """号码会话 → 对方最近用过的昵称。

        记忆块里全是「用户 10001」这种冷号码，模型对不上「眼前这位是谁」
        也不敢把号码和别的会话里的人认成同一个（实测 AI 收到完整转述记录仍去
        调工具现查）。聊天记录本身就带答案：每条 user_message 都有 session_id
        （号码）和 subject（说话时的昵称），GROUP BY 取每会话最新一条的昵称即可。
        人改名会自然跟随（取最新），不用额外建表维护。
        """
        sql = (
            "SELECT session_id, subject, MAX(id) AS mid FROM l3_events "
            "WHERE event_type='user_message' AND subject != '' "
            "GROUP BY session_id ORDER BY mid DESC LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(sql, (int(max_sessions),)).fetchall()
        return {str(r["session_id"]): str(r["subject"]) for r in rows if r["session_id"]}

    def cleanup_expired(self, now: float | None = None) -> int:
        now = now or time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "DELETE FROM l3_events WHERE expires_at > 0 AND expires_at <= ?", (now,)
            )
            cur2 = self._db.execute(
                "DELETE FROM l2_facts WHERE expires_at > 0 AND expires_at <= ? AND status='active'",
                (now,),
            )
            return cur.rowcount + cur2.rowcount

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
                "SELECT id, name, threshold, sort FROM affinity_stages ORDER BY threshold, sort"
            ).fetchall()
        return [dict(r) for r in rows]

    def affinity_stages_save(self, stages: list[dict]) -> list[dict]:
        """全量替换阶段表。接受 [{name, threshold}]，自动按阈值排序。"""
        clean: list[tuple[str, float]] = []
        for s in stages or []:
            name = str((s or {}).get("name") or "").strip()[:20]
            if not name:
                continue
            try:
                th = float((s or {}).get("threshold") or 0)
            except (TypeError, ValueError):
                th = 0.0
            clean.append((name, max(0.0, th)))
        if not clean:
            return self.affinity_stages()
        clean.sort(key=lambda x: x[1])
        with self._lock, self._db:
            self._db.execute("DELETE FROM affinity_stages")
            for i, (name, th) in enumerate(clean):
                self._db.execute(
                    "INSERT INTO affinity_stages (name, threshold, sort) VALUES (?,?,?)",
                    (name, th, i),
                )
        return self.affinity_stages()

    def affinity_stages_reset(self) -> list[dict]:
        with self._lock, self._db:
            self._db.execute("DELETE FROM affinity_stages")
            for i, (name, th) in enumerate(DEFAULT_STAGES):
                self._db.execute(
                    "INSERT INTO affinity_stages (name, threshold, sort) VALUES (?,?,?)",
                    (name, float(th), i),
                )
        return self.affinity_stages()

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
                "SELECT person, qq, note, created_at, updated_at FROM people"
            ).fetchall()
            aliases = self._db.execute(
                "SELECT alias_key, person, raw, pinyin, source, hits, updated_at"
                " FROM person_alias ORDER BY hits DESC, updated_at ASC"
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
                "SELECT person, qq, note, created_at, updated_at FROM people WHERE person=?",
                (person,),
            ).fetchone()
            if not row:
                return None
            aliases = self._db.execute(
                "SELECT alias_key, person, raw, pinyin, source, hits FROM person_alias"
                " WHERE person=? ORDER BY hits DESC, updated_at ASC", (person,),
            ).fetchall()
        d = dict(row)
        d["aliases"] = [dict(a) for a in aliases]
        return d

    def person_resolve(self, name: str) -> str:
        """一个称呼对应哪个人。只认字面（alias_key），认不出返回空串。

        读音相近的（徐老鸭 / 小红）在这里**故意不自动认**——同音不同人
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

        「用户说小红就是徐老鸭」抓出来的左半边是「用户说小红」，
        整串当然没人认识；砍到「小红」就认出来了。
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

    def person_ensure(self, name: str, qq: str = "", source: str = "auto") -> str:
        """确保这个称呼有主：没有就建一个以它命名的人物，并登记这条称呼。"""
        name = (name or "").strip()[:40]
        if not name:
            return ""
        person = self.person_resolve(name) or name
        now = time.time()
        key = alias_key(name)
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO people (person, qq, note, created_at, updated_at)"
                " VALUES (?,?,?,?,?) ON CONFLICT(person) DO NOTHING",
                (person, qq, "", now, now),
            )
            if qq:
                # 已有 QQ 就保留（先到先得），空着才补
                self._db.execute(
                    "UPDATE people SET qq=?, updated_at=? WHERE person=? AND qq=''",
                    (qq, now, person),
                )
            if key:
                self._db.execute(
                    "INSERT INTO person_alias (alias_key, person, raw, pinyin, source,"
                    " hits, created_at, updated_at) VALUES (?,?,?,?,?,1,?,?)"
                    " ON CONFLICT(alias_key) DO UPDATE SET hits=hits+1, updated_at=excluded.updated_at",
                    (key, person, name, pinyin_key(name), source, now, now),
                )
        return person

    def person_bind(self, raw: str, person: str, qq: str = "",
                    source: str = "manual", force: bool = False) -> dict:
        """把一个称呼绑到某个人身上。默认遇到「已经属于别人」的称呼就拒绝，
        force=True 才是用户明确要改绑（面板里点确认时传）。"""
        raw = (raw or "").strip()[:40]
        person = (person or "").strip()[:40]
        if not raw or not person:
            return {"ok": False, "reason": "invalid"}
        key = alias_key(raw)
        if not key:
            return {"ok": False, "reason": "invalid"}
        owner = self.person_resolve(raw)
        if owner and owner != person and not force:
            return {"ok": False, "reason": "taken", "owner": owner}
        canonical = self.person_ensure(person, qq=qq, source=source)
        now = time.time()
        qq = re.sub(r"\D", "", str(qq or ""))[:20]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO person_alias (alias_key, person, raw, pinyin, source,"
                " hits, created_at, updated_at) VALUES (?,?,?,?,?,1,?,?)"
                " ON CONFLICT(alias_key) DO UPDATE SET person=excluded.person,"
                " raw=excluded.raw, source=excluded.source, updated_at=excluded.updated_at",
                (key, canonical, raw, pinyin_key(raw), source, now, now),
            )
            if key != alias_key(canonical):
                pass
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

    def person_merge(self, keep: str, drop: str) -> dict:
        """把 drop 这个人整个并进 keep：称呼、好感度、关系边、记忆归属都搬过去。"""
        keep = (keep or "").strip()[:40]
        drop = (drop or "").strip()[:40]
        if not keep or not drop or keep == drop:
            return {"ok": False, "reason": "invalid"}
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
        return {"ok": True, "keep": keep, "drop": drop, **moved}

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

        实机踩过的坑（2026-10-28）：用户说「小红 就是徐老鸭」，总结把
        徐老鸭当成新人物写进关系图，于是同一个人在图上裂成两个节点，
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
        """私聊会话串里带着对方的 QQ：`111:FriendMessage:10001` → 10001。

        群聊/临时会话一律返回空：群号不是 QQ，套到人身上比没有更糟。
        """
        sid = str(session_id or "")
        parts = sid.split(":")
        if len(parts) < 3 or "FriendMessage" not in parts[-2]:
            return ""
        qq = re.sub(r"\D", "", parts[-1])[:20]
        return qq if 5 <= len(qq) <= 12 else ""

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
            self.person_ensure(name, qq=qq, source="bootstrap")
            created += 0 if before else 1
            if qq:
                cur = self.person_get(self.person_resolve(name) or name) or {}
                qq_bound += 1 if (cur.get("qq") or "") == qq else 0
        # 只有好感度或只有关系、没过 L3 的名字，也得登记
        for name in self._person_nodes():
            qq = ""
            if not self.person_resolve(name):
                self.person_ensure(name, source="bootstrap")
                created += 1
        learned = self.person_learn_from_evidence() if learn else []
        folded = self.person_fold_aliases()
        return {"created": created, "qq_bound": qq_bound,
                "learned": learned, "folded": folded}

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
        # 这句话里可能就写着答案：「小红就是徐老鸭」。先学着，下面解析名字
        # 时两边才能落到同一个人身上，不然同一个人会在图上裂成两个节点。
        if quote:
            try:
                self.person_learn_aliases(quote)
            except Exception:
                pass
        # 先各自归并，再拿归并结果互相比对：归并后撞成同一个人说明这层关系无从记起
        raw_src, raw_dst = source, target
        # 身份优先：认得出是谁就落到那个人身上（徐老鸭 → 小红），
        # 认不出再退回字面归并（「张三」与「张三（群友）」）。
        cs = self.person_resolve(source) or self.canonical_name(source)
        ct = self.person_resolve(target) or self.canonical_name(target)
        if cs and ct:
            source, target = cs, ct
        if source == target:
            return {"ok": False, "reason": "same_person", "source": source, "target": target}
        # 新出现的人也要有身份记录，不然面板上他就是「没名字的路人」
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

        启动时跑一遍，用来清历史脏数据：老版本不认别名，AI 写「小刚（蔗糖）」、
        面板写「小刚」，库里就有两行、图上两个节点，看着像两条一样的单向关系。
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
                     ts: float | None = None) -> int:
        """记一条用量。source=actual 是 provider 报的真实 usage，estimate 是我们估的。"""
        total = int(total if total is not None else (input + cached + output))
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO token_usage (ts, module, source, provider_id, subject, session_id,"
                " input, cached, output, total) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (float(ts or time.time()), str(module or "other")[:32], str(source or "actual")[:16],
                 str(provider_id or "")[:64], str(subject or "")[:40], str(session_id or "")[:120],
                 int(input or 0), int(cached or 0), int(output or 0), max(0, total)),
            )
            return int(cur.lastrowid)

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
