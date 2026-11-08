"""总结编号协议测试：模型选编号，不许写名字。

核心不变式：**只要给了名单，总结器就没有「新建人」的权限**。
模型编一个不存在的号、或者写了名单外的名字，这条就该被丢掉，
人物库里不许因此多出任何一个人。
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")

from sugarmemory_core.store import SugarStore  # noqa: E402
from sugarmemory_core.summarize_core import (  # noqa: E402
    apply_summary, number_transcript, roster_text,
)
from sugarmemory_core.gate import WriteGate  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  x {name}   {detail}")


def fresh(tag):
    db = os.path.join(tempfile.gettempdir(), f"sugarmem_ids_{tag}.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    return SugarStore(db)


def people(store):
    return sorted(p["person"] for p in store.person_all())


# ---------- 1. 编号渲染 ----------
rows = [
    {"event_type": "user_message", "subject": "张三", "content": "[私] 张三: 我弟今年 12 岁"},
    {"event_type": "bot_reply", "subject": "", "content": "[你] 那你得让着他点"},
    {"event_type": "user_message", "subject": "张三", "content": "[私] 张三: 他喜欢打游戏"},
    {"event_type": "user_message", "subject": "阿P", "content": "[群] 阿P: 我也玩"},
    {"event_type": "user_message", "subject": "", "content": "[群] 无名氏: 路过"},
]
lines, roster = number_transcript(rows)
check("同一个人多行同号", lines[0].startswith("[1]") and lines[2].startswith("[1]"), lines)
check("AI 固定 0 号", lines[1].startswith("[0]"), lines[1])
check("第二个人是 2 号", lines[3].startswith("[2]"), lines[3])
check("没归属人的行不给号", lines[4].startswith("[?]"), lines[4])
check("名单只有真人", roster == {1: "张三", 2: "阿P"}, roster)
check("名单渲染带 0=AI", roster_text(roster).startswith("0=AI"), roster_text(roster))

# 日期分隔行（2026-11-08）：模型看不到 created_at，「到时候」会被编成「今晚」。
# 规则：一天一行、插在当天内容之前；没有 created_at 的老数据/测试行不插。
_t0 = 1793968140.0          # 2026-11-06 20:29
_day_rows = [
    {"event_type": "user_message", "subject": "张三", "content": "a", "created_at": _t0},
    {"event_type": "user_message", "subject": "张三", "content": "b", "created_at": _t0 + 60},
    {"event_type": "user_message", "subject": "张三", "content": "c", "created_at": _t0 + 86400},
]
_dl, _ = number_transcript(_day_rows)
check("同一天只插一行日期", sum(1 for x in _dl if x.startswith("—— ")) == 2, _dl)
check("日期行排在当天内容之前", _dl[0].startswith("—— "), _dl)
check("日期行格式", _dl[0] == "—— 2026-11-06 ——", _dl[0])
check("无 created_at 不插日期行", not any(x.startswith("—— ") for x in lines), lines)

# ---------- 2. 按编号落库 ----------
s = fresh("ok")
s.person_ensure("张三", qq="10001", source="event")
s.person_ensure("阿P", qq="10002", source="event")
before = people(s)
stats = apply_summary(s, WriteGate(), {
    "summary": "聊了弟弟和游戏",
    "candidates": [
        {"subject_id": 1, "predicate": "自我相关", "value": "弟弟今年 12 岁",
         "speaker": "user", "speech_act": "statement",
         "source_quote": "我弟今年12岁", "confidence": 0.9},
        {"subject_id": 2, "predicate": "爱好技能", "value": "喜欢打游戏",
         "speaker": "user", "speech_act": "statement",
         "source_quote": "我也玩", "confidence": 0.9},
    ],
    "relations": [{"source_id": 1, "target_id": 2, "relation": "朋友",
                   "quote": "阿P 是我朋友", "confidence": 0.9}],
    "updates": [],
}, subject="张三", session_id="aiocqhttp:FriendMessage:1", roster={1: "张三", 2: "阿P"})

check("候选按编号落到对应的人", stats["l2"] == 2, stats)
check("没有凭空多出一个人", people(s) == before, people(s))
check("弟弟没被建成人", "弟弟" not in people(s), people(s))
check("关系按编号建边", stats["relations"] == 1, stats)
edges = [(e["source"], e["target"]) for e in s.relations_all()]
check("关系两端是名单里的人", edges == [("张三", "阿P")], edges)

# ---------- 3. 编号越界 / 写名单外的名字 ----------
s2 = fresh("bad")
s2.person_ensure("张三", qq="10001", source="event")
before2 = people(s2)
stats = apply_summary(s2, WriteGate(), {
    "summary": "瞎写",
    "candidates": [
        {"subject_id": 9, "predicate": "自我相关", "value": "瞎编的编号",
         "speaker": "user", "speech_act": "statement", "confidence": 0.9},
        {"subject": "弟弟", "predicate": "自我相关", "value": "我弟今年 12 岁",
         "speaker": "user", "speech_act": "statement", "confidence": 0.9},
        {"subject_id": 1, "predicate": "生活习惯", "value": "喜欢熬夜",
         "speaker": "user", "speech_act": "statement", "confidence": 0.9},
    ],
    "relations": [
        {"source_id": 1, "target_id": 7, "relation": "亲友", "confidence": 0.9},
        {"source": "张三", "target": "弟弟", "relation": "亲友", "confidence": 0.9},
    ],
    "updates": [],
}, subject="张三", roster={1: "张三"})

check("越界编号被丢", stats["unknown_subject"] >= 1, stats)
check("名单外的名字被丢", stats["l2"] == 1, stats)
check("仍然没有多出人", people(s2) == before2, people(s2))
check("越界关系不建边", stats["relations"] == 0, stats)
check("关系表是空的", s2.relations_all() == [], s2.relations_all())

# ---------- 4. 模型没按协议、但名字写对了 ----------
s3 = fresh("nameok")
s3.person_ensure("张三", qq="10001", source="event")
before3 = people(s3)
stats = apply_summary(s3, WriteGate(), {
    "summary": "旧格式",
    "candidates": [{"subject": "张三", "predicate": "生活习惯", "value": "喜欢喝无糖可乐",
                    "speaker": "user", "speech_act": "statement", "confidence": 0.9}],
    "relations": [], "updates": [],
}, subject="张三", roster={1: "张三"})
check("写对名字仍可接受", stats["l2"] == 1, stats)
check("接受名字也不造人", people(s3) == before3, people(s3))

# ---------- 5. 不给名单：老路径保持可用 ----------
s4 = fresh("legacy")
s4.person_ensure("张三", qq="10001", source="event")
stats = apply_summary(s4, WriteGate(), {
    "summary": "老调用",
    "candidates": [{"subject": "张三", "predicate": "爱好技能", "value": "喜欢听摇滚",
                    "speaker": "user", "speech_act": "statement", "confidence": 0.9}],
    "relations": [], "updates": [],
}, subject="张三")
check("没有名单时走旧路径", stats["l2"] == 1 and stats["unknown_subject"] == 0, stats)

# ---------- 6. 空名单也是严格模式 ----------
# 以前 strict = bool(roster)，名单恰好为空（整批日志都是 AI 自己的话、或者历史
# 数据没有 subject）时会静默退回老路径，总结器又拿回了建档权。空名单必须更严，
# 不是更松。
s5 = fresh("emptyroster")
s5.person_ensure("张三", qq="10001", source="event")
before5 = people(s5)
stats = apply_summary(s5, WriteGate(), {
    "summary": "整批日志没有主语",
    "candidates": [{"subject": "路人甲", "predicate": "爱好技能", "value": "喜欢篮球",
                    "speaker": "user", "speech_act": "statement", "confidence": 0.9}],
    "relations": [{"source": "张三", "target": "路人甲", "relation": "朋友",
                   "confidence": 0.9}],
    "updates": [],
}, subject="张三", roster={})
check("空名单不建人", people(s5) == before5, people(s5))
check("空名单候选全丢", stats["l2"] == 0 and stats["unknown_subject"] >= 1, stats)
check("空名单不建边", s5.relations_all() == [], s5.relations_all())
check("空名单仍可写会话摘要", stats["summary_stored"] == 1, stats)

# ---------- 7. 没有 QQ 的空壳不写边 ----------
# 面板关系图上的节点是从边推出来的，墙上多一个「没有 QQ 的人」= 多一个点进去
# 什么都没有、又永远对不上号的人。总结写关系时两端只要有一端是**档案里明确
# 空号**，这条边直接丢，只计数；补上 QQ 之后立刻能写。
s6 = fresh("unbound")
s6.person_ensure("张三", qq="10001", source="event")
s6.person_ensure("Tom", source="summary")  # 有档案、没 QQ
orb = {1: "张三", 2: "Tom"}
stats = apply_summary(s6, WriteGate(), {
    "summary": "空号在场",
    "candidates": [],
    "relations": [{"source_id": 1, "target_id": 2, "relation": "认识", "confidence": 0.9}],
    "updates": [],
}, subject="张三", roster=orb)
check("空号不建边", s6.relations_all() == [], s6.relations_all())
check("空号计入 unbound_skipped",
      stats["unbound_skipped"] == 1 and stats["relations"] == 0, stats)

s6.person_set_qq("Tom", "100010001")
stats = apply_summary(s6, WriteGate(), {
    "summary": "补号之后",
    "candidates": [],
    "relations": [{"source_id": 1, "target_id": 2, "relation": "认识", "confidence": 0.9}],
    "updates": [],
}, subject="张三", roster=orb)
check("补上 QQ 就能建边", stats["relations"] == 1 and len(s6.relations_all()) == 1, stats)
check("补号后不再计 unbound", stats["unbound_skipped"] == 0, stats)

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
