"""阶段总结的端到端：长日志切段 + 已有记忆包 + 改口落库。

不联网、不调真模型：把 `_summary_chat` 换成假的，其余全走真代码
（chunk_transcript → merge_summaries → apply_summary → write_l2/supersede）。
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _mk(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ---- astrbot / quart 桩（同 test_distill_e2e，不然 main 导不进来）----
astrbot = _mk("astrbot")
api = _mk("astrbot.api")
event_mod = _mk("astrbot.api.event")
star_mod = _mk("astrbot.api.star")
prov_mod = _mk("astrbot.api.provider")


class _Logger:
    def info(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _Filter:
    class EventMessageType:
        ALL = 0
        FRIEND_MESSAGE = 1
        GROUP_MESSAGE = 2

    def __getattr__(self, name):
        def deco(*a, **k):
            def wrap(f): return f
            return wrap if not (a and callable(a[0])) else a[0]
        return deco


class _Star:
    def __init__(self, *a, **k): pass


def _register(*a, **k):
    def wrap(cls): return cls
    return wrap


event_mod.filter = _Filter()
event_mod.AstrMessageEvent = object
star_mod.Context = object
star_mod.Star = _Star
star_mod.register = _register
prov_mod.ProviderRequest = object
api.logger = _Logger()
api.event = event_mod
api.star = star_mod
api.provider = prov_mod
astrbot.api = api
sys.modules.update({
    "astrbot": astrbot, "astrbot.api": api,
    "astrbot.api.event": event_mod, "astrbot.api.star": star_mod,
    "astrbot.api.provider": prov_mod,
})

from sugarmemory_core.store import SugarStore  # noqa: E402
from sugarmemory_core.gate import WriteGate  # noqa: E402

import main as sm  # noqa: E402

PASS = 0
FAIL = 0
NOW = time.time()
DAY = 86400


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  x {name}   {detail}")


SESS = "default:FriendMessage:1"
CALLS = []


async def run():
    global PASS, FAIL
    db = os.path.join(tempfile.gettempdir(), "sugarmem_sumpack.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    store = SugarStore(db)

    # 一条旧记忆：这次对话要推翻它
    res = store.write_l2(
        subject="小周", predicate="爱好技能", value="喜欢摇滚", speaker="user",
        source_type="user_message", gate_verdict={"decision": "allow"},
        confidence=0.9, source_ts=NOW - 20 * DAY,
    )
    old_id = int(res["id"])

    # 一段长对话：40 条 × 约 290 字 ≈ 11.6k 字符，必然切成多段
    rows = [{
        "event_type": "user_message", "subject": "小周",
        "content": f"[私] 小周: 第{i:02d}条" + "今天聊了很多关于音乐和插件的事情" * 10,
        "session_id": SESS, "created_at": NOW - (40 - i) * 60,
    } for i in range(40)]
    store.write_l3_bulk(rows, expires_at=NOW + DAY)

    inst = sm.SugarMemory.__new__(sm.SugarMemory)
    inst.store = store
    inst.gate = WriteGate()
    inst._summarizing = False
    inst._summary_pending = False
    inst._msg_count = {}
    inst._summary_fail_at = {}
    inst._summary_lock = asyncio.Lock()
    inst._cfg = lambda k, d=None: {
        "summary_window_user": 60, "summary_window_bot": 0,
        "summary_timeout_seconds": 30, "token_stats_enabled": False,
        "alias_autogen": False, "summary_related_l2": True,
        "summary_related_limit": 16, "summary_related_related": True,
    }.get(k, d)
    inst._summary_provider = lambda: object()
    inst._record_tokens = lambda *a, **k: None
    inst._summary_circuit_fail = lambda: None

    async def fake_chat(provider, transcript, timeout, related="", roster=None):
        CALLS.append({"chars": len(transcript), "related": related,
                      "transcript": transcript})
        i = len(CALLS)
        parsed = {
            "summary": f"第{i}段聊了音乐",
            "candidates": [{
                "subject": "小周", "predicate": "经历", "value": f"第{i}段新经历",
                "speaker": "user", "speech_act": "statement", "confidence": 0.9,
            }],
            # 每段都想改同一条旧记忆，看会不会被 by 偏移搞乱
            "updates": [{"id": old_id, "action": "supersede", "by": 0}],
        }
        return json.dumps(parsed, ensure_ascii=False), None

    inst._summary_chat = fake_chat

    ok = await inst._run_summary(SESS, manual=True)
    check("总结跑成功", ok is True)

    # ---------- 切段 ----------
    check("确实切成了多段", len(CALLS) >= 2, str(len(CALLS)))
    check("每段都在预算内", all(c["chars"] <= 6000 for c in CALLS),
          str([c["chars"] for c in CALLS]))
    check("整段日志没丢", sum(c["transcript"].count("第") for c in CALLS) >= 40,
          str([c["transcript"].count("第") for c in CALLS]))

    # ---------- 记忆包 ----------
    check("每段都带了已有记忆", all(c["related"] for c in CALLS))
    check("清单里有旧记忆的编号", all(f"[#{old_id}]" in c["related"] for c in CALLS),
          CALLS[0]["related"][:80])
    check("清单里带原文", "喜欢摇滚" in CALLS[0]["related"], CALLS[0]["related"][:80])
    check("清单里带主体", "小周" in CALLS[0]["related"])

    # ---------- 改口 ----------
    old = store.get_l2(old_id)
    check("旧记忆被归档", old["status"] == "archived", str(old["status"]))
    check("旧记忆指向新事实", int(old["replaced_by"]) > 0, str(old["replaced_by"]))
    check("只改了一次（不是每段都改）",
          sum(1 for c in CALLS) >= 1 and old["superseded_at"] > 0)

    # 候选都落库了（每段一条）
    actives = store.query_active_l2(subject="小周", limit=50)
    values = [r["value"] for r in actives]
    check("每段候选都落了库", len(actives) == len(CALLS), str(values))
    check("被取代的那条不在 active 里", "喜欢摇滚" not in values, str(values))
    check("新事实指向的第一段那条",
          int(old["replaced_by"]) in [int(r["id"]) for r in actives])

    # ---------- 摘要合并 ----------
    sums = store.query_l3(subject="小周", event_type="session_summary", limit=5)
    text = sums[0]["content"] if sums else ""
    check("摘要只有一条", len(sums) == 1, str(len(sums)))
    check("多段摘要接起来了", text.count("段聊了音乐") == len(CALLS), text[:80])

    print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
    return FAIL


rc = asyncio.run(run())
sys.exit(1 if rc else 0)
