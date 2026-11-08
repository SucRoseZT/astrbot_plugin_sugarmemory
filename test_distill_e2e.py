# -*- coding: utf-8 -*-
"""补录提炼的端到端测试：假 provider 跑真 _run_distill，验证按天分块、水位、落 L2。

不连真模型。provider 只是占位，_summary_chat 被替换成按段返回固定 JSON，
所以这里测的是「取数范围 / 按天倒序 / 段内水位 / 结果落库」这条链路，
不是模型质量。
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _mk(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ---- astrbot / quart 桩 ----
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

import main as sm  # noqa: E402
from sugarmemory_core.gate import WriteGate  # noqa: E402
from sugarmemory_core.store import SugarStore  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


BLOCKS = []
SEQ = [0]
DAY = 86400


def fake_summary_json(transcript: str) -> str:
    """每段产出 1 条事实，value 里带段序号，方便核对段数。"""
    BLOCKS.append(transcript)
    SEQ[0] += 1
    idx = SEQ[0]
    return json.dumps({
        "summary": f"第 {idx} 段补录",
        "candidates": [{
            "subject": "鱼", "predicate": "生日", "value": f"第{idx}段提炼出的事实",
            "speaker": "user", "speech_act": "statement",
            "source_quote": "原话", "confidence": 0.9,
        }],
        "relations": [],
    }, ensure_ascii=False)


def mk_rows(tag: str, n: int, base_ts: float, sess: str) -> list:
    return [{
        "event_type": "user_message", "subject": "鱼",
        "content": f"[私] 鱼: {tag}-{i} 我喜欢这个",
        "session_id": sess, "created_at": base_ts + i,
    } for i in range(n)]


async def main():
    db = os.path.join(tempfile.gettempdir(), "sugarmem_distill_e2e.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    store = SugarStore(db)
    now = time.time()

    # 三天数据，标签区分：T=今天 Y=昨天 E=前天
    store.write_l3_bulk(mk_rows("E", 20, now - 2 * DAY, "import:鱼:1"), expires_at=now + 86400)
    store.write_l3_bulk(mk_rows("Y", 30, now - 1 * DAY, "import:鱼:1"), expires_at=now + 86400)
    store.write_l3_bulk(mk_rows("T", 10, now, "import:鱼:1"), expires_at=now + 86400)
    # 实时会话的干扰数据，不该被补录提炼碰到
    store.write_l3_bulk([{
        "event_type": "user_message", "subject": "鱼", "content": "[私] 鱼: 实时这句",
        "session_id": "default:FriendMessage:1", "created_at": now,
    }], expires_at=now + 86400)

    inst = sm.SugarMemory.__new__(sm.SugarMemory)
    inst.store = store
    inst.gate = WriteGate()
    inst._summary_lock = asyncio.Lock()
    inst._distilling = False
    inst._distill_stop = False
    inst._distill_state = {}
    inst._cfg = lambda k, d=None: {
        "summary_window_user": 60, "summary_timeout_seconds": 30,
        "token_stats_enabled": False, "alias_autogen": False,
    }.get(k, d)
    inst._summary_provider = lambda: object()
    inst._record_tokens = lambda *a, **k: None

    async def fake_chat(provider, transcript, timeout, related="", roster=None):
        return fake_summary_json(transcript), None

    inst._summary_chat = fake_chat

    # ---------- 1. 按天倒序 ----------
    prog = store.distill_progress("鱼")
    check("开工前待提炼 60 条", prog["left"] == 60, str(prog["left"]))
    check("分成 3 天", prog["days"] == 3, str(prog["days"]))
    check("最近的一天排第一个", prog["current_day"] ==
          store.distill_days("鱼")[0]["day"], prog["current_day"])

    await inst._run_distill("鱼", 20)
    st = inst._distill_state
    check("没有报错", not st.get("error"), str(st.get("error")))
    check("跑完标记 done", st.get("done") is True)
    # 10 + ceil(30/20)=2 + ceil(20/20)=1 = 4 段
    check("分 4 段", st.get("done_blocks") == 4, str(st.get("done_blocks")))
    check("处理 60 条", st.get("items") == 60, str(st.get("items")))
    check("调用次数等于段数", len(BLOCKS) == 4, str(len(BLOCKS)))
    # 0.16.12 起 transcript 每换一天会插一行「—— 2026-11-06 ——」，
    # 数消息条数时要把分隔行排除，否则永远差 1 条。
    def _msgs(block: str) -> list[str]:
        return [ln for ln in block.splitlines()
                if not (ln.startswith("—— ") and ln.endswith(" ——"))]

    check("第 1 段是今天那 10 条", len(_msgs(BLOCKS[0])) == 10 and "T-0" in BLOCKS[0],
          str(len(_msgs(BLOCKS[0]))))
    check("第 1 段没混进昨天的话", "Y-0" not in BLOCKS[0])
    check("第 2 段是昨天前 20 条", "Y-0" in BLOCKS[1] and "Y-19" in BLOCKS[1]
          and "Y-20" not in BLOCKS[1])
    check("第 3 段接昨天的后 10 条", "Y-20" in BLOCKS[2] and "Y-29" in BLOCKS[2])
    check("第 4 段才是前天", "E-0" in BLOCKS[3])
    check("切点不跨天：每段只含一个前缀", all(
        sum(1 for t in ("T-", "Y-", "E-") if t in b) == 1 for b in BLOCKS))

    prog = store.distill_progress("鱼")
    check("没有剩余待提炼", prog["left"] == 0, str(prog["left"]))
    check("三天都标完成", prog["done_days"] == 3, str(prog["done_days"]))
    check("累计段数 4", store.distill_cursor("鱼")["blocks"] == 4,
          str(store.distill_cursor("鱼")["blocks"]))

    l2 = store.query_active_l2(subject="鱼", limit=100)
    values = [r["value"] for r in l2]
    check("L2 落库", len(l2) >= 1, str(len(l2)))
    check("四段的话都留着（多值分类不互相挤）", len(values) == 4, str(values))
    check("没有旧话被删", st.get("stale") == 0, str(st.get("stale")))
    check("实时消息没被提炼", not any("实时这句" in v for v in values))

    # ---------- 2. 新数据只补跑新的那段 ----------
    rows2 = mk_rows("N", 5, now + 100, "import:鱼:2")
    store.write_l3_bulk(rows2, expires_at=now + 86400)
    check("新批次待提炼 5 条", store.distill_progress("鱼")["left"] == 5,
          str(store.distill_progress("鱼")["left"]))
    BLOCKS.clear()
    await inst._run_distill("鱼", 20)
    check("只补跑 1 段", len(BLOCKS) == 1, str(len(BLOCKS)))
    check("累计段数 5", store.distill_cursor("鱼")["blocks"] == 5,
          str(store.distill_cursor("鱼")["blocks"]))
    vals2 = [r["value"] for r in store.query_active_l2(subject="鱼", limit=100)]
    check("新批次也写进去了", any("第5段" in v for v in vals2), str(vals2))

    # 多值分类（爱好/经历/打算）本来就该并存，旧的那条不能被判成「过时」删掉
    store.write_l3_bulk([
        {"event_type": "user_message", "subject": "小周",
         "content": "[私] 小周: 我现在喜欢摇滚", "session_id": "import:小周:1",
         "created_at": now},
        {"event_type": "user_message", "subject": "小周",
         "content": "[私] 小周: 我昨天还在听篮球播客", "session_id": "import:小周:1",
         "created_at": now - DAY},
    ], expires_at=now + 86400)

    async def fake_hobby(provider, transcript, timeout, related="", roster=None):
        kind = "摇滚" if "摇滚" in transcript else "篮球"
        return json.dumps({"summary": "", "candidates": [{
            "subject": "小周", "predicate": "爱好技能", "value": f"喜欢{kind}",
            "speaker": "user", "speech_act": "statement", "confidence": 0.9,
        }], "relations": []}, ensure_ascii=False), None

    inst._summary_chat = fake_hobby
    await inst._run_distill("小周", 20)
    hobbys = [r["value"] for r in store.query_active_l2(subject="小周", limit=50)]
    check("多值属性两天的话都留着", len(hobbys) == 2, str(hobbys))
    inst._summary_chat = fake_chat

    # ---------- 3. 旧话不能覆盖新记忆 ----------
    store.write_l2(
        subject="鱼", predicate="生日", value="现在喜欢摇滚", speaker="user",
        source_type="user_message", gate_verdict={"decision": "allow"},
    )
    store.write_l3_bulk([{
        "event_type": "user_message", "subject": "鱼",
        "content": "[私] 鱼: 我以前喜欢重金属", "session_id": "import:鱼:old",
        "created_at": now - 90 * DAY,
    }], expires_at=now + 86400)

    async def fake_old(provider, transcript, timeout, related="", roster=None):
        return json.dumps({
            "summary": "旧事", "candidates": [{
                "subject": "鱼", "predicate": "生日", "value": "以前喜欢重金属",
                "speaker": "user", "speech_act": "statement", "confidence": 0.9,
            }], "relations": [],
        }, ensure_ascii=False), None

    inst._summary_chat = fake_old
    await inst._run_distill("鱼", 20)
    old_rows = [r for r in store.query_active_l2(subject="鱼", limit=100)
                if "重金属" in str(r["value"])]
    check("90 天前的话照样留着", len(old_rows) == 1, str(len(old_rows)))
    check("但 created_at 记的是原话时间", old_rows
          and abs(float(old_rows[0]["created_at"]) - (now - 90 * DAY)) < 5,
          str(round(float(old_rows[0]["created_at"]) - now, 0)) if old_rows else "")
    check("没有旧话被删", inst._distill_state.get("stale") == 0,
          str(inst._distill_state.get("stale")))

    # ---------- 4. 中途停止：跑一段就停，剩的留着 ----------
    inst._summary_chat = fake_chat
    store.write_l3_bulk(mk_rows("Z", 40, now + 200, "import:鱼:3"), expires_at=now + 86400)
    BLOCKS.clear()
    orig = inst._run_distill

    async def stop_after_first(subject, block):
        task = asyncio.create_task(orig(subject, block))
        for _ in range(400):
            if inst._distill_state.get("running"):
                break
            await asyncio.sleep(0.005)
        for _ in range(600):
            if inst._distill_state.get("done_blocks", 0) >= 1:
                break
            await asyncio.sleep(0.005)
        inst._distill_stop = True
        await task

    await stop_after_first("鱼", 20)
    check("停止后只处理一段", inst._distill_state.get("done_blocks") == 1,
          str(inst._distill_state.get("done_blocks")))
    check("停止不标记完成", inst._distill_state.get("done") is False)
    n_left = store.distill_progress("鱼")["left"]
    check("停止后还剩 20 条", n_left == 20, str(n_left))

    # ---------- 5. 一天里句子特别长时按字符预算再切 ----------
    big_ts = now + 400
    store.write_l3_bulk([{
        "event_type": "user_message", "subject": "阿明",
        "content": f"[私] 阿明: 长-{i:02d} " + ("啊" * 280),
        "session_id": "import:阿明:big", "created_at": big_ts + i,
    } for i in range(40)], expires_at=now + 86400)
    c1 = store.distill_chunk("阿明", block=120, max_chars=5500)
    n1 = len(c1["rows"]) if c1 else 0
    check("40 条按字符预算切成小段（不是全塞）", 0 < n1 < 40, str(n1))
    check("字符数没超过预算", c1["chars"] <= 5500, str(c1["chars"]))
    check("切的段仍在同一天内、当天还有剩", c1["more_in_day"] is True)
    store.distill_mark("阿明", c1["day"], int(c1["rows"][-1]["id"]), n1)
    c2 = store.distill_chunk("阿明", block=120, max_chars=5500)
    check("下段接着同一天往下走", c2["day"] == c1["day"]
          and f"长-{n1:02d}" in c2["rows"][0]["content"], c2["rows"][0]["content"][:24])

    store.close()
    for p in (db, db + "-wal", db + "-shm"):
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"\n===== 结果：{len(PASS)} 过 / {len(FAIL)} 挂 =====")
    if FAIL:
        print("失败项：", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
