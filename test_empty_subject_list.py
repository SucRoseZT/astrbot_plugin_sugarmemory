"""记忆列表必须把「还没有记忆的人」也列出来。

实机背景（2026-11-04）：用户第一眼看的就是记忆列表。人物库里明明有这个人，
但一条记忆都没有的时候，列表里直接不出现——用的人会以为记忆被清了、插件坏了。
所以 `/api/subjects` 现在会把 people 表里的人一并送上来，标 `empty=true`，
前端显示「还没有记忆 · 等 Ta 开口」，而不是「0 条」。

不变式：
1. 有记忆的人和没记忆的人都在同一个列表里；
2. 没记忆的那条 `empty=true`、计数为 0，不会因为排序被有记忆的人顶没了；
3. 群名不算人，不能混进来。
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 插件主模块依赖 AstrBot 运行时；只能 append，insert(0) 会让 core/main.py 抢占名字
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from sugarmemory_core.store import SugarStore  # noqa: E402

try:
    import main as sm  # noqa: E402
except Exception as exc:
    print(f"SKIP: 无法导入插件主模块（{exc}）")
    sys.exit(0)

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
    db = os.path.join(tempfile.gettempdir(), f"sugarmem_subj_{tag}.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    return SugarStore(db)


def plugin(store):
    p = object.__new__(sm.SugarMemory)
    p.store = store
    return p


store = fresh("list")
# 两个人有记忆，一个人只有档案（从没说过话/一句话都没攒下来）
store.person_ensure("张三", qq="100010004", source="event")
store.person_ensure("王五", qq="100010005", source="event")
store.person_ensure("Tom", qq="100010001", source="manual")

store.write_l3(event_type="user_message", subject="张三",
               content="[私] Moon: 今天读书", session_id="s1")
store.write_l3(event_type="user_message", subject="王五",
               content="[群] 王五: 在", session_id="s2")

p = plugin(store)
resp = asyncio.run(p._api_subjects())
items = resp["data"]["items"]
subs = {x["subject"]: x for x in items}

check("三个人都在列表里", set(subs) == {"张三", "王五", "Tom"}, list(subs))
check("没记忆的人也出现", "Tom" in subs, list(subs))
check("没记忆的标了 empty", subs.get("Tom", {}).get("empty") is True, subs.get("Tom"))
check("没记忆的计数是 0",
      subs.get("Tom", {}).get("l2") == 0 and subs.get("Tom", {}).get("l3") == 0,
      subs.get("Tom"))
check("有记忆的不标 empty", not subs.get("张三", {}).get("empty"), subs.get("张三"))
check("有记忆的排在前面", items[0]["subject"] != "Tom", [x["subject"] for x in items])

# 群名不能混进人物列表
store.write_l3(event_type="user_message", subject="group:123456789",
               content="[群] 某人: 路过", session_id="g1")
store.known_group("123456789", "测试群") if hasattr(store, "known_group") else None
resp2 = asyncio.run(p._api_subjects())
subs2 = {x["subject"] for x in resp2["data"]["items"]}
check("群名不进人物列表", not any("group:" in s for s in subs2), subs2)

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
