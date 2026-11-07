"""总结带记忆包 + 改口裁决测试。

跑的是四条硬规则：
  1. 取代必须有一条「这次真的落了库」的新事实兜底，否则等于纯删除；
  2. 旧事实比这段对话还新时（补录）不许被推翻；
  3. 图钉/别人/已归档的条目不许动；
  4. 一次最多改 3 条。
另外验证错峰沉底：不同条目拿到不同相位，倒计时和清理用同一个偏移。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, ".")

from sugarmemory_core.store import (  # noqa: E402
    SugarStore, retention_days_left, retention_stagger_days,
)
from sugarmemory_core.summarize_core import (  # noqa: E402
    MAX_SUPERSEDE_PER_SUMMARY, apply_summary, format_related, pick_related_l2,
    chunk_transcript, merge_summaries,
)
from sugarmemory_core.gate import WriteGate  # noqa: E402

NOW = time.time()
DAY = 86400
PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  x {name}   {detail}")


db = os.path.join(tempfile.gettempdir(), "sugarmem_updates.db")
for _p in (db, db + "-wal", db + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store = SugarStore(db)
gate = WriteGate()


def add(subject, pred, value, *, conf=0.9, ts=None, pin=False):
    res = store.write_l2(
        subject=subject, predicate=pred, value=value, speaker="user",
        source_type="user_message", gate_verdict={"decision": "allow"},
        confidence=conf, source_ts=ts or 0,
    )
    if pin:
        store.set_l2_pinned(res["id"], True)
    return int(res["id"])


def cand(subject, pred, value, act="statement"):
    return {"subject": subject, "predicate": pred, "value": value,
            "speaker": "user", "speech_act": act, "confidence": 0.9}


# ---------- 1. 改口：旧条归档 + 指向新条 ----------
print("\n===== supersede =====")
old = add("小周", "爱好技能", "喜欢摇滚")
stats = apply_summary(store, gate, {
    "summary": "他说现在只听说唱了",
    "candidates": [cand("小周", "爱好技能", "只听说唱")],
    "updates": [{"id": old, "action": "supersede", "by": 0}],
}, subject="小周")
row = store.get_l2(old)
check("旧条被归档", row["status"] == "archived", str(row["status"]))
check("指向新事实", int(row["replaced_by"]) > 0, str(row["replaced_by"]))
check("记了改口时间", float(row["superseded_at"]) > 0, str(row["superseded_at"]))
check("原文没被改写", row["value"] == "喜欢摇滚", str(row["value"]))
check("stats 计数", stats["superseded"] == 1 and stats["l2"] == 1, str(stats))
new_id = int(row["replaced_by"])
check("新事实是 active", store.get_l2(new_id)["status"] == "active")

# ---------- 2. 没有新事实兜底 = 纯删除，必须拒绝 ----------
print("\n===== no candidate backing =====")
solo = add("小周", "经历", "去年去过日本")
stats = apply_summary(store, gate, {
    "summary": "他说不想再提这件事",
    "candidates": [],
    "updates": [{"id": solo, "action": "supersede", "by": 0}],
}, subject="小周")
check("没有候选就不改", store.get_l2(solo)["status"] == "active")
check("计入 ignored", stats["update_ignored"] == 1 and stats["superseded"] == 0, str(stats))

# ---------- 3. by 指向被门控拦掉的候选 ----------
print("\n===== denied candidate =====")
keep = add("小周", "评价认为", "觉得同事挺靠谱")
stats = apply_summary(store, gate, {
    "summary": "他开玩笑说不靠谱",
    "candidates": [cand("小周", "评价认为", "觉得同事不靠谱", act="roleplay")],
    "updates": [{"id": keep, "action": "supersede", "by": 0}],
}, subject="小周")
check("角色扮演不算改口", store.get_l2(keep)["status"] == "active",
      store.get_l2(keep)["status"])
check("计 ignored", stats["update_ignored"] >= 1, str(stats))

# ---------- 4. 图钉 ----------
print("\n===== pinned =====")
pinned = add("小周", "自我相关", "有严重的青霉素过敏", pin=True)
stats = apply_summary(store, gate, {
    "summary": "他说过敏好了",
    "candidates": [cand("小周", "自我相关", "青霉素过敏已经治好")],
    "updates": [{"id": pinned, "action": "supersede", "by": 0}],
}, subject="小周")
check("钉住的不许被改口", store.get_l2(pinned)["status"] == "active")
check("计 ignored", stats["update_ignored"] == 1, str(stats))

# ---------- 5. 补录的旧话不能推翻新记忆 ----------
print("\n===== backfill cannot rewrite newer =====")
fresh = add("小周", "打算", "明年三月考专升本")
stats = apply_summary(store, gate, {
    "summary": "半年前的对话：他说不考了",
    "candidates": [cand("小周", "打算", "放弃专升本")],
    "updates": [{"id": fresh, "action": "supersede", "by": 0}],
}, subject="小周", source_ts=NOW - 300 * DAY)
check("旧对话改不动新事实", store.get_l2(fresh)["status"] == "active")
check("计 ignored", stats["update_ignored"] == 1, str(stats))

# ---------- 6. 张冠李戴 ----------
print("\n===== subject mismatch =====")
other = add("阿明", "爱好技能", "喜欢打篮球")
stats = apply_summary(store, gate, {
    "summary": "小周说不打篮球了",
    "candidates": [cand("小周", "爱好技能", "不再打篮球")],
    "updates": [{"id": other, "action": "supersede", "by": 0}],
}, subject="小周")
check("不能替别人改口", store.get_l2(other)["status"] == "active")
check("计 ignored", stats["update_ignored"] == 1, str(stats))

# ---------- 7. 同一个事实再说一遍不算改口 ----------
print("\n===== same value is not supersede =====")
same = add("阿明", "经历", "去年跑过半马")
stats = apply_summary(store, gate, {
    "summary": "他又提了一次半马",
    "candidates": [cand("阿明", "经历", "去年跑过半马")],
    "updates": [{"id": same, "action": "supersede", "by": 0}],
}, subject="阿明")
check("同一条不会自己取代自己", store.get_l2(same)["status"] == "active")
check("计 ignored", stats["update_ignored"] == 1, str(stats))

# ---------- 8. 一次最多改 3 条 ----------
print("\n===== cap =====")
olds = [add("阿杰", "经历", f"第 {i} 段旧经历") for i in range(5)]
stats = apply_summary(store, gate, {
    "summary": "他一口气更新了五件事",
    "candidates": [cand("阿杰", "经历", f"第 {i} 段新经历") for i in range(5)],
    "updates": [{"id": o, "action": "supersede", "by": i} for i, o in enumerate(olds)],
}, subject="阿杰")
check("最多改 3 条", stats["superseded"] == MAX_SUPERSEDE_PER_SUMMARY, str(stats))
check("其余被忽略", stats["update_ignored"] == 2, str(stats))
archived_n = sum(1 for o in olds if store.get_l2(o)["status"] == "archived")
check("归档条数对得上", archived_n == 3, str(archived_n))

# ---------- 9. 捞回来要清掉改口标记 ----------
print("\n===== unarchive clears supersede =====")
store.unarchive_l2(old)
row = store.get_l2(old)
check("恢复成 active", row["status"] == "active")
check("清掉 replaced_by", int(row["replaced_by"]) == 0, str(row["replaced_by"]))
check("清掉 superseded_at", float(row["superseded_at"]) == 0, str(row["superseded_at"]))
check("数据还在", row["value"] == "喜欢摇滚")

# ---------- 10. 相关记忆挑选 ----------
print("\n===== related pack =====")
rows = [
    {"id": 1, "subject": "小周", "predicate": "自我相关", "value": "现在 21 岁",
     "created_at": NOW, "updated_at": NOW, "ask_alias": ""},
    {"id": 2, "subject": "小周", "predicate": "爱好技能", "value": "喜欢摇滚",
     "created_at": NOW, "updated_at": NOW, "ask_alias": "听什么歌"},
    {"id": 3, "subject": "小周", "predicate": "经历", "value": "去过日本",
     "created_at": NOW, "updated_at": NOW, "ask_alias": ""},
]
picked = pick_related_l2(rows, "最近还在听摇滚吗", limit=2)
ids = [r["id"] for r in picked]
check("字面相关的被选中", 2 in ids, str(ids))
check("永久底档也保底带上", 1 in ids, str(ids))
check("条数不超", len(picked) <= 2, str(len(picked)))
check("条数够时全给", len(pick_related_l2(rows, "", limit=16)) == 3)
text = format_related(rows)
check("带编号", "[#1]" in text and "[#2]" in text, text)
check("带主体和谓词", "小周 · 爱好技能：喜欢摇滚" in text, text)

# ---------- 11. 错峰沉底 ----------
print("\n===== stagger =====")
store.stagger_days = 75.0
a = add("小周", "经历", "错峰测试甲")
b = add("小周", "经历", "错峰测试乙")
ra, rb = store.get_l2(a), store.get_l2(b)
sa = retention_stagger_days(ra, 75.0)
sb = retention_stagger_days(rb, 75.0)
check("两条相位不同", abs(sa - sb) > 0.01, f"{sa:.2f}/{sb:.2f}")
check("相位落在跨度内", 0 <= sa <= 75 and 0 <= sb <= 75, f"{sa:.2f}/{sb:.2f}")
check("同一条每次算出来一样", abs(sa - retention_stagger_days(ra, 75.0)) < 1e-9)
ma = store.l2_retention_map(subject="小周")[a]
check("map 带相位", "stagger" in ma and ma["stagger"] > 0, str(ma))
left_raw = retail = retention_days_left(dict(ra, _stagger=0.0), NOW)
left_stag = retail = retention_days_left(dict(ra, _stagger=sa), NOW)
check("错峰把倒计时往后推", left_stag > left_raw + 1,
      f"{left_raw:.1f}/{left_stag:.1f}")
check("map 的倒计时和带相位的一致", abs(ma["left"] - left_stag) < 1.0,
      f"{ma['left']}/{left_stag}")
store.stagger_days = 0.0
check("关掉就退回原曲线",
      abs(retention_days_left(ra, NOW) - left_raw) < 1.0,
      f"{retention_days_left(ra, NOW):.1f}/{left_raw:.1f}")

# ---------- 12. 长日志按消息边界切段，不再硬截断 ----------
print("\n===== chunk transcript =====")
lines = [f"[私] 小周: 第{i}条" + "啊" * 200 for i in range(12)]
chunks = chunk_transcript(lines, budget=1000)
check("切成了多段", len(chunks) > 1, str(len(chunks)))
check("每段都不超预算", all(len(c) <= 1000 for c in chunks),
      str([len(c) for c in chunks]))
check("一条消息没被劈开", all(ln in c for c in chunks for ln in c.split("\n")),
      "有段落中间被切开")
check("拼回去和原文一致", "\n".join(chunks) == "\n".join(lines))
check("极短日志只有一段", chunk_transcript(["a", "b"], budget=1000) == ["a\nb"])
check("空日志给一个空段", chunk_transcript([], budget=1000) == [""])

# ---------- 13. 多段结果合并：by 要按段偏移 ----------
print("\n===== merge summaries =====")
p1 = {"summary": "上午聊了音乐", "candidates": [{"v": 1}, {"v": 2}],
      "updates": [{"id": 9, "action": "supersede", "by": 1}]}
p2 = {"summary": "下午改口了", "candidates": [{"v": 3}],
      "updates": [{"id": 8, "action": "supersede", "by": 0}]}
m = merge_summaries([p1, p2])
check("摘要接起来", m["summary"] == "上午聊了音乐；下午改口了", m["summary"])
check("候选拼起来 3 条", len(m["candidates"]) == 3, str(len(m["candidates"])))
check("第二段的 by 偏移到 2", m["updates"][-1]["by"] == 2, str(m["updates"]))
check("第一段不受影响", m["updates"][0]["by"] == 1, str(m["updates"]))
check("不信段外下标", all(0 <= u["by"] < 3 for u in m["updates"]), str(m["updates"]))

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
