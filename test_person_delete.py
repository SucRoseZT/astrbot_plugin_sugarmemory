"""person_delete 回归测试：面板删人，人、记忆、关系、好感一次清干净。

背景（2026-11-04）：面板只有「合并」，没有「删掉」——认错了的人（泛称
救回来的假人、重复建的小号）只能挂在那。合并是搬家，删除是物理清除：
称呼、好感、关系边、关系证据、L2、L3、档案本体全清，且只清这个人的。
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")

from sugarmemory_core.store import SugarStore  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  x {name}   {detail}")


db = os.path.join(tempfile.gettempdir(), "sugarmem_pdelete.db")
for _p in (db, db + "-wal", db + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store = SugarStore(db)

ALLOW = {"decision": "allow", "reason": "test"}

print("===== 造数据：三个人，A 认错了要删 =====")
store.person_ensure("阿A", qq="10001", source="event")
store.person_ensure("阿B", qq="20002", source="event")
store.person_ensure("阿C", qq="30003", source="event")
# A 的外号 + 一条外号写法的历史 L3（模拟归一前的孤儿数据）
store.write_l3(event_type="chat", content="阿A说过他叫小A", subject="小A")
store.person_bind("小A", "阿A", source="quote")
store.write_l2(subject="阿A", predicate="爱好技能", value="喜欢唱歌",
               speaker="阿A", gate_verdict=ALLOW)
store.write_l2(subject="阿A", predicate="生活习惯", value="晚睡",
               speaker="阿A", gate_verdict=ALLOW)
store.write_l3(event_type="chat", content="阿A说今天加班", subject="阿A")
store.affinity_add("阿A", 120)
store.relation_upsert(source="阿A", target="阿B", relation="朋友",
                      quote="阿A说阿B是他哥们")
# B、C 的数据不能被误伤
store.write_l2(subject="阿B", predicate="爱好技能", value="会下棋",
               speaker="阿B", gate_verdict=ALLOW)
store.write_l3(event_type="chat", content="阿B上线了", subject="阿B")
store.affinity_add("阿B", 55)
store.relation_upsert(source="阿B", target="阿C", relation="认识",
                      quote="阿B说认识阿C")
store.write_l3(event_type="chat", content="阿C水群", subject="阿C")

n_l2 = store.query_active_l2(limit=100)
n_l3 = store.query_recent_l3(limit=100)
check("造完数据 L2=3", len(n_l2) == 3, str(len(n_l2)))
check("造完数据 L3=4", len(n_l3) == 4, str(len(n_l3)))
check("A 有关系边", any(e["source"] == "阿A" for e in store.relations_all()))

print("\n===== 删除 A =====")
res = store.person_delete("阿A")
check("删除成功", res.get("ok") is True, str(res))
check("L2 删了 2 条", res.get("l2") == 2, str(res.get("l2")))
check("L3 删了 2 条（含外号写法的孤儿）", res.get("l3") == 2, str(res.get("l3")))
check("称呼删了 2 条（本名 + 外号）", res.get("aliases") == 2, str(res.get("aliases")))
check("好感带走 120", res.get("xp") == 120.0, str(res.get("xp")))
check("关系边删 1", res.get("edges") == 1, str(res.get("edges")))
check("关系证据删 1", res.get("evidence") == 1, str(res.get("evidence")))

print("\n===== 清理验证 =====")
check("档案没了", store.person_get("阿A") is None)
check("L2 只剩 B 的", [r["subject"] for r in store.query_active_l2(limit=100)] == ["阿B"],
      str([r["subject"] for r in store.query_active_l2(limit=100)]))
rest_l3 = [r["subject"] for r in store.query_recent_l3(limit=100)]
check("L3 剩 B、C 的", rest_l3.count("阿A") == 0 and "阿B" in rest_l3 and "阿C" in rest_l3,
      str(rest_l3))
check("A 的好感没了", store.affinity_get("阿A") is None)
check("B 的好感还在", (store.affinity_get("阿B") or {}).get("xp") == 55.0)
left_edges = store.relations_all()
check("A 的边没了", not any("阿A" in (e["source"], e["target"]) for e in left_edges),
      str(left_edges))
check("B-C 的边还在", any(e["source"] == "阿B" and e["target"] == "阿C"
                        for e in left_edges))
check("B 的档案还在", store.person_get("阿B") is not None)
check("C 的档案还在", store.person_get("阿C") is not None)
check("resolve 认不出 A 了", store.person_resolve("阿A") == "")
check("外号也认不出 A 了", store.person_resolve("小A") == "")

print("\n===== 用量归属与 qq 变体：删完不能再显示这个人 =====")
# 真实事故（2026-11-04）：删了叫「1」的人，人物页没了，用量里还留着
# 「1 · 入 xx / 出 xx」；另外泛称发送者会以 qq123456 当主体落库，
# 那部分当时也没被带走，记忆列表里会多出一格空壳。
store.person_ensure("阿D", qq="40004", source="event")
store.write_l3(event_type="chat", content="泛称兜底的记录", subject="qq40004")
store.token_record("chat", input=10, output=5, subject="阿D")
store.token_record("pack", input=3, output=1, subject="qq40004")
store.token_record("chat", input=7, output=2, subject="阿C")
res_d = store.person_delete("阿D")
check("删除阿D 成功", res_d.get("ok") is True, str(res_d))
check("用量归属抹掉 2 条", res_d.get("tokens") == 2, str(res_d.get("tokens")))
check("qq 变体的 L3 一起删", res_d.get("l3") == 1, str(res_d.get("l3")))
tok_left = [r["subject"] for r in store._db.execute(
    "SELECT subject FROM token_usage WHERE subject IN ('阿D','qq40004')")]
check("用量里认不出阿D", not tok_left, str(tok_left))
check("别人的用量没被动", store._db.execute(
    "SELECT COUNT(*) c FROM token_usage WHERE subject='阿C'").fetchone()["c"] == 1)
check("删干净后自检 total=0", store.person_residue("阿D").get("total") == 0,
      str(store.person_residue("阿D")))
check("自检能看到别人的残留", store.person_residue("阿C").get("total", 0) > 0,
      str(store.person_residue("阿C")))

print("\n===== 边界 =====")
check("删不存在的人 not_found", store.person_delete("阿A").get("reason") == "not_found")
# not_found 不能顺手把同名的边删掉：以前是先删边再报没找到，
# 点一下「删掉一个已经不存在的人」就把别人的关系带走了。
# 这里用「我弟」造一条没有人物档案的边（称谓词建不出人，但旧的 summary
# 边可能就挂在那），然后再对这个名字点删除。
store.relation_upsert(source="阿B", target="我弟", relation="朋友",
                      source_type="manual")
check("not_found 不动别人的边",
      not store.person_delete("我弟").get("ok")
      and any("我弟" in (e["source"], e["target"]) for e in store.relations_all()),
      str(store.relations_all()))
store.relation_delete_edge("阿B", "我弟", drop_evidence=True)
check("空名 invalid", store.person_delete("  ").get("reason") == "invalid")
check("删 B 成功", store.person_delete("阿B").get("ok") is True)
check("B 删后 C 的边跟着没了（证据也清）",
      not store.relations_all() and not store.relation_evidence("阿B", "阿C"))
check("C 毫发无损", store.person_get("阿C") is not None)

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
