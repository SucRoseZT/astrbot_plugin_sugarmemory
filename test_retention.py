"""遗忘曲线测试：保留度、图钉、沉底归档、分类配额。

跑的是一条硬规则：**没有任何一条路径会物理删除长期记忆**。
沉底只改 status='archived'，随时能捞回来。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, ".")

from sugarmemory_core.store import (  # noqa: E402
    RETENTION_ARCHIVE_BELOW, PERMANENT_PREDICATES,
    SugarStore, retention_score, retention_days_left, retention_quiet_days,
)
from sugarmemory_core.pack import PackBuilder  # noqa: E402

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


db = os.path.join(tempfile.gettempdir(), "sugarmem_retention.db")
for _p in (db, db + "-wal", db + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store = SugarStore(db)


def add(subject, pred, value, *, ts=None, conf=0.7, pin=False):
    res = store.write_l2(
        subject=subject, predicate=pred, value=value, speaker="user",
        source_type="user_message", gate_verdict={"decision": "allow"},
        confidence=conf, source_ts=ts or 0,
    )
    if pin:
        store.set_l2_pinned(res["id"], True)
    return int(res["id"])


def age(fid, days):
    """把一条事实的整体时间线往前推。用来模拟「很久没被碰到」。"""
    t = NOW - days * DAY
    store._db.execute(
        "UPDATE l2_facts SET created_at=?, updated_at=?, stored_at=? WHERE id=?",
        (t, t, t, fid),
    )


# ---------- 1. 保留度曲线 ----------
print("\n===== retention curve =====")
base = {"confidence": 0.7, "stability": 0, "last_hit_at": 0,
        "predicate": "经历", "pinned": 0}
r0 = retention_score(dict(base, created_at=NOW, updated_at=NOW), NOW)
r30 = retention_score(dict(base, created_at=NOW - 30 * DAY, updated_at=NOW - 30 * DAY), NOW)
r90 = retention_score(dict(base, created_at=NOW - 90 * DAY, updated_at=NOW - 90 * DAY), NOW)
r180 = retention_score(dict(base, created_at=NOW - 180 * DAY, updated_at=NOW - 180 * DAY), NOW)
check("刚入库接近 1", r0 > 0.99, f"{r0:.3f}")
check("越久越低", r0 > r30 > r90 > r180, f"{r0:.3f}/{r30:.3f}/{r90:.3f}/{r180:.3f}")
check("30 天还有四成", 0.3 < r30 < 0.55, f"{r30:.3f}")
check("180 天基本沉底", r180 < RETENTION_ARCHIVE_BELOW, f"{r180:.4f}")

# 反复确认过的事更牢
strong = retention_score(dict(base, created_at=NOW - 90 * DAY, updated_at=NOW - 90 * DAY,
                              stability=5, confidence=0.9), NOW)
check("确认过 5 次更牢（90 天仍过半数）", strong > 0.45, f"{strong:.3f}")
# 昨天刚被提过就重新激活
revived = retention_score(dict(base, created_at=NOW - 180 * DAY, updated_at=NOW - 180 * DAY,
                               last_hit_at=NOW - 2 * DAY), NOW)
check("刚被提过就重新激活", revived > 0.85, f"{revived:.3f}")

# 入库时间不参与 Δt：补录半年前的数据不该显示成刚聊过
backfilled = retention_score(dict(base, created_at=NOW - 180 * DAY,
                                 updated_at=NOW - 180 * DAY, stored_at=NOW), NOW)
check("入库时间不影响保留度", backfilled < RETENTION_ARCHIVE_BELOW, f"{backfilled:.4f}")

# ---------- 2. 图钉与永久分类 ----------
print("\n===== pin & permanent =====")
pinned_row = dict(base, created_at=NOW - 200 * DAY, updated_at=NOW - 200 * DAY, pinned=1)
check("图钉不给沉底倒计时", retention_days_left(pinned_row, NOW) < 0,
      str(retention_days_left(pinned_row, NOW)))
perm_row = dict(base, predicate="自我相关", created_at=NOW - 200 * DAY,
                updated_at=NOW - 200 * DAY)
check("永久分类不给沉底倒计时", retention_days_left(perm_row, NOW) < 0)
check("永久分类名单就是这两类", PERMANENT_PREDICATES == {"自我相关", "其他"},
      str(PERMANENT_PREDICATES))

# ---------- 3. 沉底归档 ----------
print("\n===== archive =====")
f_old = add("老张", "经历", "三年前跑过半马")
f_pin = add("老张", "经历", "这是他唯一在意的事", pin=True)
f_perm = add("老张", "自我相关", "对花生过敏，会休克")
f_fresh = add("老张", "经历", "昨天刚学会游泳")
for fid in (f_old, f_pin, f_perm):
    age(fid, 200)
# 补录：事实很老，但今天才入库
f_back = add("老张", "经历", "很久以前养过一只叫煤球的猫", ts=NOW - 400 * DAY)

dry = store.cleanup_decay(now=NOW, dry_run=True)
check("dry_run 不写库", dry["archived"] == 0)
check("dry_run 之后还是 active",
      len(store.query_active_l2(subject="老张", limit=50)) == 5,
      str(len(store.query_active_l2(subject="老张", limit=50))))

live = store.cleanup_decay(now=NOW)
check("只归档了那条没人管的旧事", live["archived"] == 1, str(live))
active_ids = {int(r["id"]) for r in store.query_active_l2(subject="老张", limit=50)}
arch_ids = {int(r["id"]) for r in store.query_archived_l2(subject="老张", limit=50)}
check("沉底的那条不在 active", f_old not in active_ids)
check("沉底的那条在归档区", f_old in arch_ids, str(arch_ids))
check("图钉没被收走", f_pin in active_ids)
check("疾病这种底档没被收走", f_perm in active_ids)
check("刚说的没被收走", f_fresh in active_ids)
check("补录的老事实当天不收", f_back in active_ids, str(active_ids))

check("归档不是删除：库里还在", store.get_l2(f_old) is not None)

# ---------- 4. 恢复 ----------
print("\n===== restore =====")
before = float(store.get_l2(f_old)["last_hit_at"] or 0)
ok = store.unarchive_l2(f_old)
row = store.get_l2(f_old)
check("恢复成功", ok and row["status"] == "active", row["status"])
check("恢复顺手刷新 last_hit_at（不会立刻再沉）",
      float(row["last_hit_at"] or 0) > before and
      retention_score(row, NOW) > 0.9, f"{retention_score(row, NOW):.3f}")

# ---------- 5. 图钉开关 ----------
print("\n===== pin toggle =====")
store.set_l2_pinned(f_pin, False)
age(f_pin, 400)
store.set_l2_pinned(f_fresh, True)
store.set_l2_pinned(f_fresh, True)          # 幂等
res = store.cleanup_decay(now=NOW)
ids_after = {int(r["id"]) for r in store.query_active_l2(subject="老张", limit=50)}
check("松开图钉后照样会沉", f_pin not in ids_after, str(sorted(ids_after)))
check("钉住的没事", f_fresh in ids_after)
check("这一轮收走 1 条", res["archived"] == 1, str(res))

# ---------- 6. 补录旧话不刷新 updated_at ----------
print("\n===== backfill keeps timestamps =====")
f_old_ts = add("小周", "经历", "很久以前在网吧通宵", ts=NOW - 400 * DAY)
old_created = float(store.get_l2(f_old_ts)["created_at"])
check("补录事实的 updated_at 跟着原话时间",
      abs(float(store.get_l2(f_old_ts)["updated_at"]) - old_created) < 1,
      str(store.get_l2(f_old_ts)["updated_at"] - old_created))
check("但入库时间是现在",
      float(store.get_l2(f_old_ts)["stored_at"]) > old_created + 100 * DAY,
      str(store.get_l2(f_old_ts)["stored_at"] - old_created))
# 补录的老事实保留度一进门就过线，但入库观察期还没满 90 天——
# 倒计时必须把闸门算进去，否则面板会写「约 0 天后沉底」。
check("补录当天不会显示「马上沉底」",
      retention_days_left(store.get_l2(f_old_ts), NOW) >= 89,
      str(retention_days_left(store.get_l2(f_old_ts), NOW)))
check("补录当天不会被归档",
      store.cleanup_decay(now=NOW)["archived"] == 0,
      str(store.cleanup_decay(now=NOW)))

# 同一句话又补录一遍：稳度涨，但时间线不许往前跳
add("小周", "经历", "很久以前在网吧通宵", ts=NOW - 900 * DAY)
row = store.get_l2(f_old_ts)
check("重复补录稳度 +1", int(row["stability"]) == 1, str(row["stability"]))
check("重复补录不把 updated_at 顶到今天",
      float(row["updated_at"]) < NOW - 100 * DAY,
      str((NOW - float(row["updated_at"])) / DAY))

# ---------- 7. 分类配额 ----------
print("\n===== quota =====")
for i in range(20):
    add("阿明", "经历", f"上班路上碰到的第 {i} 件小事，地铁又晚点了")
add("阿明", "自我相关", "有严重的青霉素过敏")
add("阿明", "其他", "紧急联系人是姐姐 13800000000")
pb = PackBuilder(store, subject="阿明", session_id="t", l2_limit=12,
                 l3_snippets=0, skeleton_limit=0)
pack = pb.build("地铁又晚点了", budget_tokens=900)
preds = [r["predicate"] for r in pack["l2"]]
check("配额生效：底档挤进了前 12", "自我相关" in preds and "其他" in preds, str(preds))
check("条数没超上限", len(pack["l2"]) <= 12, str(len(pack["l2"])))
check("相关性高的还在（没被配额挤空）", len(preds) >= 3, str(len(preds)))
check("配额不抢光名额", preds.count("自我相关") + preds.count("其他") <= 4, str(preds))

# 没有永久分类可用时不该硬塞
pb2 = PackBuilder(store, subject="小周", session_id="t", l2_limit=12,
                  l3_snippets=0, skeleton_limit=0)
pack2 = pb2.build("网吧通宵", budget_tokens=900)
check("没有底档就正常走相关性", len(pack2["l2"]) >= 1, str(len(pack2["l2"])))

# ---------- 8. 绝不物理删除 ----------
print("\n===== never delete =====")
import sqlite3  # noqa: E402
c = sqlite3.connect(db)
total = c.execute("SELECT COUNT(*) FROM l2_facts").fetchone()[0]
active = c.execute("SELECT COUNT(*) FROM l2_facts WHERE status='active'").fetchone()[0]
arch = c.execute("SELECT COUNT(*) FROM l2_facts WHERE status='archived'").fetchone()[0]
c.close()
check("沉底前后总数不变（active+archived=total）", active + arch == total,
      f"{active}+{arch}={total}")

store.close()
try:
    os.remove(db)
except OSError:
    pass

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
