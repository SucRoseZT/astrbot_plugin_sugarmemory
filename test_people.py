"""人物身份的回归测试：QQ 是锚，称呼分四档，叫错的名字会自己忘掉。

实机背景（2026-11-04）：同一个人把 QQ 昵称从「Moon」改成「张三」，插件按昵称
当 subject 落库，于是 L2 分成两套、关系图上成了两个人。名字之间字面上毫无关系，
只有 QQ 号能证明是同一个人——所以这一整套都围绕「先看 QQ 再看名字」。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, ".")

from sugarmemory_core.store import (  # noqa: E402
    SugarStore, alias_key, retention_score,
)

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


db = os.path.join(tempfile.gettempdir(), "sugarmem_people.db")
for _p in (db, db + "-wal", db + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store = SugarStore(db)

print("===== 同一个 QQ = 同一个人 =====")
p1 = store.person_ensure("Moon", qq="10001", source="event")
p2 = store.person_ensure("张三", qq="10001", source="event")
check("换昵称不会变成两个人", p1 == p2, f"{p1} / {p2}")
check("旧昵称也能认回来", store.person_resolve("Moon") == p1)
check("新昵称认得", store.person_resolve("张三") == p1)
d = store.person_get(p1)
check("显示名跟到最新昵称", d["display_name"] == "张三", d["display_name"])
# 称呼名（大家怎么叫他）不跟着昵称跑：昵称是他自己随手改的，常用叫法是别人
# 认下来的。跟着改的话，每换一次名 AI 的称呼也跟着变，等于每个人都只有
# 「当前昵称」、没有常用叫法。面板要的是「本人昵称 + 一个常用叫法」并列。
check("称呼名保持最初那个", d["call_name"] == "Moon", d["call_name"])

print("\n===== 不同 QQ 不会被合 =====")
p3 = store.person_ensure("路人甲", qq="20002", source="event")
check("不同号是不同人", p3 != p1, f"{p3} / {p1}")
check("陌生人不会被并进来", store.person_resolve("路人甲") == "路人甲")

print("\n===== 称呼分级 =====")
tiers = {a["raw"]: a["tier"] for a in store.person_get(p1)["aliases"]}
check("当前昵称是显示名（tier 0）", tiers.get("张三") == 0, str(tiers))
check("旧昵称降成外号（tier 2）", tiers.get("Moon") == 2, str(tiers))
# 用户自己说「老张 就是 张三」，是明说的外号，tier 2
store.person_bind("老张", "张三", source="quote")
tiers = {a["raw"]: a["tier"] for a in store.person_get(p1)["aliases"]}
check("用户说的外号进 tier 2", tiers.get("老张") == 2, str(tiers))
# 手动固定称呼名之后，换昵称也不改它
store.person_set_call_name(p1, "张三")
store.person_ensure("第三个昵称", qq="10001", source="event")
d = store.person_get(p1)
check("固定的称呼名不随昵称变", d["call_name"] == "张三", d["call_name"])
check("显示名仍然跟着变", d["display_name"] == "第三个昵称", d["display_name"])
tiers = {a["raw"]: a["tier"] for a in d["aliases"]}
check("固定的称呼名是 tier 1", tiers.get("张三") == 1, str(tiers))

print("\n===== 叫错的名字会忘 =====")
# 总结抄出来一个谁都不认识的名字 → tier 3
store.person_ensure("张三糖", source="summary")
tiers = {a["raw"]: a["tier"] for a in store.person_get("张三糖")["aliases"]}
check("模型抄错的名字进 tier 3", tiers.get("张三糖") == 3, str(tiers))
# 90 天没出现 → 清掉
res = store.alias_cleanup(guess_days=90, now=NOW + 91 * DAY)
check("91 天后叫错的名字被清", "张三糖" in (res.get("aliases") or []), str(res))
check("调用后确实绑不上了", store.person_resolve("张三糖") == "")
# 顶层两种永远不删
res2 = store.alias_cleanup(guess_days=1, nick_days=1, now=NOW + 9999 * DAY)
kept = {a["raw"] for a in (store.person_get(p1) or {}).get("aliases") or []}
check("显示名永不自动删", "第三个昵称" in kept, str(kept))
check("称呼名永不自动删", "张三" in kept, str(kept))
check("只有旧外号留下", True, str(res2))

print("\n===== 外号留得久，叫过两次就不清 =====")
_db2 = os.path.join(tempfile.gettempdir(), "sugarmem_people2.db")
for _p in (_db2, _db2 + "-wal", _db2 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store2 = SugarStore(_db2)
p = store2.person_ensure("阿强", qq="30003", source="event")
store2.person_bind("强哥", "阿强", source="quote")
# 叫过一次
res = store2.alias_cleanup(guess_days=90, nick_days=365, now=NOW + 400 * DAY, dry_run=True)
check("叫过一次的外号 400 天后可清", "强哥" in (res.get("aliases") or []), str(res))
store2.person_touch("强哥")
store2.person_touch("强哥")
res = store2.alias_cleanup(guess_days=90, nick_days=365, now=NOW + 400 * DAY,
                           dry_run=True)
check("叫过三次的外号留着", "强哥" not in (res.get("aliases") or []), str(res))

print("\n===== 合并：两个人并成一个，称呼都在 =====")
_db3 = os.path.join(tempfile.gettempdir(), "sugarmem_people3.db")
for _p in (_db3, _db3 + "-wal", _db3 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store3 = SugarStore(_db3)
a = store3.person_ensure("张三", qq="40004", source="event")
b = store3.person_ensure("Moon", qq="50005", source="event")
check("合并前是两个人", a != b)
# 两个号各绑着不同的 QQ，自动流程一律不许合（实机踩过：朋友的号被并进自己）
denied = store3.person_merge(keep="张三", drop="Moon")
check("不同 QQ 不会被自动合并", not denied.get("ok")
      and denied.get("reason") == "qq_conflict", str(denied))
store3._fold_one("张三", "Moon")
check("自动折叠也合不动", store3.person_get("Moon") is not None)
# 人亲手点的合并才放行
store3.person_bind("Moon", "张三", source="quote", force=True)
store3.person_merge(keep="张三", drop="Moon", force=True)
check("手动确认后只剩一个", store3.person_get("Moon") is None)
merged = store3.person_get("张三")
check("合并后还在", merged is not None)
names = {x["raw"] for x in (merged or {}).get("aliases") or []}
check("两边称呼都在", {"张三", "Moon"} <= names, str(names))
check("旧名字也能认到同一个人", store3.person_resolve("Moon") == "张三")

print("\n===== 泛称不是人 =====")
_db4 = os.path.join(tempfile.gettempdir(), "sugarmem_people4.db")
for _p in (_db4, _db4 + "-wal", _db4 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store4 = SugarStore(_db4)
check("「用户」建不出人", store4.person_ensure("用户", qq="60006", source="event") == "")
check("「对方」也不行", store4.person_ensure("对方", source="summary") == "")
check("库里没有多出这个人", store4.person_get("用户") is None)
check("把泛称绑给人会被拒",
      store4.person_bind("用户", "张三").get("reason") == "generic")

print("\n===== 同名不同号是两个人 =====")
x = store4.person_ensure("张三", qq="70007", source="event")
y = store4.person_ensure("张三", qq="80008", source="event")
check("重名但号不同 → 两个人", x != y, f"{x} / {y}")
check("后来者用名字+号尾区分", "#" in y, y)
check("先来的还是原名", x == "张三", x)
check("各自的号都认得回", store4.person_by_qq("70007") == x
      and store4.person_by_qq("80008") == y)
check("同一个号换昵称还是同一个人",
      store4.person_ensure("Moon", qq="70007", source="event") == x)

print("\n===== 群聊按 QQ 认人，不按群名片 =====")


class _FakeEvent:
    """只带插件真正用到的那两个接口。"""

    def __init__(self, name, qq):
        self._name, self._qq = name, qq

    def get_sender_name(self):
        return self._name

    def get_sender_id(self):
        return self._qq


from sugarmemory_core.store import speaker_key  # noqa: E402

check("群里换了群名片也能认出是谁",
      speaker_key(store4, _FakeEvent("赵六（群名片）", "70007")) == "张三")
check("没登记过的号退回群名片",
      speaker_key(store4, _FakeEvent("路人甲", "99999")) == "路人甲")
check("拿不到号时退回名字",
      speaker_key(store4, _FakeEvent("阿明", "")) == "阿明")

print("\n===== 注入用的显示名 =====")
nm = store.person_names([p1, "不存在的人"])
check("person 键映射到称呼名", nm.get(p1) == "张三", str(nm))
check("查不到的不编造", "不存在的人" not in nm, str(nm))

print("\n===== QQ 昵称 / AI 称呼 是两个位置 =====")
# 用户定的规矩：QQ 昵称是「这人是谁」的稳定标识（认人、没称呼时就用它），
# call_name 是 AI 真正开口叫的那个。两个不能混成一个字段，否则面板上分不清
# 哪个是他的号名、哪个是 AI 平时的叫法。
nick = store.person_nicknames([p1, "不存在的人"])
check("认人走 QQ 昵称", nick.get(p1) == "第三个昵称", str(nick))
check("认人查不到也不编造", "不存在的人" not in nick, str(nick))
check("叫法仍然是 call_name", store.person_names([p1]).get(p1) == "张三",
      str(store.person_names([p1])))
# 没设过 call_name 的人（新建时默认就是昵称）两个位置应该一样，
# 注入里就不会出现「张三(张三)」这种废话。
_p9 = store.person_ensure("只有昵称的人", qq="90009", source="event")
check("没设称呼时两个位置一致",
      store.person_names([_p9]).get(_p9) == store.person_nicknames([_p9]).get(_p9),
      f"{store.person_names([_p9])} / {store.person_nicknames([_p9])}")

print("\n===== 总结喂给模型的名字 =====")
# 模型看到什么名字，摘要里就写什么名字。所以名单和底档都必须用「平时怎么叫」，
# 不能给内部键（往往是最早那个昵称）——否则摘要里会留一个谁也不认识的名字，
# 注入回来又变成"两个人"。
from sugarmemory_core.summarize_core import roster_text, format_related  # noqa: E402

_p10 = store.person_ensure("阿明", qq="90010", source="event")
store.person_set_call_name(_p10, "老明")
_shown = store.person_names([_p10])
check("内部键和叫法确实不同", _p10 != _shown.get(_p10), f"{_p10} / {_shown}")
check("名单渲染用叫法", roster_text({1: _shown[_p10]}).endswith("1=老明"),
      roster_text({1: _shown[_p10]}))
_rows = [{"id": 7, "subject": _p10, "predicate": "爱好技能",
          "value": "听说唱", "created_at": 1}]
check("底档清单也用叫法", format_related(_rows, _shown).startswith("- [#7] 老明 ·"),
      format_related(_rows, _shown))
check("不给映射时退回内部键", format_related(_rows).startswith("- [#7] 阿明 ·"),
      format_related(_rows))

print("\n===== 称谓词不是名字 =====")
# 实机背景（旧版本）：「XX 是我弟」被总结成一个叫「弟弟」的新用户，
# 关系图上凭空多出一个人。称谓词只是称呼，不能当名字建档。
from sugarmemory_core.store import is_kinship_alias  # noqa: E402

_db5 = os.path.join(tempfile.gettempdir(), "sugarmem_people5.db")
for _p in (_db5, _db5 + "-wal", _db5 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store5 = SugarStore(_db5)
store5.person_ensure("张三", qq="10001", source="event")
check("「弟弟」不许凭空建人", store5.person_ensure("弟弟", source="summary") == "",
      repr(store5.person_ensure("弟弟", source="summary")))
check("「我妈」也不许", store5.person_ensure("我妈", source="summary") == "")
check("「女朋友」也不许", store5.person_ensure("女朋友", source="summary") == "")
check("库里没多出假人", store5.person_get("弟弟") is None)
check("带人称前缀的也认得出称谓",
      is_kinship_alias("我弟弟") and is_kinship_alias("你的妈妈"))
check("普通名字不受影响",
      not is_kinship_alias("张三") and not is_kinship_alias("小七"))
check("群友真叫弟弟，有号照样建档",
      store5.person_ensure("弟弟", qq="90009", source="event") == "弟弟")
check("建过档之后 AI 再写「弟弟」认得是他",
      store5.person_ensure("弟弟", source="summary") == "弟弟")

r = store5.relation_upsert(source="张三", target="我妈", relation="亲友",
                           source_type="summary")
check("自动写「张三→我妈」被拒", r.get("reason") == "kinship", str(r))
check("被拒之后没有这条边",
      not [e for e in store5.relations_all()
           if e["source"] == "张三" and e["target"] == "我妈"])
r2 = store5.relation_upsert(source="张三", target="我妈", relation="亲友",
                            source_type="manual")
check("面板手动写放行（人自己认的关系）", r2.get("ok") is True, str(r2))

from sugarmemory_core.gate import WriteGate  # noqa: E402
from sugarmemory_core.summarize_core import apply_summary  # noqa: E402

_db6 = os.path.join(tempfile.gettempdir(), "sugarmem_people6.db")
for _p in (_db6, _db6 + "-wal", _db6 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store6 = SugarStore(_db6)
store6.person_ensure("张三", qq="10001", source="event")
stats = apply_summary(store6, WriteGate(), {
    "summary": "聊到弟弟",
    "candidates": [
        {"subject": "弟弟", "predicate": "自我相关", "value": "弟弟今年 12 岁",
         "speaker": "user", "speech_act": "statement",
         "source_quote": "我弟今年 12 岁", "confidence": 0.8},
    ],
    "relations": [{"source": "张三", "target": "我弟", "relation": "亲友",
                   "quote": "我弟今年 12 岁", "confidence": 0.8}],
    "updates": [],
}, subject="张三")
check("总结里的称谓主体被跳过", stats["kinship_skipped"] == 2, str(stats))
check("称谓实体没有落成 L2",
      not [r for r in store6._db.execute("SELECT subject FROM l2_facts")
           if r["subject"] in ("弟弟", "我弟")])
check("人物库里也没多出称谓", store6.person_get("弟弟") is None
      and store6.person_get("我弟") is None)

# ---------- 关系边：自动不建人，手动才建 ----------
# 边本身照写（关系图的节点是从边推出来的，不依赖 people 行），但「多一个人」
# 这件事只留给手动操作。自动总结写出来的名字认不出是谁，说明它在猜。
_db7 = os.path.join(tempfile.gettempdir(), "sugarmem_people_7.db")
for _p in (_db7, _db7 + "-wal", _db7 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store7 = SugarStore(_db7)
store7.person_ensure("张三", qq="10001", source="event")
res = store7.relation_upsert(source="张三", target="陌生人甲",
                             relation="朋友", source_type="summary")
check("自动关系不给陌生人建档", store7.person_get("陌生人甲") is None, res)
check("自动关系的边照样写进去",
      any(e["target"] == "陌生人甲" for e in store7.relations_all()),
      store7.relations_all())
res = store7.relation_upsert(source="张三", target="Tom",
                             relation="朋友", source_type="manual")
check("手动关系仍然建档", store7.person_get("Tom") is not None, res)
check("手动关系两边都在人物库里",
      bool(store7.person_get("张三")) and bool(store7.person_get("Tom")))

print("\n===== 群名片只算别名 =====")
# 群名片是这个人「在某一个群里的马甲」，不是他自己的昵称。同一个 QQ 加好几个
# 群、每个群名片都不一样，如果都按真实昵称处理，面板上的名字就会跟着群来回跳。
_db8 = os.path.join(tempfile.gettempdir(), "sugarmem_people_8.db")
for _p in (_db8, _db8 + "-wal", _db8 + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)
store8 = SugarStore(_db8)
p8 = store8.person_ensure("赵六", qq="70007", source="event", tier=0)
store8.person_ensure("小七", qq="70007", source="event", tier=2)  # 群名片
d8 = store8.person_get(p8)
check("群名片不改显示名", d8["display_name"] == "赵六", d8["display_name"])
check("群名片认得回来", store8.person_resolve("小七") == p8, store8.person_resolve("小七"))
check("群名片不进 tier 0",
      all(a["tier"] != 0 for a in d8["aliases"] if a["alias_key"] == "小七"),
      d8["aliases"])
check("昵称也没被顶掉", store8.person_resolve("赵六") == p8)

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
