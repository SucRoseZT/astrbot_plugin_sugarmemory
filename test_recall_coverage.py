"""召回覆盖面回归：一句寒暄、一个改名、一次点名、一个单字问法。

这套用例来自 0.16.17 的一次「自造 51 条记忆 × 60 种问法」压测。压测本身
放在仓库外（.probe/recall_matrix.py），这里只留 9 条最能代表病根的断言，
每条都对应过一个真实漏洞。跑法：python test_recall_coverage.py
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sugarmemory_core.store import SugarStore      # noqa: E402
from sugarmemory_core.pack import PackBuilder      # noqa: E402

ME = "测试人"
TS = 1767225600.0

# key, subject, predicate, value, ask_alias, confidence, stability
MEM = [
    ("name",   ME, "自我相关", "本名陈岚，朋友都叫我阿岚", "", 0.95, 5),
    ("age",    ME, "自我相关", "今年23岁，2003年出生", "", 0.95, 4),
    ("guitar", ME, "爱好技能", "会弹木吉他，大学时组过乐队",
     "我会什么乐器；我弹吉他吗；我以前玩乐队吗", 0.85, 2),
    ("cat",    ME, "关系", "养了一只橘猫叫年糕，两岁了",
     "我养了什么猫；我宠物叫什么；我养宠物吗", 0.90, 4),
    ("ex",     ME, "关系", "去年和女朋友分手了，谈了三年",
     "我有对象吗；我谈过几次恋爱；我为什么分手", 0.85, 2),
    ("mom",    ME, "关系", "妈妈今年五十二，有糖尿病，每天打胰岛素",
     "我妈多大年纪；我妈身体怎么样；我妈有什么病", 0.92, 4),
    ("gastric", ME, "自我相关", "有慢性胃炎，不能空腹喝咖啡",
     "我胃怎么样；我身体有什么毛病", 0.92, 3),
    ("move",   ME, "打算", "打算明年搬到杭州去",
     "我以后想去哪；我要搬家吗", 0.86, 3),
    ("quit",   ME, "打算", "在考虑辞职做自由职业",
     "我要辞职吗；我以后想做什么", 0.82, 2),
    ("read",   ME, "爱好技能", "爱看推理小说，最喜欢东野圭吾",
     "我看什么书；我喜欢哪个作家", 0.83, 2),
    ("opdir",  ME, "评价认为", "最讨厌别人说话绕弯子，喜欢直说",
     "我讨厌什么样的人；我喜欢怎么说话", 0.84, 3),
    ("job",    ME, "自我相关", "在一家做跨境电商的小公司做运营", "", 0.90, 3),
    ("b_job",  "老周", "自我相关", "本名周维，做装修生意",
     "老周是干什么的；老周叫什么", 0.88, 3),
    # ---- 0.16.18：婆家 / 娘家 / 伴侣，都在考「这条在说谁」----
    ("birth",  ME, "自我相关", "生日是农历八月十六", "我生日是哪天", 0.88, 3),
    ("mil",    "婆婆", "关系", "婆婆跟我们一起住，有高血压，天天吃药",
     "我婆婆身体怎么样", 0.90, 3),
    ("opmil",  ME, "评价认为", "觉得婆婆管得太多，但不好明说",
     "我和婆婆关系怎么样", 0.86, 3),
    ("cook",   ME, "生活习惯", "不太会做饭，平时都是婆婆做", "我会做饭吗", 0.86, 3),
    ("hush",   ME, "关系", "老公叫阿强，跑建材业务，经常出差",
     "我老公叫什么", 0.90, 4),
    ("sweet",  ME, "生活习惯", "特别爱吃甜的，尤其是蛋糕",
     "我爱吃什么；我喜欢甜食吗", 0.88, 3),
    ("crash",  ME, "经历", "前年被电动车撞过，膝盖留了疤",
     "我受过伤吗；我出过车祸吗", 0.86, 2),
    ("wallet", ME, "经历", "在地铁上丢过钱包，后来没找回来",
     "我丢过什么东西", 0.82, 1),
    ("tool",   ME, "爱好技能", "电脑上装了 VSCode 和 PyCharm", "", 0.84, 2),
]


def _fixture(path):
    store = SugarStore(path)
    con = store._db
    ids = {}
    for i, (key, subj, pred, val, alias, conf, stab) in enumerate(MEM):
        cur = con.execute(
            "INSERT INTO l2_facts (layer,subject,predicate,value,ask_alias,speaker,"
            "source_type,fingerprint,confidence,status,stability,created_at,updated_at,stored_at)"
            " VALUES ('L2',?,?,?,?,?,'chat',?,?,'active',?,?,?,?)",
            (subj, pred, val, alias, subj, f"fp_{key}", conf, stab,
             TS + i, TS + i, TS + i))
        ids[key] = int(cur.lastrowid)
    con.commit()
    return store, ids


def _build(store, query, hist=(), extra=None, keep=3, limit=12):
    return PackBuilder(
        store, l2_limit=limit, candidate_limit=200, context_rounds=3,
        cross_limit=0, skeleton_limit=0, l3_snippets=0,
        outbound_limit=0, outbound_relay_limit=0, outbound_cron_limit=0,
        subject=ME, session_id="FRIEND:1001", extra_subjects=extra,
        l2_keep_min=keep, rel_ratio=0.3,
    ).build(query=query, budget_tokens=900, recent_queries=list(hist))


def main():
    tmp = tempfile.mkdtemp(prefix="sugarmem_cov_")
    store, ids = _fixture(os.path.join(tmp, "cov.db"))
    ok = 0

    def check(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError(msg)
        ok += 1
        print(f"  ✓ {msg}")

    # [1] 寒暄不背书：零命中就是零条，不再垫「本名/年龄/住址」三张身份卡
    for q in ("在吗", "嘿嘿想你了", "吃饭了吗"):
        pack = _build(store, q)
        check(not pack["l2"], f"「{q}」不注入任何长期事实（实际 {len(pack['l2'])} 条）")

    # [2] 别名召回的答案不能被永久槽位挤掉（_rel=0 曾被当成「无关」踢出）
    pack = _build(store, "我会什么乐器")
    check(ids["guitar"] in [r["id"] for r in pack["l2"]],
          "「我会什么乐器」→ 吉他（别名通道）留在包里，没被本名顶掉")

    # [3] 指代补位同理
    pack = _build(store, "那它呢", hist=["我养了什么猫"])
    check(ids["cat"] in [r["id"] for r in pack["l2"]],
          "「那它呢」→ 上一轮的猫（上下文通道）留在包里")

    # [4] predicate 是分类名，不是内容：不能让「关系」把整类家人一起捞出来
    pack = _build(store, "我有对象吗")
    got = [r["id"] for r in pack["l2"]]
    check(ids["ex"] in got and ids["mom"] not in got and ids["cat"] not in got,
          "「我有对象吗」→ 只带分手那条，不把妈妈/猫一起带上")

    # [5] 全是态度词的命中不算相关：问颜色不能答摄影
    pack = _build(store, "我最喜欢什么颜色")
    check(not pack["l2"], "「我最喜欢什么颜色」→ 库里没记过，就不硬凑")

    # [6] 单字锚点 + 主体对齐：问自己，家人的病不能排第一
    pack = _build(store, "我有什么病")
    got = [r["id"] for r in pack["l2"]]
    check(got and got[0] == ids["gastric"],
          "「我有什么病」→ 胃炎排第一（不是妈妈的糖尿病）")
    check(ids["mom"] in got, "「我有什么病」→ 妈妈那条仍在包里（并列比较时不丢信息）")

    # [7] 人名通道：问「老周」时，档案本身要能被主语名撞到
    pack = _build(store, "老周是干什么的", extra=["老周"])
    check(ids["b_job"] in [r["id"] for r in pack["l2"]],
          "「老周是干什么的」→ 老周自己的档案进包")

    # [8] 句末焦点：问「辞职」，不能被「打算」两个字糊弄过去
    pack = _build(store, "我打算辞职吗")
    got = [r["id"] for r in pack["l2"]]
    check(got and got[0] == ids["quit"],
          "「我打算辞职吗」→ 辞职那条排第一（不是搬家那条）")

    # [9] 常驻包（query 为空）照旧凑满：它的保底逻辑不受上面任何一条影响
    mine = sum(1 for m in MEM if m[1] == ME)   # 老周 / 婆婆那两条不在当前 subject 作用域内
    pack = _build(store, "", keep=mine, limit=mine)
    check(len(pack["l2"]) == mine,
          f"常驻包 query='' 仍拿满 {mine} 条（实际 {len(pack['l2'])}）")

    # ---- 0.16.18：主体归属（婆家 / 娘家 / 伴侣）----
    # 三条记忆都含「婆婆」，字面分不开。只有抬头是她的那条才是答案。
    pack = _build(store, "我婆婆身体怎么样", extra=["婆婆"])
    got = [r["id"] for r in pack["l2"]]
    check(got and got[0] == ids["mil"],
          f"「我婆婆身体怎么样」→ 婆婆的健康档案排第一（实际 {got[:1]}）")

    # 问伴侣的生日，不能拿我自己的生日回答。库里没记，就该只是「关于他的事」
    pack = _build(store, "我老公生日是哪天")
    got = [r["id"] for r in pack["l2"]]
    check(ids["birth"] not in got,
          f"「我老公生日是哪天」→ 不拿我自己的生日顶（实际 {got}）")

    # 纯指代追问：本轮没有实词，要接着上一轮那个人说，不能被记忆里
    # 「后来没找回来」这种叙述里的「后来」抢走。
    pack = _build(store, "后来呢", hist=["我前年出过车祸吗"])
    got = [r["id"] for r in pack["l2"]]
    check(ids["crash"] in got,
          f"「后来呢」→ 接着上一轮的车祸（实际 {got}）")

    # 主体不符的别名兜底要挡住：问婆婆口味，不能答「我爱吃甜的」
    pack = _build(store, "我婆婆喜欢吃什么", extra=["婆婆"])
    got = [r["id"] for r in pack["l2"]]
    check(ids["sweet"] not in got,
          f"「我婆婆喜欢吃什么」→ 不拿我自己的口味顶（实际 {got}）")

    # ---- 0.16.19：字符归一（纯 bigram 逐字符比较的盲区）----
    # 专业全文索引默认大小写不敏感，天然「iPhone == iphone」。我们是逐字符
    # 比较，不归一就是硬碰运气：三个字母的词只靠一两个 2-gram 凑巧重叠，
    # 两三个字符的直接零命中。
    pack = _build(store, "我用 vscode 吗")
    check(ids["tool"] in [r["id"] for r in pack["l2"]],
          "「我用 vscode 吗」→ 召回 VSCode（大小写 + 全角归一）")

    print(f"\nALL {ok} CHECKS PASSED")

if __name__ == "__main__":
    main()
