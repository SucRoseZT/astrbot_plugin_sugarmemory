"""召回升级回归：候选池 / IDF 打分 / 上下文补位 / 去冗余 / B 方案话题刷新。

纯标准库，直接运行本文件即可。不依赖 AstrBot 运行时、不碰真实 memory.db。
"""
import ast
import hashlib
import logging
import os
import sys
import tempfile
import textwrap
import time
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sugarmemory_core.store import SugarStore, PERMANENT_PREDICATES  # noqa: E402
from sugarmemory_core.pack import PackBuilder  # noqa: E402
from sugarmemory_core.injector import (  # noqa: E402
    build_block, estimate_tokens, trim_pack)

MAIN_SRC = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
TREE = ast.parse(MAIN_SRC)


def _fn(name):
    return next(n for n in ast.walk(TREE)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)


def _literal(name):
    return ast.literal_eval(next(n.value for n in TREE.body
                                 if isinstance(n, ast.Assign)
                                 and getattr(n.targets[0], "id", "") == name))


CFG_DEFAULTS = _literal("CFG_DEFAULTS")
PROMPT_CFG_KEYS = _literal("PROMPT_CFG_KEYS")
CACHE_MAX = _literal("INJECT_CACHE_MAX")
# 0.16.15 起「话题防抖窗口」已移除（fp 变了就重建）。常量保留为 0，
# 只当「把 built_at 往前推」的测试辅助，不再参与任何判定。
TOPIC_MIN_INTERVAL = 0.0

NS = {
    "hashlib": hashlib, "time": time,
    "logger": logging.getLogger("recall_upgrade"),
    "build_block": build_block, "estimate_tokens": estimate_tokens,
    "trim_pack": trim_pack, "PERMANENT_PREDICATES": PERMANENT_PREDICATES,
    "TOPIC_MIN_INTERVAL": TOPIC_MIN_INTERVAL,
    "CFG_DEFAULTS": CFG_DEFAULTS, "PROMPT_CFG_KEYS": PROMPT_CFG_KEYS,
    "INJECT_CACHE_MAX": CACHE_MAX,
}
NS.setdefault("re", __import__("re"))
for _name in ("_cfg_fp", "_inject_mode", "_is_noise_reply",
              "_resident_l2_n", "_frozen_block"):
    exec(textwrap.dedent(ast.get_source_segment(MAIN_SRC, _fn(_name))), NS)


class FakeSelf:
    _cfg_fp = NS["_cfg_fp"]
    _inject_mode = NS["_inject_mode"]
    _resident_l2_n = NS["_resident_l2_n"]
    _frozen_block = NS["_frozen_block"]

    def __init__(self, store):
        self.store = store
        self.config = dict(CFG_DEFAULTS)
        self._inject_cache = {}

    def _cfg(self, key, default):
        return self.config.get(key, default)

    def _pack_builder(self, subject="", session_id="", extra_subjects=None,
                      l2_keep_min_override=None, lite=False, bot_id="",
                      semantic_scores=None):
        return PackBuilder(
            self.store,
            l2_limit=int(self._cfg("pack_l2_limit", 12)),
            l2_keep_min=(int(l2_keep_min_override) if l2_keep_min_override
                         else int(self._cfg("pack_l2_keep_min", 3))),
            subject=subject, session_id=session_id,
            extra_subjects=list(extra_subjects or []),
            outbound_relay_limit=0, outbound_cron_limit=0,
            cross_limit=0, skeleton_limit=0, lite=lite,
            bot_id=self._iso(bot_id),
            candidate_limit=int(self._cfg("pack_l2_candidate_limit", 200)),
            context_rounds=int(self._cfg("pack_context_rounds", 3)),
        )

    def _iso(self, bot):
        return str(bot or "").strip()[:32]


def _fixture(path, noise=120):
    store = SugarStore(path)
    con = store._db
    ts = 1767225600.0
    rows = []
    for i in range(noise):
        rows.append(("闲聊碎片", f"第{i}次随口提到天气和晚饭，还有通勤", 0.99))
    rows.append(("饮品偏好", "喜欢喝拿铁", 0.35, "那个拿铁其实有点东西；我一般都点这个"))
    rows.append(("饮品偏好", "喜欢喝拿铁不加糖", 0.34))
    rows.append(("饮品偏好", "常点拿铁", 0.33))
    rows.append(("户外", "周末常去登山", 0.32))
    rows.append(("户外", "登山鞋是去年买的", 0.31))
    for i, row in enumerate(rows):
        pred, val, conf = row[0], row[1], row[2]
        alias = row[3] if len(row) > 3 else ""
        con.execute(
            "INSERT INTO l2_facts (layer,subject,predicate,value,ask_alias,speaker,"
            "source_type,fingerprint,confidence,status,stability,created_at,updated_at,stored_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("L2", "测试人", pred, val, alias, "测试人", "chat", f"fp{i}",
             conf, "active", 0, ts + i, ts + i, ts + i))
    con.commit()
    return store


def main():
    tmp = tempfile.mkdtemp(prefix="sugarmem_recall_")
    try:
        store = _fixture(os.path.join(tmp, "fixture.db"))

        # 1) 候选池：120 条高置信噪声之后，低置信但本轮相关的「拿铁」不能被挤掉
        old = store.query_active_l2(subject="测试人", limit=80)
        assert not any("拿铁" in r["value"] for r in old), \
            "fixture 失效：置信度前 80 里本来就该没有拿铁"
        got = store.query_active_l2(subject="测试人", limit=200, query="拿铁")
        assert any("拿铁" in r["value"] for r in got[:10]), \
            "先按置信度截断的老毛病又回来了：低置信高相关事实仍进不了池"
        assert got[0]["confidence"] < 0.9, "相关度排序没生效，还是置信度排最前"
        print("[1] 候选池 ✓ 低置信高相关事实从 120 条噪声里被捞出来")

        # 2) IDF / 归一 / _why：每条候选都要留下可解释的打分明细
        b = PackBuilder(store, subject="测试人", l2_limit=12,
                        candidate_limit=200, context_rounds=3,
                        cross_limit=0, skeleton_limit=0,
                        outbound_relay_limit=0, outbound_cron_limit=0)
        pack = b.build(query="上次说的拿铁", budget_tokens=1200)
        assert pack["why"], "没有排序明细，/糖记忆 why 还是只能靠猜"
        sample = next(iter(pack["why"].values()))
        for k in ("channel", "bigram", "idf", "norm", "score", "in_pack"):
            assert k in sample, f"排序明细缺 {k}"
        assert all(0.0 <= float(w["norm"]) <= 1.0 for w in pack["why"].values()), \
            "min-max 归一没落在 0~1"
        top = pack["l2"]
        assert all(float(top[i]["_score"]) >= float(top[i + 1]["_score"]) - 1e-9
                   for i in range(len(top) - 1)), "L2 不是按 _score 降序"
        print("[2] 排序 ✓ IDF 加权 / 查询内归一 / 每条挂 _why")

        # 3) MMR：三条几乎同义的「拿铁」不能同时吃掉名额
        pool = [dict(r) for r in store.query_active_l2(
            subject="测试人", limit=200, query="拿铁")[:6]]
        for r in pool:
            r["_why"] = {}
            r["_score"] = 1.0
        kept = b._apply_mmr(pool)
        latte = [r for r in kept if "拿铁" in r["value"]]
        assert len(latte) <= 2, f"近重复没让位，还留着 {len(latte)} 条"
        print(f"[3] 去冗余 ✓ 6 条候选压到 {len(kept)} 条，近重复让位")

        # 4) 上下文补位：「那它呢」本轮零命中，靠上一轮把拿铁带回来
        # 0.16.16 起 _rel/_rel_content 只统计「本轮本体」命中，历史原话的
        # 贡献单独记在 _rel_ctx —— 否则上一轮的话题会一直污染下一轮。
        p2 = b.build(query="那它呢", budget_tokens=1200,
                     recent_queries=["上次说喜欢喝拿铁"])
        hit = [r for r in p2["l2"]
               if "拿铁" in str(r.get("value") or "")
               and int(r.get("_rel_ctx") or 0) > 0]
        assert hit, "最近几轮没参与检索，指代问法依旧零命中"
        assert p2.get("query_context"), "上下文没记录，面板无法解释"
        print("[4] 上下文 ✓ 「那它呢」接上了上一轮的拿铁")

        # 5) B 方案：闲聊不重建，话题命中才刷新这一轮，并且能稳定复用
        f = FakeSelf(store)
        b1, _, _, m1 = f._frozen_block("测试人", "S1", 1200)
        b2, _, _, m2 = f._frozen_block("测试人", "S1", 1200)
        assert b1 == b2 and m2["reason"] == "same_fp", \
            "闲聊轮次竟然重建了，前缀缓存要被冲掉"
        # 让节流窗口过去，模拟真实对话里隔了半分钟换话题
        for e in f._inject_cache.values():
            e["built_at"] -= TOPIC_MIN_INTERVAL + 1
        ids = [r["id"] for r in store.query_active_l2(
            subject="测试人", limit=200, query="拿铁")
            if "拿铁" in r["value"]][:1]
        b3, _, _, m3 = f._frozen_block(
            "测试人", "S1", 1200, topic_ids=ids, topic_query="拿铁")
        assert m3["reason"] == "fp_changed", \
            f"话题命中没触发增量刷新：reason={m3['reason']}"
        assert "拿铁" in b3, "刷新了但答案没进包"
        b4, _, _, m4 = f._frozen_block(
            "测试人", "S1", 1200, topic_ids=ids, topic_query="拿铁")
        assert b3 == b4 and not m4["rebuilt"], \
            "同一话题连续两轮反复重建，缓存会掉回去"
        print("[5] B 方案 ✓ 闲聊零改动 / 话题命中刷新一次 / 同话题复用")

        # 6) 语义兜底默认不通电，注入总闸默认开
        assert CFG_DEFAULTS["inject_enabled"] is True, "注入总闸又变回默认关了"
        assert CFG_DEFAULTS["pack_semantic_fallback"] is False
        assert CFG_DEFAULTS["semantic_provider_id"] == ""
        for k in ("semantic_provider_id", "semantic_timeout_ms",
                  "inject_topic_refresh", "inject_topic_extra",
                  "pack_context_rounds", "pack_l2_candidate_limit"):
            assert k in CFG_DEFAULTS and k in PROMPT_CFG_KEYS
        print("[6] 默认值 ✓ 注入默认开、语义路默认关、新配置都进了指纹")

        # 7) 小预算：候选包自己先 trim 一次会把低置信高相关条目砍光，
        #    曾经表现为「话题刷新了但答案一条没进包」。
        f2 = FakeSelf(store)
        f2.config["inject_budget_tokens"] = 300
        c1, _, _, _ = f2._frozen_block("测试人", "T1", 300)
        for e in f2._inject_cache.values():
            e["built_at"] -= TOPIC_MIN_INTERVAL + 1
        latte_ids = [r["id"] for r in store.query_active_l2(
            subject="测试人", limit=200, query="拿铁")
            if "拿铁" in r["value"]][:3]
        c2, _, _, cm2 = f2._frozen_block(
            "测试人", "T1", 300, topic_ids=latte_ids,
            topic_query="上次说的拿铁")
        assert cm2["reason"] == "fp_changed", f"小预算没刷新：{cm2['reason']}"
        assert "拿铁" in c2, "小预算下话题增量被 trim 砍光，补了等于没补"
        print("[7] 小预算 ✓ 300 tok 下话题事实仍进包")

        # 8) 同一话题换个说法：不重建、字节不变（query 不能污染渲染）
        c3, _, _, cm3 = f2._frozen_block(
            "测试人", "T1", 300, topic_ids=latte_ids, topic_query="上次说的拿铁")
        assert not cm3["rebuilt"] and c2 == c3, \
            "相同查询仍反复重建，缓存白废"
        print("[8] 话题稳定 ✓ 相同查询不重建、字节不变")

        # 9) 命中的全是常驻包里已有的条目 → 不该白废一次前缀
        f3 = FakeSelf(store)
        d1, _, _, _ = f3._frozen_block("测试人", "T2", 1200)
        for e in f3._inject_cache.values():
            e["built_at"] -= TOPIC_MIN_INTERVAL + 1
        base_ids = [r["id"] for r in store.query_active_l2(
            subject="测试人", limit=6)]
        d2, _, _, dm2 = f3._frozen_block(
            "测试人", "T2", 1200, topic_ids=base_ids,
            topic_query="在吗今天干嘛呢")
        assert not dm2["rebuilt"] and d1 == d2, \
            "没补到任何包外事实，却白重建了一次"
        print("[9] 空增量 ✓ 没补到新事实就不重建")

        # 10) 真换话题（补进来完全不同的一组）不受防抖压住
        f4 = FakeSelf(store)
        e1, _, _, _ = f4._frozen_block("测试人", "T3", 1200)
        for e in f4._inject_cache.values():
            e["built_at"] -= TOPIC_MIN_INTERVAL + 1
        _, _, _, em2 = f4._frozen_block(
            "测试人", "T3", 1200, topic_ids=latte_ids, topic_query="拿铁")
        assert em2["reason"] == "fp_changed"
        hike_ids = [r["id"] for r in store.query_active_l2(
            subject="测试人", limit=200, query="登山")
            if "登山" in r["value"]][:3]
        # 刻意不推进时间：固定防抖若还在压，这里就会漏掉新话题
        e3, _, _, em3 = f4._frozen_block(
            "测试人", "T3", 1200, topic_ids=hike_ids, topic_query="登山")
        assert em3["reason"] == "fp_changed", "真换话题被防抖压住了"
        assert "登山" in e3
        print("[10] 换话题 ✓ 补入完全不同的一组事实，立刻刷新")

        # 11) 长噪声撑高相对阈值时，只命中一两个 bigram 的短答案不能被判死
        b3 = f._pack_builder(subject="测试人", session_id="T4", lite=True)
        fake = [
            {"id": 1, "_rel": 7, "_rel_content": 7, "_rel_w": 21.0,
             "_score": 1.00, "value": "长噪声"},
            {"id": 3, "_rel": 3, "_rel_content": 3, "_rel_w": 9.0,
             "_score": 0.80, "value": "噪声二"},
            {"id": 4, "_rel": 3, "_rel_content": 3, "_rel_w": 9.0,
             "_score": 0.78, "value": "噪声三"},
            {"id": 5, "_rel": 3, "_rel_content": 3, "_rel_w": 9.0,
             "_score": 0.76, "value": "噪声四"},
            {"id": 2, "_rel": 1, "_rel_content": 1, "_rel_w": 4.2,
             "_score": 0.55, "value": "喜欢喝拿铁"},
            {"id": 6, "_rel": 0, "_rel_content": 0, "_rel_w": 0.0,
             "_score": 0.40, "value": "无关"},
        ]
        kept = b3._filter_by_relevance(
            fake, "你还记得我上次说的那家拿铁咖啡馆叫什么名字吗")
        assert any(r["id"] == 2 for r in kept), "短答案被长噪声撑高的阈值误杀"
        print("[11] 闸门 ✓ 长噪声撑高阈值时短答案仍保留")

        # 12) 真实统计轨路径：非永久分类的强相关事实不能被永久槽位挤掉。
        # 否则统计轨 l2_rows 丢了它 → topic_ids 变空 → _frozen_block 的话题
        # 刷新门永远不开 → 表现为「输入做音乐却一点不更新」。这是 0.16.1
        # 之后修掉的召回 bug（_pick_l2 永久槽位腾位时误踢相关事实）。
        tmp2 = tempfile.mkdtemp(prefix="sugarmem_pickl2_")
        try:
            sp2 = os.path.join(tmp2, "pickl2.db")
            s2 = SugarStore(sp2)
            c2 = s2._db
            ts2 = 1767225600.0
            # 13 条高置信永久分类噪声：会触发永久槽位去抢占名额
            for i in range(13):
                c2.execute(
                    "INSERT INTO l2_facts (layer,subject,predicate,value,speaker,"
                    "source_type,fingerprint,confidence,status,stability,created_at,"
                    "updated_at,stored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("L2", "测试人", "其他", f"琐事{i}", "测试人", "chat",
                     f"fp{i}", 0.99, "active", 0, ts2 + i, ts2 + i, ts2 + i))
            # 1 条低置信但本轮强相关的非永久事实
            c2.execute(
                "INSERT INTO l2_facts (layer,subject,predicate,value,speaker,"
                "source_type,fingerprint,confidence,status,stability,created_at,"
                "updated_at,stored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("L2", "测试人", "爱好", "做音乐", "测试人", "chat",
                 "fpmusic", 0.33, "active", 0, ts2 + 100, ts2 + 100, ts2 + 100))
            c2.commit()
            b5 = PackBuilder(s2, subject="测试人", l2_limit=12, l2_keep_min=3,
                             candidate_limit=200, context_rounds=3,
                             cross_limit=0, skeleton_limit=0,
                             outbound_relay_limit=0, outbound_cron_limit=0,
                             lite=True)
            stat = b5.build(query="做音乐", budget_tokens=1200)
            assert any(r.get("value") == "做音乐" and r.get("_count_hit")
                       for r in stat["l2"]), \
                "统计轨把本轮相关的非永久事实挤掉了，topic_ids 会因此变空"
            assert any(int(r.get("id") or 0) in (stat.get("topic_ids") or [])
                       for r in stat["l2"] if r.get("value") == "做音乐"), \
                "话题命中没进 topic_ids，冻结包刷新门不开"
            print("[12] 统计轨 ✓ 非永久强相关事实不被永久槽位挤掉，topic_ids 正常")

            # 端到端：真实统计轨产出的 topic_ids 喂给 _frozen_block 能补进包
            f5 = FakeSelf(s2)
            f5._frozen_block("测试人", "P1", 1200)
            for e in f5._inject_cache.values():
                e["built_at"] -= TOPIC_MIN_INTERVAL + 1
            tb, _, _, tm = f5._frozen_block(
                "测试人", "P1", 1200, topic_ids=stat["topic_ids"],
                topic_query="做音乐")
            assert tm["reason"] == "fp_changed", f"端到端没刷新：{tm['reason']}"
            assert "做音乐" in tb, "做音乐没补进冻结包"
            print("[12b] 端到端 ✓ 输入做音乐后冻结包真刷新并补入该事实")
        finally:
            try:
                s2._db.close()
            except Exception:
                pass
            shutil.rmtree(tmp2, ignore_errors=True)

        # 13) 噪声闸门（0.16.16）：问法别名里的口语词、上一轮的话题词，
        #     都不能顶替「本轮本体命中」。实测反例：「嘿嘿想你了」靠 #46 别名
        #     里的「你了」把「定时任务插件」补进尾部；「吃东西」靠上一轮的
        #     「插件开发」把「开发了记忆系统插件」补进尾部。
        #     fixture 里「喜欢喝拿铁」的别名带「其实 / 有点 / 东西」，正好当诱饵。
        #    断言的是打分口径（_rel 只算本轮本体），不是「l2 里有没有它」——
        #    l2 还承担「保底 3 条」的职责，拿铁以零命中身份待在里面是正常的，
        #    真正决定它会不会被补进尾部的是 main 里那条 _rel>0 的候选条件。
        p13 = b.build(query="东西", budget_tokens=1200, recent_queries=[])
        for r in p13["l2"]:
            if "拿铁" in str(r.get("value") or ""):
                assert int(r.get("_rel") or 0) == 0, \
                    "问法别名里的口语词被算成了本体命中"
        p14 = b.build(query="吃东西", budget_tokens=1200,
                      recent_queries=["喜欢喝拿铁"])
        for r in p14["l2"]:
            if "拿铁" in str(r.get("value") or ""):
                assert int(r.get("_rel") or 0) == 0, \
                    "上一轮的话题词混进了本轮的命中判据"
        print("[13] 噪声闸门 ✓ 别名口语词 / 历史话题词都不再冒充本轮命中")

        print("\nALL CHECKS PASSED")
    finally:
        try:
            store._db.close()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
