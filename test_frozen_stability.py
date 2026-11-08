"""常驻记忆包的字节稳定性回归测试。纯标准库，直接运行本文件即可。

两件事分开测，别混成一个断言：
  1) 模板快照：对一份**固定假 pack** 调 build_block()，比对 sha256。
     改模板里一个空格/换行就会 fail —— 那会让所有分片的前缀缓存同时失效。
     假 pack 刻意不带任何时间字段，避开 time.localtime 的本机时区依赖。
  2) 属性测试：本轮用户原话怎么变，冻结块的字节都不许变；
     只有「改影响 prompt 的配置」或「L2 内容被改写」才允许重建，
     且 reason 必须分别是 config / l2_changed（不能互相误判）。

不依赖 AstrBot 运行时：从 main.py 里 AST 抽出被i测函数，配一份临时 SQLite
fixture 跑。不碰真实 memory.db。
"""
import ast
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sugarmemory_core.store import SugarStore, PERMANENT_PREDICATES  # noqa: E402
from sugarmemory_core.pack import PackBuilder  # noqa: E402
from sugarmemory_core.injector import (  # noqa: E402
    build_block, estimate_tokens, trim_pack)

# build_block 对固定 pack 的输出指纹。模板有**有意**改动时再更新这个值，
# 更新前请确认你接受「所有活跃会话下一轮全部 Cache Miss」的代价。
# 2026-11-08：L2 段前加了常量的时间说明行（值里的「今晚」指的是记下那天），
# 故意改了模板 → 每会话一次性 Cache Miss，之后照旧稳定。
GOLDEN_BLOCK_FP = "1cb093dd65fe46b9"

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

import logging  # noqa: E402

NS = {
    "hashlib": hashlib, "time": time,
    "logger": logging.getLogger("frozen_stability"),
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


class CountingStore:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if callable(attr):
            def wrapper(*a, **kw):
                self.calls.append(name)
                return attr(*a, **kw)
            return wrapper
        return attr


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
        )

    def _iso(self, bot):
        """多机器人隔离开关（测试里恒开，空串=不限）。"""
        return str(bot or "").strip()[:32]


FAKE_PACK = {
    "query": "", "identities": [], "probe_outbound": False,
    "summary": "固定摘要：用于模板字节快照，不要带相对时间。",
    "l2": [
        {"id": 1, "subject": "测试人", "predicate": "生活习惯",
         "value": "喜欢喝茶", "confidence": 0.95},
        {"id": 2, "subject": "测试人", "predicate": "爱好技能",
         "value": "会下棋", "confidence": 0.90},
    ],
    "l3": [], "outbound": [], "cross": [], "skeleton": [],
    "excluded": [], "trim_notes": [],
}


def _fixture_db(path):
    store = SugarStore(path)
    con = store._db
    ts = 1767225600.0  # 固定时间，不放任何相对时间依赖
    for i, (pred, val) in enumerate((
        ("生活习惯", "喜欢喝茶"), ("爱好技能", "会下棋"), ("自我相关", "今年 20 岁"))):
        con.execute(
            "INSERT INTO l2_facts (layer,subject,predicate,value,speaker,source_type,"
            "fingerprint,confidence,status,stability,created_at,updated_at,stored_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("L2", "测试人", pred, val, "测试人", "chat", f"fp{i}",
             0.95, "active", 0.9, ts + i, ts + i, ts + i))
    con.commit()
    return store


def _fixture_db_multi(path):
    """7 条事实：6 条高置信 + 1 条极低置信（喜欢潜水），用来制造一条
    「包外」事实，验证话题刷新到底往块的哪个位置追加。"""
    store = SugarStore(path)
    con = store._db
    ts = 1767225600.0
    rows = [
        ("生活习惯", "喜欢喝茶"), ("爱好技能", "会下棋"),
        ("自我相关", "今年 20 岁"), ("饮食偏好", "爱吃辣"),
        ("作息", "通常晚睡"), ("社交", "朋友很多"),
        ("运动爱好", "喜欢潜水"),  # 极低置信 → 留在常驻包外
    ]
    for i, (pred, val) in enumerate(rows):
        is_outlier = (pred == "运动爱好")
        conf = 0.20 if is_outlier else 0.92
        stab = 0.20 if is_outlier else 0.90
        con.execute(
            "INSERT INTO l2_facts (layer,subject,predicate,value,speaker,source_type,"
            "fingerprint,confidence,status,stability,created_at,updated_at,stored_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("L2", "测试人", pred, val, "测试人", "chat", f"fp{i}",
             conf, "active", stab, ts + i, ts + i, ts + i))
    con.commit()
    return store


def main():
    tmp = tempfile.mkdtemp(prefix="sugarmem_frozen_")
    store = None
    try:
        # ── 1) 模板快照 ───────────────────────────────────────────
        text = build_block(FAKE_PACK)
        fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        print(f"[1] 模板快照 fp = {fp}（golden = {GOLDEN_BLOCK_FP}）")
        if GOLDEN_BLOCK_FP != fp:
            raise AssertionError(
                f"模板字节漂移：{GOLDEN_BLOCK_FP} → {fp}。"
                "若是有意改模板，手动更新 GOLDEN_BLOCK_FP 并接受全量 Cache Miss。")
        print("    ✓ 模板字节未漂移")

        store = _fixture_db(os.path.join(tmp, "fixture.db"))
        f = FakeSelf(store)

        # ── 2) 属性测试：8 轮不同原话，冻结块恒定 ─────────────────
        queries = ["插件", "阿蓝", "你还记得我几岁吗", "昨天转述了什么",
                   "定时任务还在吗", "聊聊棋类", "记忆包预算多少", "随便说点什么"]
        blocks, metas = [], []
        for q in queries:
            # 统计轨（lite）按原话检索，结果不进 prompt；再跑注入轨
            fb = f._pack_builder(subject="测试人", session_id="S1", lite=True)
            fb.build(query=q, budget_tokens=1200)
            b, _pt, _rt, m = f._frozen_block("测试人", "S1", 1200, rel_text="")
            blocks.append(b)
            metas.append(m)
        print(f"[2] 8 轮不同原话 → 不同块数 {len(set(blocks))}，"
              f"fp 种类 {len({m['fp'] for m in metas})}，"
              f"reason {[m['reason'] for m in metas]}")
        assert len(set(blocks)) == 1, "本轮原话泄漏进 system_prompt 了"
        assert len({m["fp"] for m in metas}) == 1
        assert metas[0]["rebuilt"] and all(not m["rebuilt"] for m in metas[1:])
        assert all(m["arch_ver"] == "frozen" for m in metas)
        print("    ✓ query 不进 system，fp 恒定，归因正确")

        # ── 3) 无关配置不动指纹，影响 prompt 的配置才动 ────────────
        fp_cfg = f._cfg_fp()
        f.config["inject_freeze_seconds"] = 999
        f.config["token_stats_enabled"] = False
        assert f._cfg_fp() == fp_cfg
        f.config["pack_l2_limit"] = 8
        b, _pt, _rt, m = f._frozen_block("测试人", "S1", 1200, rel_text="")
        print(f"[3] 改 pack_l2_limit → reason={m['reason']}")
        assert m["rebuilt"] and m["reason"] == "config", \
            "配置变更被误判成「用户刚改了记忆」"
        print("    ✓ 配置变更不再伪装成 l2_changed")

        # ── 4) 真实 L2 改写才允许 important ───────────────────────
        store._db.execute("UPDATE l2_facts SET value=value || '（改写）' WHERE id=1")
        store._db.commit()
        b, _pt, _rt, m = f._frozen_block("测试人", "S1", 1200, rel_text="")
        print(f"[4] 改写 L2 → reason={m['reason']}")
        assert m["rebuilt"] and m["reason"] == "l2_changed"
        print("    ✓ 真实记忆改写立即重建")

        # ── 5) lite 统计轨确实省掉重查询 ──────────────────────────
        c_full, c_lite = CountingStore(store), CountingStore(store)
        PackBuilder(c_full, l2_limit=12, subject="测试人", session_id="S1",
                    l3_snippets=0).build(query="插件", budget_tokens=1200)
        p_lite = PackBuilder(c_lite, l2_limit=12, subject="测试人", session_id="S1",
                             l3_snippets=0, lite=True).build(
            query="插件", budget_tokens=1200)
        print(f"[5] store 调用 全量 {len(c_full.calls)} → lite {len(c_lite.calls)}；"
              f"lite 标记 {p_lite.get('_lite')}")
        assert len(c_lite.calls) < len(c_full.calls)
        assert p_lite.get("_lite") is True
        print("    ✓ 统计轨不再付两次全量检索的钱")

        # ── 6) 话题命中 = 只追加后缀，前缀（常驻底档）字节不变 ────────
        # 用户最关心的缓存语义：话题刷新到底会不会废掉前缀。直接冻结路径验证。
        tmp2 = tempfile.mkdtemp(prefix="sugarmem_fz6_")
        store2 = None
        try:
            store2 = _fixture_db_multi(os.path.join(tmp2, "fx.db"))
            f2 = FakeSelf(store2)
            f2.config["pack_l2_limit"] = 6  # 6 进常驻，极低那条「喜欢潜水」留在包外
            # 先拿基线冻结块（闲聊，无话题）
            b0, _p0, _r0, m0 = f2._frozen_block("测试人", "S2", 4000, rel_text="")
            assert "潜水" not in b0, "包外事实泄漏进常驻底档"
            # 本轮聊到「潜水」：统计轨给出 topic_ids=[潜水那条 id=7]，触发增量
            b1, _p1, _r1, m1 = f2._frozen_block(
                "测试人", "S2", 4000, rel_text="",
                topic_ids=[7], topic_signal=True,
                topic_query="你潜水的时候害怕吗",
                recent_queries=["你潜水的时候害怕吗"])
            print(f"[6] 基线 fp={m0['fp']} → 话题命中 fp={m1['fp']} reason={m1['reason']}")
            assert m1["reason"] == "fp_changed", "话题刷新没触发"

            def _l2(block):
                return [ln for ln in block.splitlines() if "L2|" in ln]

            l0, l1 = _l2(b0), _l2(b1)
            n = len(l0)
            assert n == 6, "L2 条数异常"
            pfx = max(1, f2._cfg("pack_l2_limit", 6) - f2._cfg("inject_topic_extra", 3))
            # 关键缓存语义：话题刷新只动【尾部】事实，前 pfx 条逐字节不变 →
            # provider 前缀缓存（占 token 大头）整段命中，只重算尾部槽位。
            assert l0[:pfx] == l1[:pfx], "话题刷新改动了前缀事实，前缀缓存白废"
            # 话题事实进了尾部（最后 tail_k 条），不在前缀
            tail = " ".join(l1[pfx:])
            assert "潜水" in tail and "潜水" not in " ".join(l0[pfx:]), \
                "话题事实没补进尾部"
            assert b1 != b0, "话题刷新没产生任何字节变化"
            print(f"    ✓ 话题命中=只换尾部事实，前 {pfx} 条前缀逐字节不变 → 前缀缓存命中率最大化")
        finally:
            if store2 is not None:
                store2._db.close()
            shutil.rmtree(tmp2, ignore_errors=True)

        print("\nALL CHECKS PASSED")
    finally:
        if store is not None:
            store._db.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
