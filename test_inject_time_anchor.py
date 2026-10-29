# -*- coding: utf-8 -*-
"""注入块的时间锚 + 「用户是不是在问这条」的判定。

两条独立的约束：

1. 时间锚必须是绝对时间。相对写法（「·约5分钟前」）依赖 now，每跨一个
   分钟桶就换字面量；而块在 system_prompt 尾部、后面跟着整段历史，模型缓存
   是纯前缀匹配，改一个字后面全部重算（实测 309 次调用 127 次 cached=0）。

2. 「要不要把话挑明」看的是用户这一句，不是记录多旧。曾经用时间窗口做闸门
   （600s，后来放宽到 2h），等于规定用户什么时候才能问——实测窗口外真实模型
   1/3 会答「我什么都没收到呀」。现在窗口整个删掉，判据换成
   pack._asks_about_message(query)：隔三分钟问和隔三天问，同一句话同样触发。
"""
import time

import sugarmemory_core.injector as injector
from sugarmemory_core.injector import build_block, _abs_when
from sugarmemory_core.pack import _asks_about_message

NOW = time.time()
T_2MIN = NOW - 120
T_3H = NOW - 3 * 3600
T_9D = NOW - 9 * 86400


def _stamp(ts: float) -> str:
    lt = time.localtime(ts)
    return (f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"
            f" {lt.tm_hour:02d}:{lt.tm_min:02d}")


def _pack(outbound, probe=False, **kw):
    p = {"summary": "", "l2": [], "l3": [], "outbound": outbound,
         "cross": [], "excluded": [], "trim_notes": [],
         "probe_outbound": probe}
    p.update(kw)
    return p


def _row(ts, content="[你转述给用户 10002] 喵喵喵｜起因：带个话", nick="小刚"):
    return {"content": content, "ts": ts, "session_id": "111:FriendMessage:10002",
            "here": True, "peer_nick": nick}


# ---------- 1. 锚必须存在，且是绝对时间 ----------

def test_row_gets_abs_anchor():
    block = build_block(_pack([_row(T_2MIN)]))
    assert f"[你转述给当前用户（小刚·{_stamp(T_2MIN)}）]" in block


def test_row_without_nick_still_anchored():
    block = build_block(_pack([_row(NOW - 30, nick="")]))
    assert f"[你转述给当前用户·{_stamp(NOW - 30)}]" in block


def test_scheduled_task_row_keeps_task_name_and_abs_anchor():
    row = _row(T_2MIN,
               content="[你按定时任务「跨对话测试2」发给用户 10001] 喵~｜起因：建个测试")
    block = build_block(_pack([row]))
    assert f"[你按定时任务「跨对话测试2」发给当前用户（小刚·{_stamp(T_2MIN)}）]" in block
    assert "起因：建个测试" in block


def test_stale_row_still_anchored():
    """旧实现的缺陷：超过 24 小时直接返回空串，9 天前的记录干脆没有时间。
    绝对时间没有这个问题——越老越该知道它老。"""
    block = build_block(_pack([_row(T_9D)]))
    assert f"小刚·{_stamp(T_9D)}" in block


def test_no_ts_no_anchor():
    block = build_block(_pack([_row(0)]))
    assert "[你转述给当前用户（小刚）]" in block


# ---------- 2. 挑明与否只看「用户这句话」，与时间无关 ----------

def test_probe_adds_nudge_regardless_of_age():
    """同一个 probe，记录是 2 分钟前还是 9 天前，都要挑明。
    这正是「不规定用户」：什么时候问都算数。"""
    for ts in (T_2MIN, T_3H, T_9D):
        block = build_block(_pack([_row(ts)], probe=True))
        assert "先承认你发过" in block, f"ts={ts} 该挑明却没挑明"


def test_no_probe_keeps_block_quiet():
    """用户这句话不是在问消息（比如「今天好热」），就别硬扯到定时任务上。"""
    for ts in (T_2MIN, T_3H):
        block = build_block(_pack([_row(ts)], probe=False))
        assert "先承认你发过" not in block
        assert "绝不要再说" not in block


def test_probe_false_when_no_outbound():
    """没有外发记录时无从挑明——提示语里「下面列的才是权威记录」会指向空气。"""
    block = build_block(_pack([], probe=True))
    assert "先承认你发过" not in block


# ---------- 3. 判定本身：这句话是不是在问一条消息 ----------

def test_detects_vague_questions():
    for q in ("这是什么？", "这是啥", "什么意思", "啥意思啊", "啊？", "？？？",
              "唉？", "你发的？", "刚才那条是什么", "怎么回事", "没看懂"):
        assert _asks_about_message(q), f"应判为在问：{q}"


def test_ignores_normal_chat():
    for q in ("谢谢你呀", "好的，我知道啦", "今天好热", "我去写代码了",
              "帮我看看这个插件为什么报错", ""):
        assert not _asks_about_message(q), f"不该判为在问：{q}"


def test_long_message_not_treated_as_probe():
    """长句子多半在讲正事，「这是什么原理」这种嵌在长论述里的不算指代。"""
    long_q = "我在调试那个定时任务的插件，它每次都会把消息发到别的会话去，" \
             "然后我这边收不到回调，这是什么原理啊，你能帮我分析一下吗"
    assert len(long_q) > 30
    assert not _asks_about_message(long_q)


# ---------- 4. 整块完全不随时间变化（缓存命中的前提） ----------

def test_block_identical_at_any_time():
    """build_block 现在一次都不读 now：推 5 分钟、推 3 天，结果逐字节相同。
    这是「前缀缓存只在真新增记录时才断」的保证。"""
    pack = _pack([_row(T_2MIN)], probe=True)
    pack["summary"] = "他在调试定时投递"
    pack["l2"] = [{"subject": "小明", "predicate": "在测", "value": "定时投递",
                   "confidence": 0.8, "source_msg_id": 1144}]

    base = build_block(pack)
    real = injector.time.time
    try:
        for offset in (300, 3600, 86400 * 3):
            injector.time.time = lambda o=offset: NOW + o
            assert build_block(pack) == base, f"now+{offset}s 时块变了"
    finally:
        injector.time.time = real


def test_no_drifting_time_words():
    for ts in (NOW - 30, T_2MIN, T_3H, T_9D):
        block = build_block(_pack([_row(ts)], probe=True))
        for drifting in ("分钟前", "小时前", "1分钟内", "约1分", "刚刚"):
            assert drifting not in block, f"ts={ts} 仍含会漂移的词「{drifting}」"


def test_future_ts_does_not_crash():
    block = build_block(_pack([_row(NOW + 60)]))
    assert _stamp(NOW + 60) in block


def test_abs_when_direct():
    assert _abs_when(0) == ""
    assert _abs_when(-1) == ""
    assert _abs_when(T_2MIN) == f"·{_stamp(T_2MIN)}"
    assert _abs_when(T_2MIN) == _abs_when(T_2MIN)
