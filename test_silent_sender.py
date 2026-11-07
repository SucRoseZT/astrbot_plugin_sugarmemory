"""空文本消息（图片 / 表情 / 语音）也必须认人。

实机背景（2026-11-04）：Jerry/小杰 给机器人发过消息，people 表里却是空壳、
没有 QQ，面板上永远显示「未绑 QQ」。查日志发现那两条消息的 message_str 是
空的（表情包一类），而处理链在 `if not text: return` 就断了——**连
person_ensure 都没走**。他后来能出现在人物库里，是阶段总结从张三那句
「给100010002也打个招呼，他叫Jerry/小杰」里猜出来的，猜出来的当然没 QQ。

不变式：
1. 有 QQ、有昵称的空文本消息 → 建档并绑 QQ；
2. 空文本不写 L3（一堆「[图片]」进正文只会挤占总结预算）；
3. 伪发送者（定时唤醒）、泛称、没有 QQ 的，一律不建档。
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 插件主模块依赖 AstrBot 运行时，它就在实例的 core/ 下（插件目录往上三层）。
# 只能 append：insert(0) 会让 core/main.py 抢占插件自己的 main 模块名。
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from sugarmemory_core.store import SugarStore  # noqa: E402

try:
    import main as sm  # noqa: E402
except Exception as exc:  # 没装 astrbot 等依赖时不算失败
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


class FakeEvent:
    """只实现 _sender_name / _identify_quietly 会用到的三个方法。"""

    def __init__(self, qq, name, text=""):
        self._qq = qq
        self._name = name
        self.message_str = text

    def get_sender_id(self):
        return self._qq

    def get_sender_name(self):
        return self._name

    def get_self_id(self):
        return "10000"


def fresh(tag):
    db = os.path.join(tempfile.gettempdir(), f"sugarmem_silent_{tag}.db")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    return SugarStore(db)


def plugin(store):
    """跳过 __init__：这里只测认人这一段，不需要完整插件上下文。"""
    p = object.__new__(sm.SugarMemory)
    p.store = store
    p._cfg = lambda key, default=None: default
    return p


def l3_count(store):
    return len(store.query_recent_l3(limit=1000))


# ---------- 1. 表情消息也要把 QQ 认下来 ----------
s1 = fresh("emoji")
p1 = plugin(s1)
ev = FakeEvent("100010002", "Jerry")
p1._identify_quietly(ev, "100010002")

person = s1.person_by_qq("100010002")
check("空文本消息也认人", person == "Jerry", person)
check("QQ 绑定成功", s1.person_qq(person) == "100010002" if person else False, person)
check("空文本不写 L3", l3_count(s1) == 0, l3_count(s1))
check("昵称进别名表", bool(s1.person_resolve("Jerry")), "")

# ---------- 2. 同一个人换个昵称，还落回同一个 person ----------
p1._identify_quietly(FakeEvent("100010002", "小郁"), "100010002")
check("换昵称不裂成两个人",
      s1.person_by_qq("100010002") == person and len(s1.person_all()) == 1,
      [x["person"] for x in s1.person_all()])

# ---------- 3. 该拦的一个都不能进 ----------
s2 = fresh("guard")
p2 = plugin(s2)
p2._identify_quietly(FakeEvent("", "Jerry"), "")       # 没 QQ
p2._identify_quietly(FakeEvent("8888", "系统唤醒"), "8888")     # 定时唤醒伪发送者
p2._identify_quietly(FakeEvent("7777", "用户"), "7777")         # 泛称
check("拦截后人物库是空的", s2.person_all() == [], s2.person_all())
check("拦截后也没有记忆", l3_count(s2) == 0, l3_count(s2))

# ---------- 4. 昵称缺失时拿 QQ 号兜底（和正文路径一致） ----------
# 群名片/昵称取不到的时候，event_sender_name 会退回 QQ 号。这条路径也得能建档：
# 有个坏名字总比没有档案强，等真昵称来了 person_by_qq 会认回同一个人。
s3 = fresh("noname")
p3 = plugin(s3)
p3._identify_quietly(FakeEvent("9999", ""), "9999")
check("没昵称也能建档", s3.person_by_qq("9999") != "", s3.person_all())
check("也就一个人", len(s3.person_all()) == 1, s3.person_all())

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
