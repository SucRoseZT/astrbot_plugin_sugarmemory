"""写入门控回归测试：字面兜底不许误伤正常陈述。

「听说」是最典型的坑：中文没有词边界，「我只听说唱」会被切成
「听说」+「唱」，旧规则把整句判成转述、降级进 L3（14 天后就没了），
长期记忆永远收不到这条口味。这里把正反例都钉住。
"""
import sys

sys.path.insert(0, ".")

from sugarmemory_core.gate import WriteGate  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  x {name}   {detail}")


gate = WriteGate()


def verdict(text, speech_act="statement"):
    return gate.judge({
        "text": text,
        "source_type": "user_message",
        "speaker": "user",
        "speech_act": speech_act,
    })["decision"]


def allow(text, name):
    """正常事实：必须 allow。"""
    d = verdict(text)
    check(name, d == "allow", f"{text} -> {d}")


def quoted(text, name):
    """转述：必须 transform（降到 L3）。"""
    d = verdict(text)
    check(name, d == "transform", f"{text} -> {d}")


print("===== 「听 + 说X」不许当转述 =====")
allow("我只听说唱", "只听说唱")
allow("平时听听说唱，也听摇滚", "听听说唱")
allow("喜欢听相声", "听相声")
allow("最近在听播客", "听播客")
allow("只听说唱和嘻哈", "听说唱和嘻哈")
allow("爱听书，一周一本", "听书")

print("\n===== 真转述还得拦住 =====")
quoted("他说他下周要去上海", "他说…")
quoted("她说不想干了", "她说…")
quoted("听朋友说这家店不错", "听朋友说…")
quoted("听说他考上研了", "听说他…")
quoted("我听说你换工作了", "听说你…")
quoted("据说这个月底会涨价", "据说…")
quoted("网上说这家店很坑", "网上说…")
quoted("有人觉得我太内向", "有人觉得…")
quoted("假如我是他，我就不去了", "假如我是…")

print("\n===== speech_act 结构化字段仍然优先 =====")
d = verdict("我上周去了杭州", speech_act="quote")
check("LLM 标了 quote 就降级", d == "transform", d)
d = verdict("我上周去了杭州", speech_act="statement")
check("正常陈述放行", d == "allow", d)

print(f"\n===== 结果：{PASS} 过 / {FAIL} 挂 =====")
sys.exit(1 if FAIL else 0)
