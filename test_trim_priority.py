"""v0.7.4 trim 优先级修复单测：fresh 对题 cross 必须活过 not-here outbound。"""
import sys, types, importlib.util, asyncio

def _mk(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

# ---- astrbot 桩（与 test_memory_search_fallback 相同模式）----
astrbot = _mk("astrbot"); astrbot.__path__ = []
astrbot_api = _mk("astrbot.api"); astrbot_api.__path__ = []
event_mod = _mk("astrbot.api.event")
star_mod = _mk("astrbot.api.star")
comp_mod = _mk("astrbot.api.message_components")

class AstrBotConfig(dict):
    def __init__(self, d=None):
        super().__init__(d or {})

class _Logger:
    def info(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass

class _Filter:
    def __getattr__(self, name):
        def deco(*a, **k):
            def wrap(f):
                return f
            return wrap if not (a and callable(a[0])) else a[0]
        return deco

filter = _Filter()
event_mod.filter = filter
event_mod.AstrMessageEvent = object
event_mod.MessageEventResult = object
event_mod.MessageChain = object
star_mod.Context = object
star_mod.Star = object
comp_mod.Plain = object
comp_mod.Image = object
astrbot_api.event = event_mod
astrbot_api.star = star_mod
astrbot_api.message_components = comp_mod
astrbot_api.AstrBotConfig = AstrBotConfig
astrbot_api.logger = _Logger()
astrbot.api = astrbot_api
astrbot.core = _mk("astrbot.core"); astrbot.core.__path__ = []
platform_mod = _mk("astrbot.core.platform"); platform_mod.__path__ = []
msgtype_mod = _mk("astrbot.core.platform.message_type")

class MessageType:
    FRIEND_MESSAGE = 1
    GROUP_MESSAGE = 2
    ALL = 0

msgtype_mod.MessageType = MessageType
sess_mod = _mk("astrbot.core.platform.message_session")
sess_mod.MessageSession = object
platform_mod.message_type = msgtype_mod
platform_mod.message_session = sess_mod
sys.modules.update({
    "astrbot": astrbot, "astrbot.api": astrbot_api,
    "astrbot.api.event": event_mod, "astrbot.api.star": star_mod,
    "astrbot.api.message_components": comp_mod,
    "astrbot.core": astrbot.core, "astrbot.core.platform": platform_mod,
    "astrbot.core.platform.message_type": msgtype_mod,
    "astrbot.core.platform.message_session": sess_mod,
})

spec = importlib.util.spec_from_file_location(
    "sm_injector",
    r"C:/Users/Administrator/.astrbot_launcher/instances/1d216365-92e9-4fc5-a411-3d8aa5ecc6c8/core/data/plugins/astrbot_plugin_sugarmemory/sugarmemory_core/injector.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
trim_pack = mod.trim_pack
build_block = mod.build_block

failures = []

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", detail)
    if not cond:
        failures.append(name)

# ---- 场景：600 预算，outbound 4 条 not-here + L3 5 条 + cross 2 条（1 fresh 对题）----
pack = {
    "summary": "这是一个足够长的摘要文本" * 10,  # ~110字
    "l2": [
        {"subject": "小刚", "predicate": f"喜好{i}", "value": f"东西{i}",
         "confidence": 0.8, "status": "active"} for i in range(5)
    ],
    "l3": [
        {"id": 100 + i, "content": f"[私] 小刚: 历史消息内容第{i}条，内容比较长一些用来占token",
         "event_type": "user_message"} for i in range(5)
    ],
    "outbound": [
        {"content": f"[你主动发给用户 10001] 外发消息{i}，较长内容占token用",
         "ts": 1000.0 + i, "session_id": "x", "here": False} for i in range(4)
    ],
    "cross": [
        {"content": "[私] 小明: 请求添加你为好友", "subject": "小明",
         "ts": 2000.0, "_rel": 3, "_fresh": True},
        {"content": "[私] 小明: 同意一下我的好友申请", "subject": "小明",
         "ts": 1900.0, "_rel": 7, "_fresh": True},
    ],
    "relation": [],
}

trimmed = trim_pack(pack, budget_tokens=600, l3_keep_min=2, l2_keep_min=3)
cross_contents = [c.get("content") for c in trimmed.get("cross", [])]
check("fresh对题cross存活", any("请求添加你为好友" in c for c in cross_contents if c),
      str(cross_contents))
ob = trimmed.get("outbound", [])
# outbound 只在预算不够时才让路；本场景 cross 无损失即达标
check("cross无损失时outbound不强求清空",
      (len(ob) < 4) or len(cross_contents) >= 2, f"剩余{len(ob)}条")
check("token预算内", trimmed.get("token_estimate", 9999) <= 600,
      str(trimmed.get("token_estimate")))
notes = trimmed.get("trim_notes", [])
# 新优先级：L3（当前会话原文已在 astrbot 上下文里）最先让位，优先级低于 outbound。
# 所以预算紧张时应先看到「裁剪 L3 事件」，而不是「裁剪外发」。
check("L3先于外发让位", any("L3" in n for n in notes)
      and not any("外发" in n for n in notes), str(notes))

# ---- 场景2：预算极其紧张，cross 至少保 1 条 rel 最高的 ----
pack2 = {
    "summary": "摘要" * 60,
    "l2": [{"subject": "A", "predicate": f"p{i}", "value": f"很长的值" * 5,
            "confidence": 0.9, "status": "active"} for i in range(8)],
    "l3": [{"id": 200 + i, "content": f"[私] 小刚: 消息{i}，" + "内容" * 20,
            "event_type": "user_message"} for i in range(6)],
    "outbound": [{"content": f"[你主动发给用户 555] 外发{i}，" + "占位" * 15,
                  "ts": 100.0 + i, "session_id": "y", "here": False} for i in range(3)],
    "cross": [
        {"content": "[私] 小明: 低相关旧消息", "subject": "小明",
         "ts": 500.0, "_rel": 0, "_fresh": False},
        {"content": "[私] 小明: 同意一下我的好友申请", "subject": "小明",
         "ts": 900.0, "_rel": 7, "_fresh": True},
    ],
    "relation": [],
}
t2 = trim_pack(pack2, budget_tokens=600, l3_keep_min=2, l2_keep_min=3)
c2 = [c.get("content") for c in t2.get("cross", [])]
check("极端预算cross保1条", len(c2) >= 1, str(c2))
if c2:
    check("保的是rel最高那条", "同意一下我的好友申请" in c2[0] or len(c2) > 1, str(c2))

# ---- 场景3：预算充裕时什么都不裁 ----
pack3 = {
    "summary": "短摘要",
    "l2": [{"subject": "A", "predicate": "p", "value": "v", "confidence": 0.9,
            "status": "active"}],
    "l3": [{"id": 1, "content": "一条消息", "event_type": "user_message"}],
    "outbound": [{"content": "[你主动发给用户 555] 一条", "ts": 1.0,
                  "session_id": "y", "here": False}],
    "cross": [{"content": "[私] 小明: 你好", "subject": "小明", "ts": 2.0,
               "_rel": 5, "_fresh": True}],
    "relation": [],
}
t3 = trim_pack(pack3, budget_tokens=600, l3_keep_min=2, l2_keep_min=3)
check("预算充裕不裁", not t3.get("trim_notes"), str(t3.get("trim_notes")))

print()
print("FAILED:", failures if failures else "无，全部通过")
sys.exit(1 if failures else 0)
