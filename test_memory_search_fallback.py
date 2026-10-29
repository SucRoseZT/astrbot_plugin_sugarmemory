# -*- coding: utf-8 -*-
"""验证 memory_search 的「最近其他会话来信」兜底：关键词查不到 小明 的消息时,
兜底段也要给出。"""
import asyncio
import sys
import time
import types

astrbot = types.ModuleType("astrbot")
astrbot_api = types.ModuleType("astrbot.api")
event_mod = types.ModuleType("astrbot.api.event")
star_mod = types.ModuleType("astrbot.api.star")


class _Filter:
    class EventMessageType:
        ALL = "ALL"
        GROUP_MESSAGE = "GROUP_MESSAGE"
        PRIVATE_MESSAGE = "PRIVATE_MESSAGE"
        OTHER_MESSAGE = "OTHER_MESSAGE"

    def llm_tool(self, name=None, **kw):
        def deco(f):
            return f
        return deco

    def on_llm_request(self, priority=0):
        def deco(f):
            return f
        return deco

    def __getattr__(self, name):
        def deco(*a, **kw):
            if len(a) == 1 and callable(a[0]) and not kw:
                return a[0]
            def inner(f):
                return f
            return inner
        return deco


class _Logger:
    def info(self, *a):
        pass

    def warning(self, *a):
        pass

    def error(self, *a):
        print("[ERROR]", *a)

    def debug(self, *a):
        pass


class AstrBotConfig(dict):
    pass


filter = _Filter()
event_mod.filter = filter
event_mod.AstrMessageEvent = object
event_mod.MessageEventResult = object
event_mod.MessageChain = object
star_mod.Context = object
star_mod.Star = object
star_mod.register = lambda *a, **kw: (lambda cls: cls)
comp_mod = types.ModuleType("astrbot.api.message_components")
comp_mod.Plain = object
comp_mod.Image = object
prov_mod = types.ModuleType("astrbot.api.provider")
prov_mod.ProviderRequest = object
astrbot_api.provider = prov_mod
sys.modules["astrbot.api.provider"] = prov_mod
astrbot_api.event = event_mod
astrbot_api.star = star_mod
astrbot_api.AstrBotConfig = AstrBotConfig
astrbot_api.logger = _Logger()
astrbot_api.message_components = comp_mod
astrbot.api = astrbot_api
core = types.ModuleType("astrbot.core")
platform_mod = types.ModuleType("astrbot.core.platform")
msgtype_mod = types.ModuleType("astrbot.core.platform.message_type")
msgtype_mod.MessageType = types.SimpleNamespace(
    GROUP_MESSAGE="G", FRIEND_MESSAGE="F", OTHER_MESSAGE="O")
platform_mod.message_type = msgtype_mod
sess_mod = types.ModuleType("astrbot.core.platform.message_session")
sess_mod.MessageSession = object
platform_mod.message_session = sess_mod
core.platform = platform_mod
core.__path__ = []
platform_mod.__path__ = []
sys.modules.update({
    "astrbot": astrbot, "astrbot.api": astrbot_api,
    "astrbot.api.event": event_mod, "astrbot.api.star": star_mod,
    "astrbot.api.message_components": comp_mod,
    "astrbot.core": core, "astrbot.core.platform": platform_mod,
    "astrbot.core.platform.message_type": msgtype_mod,
    "astrbot.core.platform.message_session": sess_mod,
})

import importlib.util
spec = importlib.util.spec_from_file_location(
    "sm_main",
    r"C:/Users/Administrator/.astrbot_launcher/instances/1d216365-92e9-4fc5-a411-3d8aa5ecc6c8/core/data/plugins/astrbot_plugin_sugarmemory/main.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PluginCls = mod.SugarMemory

plugin = object.__new__(PluginCls)
plugin.config = AstrBotConfig({"tool_memory_search": True})

NOW = time.time()

class FakeStore:
    def query_active_l2(self, limit=400):
        return []

    def query_recent_l3(self, limit=40, event_type=None, **kw):
        if event_type == "user_message":
            return [
                {"id": 837, "event_type": "user_message",
                 "content": "[私] 小刚: 刚刚有没有人给你发消息",
                 "session_id": "111:FriendMessage:10002",
                 "subject": "小刚", "created_at": NOW - 300},
                {"id": 834, "event_type": "user_message",
                 "content": "[私] 小明: 你好",
                 "session_id": "111:FriendMessage:10001",
                 "subject": "小明", "created_at": NOW - 170},
                {"id": 833, "event_type": "user_message",
                 "content": "[私] 小明: 请求添加你为好友",
                 "session_id": "111:FriendMessage:10001",
                 "subject": "小明", "created_at": NOW - 175},
            ]
        if event_type == "bot_reply":
            return []
        if event_type == "bot_outbound":
            return []
        return []

plugin.store = FakeStore()
plugin._record_tokens = lambda *a, **kw: None

class FakeEvent:
    unified_msg_origin = "111:FriendMessage:10002"

failures = []

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", detail)
    if not cond:
        failures.append(name)

async def run():
    # 场景 1：主人问「有没有人给你发消息」→ AI 用「发消息」当关键词 →
    # 关键词只命中主人自己的话，但兜底段必须出现 小明 的两条
    out = await plugin.tool_memory_search(
        FakeEvent(), query="发消息", scope="all", limit=8)
    print("--- 输出 ---")
    print(out)
    print("-----------")
    check("兜底: 小明 的你好出现在结果里", "小明: 你好" in out)
    check("兜底: 小明 的申请文本出现在结果里", "请求添加你为好友" in out)
    check("兜底: 标注了其他会话来信", "其他会话来信" in out)

    # 场景 2：30 分钟内其他会话无来信 → 明确说「无」
    global NOW
    # 直接改 store 的时间戳
    async def out2():
        return await plugin.tool_memory_search(
            FakeEvent(), query="存在吗", scope="all", limit=8)
    # 临时把 小明 的行变成 2 小时前
    orig_q = plugin.store.query_recent_l3
    def q2(limit=40, event_type=None, **kw):
        rows = orig_q(limit=limit, event_type=event_type, **kw)
        for r in rows:
            r["created_at"] = NOW - 7200
        return rows
    plugin.store.query_recent_l3 = q2
    out = await out2()
    print("--- 场景2输出 ---")
    print(out)
    check("兜底: 无来信时明说无", "无" in out and "半小时" in out)

asyncio.run(run())
print()
print("FAILED:", failures if failures else "无，全部通过")
sys.exit(1 if failures else 0)
