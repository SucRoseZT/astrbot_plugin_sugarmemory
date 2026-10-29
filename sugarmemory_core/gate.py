"""写入门控：纯逻辑、可单测、零 LLM 调用。
输入候选断言（结构化字段+上下文元数据），输出 allow / deny / transform 与原因。
规则优先信任结构化字段（speech_act / source_type / speaker），
正则只作为 Bot 自述与临时状态的兜底安全网。
"""
import re

# Bot 自述兜底：意图、工具尝试、行为描述
BOT_SELF_RE = re.compile(
    r"(我|本?(鱼|bot|助手|模型))[^。！？]{0,12}(正在?|刚才|刚刚)?[^。！？]{0,8}"
    r"(调用|尝试|执行|发送|回复|触发|拦截|冷却|记忆|工具|验证|检查)|"
    r"我(会|要|打算|准备|正在|刚刚)(去|来)?(发|回|说|做|调用|尝试|执行)"
)
# 临时状态
TEMP_STATE_RE = re.compile(
    r"(今天|现在|此刻|这会儿|刚才|最近)?(好累|太累|好困|好饿|好渴|好烦|好开心|好难过|"
    r"很累|很困|很饿|疲惫|困了|饿了|渴了|无聊|累了|有点累|困死|累死)"
)
# 未兑现打算
PLAN_RE = re.compile(r"(打算|准备|计划|想要|想去|下次|以后|回头|改天)(去|做|学|买|玩|写|搞|看|试)")
# 转述/引用/角色扮演/假设
QUOTE_RE = re.compile(r"(他说|她说|他们说|有人(说|觉得)|网上(说|讲)|听说|据说|引用|扮演|假设|如果我是|比如说|举个例子)")

ALLOW = "allow"
DENY = "deny"
TRANSFORM = "transform"


class Verdict(dict):
    @property
    def decision(self) -> str:
        return self["decision"]

    @property
    def reason(self) -> str:
        return self["reason"]


class WriteGate:
    """judge(candidate) -> Verdict
    candidate 结构化字段：
      text: str                  候选内容
      source_type: user_message | system_event | bot_utterance | unknown
      speaker: user | bot | other | unknown
      speech_act: statement | quote | roleplay | hypothesis | temp_state | plan | lyric | unknown
      subject_confirmed: bool    说话主体能否确认（默认 speaker=='user' 时为 True）
      quote_of: str              被转述的原话人（可空）
    """

    def judge(self, cand: dict) -> Verdict:
        text = str(cand.get("text", "") or "")
        source_type = str(cand.get("source_type", "unknown") or "unknown")
        speaker = str(cand.get("speaker", "unknown") or "unknown")
        speech_act = str(cand.get("speech_act", "unknown") or "unknown")
        subject_confirmed = bool(cand.get("subject_confirmed", speaker == "user"))

        def deny(reason: str) -> Verdict:
            return Verdict({"decision": DENY, "reason": reason, "target_layer": ""})

        def to_l3(reason: str) -> Verdict:
            return Verdict({"decision": TRANSFORM, "reason": reason, "target_layer": "L3"})

        # 1) Bot 来源 / 自述：一律不得进 L2（L4 落盘禁令的延伸）
        if source_type == "bot_utterance" or speaker == "bot":
            return deny("来源为 Bot 自述：意图/行为描述不落 L2，置信度归零")
        if BOT_SELF_RE.search(text):
            return deny("命中 Bot 自述特征（工具尝试/意图/行为描述），拒绝写入 L2")

        # 2) 系统结构化事件：允许（工具执行成功等）
        if source_type == "system_event":
            return Verdict({"decision": ALLOW, "reason": "系统结构化事件", "target_layer": "L2"})

        # 3) 显式标记的转述/引用/角色扮演/假设/歌词：降级 L3
        if speech_act in ("quote", "roleplay", "hypothesis", "lyric") or QUOTE_RE.search(text):
            return to_l3("转述/引用/角色扮演/假设内容只作事件日志，不构成用户事实")

        # 4) 临时状态：降级 L3
        if speech_act == "temp_state" or TEMP_STATE_RE.search(text):
            return to_l3("临时状态（情绪/体感），只入 L3 事件日志")

        # 5) 未兑现打算：降级 L3（不得写成已具备事实）
        if speech_act == "plan" or PLAN_RE.search(text):
            return to_l3("未兑现的打算/计划，只入 L3 事件日志")

        # 6) 归属不可确认：拒绝
        if speaker not in ("user", "system") or not subject_confirmed:
            return deny("说话主体不是用户本人或无法确认归属，拒绝写入 L2")

        # 7) 用户显式陈述：允许
        return Verdict({"decision": ALLOW, "reason": "用户显式陈述", "target_layer": "L2"})
