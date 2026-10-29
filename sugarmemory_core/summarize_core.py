"""阶段总结核心：解析一次 LLM 调用的 JSON 输出，走门控落库。
本模块零 LLM 调用（调用方只发一次请求，这里只做解析+门控+存储）。
"""
import json
import re

from .gate import WriteGate
from .social import normalize_relation


def parse_summary_json(content: str) -> dict | None:
    """健壮解析：剥 <think>、剥围栏、截断修复、中文标点兜底。"""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*", text, re.S)
    if not m:
        return None
    frag = m.group(0)

    def repair(s: str) -> str:
        stack: list[str] = []
        in_str = esc = False
        for ch in s:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
        if in_str:
            s += '"'
        s = s.rstrip()
        if s.endswith(","):
            s = s[:-1]
        elif s.endswith(":"):
            s += " null"
        return s + "".join(reversed(stack))

    for cand in (frag, repair(frag), repair(frag).replace("，", ",").replace("：", ":")):
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


SUMMARY_SYSTEM_PROMPT = (
    "你是记忆整理器。根据最近的对话事件日志，输出严格的 JSON（不要任何其他文字）：\n"
    '{"summary": "一段 100 字以内的会话摘要", "candidates": ['
    '{"subject": "<说话人的名字>", "predicate": "<从属性清单里选一个>", "value": "内容", '
    '"speaker": "user", "speech_act": "statement|quote|roleplay|hypothesis|temp_state|plan", '
    '"source_quote": "支撑该断言的对话原话", "confidence": 0.0-1.0}], '
    '"relations": [{"source": "<说话人的名字>", "target": "<被提到的人的名字>", '
    '"relation": "认识|朋友|亲友|恋人", "quote": "对话原话", "confidence": 0.0-1.0}]}\n'
    "主体规则（subject，这一栏最容易写错，必须照做）：\n"
    "- 日志每行有两种开头：`名字: 内容` 表示这个人说的，`[你] 内容` 表示 AI 自己说的；\n"
    "- subject 必须原样照抄 `名字:` 里的那个名字，一个字符都不许改：不翻译、不简化、不换成中文；\n"
    "- 严禁写「用户」「user」「我」「他」「她」「对方」「本人」这类泛称当 subject；\n"
    "- `[你]` 开头的行是 AI 自己的话，不能当 subject，也不能拿来当断言依据；\n"
    "- 只有当断言讲的是对话里提到的第三个人时（比如「A 说 B 是他女朋友」里的 B），"
    "subject 才填那个第三人；\n"
    "- 拿不准是谁说的，就丢掉这条候选，别猜。\n"
    "事实规则：\n"
    "- 只从对话里提取说话人亲口陈述的事实/偏好；转述、角色扮演、玩笑、临时情绪一律标对应 speech_act；\n"
    "- 绝不要把 AI 自己说过的话写成断言；\n"
    "- 没有可提取的就给空数组；宁缺毋滥。\n"
    "属性规则（predicate，必须从下面清单里原样选一个，严禁自己造词）：\n"
    "- 爱好技能 / 生活习惯 / 经历 / 打算 / 评价认为 / 自我相关 / 其他\n"
    "- 一条 candidate 只写一件独立的事，别把多件事用分号塞进同一条；\n"
    "- 同一个属性下有多个独立事实（比如「被诊断双向」「被霸凌休学」是两件事），就拆成多条 candidate，每条单独一行；\n"
    "- 同一个人的同一个属性只保留一条、新的覆盖旧的，所以只写长期成立的东西；\n"
    "- 「喜欢听歌」「喜欢看花」「爱打音游」「会写代码」「会编曲」一律归 爱好技能（技能和爱好不分开）；\n"
    "- 「喜欢几点睡觉」「不喜欢吃泡面」「熬夜」这类作息/口味/日常规律，一律归 生活习惯；\n"
    "- 已经发生的各种经历（休学、被霸凌、送过礼、去过哪）一律归 经历；\n"
    "- 未来想做的事、计划、目标一律归 打算；\n"
    "- 对某人/某物/某事的看法、评价（自己的或别人的）一律归 评价认为，这只是主观看法，不是事实；\n"
    "- 年龄、身高、体重、籍贯、星座、身体状况这类客观事实一律归 自我相关；\n"
    "- 两个人之间的关系（谁是谁的女友/朋友）不要写进 candidates，放到 relations 里；\n"
    "- 实在归不进去才写 其他。\n"
    "关系规则（relations）：\n"
    "- 只记「对话里明确说出来的、两个真人之间的关系」，比如「A 说 B 是我女朋友」→ source=A, target=B；\n"
    "- relation 只能从 认识 / 朋友 / 亲友 / 恋人 里选一个，拿不准就不要输出这条；\n"
    "- source 和 target 必须是两个不同的真人，且名字必须跟日志里出现的名字一字不差；\n"
    "- 不要把 AI/机器人/自己写进 source 或 target；\n"
    "- 没有就空数组，不要猜。"
)


GENERIC_SUBJECTS = {
    "用户", "用户自己", "使用者", "本人", "我", "他", "她", "对方", "这人", "该用户",
    "user", "the user", "ta", "ai", "机器人", "机器人自己", "bot", "assistant", "narrator",
}


def _normalize_subject(raw, fallback: str) -> str:
    """把模型写的泛称主体拉回真实名字。

    日志每行本来就是 `名字: 内容`，但模型见到示例里的「用户」就会照抄。
    落库后 subject 对不上，注入时 PackBuilder 按 subject 过滤会直接查不到，
    等于这条记忆白存。所以泛称一律换成这次总结的说话人。
    """
    name = str(raw or "").strip()
    fb = str(fallback or "").strip()
    if name and name not in GENERIC_SUBJECTS and name.lower() not in GENERIC_SUBJECTS:
        return name[:40]
    if fb:
        return fb[:40]
    return (name or "用户")[:40]


PREDICATE_VOCAB = (
    "爱好技能", "生活习惯", "经历", "打算", "评价认为", "自我相关", "其他",
)

# 多值分类：同一个人名下可以有多条（不同的事/爱好/经历），不做单值合并。
# 自我相关（被诊断、被霸凌、家人不给治疗是多个独立事件）、评价认为（对不同人/
# 不同事的看法各不相同）同样允许多条并存，别硬拼成一大坨。
_MULTI_VALUED = {"爱好技能", "生活习惯", "经历", "打算", "评价认为", "自我相关", "其他"}


def is_single_valued(predicate: str) -> bool:
    """多值分类允许多条并存，其余单值合并。"""
    return (predicate or "").strip() not in _MULTI_VALUED


# 同义造词 → 规范词。顺序即优先级：带强信号词的排在宽泛词之前，
# 否则「评价认为」里的「觉得」会抢走「自我相关」里的「年龄」这类客观信息。
_PREDICATE_RULES = (
    # 注意顺序：靠前优先命中。
    # 客观事实（自我相关）的信号词要排前面，避免被「评价/认为」抢走。
    (("年龄", "几岁", "岁", "身高", "体重", "体型", "籍贯", "星座", "血型", "生日",
      "出生", "生肖", "职业", "工作", "学校", "专业", "学历", "毕业", "公司", "单位",
      "岗位", "身份", "外貌", "长相", "形象", "模样", "颜值", "性格", "脾气",
      "内向", "外向", "社恐", "社牛", "为人"), "自我相关"),
    (("健康", "身体", "症状", "病", "疼", "痛", "过敏", "受伤", "生病", "住院", "吃药",
      "状况", "双向", "抑郁", "焦虑", "失眠"), "自我相关"),
    # 评价/认为（主观看法）排在客观事实之后
    (("评价", "看法", "态度", "觉得", "认为", "印象", "口碑", "意见", "评价为",
      "觉得别人", "觉得周围", "怎么看"), "评价认为"),
    # 打算（未来）
    (("计划", "打算", "准备", "待办", "todo", "想做", "想学", "目标", "规划",
      "将要", "以后", "准备做", "准备改"), "打算"),
    # 生活习惯（作息/口味/日常规律）
    (("习惯", "作息", "日常", "规律", "熬夜", "起床", "睡眠", "生物钟", "生活习惯",
      "作息时间", "晚睡", "早起", "口味", "不吃", "不爱吃", "爱吃", "饮食", "吃饭"), "生活习惯"),
    # 爱好技能（喜欢什么 + 会什么，不分开）
    (("爱好", "喜欢", "讨厌", "偏好", "兴趣", "迷恋", "反感", "喜好", "爱打", "爱听",
      "爱看", "擅长", "技能", "技术", "能力", "会写", "会做", "会编", "会弹", "水平",
      "编程", "代码", "动手", "技术能力", "技术行为", "技术活动"), "爱好技能"),
    # 经历（已经发生的过去）
    (("经历", "曾经", "以前", "过去", "小时候", "休学", "霸凌", "送过", "去过",
      "做过", "完成", "搞定", "修复", "开发", "改造", "项目", "成果", "作品"), "经历"),
    # 兜底：称呼/自称等先归「其他」的强信号，再让「其他」兜底
    (("称呼", "昵称", "自称", "外号", "笔名", "小名", "绰号", "怎么叫"), "其他"),
    (("设备", "电脑", "手机", "配置", "显卡", "系统", "型号", "主机", "显示器",
      "键盘", "鼠标", "token", "用量"), "其他"),
)


def _normalize_predicate(raw) -> str:
    """把模型自造的属性名拉回受控词表。

    predicate 是 L2 去重的三要素之一（subject|predicate|value 一起算指纹），
    模型每次总结都现编一个新词，「技术能力」「技术行为」「技术活动」就会被
    当成三件不同的事各存一条，同一条事实永远不会被认出是重复。
    归不进词表时归到「其他」兜底，保证面板里永远只有受控的 7 个分类。
    """
    p = str(raw or "").strip()
    if not p:
        return "其他"
    if p in PREDICATE_VOCAB:
        return p
    low = p.lower()
    for keys, canon in _PREDICATE_RULES:
        for k in keys:
            if k in low:
                return canon
    return "其他"


def reclassify_value(value: str, current_predicate: str = "") -> str:
    """按 value 内容重新判断分类（用于清洗旧数据 + 修正「其他」里被塞错的条目）。

    旧数据里大量条目因为谓词被写成「其他」或自造词，value 明明属于某类却没归对。
    这里不看旧谓词标签，只看 value 语义骨架，按优先级归类：

    1. 客观事实（年龄/身高/身体状况/身份）→ 自我相关
    2. 未来打算 → 打算
    3. 主观评价（觉得/认为/看法，明确的观点判断）→ 评价认为
    4. 作息/口味/习惯 → 生活习惯
    5. 过去的经历 → 经历
    6. 爱好/技能 → 爱好技能
    7. 其余 → 其他

    判断靠关键词启发式（纯本地，无 LLM），宁可归「其他」也不误判。
    注意「喜欢/爱/讨厌」这类情感词既是爱好又是口味，不放进「评价认为」——
    评价认为只认「觉得/认为/看法/评价/改观」这类明确的观点判断词。
    """
    v = str(value or "")
    # 客观事实：年龄/身高/体重/籍贯/职业/身份/身体
    if re.search(r"(\d+\s*岁|年龄|身高|体重|籍贯|星座|血型|生日|出生|生肖|职业|"
                 r"学校|专业|学历|毕业|公司|单位|岗位|身份|双向|双相|抑郁|焦虑|"
                 r"失眠|身体|病情|休学|在读|在读生|患者|唯物主义者|留学)", v):
        return "自我相关"
    # 生活习惯（作息/口味/饮食）——「吃/喝」这类具体行为先于泛化的「喜欢」判断
    if re.search(r"(熬夜|起床|睡觉|睡眠|作息|生物钟|吃|喝|口味|泡面|饮食|"
                 r"点东西吃|晚睡|早起|不睡|心情不好|茶|咖啡|饮料|零食)", v):
        return "生活习惯"
    # 经历（明确的完成时动作：开发了/修好/做了/写过）
    if re.search(r"(开发了|修好了|修好|修改了|做了|写过|做过|搞定|完成|改造了|"
                 r"曾经|以前|过去|小时候|送过|去过|被霸凌|五年前|前任|初恋|"
                 r"靠.*挣钱|长期.*放不下|疑似.*倾向|给机器人写)", v):
        return "经历"
    # 主观评价：明确的观点判断词 + 对某人的情感态度
    if re.search(r"(觉得|认为|看法|评价|印象|改观|不屑|看不起|看不上|"
                 r"对.*表达.*(爱|心疼|好感|喜欢|不满)|觉得周围|觉得别人|"
                 r"希望.*能|大部分.*只是)", v):
        return "评价认为"
    # 未来打算
    if re.search(r"(打算|计划|准备|想学|想做|目标|规划|将要|以后|准备做|准备改|"
                 r"待办|todo|暂缓|明年|打算开源|准备上学)", v):
        return "打算"
    # 爱好技能（会/擅长/做音乐这类能力信号）
    if re.search(r"(会写|会做|会用|会编|会弹|擅长|技能|爱听|爱看|爱打|爱玩|"
                 r"音游|摇滚|重金属|做音乐|做电音|编曲|画画|画稿|写.*插件|"
                 r"改.*代码|形象叫|网络形象|乐天派|喜欢|爱好|兴趣|迷恋|反感|讨厌|偏好)", v):
        return "爱好技能"
    return "其他"


def apply_summary(store, gate: WriteGate, parsed: dict, source_msg_id: str = "",
                  subject: str = "", session_id: str = "") -> dict:
    """把一次总结的产出（已解析 JSON）经门控落库。绝不调用 LLM。
    总结行会带上归属用户/会话，长期记忆通过 source_summary_id 反向指回这次总结。"""
    summary = str(parsed.get("summary", "") or "").strip()
    stats = {"summary_stored": 0, "l2": 0, "l3": 0, "deny": 0, "summary_id": 0,
             "relations": 0}
    summary_id = 0
    if summary:
        summary_id = store.write_l3(
            event_type="session_summary", subject=subject, content=summary[:500],
            source_msg_id=source_msg_id, session_id=session_id,
        )
        stats["summary_stored"] = 1
        stats["summary_id"] = summary_id
    for cand in parsed.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        text = str(cand.get("value", "") or cand.get("source_quote", "") or "").strip()
        if not text:
            continue
        cand_subject = _normalize_subject(cand.get("subject"), subject)
        payload = {
            "text": text,
            "source_type": "user_message" if str(cand.get("speaker", "user")) == "user" else "unknown",
            "speaker": str(cand.get("speaker", "user") or "user"),
            "speech_act": str(cand.get("speech_act", "statement") or "statement"),
        }
        verdict = gate.judge(payload)
        store.write_audit(
            candidate=text[:200], decision=verdict.decision, reason=verdict.reason,
            source_type=payload["source_type"], speech_act=payload["speech_act"],
            target_layer=str(verdict.get("target_layer", "")),
        )
        if verdict.decision == "allow":
            pred = _normalize_predicate(cand.get("predicate"))
            store.write_l2(
                subject=cand_subject,
                predicate=pred,
                value=text[:200],
                speaker=payload["speaker"],
                speech_act=payload["speech_act"],
                source_type=payload["source_type"],
                source_msg_id=source_msg_id,
                source_quote=str(cand.get("source_quote", "") or "")[:200],
                source_summary_id=summary_id,
                confidence=float(cand.get("confidence", 0.6) or 0.6),
                gate_verdict={"decision": "allow", "reason": verdict.reason},
                gate_reason=verdict.reason,
                single_valued=is_single_valued(pred),
            )
            stats["l2"] += 1
        elif verdict.decision == "transform":
            store.write_l3(
                event_type="demoted_candidate", subject=cand_subject,
                content=f"[{verdict.reason}] {text}"[:300],
                source_msg_id=source_msg_id, session_id=session_id, summary_id=summary_id,
            )
            stats["l3"] += 1
        else:
            stats["deny"] += 1
    for rel in parsed.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        src = _normalize_subject(rel.get("source"), subject)
        dst = _normalize_subject(rel.get("target"), subject)
        rank = normalize_relation(rel.get("relation"))
        if not src or not dst or not rank or src == dst:
            continue
        try:
            conf = float(rel.get("confidence", 0.6) or 0.6)
        except (TypeError, ValueError):
            conf = 0.6
        res = store.relation_upsert(
            source=src, target=dst, relation=rank,
            quote=str(rel.get("quote", "") or ""),
            confidence=conf, source_type="summary",
            session_id=session_id, summary_id=summary_id,
        )
        if res.get("ok"):
            stats["relations"] += 1
    return stats
