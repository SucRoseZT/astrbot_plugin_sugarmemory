"""阶段总结核心：解析一次 LLM 调用的 JSON 输出，走门控落库。
本模块零 LLM 调用（调用方只发一次请求，这里只做解析+门控+存储）。
"""
import json
import re
import time

from .gate import WriteGate
from .social import normalize_relation
from .store import PERMANENT_PREDICATES, is_generic_name, is_kinship_alias


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
    '{"summary": "一段 100 字以内的会话摘要（人名只用在场名单上的写法）", "candidates": ['
    '{"subject_id": 1, "predicate": "<从属性清单里选一个>", "value": "内容", '
    '"speaker": "user", "speech_act": "statement|quote|roleplay|hypothesis|temp_state|plan", '
    '"source_quote": "支撑该断言的对话原话", "confidence": 0.0-1.0}], '
    '"relations": [{"source_id": 1, "target_id": 2, '
    '"relation": "认识|朋友|亲友|恋人", "quote": "对话原话", "confidence": 0.0-1.0}], '
    '"updates": [{"id": 12, "action": "supersede", "by": 0}]}\n'
    "名字规则（只约束 summary 那段文字，编号规则管的是字段）：\n"
    "- 摘要里提到在场的人，只用「在场的人」名单上那个名字，一个字都不许改；\n"
    "- 同一个人全程只用一个名字：不许换成他的 QQ 昵称、群名片、拼音写法或外号，"
    "哪怕日志原文里就是这么写的；\n"
    "- 例：名单写着 1=张三，日志里有人喊「Moon」或「M0on」，摘要一律写「张三」；\n"
    "- 日志里出现、但名单上没有的名字，不要搬进摘要，写成「某人」或者干脆不提。\n"
    "编号规则（subject_id / source_id / target_id 都是**数字**，不许写名字）：\n"
    "- 提示里会给一份「在场的人」名单，形如 0=AI  1=张三  2=阿P，编号只认这份名单；\n"
    "- 日志每行以 `[编号]` 开头，那就是这句话的说话人，照抄这个数字就行；\n"
    "- 0 是 AI 自己：不能当 subject，也不能拿来当断言依据；\n"
    "- **名单里没有的人一律不要写**。想不出该填哪个编号，就把这条候选/关系整条丢掉，"
    "不要自己编编号、也不要把名字写进这个字段；\n"
    "- 「弟弟」「我妈」「我老板」这类关系称呼不是人：名单里没有对应编号就丢掉这条；\n"
    "- 只有当断言讲的是名单里的第三个人时（比如「A 说 B 是他女朋友」里的 B），"
    "subject_id 才填 B 的编号；B 不在名单里就丢掉。\n"
    "事实规则：\n"
    "- 只从对话里提取说话人亲口陈述的事实/偏好；转述、角色扮演、玩笑、临时情绪一律标对应 speech_act；\n"
    "- 绝不要把 AI 自己说过的话写成断言；\n"
    "- 没有可提取的就给空数组；宁缺毋滥。\n"
    "- source_quote 必须填用户说这句话时的【完整原句】，不要只截取支撑该断言的一小段；"
    "若一句长话被拆成多条断言，每条的 source_quote 都填那句完整原话"
    "（之后才能按整句原话召回，而不是只剩碎片）。\n"
    "时间与因果规则：\n"
    "- 日志里形如「—— 2026-11-06 ——」的是日期分隔线：它下面每一行都发生在那一天（本地时间）；\n"
    "- 有日期分隔线时，用户用了相对时间（今晚/明早/昨天/前几天/上周/这周末/刚才）的，"
    "value 必须换算成那一天的绝对日期（✗「打算今晚好好睡一觉」→ ✓「打算 2026-11-06 晚上"
    "好好睡一觉」）：你并不知道今天是几号，照抄「今晚」以后会被当成当天说的事；\n"
    "- 日志里没有日期分隔线时，不许自己补任何日期，也不许把「今晚」改成具体某天；\n"
    "- 无论有没有分隔线，用户没说时间就严禁自己加时间词。原话是「到时候」「改天」"
    "「以后有机会」，value 里就照原样含糊（✗ 原话「我到时候好好睡一觉」写成"
    "「打算今晚好好睡一觉」——「今晚」是凭空来的）；\n"
    "- 对话里出现先后变化（「之前做A后来改做B」「以前…现在不…了」「换成/改用」）时，"
    "value 必须保留先后顺序：写「以前做A风格音乐，现已改做B风格」，"
    "严禁压成并列的现在时（✗「认为A难听，做B风格」——这丢掉了先后）；\n"
    "- 「做过后来放弃/不做了」的旧事单独写一条 经历，value 点明已放弃；"
    "现在的状态另写一条，两件事不许合并；\n"
    "- 用户解释了变化的原因（「因为A太难听」）时原因不能丢：在对应条目的 value 末尾"
    "带上「（因为…）」，原因写不下就单独写一条 评价认为；\n"
    "- 对话里有明确时间线索（上周/三月初/去年/昨天）的，把时间写进 value，别只留事情本身；\n"
    "属性规则（predicate，必须从下面清单里原样选一个，严禁自己造词）：\n"
    "- 爱好技能 / 生活习惯 / 经历 / 打算 / 评价认为 / 自我相关 / 其他\n"
    "- 一条 candidate 只写一件独立的事，别把多件事用分号塞进同一条；\n"
    "- 同一个属性下有多个独立事实（比如「喜欢喝茶」「会下棋」是两件事），就拆成多条 candidate，每条单独一行；\n"
    "- 同一个属性下允许多条并存（不同的爱好、经历、打算各写一条），不要为了「只留一条」把旧事丢掉；只写长期成立的东西；\n"
    "- 「喜欢听歌」「喜欢看花」「会下棋」「会写代码」「会摄影」一律归 爱好技能（技能和爱好不分开）；\n"
    "- 「喜欢几点睡觉」「不爱喝咖啡」「早起」这类作息/口味/日常规律，一律归 生活习惯；\n"
    "- 已经发生的各种经历（搬家、换过工作、送过礼、去过哪）一律归 经历；\n"
    "- 未来想做的事、计划、目标一律归 打算；\n"
    "- 对某人/某物/某事的看法、评价（自己的或别人的）一律归 评价认为，这只是主观看法，不是事实；\n"
    "- 年龄、身高、体重、籍贯、星座、身体状况这类客观事实一律归 自我相关；\n"
    "- 两个人之间的关系（谁是谁的女友/朋友）不要写进 candidates，放到 relations 里；\n"
    "- 实在归不进去才写 其他。\n"
    "改口规则（updates，只在明确推翻旧说法时才用）：\n"
    "- 用户消息后面可能附一段「已有长期记忆」，每行带 [#编号]，那是这个人的底档；\n"
    "- 只有这次对话明确否定了某条旧内容（改口、状态变了、纠正了以前的说法）才写 update；\n"
    "- 只是多了一件同类的事（又喜欢上一首歌、又去了一个地方）不算改口，直接写进 candidates，不要 update；\n"
    "- id 只能引用清单里出现过的编号，严禁自己编；by 是这次 candidates 里取代它的那条的下标（从 0 开始）；\n"
    "- 拿不准就别写，宁可多留一条旧的，也不要误删；一次最多写 3 条；\n"
    "- 清单里的内容不要抄进 candidates，除非这次对话确实又提到了它。\n"
    "改口例子：\n"
    "- 清单有 [#7] 爱好技能：喜欢摇滚；这次说「我现在只听说唱了」→ candidate 写「只听说唱」，"
    "update 写 {\"id\":7,\"action\":\"supersede\",\"by\":0}；\n"
    "- 清单有 [#7] 爱好技能：喜欢摇滚；这次说「最近也在听说唱」→ 只写 candidate，不写 update；\n"
    "- 清单有 [#9] 自我相关：20 岁；这次说「昨天过完 21 岁生日」→ candidate 写「21 岁」，"
    "update 指向 #9。\n"
    "关系规则（relations）：\n"
    "- 只记「对话里明确说出来的、两个真人之间的关系」，比如「A 说 B 是我女朋友」→ "
    "source_id=A 的编号, target_id=B 的编号；\n"
    "- relation 只能从 认识 / 朋友 / 亲友 / 恋人 里选一个，拿不准就不要输出这条；\n"
    "- source_id 和 target_id 必须是名单里两个不同的编号，不能相同、不能是 0；\n"
    "- 「弟弟」「我妈」「老板」这类称呼不是人：名单里没有他的编号就别输出这条关系；\n"
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
# 自我相关（不同方面的客观事实是多个独立条目）、评价认为（对不同人/
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
      "状况", "抑郁", "焦虑", "失眠"), "自我相关"),
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
    (("经历", "曾经", "以前", "过去", "小时候", "搬家", "转学", "送过", "去过",
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
                 r"学校|专业|学历|毕业|公司|单位|岗位|身份|抑郁|焦虑|"
                 r"失眠|身体|病情|住院|在读|在读生|患者|唯物主义者|留学)", v):
        return "自我相关"
    # 生活习惯（作息/口味/饮食）——「吃/喝」这类具体行为先于泛化的「喜欢」判断
    if re.search(r"(熬夜|起床|睡觉|睡眠|作息|生物钟|吃|喝|口味|外卖|饮食|"
                 r"点东西吃|晚睡|早起|不睡|心情不好|茶|咖啡|饮料|零食)", v):
        return "生活习惯"
    # 经历（明确的完成时动作：开发了/修好/做了/写过）
    if re.search(r"(开发了|修好了|修好|修改了|做了|写过|做过|搞定|完成|改造了|"
                 r"曾经|以前|过去|小时候|送过|去过|搬过|多年前|前任|初恋|"
                 r"靠.*挣钱|长期.*放不下|独立完成|参与开发)", v):
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
    # 爱好技能（会/擅长这类能力信号）
    if re.search(r"(会写|会做|会用|会编|会弹|擅长|技能|爱听|爱看|爱打|爱玩|"
                 r"棋类|摇滚|重金属|会乐器|做电音|编曲|书法|摄影|写.*工具|"
                 r"改.*代码|形象叫|网络形象|乐天派|喜欢|爱好|兴趣|迷恋|反感|讨厌|偏好)", v):
        return "爱好技能"
    return "其他"


# 一次总结最多改几条旧记忆。改口是低频事件，一次冒出四五条基本是模型在
# 拿「同类但不同事」当冲突，限制住比事后回滚便宜。
MAX_SUPERSEDE_PER_SUMMARY = 3
# 喂给总结的相关记忆条数。宁多勿少：漏一条，模型就没机会判断改口；
# 多带几条只是几十个 token。每人十几条时基本等于全量。
RELATED_L2_LIMIT = 16


def _bigrams_of(text: str) -> set:
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", str(text or "").lower())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()


def _day_label(ts) -> str:
    """created_at → 「2026-11-06」。取不到或非法就返回空串。

    老数据/测试构造的行没有 created_at，返回空串让调用方跳过日期行，
    行为与历史版本逐字节一致。
    """
    try:
        v = float(ts or 0)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    try:
        return time.strftime("%Y-%m-%d", time.localtime(v))
    except (OSError, OverflowError, ValueError):
        return ""


def number_transcript(rows) -> tuple[list[str], dict]:
    """给这次总结的每条日志打上说话人编号，返回 (带编号的行, {编号: 真名})。

    为什么不再让模型写名字：名字是开放集合，模型写一个库里没有的写法，我们
    就只能新建一个人——「弟弟」「1」就是这么来的。编号是封闭集合，模型只能
    从名单里挑；挑不出来（或编一个不存在的号）这条就直接丢，不会造人。

    0 号固定留给 AI。名单按出现顺序编号，同一会话里通常只有 1~2 个人。

    每换一天插一行「—— 2026-11-06 ——」（2026-11-08）：以前只给内容不给时间，
    模型看见了「到时候」「今晚」却不知道这批话是哪天说的，实测把它编成了
    「打算今晚好好睡一觉」并永久落库——那条原话其实是「我到时候好好睡一觉」。
    分隔线一天一行，比每行都挂时间戳省 token，语义也够（「今晚」= 那天晚上）。
    """
    id_of: dict[str, int] = {}
    lines: list[str] = []
    last_day = ""
    for r in rows:
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        day = _day_label(r.get("created_at"))
        if day and day != last_day:
            lines.append(f"—— {day} ——")
            last_day = day
        subject = str(r.get("subject") or "").strip()
        is_bot = (str(r.get("event_type") or "") == "bot_reply"
                  or content.startswith("[你]"))
        if is_bot:
            pid = 0
        elif not subject:
            # 没归属人的行：给个 ? 号，模型选不到，写进去也会被丢掉
            pid = "?"
        else:
            pid = id_of.setdefault(subject, len(id_of) + 1)
        lines.append(f"[{pid}] {content}")
    roster = {v: k for k, v in id_of.items()}
    return lines, roster


def roster_text(roster: dict) -> str:
    """把编号名单渲染成一行给模型看。0=AI 永远排最前面。"""
    if not roster:
        return ""
    head = "0=AI（不是你，不能当 subject）"
    body = "  ".join(f"{k}={v}" for k, v in sorted(roster.items()))
    return f"{head}  {body}" if body else head


def _unbound(store, name: str) -> bool:
    """这个人是不是「没有 QQ 的空壳」。

    只有**明确查到档案、且 QQ 是空的**才算。查不到档案说明是没登记过的新
    名字，不能凭这个判人家没 QQ——那种情况放行，漏记比错杀划算。
    空壳头像面板上点进去什么都没有，多一个就多一分认错人的机会，
    关系照写只会在图上留一个对不上号的节点，不如不写。
    """
    name = (name or "").strip()
    if not name:
        return True
    try:
        p = store.person_get(name)
    except Exception:
        return False
    if not p:
        return False
    return not str(p.get("qq") or "").strip()


def _roster_pick(store, item: dict, id_key: str, name_key: str, roster: dict) -> str:
    """按编号从名单里取真名。取不到就返回空串（调用方据此丢弃）。

    兼容模型没按协议走的情况：它写的是名字、但恰好跟名单里某个人对得上，
    就当作写对了编号；对不上一律不认——总结器没有新建人的权限。
    """
    raw = item.get(id_key)
    if raw not in (None, ""):
        try:
            pid = int(str(raw).strip())
        except (TypeError, ValueError):
            pid = None
        if pid is not None:
            return roster.get(pid, "")
        raw = None
    name = str(item.get(name_key) if raw is None else raw).strip()
    if not name:
        return ""
    if name in roster.values():
        return name
    try:
        canon = store.person_resolve(name) or ""
    except Exception:
        canon = ""
    return canon if canon in roster.values() else ""


def chunk_transcript(lines: list[str], budget: int = 6000) -> list[str]:
    """把日志按消息边界切段，每段不超过 budget 字符。

    以前是 `"\\n".join(...)[:6000]` 硬截断：超出那半段被静默丢掉，可总结行
    照样写进库、计数照样归零，等于这段对话永远没被总结过。现在宁可多打一次
    模型，也不切在句子中间。L3 入库时每条正文已截到 300 字，所以窗口再长
    也是有界的（90 条 × 300 ≈ 27k，最多 5 段）。
    """
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    # 切块可能正好切在日期分隔线之后：下一块开头就只剩「[1] 内容」，
    # 模型又不知道这是哪天的对话了。记住当前所在的那一天，换块时补到开头。
    day_ctx = ""
    for ln in lines:
        if ln.startswith("—— ") and ln.endswith(" ——"):
            day_ctx = ln
        if cur and n + len(ln) + 1 > budget:
            # 末尾孤立的日期行要摘下来交下一块：留在这一块的尾巴，
            # 模型会把上面那些话算进这一天（其实这一天还没开始说）。
            carry = ""
            if cur[-1].startswith("—— ") and cur[-1].endswith(" ——"):
                carry = cur.pop()
            chunks.append("\n".join(cur))
            cur, n = [], 0
            if carry:
                cur.append(carry)
                n += len(carry) + 1
            elif day_ctx and not (ln.startswith("—— ") and ln.endswith(" ——")):
                cur.append(day_ctx)
                n += len(day_ctx) + 1
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [""]


def merge_summaries(parts: list[dict]) -> dict:
    """把多段日志各自的解析结果合并成一次落库的输入。

    updates 里的 `by` 是「这一段 candidates 里的下标」，直接相加会指到别人
    的候选上，必须按前面已攒的条数偏移。摘要用「；」接起来，落库时统一截断。
    """
    merged = {"summary": "", "candidates": [], "relations": [], "updates": []}
    for p in parts:
        if not isinstance(p, dict):
            continue
        if p.get("summary"):
            merged["summary"] = (
                f"{merged['summary']}；{p['summary']}".strip("；")
                if merged["summary"] else str(p["summary"]))
        offset = len(merged["candidates"])
        for upd in p.get("updates") or []:
            if not isinstance(upd, dict):
                continue
            upd = dict(upd)
            try:
                upd["by"] = int(upd.get("by")) + offset
            except (TypeError, ValueError):
                continue
            merged["updates"].append(upd)
        merged["candidates"].extend(p.get("candidates") or [])
        merged["relations"].extend(p.get("relations") or [])
    return merged


def pick_related_l2(rows: list[dict], query: str = "",
                    limit: int = RELATED_L2_LIMIT) -> list[dict]:
    """挑出「这次总结可能涉及」的旧记忆，喂给模型判断要不要改口。

    两个来源，缺一不可：
      1) 和这段对话字面重合的（聊到了才会改口）；
      2) 永久底档（年龄/身份/联系方式）。它们几乎不会出现在对话原文里，
         但恰恰最容易被改口——「20 岁」变「21 岁」时原话是「昨天过完生日」，
         2-gram 一个都不沾。
    条数够少（大多数用户十几条）时直接全给，省得算分还算漏。
    """
    rows = [r for r in (rows or []) if r]
    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return sorted(rows, key=lambda r: int(r.get("id") or 0))
    qg = _bigrams_of(query)

    def score(r):
        text = f"{r.get('predicate', '')}{r.get('value', '')}{r.get('ask_alias', '')}"
        return (len(qg & _bigrams_of(text)),
                float(r.get("updated_at") or 0))

    perm: list[dict] = []
    other: list[dict] = []
    for r in rows:
        if str(r.get("predicate") or "").strip() in PERMANENT_PREDICATES:
            perm.append(r)
        else:
            other.append(r)
    keep_perm = sorted(perm, key=score, reverse=True)[:max(2, limit // 4)]
    keep_other = sorted(other, key=score, reverse=True)[:max(0, limit - len(keep_perm))]
    return sorted(keep_perm + keep_other, key=lambda r: int(r.get("id") or 0))


def format_related(rows: list[dict], names: dict | None = None) -> str:
    """把候选记忆渲染成给模型看的一行一条。id 必须留，updates 只能引用它。

    names 是 {内部键: 平时怎么叫}。模型看到什么名字，写摘要就会照着用；
    给它内部键（往往是一个人最早的昵称）就会出现「旧名 + 现名」并存的摘要。
    """
    names = names or {}
    lines = []
    for r in rows:
        when = ""
        try:
            ts = float(r.get("created_at") or 0)
            if ts > 0:
                when = time.strftime("%Y-%m-%d", time.localtime(ts))
        except (ValueError, OSError):
            when = ""
        value = str(r.get("value") or "").replace("\n", " ")[:80]
        subj = str(r.get("subject") or "")
        who = names.get(subj) or subj
        lines.append(
            f"- [#{r.get('id')}] {who} · {r.get('predicate')}："
            f"{value}{'（' + when + '）' if when else ''}"
        )
    return "\n".join(lines)


def apply_summary(store, gate: WriteGate, parsed: dict, source_msg_id: str = "",
                  subject: str = "", session_id: str = "", source_ts: float = 0.0,
                  roster: dict | None = None) -> dict:
    """把一次总结的产出（已解析 JSON）经门控落库。绝不调用 LLM。
    总结行会带上归属用户/会话，长期记忆通过 source_summary_id 反向指回这次总结。

    roster 是 {编号: 真名}。给了就按编号取人，名单之外的名字一律丢弃、绝不建档
    （总结器不是人，没有「发现新人」的权限，它只会在名字上犯错）；没给就走旧的
    按名字解析的路径，留给补录之外的直接调用和老测试。
    """
    # 传了名单就一定是严格模式，空名单也算——以前用 bool(roster) 判断，名单
    # 恰好为空（整批日志都是 AI 自己的话、或者历史数据没有 subject）时会静默
    # 退回按名字解析的老路径，那一瞬间总结器又有了建档权。宁可不写，也不建人。
    strict = roster is not None
    roster = roster or {}
    summary = str(parsed.get("summary", "") or "").strip()
    stats = {"summary_stored": 0, "l2": 0, "l3": 0, "deny": 0, "summary_id": 0,
             "relations": 0, "stale": 0, "superseded": 0, "update_ignored": 0,
             "kinship_skipped": 0, "unknown_subject": 0, "unbound_skipped": 0}
    summary_id = 0
    # 候选下标 → 落库后的 id / 主体。updates 里的 by 指的就是这个下标，
    # 所以必须按 enumerate 跑，不能只 for cand in candidates。
    cand_ids: dict[int, int] = {}
    cand_subjects: dict[int, str] = {}
    if summary:
        summary_id = store.write_l3(
            event_type="session_summary", subject=subject, content=summary[:500],
            source_msg_id=source_msg_id, session_id=session_id,
        )
        stats["summary_stored"] = 1
        stats["summary_id"] = summary_id
    for idx, cand in enumerate(parsed.get("candidates") or []):
        if not isinstance(cand, dict):
            continue
        text = str(cand.get("value", "") or cand.get("source_quote", "") or "").strip()
        if not text:
            continue
        if strict:
            # 只认名单里的编号。模型编一个不存在的号、或者写了名单外的名字，
            # 这条就丢——宁可漏一条，也不在人物库里留一个认不回来的人。
            cand_subject = _roster_pick(store, cand, "subject_id", "subject", roster)
            if not cand_subject:
                stats["unknown_subject"] += 1
                continue
        else:
            cand_subject = _normalize_subject(cand.get("subject"), subject)
        # 模型写的人名可能是旧昵称、外号、或者听岔的写法。能认出是谁就落到
        # 那个人的规范名上——否则「张三」和「Moon」会各记一套记忆。认不出
        # 才建新人，按「叫错的」等级登记（tier 3），久了没提就自动忘。
        # （strict 模式下上面已经取到真名，这一段只服务老路径。）
        if cand_subject and not strict:
            try:
                resolved = store.person_resolve(cand_subject)
            except Exception:
                resolved = ""
            if resolved:
                cand_subject = resolved
            elif is_generic_name(cand_subject):
                # 「用户」「对方」不是人。这句话的主语就是对话者本人，
                # 不该为它造一个永远认不回来的假人。
                cand_subject = subject
            elif is_kinship_alias(cand_subject):
                # 「弟弟」「我妈」「女朋友」是称呼不是名字。硬记只会在人物库
                # 里多一个假人；记到说话人名下又是把别人的事安到他头上——
                # 两条都错。原话还在 L3 里，要查得回来；这条候选就不要了。
                stats["kinship_skipped"] += 1
                continue
            else:
                try:
                    cand_subject = store.person_ensure(
                        cand_subject, source="summary") or cand_subject
                except Exception:
                    pass
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
            res = store.write_l2(
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
                source_ts=source_ts,
            )
            if res.get("action") == "stale":
                # 这批原话比现有记忆还旧，跳过不算写入
                stats["stale"] += 1
            else:
                stats["l2"] += 1
                try:
                    cand_ids[idx] = int(res.get("id") or 0)
                    cand_subjects[idx] = cand_subject
                except (TypeError, ValueError):
                    pass
        elif verdict.decision == "transform":
            store.write_l3(
                event_type="demoted_candidate", subject=cand_subject,
                content=f"[{verdict.reason}] {text}"[:300],
                source_msg_id=source_msg_id, session_id=session_id, summary_id=summary_id,
            )
            stats["l3"] += 1
        else:
            stats["deny"] += 1
    # 改口裁决：把旧事实归档并指向新事实。四道闸门，缺一不改：
    #   1) 新事实必须是这次通过门控、真的落了库的那条（by 指向候选下标）
    #   2) 旧事实必须还是 active、没被图钉钉住、跟新事实是同一个人
    #   3) 旧事实不能比这段对话还新（补录半年前的话不许推翻今天的事实）
    #   4) 一次最多改 3 条
    for upd in parsed.get("updates") or []:
        if not isinstance(upd, dict):
            continue
        if str(upd.get("action") or "").strip().lower() != "supersede":
            stats["update_ignored"] += 1
            continue
        if stats["superseded"] >= MAX_SUPERSEDE_PER_SUMMARY:
            stats["update_ignored"] += 1
            continue
        try:
            old_id = int(upd.get("id"))
            by = int(upd.get("by"))
        except (TypeError, ValueError):
            stats["update_ignored"] += 1
            continue
        new_id = cand_ids.get(by) or 0
        if not new_id or new_id == old_id:
            # 没有本次落库的新事实兜底 = 纯删除，绝不允许
            stats["update_ignored"] += 1
            continue
        old = store.get_l2(old_id)
        if not old or str(old.get("status") or "") != "active":
            stats["update_ignored"] += 1
            continue
        if int(old.get("pinned") or 0):
            # 钉住 = 用户手动说了「留着」，改口也不动它
            stats["update_ignored"] += 1
            continue
        if str(old.get("subject") or "") != str(cand_subjects.get(by) or ""):
            stats["update_ignored"] += 1
            continue
        old_ts = max(float(old.get("updated_at") or 0), float(old.get("created_at") or 0))
        if source_ts and old_ts > float(source_ts):
            stats["update_ignored"] += 1
            continue
        if store.supersede_l2(old_id, new_id):
            stats["superseded"] += 1
        else:
            stats["update_ignored"] += 1
    for rel in parsed.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        if strict:
            src = _roster_pick(store, rel, "source_id", "source", roster)
            dst = _roster_pick(store, rel, "target_id", "target", roster)
            if not src or not dst:
                stats["unknown_subject"] += 1
                continue
        else:
            src = _normalize_subject(rel.get("source"), subject)
            dst = _normalize_subject(rel.get("target"), subject)
        rank = normalize_relation(rel.get("relation"))
        if not src or not dst or not rank or src == dst:
            continue
        # 没有 QQ 的空壳不写边。这种人本来就没法对应到具体账号，留一条边等于
        # 在图上插一个点进去什么都没有的节点——面板上还要多认一次人。
        # 只拦「档案里明确空号」的，没登记过的新名字照写（见 _unbound）。
        if _unbound(store, src) or _unbound(store, dst):
            stats["unbound_skipped"] += 1
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
        elif res.get("reason") == "kinship":
            # 「我弟」「我妈」这种称呼，没有真名就不建边（原因见 relation_upsert）
            stats["kinship_skipped"] += 1
    return stats


# ---------- L3 过期骨架：从关键句忘到只剩关键词（零 LLM） ----------
# 设计：原文到期后不直接物理删除，先按「会话×天」压成一行主题词索引再删，
# 骨架再按自己的保留期过期。这样「上个月聊过啥」至少有个索引可查，
# AI 被问起时不是两眼一抹黑。主题词提取 = 高频 2-gram + 停用过滤 +
# 不重叠截取，全程纯文本操作，成本可忽略。

# 含这些内容的 2-gram 一律不进骨架：虚词骨架、数字、字母、标点——
# 「了记」「20」「，就」这类碎片既不达意又占名额
_SKELETON_STOP_RE = re.compile(
    r"[的是了就在有和与或都也很还吗呢啊吧呀哪什么怎么为何因为所以这个那咋咱你我他她它]"
    r"|\d|[a-zA-Z]|[\s，。！？…、；：~～《》\[\]（）()「」\"'',;:?!]"
)

# 会话前缀：骨架只关心聊了什么主题，不关心「[私] 张三:」这个信封。
# 不剥的话，信封碎片（「张三:」「私」）会被当成主题词挤进骨架。
_L3_ENVELOPE_RE = re.compile(r"^\[(?:私|群)\]\s*[^:：]{1,40}[:：]\s*")
# 引用回复信封：[引用消息(昵称: 被引内容 … MSG_ID:123)] 用户真正说的话在括号外
_QUOTE_ENVELOPE_RE = re.compile(r"\[引用消息\(.+?MSG_ID:\s*\d+\)\]", re.S)


def _grams_for_skeleton(text: str) -> set:
    """2~3 字片段全收。曾经试过 4 字：滑窗把「倒库又压线」切成跨词碎片
    「库又压线」，真词「倒库」反而被挤出骨架——用户拿「倒库」来问就永远
    召不回。中文词以 2~3 字为主，4 字片段信息量低、破坏词完整性，弃用。"""
    t = re.sub(r"\s+", "", str(text or ""))
    out = set()
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            out.add(t[i: i + n])
    return out


# ---------- 选句版骨架打分（纯本地，零 LLM） ----------
# 事实的数量骨架是数字：「九十分钟」「五分钟」里的数量词必须整体存活。
_SENT_NUM_RE = re.compile(r"[0-9０-９]|[一二两三四五六七八九十百千万几半]+")
# 时间词：什么时候发生的是骨架的另一半。
_SENT_TIME_RE = re.compile(
    r"今天|昨天|前天|明天|今晚|早上|中午|下午|晚上|凌晨|周末|月底|月初"
    r"|上?[周礼拜][一二三四五六日天]|[0-9０-９]+月|[0-9０-９]+[号日]"
    r"|分钟|小时|点半?|[0-9０-９]点")
# 转述/约定动词：带出处的句子（谁说了什么）优先保留。
_SENT_QUOTE_RE = re.compile(r"(说|讲|告诉|答应|提醒|建议|约好?|提议|问)")
# 填充句：哈哈刷屏、嗯哦、省略号——没有记忆价值。
_SENT_FILLER_RE = re.compile(r"哈哈|嘿嘿|嘻嘻|嗯+|哦+|额|呸|诶{2,}|。。。|……\s*$")


def _score_skeleton_sentence(t: str, subj: str, known_names: set) -> float:
    """给一句候选原话打分：数字/转述/时间词加分，填充句扣分。"""
    s = 0.0
    # 纯数字/ID/乱码句（中文占比低）：数字加分规则会偏爱「123456001」
    # 这种 QQ 号消息，必须先把没有语义载体的句子重罚下去
    cn = len(re.findall(r"[\u4e00-\u9fff]", t))
    if len(t) >= 4 and cn / max(1, len(t)) < 0.4:
        s -= 3.0
    if _SENT_NUM_RE.search(t):
        s += 2.0
    if _SENT_QUOTE_RE.search(t):
        s += 1.5
    # 内容里点名了本组说话人以外的在场者 → 第三方/转述线索
    others = known_names - {subj, ""}
    if any(n and n in t for n in others):
        s += 1.0
    if _SENT_TIME_RE.search(t):
        s += 1.0
    if 8 <= len(t) <= 48:
        s += 1.0
    if _SENT_FILLER_RE.search(t):
        s -= 2.0
    return s


def _skeleton_head_grams(texts: list) -> str:
    """兜底：没有句子过线时退回旧版词频头部，保证骨架不空转。"""
    from collections import Counter
    grams: Counter = Counter()
    for t in texts:
        for g in _grams_for_skeleton(t):
            if _SKELETON_STOP_RE.search(g):
                continue
            grams[g] += 1
    common = sorted(grams.items(), key=lambda x: (-x[1], -len(x[0])))[:8]
    if common:
        return "/".join(g for g, _ in common)
    return re.sub(r"\s+", "", texts[0])[:16] if texts else ""


def make_skeleton(rows: list, *, keep_until: float, now: float | None = None) -> list:
    """把一组即将过期的 L3 行压成骨架行描述，供 store.cleanup_expired 写回。

    rows：同一次清理批次的过期 L3（内部按 会话×天 自行分组）。
    返回 [{event_type, subject, session_id, content, created_at, expires_at}]。

    v2 选句版（零 LLM）：每组挑 1~2 句信息密度最高的**原话**截断保留，
    取代旧版 2-gram 词频关键词。滑窗会把「九十分钟」切成「九十分」、
    「蓝色柜子」切成「后台蓝/色柜子」——碎片即使召回也无法支撑作答。
    打分：数字 +2、转述动词 +1.5、点名在场他人 +1、时间词 +1、
    句长 8~48 字 +1、填充句 -2。选不出过线句子时退回旧版词频头部。
    """
    import time as _time

    now = now or _time.time()
    lt = _time.localtime
    groups: dict = {}
    for r in rows:
        ts = float(r.get("created_at") or 0)
        day = _time.strftime("%m-%d", lt(ts)) if ts > 0 else "未知"
        key = f"{r.get('session_id') or ''}|{r.get('subject') or ''}|{day}"
        groups.setdefault(key, []).append(r)

    # 全批次出现过的说话人 = 在场名单的近似，用于「点名了别人」的加分
    known_names = {str(r.get("subject") or "").strip() for r in rows}
    known_names.discard("")

    out: list = []
    for key, grp in groups.items():
        sid, subj, day = key.split("|", 2)
        # 转述标注：组里没有用户原话、只有 bot_outbound（AI 在别的会话主动
        # 说的内容）时，标题带上「·转述」，面板和注入里都能一眼区分来源
        types = {str(r.get("event_type") or "") for r in grp}
        tag = "·转述" if ("user_message" not in types and "bot_outbound" in types) else ""
        # 与旧版一致：优先本组用户原话，没有再考虑组内其他行
        texts = [_QUOTE_ENVELOPE_RE.sub("", _L3_ENVELOPE_RE.sub("", str(r.get("content") or "")))
                 for r in grp
                 if r.get("event_type") == "user_message"
                 and str(r.get("content") or "").strip()]
        if not texts:
            texts = [_QUOTE_ENVELOPE_RE.sub("", _L3_ENVELOPE_RE.sub("", str(r.get("content") or "")))
                     for r in grp if str(r.get("content") or "").strip()]
        # 打分选句：top2，近似句（2-gram 重叠过半）去重
        scored = []
        for t in texts:
            t = re.sub(r"\s+", "", t)
            if not t:
                continue
            scored.append((_score_skeleton_sentence(t, subj, known_names), len(t), t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked: list = []
        picked_grams: list = []
        for s, _ln, t in scored:
            if len(picked) >= 2 or s <= 0:
                break
            gs = {t[i:i + 2] for i in range(len(t) - 1)}
            if any(len(gs & pg) / max(1, min(len(gs), len(pg))) > 0.5 for pg in picked_grams):
                continue  # 和已选句大半重复（如连发的同款指令），换下一个候选
            picked_grams.append(gs)
            picked.append(t[:46] + ("…" if len(t) > 46 else ""))
        if not picked:
            head = _skeleton_head_grams(texts)
            if not head:
                continue
            picked = [head]
        latest = max(float(r.get("created_at") or 0) for r in grp)
        out.append({
            "event_type": "l3_skeleton",
            "subject": subj,
            "session_id": sid,
            # 两句 ×46 字 + 前缀，一行 ~110 字封顶
            "content": f"[骨架·{day} 聊了{len(grp)}句{tag}] " + "；".join(picked),
            "created_at": latest,
            "expires_at": float(keep_until),
        })
    return out
