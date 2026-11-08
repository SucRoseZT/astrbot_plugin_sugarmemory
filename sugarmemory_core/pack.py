"""记忆包组装：检索、排序、裁剪、可解释输出。零 LLM 调用。
关键词 2-gram 检索独立工作；向量检索为可选增强（预留接口，不默认依赖）。
"""
import math
import re
import time
import unicodedata

from .injector import estimate_tokens, trim_pack
from .store import PERMANENT_PREDICATES, alias_key, retention_score

# 裁剪告警节流：常驻包每轮都要 build 一次，预算调低时会每轮都触发同样的
# 裁剪。相同原因 60 秒只报一次，别把日志刷爆。
_TRIM_WARN: dict = {"sig": "", "at": 0.0}


def _norm_text(text) -> str:
    """检索用的最小归一：兼容字符（全角数字/字母）+ 大小写。

    纯 bigram 匹配是逐字符比较的，比任何全文索引都「笨」在这一处：
    专业全文索引默认大小写不敏感，天然 `iPhone == iphone`，我们这边不是。
    实测库侧「iPhone 12」切成 {iP, Ph, ho, ne, e1, 12}，问侧「iphone」切成
    {ip, ph, ho, ne}，只靠 ho/ne 两个凑巧重叠活着；换成 AI / QQ / B站 这种
    两三个字符的，直接零命中。
    NFKC 顺带把全角「ｉＰｈｏｎｅ」「１２３」折成半角，中文汉字不受影响。
    """
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", _norm_text(text))
    if len(text) < 2:
        return {text} if text else set()
    return {text[i: i + 2] for i in range(len(text) - 1)}


# 意图规则只用来微调各通道权重，不用来开关通道：判错顶多是排序差一点，
# 判成开关则会整条路召回不到。规则一律是通用中文口语，不为个案硬编码。
_REL_Q = re.compile(r"谁|认识|朋友|关系|对象|恋人|男朋友|女朋友|老婆|老公|叫什么|名字|一起")
_TIME_Q = re.compile(r"上次|之前|刚才|以前|曾经|那时候|那天|聊过|说过|发过|还记得")
_FACT_Q = re.compile(r"是什么|怎么|为什么|几|多少|喜欢|在哪|叫|记得吗")
_REL_PRED = re.compile(r"关系|朋友|对象|恋人|称呼|名字|认识|家人|同事")
_NUM_RE = re.compile(r"[0-9零一二两三四五六七八九十百千万]+")


def _l2_text(r: dict) -> str:
    """事实本体 + 主体名。**刻意不含 ask_alias，也不含 predicate**。

    ask_alias 是模型写的三句口语化问法，每条覆盖「其实 / 有点 / 东西 /
    你了」这类日常词汇，整段拼进检索文本等于给每条记忆挂一个万能噪声
    接收器：一句「嘿嘿想你了」能靠别名里的「你了」撞上「定时任务插件」。
    0.16.16 起 alias 走独立低权通道，只在本体零命中时兜底。

    predicate 是**分类名**（关系 / 打算 / 经历 / 评价认为），不是内容。
    拼进来等于给同一类下每条记忆盖同一个章：概念词「关系」一激活，
    「我有对象吗」就把妈妈、妹妹、猫、最好的朋友五条全捞进来（实测 5 条
    274 token，只有 1 条是答案）。分类不进入检索文本，让 value 自己说话。
    """
    return f"{r.get('subject', '')}{r.get('value', '')}"


def _l2_content(r: dict) -> str:
    """事实内容，不含主体名、也不含分类名。

    两个都不该进：主体名用来区分「只提到这个人」和「内容被提到」，分类名
    （关系 / 打算 / 经历）是给人类看的标签，同一类下每条都带同样的两字，
    概念词一旦等于分类名（「对象 恋爱 关系」里的「关系」）整类会一起中招。
    """
    return str(r.get("value", ""))


def _l2_alias(r: dict) -> str:
    """问法别名：只做兜底召回，不进主相关性判据。"""
    return str(r.get("ask_alias") or "")


# ---- 常驻底档的 predicate 权重 ----
# 只回答一个问题：「这条事实，不看这轮聊什么也该知道吗？」
#
# 为什么需要它：常驻排序原来是 conf + min(stab,5)*0.1 + retention*0.3。
# stability 每级 0.1、最高 +0.5，是置信度差距（通常 0.05）的十倍——**越反复
# 聊的话题 stability 越高，就越永久霸占常驻前排**。实测用户库里 stab=5 的
# 「自己给机器人写了一堆插件」稳坐第 1 位，四条 stab=4 的「觉得别人插件差」
# 把 conf 0.9 抬进前 16。这些恰恰是最该等话题再召回的。
#
# 分档依据：
#   身份/关系 —— 每条都在决定 AI 怎么称呼、怎么关心这个人，必须常驻；
#   习惯/技能 —— 人物画像的骨架，常驻但让位于身份；
#   打算       —— 有时效，常驻价值中等；
#   经历/评价  —— 某个话题下才有用，聊到再召回（0.55 足够压过 stab 满级加成）。
# 不在这张表里的 predicate 走 0.8，宁可保守也不误伤新分类。
_RESIDENT_WEIGHT = {
    "自我相关": 1.0, "其他": 1.0, "关系": 1.0, "称呼": 1.0,
    "生活习惯": 0.9, "爱好技能": 0.9,
    "打算": 0.7,
    "经历": 0.55, "评价认为": 0.55,
}
_RESIDENT_WEIGHT_DEFAULT = 0.8


# 泛用 bigram：疑问词、指代词、口头禅。整句只靠这些撞上一条事实时，那不是
# 「相关」而是巧合。实测：「我养过什么宠物」把「什么也不知道」那条评价记忆
# 捞进尾部（库里根本没有宠物事实）；「我平时爱吃什么」捞到「平时基本不运动」。
# 做法是直接从命中集合里扣掉，而不是「有泛用词就整条作废」——「那个插件」
# 里的「插件」是真信号，只扣「那个」。扣完为空就等于没命中。
_STOP_GRAMS = frozenset({
    "什么", "怎么", "为什", "么样", "那个", "这个", "哪个", "哪些", "这些",
    "那些", "一些", "一下", "一样", "一点", "有点", "有些", "很多", "其实",
    "但是", "因为", "所以", "然后", "现在", "时候", "感觉", "觉得", "应该",
    "可以", "真的", "就是", "还是", "或者", "我们", "你们", "他们", "她们",
    "自己", "别人", "平时", "最近", "以前", "刚才", "上次", "这次", "了吗",
    "好的", "知道", "不知",
    # 量词/泛用量：「一个月挣多少钱」靠「一个」撞上「去年一个人去北海道」，
    # 一条关于旅行的经历压过了工资那条真答案（实测排第 1）。「一个」在任何
    # 句子里都不指向任何具体话题。
    "一个", "一种", "一件", "一次",
    # 泛用指代：「东西 / 事情」在任何句子里都不指具体话题，却因为口语里
    # 到处出现（「买东西」「吃的东西」）而撞得极准。实测「我丢过什么东西」
    # 被「喜欢在拼多多买东西」抢了第一（IDF 算下来 5.05，比真答案「赶集的
    # 时候丢过钱包」高一倍），问丢东西答爱购物。
    "东西", "事情", "事儿",
    # 副词 / 连词 / 泛指词。只收能构成 bigram 的双字词——检索是字粒度，
    # 单字停用词在这里加是自欺欺人（「很」「太」本来就过不了 _FUNC_CHARS
    # 的两侧过滤）。
    #
    # 时间是**话题之外的维度**：「一直」「已经」「突然」修饰任何事，
    # 却不指向任何一件事，跟「以前」「最近」是同一类，处理方式也该一致。
    "已经", "正在", "总是", "一直", "从来", "马上", "立刻", "顿时",
    "忽然", "突然", "渐渐", "逐渐", "慢慢", "刚刚", "快要", "就要",
    "非常", "特别", "十分", "极其", "相当", "几乎", "稍微", "比较",
    "而且", "并且", "不过", "可是", "然而", "因此", "如果", "虽然",
    "尽管", "除非", "只要", "于是", "要么", "不但", "反而",
    "各种", "任何", "所有", "每个", "某些", "其他", "一样", "之间",
    "是不是", "有没有", "能不能", "会不会", "要不要", "对不对",
})

# 功能字：两个都是功能字的 bigram 是跨词边界的碎片（「的那」「你了」「是我」
# 「在这」），不携带任何主题信息。它们反而因为在整个库里罕见，IDF 给得高得
# 离谱 —— 实测「上次说的那个事」就是靠 #79 里「用的**的那**个记忆插件」这一
# 段碎片把一条完全无关的评价记忆顶到尾部第一位。词表治不了这种，只能按字判。
_FUNC_CHARS = set("的了是在有和就都也还很太更最这那个什么怎谁哪咋嘛么之其而但所因如若虽"
                  "你我他她它们啊吧呢吗呀哦哈嘿呀啦呀")


_ASCII_WORD = re.compile(r"[a-z0-9]{2,}")


def _content_grams(text: str) -> set[str]:
    """带虚字过滤的 bigram + 英文/数字整词。"""
    t = _norm_text(text)
    grams = {g for g in _bigrams(t)
             if not (len(g) == 2 and g[0] in _FUNC_CHARS and g[1] in _FUNC_CHARS)}
    # 英文和数字另外给一份**整词**。2-gram 拿来做检索有个绕不过去的坑：
    # Java / JavaScript 共享 {ja, av, va}，问「Java 好用吗」能把 JS 那条
    # 拉上来；「6800」和「68000」也共享四个 gram。整词的 df 天然低，
    # IDF 就高，命中一个整词等于「说的是同一个东西」，比五个 2-gram 都硬。
    # 不删 2-gram：整词挡不住的形态变化（iphone / iphone12）还得靠它。
    grams |= set(_ASCII_WORD.findall(re.sub(r"\s+", "", t)))
    return grams


# 多意图连接词：「我妈的糖尿病和我胃病哪个严重」这种句子里有两个主语，
# 不能因为点了「妈」就把「我」的那条排掉，否则对比题只剩一半。
_MULTI_RE = re.compile(r"和|跟|与|以及|还有|哪个|谁更|比较|谁大|谁小")

# 接续词：只表示「接着说」，本身不指向任何话题。它们留在 grams 里会让
# 「后来呢」「然后呢」被当成「这轮有实词」，从而关掉历史补位通道——实测
# 「后来呢」在上一轮问过车祸之后返回空，因为「后来」两个字被算成了主题。
_CONT_RE = re.compile(r"后来|然后|接着|于是|接下来|再然后|所以说|所以|那么|继续|还有呢|再说")


def _query_has_topic(text: str) -> bool:
    """这句里有没有指向某个话题的实词（接续词不算）。

    「后来呢」→ 剥掉「后来」只剩「呢」→ 没有话题，该走历史补位。
    「吃东西」→ 有 → 换了话题，别拿上一轮的答案硬凑。
    """
    t = _CONT_RE.sub("", str(text or ""))
    # 只认长度 ≥2 的：剩下一个「呢」是 _bigrams 对单字的兜底产物，不是话题。
    return bool({g for g in _content_grams(t) if len(g) >= 2} - _STOP_GRAMS)


# ---- query 概念归一（零成本补召回的第一层）----
# 中文口语的改写问法（「多大了」vs「16岁」）2-gram 字面零重叠，闸门必砍。
# 这里把常见问法映射到「概念词」，检索前拼接到 query 尾部：原文保留（字面
# 命中不受影响），概念词让记忆条里的对应说法有机会撞上 bigram。
# 规则要泛化、不许为个案硬编码：每条都是通用中文口语知识，宁可漏配
# （还有 alias 列和语义兜底接着），不要配错（概念词乱激活会拉低精度）。
_QUERY_CONCEPTS: list[tuple[re.Pattern, str]] = [
    (re.compile(p), concepts) for p, concepts in (
        (r"多大|几岁|年龄|年纪|岁数", "岁 年龄"),
        (r"吃|喝|宵夜|夜宵|零食|做饭|下饭|点东西|饿|馋", "吃 食物 饮食"),
        # 「困」单字命中睡眠，但要排除「困难 / 贫困 / 围困 / 困境 / 困惑 /
        # 困扰 / 被困住」这些非睡眠义。前缀看一字符足够：中文里这些词的第二
        # 字固定，往后多看会误伤「好困得不行」。
        (r"睡|熬夜|失眠|作息|通宵|起床|才醒|补觉|赖床|疲"
         r"|(?<![难贫围境扰惑被])困(?!难|境|惑|扰|住|局)", "睡 作息 熬夜"),
        (r"心理医生|看病|去医院|抑郁|焦虑|失眠|崩溃|状态不好|休息", "心理 病 状态"),
        (r"人设|形象|设定|马甲|网名", "形象 人设"),
        (r"创作|作品|搞什么|玩什么|接稿|约稿|画稿", "创作 作品 稿"),
        (r"礼物|送过|送了|收过|收到", "礼物 送"),
        (r"去得成|来吗|过来|见面|去找|来访|打算.{0,8}来|暂缓", "来 见面 计划"),
        (r"挣钱|赚钱|收入|工资|生活费|零花", "钱 挣钱"),
        # 问「干啥的 / 什么工作」时，答案在 value 里往往写成「在xx公司做运营」
        # 「做装修生意」，字面和「工作」零重叠（实测「我是做什么工作的」把
        # 三条无关记忆拉进包，真答案的 value 里一个「工作」都没有）。
        (r"工作|上班|职业|干什么的|做什么的|做啥的|干啥的|混哪行", "工作 公司 上班 生意"),
        # 「老公 / 老婆」是**具体的人**，不是「恋爱」这个概念。原来一起映射到
        # 「对象 恋爱 关系」，等于每句问伴侣的话都被拼上「恋爱」「关系」——
        # 实测问「我老公生日是哪天」，答案本该是「老公叫阿强…」，但概念词把
        # 一条评价（「觉得婆婆管得太多」）也点亮了，还让「生日是农历八月十六」
        # 稳稳排第一。同义映射只放同义的**称呼**，不放抽象类目。
        (r"对象|男女朋友|恋人|男朋友|女朋友|前任|老婆|老公|丈夫|妻子",
         "老公 老婆 对象 丈夫 妻子"),
        (r"留学|出国|哪个国家|加拿大|日本|美国", "留学 国家"),
        (r"音乐|电音|歌曲|编曲|做歌|听歌", "音乐"),
        (r"考试|备考|专升本|升学|读书|上学", "考试 学"),
        (r"插件|开发|写代码|编程|程序|bot", "插件 开发"),
    )
]

# 扩展标记：概念词只参与检索，不进注入块渲染，也不参与外发/追问等问法判断。
_CONCEPT_MARK = "\u2063"  # 不可见分隔符（INVISIBLE SEPARATOR）


def expand_query(query: str) -> str:
    """把口语问法归一成「原文 + 命中的概念词」。没命中任何模式时原样返回。"""
    q = str(query or "")
    if not q.strip():
        return q
    extra: list[str] = []
    for pat, concepts in _QUERY_CONCEPTS:
        if pat.search(q):
            extra.append(concepts)
    if not extra:
        return q
    return q + _CONCEPT_MARK + " ".join(extra)


# 用户在问「这哪来的 / 这是什么」的典型说法。注意这不是关键词检索——它不跟
# 记忆内容比对（「这是什么」四个字在库里永远匹配不到任何东西），而是判断
# 「这一句是不是在指代一条消息」。
_PROBE_RE = re.compile(
    r"(这|那)(一)?(句|条|个)?(是|到底)?(什么|啥)"
    r"|什么意思|啥意思|啥呀|啥啊|什么鬼"
    r"|你(刚|刚才|刚刚|前面)?(发|说|讲)的"
    r"|(刚才|刚刚|开始)(那|这)"
    r"|怎么回事|咋回事|没看懂|看不懂|不懂"
)
# 纯语气/纯标点：「啊？」「？？？」「……」——这类反而最典型，用户看到一条
# 莫名其妙的定时消息，第一反应就是一个字加问号。
_PURE_MOOD_RE = re.compile(r"[?？!！。…~～啊唉诶欸咦哦噢嗯呐]+")


# 强信号：直接说了「转述/带话/替我」这类词。无论指向谁，都说明这轮在聊
# 「你替谁给谁带过什么话」，跨会话那条值得带上。
_ASK_AWAY_STRONG_RE = re.compile(r"转述|带[个句]?话|捎话|带过去|代我|替我")

# 「给/跟/向 某某 发/说/讲」。关键是必须指向**第三人称**：目标要是「我/你/咱」，
# 问的就是当前这个会话收到的东西，由 here 段负责，不该拿跨会话记录去凑。
_ASK_AWAY_RE = re.compile(
    r"(给|跟|和|向|对|帮)(?!我|你|咱).{1,16}(发|说|讲|转述|带|问好|问候|聊)"
)

# 自指问法：「你刚才发了什么」「还记得自己发了啥吗」「你之前说过什么」。
# 这类句子既没点名、也没说「转述」，但问的正是「你自己发出去的话」——
# 只靠上面两条会整句漏掉（实测那轮：用户问「你读记忆包时发了什么」
# 被判成不追问，那条转述就没进包，模型当场答「没记录」）。
# 必须带「了/过」才算：`你说什么` 是「没听清，再说一遍」，不是问你发过什么。
# 「你」前面排除「给/跟/和/向/对/帮」：`他给你发了什么` 问的是对方发来的，
# 拿自己发出去的东西去凑等于答非所问。
_ASK_SELF_SENT_RE = re.compile(
    r"(?<![给跟和向对帮])(你|自己)[^。？！?!\n]{0,12}?"
    r"(发|说|讲|带)(了|过)(什么|啥|哪些|哪句|什么话|啥话)"
)

# 只提到人名还不够：「路人乙 最近怎么样」在聊她的近况，不该把一条转述
# 塞进来（实测这条会误命中，白烧 token 还容易让模型跑题）。名字/号码必须
# 跟「回忆或发送」这类词同时出现，才算在问「你发过什么」。
_RECALL_RE = re.compile(
    r"记得|记不记得|还记得|忘了|回忆|发|说|讲|转述|带话|带过去|聊|问好|问候"
)


def _asks_outbound_away(query: str, rows: list[dict], nick_map: dict) -> bool:
    """这一句是不是在追问「你给别的人/别的会话发了什么」。

    为什么要专门判一句：跨会话转述（在 A 会话让 AI 给 B 带话，回 A 问「你刚才
    跟 B 说了什么」）的记录，session_id 记的是**收件人 B**，永远不会落进
    「发到当前会话」那一栏。默认策略是整段不带（省 token，也免得拿别人的话
    污染当前对话），于是这一问永远答不上来。

    判据从宽到严三层，任何一层命中就补 1 条，全不中一条不带：
      ① 「转述/带话/替我」这类词；
      ② 「给/跟/向 某某 发/说」这种指向第三人称的发送动作；
      ③ 本轮原话里出现了某条记录的目标昵称——最精准，几乎不误报。
    """
    q = str(query or "").strip()
    if not q:
        return False
    if _ASK_AWAY_STRONG_RE.search(q) or _ASK_AWAY_RE.search(q):
        return True
    if _ASK_SELF_SENT_RE.search(q):
        return True
    # 下面两条都要「有人名/号码」+「回忆或发送词」同时成立，缺一不算。
    asks_what = bool(_RECALL_RE.search(q))
    if not asks_what:
        return False
    for r in rows[:8]:
        nick = str(nick_map.get(str(r.get("session_id") or ""), "") or "").strip()
        # 单字昵称（「鱼」「喵」）在正文里到处都是，按 2 字起步免得句句命中
        if len(nick) >= 2 and nick in q:
            return True
        # 直接报号码的情况：「你给 100010001 发了什么」/「对 100010001」
        tail = str(r.get("session_id") or "").rsplit(":", 1)[-1].strip()
        if len(tail) >= 5 and tail in q:
            return True
    return False


# 定时任务专属信号。转述和定时是两件事：前者是「主人让我带句话」，
# 后者是「到点自动发」，混在一段里模型会把定时提醒说成「我替你带话」，
# 或者反过来把转述当成闹钟。所以判据也分开。
_ASK_CRON_RE = re.compile(r"定时|闹钟|提醒|预约|那个任务|任务发|自动发|几点发")


def _asks_outbound_away_cron(query: str, rows: list[dict], nick_map: dict) -> bool:
    """这一句是不是在追问「你按定时任务给别的人发了什么」。

    和 ``_asks_outbound_away`` 同一套思路，只是判据换成定时任务的说法。
    分开的理由：问「你给 B 带什么话了」时补一条定时提醒进包，模型会
    把闹钟内容当成转述原话复述出来，比不补更糟。
    """
    q = str(query or "").strip()
    if not q:
        return False
    if _ASK_CRON_RE.search(q):
        return True
    for r in rows[:8]:
        nick = str(nick_map.get(str(r.get("session_id") or ""), "") or "").strip()
        if len(nick) >= 2 and nick in q:
            return True
    return False


def _outbound_kind(content: str) -> str:
    """把一条外发记录归到 relay（转述）/ cron（定时任务）/ plain（普通外发）。

    数据层只有 content，没有单独的类别列，所以靠写入时打的固定前缀判断。
    前缀由 sugarmemory 自己在 hook send_message 时拼，见 main.py 的
    ``[你转述给用户 …]`` / ``[你按定时任务「…」发给用户 …]``。
    """
    c = str(content or "")
    if c.startswith("[你转述给"):
        return "relay"
    if c.startswith("[你按定时任务"):
        return "cron"
    return "plain"


def _asks_about_message(query: str, max_len: int = 30) -> bool:
    """这一句是不是在问「你刚才发的那个是什么」。

    为什么要靠这句话、而不是靠「记录离现在多久」：用时间窗口等于规定用户
    什么时候能问。实测（07:00，真实模型跑 40 分钟前的记录）600 秒窗口一过，
    三次里一次答「我什么都没收到呀，是不是图没发出来」——看到提醒、去忙别的、
    过一阵才回头问一句，是最普通不过的行为，不该被窗口卡住。
    反过来也不能「永远当新的」：三天前的记录还标「刚发」会误导。

    所以判据落在这一句本身：短句 + 指代/疑问指向一条消息。跟时间无关，
    隔三分钟问和隔三天问，同一句话得到同一个答案。
    """
    q = re.sub(r"\s+", "", str(query or ""))
    if not q or len(q) > max_len:
        return False
    if _PURE_MOOD_RE.fullmatch(q):
        return True
    return bool(_PROBE_RE.search(q))


def resolve_preview_scope(subject: str, session_id: str, cache_items,
                          recent_row, scope_all: bool = False):
    """注入预览 / 配置引导该用「谁」的包：返回 (subject, session_id, subject_auto)。

    面板和引导都可能没传 subject。这时**绝不能**退化成
    `query_active_l2(subject="")` —— 那是不分人的全库分支，会把全库所有人的 L2
    按置信度混进同一个包（实测某个人的包里混进两条别人的隐私事实），而界面把
    这块标成「真正追加到 system_prompt 的字节」。真实注入永远带 subject，
    所以预览也必须带：

      1. 运行时冻结缓存里最近生成的那个身份键 —— 和 system_prompt 里的逐字节相同；
      2. 没有缓存时，退回「最近说过话的那个人」。

    scope_all=True 是面板里显式的「全库，会串台，仅调试用」入口，原样返回。
    """
    if scope_all or subject:
        return subject, session_id, False
    best_key, best_at = "", -1.0
    for k, e in (cache_items or []):
        try:
            at = float((e or {}).get("built_at", 0))
        except (TypeError, ValueError):
            at = 0.0
        if at > best_at:
            best_key, best_at = str(k), at
    parts = best_key.split("|")
    if len(parts) >= 3:          # bot|who|sess（开了机器人隔离）
        sub, sid = parts[1], "|".join(parts[2:])
    elif len(parts) == 2:        # who|sess（老 key / 单账号）
        sub, sid = parts[0], parts[1]
    else:
        sub, sid = "", ""
    if sub:
        return sub, (session_id or sid), True
    row = recent_row or {}
    sub = str(row.get("subject") or "").strip()
    if sub:
        return sub, (session_id or str(row.get("session_id") or "")), True
    return "", session_id, False


# 硬锚点：只认单字的实体词，2-gram 机制永远撞不上它们。
#
# 「我有什么病」问的是「病」，value 写的是「慢性胃炎」——「么病」对「胃炎」
# 零重叠，整句召回空。「我几岁」对「今年23岁」同理（「几岁」vs「3岁」）。
# 这类词在中文里就那么几个，而且**只在对应问法下才算数**（「吃饭了吗」里
# 的「吃」不算，否则又要给每句寒暄塞饮食偏好），所以按问法触发、按字匹配。
#
# 权重给 0.45：能过闸，但排在真字面命中之后。单字命中噪音大，不该抢第一。
_CONCEPT_ANCHORS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(p), words) for p, words in (
        (r"多大|几岁|年龄|年纪|岁数", ("岁",)),
        (r"病|症状|健康|吃药|住院|手术|疼|痛|不舒服", ("病", "炎", "医")),
        (r"睡|作息|熬夜|失眠|几点起|起床", ("睡", "醒", "熬夜")),
    )
]

# ---- 称呼槽位：谁是谁 ----
# 中文里「我妈」和「婆婆」都是「妈」，可她们是两个人。系统原本没有「这个
# 人是谁」的概念，全靠字面——只要某条记忆提过「婆婆」，就一样能撞上问句
# 里的「妈」。实测（0.16.18 前）问「我婆婆身体怎么样」，三条含「婆婆」的
# 记忆平分，正确答案「婆婆跟我们一起住，有高血压」被「觉得婆婆管得太多」
# 和「平时都是婆婆做」挤出了包；问「我老公生日是哪天」，直接拿「我自己
# 的生日」回答，因为库里只有那一条含「生日」。
#
# 槽位把称呼归一到「同一个人」：只有落在同一个槽里才算在说这个人。
# 「我妈」→ mother，「我婆婆」→ mil，两者永不互相匹配。
#
# 顺序即优先级：长词必须排在短词前，否则「老公的妈」会被「老公」吃掉、
# 判成丈夫。「他妈」故意归 mil —— 聊天里说「他妈」几乎都是指婆婆。
_KIN_SLOTS: tuple[tuple[str, str], ...] = (
    ("老公的妈", "mil"), ("老公他妈", "mil"), ("婆婆", "mil"),
    ("家婆", "mil"), ("他妈", "mil"),
    ("老公的爸", "fil"), ("公公", "fil"),
    ("孩子他爸", "husband"), ("娃他爸", "husband"),
    ("老公", "husband"), ("丈夫", "husband"), ("爱人", "husband"),
    ("儿子", "child"), ("女儿", "child"), ("孩子", "child"),
    ("小孩", "child"), ("闺女", "child"), ("娃", "child"), ("崽", "child"),
    ("我妈", "mother"), ("妈妈", "mother"), ("母亲", "mother"),
    ("老妈", "mother"), ("娘家", "mother"),
    ("我爸", "father"), ("爸爸", "father"), ("父亲", "father"), ("老爸", "father"),
    ("弟弟", "brother"), ("我弟", "brother"),
    ("哥哥", "brother"), ("我哥", "brother"),
    ("妹妹", "sister"), ("我妹", "sister"),
    ("姐姐", "sister"), ("我姐", "sister"),
    ("小姑子", "sil"), ("小姑", "sil"),
    ("侄女", "niece"), ("侄子", "nephew"),
    ("奶奶", "grandma"), ("外婆", "grandma"),
    ("爷爷", "grandpa"), ("外公", "grandpa"),
)
_KIN_SLOTS_LONGEST_FIRST = tuple(sorted(_KIN_SLOTS, key=lambda x: -len(x[0])))

# 「这条事实的主语是这个人」的词表。只判开头，且允许「我 / 我家 / 家里」这类
# 前缀先剥掉（记忆里常写「我妈…」「家里…」）。
# 和上面那张表分开，是因为判开头必须容忍单字（「妈在梅州」去掉「我」之后
# 只剩「妈」），而全文匹配不能收单字——「他妈」会被拆成 mother。
_HEAD_TERMS: tuple[tuple[str, str], ...] = (
    ("老公的妈", "mil"), ("婆婆", "mil"), ("家婆", "mil"), ("他妈", "mil"),
    ("老公的爸", "fil"), ("公公", "fil"),
    ("老公", "husband"), ("丈夫", "husband"), ("爱人", "husband"),
    ("儿子", "child"), ("女儿", "child"), ("孩子", "child"),
    ("小孩", "child"), ("闺女", "child"), ("娃", "child"), ("崽", "child"),
    ("妈", "mother"), ("娘", "mother"), ("爸", "father"), ("爹", "father"),
    ("弟弟", "brother"), ("哥哥", "brother"),
    ("妹妹", "sister"), ("姐姐", "sister"),
    ("小姑", "sil"), ("侄女", "niece"), ("侄子", "nephew"),
    ("奶奶", "grandma"), ("外婆", "grandma"),
    ("爷爷", "grandpa"), ("外公", "grandpa"),
)
_HEAD_TERMS_LONGEST_FIRST = tuple(sorted(_HEAD_TERMS, key=lambda x: -len(x[0])))
_HEAD_PREFIXES = ("我的", "我家", "我", "咱家", "咱", "家里", "家")


def _kin_slots(text: str) -> set[str]:
    """一段中文里出现过的称呼 → 槽位集合。长词优先，一个字只归一次。"""
    t = str(text or "")
    if not t:
        return set()
    taken = [False] * len(t)
    slots: set[str] = set()
    for term, slot in _KIN_SLOTS_LONGEST_FIRST:
        i = t.find(term)
        while i >= 0:
            if not any(taken[i:i + len(term)]):
                slots.add(slot)
                for k in range(i, i + len(term)):
                    taken[k] = True
            i = t.find(term, i + 1)
    return slots


def _head_slots(text: str) -> set[str]:
    """这条事实的**主语**是谁 —— 只看 value 开头，不看中间。

    「婆婆跟我们一起住，有高血压」→ mil（在讲婆婆）
    「觉得婆婆管得太多」        → 空（在讲我的看法，只是顺带提到她）
    「生儿子的时候是剖腹产」    → 空（是我的经历）
    分不开这两种，问「婆婆身体怎么样」就会被我的评价抢走。
    """
    t = str(text or "").strip()
    for p in _HEAD_PREFIXES:
        if t.startswith(p):
            t = t[len(p):].lstrip("的了")
            break
    for term, slot in _HEAD_TERMS_LONGEST_FIRST:
        if t.startswith(term):
            return {slot}
    return set()

# 态度谓词：喜欢 / 觉得 / 认为 / 想要 / 知道。它们在记忆里出现的频率极高
# （几乎每条评价都带一个），而且**不指向任何具体对象**——「喜欢」本身不是
# 话题，「喜欢什么」才是。
#
# 实测「我最喜欢什么颜色」：靠「喜欢」这一个词撞出三条「喜欢 X」的记忆
# （推理小说 / 说话直说 / 胶片摄影），模型拿着它们会答「你喜欢胶片摄影」，
# 问颜色答摄影，比空手更糟。库里根本没记过颜色。
#
# 不放进 _STOP_GRAMS：它们是别的好句子里唯一的锚点（「我讨厌什么样的人」
# 靠「讨厌」找到那条评价）。这里只在**整句命中全是这类词**时判定「这轮
# 没有可检索的实体」，闸门放行与否看 _rel_strong。
_WEAK_CHARS = set("喜欢觉得认想要知道感觉在意希望")


def _is_weak_gram(g: str) -> bool:
    """除功能字外全是态度字 → 这个 bigram 不指向任何实体。"""
    chars = [c for c in str(g) if c not in _FUNC_CHARS]
    return bool(chars) and all(c in _WEAK_CHARS for c in chars)


class PackBuilder:
    def __init__(self, store, l3_snippets: int = 0, excluded_limit: int = 5,
                 l2_limit: int = 12, subject: str = "", session_id: str = "",
                 extra_subjects: list[str] | None = None,
                 rel_ratio: float = 0.3, l2_keep_min: int = 3, l3_keep_min: int = 0,
                 outbound_limit: int = 4, cross_limit: int = 5,
                 outbound_labeled_only: bool = True, outbound_pinned: bool = True,
                 outbound_relay_limit: int | None = None,
                 outbound_cron_limit: int | None = None,
                 outbound_relay_window_hours: float = 24.0,
                 outbound_cron_window_hours: float = 24.0,
                 outbound_away_limit: int = 1,
                 outbound_away_cron_limit: int | None = None,
                 cross_include_self: bool = False,
                 query_expand: bool = True,
                 semantic_fn=None,
                 skeleton_limit: int = 0,
                 bot_id: str = "",
                 lite: bool = False,
                 logger=None,
                 candidate_limit: int = 200,
                 context_rounds: int = 3,
                 mmr_enabled: bool = True,
                 idf_enabled: bool = True):
        # 日志由调用方注入：运行路径在 main.py 传入从 astrbot.api 导入的 logger。
        # 本模块不导入任何日志库（内置日志模块 / 第三方日志库都不用），
        # 离线测试直接构造 PackBuilder 时 logger 为 None，裁剪告警静默。
        self.logger = logger
        self.store = store
        # 机器人命名空间（unified_msg_origin 第一段）。多账号共用一份库时，
        # 外发/原话/骨架都必须按它过滤：否则 A 机器人给用户发的定时提醒会
        # 出现在 B 机器人的记忆包里，被 B 当成「我自己说过的话」。
        # 空串 = 不限（单账号老库/面板预览的兼容行为）。
        self.bot_id = str(bot_id or "").strip()[:32]
        # lite：只跑 L2 相关性打分 + _count_hit 标记，给「统计轨」用。
        # A 方案后统计轨的结果不进 system_prompt，只为面板冷热计数服务，
        # 没必要每轮再拉 l3(60) / 骨架(2000) / outbound / cross 那一堆。
        # 影子模式和 dynamic 对照模式仍走全量。
        self.lite = bool(lite)
        self.l3_snippets = int(l3_snippets)
        self.excluded_limit = int(excluded_limit)
        self.l2_limit = int(l2_limit)
        # 跨会话两条通道，都是「不管当前是谁」都能看到的：
        #   outbound = 你自己主动发到别的会话的话（转述、定时通知）。
        #     不注入这段，AI 换个会话被问「你刚才跟 X 说了啥」就查不到任何记录，
        #     会当成自己编的，然后否认。这是「转述被当成假的」的根因。
        #   cross = 别的会话里别人说的话，按本轮话题筛过再给，免得串台。
        self.outbound_limit = max(0, int(outbound_limit))
        # 转述 / 定时各带几条。以前是一个 outbound_limit 混着算，模型看到的
        # 一段里既有「主人让我带的话」又有「闹钟替我发的」，问答时会串。
        # 现在两个独立额度；不显式传就退回 outbound_limit（老测试脚本兼容）。
        self.outbound_relay_limit = max(
            0, int(outbound_limit if outbound_relay_limit is None else outbound_relay_limit))
        self.outbound_cron_limit = max(
            0, int(outbound_limit if outbound_cron_limit is None else outbound_cron_limit))
        # 外发的新鲜度窗口（小时）。转述/定时都是「AI 不知道自己刚才干了啥」的
        # 短期答案：过一两天没人再问 10 天前的闹钟发了什么，老条目却因为「最近
        # 没有新的」一直占着常驻位还 pinned 不许裁，记忆包越来越陈旧臃肿。
        # 窗内照旧常驻进包；过窗的降级成普通外发回声（labeled_only 下不再进包，
        # 关掉 labeled_only 的高级模式里还能看到、但不再 pinned）。
        # 两个池子各配各的窗，0 = 不过期（老行为）。跨会话的追问补位不受窗口——
        # 它只有本轮真的在问才带，平时零 token，正是过窗内容唯一的召回路径。
        self.outbound_relay_window = max(0.0, float(outbound_relay_window_hours)) * 3600.0
        self.outbound_cron_window = max(0.0, float(outbound_cron_window_hours)) * 3600.0
        self.cross_limit = max(0, int(cross_limit))
        # cross 是否也带 AI 自己在别的会话说过的话（bot_reply）。AI 一句往往比人多、
        # 信息量少，默认关；用户要「我记得我在别的会话说过啥」时再开。
        self.cross_include_self = bool(cross_include_self)
        # 外发段的两道闸（默认都开，走「平时不占预算、触发了就顶到最前」）：
        #   labeled_only = 只有真定时任务/转述才进包。平时那句「你主动发给xxx」
        #     多半是分段回复的回声，既占预算又没信息量——没触发定时任务就整段不进。
        #   pinned = 进包的那几条不参与 token 裁剪，预算再紧也留着。
        self.outbound_labeled_only = bool(outbound_labeled_only)
        self.outbound_pinned = bool(outbound_pinned)
        # 跨会话补位：labeled_only 挡掉了「发到别的会话」的转述，但「你刚才跟 X
        # 说了什么」这一问的答案恰恰就在那里面。这里留一个只在被追问时才开的
        # 小口子，条数默认 1——不问就不带，这是省 token 的底线。0 = 连追问也不补。
        self.outbound_away_limit = max(0, int(outbound_away_limit))
        # 定时任务的跨会话补位单独一份额度，默认跟转述一致。分开的好处：
        # 问「你给 B 带什么话了」时不会补进来一条闹钟内容。
        self.outbound_away_cron_limit = max(
            0, int(outbound_away_limit
                   if outbound_away_cron_limit is None else outbound_away_cron_limit))
        # 相关性门槛：低于「本轮最高命中数 × rel_ratio」的 L2 直接不进池。
        # 0 或负数 = 关闭过滤（回到「有钱就全塞」的旧行为）。
        # 为什么要这道闸：预算是有限资源，一句「在吗，今天干嘛呢」不该把
        # 十几条跟本轮无关的事实全拽进来，把原话和摘要挤死。
        self.rel_ratio = float(rel_ratio or 0)
        self.l2_keep_min = max(1, int(l2_keep_min))
        self.l3_keep_min = max(0, int(l3_keep_min))
        # query 概念归一开关（expand_query）：把「多大了」这类改写问法映射成
        # 概念词拼到检索文本后，补 2-gram 字面匹配的盲区。关掉回到纯字面。
        self.query_expand = bool(query_expand)
        # 语义兜底钩子（可选）：字面/概念/别名全部零命中时才调用。
        # 签名 semantic_fn(query_raw, rows) -> {l2_id: 0~1 分}。
        # 线上由 main 在异步上下文里预计算好分数，这里只做纯查表，不碰网络。
        # 没配 embedding provider 时传 None，整条语义路完全不通电。
        self.semantic_fn = semantic_fn
        # L2 候选池上限：SQL 先按作用域/状态/有效期过滤，不按置信度 LIMIT，
        # 再由 _score_l2 按本轮相关度取用。0 = 不设上限（全库）。
        self.candidate_limit = max(0, int(candidate_limit or 0))
        # 最近几轮用户原话参与检索（只给检索，不进 _asks_* 问法判断）。
        self.context_rounds = max(0, int(context_rounds or 0))
        # IDF 加权 + 查询内 min-max 归一 / L2 近重复去重，都能单独回退。
        self.idf_enabled = bool(idf_enabled)
        self.mmr_enabled = bool(mmr_enabled)
        # 旧事骨架常驻条数：l3_skeleton 行（原话过期后压成的主题词索引）。
        # 常驻不按 query 挑——骨架就几个词，按相关性挑会重演「改写零命中
        # 永远召不回」，而它一行才 ~30 token，常驻 N 条的成本可预期。
        # 0 = 不注入（骨架只在库里躺着，供 memory_search 查）。
        self.skeleton_limit = max(0, int(skeleton_limit))
        # 作用域：subject 非空 = 只取这个人的记忆；session_id 非空 = 摘要取本会话的。
        # 两者都为空 = 旧的全库行为（会串台，仅供面板调试用）。
        self.subject = str(subject or "").strip()
        self.session_id = str(session_id or "").strip()
        # 这轮对话里被点名的其他人：A 问「B 是不是说过……」时答案在 B 的档案里，
        # 只取 A 自己的记忆等于没取。上限 3 个，且绝不包含说话人自己。
        self.extra_subjects = [
            str(s).strip() for s in (extra_subjects or [])
            if str(s).strip() and str(s).strip() != self.subject
        ][:3]
        # 人物归一缓存：self.subject 可能是某个昵称（路人甲），库里同一个人可能
        # 记在别的昵称（张三）名下。缓一份「归一后的 person + 全部别名」，cross 通道
        # 用它判断「这条是不是我自己说的」，避免同人换昵称后被当成「别人」。
        self._self_person = ""
        self._self_aliases: set[str] = set()
        if self.subject:
            try:
                self._self_person = str(store.person_resolve(self.subject) or "") or self.subject
                info = store.person_get(self._self_person)
                for a in (info or {}).get("aliases") or []:
                    self._self_aliases.add(str(a.get("raw") or "").strip())
                    self._self_aliases.add(str(a.get("alias_key") or "").strip())
            except Exception:
                self._self_person = self.subject
            self._self_aliases.add(self.subject)

    def _is_self(self, name: str) -> bool:
        """判断某个 subject 是不是「当前说话人自己」（含同人换昵称）。"""
        n = str(name or "").strip()
        if not n or not self.subject:
            return False
        if n == self.subject:
            return True
        # 别名命中（raw 或 alias_key 任一形式）
        if n in self._self_aliases:
            return True
        # 归一后同 person
        try:
            r = self.store.person_resolve(n)
            if r and self._self_person and r == self._self_person:
                return True
        except Exception:
            pass
        return False

    def _score_l2(self, rows: list[dict], query: str,
                  context: str = "") -> list[dict]:
        """给 L2 候选打分。

        三路命中必须分开计数，绝不合并成一个数（0.16.16 的关键修正）：
          本轮本体 qg∩content —— 主判据，决定「这轮到底在说什么」；
          历史本体 cg∩content —— 只给指代追问（「那它呢」）补位，不能顶替本轮；
          问法别名 qall∩alias —— 兜底通道，整句口语别名撞上一个泛用词不算命中。
        合并的后果实测过：「吃东西」那轮，上一轮的「插件开发」还在
        recent_queries 里，#45「开发了记忆系统插件」靠两个历史词拿到
        w=2.16，排在有真命中的「不喜欢吃泡面」前面，被补进尾部当噪声。

        另外四条老规矩保持不变：
          1) bigram 命中按 IDF 加权（候选池内的文档频率），「的/了/是」自动降权；
          2) 加权相关度在本查询内 min-max 归一，长 value 不再靠字数天然领先；
          3) 置信度 / 保留度各自只占 0.15，不再当常量偏移主导排序；
          4) 本轮原话仍是主判据，历史只按折扣补位。
        _rel 的语义已改为「本轮本体命中数」，_filter_by_relevance 的闸门、
        _pick_l2 的「本轮真的被提到」判据都用它。
        """
        # 本轮原话 和 概念归一拼上来的词，分开算。
        #
        # 概念词是「猜测」：把「多大」猜成「岁」、把「干啥的」猜成「工作」。
        # 猜得中能救回零重叠的整句，猜不中就直接制造误命中——实测问「我在
        # 哪上班」，概念词「工作」点亮了「想换工作，嫌现在厂里太远」（3.26
        # 分），真答案「在顺德一家五金厂做会计」只能靠别名拿 1.96 分，问在
        # 哪上班答想换工作。原话里一个字都没提到的东西，凭什么比别名更硬。
        _raw_q, _sep, _concept_q = str(query or "").partition(_CONCEPT_MARK)
        # 接续词在检索里也要剥掉，不只是判「有没有话题」时剥。实测「后来呢」
        # 期望接着上一轮说手术，结果被一条「在地铁上丢过一部手机，**后来**
        # 没找回来」抢走——记忆里出现「后来」是正常叙述，问句里出现「后来」
        # 就只是「接着说」。
        qg_raw = _content_grams(_CONT_RE.sub("", _raw_q))
        qg_con = _content_grams(_concept_q) if _sep else set()
        qg = qg_raw | qg_con
        cg = _content_grams(context) if context else set()
        qall = qg | cg
        # 单字锚点只在「本轮原话」里判，不看历史：上一轮问过病，这一轮
        # 只是在聊别的，不该继续往包里塞病历。
        raw_q = _raw_q
        anchor_words: list[str] = []
        for _pat, _words in _CONCEPT_ANCHORS:
            if _pat.search(raw_q):
                anchor_words.extend(w for w in _words if w not in anchor_words)
        # 问句点了谁。没点人（「我有什么病」）就是空集，下面所有主体逻辑跳过。
        q_slots = _kin_slots(raw_q)
        # 上一轮点的是谁。「那他呢」「后来呢」本轮没有主体，得接着上一位说，
        # 否则 IDF 会把「想换工作」这种稀有词派上去，答成我自己的工作。
        h_slots = _kin_slots(context) if (context and not q_slots) else set()
        grams = [_content_grams(_l2_text(r)) for r in rows]
        cgrams = [_content_grams(_l2_content(r)) for r in rows]
        agrams = [_content_grams(_l2_alias(r)) for r in rows]
        # 主体名单独一路：问「老周呢」「老周最近怎么样」时，全句唯一的实词
        # 就是人名，而这个字只出现在 subject 里、不在 value 里（value 记的是
        # 「本名周维，做装修生意」）。不算这一路，点名人名等于没点名——实测
        # 「老周是干什么的」整句零命中，返回空。
        # 只对**别人**生效：自己名字出现在句句里都可能（「阿岚今天好累」），
        # 算了会让全库每一行都 +1，闸门形同虚设。
        sgrams = [_content_grams(str(r.get("subject") or "")) for r in rows]
        # 句末焦点：中文问句的信息重心通常在最后几个字（「…加糖吗」的重点是
        # 糖，「…辞职吗」的重点是辞职）。命中它的行给一点加分，用来分开
        # 「沾到同一个话题」和「答的就是这一问」——实测「我打算辞职吗」，
        # 「打算明年搬到杭州」靠「打算」两个字和「在考虑辞职」打平，然后靠
        # 置信度赢了，答非所问。只加分，不做闸门。
        # 窗口只取最后 3 个字。取 5 个会把「打算辞职吗」里的「打算」也算进
        # 焦点，「打算明年搬到杭州」跟着白拿一份加分，和「在考虑辞职」打平。
        tail_g = _content_grams(raw_q[-3:]) - _STOP_GRAMS if raw_q else set()
        if not qall:
            for r in rows:
                r["_rel"] = 0
                r["_rel_content"] = 0
                r["_rel_ctx"] = 0
                r["_rel_alias"] = 0
                r["_rel_name"] = 0
                r["_rel_w"] = 0.0
                base = (float(r.get("confidence", 0) or 0)
                        + min(int(r.get("stability", 0) or 0), 5) * 0.1
                        + retention_score(r) * 0.3)
                w = _RESIDENT_WEIGHT.get(
                    str(r.get("predicate") or "").strip(),
                    _RESIDENT_WEIGHT_DEFAULT)
                r["_score"] = base * w
                r["_why"] = {"channel": "none", "bigram": 0, "idf": 0.0,
                             "norm": 0.0, "intent": "", "mmr": "",
                             "resident_w": w}
            return sorted(rows, key=lambda r: r["_score"], reverse=True)
        n = max(1, len(rows))
        if self.idf_enabled:
            # df 只在「本体」里统计：alias 是模型生成的三句口语，算进 df
            # 会把「东西 / 其实」这类词的文档频率抬高、权重压低，但它们本来
            # 也不该靠 alias 拿分，压不压无所谓；真事实词的 df 反而更干净。
            w = {}
            for g in qall:
                df = sum(1 for s in grams if g in s)
                w[g] = math.log(1.0 + n / (1.0 + df))
        else:
            w = {g: 2.0 for g in qall}
        intent_rel = bool(_REL_Q.search(query))
        intent_time = bool(_TIME_Q.search(query))
        intent_fact = bool(_FACT_Q.search(query))

        def _sat(hs: set[str]) -> float:
            # 长 value 天然能蹭到更多 bigram，裸求和会让它永远压过
            # 「喜欢喝拿铁」这种只命中一两个稀有词的短事实。按命中数做
            # 次线性饱和：命中越多边际收益越低，短答案才有机会排上来。
            return (sum(w.get(g, 0.0) for g in hs)
                    / (1.0 + 0.35 * max(0, len(hs) - 1)))

        rel_ws: list[float] = []
        for r, tg, ctg, ag, sg in zip(rows, grams, cgrams, agrams, sgrams):
            _val = str(r.get("value") or "")
            # 这条在说谁。问句点了人时，槽位对不上就是「答的不是这个人」。
            # subject 也算：subject=「婆婆」的档案行，value 里写的是「本名
            # 李桂芳」，不提称呼，但这条事实本来就是婆婆的。
            _subj_slots = _kin_slots(str(r.get("subject") or ""))
            r_slots = _kin_slots(_val) | _subj_slots
            r["_slots"] = r_slots
            same = q_slots & r_slots if q_slots else set()
            hits_q = (qg_raw & ctg) - _STOP_GRAMS      # 本轮原话：主判据
            # 概念词猜出来的命中单独一路、单独打折。它回答的是「这句可能在问
            # 什么」，不是「这句提到了什么」。
            hits_con = ((qg_con & ctg) - hits_q) - _STOP_GRAMS
            hits_c = ((cg & ctg) - hits_q - hits_con) - _STOP_GRAMS   # 历史本体：只补位
            # 别名只认本轮问法：历史词走 ctx 通道，混进来会让上一轮的话题
            # 借别名的壳复活（「吃东西」+ 上轮「拿铁」，别名里撞到「东西」和
            # 「拿铁」两个 bigram，就够格当兜底了）。概念词也不进：别名是模型
            # 写的口语，概念词是规则猜的，两个猜的东西叠一起等于自证。
            hits_a = ((qg_raw & ag) - hits_q) - _STOP_GRAMS    # 本轮撞上别名：兜底
            # 人名通道：只算别人。名字命中说明「这一条就是在说被点名的人」，
            # 但不算内容被提起（_rel_content 仍是 0，统计轨不会因此加一次）。
            hits_n = set()
            if (str(r.get("subject") or "").strip()
                    and str(r.get("subject") or "").strip() != self.subject):
                hits_n = (qg_raw & sg) - hits_q - _STOP_GRAMS
            # 单字锚点：只在本轮问法命中时才算，且问某位亲属时只认**这一位**
            # （问「我妈有什么病」，自己那条「慢性胃炎」也含「病」，婆婆那条
            # 含「高血压」——都不该靠一个「病」字混进来）。
            aw: list[str] = []
            if anchor_words and (not q_slots or same):
                aw = [w for w in anchor_words if w in _val]
            # 主体权重：同一句话，主语是这个人 和 顺带提到这个人，差别很大。
            # 「婆婆跟我们一起住，有高血压」才是答案；「觉得婆婆管得太多」
            # 是态度；「平时都是婆婆做」是家务分工。三条都含「婆婆」，字面
            # 分不开，只能靠抬头判。
            slot_w = 0.0
            if q_slots:
                # subject 本身就是这个人（「婆婆」名下的档案行）和「value 抬头
                # 是这个人」等价：那条事实本来就是他的，不该因为 value 写成
                # 「本名李桂芳，今年六十八」不带称呼就被当成顺带一提。
                if (_head_slots(_val) & q_slots) or (_subj_slots & q_slots):
                    slot_w = 0.80
                elif same:
                    slot_w = 0.25
            # 纯指代追问（「那他呢」「后来呢」）：沿用上一轮那个人。
            ctx_slot_w = 0.0
            if h_slots:
                if (_head_slots(_val) & h_slots) or (_subj_slots & h_slots):
                    ctx_slot_w = 0.80
                elif r_slots & h_slots:
                    ctx_slot_w = 0.20
            # 折扣写在这里而不是 w 里：w 只放 idf，三种命中的量纲才可比。
            # 别名只在**这一行自己没有任何本体/锚点命中**时才算数——兜底通道
            # 的语义就是「别的路都走不通才用它」。整池一起判太粗：实测
            # 「我喝咖啡加糖吗」，池子里有「空腹喝咖啡」那条命中了咖啡，于是
            # 别名通道被整体关掉，真正答「不加糖」的那条反而掉到第二。
            _alias_w = 0.30 if not (hits_q or hits_con or aw) else 0.0
            # 句末焦点命中按 0.55 计入相关度：它回答的是「问的那一点」。
            # 平摊成固定加分不够用——「我喝咖啡加糖吗」里「空腹喝咖啡」命中
            # 「喝咖 + 咖啡」两个词，归一后直接 1.0，压过只命中「加糖」的
            # 那条正确答案。焦点词必须能翻盘，才算真的在看这一问。
            rw = (_sat(hits_q) + 0.55 * _sat(hits_con) + 0.45 * _sat(hits_c)
                  + _alias_w * _sat(hits_a)
                  + 0.70 * _sat(hits_n) + 0.45 * min(3, len(aw))
                  + 0.55 * _sat(hits_q & tail_g) + slot_w + ctx_slot_w)
            rel_ws.append(rw)
            r["_rel"] = len(hits_q) + len(hits_con) + len(hits_n) + len(aw)
            r["_rel_content"] = len(hits_q) + len(hits_con) + len(aw)
            r["_rel_ctx"] = len(hits_c)
            r["_rel_alias"] = len(hits_a)
            r["_rel_name"] = len(hits_n)
            r["_rel_concept"] = len(hits_con)
            # 实体命中数：扣掉纯态度词。闸门看这个，不看 _rel ——
            # 否则一句「我最喜欢什么颜色」靠「喜欢」就能过闸进包。
            # 主体相符也算一票：「我生过孩子吗」字面撞不到「生儿子的时候是
            # 剖腹产」，但那条确实是答案，放它过闸才有机会被看到。
            r["_rel_strong"] = (sum(1 for g in (hits_q | hits_con)
                                    if not _is_weak_gram(g))
                                + len(hits_n) + len(aw) + (1 if same else 0))
            r["_rel_w"] = rw
        lo, hi = min(rel_ws, default=0.0), max(rel_ws, default=0.0)
        span = hi - lo
        concept = _CONCEPT_MARK in query
        for r, rw, ctg in zip(rows, rel_ws, cgrams):
            norm = (rw - lo) / span if span > 0 else 0.0
            conf = float(r.get("confidence", 0) or 0)
            ret = retention_score(r)
            bonus, why_intent = 0.0, ""
            pv = f"{r.get('predicate') or ''}{r.get('value') or ''}"
            if intent_rel and _REL_PRED.search(pv):
                bonus, why_intent = 0.08, "relation"
            elif intent_time and r.get("_rel_content"):
                bonus, why_intent = 0.05 * min(1.0, ret), "time"
            elif intent_fact and r.get("_rel_content"):
                bonus, why_intent = 0.04, "fact"
            if tail_g & ctg:
                why_intent = why_intent or "tail"
            r["_score"] = 0.70 * norm + 0.15 * conf + 0.15 * ret + bonus
            # 主体冲突：问句点了某个人，这条记忆里压根没有这个人（提的是别人，
            # 或者通篇在讲我自己）→ 不是答案。实测「我老公生日是哪天」，库里
            # 只有「生日是农历八月十六」，字面命中「生日」「日是」，0.93 分
            # 稳坐第一，模型就会答「你老公生日是农历八月十六」。
            #
            # 只降权不删除：「我妈的糖尿病和我胃病哪个严重」需要两边都在，
            # 一刀切掉就答不成对比题了。0.55 足够让 0.93 的噪声让位给 0.70
            # 的正确答案，又不至于把它踢出包。
            subj_pen = 1.0
            if raw_q.strip():
                if q_slots:
                    if not ((r.get("_slots") or set()) & q_slots):
                        subj_pen = 0.55
                        # 标记「问的是别人，这条不是那个人」。排除是后面
                        # _filter_by_relevance 的事：只有池里确有这个人的
                        # 档案时才敢排掉别人的，否则会把仅有的线索也删光。
                        r["_subj_off"] = True
                elif not h_slots and _head_slots(str(r.get("value") or "")):
                    # 问句里没点人 = 在问我自己。主语是「妈妈」「弟弟」「婆婆」
                    # 的行是**别人的档案**，不能拿去回答「我有什么病」——
                    # 实测排第一的是「妈妈今年五十二，有糖尿病」，模型张口就
                    # 说「你有糖尿病」。
                    # 只看抬头，不看整句：自己那条「生完孩子胖了二十斤」中间
                    # 也有「孩子」，但它讲的是我的事，不该降权。
                    # h_slots 非空是例外：那是「那她呢」「后来呢」这种纯指代
                    # 追问，上一轮点的就是别人，这正是要捞的档案。
                    subj_pen = 0.55
            if subj_pen != 1.0:
                r["_score"] = float(r["_score"]) * subj_pen
            # 同分破平：两条都只撞到同一个泛词时（「我一个月挣多少钱」，
            # 「房贷每个月还四千二」和「一个月工资六千八」都只撞「个月」），
            # 归一后双双 1.0、连 conf 和 stability 都一样，谁排前面全看
            # 数据库返回顺序。这时别名命中数是最可靠的区分：模型给这条写了
            # 五句口语问法，恰好撞上这一问的，比只碰上一个「个月」更像答案。
            # 系数压到 1e-3 量级，只在真同分时起作用，不改变正常排序。
            r["_score"] = (float(r["_score"])
                           + 0.0012 * min(6, int(r.get("_rel_alias") or 0)))
            rel_q = int(r.get("_rel") or 0)
            ch = ("concept" if rel_q and concept
                  else "literal" if rel_q
                  else "context" if int(r.get("_rel_ctx") or 0)
                  else "alias" if int(r.get("_rel_alias") or 0)
                  else "none")
            r["_why"] = {
                "channel": ch, "bigram": rel_q, "idf": round(rw, 3),
                "norm": round(norm, 3), "confidence": round(conf, 3),
                "retention": round(ret, 3), "intent": why_intent,
                "bonus": round(bonus, 3), "mmr": "", "consensus": 0.0,
                "tail": 1 if (tail_g & ctg) else 0,
                "subj_pen": subj_pen,
                "rel_ctx": int(r.get("_rel_ctx") or 0),
                "rel_alias": int(r.get("_rel_alias") or 0),
                "score": round(float(r["_score"]), 4),
            }
        return sorted(rows, key=lambda r: r["_score"], reverse=True)

    def _apply_mmr(self, rows: list[dict], lam: float = 0.7) -> list[dict]:
        """L2 之间去冗余：同一件事被总结成三条近重复会白占三个名额。

        只比 value。**不能带 ask_alias** —— 重叠系数是 |交集| / 较短者，
        别名是模型写的三句口语（动辄 40+ 个 bigram），把它算进来分母被撑到
        两倍多，#9「开发了记忆插件」和 #45「开发了记忆系统插件」这种真近重复
        只有 0.1，永远判不出来（实测就是这样，两条一起占掉尾部两个名额）。
        带独有数字的细节（「九十分钟」）视为不同信息，不让位。
        """
        if not self.mmr_enabled or len(rows) < 2:
            return rows
        kept: list[dict] = []
        kept_grams: list[set[str]] = []
        kept_units: list[set[str]] = []
        for r in rows:
            text = str(r.get("value") or "")
            g = _bigrams(text)
            units = set(_NUM_RE.findall(text))
            dup = False
            for sg, su in zip(kept_grams, kept_units):
                # 重叠系数（|交集| / 较短者）而不是 Jaccard：Jaccard 对
                # 「同一句多几个字」惩罚太重，「喜欢喝拿铁」和「喜欢喝拿铁不加糖」
                # 只有 0.67，永远判不成重复。骨架让位用的也是这一套。
                shared = len(g & sg)
                ov = shared / max(1, min(len(g), len(sg)))
                if shared >= 2 and ov > lam and not (units - su):
                    dup = True
                    break
            if dup:
                r.setdefault("_why", {})["mmr"] = "duplicate"
                continue
            r.setdefault("_why", {})["mmr"] = "kept"
            kept.append(r)
            kept_grams.append(g)
            kept_units.append(units)
        return kept

    def _filter_by_relevance(self, rows: list[dict], query: str = "") -> list[dict]:
        """按本轮话题命中数砍掉「压根没被提到」的条目。

        只看命中数（_rel 本轮本体），不看 _score：_score 里混了置信度和稳定性，
        即使一句话都没命中也会有 0.9 分，用它做阈值等于没设。

        本体零命中有四级退路，顺序是「越准的越前」：
          语义钩子（若配了 provider）→ 问法别名（撞 ≥2 词）→
          历史本体（仅当本轮挑不出实词）→ 按 _score 保底 3 条。
        保底那几条 _rel 是 0，注入轨的候选条件会把它们挡在尾部之外，
        只用于「常驻包总得有几条垫底」。
        """
        if self.rel_ratio <= 0 or not rows:
            return rows
        qg = _content_grams(query)
        q_slots_ctx = _kin_slots(query)
        top = max(int(r.get("_rel_strong", r.get("_rel", 0)) or 0) for r in rows)
        if top <= 0:
            keep = min(self.l2_keep_min, len(rows))
            # 语义兜底：字面/概念/别名全零命中时，问 semantic_fn 要一个
            # {id: 分} 的语义相关度，挑分最高且过阈值的几条。没有钩子或
            # 全不过线，退回按置信度保底（旧行为）。
            if self.semantic_fn is not None and keep < len(rows):
                try:
                    sem = self.semantic_fn(query, rows) or {}
                except Exception:
                    sem = {}
                if sem:
                    scored = [(float(sem.get(int(r.get("id", 0)), 0.0)), r)
                              for r in rows]
                    picked = [r for s, r in sorted(scored, key=lambda x: -x[0])
                              if s >= 0.5][:keep]
                    if picked:
                        for s, r in ((float(sem.get(int(r.get("id", 0)), 0.0)), r)
                                     for r in picked):
                            # 语义命中也是 query 驱动的，算「被想起」，计数标记
                            r["_sem"] = True
                            r.setdefault("_why", {})["sem"] = round(s, 3)
                            r.setdefault("_why", {})["channel"] = "semantic"
                            # 抬 _score，避免这些行在 trim 阶段被当成最弱先裁掉；
                            # 但压在字面命中行之下（_rel>=1 意味着 _score>=2）
                            r["_score"] = max(float(r.get("_score", 0) or 0),
                                              0.4 + 0.6 * s)
                        return picked
            # 本体零命中不等于没话题：「多大了」在值里撞不到任何 bigram，
            # 但 alias（模型写的口语问法「今年多大了呀」）能撞上。此时才
            # 允许别名兜底，且要求至少 2 个 bigram —— alias 是整句口语，
            # 只撞一个词纯属巧合（「嘿嘿想你了」撞别名里的「你了」）。
            # 本体有命中的轮次完全不看 alias，这就是 0.16.16 堵住噪声的闸门。
            alias_rows = sorted(
                (r for r in rows if int(r.get("_rel_alias") or 0) >= 2),
                key=lambda r: -float(r.get("_rel_w") or 0))
            if alias_rows:
                return alias_rows[:keep]
            # 再退一步：只有本轮这句话里压根挑不出实词（「那它呢」「后来呢」
            # 「那个呢」），才用历史轮的原话补位。本轮有实词（「吃东西」）却
            # 零命中，说明库里没这条，别拿上一轮的话题硬凑 —— 否则换了话题
            # 还挂着上一条，用户一眼就看出是噪声。
            # 判据是「剥掉接续词后还剩不剩实词」：「后来呢」剩个「呢」，
            # 不算话题，得接着上一轮说；原来的 `not qg` 判不出来，因为
            # 「后来」本身是两个实字。
            if not _query_has_topic(query):
                ctx_rows = sorted(
                    (r for r in rows if int(r.get("_rel_ctx") or 0) > 0),
                    key=lambda r: -float(r.get("_rel_w") or 0))
                if ctx_rows:
                    return ctx_rows[:keep]
            # 整句话一个 2-gram 都没命中：判断不了相关性。
            #
            # **有 query 就一条不垫**（0.16.17）。原来这里无条件返回
            # rows[:keep]，而 rows 按 _score 排，头部恒为「自我相关 + 高稳定」
            # 那几条，于是每一句「在吗」「嘿嘿想你了」「吃饭了吗」都会带上
            # 「本名陈岚 / 今年23岁 / 现在住在成都」——206 token 的固定身份卡，
            # 九句问候九次一模一样。这不是「保底」，是每轮强行背资料卡。
            # 真没答案的句子，空着比塞三条无关事实好：模型手里没有资料，
            # 就不会在一句打招呼里突然自我介绍。
            #
            # query 为空是另一回事：常驻包 / 面板预览本来就没有话题，那几条
            # 永久事实正是它该有的内容，保底逻辑保留。
            if str(query or "").strip():
                return []
            return rows[:keep]
        # top 取自整个候选池（0.16.0 起不再预截断），一条长 value 蹭到 6~8
        # 个 bigram 就能把阈值抬到 2.4，短答案只命中 1~2 个的会被误杀。
        # 天花板按本轮 query 自己的 bigram 数算，再压一个绝对上限 3：
        # 命中 3 个已经是强信号，再往上要求只会漏召回。放宽闸门不会把噪声
        # 放进包——后面还有 _pick_l2 按 _score 取前 N 和预算裁剪。
        cap = max(1, len(qg)) if qg else top
        th = min(2.0, max(1.0, min(top, cap) * self.rel_ratio))
        kept = [r for r in rows
                if int(r.get("_rel_strong", r.get("_rel", 0)) or 0) >= th]
        # 问的是某个人，池里确实有这个人的档案 → 别人的一律不进包。
        # 只靠 0.55 降权不够：「我老公生日是哪天」里我自己的生日 0.93 分，
        # 乘完还有 0.51，照样压过 0.70 分的伴侣档案，模型就会答错人。
        # 多意图（「我妈的糖尿病和我胃病哪个严重」）不排，两边都要。
        if q_slots_ctx and not _MULTI_RE.search(query):
            mine = [r for r in kept if not r.get("_subj_off")]
            if mine and len(mine) < len(kept):
                kept = mine
        # 短事实往往只有一两个 bigram 命中，可那可能正是答案本身。长噪声
        # 把 top 撑高时，相对阈值等于宣布「短事实永不进包」。显式把内容
        # 真的命中过的落选行按分数补回来，后面还有 _pick_l2 和预算裁剪，
        # 不会因为放宽闸门把包撑大。
        kept_ptr = {id(r) for r in kept}
        # 补回来的有两类，原来只补第一类：
        #   ① 本体真命中过、被相对阈值卡掉的短事实；
        #   ② 靠问法别名撞了 ≥2 个词的兜底行。
        # 第二类原来只在「整池零命中」时才启用，于是池子里只要有一条泛词命中
        # 就全废：实测问「我以前做什么工作」，先被拼上概念词「工作」，命中
        # 一条「想换工作」，于是真答案「以前在电子厂做过六年统计」（别名撞了
        # 5 个词）连包都进不来。
        weak = [r for r in rows if id(r) not in kept_ptr
                and not r.get("_subj_off")
                and (int(r.get("_rel_strong", r.get("_rel_content", 0)) or 0) > 0
                     or (int(r.get("_rel_alias") or 0) >= 2
                         # 别名兜底不许跨人：问「我婆婆喜欢吃什么」，一条
                         # 「特别爱吃甜的」靠别名「我爱吃什么」也撞了两个
                         # 词，照样被补进包 —— 问婆婆口味答我爱吃甜的。
                         and (not q_slots_ctx
                              or (r.get("_slots") or set()) & q_slots_ctx)))]
        if weak:
            kept.extend(sorted(weak,
                               key=lambda r: -float(r.get("_score") or 0))
                        [:max(1, self.l2_keep_min)])
        if len(kept) >= min(self.l2_keep_min, len(rows)):
            return kept
        # 过滤太狠（只剩一两条）时，原来会补足到保底条数。有 query 就不补：
        # 只有 1 条真答案就交 1 条，凑满 3 条只会把「本名/年龄」这类零相关
        # 的常驻事实塞回包里。query 为空（常驻包 / 面板预览）才需要垫底。
        if str(query or "").strip():
            return kept
        return rows[: min(self.l2_keep_min, len(rows))]

    def _pick_l2(self, rows: list[dict], pool: list[dict],
                 limit: int) -> list[dict]:
        """按分类配额挑长期事实：给「人的底档」留几个固定名额。

        身份、疾病、年龄、联系方式这类事实本来就不该每轮被提起，但它们
        必须一直在。只按话题相关性排的话，它们永远挤不进前 12 条，在面板
        上永远显示「❄ 冷」——不是没人提，是根本没机会被提。将来按热度淘汰，
        第一批死的就是它们。

        所以先看头部里永久分类占了几条，不够就按分数从候选池补，
        挤掉末尾的非永久条目。挤的永远是排名最靠后的，不动相关性高的。

        关键约束：腾槽位时绝不踢掉「本轮真的被提到」的相关事实
        （_rel>0）。没有这条规定时，一个 query 命中的高相关非永久事实
        （比如爱好=做音乐）会被硬生生挤掉去给一个零相关的永久事实（其他/
        琐事）让位 —— 直接导致统计轨的 l2_rows 丢了它、topic_ids 变空、
        _frozen_block 的话题刷新门因此永远不开，出现「输入做音乐却一点不更新」
        的假象。query="" 的常驻包里所有事实 _rel 都是 0，这条约束自动失效，
        永久槽位照常生效，行为不受影响。
        """
        if limit <= 0:
            return []
        head = list(rows[:limit])

        def _perm(r: dict) -> bool:
            return str(r.get("predicate") or "").strip() in PERMANENT_PREDICATES

        def _relevant(r: dict) -> bool:
            """这条算不算「本轮真的在问的东西」。

            必须把别名 / 上下文 / 语义三条通道一起算进来。只看 _rel（本体
            字面命中）时，靠别名召回的答案会被判成「零相关」，然后被一个
            零相关的永久事实（本名/年龄）顶掉 —— 实测「我会什么乐器」把
            「会弹木吉他」（alias 命中 4 个 bigram）换成了「本名陈岚」，
            「那它呢」把「养了一只橘猫」（ctx 命中）换成了「今年23岁」。
            别名兜底刚刚捞回来的东西，转头就被这里扔掉，等于没做。
            """
            return (int(r.get("_rel") or 0) > 0
                    or int(r.get("_rel_ctx") or 0) > 0
                    or int(r.get("_rel_alias") or 0) > 0
                    or bool(r.get("_sem")))

        quota = min(4, max(1, limit // 3))
        n_perm = sum(1 for r in head if _perm(r))
        if n_perm >= quota:
            return head
        have = {id(r) for r in head}
        extra = [r for r in pool if _perm(r) and id(r) not in have]
        if not extra:
            return head
        extra = extra[: quota - n_perm]
        keep = list(head)
        for r in extra:
            # 只从「既非永久、又和本轮无关」的条目里挑受害者。若头部里
            # 仅剩的相关事实都是非永久的，宁可永久槽位没填满，也绝不拿
            # 它们去换一个零相关的永久事实。
            victims = [i for i, x in enumerate(keep)
                       if not _perm(x) and not _relevant(x)]
            if not victims:
                break
            keep.pop(victims[-1])
            keep.append(r)
        keep.sort(key=lambda r: -float(r.get("_score") or 0))
        return keep

    def _filter_l3(self, rows: list[dict], query: str) -> list[dict]:
        if not query:
            return rows[: self.l3_snippets]
        qg = _bigrams(query)
        scored = []
        for r in rows:
            overlap = len(qg & _bigrams(r.get("content", "")))
            scored.append((overlap, r))
        scored.sort(key=lambda x: (x[0], x[1].get("id", 0)), reverse=True)
        hits = [r for o, r in scored if o > 0][: self.l3_snippets]
        if not hits:
            hits = [r for _, r in scored[: self.l3_snippets]]
        return hits

    def build(self, query: str = "", budget_tokens: int = 600,
              recent_queries: list[str] | None = None) -> dict:
        now = time.time()
        sub = self.subject or None
        sid = self.session_id

        # 三条 query 各管各的，绝不混用：
        #   q_raw   —— 本轮原话。问法判断（_asks_* / probe）只认它，历史消息和
        #              概念词混进去会把「这句是不是在指代一条消息」判偏。
        #   q       —— 本轮原话 + 概念归一词，主检索文本，也是渲染子项折叠的依据。
        #   ctx     —— 最近几轮用户原话，只按 0.4 权重补召回（「那它呢」「刚才
        #              那个」）。不进问法判断，也不进 pack["query"] 影响渲染。
        q_raw = str(query or "")
        q = expand_query(q_raw) if self.query_expand else q_raw
        hist = [str(x).strip() for x in (recent_queries or []) if str(x).strip()]
        if self.context_rounds > 0 and hist:
            hist = hist[-self.context_rounds:]
        else:
            hist = []
        ctx = " ".join(hist)
        q_search = f"{ctx} {q}".strip() if ctx else q

        # 摘要优先取「本会话」那条（同一人可能有多个会话，别串）
        summary = ""
        if sid:
            srows = self.store.query_recent_l3(
                limit=1, event_type="session_summary", session_id=sid,
                bot_id=self.bot_id)
            if srows:
                summary = srows[0]["content"] or ""
        if not summary and sub:
            # 本会话还没总结过：退一步用这个人最近一次总结。必须直接按 subject 查，
            # 不能指望它出现在「最近 60 条 L3」里 —— 消息一多，摘要当天就被挤出
            # 窗口，注入包里一个字都看不到（实测就丢过整段摘要）。
            try:
                srows = self.store.query_recent_l3(limit=1, event_type="session_summary",
                                                   subject=sub, bot_id=self.bot_id)
            except TypeError:
                srows = []
            if srows:
                summary = srows[0]["content"] or ""

        # 候选池：SQL 先按作用域 / 状态 / 有效期过滤，再按本轮相关度取，
        # 不再先按置信度 LIMIT。老实现「confidence DESC ... LIMIT 80」会让
        # 攒到 80 条以上的用户，低置信但本轮高度相关的事实永远进不了池。
        # 问法为空（常驻包 / 面板预览）时 store 自动退回旧的置信度排序。
        cand_lim = self.candidate_limit if self.candidate_limit > 0 else 100000

        def _fetch(name):
            try:
                raw = self.store.query_active_l2(
                    subject=name, limit=cand_lim, query=q_search,
                    candidate_limit=cand_lim)
            except TypeError:
                try:
                    raw = self.store.query_active_l2(subject=name, limit=cand_lim)
                except TypeError:
                    raw = self.store.query_active_l2(cand_lim)
            return [r for r in raw
                    if not r.get("expires_at") or r["expires_at"] > now]

        l2_rows = _fetch(sub)
        for name in self.extra_subjects:
            l2_rows.extend(_fetch(name))
        l2_rows = self._score_l2(l2_rows, q, ctx)
        # 先按相关性砍掉压根没被提到的，再去近重复、再截条数上限。
        # 反过来做的话，12 条名额会被噪声占满，过滤等于白设。
        l2_pool = l2_rows
        l2_rows = self._filter_by_relevance(l2_pool, q_raw)
        l2_rows = self._apply_mmr(l2_rows)
        l2_rows = self._pick_l2(l2_rows, l2_pool, self.l2_limit)
        # 点名保底：被点名的人（extra_subjects）至少 1 条档案进包。
        # 12 个名额可能被主人的事实挤满，那样「问 B 的事」时 AI 手里
        # 一条 B 的记录都没有——点名却查无此人，比不点名还糟。
        # 超出 l2_limit 的 1~3 条由预算裁剪兜底，不在这里硬砍。
        _have_subj = {r.get("subject") for r in l2_rows}
        for _name in self.extra_subjects:
            if _name and _name not in _have_subj:
                _cand = sorted(
                    (r for r in l2_pool if r.get("subject") == _name),
                    key=lambda r: -float(r.get("_score") or 0))
                if _cand:
                    _cand[0]["_extra_pin"] = True
                    l2_rows.append(_cand[0])
        # 提起度标记：这条进包算不算「被想起」。只认两类——
        #   1) 事实内容（predicate/value/ask_alias）和 query 字面重叠
        #   2) 语义钩子命中
        # 只蹭到人名（query 提到「张三」→ 整份档案 _rel>0 但内容零命中）
        # 和预算兜底硬塞的（rel 全零取保底条数）都不算，防止被动注入把
        # 计数刷成「系统选中次数」。main.py 的 bump 只数 _count_hit 的行。
        for r in l2_rows:
            r["_count_hit"] = bool(
                r.get("_sem")
                or (int(r.get("_rel") or 0) > 0 and int(r.get("_rel_content") or 0) > 0)
            )
        # 给每条候选打「这一轮进没进包」的标记。/糖记忆 why 要能解释的不只是
        # 「排第几」，还有「为什么没进包」——落选行也必须留下排序明细。
        _in_pack = {id(r) for r in l2_rows}
        for _r in l2_pool:
            _r.setdefault("_why", {})["in_pack"] = id(_r) in _in_pack
        # 统计轨早退：_count_hit 已经打好了，后面全是给「真正要注入的块」
        # 准备的重查询（l3 60 条 / 骨架 2000 条 / 外发池 / 跨会话池）。
        # lite 包不进 prompt、不裁剪、不渲染，只用来数提及度。
        if self.lite:
            # 话题触发信号交给注入轨：本轮「真被想起」的事实 id 列表 +
            # 是否出现指代 / 追问 / 回忆类问法。B 方案据此决定要不要在这一轮
            # 刷新一次常驻包，没有信号时一个字节都不动。
            return {"l2": l2_rows, "l3": [], "outbound": [], "cross": [],
                    "skeleton": [], "summary": "", "excluded": [],
                    "token_estimate": 0, "trim_notes": [], "_lite": True,
                    "topic_ids": [r.get("id") for r in l2_rows
                                  if r.get("_count_hit") and r.get("id")][:5],
                    "topic_signal": bool(
                        _asks_about_message(q_raw) or _PROBE_RE.search(q_raw)
                        or _TIME_Q.search(q_raw)),
                    "query_context": ctx,
                    "why": {int(r["id"]): dict(r.get("_why") or {})
                            for r in l2_pool if r.get("id")},
                    "_l2_pool": l2_pool}

        # 保留 _score 给裁剪阶段用：预算不够时该按「和这轮话题有多相关」砍，
        # 而不是按置信度砍。否则「路人乙 是不是说过他叫滚木」这种问法，
        # 会把正好相关的、只是当初记得不太确定的那条第一个砍掉。

        l3_rows = self.store.query_recent_l3(limit=60, subject=sub,
                                             bot_id=self.bot_id)
        # L3 通道只收「当前用户本人说过的原话」(user_message)。bot 自己在当前会话的
        # 闲聊回复（bot_reply）就挂在对话上下文里，模型看得见，塞进记忆块毫无信息量，
        # 只会把真正有价值的用户原话挤出预算（实测 L3 五条里四条是「喵~这就去打招呼」
        # 这类 bot 客套话）。bot 的跨会话回复归 outbound 通道管，不进这里。
        l3_rows = [r for r in l3_rows
                   if r.get("event_type") == "user_message"]
        # 当前会话最近 15 分钟的行不进 L3 通道：这些话就挂在本次对话上下文里，
        # 模型看得见原话，记忆块里再带一遍是双花 token（实测一次测试里 L3 三条
        # 全是上下文已有内容，把跨会话的行活活挤出 600 预算）。15 分钟与外发回声
        # 的判定保持一致：超过它的才算「可能已掉出上下文、值得记忆补」。
        if sid:
            l3_rows = [
                r for r in l3_rows
                if str(r.get("session_id") or "") != sid
                or now - float(r.get("created_at") or 0) > 900
            ]
        l3_hits = self._filter_l3(l3_rows, q_search)
        # 旧事骨架：只取当前说话人名下的（骨架跟着人走，跨人的是 cross 通道的事）
        skeletons: list = []
        if self.skeleton_limit and sub:
            try:
                # 候选窗口给足：骨架按人保留 90 天 × 多会话，几百到两千行。
                # 只取最近 60 行时，问 60 天前的旧事（骨架窗口外）永远召不回。
                # 全量打分的开销是 2-gram 集合交集 × 行数，实测 <2ms，可忽略。
                srows = self.store.query_recent_l3(
                    limit=2000,
                    event_type="l3_skeleton", subject=sub, bot_id=self.bot_id)
            except Exception:
                srows = []
            qg_skel = _bigrams(q_search) if q_search else set()
            if qg_skel:
                for r in srows:
                    r["_skel_rel"] = len(qg_skel & _bigrams(str(r.get("content") or "")))
                srows.sort(key=lambda r: (-r["_skel_rel"], -float(r.get("created_at") or 0)))
            # 相关命中优先，不足用最近的补齐（零命中时退化为原「常驻最新」行为）
            cand_skel = [
                {"id": r["id"], "content": r["content"], "ts": r.get("created_at", 0)}
                for r in srows[: self.skeleton_limit]
            ]
            # 骨架让位去重（细节感知）：骨架的定位是「补 L2 的空」，只有某条
            # 骨架的内容已**几乎逐字**躺在某条已选 L2 里（2-gram 重叠 >0.7）
            # 且它没带任何 L2 没有的数字信息时，才是纯重复、让位。
            # 不能按「话题重叠」去重——L2 是压缩断言（「检查灯用了很久」），
            # 骨架是原话摘录（「整整用了九十分钟」），同话题不同细节；
            # 数字是细节的主要载体，骨架带独有数字时永远保留。
            _num_re = re.compile(r"[0-9零一二两三四五六七八九十百千万]+")
            l2_texts = [
                f"{r.get('value', '')} {r.get('ask_alias') or ''}" for r in l2_rows
            ]
            l2_grams = [_bigrams(t) for t in l2_texts]
            l2_units: set = set()
            for _t in l2_texts:
                l2_units |= set(_num_re.findall(_t))
            skeletons = []
            self.skel_yielded = 0
            for _s in cand_skel:
                _sg = _bigrams(_s["content"])
                _best = max(
                    (len(_sg & _g) / max(1, min(len(_sg), len(_g))) for _g in l2_grams),
                    default=0.0,
                )
                _units = set(_num_re.findall(_s["content"]))
                if _best > 0.7 and not (_units - l2_units):
                    self.skel_yielded += 1
                    continue  # 纯重复：L2 已逐字覆盖且无独有细节
                skeletons.append(_s)
        # 多通道共识：同一件事实在 L2、旧事骨架、L3 原话里都和本轮对上，
        # 比单路命中可信得多。只给已进包的行加分，不改条数、不额外查库。
        if l2_rows and (skeletons or l3_hits):
            cg = _bigrams(" ".join(
                [str(s.get("content") or "") for s in skeletons]
                + [str(r.get("content") or "") for r in l3_hits]))
            for r in l2_rows:
                vg = _bigrams(f"{r.get('value') or ''}{r.get('ask_alias') or ''}")
                if cg & vg and int(r.get("_rel") or 0) > 0:
                    r["_score"] = float(r.get("_score") or 0) + 0.08
                    r.setdefault("_why", {})["consensus"] = 0.08
            l2_rows.sort(key=lambda r: -float(r.get("_score") or 0))

        excluded = self.store.recent_audit(limit=self.excluded_limit)

        # 跨会话：你自己主动发出去的话。不按 subject 过滤 —— 转述的本质就是
        # 「这个会话的我，替那个会话的我说话」，两边都得认账。
        #
        # 为什么不能直接拿「最近 N 条」：同一个会话里 AI 自己的对话回复也会走
        # context.send_message 这条路（分段器、send 工具），被记成大量
        # 「你主动发给当前用户」的回声记录，真转述几秒钟就被挤出窗口。
        # 所以这里分三档挑选：
        #   1. 发到当前会话、且带标签（转述/定时任务）—— 直接回答「你刚才发了什么」
        #   2. 发到其他会话的 —— 跨会话「我干过什么」的凭据
        #   3. 发到当前会话但超过 15 分钟的普通外发 —— 可能已掉出对话上下文，兜底带上
        # 15 分钟内的普通外发不进包：用户眼前就挂着这段对话，AI 不需要记忆来回忆它。
        # 号码 → 昵称映射：渲染层要把「用户 100010003」变成「用户 100010003(乐乐)」，
        # 否则模型认不出号码背后是谁（实测：AI 收到完整转述记录仍不敢用，去调工具现查）。
        nick_map: dict = {}
        try:
            nick_map = self.store.recent_nick_by_session()
        except Exception:
            nick_map = {}

        outbound: list[dict] = []
        # 每一类「候选 / 入选」的条数，仅供面板展示。用户看预览时要能分清
        # 「这一类一条都没有」和「有一堆候选但额度只给了 2 条」——这两种情况
        # 在注入块里长得一模一样（都是没有或很少几行），不标出来会以为是漏了。
        outbound_cand = {"here_relay": 0, "here_cron": 0, "here_plain": 0,
                         "away_relay": 0, "away_cron": 0, "away_plain": 0}
        # 追问补带进来的跨会话记录（只给日志/面板看，不进注入块）。
        # 出问题时能一眼确认「这轮到底补没补、补的是哪条」，不用再靠猜。
        added_away: list[dict] = []
        total_cap = max(self.outbound_relay_limit, self.outbound_cron_limit,
                        self.outbound_limit)
        if total_cap:
            try:
                rows = self.store.query_recent_l3(
                    limit=max(40, total_cap * 10), event_type="bot_outbound",
                    bot_id=self.bot_id,
                )
            except Exception:
                rows = []
            # 按「转述 / 定时 / 普通外发」×「当前会话 / 别的会话」分成五堆。
            # 混在一起时模型分不清哪句是替人带的话、哪句是闹钟发的，
            # 被问「你给 B 带了什么」会把定时提醒当转述复述出来。
            here_relay: list[dict] = []
            here_cron: list[dict] = []
            away_relay: list[dict] = []
            away_cron: list[dict] = []
            away_plain: list[dict] = []
            here_old: list[dict] = []
            stale_ids: set = set()
            for r in rows:  # rows 按新→旧
                content = str(r.get("content") or "")
                kind = _outbound_kind(content)
                is_here = bool(sid) and str(r.get("session_id") or "") == sid
                # 新鲜度窗口：过窗的当前会话转述/定时降级为普通回声，掉进
                # here_old——labeled_only 下不再常驻进包，也不再 pinned。
                # 追问补位的 away_* 不在此列：只有问了才带，不过窗。
                stale = (
                    (kind == "relay" and self.outbound_relay_window
                     and now - float(r.get("created_at") or 0) > self.outbound_relay_window)
                    or (kind == "cron" and self.outbound_cron_window
                        and now - float(r.get("created_at") or 0) > self.outbound_cron_window)
                )
                if stale and is_here:
                    stale_ids.add(r.get("id"))
                if is_here and kind == "relay" and not stale:
                    here_relay.append(r)
                elif is_here and kind == "cron" and not stale:
                    here_cron.append(r)
                elif is_here:
                    if now - float(r.get("created_at") or 0) > 900 or stale:
                        here_old.append(r)
                elif kind == "relay":
                    away_relay.append(r)
                elif kind == "cron":
                    away_cron.append(r)
                else:
                    away_plain.append(r)
            outbound_cand = {
                "here_relay": len(here_relay), "here_cron": len(here_cron),
                "here_plain": len(here_old),
                "away_relay": len(away_relay), "away_cron": len(away_cron),
                "away_plain": len(away_plain),
            }
            if self.outbound_labeled_only:
                # 只保留「发到当前会话」的真定时任务/转述。这个区块的初衷就是
                # 「AI 不知道自己刚才在本会话转述了啥」，只需要当前会话相关的
                # 转述。发到其他会话的转述（给 wind/Tom带话这类）跟眼前这个
                # 人毫无关系，塞进来只会污染。
                selected = (here_relay[: max(0, self.outbound_relay_limit)]
                            + here_cron[: max(0, self.outbound_cron_limit)])
                # 例外一：本轮这句话在追问「你给某某发了什么」。这时那条记录
                # 就是答案本身，必须补进来。早先这里写的是「跨会话该靠
                # group_history_agent 的 read_friend_recent_messages 兜底」——
                # 实测兜不住：那个工具读的是协议端 get_friend_msg_history，
                # 跟这份 L3 库不是一个数据源，转述根本不在里面。
                # 所以改成按问句补位：不问 0 条，问了最多 N 条。
                if (
                    away_relay
                    and self.outbound_away_limit > 0
                    and _asks_outbound_away(query, away_relay, nick_map)
                ):
                    added_away = away_relay[: self.outbound_away_limit]
                    selected = selected + added_away
                # 例外二：问的是定时任务/提醒。判据和上面分开，免得「给 B 带了
                # 什么话」把一条闹钟补进来，模型张冠李戴。
                if (
                    away_cron
                    and self.outbound_away_cron_limit > 0
                    and _asks_outbound_away_cron(query, away_cron, nick_map)
                ):
                    selected = selected + away_cron[: self.outbound_away_cron_limit]
            else:
                # 关掉 labeled_only = 用户明确要看全部外发声。四类各给自己那份额度，
                # 不互相挤占（以前混在一个列表里取前 N 条，普通回声永远排不上）。
                cap_plain = max(0, self.outbound_limit)
                selected = (here_relay[: max(0, self.outbound_relay_limit)]
                            + here_cron[: max(0, self.outbound_cron_limit)]
                            + here_old[: cap_plain]
                            + away_relay[: cap_plain]
                            + away_cron[: cap_plain]
                            + away_plain[: cap_plain])
            for r in selected:
                content = str(r.get("content") or "")
                is_here = bool(sid) and str(r.get("session_id") or "") == sid
                kind = _outbound_kind(content)
                # 过窗降级条目（已掉进 here_old 的老转述/定时）：高级模式
                # （labeled_only=False）里仍可见，但不再算「带标签的铁证」，
                # 也不吃 pinned 保护——窗口的语义就是它不再是新鲜答案。
                still_fresh = r.get("id") not in stale_ids
                labeled = kind != "plain" and still_fresh
                outbound.append({
                    "content": content,
                    "ts": r.get("created_at", 0),
                    "session_id": r.get("session_id", ""),
                    "here": is_here,
                    # relay / cron / plain：渲染时按这个分段，模型才不会把
                    # 「替你带的话」和「闹钟发的话」混成一件。
                    "kind": kind,
                    # 带任务/转述标签 = 真定时任务/真转述，跟「你主动发给xxx」的
                    # 普通外发回声区分开。裁剪时带标签的优先保留（见 injector）。
                    "labeled": labeled,
                    # 发到当前会话、且带任务/转述标签的：这是「你刚才定时发了什么」的
                    # 直接答案，预算再紧也不许裁（见 injector 第 5 步）。
                    "pinned": self.outbound_pinned and is_here and labeled,
                    "peer_nick": nick_map.get(str(r.get("session_id") or ""), ""),
                })

        # 跨会话：别的会话里发生的事。
        # 默认只收 user_message（别人说的话）。cross_include_self 开了才额外收
        # bot_reply（AI 自己在别的会话里的回复）——AI 一句往往比人多、信息量少，
        # 默认不带，靠 user_message 撑起跨会话的信息量。bot_outbound（主动转述/
        # 定时）归 outbound 管道管，不进这里。
        cross: list[dict] = []
        if self.cross_limit:
            try:
                pool = self.store.query_recent_l3(
                    limit=max(40, self.cross_limit * 8), bot_id=self.bot_id)
            except Exception:
                pool = []
            allowed = ("user_message", "bot_reply") if self.cross_include_self else ("user_message",)
            pool = [
                r for r in pool
                if r.get("event_type") in allowed
                and str(r.get("session_id") or "") != sid
            ]
            # user_message 要排除「当前用户自己在别的会话说的话」（含同人换昵称），
            # 否则会把张三/路人甲 的旧消息当「别人」塞进来；bot_reply 的 subject 是
            # 「AI 回复的那个会话的人」而不是说话人，不需要这个排除。
            pool = [
                r for r in pool
                if r.get("event_type") != "user_message"
                or not self._is_self(str(r.get("subject") or ""))
            ]
            # 会话级归因：找出「历史消息里提到过当前用户」的会话。路人乙 私聊里
            # 用「她/被子」指代张三，字面没「张三」二字，AI 无法归因。只要这个会话
            # 的历史消息里出现过当前用户的名字，就判定整段会话在聊 ta。
            about_sessions: set = set()
            if self.subject:
                try:
                    about_sessions = self.store.sessions_mentioning(self.subject)
                except Exception:
                    about_sessions = set()

            def _about(r: dict) -> bool:
                return str(r.get("session_id") or "") in about_sessions

            fresh = sorted(
                (r for r in pool if now - float(r.get("created_at") or 0) <= 600),
                key=lambda r: r.get("id", 0), reverse=True)
            picked: list[dict] = []
            seen: set = set()
            # 挑选优先级：
            #   ① 最近 10 分钟、且来自「提到过当前用户」会话的 —— 正在聊 ta，最高优先
            #   ② 最近 10 分钟的其它动态 —— 正在发生的别处事情
            #   ③ 来自「提到过当前用户」会话的近期消息 —— 聊过 ta，比关键词噪声更该带
            #   ④ 按本题关键词补足
            for r in fresh[: max(1, self.cross_limit - 1)]:
                if _about(r):
                    picked.append(r)
                    seen.add(r.get("id"))
            if len(picked) < self.cross_limit:
                for r in fresh:
                    if r.get("id") in seen:
                        continue
                    picked.append(r)
                    seen.add(r.get("id"))
                    if len(picked) >= self.cross_limit:
                        break
            if len(picked) < self.cross_limit:
                about_rest = [r for r in pool
                              if r.get("id") not in seen and _about(r)]
                for r in about_rest[: self.cross_limit]:
                    picked.append(r)
                    seen.add(r.get("id"))
            if len(picked) < self.cross_limit:
                for r in self._filter_l3(pool, q):
                    if r.get("id") in seen:
                        continue
                    picked.append(r)
                    if len(picked) >= self.cross_limit:
                        break
            qg = _bigrams(q) if q else set()
            for r in picked:
                ts = float(r.get("created_at") or 0)
                is_about = _about(r)
                # 区分「别人说的话」和「AI 自己在别的会话说的话」，渲染时措辞不同
                is_self_talk = r.get("event_type") == "bot_reply"
                cross.append({
                    "content": r.get("content", ""),
                    "subject": r.get("subject", ""),
                    "ts": r.get("created_at", 0),
                    # 带上和本轮话题的命中数，裁剪时要区分「对上题的答案」和「兜底杂闻」
                    "_rel": len(qg & _bigrams(r.get("content", ""))),
                    # 10 分钟内的其他会话动态，裁剪时不当杂闻砍
                    "_fresh": now - ts <= 600,
                    # 归因：这条来自「提到过当前用户」的会话，很可能就是在说 ta
                    "_about_subject": is_about,
                    # 这是 AI 自己在别的会话里说过的话（bot_reply），渲染时标注「你」开头
                    "_self_talk": is_self_talk,
                })
                # 旧数据 subject 可能为空，渲染会显示「[别人]」；用昵称映射兜底
                if not cross[-1]["subject"]:
                    cross[-1]["subject"] = nick_map.get(
                        str(r.get("session_id") or ""), "")

        # 身份对照：主人 + 本轮被点名的人。系统侧 subjects_mentioned 靠别名
        # 认得出「滚木=用户乙」并拉了档案，但模型看不到这层映射——对话里
        # 喊「阿沃」、记忆块写「用户甲的xxx」，字面不像时模型认不出是同一人。
        # 这里把 显示名/QQ/别名清单一并带出，渲染层拼成「谁是谁」段。
        identities: list = []
        _id_srcs = list(dict.fromkeys(
            ([self.subject] if self.subject else []) + list(self.extra_subjects)))
        if _id_srcs:
            try:
                _idmap = self.store.person_identity(_id_srcs)
                for _n in _id_srcs:
                    if _idmap.get(_n):
                        identities.append({"subject": _n, **_idmap[_n]})
            except Exception:
                identities = []
        # 别名按对话过滤（2026-11-06 用户定稿）：一个人名下所有称呼全列进
        # 注入是烧 token，而模型真正需要的只有一件事——「这轮对话里出现的
        # 这个写法，是我平时叫他的那个谁」。所以只保留用户本轮原话里字面
        # 出现过的别名，全量清单留给 alias_lookup 工具按需查。
        # 注意只能对 query 匹配，不能把记忆包原文也算进「对话」：转述行、
        # cross 行的渲染自带号码和 subject（[你转述给当前用户（用户甲…）]、
        # [私] 用户乙: …），拿它们当语料的话号码恒命中，过滤形同虚设。
        # 单字别名不做匹配（中文单字撞车率太高，「鸭」会命中一堆无关句子）。
        _conv_text = f"{q_raw or ''}\n{q or ''}"
        for _it in identities:
            _it["aliases"] = [
                (a, t) for a, t in (_it.get("aliases") or [])
                if len(a) >= 2 and a in _conv_text]

        pack = {
            "summary": summary,
            "scope": {"subject": self.subject, "session_id": sid},
            "query": q,
            "query_raw": q_raw,
            "query_context": ctx,
            # 每条 L2 的排序明细：/糖记忆 why <id> 和面板据此解释
            # 「为什么排第 3 / 为什么没进包」，不用再靠猜。
            "why": {int(r["id"]): dict(r.get("_why") or {})
                    for r in l2_pool if r.get("id")},
            "_l2_pool": l2_pool,
            "l2": l2_rows,
            "identities": identities,
            "l3": [{"id": r["id"], "content": r["content"], "event_type": r["event_type"],
                    "ts": r.get("created_at", 0)} for r in l3_hits],
            "skeleton": skeletons,
            # 本轮因「L2 逐字覆盖且无独有细节」而让位的骨架条数（面板可见）
            "skeleton_yielded": getattr(self, "skel_yielded", 0),
            "outbound": outbound,
            # 本轮「追问命中」补带进来的跨会话记录：给日志和面板留个凭据。
            # 用户在问「你给 B 发了什么」时，这一行就是他能不能看到答案的分水岭。
            "outbound_away_added": [
                {"id": r.get("id"), "session_id": r.get("session_id")}
                for r in added_away
            ],
            # 各类外发的候选条数（未按额度截断前），面板据此显示「有 5 条候选、
            # 只给了 2 条」，把「额度卡住」和「确实没有」区分开。
            "cand": {"outbound": outbound_cand},
            # 每类的条数上限。光有候选数还说不清 0 条的原因——可能是没触发补位，
            # 也可能是额度就是 0，两个数字摆一起用户自己就能判断。
            "limits": {"here_relay": self.outbound_relay_limit,
                       "here_cron": self.outbound_cron_limit,
                       "here_plain": self.outbound_limit,
                       "away_relay": self.outbound_away_limit,
                       "away_cron": self.outbound_away_cron_limit,
                       "away_plain": self.outbound_limit},
            # 这次用的两道闸，面板要据此解释「普通外发为什么一条都没进」。
            "gates": {"labeled_only": self.outbound_labeled_only,
                      "pinned": self.outbound_pinned},
            "cross": cross,
            "excluded": [{"decision": r["decision"], "reason": r["reason"]} for r in excluded],
            "trim_notes": [],
            # 这一句是不是在问「你刚才发的那个是什么」。判断依据是本轮用户的话，
            # 跟记录距今多久无关——隔三天问同一句话，结果一样。
            # injector 据此决定要不要把话挑明（先承认发过），见 _asks_about_message。
            "probe_outbound": bool(outbound) and _asks_about_message(query),
        }
        # 展示名：subject 是内部身份键（常常是这个人最早的那个昵称），换了昵称
        # 之后直接注入就是旧名字，AI 会拿一个他自己都改掉的称呼叫人。这里只
        # 加一个渲染用的 _shown，subject 本身不动——检索/归并/关系边全靠它稳定。
        try:
            persons = [r.get("subject") for r in l2_rows]
            persons += [e.get("subject") for e in cross]
            nm = self.store.person_names(persons)
            nick = self.store.person_nicknames(persons)
            for r in l2_rows:
                shown = nm.get(str(r.get("subject") or ""))
                if shown:
                    r["_shown"] = shown
                    qq_nick = nick.get(str(r.get("subject") or ""))
                    if qq_nick and alias_key(qq_nick) != alias_key(shown):
                        r["_nick"] = qq_nick
            for e in cross:
                shown = nm.get(str(e.get("subject") or ""))
                if shown:
                    e["_shown"] = shown
                    qq_nick = nick.get(str(e.get("subject") or ""))
                    if qq_nick and alias_key(qq_nick) != alias_key(shown):
                        e["_nick"] = qq_nick
        except Exception:
            pass
        result = trim_pack(pack, budget_tokens,
                           l3_keep_min=self.l3_keep_min,
                           l2_keep_min=self.l2_keep_min)
        # 以前裁剪是完全静默的：预算调到低于自然体积会把 L2 从 12 条砍到 1 条，
        # 用户只会觉得「AI 突然变笨」，面板预览之外没有任何线索。
        notes = [str(x) for x in (result.get("trim_notes") or []) if x]
        if notes:
            sig = " | ".join(notes)
            now_t = time.time()
            if sig != _TRIM_WARN["sig"] or now_t - _TRIM_WARN["at"] >= 60.0:
                if self.logger is not None:
                    self.logger.warning(
                        "[SugarMemory] 记忆包触发预算裁剪（budget=%s，上限不等于目标）：%s",
                        budget_tokens, sig[:400])
                _TRIM_WARN["sig"] = sig
                _TRIM_WARN["at"] = now_t
        return result
