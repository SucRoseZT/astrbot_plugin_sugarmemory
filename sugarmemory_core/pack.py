"""记忆包组装：检索、排序、裁剪、可解释输出。零 LLM 调用。
关键词 2-gram 检索独立工作；向量检索为可选增强（预留接口，不默认依赖）。
"""
import re
import time

from .injector import estimate_tokens, trim_pack
from .store import PERMANENT_PREDICATES, alias_key, retention_score

# 裁剪告警节流：常驻包每轮都要 build 一次，预算调低时会每轮都触发同样的
# 裁剪。相同原因 60 秒只报一次，别把日志刷爆。
_TRIM_WARN: dict = {"sig": "", "at": 0.0}


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", str(text or ""))
    if len(text) < 2:
        return {text} if text else set()
    return {text[i: i + 2] for i in range(len(text) - 1)}


# ---- query 概念归一（零成本补召回的第一层）----
# 中文口语的改写问法（「多大了」vs「16岁」）2-gram 字面零重叠，闸门必砍。
# 这里把常见问法映射到「概念词」，检索前拼接到 query 尾部：原文保留（字面
# 命中不受影响），概念词让记忆条里的对应说法有机会撞上 bigram。
# 规则要泛化、不许为个案硬编码：每条都是通用中文口语知识，宁可漏配
# （还有 alias 列和语义兜底接着），不要配错（概念词乱激活会拉低精度）。
_QUERY_CONCEPTS: list[tuple[re.Pattern, str]] = [
    (re.compile(p), concepts) for p, concepts in (
        (r"多大|几岁|年龄|年纪|岁数", "岁 年龄"),
        (r"吃|喝|宵夜|夜宵|零食|做饭|下饭|点东西", "吃 食物 饮食"),
        (r"睡|熬夜|失眠|作息|通宵|起床|才醒", "睡 作息 熬夜"),
        (r"心理医生|看病|去医院|抑郁|焦虑|失眠|崩溃|状态不好|休息", "心理 病 状态"),
        (r"人设|形象|设定|马甲|网名", "形象 人设"),
        (r"创作|作品|搞什么|玩什么|接稿|约稿|画稿", "创作 作品 稿"),
        (r"礼物|送过|送了|收过|收到", "礼物 送"),
        (r"去得成|来吗|过来|见面|去找|来访|打算.{0,8}来|暂缓", "来 见面 计划"),
        (r"挣钱|赚钱|收入|工资|生活费|零花", "钱 挣钱"),
        (r"对象|在一起|男女朋友|恋人|男朋友|女朋友|老婆|老公|前任|分手", "对象 恋爱 关系"),
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
                 logger=None):
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
        # 线上由 main 注入 embedding 实现（provider 不支持时传 None）。
        self.semantic_fn = semantic_fn
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

    def _score_l2(self, rows: list[dict], query: str) -> list[dict]:
        if not query:
            for r in rows:
                r["_rel"] = 0
                r["_rel_content"] = 0
            return rows
        qg = _bigrams(query)
        for r in rows:
            # 比对文本 = subject + predicate + value + ask_alias。
            # ask_alias 是总结阶段生成的「用户可能怎么问」的说法列——
            # 改写问法跟 value 零重叠，但跟问法别名是字面重叠，靠它进闸门。
            text = (f"{r.get('subject','')}{r.get('predicate','')}"
                    f"{r.get('value','')}{r.get('ask_alias','')}")
            # 内容匹配单独算一遍（不含 subject）：query 里只出现了人名时，
            # 整份档案都会蹭到 _rel>0——那是「点名带出档案」，不是「想起这条
            # 事实」。提起度只认内容匹配，防止随口叫人名把计数刷爆。
            content = (f"{r.get('predicate','')}{r.get('value','')}"
                       f"{r.get('ask_alias','')}")
            r["_rel"] = len(qg & _bigrams(text))
            r["_rel_content"] = len(qg & _bigrams(content))
            # 保留度只做同等相关时的次级权重（0~0.3），刻意压得很低：
            # 一个 2-gram 命中值 2.0 分，相关性永远优先。它的作用是
            # 「两条一样相关的，先给还记得牢的那条」。
            r["_score"] = (r["_rel"] * 2.0 + float(r.get("confidence", 0))
                           + min(int(r.get("stability", 0)), 5) * 0.1
                           + retention_score(r) * 0.3)
        return sorted(rows, key=lambda r: r["_score"], reverse=True)

    def _filter_by_relevance(self, rows: list[dict], query: str = "") -> list[dict]:
        """按本轮话题命中数砍掉「压根没被提到」的条目。

        只看 2-gram 命中数（_rel），不看 _score：_score 里混了置信度和稳定性，
        即使一句话都没命中也会有 0.9 分，用它做阈值等于没设。
        本轮一个都没命中（闲聊、"在吗"）时不过滤，交给预算按保底条数裁。
        """
        if self.rel_ratio <= 0 or not rows:
            return rows
        top = max(int(r.get("_rel", 0)) for r in rows)
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
                            # 抬 _score，避免这些行在 trim 阶段被当成最弱先裁掉；
                            # 但压在字面命中行之下（_rel>=1 意味着 _score>=2）
                            r["_score"] = max(float(r.get("_score", 0) or 0),
                                              0.4 + 0.6 * s)
                        return picked
            # 整句话一个 2-gram 都没命中：判断不了相关性，只给保底条数（按置信度
            # 挑最可信的几条）。别整池子扔进去 —— 一句「在吗，今天干嘛呢」塞
            # 8 条无关事实，AI 会在一句打招呼里强行背资料卡。
            return rows[:keep]
        th = max(1.0, top * self.rel_ratio)
        kept = [r for r in rows if int(r.get("_rel", 0)) >= th]
        if len(kept) >= min(self.l2_keep_min, len(rows)):
            return kept
        # 过滤太狠（只剩一两条）时补足到保底条数，取分最高的那几条。
        # 注意不能像早期版本那样直接退回整个 rows —— 那等于把 40 条噪声
        # 全放回池子，过滤白设，最后还是靠预算随机砍。
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
        """
        if limit <= 0:
            return []
        head = list(rows[:limit])

        def _perm(r: dict) -> bool:
            return str(r.get("predicate") or "").strip() in PERMANENT_PREDICATES

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
            victims = [i for i, x in enumerate(keep) if not _perm(x)]
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

    def build(self, query: str = "", budget_tokens: int = 600) -> dict:
        now = time.time()
        sub = self.subject or None
        sid = self.session_id

        # 两条 query 并行：
        #   q_raw —— 原话。问法判断（_asks_outbound_away / probe）必须用它，
        #            概念词混进去会把「这不是在问转述吗」的判断带偏。
        #   q     —— 检索文本。原话 + 概念归一词（expand_query），只给
        #            _score_l2 / _filter_l3 / 渲染子项筛选用。
        q_raw = str(query or "")
        q = expand_query(q_raw) if self.query_expand else q_raw

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

        l2_rows = [r for r in self.store.query_active_l2(subject=sub, limit=80)
                   if not r.get("expires_at") or r["expires_at"] > now]
        for name in self.extra_subjects:
            # 被点名的人：档案按「和这轮话的相关性」统一排序选出，所以候选
            # 必须给够。此前 limit=8 且按置信度预排序——点名问「王五在不在
            # 找工作」，跳槽条（conf 0.7）被 8 个高置信琐碎条挤出候选，
            # 相关性排序根本没机会发力（压测抓到）。80 = 与主 subject 一致。
            l2_rows.extend(
                r for r in self.store.query_active_l2(subject=name, limit=80)
                if not r.get("expires_at") or r["expires_at"] > now
            )
        l2_rows = self._score_l2(l2_rows, q)
        # 先按相关性砍掉压根没被提到的，再截条数上限——反过来做的话，
        # 12 条名额会被噪声占满，过滤等于白设
        l2_pool = l2_rows
        l2_rows = self._filter_by_relevance(l2_pool, q_raw)
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
        # 统计轨早退：_count_hit 已经打好了，后面全是给「真正要注入的块」
        # 准备的重查询（l3 60 条 / 骨架 2000 条 / 外发池 / 跨会话池）。
        # lite 包不进 prompt、不裁剪、不渲染，只用来数提及度。
        if self.lite:
            return {"l2": l2_rows, "l3": [], "outbound": [], "cross": [],
                    "skeleton": [], "summary": "", "excluded": [],
                    "token_estimate": 0, "trim_notes": [], "_lite": True}

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
        l3_hits = self._filter_l3(l3_rows, q)
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
            qg_skel = _bigrams(q) if q else set()
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
