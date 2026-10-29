"""记忆包组装：检索、排序、裁剪、可解释输出。零 LLM 调用。
关键词 2-gram 检索独立工作；向量检索为可选增强（预留接口，不默认依赖）。
"""
import re
import time

from .injector import estimate_tokens, trim_pack


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", str(text or ""))
    if len(text) < 2:
        return {text} if text else set()
    return {text[i: i + 2] for i in range(len(text) - 1)}


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


class PackBuilder:
    def __init__(self, store, l3_snippets: int = 0, excluded_limit: int = 5,
                 l2_limit: int = 12, subject: str = "", session_id: str = "",
                 extra_subjects: list[str] | None = None,
                 rel_ratio: float = 0.3, l2_keep_min: int = 3, l3_keep_min: int = 0,
                 outbound_limit: int = 4, cross_limit: int = 5,
                 outbound_labeled_only: bool = True, outbound_pinned: bool = True,
                 cross_include_self: bool = False):
        self.store = store
        self.l3_snippets = int(l3_snippets)
        self.excluded_limit = int(excluded_limit)
        self.l2_limit = int(l2_limit)
        # 跨会话两条通道，都是「不管当前是谁」都能看到的：
        #   outbound = 你自己主动发到别的会话的话（转述、定时通知）。
        #     不注入这段，AI 换个会话被问「你刚才跟 X 说了啥」就查不到任何记录，
        #     会当成自己编的，然后否认。这是「转述被当成假的」的根因。
        #   cross = 别的会话里别人说的话，按本轮话题筛过再给，免得串台。
        self.outbound_limit = max(0, int(outbound_limit))
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
        # 相关性门槛：低于「本轮最高命中数 × rel_ratio」的 L2 直接不进池。
        # 0 或负数 = 关闭过滤（回到「有钱就全塞」的旧行为）。
        # 为什么要这道闸：预算是有限资源，一句「在吗，今天干嘛呢」不该把
        # 十几条跟本轮无关的事实全拽进来，把原话和摘要挤死。
        self.rel_ratio = float(rel_ratio or 0)
        self.l2_keep_min = max(1, int(l2_keep_min))
        self.l3_keep_min = max(0, int(l3_keep_min))
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
        # 人物归一缓存：self.subject 可能是某个昵称（小刚），库里同一个人可能
        # 记在别的昵称（蔗糖）名下。缓一份「归一后的 person + 全部别名」，cross 通道
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
            return rows
        qg = _bigrams(query)
        for r in rows:
            text = f"{r.get('subject','')}{r.get('predicate','')}{r.get('value','')}"
            rg = _bigrams(text)
            hit = len(qg & rg)
            r["_rel"] = hit
            r["_score"] = hit * 2.0 + float(r.get("confidence", 0)) + min(int(r.get("stability", 0)), 5) * 0.1
        return sorted(rows, key=lambda r: r["_score"], reverse=True)

    def _filter_by_relevance(self, rows: list[dict]) -> list[dict]:
        """按本轮话题命中数砍掉「压根没被提到」的条目。

        只看 2-gram 命中数（_rel），不看 _score：_score 里混了置信度和稳定性，
        即使一句话都没命中也会有 0.9 分，用它做阈值等于没设。
        本轮一个都没命中（闲聊、"在吗"）时不过滤，交给预算按保底条数裁。
        """
        if self.rel_ratio <= 0 or not rows:
            return rows
        top = max(int(r.get("_rel", 0)) for r in rows)
        if top <= 0:
            # 整句话一个 2-gram 都没命中：判断不了相关性，只给保底条数（按置信度
            # 挑最可信的几条）。别整池子扔进去 —— 一句「在吗，今天干嘛呢」塞
            # 8 条无关事实，AI 会在一句打招呼里强行背资料卡。
            return rows[: min(self.l2_keep_min, len(rows))]
        th = max(1.0, top * self.rel_ratio)
        kept = [r for r in rows if int(r.get("_rel", 0)) >= th]
        if len(kept) >= min(self.l2_keep_min, len(rows)):
            return kept
        # 过滤太狠（只剩一两条）时补足到保底条数，取分最高的那几条。
        # 注意不能像早期版本那样直接退回整个 rows —— 那等于把 40 条噪声
        # 全放回池子，过滤白设，最后还是靠预算随机砍。
        return rows[: min(self.l2_keep_min, len(rows))]


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

        # 摘要优先取「本会话」那条（同一人可能有多个会话，别串）
        summary = ""
        if sid:
            srows = self.store.query_recent_l3(limit=1, event_type="session_summary", session_id=sid)
            if srows:
                summary = srows[0]["content"] or ""
        if not summary and sub:
            # 本会话还没总结过：退一步用这个人最近一次总结。必须直接按 subject 查，
            # 不能指望它出现在「最近 60 条 L3」里 —— 消息一多，摘要当天就被挤出
            # 窗口，注入包里一个字都看不到（实测就丢过整段摘要）。
            try:
                srows = self.store.query_recent_l3(limit=1, event_type="session_summary", subject=sub)
            except TypeError:
                srows = []
            if srows:
                summary = srows[0]["content"] or ""

        l2_rows = [r for r in self.store.query_active_l2(subject=sub, limit=80)
                   if not r.get("expires_at") or r["expires_at"] > now]
        for name in self.extra_subjects:
            l2_rows.extend(
                r for r in self.store.query_active_l2(subject=name, limit=8)
                if not r.get("expires_at") or r["expires_at"] > now
            )
        l2_rows = self._score_l2(l2_rows, query)
        # 先按相关性砍掉压根没被提到的，再截条数上限——反过来做的话，
        # 12 条名额会被噪声占满，过滤等于白设
        l2_rows = self._filter_by_relevance(l2_rows)[: self.l2_limit]
        # 保留 _score 给裁剪阶段用：预算不够时该按「和这轮话题有多相关」砍，
        # 而不是按置信度砍。否则「小红 是不是说过他叫滚木」这种问法，
        # 会把正好相关的、只是当初记得不太确定的那条第一个砍掉。

        l3_rows = self.store.query_recent_l3(limit=60, subject=sub)
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
        l3_hits = self._filter_l3(l3_rows, query)
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
        # 号码 → 昵称映射：渲染层要把「用户 10001」变成「用户 10001(小明)」，
        # 否则模型认不出号码背后是谁（实测：AI 收到完整转述记录仍不敢用，去调工具现查）。
        nick_map: dict = {}
        try:
            nick_map = self.store.recent_nick_by_session()
        except Exception:
            nick_map = {}

        outbound: list[dict] = []
        if self.outbound_limit:
            try:
                rows = self.store.query_recent_l3(
                    limit=max(40, self.outbound_limit * 8), event_type="bot_outbound"
                )
            except Exception:
                rows = []
            here_top: list[dict] = []
            away_labeled: list[dict] = []
            away_plain: list[dict] = []
            here_old: list[dict] = []
            for r in rows:  # rows 按新→旧
                content = str(r.get("content") or "")
                is_here = bool(sid) and str(r.get("session_id") or "") == sid
                labeled = content.startswith("[你转述给") or content.startswith("[你按定时任务")
                if is_here and labeled:
                    here_top.append(r)
                elif is_here:
                    if now - float(r.get("created_at") or 0) > 900:
                        here_old.append(r)
                else:
                    # 发到其他会话的，按带不带标签再分：带标签（定时任务/转述）
                    # 是「我在别的会话干过什么」的铁证，永远该进包；
                    # 不带标签的普通外发才是可被 labeled_only 过滤的回声。
                    (away_labeled if labeled else away_plain).append(r)
            if self.outbound_labeled_only:
                # 只保留「发到当前会话」的真定时任务/转述。这个区块的初衷就是
                # 「AI 不知道自己刚才在本会话转述了啥」，只需要当前会话相关的
                # 转述。发到其他会话的转述（给 wind/披萨带话这类）跟眼前这个
                # 人毫无关系，塞进来只会污染——跨会话「我给 X 说过什么」该靠
                # group_history_agent 的工具（read_friend_recent_messages）兜底，
                # 不靠记忆包堆一堆不相关的转述。
                selected = here_top[: max(0, self.outbound_limit)]
            else:
                selected = (here_top + away_labeled + away_plain + here_old)[: max(0, self.outbound_limit)]
            for r in selected:
                content = str(r.get("content") or "")
                is_here = bool(sid) and str(r.get("session_id") or "") == sid
                labeled = content.startswith("[你转述给") or content.startswith("[你按定时任务")
                outbound.append({
                    "content": content,
                    "ts": r.get("created_at", 0),
                    "session_id": r.get("session_id", ""),
                    "here": is_here,
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
                pool = self.store.query_recent_l3(limit=max(40, self.cross_limit * 8))
            except Exception:
                pool = []
            allowed = ("user_message", "bot_reply") if self.cross_include_self else ("user_message",)
            pool = [
                r for r in pool
                if r.get("event_type") in allowed
                and str(r.get("session_id") or "") != sid
            ]
            # user_message 要排除「当前用户自己在别的会话说的话」（含同人换昵称），
            # 否则会把蔗糖/小刚 的旧消息当「别人」塞进来；bot_reply 的 subject 是
            # 「AI 回复的那个会话的人」而不是说话人，不需要这个排除。
            pool = [
                r for r in pool
                if r.get("event_type") != "user_message"
                or not self._is_self(str(r.get("subject") or ""))
            ]
            # 会话级归因：找出「历史消息里提到过当前用户」的会话。小红 私聊里
            # 用「她/被子」指代蔗糖，字面没「蔗糖」二字，AI 无法归因。只要这个会话
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
                for r in self._filter_l3(pool, query):
                    if r.get("id") in seen:
                        continue
                    picked.append(r)
                    if len(picked) >= self.cross_limit:
                        break
            qg = _bigrams(query) if query else set()
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

        pack = {
            "summary": summary,
            "scope": {"subject": self.subject, "session_id": sid},
            "query": query,
            "l2": l2_rows,
            "l3": [{"id": r["id"], "content": r["content"], "event_type": r["event_type"],
                    "ts": r.get("created_at", 0)} for r in l3_hits],
            "outbound": outbound,
            "cross": cross,
            "excluded": [{"decision": r["decision"], "reason": r["reason"]} for r in excluded],
            "trim_notes": [],
            # 这一句是不是在问「你刚才发的那个是什么」。判断依据是本轮用户的话，
            # 跟记录距今多久无关——隔三天问同一句话，结果一样。
            # injector 据此决定要不要把话挑明（先承认发过），见 _asks_about_message。
            "probe_outbound": bool(outbound) and _asks_about_message(query),
        }
        return trim_pack(pack, budget_tokens,
                         l3_keep_min=self.l3_keep_min,
                         l2_keep_min=self.l2_keep_min)
