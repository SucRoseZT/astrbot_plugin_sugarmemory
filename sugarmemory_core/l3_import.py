# -*- coding: utf-8 -*-
"""补录历史的解析与体检：时间戳解析、行式文本解析、导入前分布检查。

纯标准库，不碰 astrbot，方便离线跑测试。
"""
import re
import time

# 补录历史记忆时认的时间格式。datetime-local 给的是 2026-08-01T21:30，
# 从别处导出来的多半是 2026-08-01 21:30:00 或纯时间戳，三种都得吃得下。
TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
)

# 聊天记录里「谁在什么时候说的」那一行。QQ/微信复制出来的形态五花八门，
# 认这四种就够了（按顺序试，先具体后宽松）：
#   [2026-11-04 09:59:33] 昵称: 内容
#   2026-11-04 09:59:33 昵称: 内容
#   昵称: 11-04 09:59:33      （内容在下一行）
#   昵称 11-04 09:59:33        （没冒号）
_RE_HEAD_BRACKET = re.compile(
    r"^[\[\(（]\s*(?P<ts>[^\]\)）]{6,32})\s*[\]\)）]\s*(?P<name>[^:：\n|]{1,40}?)\s*[:：]\s*(?P<text>.*)$"
)
_RE_TIME_FIRST = re.compile(
    r"^(?P<ts>\d{2,4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s+"
    # 后面的名字不能是 `09:59` 里被回溯切出来的「09」，否则纯内容行会被拆成
    # spk=09 / content=59:33 随便说一句
    r"(?!(?:\d{1,2}:\d{2}))(?P<name>[^:：\n|]{1,40}?)\s*[:：]\s*(?P<text>.+)$"
)
_RE_HEAD_TIME = re.compile(
    r"^(?P<name>[^:：\n|]{1,40}?)\s*[:：]\s*"
    r"(?P<ts>\d{2,4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)
_RE_HEAD_TIME_NC = re.compile(
    r"^(?P<name>[^:：\n\d|][^:：\n|]{0,39}?)\s+(?P<ts>\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)
_RE_TIME_ONLY = re.compile(
    r"^(?P<ts>\d{2,4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)[ ]{1,}"
    # 只认「时间 空格 内容」：用 tab 或竖线分隔的交给行式解析，别在这里抢走
    r"(?P<text>(?![|\t]).+)$"
)
# 已经带信封的行（插件自己录出来的格式），不重复套一层
_RE_ENVELOPE = re.compile(r"^\[(?:你|私|群)\]\s*")
# 名字里混进来的零宽字符/软连字符——「隐形字符那位」就是被这个坑过
_RE_INVISIBLE = re.compile(r"[\s\u200b-\u200f\u2060\ufeff\u00ad]+")


def name_key(name) -> str:
    """名字归一：去掉空白和不可见字符、大小写折叠。

    聊天记录里的昵称经常混零宽空格或前后空格，直接比字符串会认成两个人。
    """
    return _RE_INVISIBLE.sub("", str(name or "")).casefold()


def _parse_month_day(s: str, now: float) -> float | None:
    """认 `11-04 09:59:33` / `11/04 09:59` / `11-04` 这类没有年份的写法。

    年份靠猜：先按当前年算，如果算出来的时间比现在还晚两天以上，
    说明是去年的记录（跨年补录）。猜错两天以内无所谓，猜错一年很致命。
    """
    m = re.match(
        r"^(\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$", s.strip())
    if not m:
        return None
    mon, day = int(m.group(1)), int(m.group(2))
    if not (1 <= mon <= 12 and 1 <= day <= 31):
        return None
    hh = int(m.group(3) or 0)
    mi = int(m.group(4) or 0)
    ss = int(m.group(5) or 0)
    if hh > 23 or mi > 59 or ss > 59:
        return None
    year = time.localtime(now).tm_year
    for y in (year, year - 1):
        try:
            t = time.mktime((y, mon, day, hh, mi, ss, 0, 0, -1))
        except (ValueError, OverflowError):
            continue
        if t <= now + 2 * 86400:
            return t
    return None


def parse_ts(val, *, now: float = 0.0) -> float | None:
    """把面板/导入进来的时间解析成 epoch 秒。

    认四种写法：秒/毫秒时间戳、`2026-08-01 21:30(:00)`、`2026-08-01T21:30`、
    以及聊天记录里常见的不带年份的 `11-04 09:59:33`。
    认不出返回 None，调用方退回当前时间——补录数据宁可整批按导入时刻算，
    也不要因为一个脏字段整批报错。
    """
    if val is None or val == "" or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
    else:
        s = str(val).strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            s = s.replace("Z", "").strip()
            for fmt in TS_FORMATS:
                try:
                    return time.mktime(time.strptime(s, fmt))
                except ValueError:
                    continue
            return _parse_month_day(s, now or time.time())
    if v <= 0:
        return None
    # 13 位是毫秒（前端 Date.now() 常见），10 位是秒
    return v / 1000.0 if v > 1e11 else v


def fmt_ts(ts: float) -> str:
    if not ts or ts <= 0:
        return "—"
    lt = time.localtime(ts)
    return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"


def _match_head(line: str, now: float):
    """判断这一行是不是「谁在什么时候说的」。

    返回 (ts, 名字, 同行内容 or None)；不是就返回 None。
    同行内容为 None 表示真正的发言在下一行（QQ 复制出来就是这形态）。
    时间解析必须成功才算头行——`[图片] 昵称: 内容` 这种不能误吃。
    """
    for pat, has_text in ((_RE_HEAD_BRACKET, True), (_RE_TIME_FIRST, True),
                          (_RE_HEAD_TIME, False), (_RE_HEAD_TIME_NC, False)):
        m = pat.match(line)
        if not m:
            continue
        ts = parse_ts(m.group("ts"), now=now)
        if ts is None:
            continue
        name = (m.group("name") or "").strip()
        if not name:
            continue
        text = (m.group("text") or "").strip() if has_text else None
        return ts, name, text
    m = None
    if "|" not in line and "\t" not in line:
        # 竖线/制表符行交给行式解析；这里再抢一遍会把 `时间 | 用户 | 内容` 拆坏
        m = _RE_TIME_ONLY.match(line)
    if m:
        ts = parse_ts(m.group("ts"), now=now)
        if ts is not None:
            return ts, "", m.group("text").strip()
    return None


def parse_import_text(text: str, default_subject: str = "",
                      *, now: float = 0.0) -> list[dict]:
    """把粘贴进来的东西解析成一条条事件。

    三种格式都吃：
      1. 聊天记录复制体（QQ/微信最常见，两行一条）：
             昵称: 11-04 09:59:33
             说了什么
      2. 竖线式：`时间 | 用户 | 内容`，字段少写也能认
      3. 裸文本：一行一条，没时间没归属

    返回每项：ts / speaker（谁说的）/ subject（归属用户）/ content。
    没写时间就按导入时刻算——不猜、不乱填，猜错比不猜更糟。
    以 # 开头的行当注释跳过。
    """
    now = now or time.time()
    rows: list[dict] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is not None:
            body = " ".join(cur["buf"]).strip()
            if body:
                rows.append({
                    "ts": cur["ts"], "speaker": cur["speaker"],
                    "subject": default_subject, "content": body,
                })
            cur = None

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 竖线式先判：`2026-08-01 21:30 | 用户甲 | 内容` 里的 `21:30`
        # 会被「名字+时间」的正则回溯成 name=21，必须先在这里截住。
        if cur is None and ("|" in line or "\t" in line):
            parts = [p.strip() for p in re.split(r"\s*[|\t]\s*", line)]
            ts = None
            speaker = ""
            content = ""
            if len(parts) >= 3:
                ts = parse_ts(parts[0], now=now)
                speaker = parts[1]
                content = " | ".join(p for p in parts[2:] if p)
            elif len(parts) == 2:
                ts = parse_ts(parts[0], now=now)
                if ts is not None:
                    content = parts[1]
                else:
                    speaker = parts[0]
                    content = parts[1]
            else:
                content = parts[0]
            if content:
                rows.append({"ts": ts, "speaker": speaker,
                             "subject": speaker or default_subject, "content": content})
            continue
        head = _match_head(line, now)
        if head:
            flush()
            ts, name, same_line = head
            if same_line:
                rows.append({"ts": ts, "speaker": name,
                             "subject": default_subject, "content": same_line})
            else:
                cur = {"ts": ts, "speaker": name, "buf": []}
            continue
        if cur is not None:
            cur["buf"].append(line)
            continue
        rows.append({"ts": None, "speaker": "", "subject": default_subject, "content": line})
    flush()
    return rows


def assign_roles(rows: list[dict], *, default_type: str = "user_message",
                 ai_names=None, owner: str = "", scope: str = "私") -> dict:
    """判每行是 AI 说的还是人说的，并拼成插件录制时的原始格式。

    为什么非得拼信封：总结器的提示词只认两种开头——`名字: 内容` 是人说的，
    `[你] 内容` 是 AI 说的。裸文本进去，AI 的话会被提炼成用户的事实，
    等于把编造的东西写进长期记忆（这一层污染不可自愈）。

    判定优先级：
      1. 行里已经写了 event_type（JSON 导入）→ 尊重，不改
      2. 填了 AI 昵称 → 在名单里算 AI，否则算人
      3. 没填但填了归属用户 → 私聊默认：跟归属用户同名算人，其他算 AI
      4. 都没有 → 用默认类型

    返回统计：{ai, user, other, speakers:[{name,n,kind}]}
    """
    ai_set = {name_key(n) for n in (ai_names or []) if str(n).strip()}
    owner_key = name_key(owner)
    stat = {"ai": 0, "user": 0, "other": 0, "speakers": {}}
    last_human = owner

    for r in rows:
        speaker = (r.get("speaker") or "").strip()
        raw = (r.get("content") or "").strip()
        if not raw:
            continue
        etype = r.get("event_type") or ""
        if not etype:
            key = name_key(speaker)
            if key and ai_set:
                etype = "bot_reply" if key in ai_set else "user_message"
            elif key and owner_key:
                etype = "user_message" if key == owner_key else "bot_reply"
            else:
                etype = default_type
        r["event_type"] = etype

        if etype == "bot_reply":
            subject = last_human or owner
            r["subject"] = subject
            if not _RE_ENVELOPE.match(raw):
                r["content"] = f"[你] {raw}"[:300]
            stat["ai"] += 1
        elif etype == "user_message":
            subject = speaker or r.get("subject") or owner
            r["subject"] = subject
            if subject:
                last_human = subject
            if not _RE_ENVELOPE.match(raw):
                r["content"] = (f"[{scope}] {subject}: {raw}" if subject
                                else f"[{scope}] {raw}")[:300]
            stat["user"] += 1
        else:
            r["subject"] = speaker or r.get("subject") or owner
            r["content"] = raw[:300]
            stat["other"] += 1

        if speaker:
            slot = stat["speakers"].setdefault(
                speaker, {"name": speaker, "n": 0, "ai": 0, "user": 0})
            slot["n"] += 1
            slot["ai" if etype == "bot_reply" else "user"] += 1
    return stat


def import_report(rows: list[dict], cap: int = 14, ai_names=None, owner: str = "") -> dict:
    """导入前体检。

    这不是防呆，是防真事故：几千条挤在同一天，骨架会退化成一锅高频词、
    时序锚全部失效、总结任务连环触发。等提炼乱了再回头看，L2 已经被污染。
    """
    total = len(rows)
    owner = owner or ""
    timed = [r for r in rows if r.get("ts")]
    bad = total - len(timed)
    days: dict[str, int] = {}
    hours: dict[str, int] = {}
    for r in timed:
        lt = time.localtime(r["ts"])
        d = f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"
        h = f"{d} {lt.tm_hour:02d}"
        days[d] = days.get(d, 0) + 1
        hours[h] = hours.get(h, 0) + 1
    per_day = sorted(days.items(), key=lambda kv: -kv[1])[:cap]
    per_day.sort(key=lambda kv: kv[0])
    warnings: list[str] = []
    hints: list[str] = []
    if timed:
        top_day, top_day_n = max(days.items(), key=lambda kv: kv[1])
        top_hour, top_hour_n = max(hours.items(), key=lambda kv: kv[1])
        if total >= 100 and top_day_n / max(1, len(timed)) >= 0.8:
            warnings.append(
                f"{top_day_n} 条（{top_day_n * 100 // max(1, len(timed))}%）挤在 {top_day} "
                f"这一天。骨架按天分桶，同一天塞太多会把话题词全当套话滤掉，"
                f"提炼出来的是废的。建议按真实日期拆开再导。"
            )
        if top_hour_n >= 100 and top_hour_n / max(1, len(timed)) >= 0.5:
            warnings.append(
                f"{top_hour_n} 条落在 {top_hour} 时这一个小时里，时序完全不可分。"
                f"补录数据至少按天摊开。"
            )
    if bad:
        first_bad = next((r for r in rows if not r.get("ts")), None)
        sample = ((first_bad or {}).get("content") or "")[:24]
        hint = f"（第一条没时间的是「{sample}」）" if sample else ""
        warnings.append(
            f"{bad} 条没有可用时间，会按导入时刻落库（等于全挤在现在）{hint}。"
            "查一下这些行是不是没带时间，或者格式没被认出来。"
        )
    no_subject = sum(1 for r in rows if not (r.get("subject") or "").strip())
    if no_subject:
        warnings.append(f"{no_subject} 条没有归属用户，会落在「未归类」里，之后认不出是谁的。")

    ai_n = sum(1 for r in rows if r.get("event_type") == "bot_reply")
    user_n = sum(1 for r in rows if r.get("event_type") == "user_message")
    spk: dict[str, dict] = {}
    for r in rows:
        name = (r.get("speaker") or "").strip()
        if not name:
            continue
        slot = spk.setdefault(name, {"name": name, "n": 0, "ai": 0, "user": 0})
        slot["n"] += 1
        if r.get("event_type") == "bot_reply":
            slot["ai"] += 1
        elif r.get("event_type") == "user_message":
            slot["user"] += 1
    speakers = sorted(spk.values(), key=lambda x: -x["n"])[:8]
    if ai_names and speakers and ai_n == 0:
        warnings.append(
            "填了 AI 昵称，但一条都没匹配上——检查名字有没有写错"
            "（表情、空格、繁体都算不同名字）。现在全部按「人说的」处理了。"
        )
    elif not ai_names and len(speakers) >= 2 and ai_n == 0:
        warnings.append(
            "这批里有不止一个人在说话，但一条 AI 发言都没认出来，"
            "会全部当成「人说的」写进记忆。去上面「AI 的昵称」填上它的名字再体检一次。"
        )
    # 归属用户写错名字是最容易犯的错：聊天记录里叫「Moon」，插件里叫「张三」，
    # 一对不上就人人都成了 AI。这比沉默地写错强。
    if owner.strip() and speakers:
        ok = name_key(owner)
        if not any(name_key(s["name"]) == ok for s in speakers):
            names = "、".join(f"「{s['name']}」" for s in speakers[:4])
            warnings.append(
                f"「{owner}」在记录里一次都没出现（记录里是 {names}）——"
                f"聊天记录里的昵称可能跟插件里的不一样。把「归属用户」改成你自己的那个，"
                f"否则会认错谁是谁。"
            )
    if speakers and ai_n and user_n == 0:
        warnings.append(
            f"{ai_n} 条全被判成 AI 说话了，一条人说的都没有。检查「归属用户」"
            f"和「AI 的昵称」是不是写反了。"
        )

    # 提炼是按人走的：群聊记录里别人的话不会变成这个人的长期记忆。
    # 体检先说清楚，省得导完以为「怎么只记住了我自己」。
    human = [s["name"] for s in speakers if s.get("user")]
    if owner.strip() and len(human) > 1:
        names = "、".join(f"「{n}」" for n in human[:4])
        hints.append(
            f"这批里有 {len(human)} 个人说话（{names}），提炼只按归属用户「{owner}」走，"
            f"其他人的原话不会变成长期记忆。要清算别人，换归属用户分批导。"
        )

    span_days = 0
    if timed:
        first = min(r["ts"] for r in timed)
        last = max(r["ts"] for r in timed)
        span_days = int((last - first) // 86400) + 1
    return {
        "total": total,
        "timed": len(timed),
        "bad_ts": bad,
        "span_days": span_days,
        "first": fmt_ts(min((r["ts"] for r in timed), default=0)),
        "last": fmt_ts(max((r["ts"] for r in timed), default=0)),
        "per_day": [{"day": d, "n": n} for d, n in per_day],
        "day_total": len(per_day),
        "ai_count": ai_n,
        "user_count": user_n,
        "speakers": speakers,
        "warnings": warnings,
        "hints": hints,
    }
