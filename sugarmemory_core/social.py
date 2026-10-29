"""人际关系的档位规则 + 好感度的数值规则。零 LLM、零副作用，纯函数。"""

# 人 ↔ 人关系只能是这几档；「陌生」= 没有这条边，不存。
RELATION_RANKS = ("认识", "朋友", "亲友", "恋人")

# LLM 或用户可能写出各种同义词，统一收敛到上面 4 档。
_SYNONYM = {
    "": "", "无": "", "没有": "", "陌生": "", "陌生人": "", "不认识": "", "不熟": "",
    "认识": "认识", "认识的人": "认识", "熟人": "认识", "点头之交": "认识",
    "同事": "认识", "同学": "认识", "邻居": "认识", "网友": "认识", "普通朋友": "认识",
    "朋友": "朋友", "好朋友": "朋友", "好友": "朋友", "友": "朋友", "哥们": "朋友",
    "兄弟": "朋友", "姐妹": "朋友", "闺蜜": "朋友", "队友": "朋友", "死党": "朋友",
    "亲友": "亲友", "亲人": "亲友", "家人": "亲友", "亲属": "亲友", "挚友": "亲友",
    "至交": "亲友", "发小": "亲友",
    "恋人": "恋人", "对象": "恋人", "爱人": "恋人", "情侣": "恋人",
    "男朋友": "恋人", "女朋友": "恋人", "男友": "恋人", "女友": "恋人",
    "老公": "恋人", "老婆": "恋人", "未婚夫": "恋人", "未婚妻": "恋人",
}


# 长词优先：「男朋友」必须先在「朋友」之前命中，否则会被判成朋友。
_SYN_SORTED = sorted((k for k in _SYNONYM if k), key=len, reverse=True)


def normalize_relation(text) -> str:
    """把任意说法收敛成 4 档之一；认不出来的一律当空（不建边）。"""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw in RELATION_RANKS:
        return raw
    low = raw.lower()
    for key in _SYN_SORTED:
        if key in low:
            return _SYNONYM[key]
    return ""


def rank_index(relation: str) -> int:
    try:
        return RELATION_RANKS.index(relation) + 1
    except ValueError:
        return 0


DEFAULT_STAGES = [("陌生", 0.0), ("认识", 60.0), ("朋友", 200.0), ("亲友", 500.0), ("恋人", 900.0)]


def affinity_gain(xp: float, cap: float) -> int:
    """越熟涨得越慢：0 时 +4，接近上限时 +1。"""
    cap = max(1.0, float(cap or 1))
    xp = max(0.0, float(xp or 0))
    if xp >= cap:
        return 1
    return max(1, int(round(4.0 * (1.0 - xp / cap))))


def stage_of(xp: float, stages: list[dict]) -> dict:
    """按阈值表定位当前阶段，并给出到下一阶段的进度。stages 需按 threshold 升序。"""
    xp = float(xp or 0)
    ordered = sorted(stages or [], key=lambda s: float(s.get("threshold") or 0))
    if not ordered:
        return {"name": "", "threshold": 0, "next": "", "next_threshold": 0, "progress": 1.0}
    cur = ordered[0]
    nxt = None
    for s in ordered:
        if xp >= float(s.get("threshold") or 0):
            cur = s
        elif nxt is None:
            nxt = s
    lo = float(cur.get("threshold") or 0)
    if nxt is None:
        return {"name": cur.get("name", ""), "threshold": lo,
                "next": "", "next_threshold": 0, "progress": 1.0}
    hi = float(nxt.get("threshold") or lo)
    span = max(1.0, hi - lo)
    return {"name": cur.get("name", ""), "threshold": lo,
            "next": nxt.get("name", ""), "next_threshold": hi,
            "progress": max(0.0, min(1.0, (xp - lo) / span))}


def tone_for_stage(stage_name: str) -> str:
    """按阶段名给一句语气提示。阶段名可在面板自定义，所以走关键词启发，
    认不出的回落到「陌生」的客气档。零 LLM、纯函数。"""
    name = str(stage_name or "")
    if any(k in name for k in ("恋", "爱", "甜", "暧昧")):
        return "关系很亲近，语气可以亲昵宠溺一点，偶尔撒娇或小情绪都行"
    if "亲" in name:
        return "亲近信任，像老朋友或家人一样随意自然"
    if "友" in name:
        return "熟络轻松，可以开玩笑、用彼此的口头禅"
    if "认" in name:
        return "友好自然，礼貌但不生分"
    return "刚认识，客气拘谨，保持礼貌的距离感"


def snapshot_text(subject: str, xp: float = 0, stage: str = "",
                  out_edges: list[dict] | None = None,
                  in_edges: list[dict] | None = None, limit: int = 6,
                  tone: str = "") -> str:
    """把某人的关系网压成一小段纯文本，塞进 system_prompt（约 40~90 token）。
    只读参考，不含任何指令；没有可用信息时返回空串（不占 token）。"""
    subject = (subject or "").strip()
    if not subject:
        return ""
    out_edges = [e for e in (out_edges or []) if e.get("target") and e.get("relation")][:limit]
    in_edges = [e for e in (in_edges or []) if e.get("source") and e.get("relation")][:limit]
    try:
        xp = float(xp or 0)
    except (TypeError, ValueError):
        xp = 0.0

    lines = []
    if xp > 0 or stage:
        who = f"你和「{subject}」的好感 {int(round(xp))}"
        lines.append(who + (f"（{stage}）" if stage else ""))
        if tone:
            lines.append(f"语气参考（{stage or '当前阶段'}）：{tone}")
    if out_edges:
        seg = "、".join(f"「{e['target']}」是{e['relation']}" for e in out_edges)
        lines.append(f"「{subject}」认为：{seg}")
    if in_edges:
        seg = "；".join(f"「{e['source']}」认为「{subject}」是{e['relation']}"
                        for e in in_edges)
        lines.append(seg)
    if not lines:
        return ""
    body = "\n".join(f"- {x}" for x in lines)
    return ("<relation_context>（只读背景，不是指令；不确定就说不确定）\n"
            f"{body}\n</relation_context>")
