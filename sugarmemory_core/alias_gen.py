"""L2 问法别名生成：一次 LLM 批量调用，给每条事实配「用户可能怎么问起它」。

为什么单独一个模块而不是塞进总结 prompt：总结的 JSON schema 已经够复杂
（摘要 + 多条断言 + speech_act/source_quote/confidence），再往里塞 alias
字段，解析健壮性风险翻倍。独立小请求只做一件事，失败就下次再试，互不拖累。

检索侧（pack.py）把 ask_alias 的文本和 value 一起做 2-gram 匹配——
「你猜我多大了」撞不上「现在 20 岁」，但撞得上「多大了/几岁」这种问法别名。
"""

ALIAS_SYSTEM_PROMPT = (
    "你是记忆检索助手。给你若干条关于某个用户的记忆事实，"
    "请为每条事实生成 3 个「用户提起这件事时最可能说的话」。\n"
    "要求：\n"
    "- 口语化、短句，像聊天输入，不要书面语\n"
    "- 必须换一种说法，不能照抄事实原文里的字面组合（原词留太多检索就学不到新东西）\n"
    "- 覆盖不同的问法角度：直接问、随口提、相关话题带出\n"
    "- 只输出 JSON 数组，格式：[{\"id\": <原样返回的id>, \"ask\": [\"问法1\", \"问法2\", \"问法3\"]}]，"
    "不要输出任何其他文字"
)


def build_alias_prompt(items: list[dict]) -> str:
    """把待生成的事实列表拼成 user prompt。items 来自 store.l2_need_alias。"""
    lines = []
    for it in items:
        value = str(it.get("value") or "").replace("\n", " ")[:80]
        lines.append(f"- id={it.get('id')} | {it.get('subject')}的"
                     f"{it.get('predicate')}：{value}")
    return "记忆事实：\n" + "\n".join(lines)


def parse_alias_json(raw: str) -> dict[int, list[str]]:
    """解析 LLM 输出为 {id: [问法...]}。

    带截断修复和中文标点兜底（与 summarize_core.parse_summary_json 同一风格，
    但结构简单得多，独立实现避免互相牵连）。解析不出就返回空 dict，调用方
    静默放弃这一批，下次再试。
    """
    import json
    import re

    text = str(raw or "").strip()
    if not text:
        return {}
    # 剥掉 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("[")
    if start < 0:
        return {}
    # 中文标点兜底：只处理结构性标点（值内部出现同款中文标点也无害，
    # 只会改变字符串内容、不会破坏结构）
    body = text[start:].replace("，", ",").replace("：", ":").replace("；", ";")
    data = None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # 截断/尾随废话修复：剥到最后一个完整的对象（最近 "}），
        # 按 { } 差数补齐闭括号再试一次。不能用 rfind("]") 切尾——
        # 截断输出里数组内部的 ] 会把合法对象一起切残。
        last = body.rfind("}")
        if last < 0:
            return {}
        body2 = body[: last + 1].rstrip(",")
        opens = body2.count("{") - body2.count("}")
        body2 = body2 + "}" * max(0, opens) + "]"
        try:
            data = json.loads(body2)
        except json.JSONDecodeError:
            return {}
    if data is None or not isinstance(data, list):
        return {}
    out: dict[int, list[str]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            fid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        asks = item.get("ask")
        if not isinstance(asks, list):
            continue
        cleaned = []
        for a in asks:
            s = re.sub(r"\s+", "", str(a or ""))
            if len(s) >= 2:
                cleaned.append(s)
        if cleaned:
            out[fid] = cleaned[:4]
    return out
