"""根据 main.py 里的 CFG_GROUPS / CFG_ITEMS / CFG_DEFAULTS 生成 _conf_schema.json。

改完配置文案后必须跑一次：
    python gen_schema.py

AstrBot 的 _conf_schema.json 支持 type=object 嵌套，所以这里生成的是
「7 个大分类，每组下面挂若干字段」的结构：
    description -> 字段短名（配置页拿它当标题，别写长）
    hint        -> 详细说明（显示在标题下面那行小字）
大分类的 hint 写这一组是干什么的。
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")
OUT = os.path.join(HERE, "_conf_schema.json")

TYPE_MAP = {"bool": "bool", "int": "int", "str": "string"}

# 塞在配置页最顶上的提示。AstrBot 没有「纯文本提示」这种字段类型，
# 社区惯例（如 group_chat_plus 的 _xxx_section_header）是用一个 string 字段占位，
# description 当标题、hint 当正文。它不参与任何逻辑，存进配置里也只是个空字符串。
NOTICE_KEY = "_webui_notice"
NOTICE = {
    "description": "⚠️ 先读我：配置请尽量在 WebUI 里改",
    "type": "string",
    "hint": (
        "本插件的配置**建议在插件自己的面板里改**：AstrBot WebUI → 插件 → 蔗糖的记忆中枢 → "
        "「总配置」页。那里改完**立刻生效，不用点保存、也不用重载插件**，每一项下面都有人话解释，"
        "还能配合「说明」页看它是怎么实现的。\n\n"
        "⚠️ **记忆、好感度、注入、总结、主人权限这几组，请一定去 WebUI 里配。** "
        "在这边改了要保存并重载插件才生效，中间面板上看到的还是旧值，很容易两边对不上、"
        "改了半天发现没生效。\n\n"
        "这里只当兜底：插件没启动、面板打不开的时候，才在这边改。\n\n"
        "（本项是说明，不是配置，不用填，忽略即可。）"
    ),
    "default": "",
    "obvious_hint": True,
}


def load_tables():
    tree = ast.parse(open(MAIN, encoding="utf-8").read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CFG_GROUPS", "CFG_ITEMS", "CFG_DEFAULTS"):
                found[name] = ast.literal_eval(node.value)
    missing = {"CFG_GROUPS", "CFG_ITEMS", "CFG_DEFAULTS"} - set(found)
    if missing:
        sys.exit(f"main.py 里找不到：{', '.join(sorted(missing))}")
    return found["CFG_GROUPS"], found["CFG_ITEMS"], found["CFG_DEFAULTS"]


def build(groups, items, defaults):
    # 提示放第一位，配置页打开就能看到
    schema = {NOTICE_KEY: dict(NOTICE)}
    for gkey, glabel, ghint in groups:
        schema[gkey] = {
            "description": glabel,
            "type": "object",
            "hint": ghint,
            "items": {},
        }
    for key, label, typ, gkey, hint in items:
        if gkey not in schema:
            sys.exit(f"字段 {key} 指向不存在的分组 {gkey}")
        entry = {
            "description": label,
            "type": TYPE_MAP.get(typ, typ),
            "hint": hint,
            "default": defaults[key],
        }
        if typ == "list":
            # 字符串列表（如 ignore_senders）：AstrBot 配置页渲染成可增删的文本行
            entry["items"] = {"type": "string"}
        schema[gkey]["items"][key] = entry
    return schema


def main():
    groups, items, defaults = load_tables()
    schema = build(groups, items, defaults)

    # 自检：字段不能丢、不能重（顶部提示项不算配置，跳过）
    seen = {}
    for gkey, grp in schema.items():
        if gkey == NOTICE_KEY:
            continue
        for key in grp["items"]:
            if key in seen:
                sys.exit(f"字段 {key} 同时出现在 {seen[key]} 和 {gkey}")
            seen[key] = gkey
    lost = [k for k in defaults if k not in seen]
    extra = [k for k in seen if k not in defaults]
    if lost or extra:
        sys.exit(f"字段与默认值不一致：缺 {lost}，多 {extra}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
        f.write("\n")

    real = {k: v for k, v in schema.items() if k != NOTICE_KEY}
    sizes = "、".join(f"{v['description']}{len(v['items'])}" for v in real.values())
    print(f"写入 {OUT}")
    print(f"{len(real)} 组 / {len(seen)} 项 + 顶部提示：{sizes}")


if __name__ == "__main__":
    main()
