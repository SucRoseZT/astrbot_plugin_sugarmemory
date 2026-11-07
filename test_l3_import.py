# -*- coding: utf-8 -*-
"""L3 补录/批量导入的离线测试：时间解析、行式解析、导入体检、批量写入。

不依赖 astrbot，直接加载 sugarmemory_core 下的纯模块。
"""
import importlib.util
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


li = _load("sugarmemory_core.l3_import", "sugarmemory_core/l3_import.py")
st = _load("sugarmemory_core.store", "sugarmemory_core/store.py")
parse_ts = li.parse_ts
parse_import_text = li.parse_import_text
import_report = li.import_report

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ---------- 1. 时间解析 ----------
print("\n===== parse_ts =====")
expect = time.mktime(time.strptime("2026-08-01 21:30", "%Y-%m-%d %H:%M"))
check("空格格式", parse_ts("2026-08-01 21:30") == expect)
check("T 格式(datetime-local)", parse_ts("2026-08-01T21:30") == expect)
check("带秒", parse_ts("2026-08-01 21:30:00") == expect)
check("只有日期", parse_ts("2026-08-01") == time.mktime(time.strptime("2026-08-01", "%Y-%m-%d")))
check("秒级时间戳", parse_ts(1770000000) == 1770000000.0)
check("毫秒时间戳", parse_ts(1770000000000) == 1770000000.0)
check("字符串时间戳", parse_ts("1770000000") == 1770000000.0)
check("空串返回 None", parse_ts("") is None)
check("None 返回 None", parse_ts(None) is None)
check("垃圾返回 None", parse_ts("昨天下午") is None)
check("0 返回 None", parse_ts(0) is None)
check("斜杠格式", parse_ts("2026/08/01 21:30") == expect)
check("bool 不当数字", parse_ts(True) is None)

# 聊天记录里常见的不带年份写法：按当前年推断，跨年自动回退一年
now_ref = time.mktime(time.strptime("2026-11-04 12:00", "%Y-%m-%d %H:%M"))
check("无年份按今年算",
      abs(parse_ts("11-04 09:59:33", now=now_ref)
          - time.mktime(time.strptime("2026-11-04 09:59:33", "%Y-%m-%d %H:%M:%S"))) < 1)
now_jan = time.mktime(time.strptime("2027-01-05 10:00", "%Y-%m-%d %H:%M"))
check("无年份跨年回退一年",
      abs(parse_ts("12-28 21:00", now=now_jan)
          - time.mktime(time.strptime("2026-12-28 21:00", "%Y-%m-%d %H:%M"))) < 1)
check("无年份非法月日返回 None", parse_ts("13-45 09:00", now=now_ref) is None)

# ---------- 2. 行式解析 ----------
print("\n===== parse_import_text =====")
rows = parse_import_text("2026-08-01 21:30 | 用户甲 | 今天下班好晚\n# 注释行\n\n用户乙 | 早上啃了个面包\n就一句话")
check("跳过空行与注释", len(rows) == 3, f"got {len(rows)}")
check("三字段取时间/用户/内容",
      rows[0]["ts"] is not None and rows[0]["subject"] == "用户甲" and rows[0]["content"] == "今天下班好晚")
check("两字段无时间当用户",
      rows[1]["ts"] is None and rows[1]["subject"] == "用户乙" and rows[1]["content"] == "早上啃了个面包")
check("单字段当内容", rows[2]["content"] == "就一句话" and rows[2]["ts"] is None)
check("默认用户兜底",
      parse_import_text("随便说点什么", "用户丙")[0]["subject"] == "用户丙")
rows2 = parse_import_text("2026-08-01 21:30 | 用户甲 | 内容里有 | 竖线")
check("内容里的竖线不被切碎", rows2[0]["content"] == "内容里有 | 竖线", rows2[0]["content"])
check("制表符分隔",
      parse_import_text("2026-08-01\t用户甲\t吃了吗")[0]["content"] == "吃了吗")

# ---------- 2b. 聊天记录复制体（QQ/微信最常见） ----------
print("\n===== 聊天记录格式 =====")
chat = parse_import_text(
    "Алиса: 11-04 09:59:33\n鱼来试试！\n主人: 11-04 10:01:17\n太好了。", "主人")
check("两行式解析出 2 条", len(chat) == 2, f"got {len(chat)}")
check("头行时间被认出来", chat[0]["ts"] is not None)
check("头行名字进 speaker", chat[0]["speaker"] == "Алиса", chat[0]["speaker"])
check("下一行才是内容", chat[0]["content"] == "鱼来试试！", chat[0]["content"])
check("归属用默认值", chat[0]["subject"] == "主人", chat[0]["subject"])
check("第二条也对", chat[1]["speaker"] == "主人" and chat[1]["content"] == "太好了。")

chat2 = parse_import_text("Алиса: 11-04 09:59:33\n第一句\n第二句\n主人: 11-04 10:00:00\n嗯")
check("同一人的续行合并成一条", len(chat2) == 2, f"got {len(chat2)}")
check("续行用空格拼起来", chat2[0]["content"] == "第一句 第二句", chat2[0]["content"])

brk = parse_import_text("[2026-11-04 09:59:33] Алиса: 鱼来试试！")
check("方括号单行式", brk[0]["speaker"] == "Алиса" and brk[0]["content"] == "鱼来试试！"
      and brk[0]["ts"] is not None, str(brk))
tf = parse_import_text("2026-11-04 09:59:33 Алиса: 鱼来试试！")
check("时间在前", tf[0]["speaker"] == "Алиса" and tf[0]["content"] == "鱼来试试！", str(tf))
nc = parse_import_text("Алиса 11-04 09:59:33\n鱼来试试！")
check("头行没冒号", nc[0]["speaker"] == "Алиса" and nc[0]["content"] == "鱼来试试！", str(nc))
tonly = parse_import_text("2026-11-04 09:59:33 随便说一句")
check("只有时间+内容时不被拆成 spk=09",
      tonly[0]["speaker"] == "" and tonly[0]["content"] == "随便说一句", str(tonly))
check("竖线式不被时间正则抢走",
      parse_import_text("2026-08-01 21:30 | 用户甲 | 今天下班好晚")[0]["speaker"] == "用户甲")
check("隐形字符归一后同名",
      li.name_key("А\u200bлиса ") == li.name_key("Алиса"))

# ---------- 2c. 说话人判定与信封 ----------
print("\n===== assign_roles =====")
rr = parse_import_text("Алиса: 11-04 09:59:33\n猫娘说一句\n主人: 11-04 10:01:17\n主人说一句")
st1 = li.assign_roles(rr, ai_names=["Алиса"], owner="主人", scope="私")
check("AI 名单命中判 bot_reply", rr[0]["event_type"] == "bot_reply", rr[0]["event_type"])
check("其他人判 user_message", rr[1]["event_type"] == "user_message", rr[1]["event_type"])
check("AI 的话挂到对话对象名下", rr[0]["subject"] == "主人", rr[0]["subject"])
check("AI 内容带 [你] 信封", rr[0]["content"].startswith("[你] "), rr[0]["content"])
check("人说的话带 名字: 信封", rr[1]["content"].startswith("[私] 主人: "), rr[1]["content"])
check("统计对得上", st1["ai"] == 1 and st1["user"] == 1, str(st1))

rr2 = parse_import_text("Алиса: 11-04 09:59:33\n猫娘说一句\n主人: 11-04 10:01:17\n主人说一句")
li.assign_roles(rr2, ai_names=[], owner="主人", scope="私")
check("没填 AI 昵称时：非归属用户当 AI",
      rr2[0]["event_type"] == "bot_reply" and rr2[1]["event_type"] == "user_message",
      f"{rr2[0]['event_type']}/{rr2[1]['event_type']}")

rr3 = [{"ts": 0, "speaker": "", "subject": "", "content": "纯文本", "event_type": ""}]
li.assign_roles(rr3, default_type="manual_note", ai_names=[], owner="")
check("都没填就走默认类型", rr3[0]["event_type"] == "manual_note", rr3[0]["event_type"])

rr4 = [{"ts": 0, "speaker": "Алиса", "subject": "", "content": "你之前说过这句",
        "event_type": "user_message"}]
li.assign_roles(rr4, ai_names=["Алиса"], owner="主人")
check("JSON 里写明类型就不改判", rr4[0]["event_type"] == "user_message", rr4[0]["event_type"])

rr5 = [{"ts": 0, "speaker": "主人", "subject": "", "content": "[私] 主人: 已经有信封了",
        "event_type": "user_message"}]
li.assign_roles(rr5, ai_names=[], owner="主人")
check("已有信封不重复套", rr5[0]["content"] == "[私] 主人: 已经有信封了", rr5[0]["content"])

# ---------- 3. 导入体检 ----------
print("\n===== import_report =====")
base = time.mktime(time.strptime("2026-08-01 12:00", "%Y-%m-%d %H:%M"))
same_day = [{"ts": base + i, "subject": "用户甲", "content": f"m{i}"} for i in range(500)]
r = import_report(same_day)
check("同一天 500 条触发警告", any("挤在" in w for w in r["warnings"]), str(r["warnings"])[:120])
check("跨度算出来是 1 天", r["span_days"] == 1, str(r["span_days"]))
check("总数正确", r["total"] == 500 and r["timed"] == 500)

spread = [{"ts": base + i * 86400, "subject": "用户甲", "content": f"m{i}"} for i in range(40)]
r2 = import_report(spread)
check("分散 40 天不告警", not any("挤在" in w for w in r2["warnings"]), str(r2["warnings"]))
check("跨度 40 天", r2["span_days"] == 40, str(r2["span_days"]))

r3 = import_report([{"ts": None, "subject": "", "content": "x"}])
check("无时间告警", any("没有可用时间" in w for w in r3["warnings"]))
check("无归属告警", any("没有归属用户" in w for w in r3["warnings"]))
check("全无时间也不炸", r3["timed"] == 0 and r3["bad_ts"] == 1)

small = [{"ts": base + i, "subject": "用户甲", "content": f"m{i}"} for i in range(30)]
check("30 条同日不告警（阈值 100）", not any("挤在" in w for w in import_report(small)["warnings"]))

spk_rows = [
    {"ts": base, "speaker": "Алиса", "subject": "主人", "content": "x", "event_type": "bot_reply"},
    {"ts": base, "speaker": "Алиса", "subject": "主人", "content": "y", "event_type": "bot_reply"},
    {"ts": base, "speaker": "主人", "subject": "主人", "content": "z", "event_type": "user_message"},
]
rep_spk = import_report(spk_rows, ai_names=["Алиса"])
check("说话人统计", rep_spk["speakers"][0]["name"] == "Алиса"
      and rep_spk["speakers"][0]["n"] == 2 and rep_spk["speakers"][0]["ai"] == 2,
      str(rep_spk["speakers"]))
check("AI/人计数", rep_spk["ai_count"] == 2 and rep_spk["user_count"] == 1)
rep_bad = import_report([{"ts": base, "speaker": "主人", "subject": "主人",
                          "content": "x", "event_type": "user_message"}],
                        ai_names=["驴头不对马嘴的名字"])
check("AI 昵称一个都没匹配上要告警",
      any("一条都没匹配上" in w for w in rep_bad["warnings"]), str(rep_bad["warnings"]))

# 多人批次：提炼是按人走的，必须提前说清只清算归属用户
multi = [
    {"ts": None, "speaker": "阿明", "subject": "阿明", "content": "x", "event_type": "user_message"},
    {"ts": None, "speaker": "路人乙", "subject": "路人乙", "content": "y", "event_type": "user_message"},
]
rm = import_report(multi, owner="阿明")
check("多人批次提示只提炼归属用户", any("提炼只按归属用户" in h for h in rm["hints"]),
      str(rm["hints"]))
one = import_report(multi[:1], owner="阿明")
check("单人批次不提示", not one["hints"], str(one["hints"]))

# ---------- 4. 批量写入 ----------
print("\n===== write_l3_bulk =====")
db_path = os.path.join(tempfile.gettempdir(), "sugarmem_import_test.db")
if os.path.exists(db_path):
    os.remove(db_path)
store = st.SugarStore(db_path)
exp = time.time() + 14 * 86400
batch = [{"event_type": "user_message", "subject": "用户甲", "content": f"第{i}条",
          "session_id": "s1", "created_at": base + i * 60} for i in range(800)]
n = store.write_l3_bulk(batch, expires_at=exp)
check("写入条数正确", n == 800, str(n))
got = store.query_l3(subject="用户甲", limit=1000)
check("查回条数一致", len(got) == 800, str(len(got)))
first = min(got, key=lambda x: x["created_at"])
check("时间戳保留（不是导入时刻）", abs(first["created_at"] - base) < 1, str(first["created_at"]))
check("内容完整", first["content"] == "第0条", first["content"])
check("session 保留", first["session_id"] == "s1")
check("expires_at 按导入时刻算", abs(first["expires_at"] - exp) < 2)

n2 = store.write_l3_bulk([], expires_at=exp)
check("空列表返回 0", n2 == 0)

# ---------- 5. 补录提炼：范围与按天水位 ----------
print("")
print("===== distill by day =====")
db2 = os.path.join(tempfile.gettempdir(), "sugarmem_distill_test.db")
if os.path.exists(db2):
    os.remove(db2)
s2 = st.SugarStore(db2)
now2 = time.time()
y2 = now2 - 86400          # 昨天
rows2 = []
# 实时消息：session 是真实会话，不该被补录提炼捞走（否则等于重复烧 token）
rows2 += [{"event_type": "user_message", "subject": "鱼", "content": f"[私] 鱼: 实时{i}",
           "session_id": "default:FriendMessage:1", "created_at": now2 + i} for i in range(5)]
# 补录：新批次 import: 前缀 + 老数据空 session，两种都要能捞
rows2 += [{"event_type": "user_message", "subject": "鱼", "content": f"[私] 鱼: 昨天{i}",
           "session_id": "import:鱼:1", "created_at": y2 + i} for i in range(4)]
rows2 += [{"event_type": "user_message", "subject": "鱼", "content": f"[私] 鱼: 今天{i}",
           "session_id": "import:鱼:2", "created_at": now2 + 100 + i} for i in range(4)]
rows2 += [{"event_type": "user_message", "subject": "鱼", "content": f"[私] 鱼: 老补录{i}",
           "session_id": "", "created_at": now2 + 200 + i} for i in range(2)]
# AI 回复：跟人同一 subject，要一起进窗口
rows2 += [{"event_type": "bot_reply", "subject": "鱼", "content": "[你] 嗯",
           "session_id": "import:鱼:2", "created_at": now2 + 300 + i} for i in range(2)]
# 别人的补录：不该混进来
rows2 += [{"event_type": "user_message", "subject": "阿明", "content": "[私] 阿明: 你好",
           "session_id": "import:阿明:1", "created_at": now2}]
s2.write_l3_bulk(rows2, expires_at=now2 + 86400)

check("只捞补录，不碰实时会话", s2.count_l3_for_distill("鱼") == 12,
      str(s2.count_l3_for_distill("鱼")))
prog2 = s2.distill_progress("鱼")
check("归成 2 天", prog2["days"] == 2, str(prog2["days"]))
check("待提炼 12 条", prog2["left"] == 12, str(prog2["left"]))
check("最近的一天打头", prog2["current_day"] == s2.distill_days("鱼")[0]["day"])

c1 = s2.distill_chunk("鱼", block=100)
check("先给最近的今天", "今天0" in c1["rows"][0]["content"], c1["rows"][0]["content"])
check("今天这批 8 条一起给", len(c1["rows"]) == 8, str(len(c1["rows"])))
check("实时消息不混入",
      all(r["session_id"] == "" or str(r["session_id"]).startswith("import:")
          for r in c1["rows"]))
check("不含别人的记录", all(r["subject"] == "鱼" for r in c1["rows"]))
check("AI 回复也在段里", sum(1 for r in c1["rows"] if r["event_type"] == "bot_reply") == 2)
check("这天一段就够", c1["more_in_day"] is False)

s2.distill_mark("鱼", c1["day"], int(c1["rows"][-1]["id"]), len(c1["rows"]))
s2.distill_tally("鱼", items=len(c1["rows"]))
prog2 = s2.distill_progress("鱼")
check("今天标完成", prog2["done_days"] == 1, str(prog2["done_days"]))
check("只剩昨天的 4 条", prog2["left"] == 4, str(prog2["left"]))

c2 = s2.distill_chunk("鱼", block=100)
check("接着给昨天", c2 is not None and "昨天0" in c2["rows"][0]["content"],
      c2["rows"][0]["content"] if c2 else "None")
check("昨天 4 条", len(c2["rows"]) == 4, str(len(c2["rows"])))
s2.distill_mark("鱼", c2["day"], int(c2["rows"][-1]["id"]), len(c2["rows"]))
s2.distill_tally("鱼", items=len(c2["rows"]))
prog2 = s2.distill_progress("鱼")
check("两天都完成", prog2["done_days"] == 2 and prog2["left"] == 0,
      f'{prog2["done_days"]}/{prog2["left"]}')
check("没有待提炼时返回空", s2.distill_chunk("鱼", block=100) is None)
cur2 = s2.distill_cursor("鱼")
check("累计条数 12", cur2["items"] == 12, str(cur2["items"]))
check("累计段数 2", cur2["blocks"] == 2, str(cur2["blocks"]))
check("累计表不掺进全局水位", cur2["last_id"] == 0, str(cur2["last_id"]))
s2.close()

store.close() if hasattr(store, "close") else None
try:
    os.remove(db_path)
except OSError:
    pass
try:
    os.remove(db2)
except OSError:
    pass

print(f"\n===== 结果：{len(PASS)} 过 / {len(FAIL)} 挂 =====")
if FAIL:
    print("失败项：", FAIL)
    sys.exit(1)
