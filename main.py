import asyncio
import hashlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from quart import Response, send_file

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # noqa
    get_astrbot_data_path = None

PLUGIN_NAME = "astrbot_plugin_sugarmemory"
# qq_tools 定时唤醒伪造的消息正文前缀。这类消息的 sender_id 是真人 QQ，
# 只有昵称和正文能认出来，所以两边都要判。
PSEUDO_SENDER_PREFIX = "[系统唤醒]"
PANEL_PREFIX = f"{PLUGIN_NAME}/panel"
API_PREFIX = f"{PLUGIN_NAME}/api"

# 补录提炼：一天算一个自然分块。一天之内条数特别多时再切段，
# 条数和字符各卡一道 —— 只卡条数的话，一天几十条长消息照样能撑爆上下文。
DISTILL_BLOCK = 120
DISTILL_CHUNK_CHARS = 5500


def _read_plugin_version(default: str = "0.0.0") -> str:
    """从同目录 metadata.yaml 读版本号。

    @register 的版本串以前是手写字面量，改 metadata.yaml 时一漏改，面板和
    插件列表就一直显示旧版本（「明明升到 0.16.x 还显示 0.15.4」就是这么来的）。
    这里统一以 metadata.yaml 为准，只有读不到时才回落默认值。
    """
    try:
        text = Path(__file__).with_name("metadata.yaml").read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("version:"):
                v = s.split(":", 1)[1].strip().strip("'\"")
                if v:
                    return v
    except Exception:  # noqa: BLE001
        pass
    return default


PLUGIN_VERSION = _read_plugin_version()

def _is_noise_reply(query: str) -> bool:
    """这一句是不是纯应答/寒暄（没有可检索的话题内容）。

    判据是「把应答语素全拆掉后几乎不剩实词」，而不是简单的开头匹配：
    「好的，那我先睡了」拆完还剩「那我先睡了」，必须照常检索；
    「ok我明白了」「可以的」「哈喽」拆完是空的，才按闲聊处理。
    带疑问尾巴的（「可以吗」「然后呢」）一律不算应答——那是真在问。

    常量写在函数体内是刻意的：离线回归测试用 AST 抽取单个函数来跑，
    依赖模块级常量会让那些测试在抽取时直接 NameError。
    """
    tokens = (
        "okay", "ok", "我明白了", "我懂了", "我知道了", "明白了", "知道了", "懂了",
        "了解", "收到", "没问题", "没事", "辛苦了", "辛苦", "谢谢", "谢了",
        "可以", "好的", "好滴", "好吧", "行", "嗯", "哦", "啊", "哈",
        "嘿", "对", "是", "在吗", "在么", "没有",
    )
    s = re.sub(r"[\s。，、！!？?~～…\.\,；;：:'\"“”‘’]+", "",
               str(query or "")).lower()
    if not s or len(s) > 10:
        return False
    if s.endswith(("吗", "呢", "么", "啥", "什么", "不")):
        return False
    rest = s
    for w in sorted(tokens, key=len, reverse=True):
        rest = rest.replace(w, "")
    return len(rest) <= 1


def _idem_key(*parts: str) -> str:
    """幂等指纹：参数顺序固定、大小写与空白归一，同一件事永远同一个 key。"""
    raw = "|".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def outbound_history_lines(rows: list, kind: str = "all", hours: float = 0,
                           limit: int = 10, nick_map: dict | None = None,
                           now: float | None = None) -> list[str]:
    """outbound_history 工具的正文：按新→旧列出转述/定时外发记录。

    独立成模块函数是为了能脱离 AstrBot 运行时直接测（longterm_sim）——
    工具方法只做参数校验和取数，格式化全在这里。
    """
    now = now or time.time()
    nick_map = nick_map or {}
    kind = str(kind or "all").strip().lower()
    if kind not in ("all", "relay", "cron"):
        kind = "all"
    window = max(0.0, float(hours or 0)) * 3600.0
    limit = max(1, min(int(limit or 10), 20))

    def _fmt_ts(ts) -> str:
        try:
            return time.strftime("%m-%d %H:%M", time.localtime(float(ts or 0)))
        except (TypeError, ValueError, OSError):
            return "?"
    keep: list[tuple[str, dict]] = []
    for r in rows:  # rows 按新→旧
        c = str(r.get("content") or "")
        k = _outbound_kind(c)
        if kind != "all" and k != kind:
            continue
        ts = float(r.get("created_at") or 0)
        if window and now - ts > window:
            continue
        nick = str(nick_map.get(str(r.get("session_id") or ""), "") or "").strip()
        if nick:
            # 冷号码「用户 123456003」AI 认不出是谁，补上昵称
            c = re.sub(r"(用户\s?\d{5,12})(?!\()", rf"\g<1>（{nick}）", c, count=1)
        tag = "转述" if k == "relay" else ("定时" if k == "cron" else "外发")
        keep.append((f"[{tag}|{_fmt_ts(ts)}] {c}", r))
        if len(keep) >= limit:
            break
    if not keep:
        span = "全部历史" if not window else f"最近 {int(max(0.0, float(hours or 0)))} 小时"
        what = {"relay": "转述", "cron": "定时任务"}.get(kind, "外发")
        return [f"{span}内没有任何{what}外发记录。"
                f"如实告诉用户这个时间段没有记录即可，不要猜测或编造。"]
    lines = [t for t, _ in keep]
    if kind == "all":
        # 混排时转述和定时分段，模型才不会把闹钟内容说成「我替你带的话」
        relay = [t for t, _ in keep if "[转述|" in t]
        cron = [t for t, _ in keep if "[定时|" in t]
        other = [t for t, _ in keep if "[外发|" in t]
        out: list[str] = []
        if relay:
            out.append("—— 你替人带的话（转述）——")
            out.extend(relay)
        if cron:
            out.append("—— 你按定时任务发的话 ——")
            out.extend(cron)
        if other:
            out.append("—— 其他外发 ——")
            out.extend(other)
        return out
    return lines


# 配置分组：schema key、显示名、这一组是干什么的。
# AstrBot 官方配置页拿它当大分类（type=object），组说明显示在组标题下面。
CFG_GROUPS = [
    ("memory", "记忆",
     "记忆怎么写、留多久。这是整个插件的地基，后面所有功能都建立在这些数据上。"),
    ("inject", "注入",
     "每轮对话往 AI 提示词里塞什么、塞多少。改动这一组会直接影响回复质量和 token 花费。"),
    ("summary", "总结",
     "什么时候把聊天压成摘要和长期事实。总结会真的调用模型，所以这里有频率、超时和熔断。"),
    ("affinity", "好感度",
     "分数怎么涨、阶段怎么分。阶段名同时决定 AI 用什么语气跟你说话。"),
    ("relation", "人际关系",
     "人和人关系网的全部配置都在这一组：总开关、注入快照、AI 的读写工具和权限。"
     "总开关默认关——关掉后对话不注入关系、记忆包里不带关系、AI 也不能读写关系，"
     "面板上会标「当前此功能为关闭」。不需要这个功能就让它一直关着，一点 token 都不花。"),
    ("usage", "用量",
     "token 记账开关。只影响统计，不影响任何功能。"),
    ("tool", "给 AI 的工具",
     "工具本身的行为：挂不挂载、群里能不能写、写之前校验什么。"),
]

# 字段表：key、短名、类型、分组 key、详细说明。
# 短名要短到能在一行里放下（官方配置页拿它当标题），所有解释都写在最后一个字段。
# 说明一律写成人话：先说这个东西干什么，再说开了/关了会怎样，少用术语。
CFG_ITEMS = [
    # ---- 记忆 ----
    ("record_enabled", "记录聊天", "bool", "memory",
     "把你说的和 AI 回的存成短期记录。关掉之后什么都不记，已有的不受影响"),
    ("record_in_group", "群聊写记忆", "bool", "memory",
     "【强烈不建议开启】本插件没有做群聊方面的适配，也不是为群聊功能服务的。"
     "默认就是群聊只读：记忆只给 AI 看，不写入、不加好感度、也不计入总结条数。"
     "强行打开会遇到说话人归属错乱、记忆互相串台、额度被群聊白白吃掉"),
    ("ignore_senders", "不记这些发送者", "list", "memory",
     "按昵称排除伪消息。qq_tools 的定时唤醒会伪装成「系统唤醒」发一条消息，"
     "sender_id 还是真人 QQ，不排除的话会被当成用户亲口说的话记进记忆和好感度"),
    ("l3_retention_days", "聊天保留天数", "int", "memory",
     "聊天原文存多少天后自动删除。0 = 永久保留（数据库会一直变大）"),
    ("l3_skeleton_enabled", "原文过期压成骨架", "bool", "memory",
     "开：聊天原文到期后不直接删，先压成一行主题词骨架（如「天气/通勤/咖啡」）"
     "再删，骨架自己再保留一阵——「从关键句忘到只剩关键词」。关：到期直接删（原行为）"),
    ("skeleton_retention_days", "骨架保留天数", "int", "memory",
     "骨架（原文压缩后的主题词索引）再保留多少天后彻底删除。只在上一项开启时生效"),
    ("decay_stagger_enabled", "错峰沉底", "bool", "memory",
     "默认关。同批入库的记忆过期时间几乎一样，会在同一天集体沉底——一沉几十条。"
     "打开后每条按自己的编号拿一个固定偏移，在两三个月里陆续沉，面板上看到的"
     "是「一直在慢慢忘」，而不是「半年后清一次仓」。只改什么时候沉，不改沉不沉，"
     "对注入内容没有影响；沉底是单向的（沉了就不再被检索到），"
     "所以真正该调的是保留度参数，这个只是观感开关"),
    ("decay_stagger_days", "错峰跨度（天）", "int", "memory",
     "相位偏移的最大天数，默认 75。设 0 等于关掉错峰。跟保留度曲线叠加："
     "原本 150 天沉底的条目，会落在 150~225 天之间"),
    ("record_outbound", "记录主动发出去的话", "bool", "memory",
     "AI 通过转述/跨会话发送发给别人的话，也存进记忆。关掉的话，换个会话被问"
     "「你刚才跟谁说了什么」，它会查不到记录，然后把自己真发过的话当成编的否认掉"),
    ("outbound_cause_window", "找起因的时间范围（秒）", "int", "memory",
     "记主动发送时，往回找这么久内「谁让它发的、原话怎么说的」，一起钉在记录上。"
     "隔几天别人问「你为什么发这个」，靠的就是这段。0 = 不记起因"),
    ("record_outbound_other", "其他主动发送也要记", "bool", "memory",
     "转述工具发的每一条都带标记、一定记。这项管的是**没带标记**的主动发送——"
     "定时推送、插件自发通知、其他插件的带话。开着=照旧记成「你主动发给xxx」；"
     "关掉=只记真正的转述，怕定时消息污染记忆就关"),
    ("outbound_dedup_seconds", "本会话外发去重（秒）", "int", "memory",
     "AI 往「刚刚还在聊天的会话」发的不带标记消息（分段回复、send 工具的碎碎念）"
     "不再记成「主动发送」——那些话就在当前对话里，再记一遍会把转述通道堵死，"
     "真转述反而被挤丢。带标记的转述/定时任务永远照记。0 = 关闭去重，全部照记"),
    ("bot_isolation", "多机器人记忆隔离", "bool", "memory",
     "一台 AstrBot 跑多个机器人账号（多个平台实例，各自一个 QQ 号）时，记忆按机器人"
     "分开：A 的定时任务、转述和对话不会被 B 的记忆包读到，B 也查不到 A 发过什么。"
     "以前外发记录是全局的，任何一个插件调 context.send_message 都会被记成「你说的」，"
     "于是没装本插件的工具机器人发的定时消息也会串进主机器人的主动记忆。"
     "关掉 = 所有账号共用一套记忆（老行为），只在「多个账号其实是同一个人格」时才关"),

    # ---- 注入 ----
    ("inject_enabled", "注入记忆", "bool", "inject",
     "开着才会把记忆塞进 AI 的 prompt。**默认开**：测试期结束，装好即用，不用手动开开关。"
     "关掉是影子模式：照常记录，但 AI 看不见——记忆包和关系快照一并停（人际关系网开关"
     "单独开着也不会注入，只保留 AI 记/查关系的工具）。只有两种情况需要关：想先观察几天"
     "记录得准不准，或者还装了别的记忆类插件、怕同样的内容注入两遍"),
    ("inject_scope_subject", "记忆包按人过滤", "bool", "inject",
     "只注入当前说话人的长期事实和本会话的摘要。关掉＝全库检索，别人说过的事会串到他头上"),
    ("inject_outbound_relay_limit", "带几条转述过的话", "int", "inject",
     "「你替人带给谁的话」最多带几条（发到当前会话的）。转述和定时消息在记忆包"
     "里是分开的两段，条数也分开算——混在一起时 AI 会把闹钟提醒说成"
     "「我替你带的话」。建议 3（转述是最常被追问的），0 = 不带转述"),
    ("inject_outbound_relay_window_hours", "转述新鲜窗（小时）", "int", "inject",
     "只有这个时间内的转述才常驻进记忆包，过窗的自动让位——没人会一直追问"
     "三天前带的什么话，老转述却因为「最近没新的」一直占坑还优先保留，包越滚"
     "越臃肿。第二天有新转述就照旧进，每天都一样。被追问时仍能补召回（不受此"
     "窗限制，问了才带）。建议 24，0 = 不过期（一直带）"),
    ("inject_outbound_cron_limit", "带几条定时任务发的话", "int", "inject",
     "「你按定时任务发出去的话」最多带几条（发到当前会话的）。这类消息是你"
     "自己写的提醒原文，不是替谁带的话，所以单独一段、单独配额。"
     "建议 2，0 = 不带定时消息"),
    ("inject_outbound_cron_window_hours", "定时新鲜窗（小时）", "int", "inject",
     "和转述新鲜窗同理，但管的是定时任务池——两个池子独立计时、互不影响。"
     "过窗的定时消息不再常驻进包，追问到仍可召回。建议 24，0 = 不过期"),
    ("inject_outbound_away_limit", "被追问时补带几条发到别的会话的转述", "int", "inject",
     "在 A 会话让 AI 给 B 带完话、回 A 问「你刚才跟 B 说了什么」时，那条记录"
     "记在 B 名下，默认进不了记忆包，AI 只能回「我凭记忆」。这一项就是给这种"
     "追问开的补位：只有本轮这句话确实在问（说到转述/带话，或点名了对方）才补，"
     "平时一条都不带，几乎不占 token。建议 1，0 = 追问也不补"),
    ("inject_outbound_away_cron_limit", "被追问时补带几条发到别的会话的定时消息", "int", "inject",
     "和上一项同理，但管的是定时任务。只有本轮明确在问定时/提醒/闹钟/那个任务"
     "时才补——问「你给 B 带了什么话」不会把闹钟内容塞进来，免得 AI 张冠李戴。"
     "建议 1，0 = 追问也不补"),
    ("inject_outbound_labeled_only", "外发段只带定时任务/转述", "bool", "inject",
     "开着：没触发定时任务或转述时，「你发出去的话」整段不进记忆包，把预算让给"
     "摘要和事实；一旦有定时任务/转述，它们会顶到记忆包最前面。关掉：连「你主动"
     "发给xxx」这类普通回声也一起进包（信息量低，还会挤预算）"),
    ("inject_outbound_pinned", "定时任务/转述不参与裁剪", "bool", "inject",
     "记忆包超预算时的裁剪顺序里，发给当前会话的定时任务/转述排在最后一道且默认"
     "不许砍——它就是「你刚才定时发了什么」的答案本身，砍了 AI 又会把真发过的"
     "东西当幻觉否认。关掉=让它跟其他内容一样正常参与裁剪（预算很紧时可以关）"),
    ("inject_cross_limit", "带几条其他会话的近况", "int", "inject",
     "顺便带几条别的会话里最近发生的事，按本轮话题筛过才给，会标出是谁说的。"
     "填 0 = 不带（只认自己这个会话）。开太大容易串台"),
    ("inject_cross_include_self", "跨会话也带 AI 自己说的话", "bool", "inject",
     "其他会话近况里，除了别人说的，要不要把 AI 自己在别的会话说过的话也带进来。"
     "AI 一句往往比人多、信息量少，默认关；需要「我记得我在别的会话说过啥」时再开"),
    ("inject_mode", "记忆包注入架构", "str", "inject",
     "填 frozen 或 dynamic。frozen＝按（人,会话）冻结一份常驻包吃满前缀缓存（默认，A 方案）；"
     "dynamic＝回到旧版按本轮原话实时检索，每轮都变、前缀缓存基本不命中。"
     "dynamic 只用于 A/B 对照或紧急回退，日常请保持 frozen。"
     "切换这一项会让所有活跃分片各重建一次，属预期行为；用量表里用 arch_ver 区分两批数据"),
    ("inject_dedup", "冻结常驻记忆包", "bool", "inject",
     "冻结的总闸，不是架构开关（架构由上方 inject_mode 决定）。开启后：这一轮算出来的"
     "块跟上一次挂出去的一模一样，就原样复用，一个字节都不重算；不一样才重建。"
     "关掉＝每轮无条件重算 query=\"\" 的常驻包，等于放弃冻结，只用于排查缓存问题。"
     "建议保持开启"),
    ("inject_budget_tokens", "注入预算", "int", "inject",
     "这是上限、不是目标：常驻包实际多大由记忆库和各项配额决定（本机实测最厚约 900 tok）。"
     "调大不会多花一分钱，只是给记忆增长留余量；调到低于实际体积就会触发裁剪，"
     "而裁剪最后会「突破保底」把长期事实砍到 1 条，反而毁掉召回。"
     "默认 1200 够用；若把 pack_l2_limit / 转述配额调大，建议同步提到 1400~1600"),
    ("pack_l3_snippets", "附带聊天原文", "int", "inject",
     "记忆包里最多带几条聊天原文当参考"),
    ("pack_l2_limit", "长期事实上限", "int", "inject",
     "记忆包里最多带几条从以往对话里提炼出来的事实"),
    ("inject_l2_resident_limit", "常驻事实条数", "int", "inject",
     "每一轮都进 system 的长期事实只放几条，剩下的改成「聊到再召回」。"
     "0 ＝ 不额外限制（等于「长期事实上限」）。设成 6 就是常驻只带 6 条身份/习惯，"
     "「开发过什么插件」这类经历退到按话题召回，闲聊时不再白占 token。"
     "改动会让前缀重算一次，之后照旧稳定"),
    ("pack_l2_keep_min", "长期事实保底条数", "int", "inject",
     "预算不够时，再挤也要给事实留几条。留 3 条＝至少保住「这个人是谁」的基本盘"),
    ("pack_l3_keep_min", "聊天原文保底条数", "int", "inject",
     "预算不够时，再挤也要给聊天原文留几条。原文是「他当时原话怎么说的」的唯一来源，"
     "砍了补不回来。留 0 ＝允许全砍（旧行为）"),
    ("pack_l2_rel_ratio", "相关性门槛（0~100）", "int", "inject",
     "本轮消息没提到的事实，命中率低于最高命中的这个百分比就不注入。"
     "30 ＝ 命中数不到头名的三成直接丢掉，防止一句「在吗」把十几条无关事实全拽进来。0 ＝ 关闭"),
    ("pack_query_expand", "问法归一（改写补召回）", "bool", "inject",
     "把「多大了」「搞什么创作」这类口语问法映射成概念词再检索，补上字面"
     "匹配抓不住改写的盲区（实测改写召回 20% → 90%+）。关掉回到纯字面匹配"),
    ("inject_skeleton_limit", "带几条旧事骨架", "int", "inject",
     "记忆包常驻几条「旧事骨架」（原文过期后压缩的主题词索引，一行约 30 token）。"
     "用户问「咱们以前聊过啥」时 AI 至少有个索引可查。0 = 不注入，骨架只在库里躺着"),
    ("pack_semantic_fallback", "语义兜底", "bool", "inject",
     "字面/概念/问法别名全部零命中时，按语义相似度挑保底事实，兜「打字很响的设备」"
     "→「机械键盘」这类完全不同字的改写。**必须同时填下面的 semantic_provider_id 才会通电**："
     "留空或者 Provider 不支持 embedding 时整条路静默跳过，零调用零开销，等于没开"),
    ("semantic_provider_id", "语义兜底专用 Provider ID", "str", "inject",
     "留空 = 整条语义路不通电，一点钱都不花。填上支持 embedding 的 Provider ID，"
     "并打开 pack_semantic_fallback，才会在前两层全零命中时调向量接口。"
     "异步 / 同步接口都认，超时或报错会自动退回字面路结果，不会拖慢对话"),
    ("semantic_timeout_ms", "语义兜底超时（毫秒）", "int", "inject",
     "向量接口最多等多久。默认 1500；超时、报错、Provider 不存在都静默降级，"
     "本轮照常用字面/概念/别名路的结果，绝不让记忆检索卡住聊天"),
    ("inject_topic_refresh", "话题命中时刷新常驻包", "bool", "inject",
     "B 方案：常驻包继续冻结吃满前缀缓存，只有本轮真的聊到某条包外相关记忆，"
     "或出现「那它呢 / 上次 / 刚才」这类指代追问时，才重建这一轮把答案补进去。"
     "闲聊轮次一个字节都不动，命中率不受影响；关掉＝回到纯冻结（旧行为）"),
    ("inject_topic_extra", "话题刷新最多补几条", "int", "inject",
     "话题触发那一轮最多补几条包外高相关长期事实。用「替换最弱常驻条」的方式补，"
     "不把包撑大、不额外烧预算。建议 3，0 = 只刷新不补事实"),
    ("pack_context_rounds", "最近几轮一起参与检索", "int", "inject",
     "「那它呢」「刚才那个」本轮字面零命中，把最近几轮用户原话按 0.4 权重一起参与"
     "检索就能接上。只给检索用，不进问法判断、不进注入块渲染。建议 3，0 = 关"),
    ("pack_l2_candidate_limit", "长期事实候选池上限", "int", "inject",
     "检索时先按（人 / 状态 / 有效期）过滤出这么多条候选，再按本轮相关度挑，"
     "不再先按置信度截断——低置信但本轮正好相关的事实才有机会被想起。"
     "建议 200；库特别大时可调小省 CPU，调太小会重新出现漏召回"),

    # ---- 总结 ----
    ("summarize_enabled", "启用阶段总结", "bool", "summary",
     "攒够一定条数就让 AI 回头看一遍：写一段摘要，顺便挑出值得长期记住的事"),
    ("summary_interval_messages", "攒多少条触发", "int", "summary",
     "同一个会话攒够多少条你说的话触发一次。AI 的回复不计数"),
    ("summary_idle_seconds", "静默等待（秒）", "int", "summary",
     "攒够之后再等这么久，确认这轮聊完了才开始跑，免得跟你正在进行的对话抢模型"),
    ("summary_window_user", "回顾用户消息数", "int", "summary",
     "总结时往回取最近多少条用户消息"),
    ("summary_window_bot", "回顾 AI 回复数", "int", "summary",
     "总结时往回取最近多少条 AI 回复。0 = 不带回复"),
    ("summary_related_l2", "总结时带上已有记忆", "bool", "summary",
     "把这个人已经记住的长期事实（每条带 #编号）一起喂给总结模型，让它能判断"
     "这次说的话是不是在推翻旧内容（改口）。不带的话总结是闭卷答题，只会不断新增，"
     "「喜欢摇滚」和「现在只听说唱」会一起躺在库里"),
    ("summary_related_limit", "最多带多少条", "int", "summary",
     "带太多会稀释注意力也费 token；每人十几条时直接全带。0 = 一条都不带"),
    ("summary_timeout_seconds", "单次最长等待（秒）", "int", "summary",
     "一次总结最多等多久。超时会自动重试 3 次（间隔 2 秒、4 秒）"),
    ("summary_provider_id", "总结专用模型", "str", "summary",
     "留空＝跟当前对话用同一个模型。填 Provider ID 可以挂个便宜的小模型省钱"),
    ("alias_autogen", "自动生成问法别名", "bool", "summary",
     "每次总结完成后，让模型给新记下的长期事实顺手生成几个「用户可能怎么问起"
     "它」的说法存进库里。检索时问法跟别名能对上字面，改个说法也召回得回来"
     "（「多大了」→「16岁」这种）。存量旧事实会在下次总结后分批补齐"),
    ("summary_reasoning_effort", "总结思考强度", "str", "summary",
     "留空＝不干预，由模型自己决定。可选 low / medium / high / max。"
     "推理型模型默认能「想」三万多字，实测同一份日志："
     "high 要 190 秒、low 只要 7 秒，摘要质量几乎一样。"
     "只作用于总结这一次请求，不影响你聊天"),
    ("summary_prompt_extra", "总结提示词追加内容", "str", "summary",
     "只追加、不覆盖：内置提示词一个字都不改（面板里只读预览），这段会接在它末尾"
     "一起发给总结模型。适合写个人偏好，比如「游戏名一律保留原名，不要翻译」"
     "「我的事只记事实，不要记情绪」。留空＝只用内置提示词。"
     "完整预览在「设置 → 总结提示词」。改这里不会动注入缓存"),
    ("summary_circuit_threshold", "连续失败几次就暂停", "int", "summary",
     "总结连续失败这么多次就暂时不自动跑了，避免一直烧钱。面板上手动点不受影响"),
    ("summary_circuit_cooldown", "暂停多久（秒）", "int", "summary",
     "暂停多久之后自动试一次，成功就恢复正常"),

    # ---- 好感度 ----
    ("affinity_enabled", "启用好感度", "bool", "affinity",
     "你每说一条消息就涨一点。越接近上限涨得越慢，到顶之后基本不涨"),
    ("affinity_cap", "好感度上限", "int", "affinity",
     "涨到这个数就不再明显增长，之后每条只 +1"),
    ("affinity_tone", "好感度影响语气", "bool", "affinity",
     "按当前阶段给 AI 一句语气提示，让它回复的亲近程度跟着好感度走"),

    # ---- 用量 ----
    ("token_stats_enabled", "记录 Token 用量", "bool", "usage",
     "记录对话、总结、注入各花了多少 token。关掉之后面板「Token 用量」就没有数据"),

    # ---- 人际关系：关系网全部配置归这一组 ----
    ("inject_relations", "人际关系网总开关", "bool", "relation",
     "整个关系网系统的总开关。关（默认）＝三件事全停：对话不注入关系快照、"
     "记忆包里不带关系、AI 的记/查关系工具一并禁用，一点 token 都不花；"
     "面板「人际关系」页会显示「当前此功能为关闭」。开＝（注入总闸也开着时）"
     "每轮对话附一小段「你和这个人是什么关系」（几十 token，不用 AI 调工具）"
     "+ AI 可以从对话里记关系"),
    ("inject_relation_limit", "关系快照最多几条", "int", "relation",
     "这段快照最多列几条关系。列得越多越占 token"),
    ("tool_relation_write", "允许 AI 记关系", "bool", "relation",
     "让 AI 能调用「记关系」，从对话里自己抓到「A 是 B 的某某」并写下来。"
     "总开关关着时这项不生效"),
    ("tool_relation_query", "允许 AI 查关系", "bool", "relation",
     "让 AI 能调用「查关系」，回答之前先查一下谁跟谁是什么关系。"
     "总开关关着时这项不生效"),
    ("tool_write_in_group", "群里允许改关系", "bool", "relation",
     "默认群里只能查、不能改。开之前想清楚：群里任何人说话都可能触发写入"),
    ("tool_relation_quote_check", "只认本轮原话", "bool", "relation",
     "AI 想记的两个人，至少有一个要出现在你这条消息里。防止它被记忆包、别人的转述、贴进来的网页骗去乱写"),
    ("tool_idem_seconds", "防重复写入（秒）", "int", "relation",
     "同一条关系在这个时间内重复提交只算一次，防 AI 重试时反复写。0 = 关闭"),
    ("tool_relation_admin_ids", "额外授权名单", "str", "relation",
     "填 QQ 号 / sender_id，多个用逗号或空格隔开。名单里的人当主人看待。"
     "注意：AstrBot 管理员本身就算主人，不用填；你自己不是管理员又想用下面那些限制，就把自己填进来"),
    ("tool_relation_write_admin_only", "改关系只限主人", "bool", "relation",
     "开启后只有主人能通过 AI 改关系，普通用户明说也不写。默认关（谁都能让 AI 记，但受下面「只能记跟自己有关」约束）"),
    ("tool_relation_query_admin_only", "查关系只限主人", "bool", "relation",
     "开启后只有主人能通过 AI 查关系。默认关，因为查是只读的，一般不需要拦"),
    ("tool_relation_self_only", "只能记跟自己有关的关系", "bool", "relation",
     "开启后非主人只能记「他自己」跟别人的关系，碰不到两个第三人之间的关系。"
     "防的是：群里随便谁一句「张三和李四是情侣」，AI 就替他把别人的关系改了。默认开"),

    # ---- 人物身份：QQ 是身份锚，称呼分四档 ----
    ("alias_cleanup_enabled", "称呼自带遗忘", "bool", "relation",
     "默认开。人的称呼分四档：显示名、能被称呼的名、绰号外号、叫错的名字。"
     "只有后两档会被忘掉，前两档永不自动删（那是这个人的身份）。"
     "关掉的话所有称呼都永久留着，面板「人际关系」里会越堆越多"),
    ("alias_guess_retention_days", "叫错的名字留多久（天）", "int", "relation",
     "默认 90。AI 从对话里听岔、写错的名字归这一档，这么多天没再出现就自动删掉。"
     "只删称呼映射，人物和记忆一条不动"),
    ("alias_nick_retention_days", "外号留多久（天）", "int", "relation",
     "默认 365。绰号外号这么久没被叫过、而且从头到尾只叫过一次，才会被清。"
     "叫过两次以上的一直留着"),

    # ---- 给 AI 的工具 ----
    ("tool_memory_search", "允许 AI 查记忆", "bool", "tool",
     "给 AI 一个「翻自己的记忆」的工具（memory_search）。注入包是被动的、按预算裁过，"
     "用户问「你刚才发什么」「我们之前聊过什么」时包里未必有——有这个工具 AI 就能"
     "主动去库里翻，而不是两手一摊说没印象。只读，不写任何东西"),
    ("tool_outbound_history", "允许 AI 翻外发历史", "bool", "tool",
     "给 AI 一个「翻自己发出去的话」的工具（outbound_history）：按时间列出转述/"
     "定时任务的历史记录，可按类型和时间范围筛。记忆包只带新鲜窗内的转述/定时，"
     "更早的只有这个工具能完整翻到。只读，不写任何东西"),
]
CFG_KEYS = {k for k, *_ in CFG_ITEMS}
CFG_GROUP_OF = {k: g for k, _l, _t, g, _h in CFG_ITEMS}
# 这几项是长文本，面板要渲染成多行输入框（单行 input 里改一段提示词没法看）。
# 只影响前端控件形态，读写仍走 str，配置文件和 schema 都不用特殊处理。
CFG_MULTILINE = {"summary_prompt_extra"}

# 只有这些配置会改变 system_prompt 的字节。改它们时统一失效所有常驻包分片；
# 节流参数、用量统计、总结开关等不影响 prompt，改了不该动缓存。
# 注意 inject_dedup / inject_freeze_* 故意不在里面：它们只决定「内容变了
# 要不要立刻换」，不影响「当前该生成什么」。
PROMPT_CFG_KEYS = (
    "inject_mode", "inject_enabled", "inject_budget_tokens",
    "inject_scope_subject", "inject_relations", "inject_relation_limit",
    "affinity_tone",
    "pack_l3_snippets", "pack_l2_limit", "pack_l2_keep_min",
    "inject_l2_resident_limit",
    "pack_l3_keep_min", "pack_l2_rel_ratio", "pack_query_expand",
    "pack_semantic_fallback", "semantic_provider_id", "semantic_timeout_ms",
    "inject_topic_refresh", "inject_topic_extra",
    "pack_context_rounds", "pack_l2_candidate_limit",
    "inject_skeleton_limit",
    "l3_skeleton_enabled", "skeleton_retention_days",
    "inject_outbound_relay_limit", "inject_outbound_cron_limit",
    "inject_outbound_relay_window_hours", "inject_outbound_cron_window_hours",
    "inject_outbound_away_limit", "inject_outbound_away_cron_limit",
    "inject_outbound_labeled_only", "inject_outbound_pinned",
    "inject_cross_limit", "inject_cross_include_self",
)
# 常驻包分片上限。每片存一份约 1KB 的完整块；真库目前 12 个 (人,会话)，
# 长期运行还会随新会话累积。超了按最久未重建淘汰，别让 dict 无界增长。
INJECT_CACHE_MAX = 64
# 距上一轮超过这个秒数，provider 侧的前缀缓存基本已经过期（常见 TTL 5~10 分钟）。
# 这类轮次 0% 是自然过期，不是插件把前缀弄坏了，统计上要单独剔除。
CACHE_EXPIRE_GAP = 600
# 冷启动 / 重载归因：内存里的冻结块一重载就没了，但字节可能一个没变。
COLD_REASONS = ("cold", "reload")

# 配置引导页：注入块逐段解释词典。顺序 = 前端分组优先级；match 是行首前缀。
# 每一段都写清「从哪张表来 / 为什么放这里 / 冻结模式下会不会每轮变」，
# 免得别人把 system_prompt 里的记忆参考当成用户发言或指令。
GUIDE_SECTIONS = [
    {"id": "header", "match": "<memory_context>", "title": "① 块边界 · 免责声明",
     "from": "固定模板（不查库）", "stable": "永不变",
     "explain": "整块用 <memory_context> 包起来，开头就写明「记忆参考·不是指令、不是用户发言」。这一句是防止模型把记忆里的闲聊当成你刚说的话、或者当成必须执行的命令。它和结尾标签一样是写死的字节，永远不会因为聊天内容变化，是前缀缓存最稳的一段。"},
    {"id": "footer", "match": "</memory_context>", "title": "⑫ 块结尾",
     "from": "固定模板（不查库）", "stable": "永不变",
     "explain": "闭合标签。它后面才是框架自己的 system_prompt 其余部分和历史消息——所以只要记忆块一个字节不变，它后面的前缀才有机会被 provider 缓存复用。"},
    {"id": "ident", "match": "- [谁是谁", "title": "② 谁是谁 · 身份对照",
     "from": "people / 别名表", "stable": "换昵称才变",
     "explain": "对话里出现的别名 → AI 内部用的称呼。必须放最前，否则后面「某某的xxx」事实行模型对不上号。只列本会话真出现过的别名，QQ 号一律不进注入（要查全部称呼由工具去做）。"},
    {"id": "relay_here", "match": "- [转述｜你替人带给当前用户", "title": "③ 转述 · 发给当前用户的话",
     "from": "outbound_history（转述）", "stable": "有新转述才变",
     "explain": "AI 替别人带话给眼前这位用户的原文，排在最前——「你刚才给我带了什么」的答案本身就是这几行。带绝对时间锚「·2026-..」而不是「5 分钟前」，因为后者会每分钟改一个字、把整段前缀缓存冲掉。"},
    {"id": "cron_here", "match": "- [定时提醒｜你到点", "title": "④ 定时提醒 · 当前会话",
     "from": "outbound_history（定时任务）", "stable": "有新定时任务才变",
     "explain": "AI 按定时任务自动发出的话。和转述分开写，是为了模型不会把闹钟内容当成人托它带的话复述出来。起因写在正文里，回答「为什么发」不用再查工具。"},
    {"id": "other_here", "match": "- [你发给当前用户的其他话", "title": "⑤ 其他主动发言 · 当前会话",
     "from": "outbound_history（普通）", "stable": "有新记录才变",
     "explain": "分段回复、碎碎念这类没有信息量的主动发言。预算不够时，这一段（以及其他会话的无标签回声）是最先被裁掉的候选之一。"},
    {"id": "nickmap", "match": "称呼对照：", "title": "⑥ 称呼对照",
     "from": "L2 / cross 里的昵称映射", "stable": "仅点名 QQ 昵称时出现",
     "explain": "只有本轮原话里出现了某个 QQ 昵称时才写这一行，告诉模型「认人看昵称、说话叫 AI 称呼」。冻结包是按（人,会话）建的、与本轮原话无关，所以它基本不会出现在这里——这正是 A 方案的取舍：少一行对照，换整段前缀稳定。"},
    {"id": "summary", "match": "相关会话摘要：", "title": "⑦ 会话摘要",
     "from": "阶段总结（summaries 表）", "stable": "触发新总结才变",
     "explain": "每隔若干条消息把这段对话压成一段摘要。预算不够时会被逐步压短，但压到 60 字就停（再短就只剩废话，不如整段砍掉）。"},
    {"id": "l2", "match": "- [L2|", "title": "⑧ L2 · 长期事实",
     "from": "l2_facts（长期记忆）", "stable": "改写/新增才变",
     "explain": "「谁的什么 = 什么」这类从以往对话里提炼出来的结论，带置信度和出处编号。这是记忆包里最值钱的部分：上下文里没有、只能靠记忆补。时间锚取 created_at 而非 updated_at，因为后者会随稳度累加变化、平白废掉缓存。预算不足时 L2 退让到保底条数，砍完就补不回来。"},
    {"id": "relay_away", "match": "- [转述｜你替人带给其他会话", "title": "⑨ 转述 · 其他会话",
     "from": "outbound_history（跨会话）", "stable": "有新转述才变",
     "explain": "AI 在别的会话替人带过的话。跨会话回忆时，「我到底有没有说过」全靠这些凭据；不带标签的普通回声会被先裁掉，带标签的转述排得更靠后。"},
    {"id": "cron_away", "match": "- [定时提醒｜你按任务发到其他会话", "title": "⑨ 定时任务 · 其他会话",
     "from": "outbound_history（跨会话）", "stable": "有新定时任务才变",
     "explain": "别的会话里按定时任务发过的话。闹钟过期就没价值，所以比转述更早让位。"},
    {"id": "other_away", "match": "- [你发到其他会话的其他话", "title": "⑨ 其他发言 · 其他会话",
     "from": "outbound_history（跨会话）", "stable": "有新记录才变",
     "explain": "跨会话的普通主动发言，信息密度最低的一类外发记录，裁剪顺序里排在很前面。"},
    {"id": "cross", "match": "- [其他会话里的消息", "title": "⑩ 其他会话里的消息",
     "from": "l3_events（跨会话）", "stable": "有新消息才变",
     "explain": "别的会话里别人说过的话、以及 AI 自己在那边说过的话。开头带「你」的是 AI 自己说的，措辞分开是为了不让模型搞错立场。「这段可能是在说你」是归因标注，提示模型别把聊到当前用户的近况当成无关闲聊漏掉。"},
    {"id": "skeleton", "match": "- [旧事索引", "title": "⑪ 旧事索引",
     "from": "阶段总结的原话摘录", "stable": "触发新总结才变",
     "explain": "老话题的一行主题词索引，原话已经没了，只剩关键词。附一句「没写的细节不许脑补」，防止模型拿碎片编日期和数字。信息密度低，裁剪时可一直砍到 0 条。"},
    {"id": "l3", "match": "- [L3·", "title": "⑪ L3 · 只读事件（原话）",
     "from": "l3_events（当前会话）", "stable": "有新消息才变",
     "explain": "用户当时说过的原话。框架本身已经把当前会话原文放在上下文里，记忆包再带一遍属于双花 token，所以它是预算不足时第一个被裁的。"},
    {"id": "gated", "match": "- [另有 ", "title": "⑪ 门控筛除说明",
     "from": "写入审核（门控）", "stable": "有新筛除才变",
     "explain": "告诉模型「还有 N 条候选被筛掉了，是重复/临时状态，别提」。不写这句，模型容易向用户复述「似乎还有藏起来的内容」。"},
]

_VERSION_FRAGMENT = re.compile(r"^[\d.]+$")


def normalize_provider_label(value: str) -> str:
    """把 `5.3/glm-5.3-flash/glm-5.3-flash` 这类重复拼段收敛成 `glm-5.3-flash`。

    框架的 provider id 偶尔会把模型名截断的版本碎片和模型名拼在一起，还重复
    两段，分组卡看着像三个不同服务商。只丢「被更长段包含的纯数字版本碎片」
    和完全重复的段，不动正常的 `provider/model` 组合。
    """
    s = str(value or "").strip()
    if not s:
        return ""
    parts: list[str] = []
    for p in s.split("/"):
        p = p.strip()
        if p and p not in parts:
            parts.append(p)
    if not parts:
        return ""
    kept = [p for p in parts if not (
        _VERSION_FRAGMENT.fullmatch(p)
        and any(p != q and p in q for q in parts))]
    return "/".join(kept or parts)


CFG_GROUP_LABEL = {k: label for k, label, _h in CFG_GROUPS}
CFG_GROUP_HINT = {k: hint for k, _label, hint in CFG_GROUPS}
CFG_TYPE_OF = {k: t for k, _l, t, _g, _h in CFG_ITEMS}
CFG_ITEM_HINT = {k: h for k, _l, _t, _g, h in CFG_ITEMS}
CFG_DEFAULTS = {
    "record_enabled": True, "record_in_group": False, "ignore_senders": ["系统唤醒"],
    "l3_retention_days": 14, "inject_enabled": True,
    "record_outbound": True,
    "outbound_cause_window": 180,
    "record_outbound_other": True,
    "outbound_dedup_seconds": 120,
    "bot_isolation": True,
    "inject_relations": False, "inject_relation_limit": 6,
    "alias_cleanup_enabled": True,
    "alias_guess_retention_days": 90, "alias_nick_retention_days": 365,
    "inject_scope_subject": True,
    "inject_outbound_relay_limit": 3, "inject_outbound_cron_limit": 2,
    "inject_outbound_relay_window_hours": 24, "inject_outbound_cron_window_hours": 24,
    "inject_outbound_away_limit": 1, "inject_outbound_away_cron_limit": 1,
    "inject_outbound_labeled_only": True, "inject_outbound_pinned": True,
    "inject_cross_limit": 5,
    "inject_cross_include_self": False,
    "inject_mode": "frozen",
    "inject_dedup": True,
    "inject_budget_tokens": 1200, "pack_l3_snippets": 0, "pack_l2_limit": 12,
    "inject_l2_resident_limit": 0,
    "pack_l2_keep_min": 3, "pack_l3_keep_min": 0, "pack_l2_rel_ratio": 30,
    "pack_query_expand": True, "pack_semantic_fallback": False,
    "semantic_provider_id": "", "semantic_timeout_ms": 1500,
    "inject_topic_refresh": True, "inject_topic_extra": 3,
    "pack_context_rounds": 3, "pack_l2_candidate_limit": 200,
    "l3_skeleton_enabled": True, "skeleton_retention_days": 90,
    "inject_skeleton_limit": 2,
    "decay_stagger_enabled": False, "decay_stagger_days": 75,
    "alias_autogen": True,
    "summarize_enabled": True,
    "summary_interval_messages": 50, "summary_idle_seconds": 10,
    "summary_window_user": 60, "summary_window_bot": 30, "summary_timeout_seconds": 120,
    "summary_related_l2": True, "summary_related_limit": 16,
    "summary_provider_id": "", "summary_reasoning_effort": "",
    "summary_prompt_extra": "",
    "affinity_enabled": True, "affinity_cap": 1000,
    "summary_circuit_threshold": 3, "summary_circuit_cooldown": 600,
    "affinity_tone": True, "token_stats_enabled": True,
    "tool_relation_write": False, "tool_relation_query": False,
    "tool_memory_search": True, "tool_outbound_history": True,
    "tool_relation_write_admin_only": False,
    "tool_relation_query_admin_only": False,
    "tool_write_in_group": False,
    "tool_relation_self_only": True, "tool_relation_admin_ids": "",
    "tool_relation_quote_check": True,
    "tool_idem_seconds": 120,
}

# ---------- 联动插件 ----------
# 本插件的「转述 / 定时」记忆全部来自外部插件的写入：QQ 全套工具的
# relay_message 打标、sugarcron 的定时任务推送。没装对应插件时，相关配置
# 形同虚设——配额再大也永远等不到一条记录。所以：注入端把相关配额按 0
# 处理（等检测到插件恢复默认值），面板在配置旁挂提示，主页「联动」卡直观
# 展示两边状态。
#
# 2026-11-06 修正：以前这里把「QQ 全套工具」当成 astrbot_plugin_qq_tools，
# 但那是 YUMU1658 的 QQ 平台工具集（定时唤醒/群管理等），并不提供
# relay_message；真正打 _sugarmemory_relay 标记的是「蔗糖的QQ全套工具~」
# = astrbot_plugin_group_history_agent（融合了原 xrelay）。用户装了并启用了
# 全套工具、只是把 qq_tools 停用了，检测就报「未装」，转述配额还被 gating
# 压成 0。现在以真正的数据来源为主判据，qq_tools / xrelay 降为兼容别名，
# 并把「停用」和「未安装」分成两种状态，面板不再对着装好的插件说没装。
DEP_PLUGINS = {
    "qq_tools": {
        "module": "astrbot_plugin_group_history_agent",
        "aliases": ("astrbot_plugin_qq_tools", "astrbot_plugin_xrelay"),
        "display_names": ("蔗糖的QQ全套工具~", "QQ 全套工具", "QQ全套工具"),
        "label": "QQ 全套工具",
        "provides": "转述（relay_message 打标）与跨会话发送记录",
    },
    "sugarcron": {
        "module": "astrbot_plugin_sugarcron",
        "aliases": (),
        "display_names": ("蔗糖的定时任务~", "定时任务"),
        "label": "定时任务",
        "provides": "定时任务推送（外发记录带任务标记）",
    },
}

# 哪些配置依赖哪个联动插件。元组＝「任一装了就有用」；
# 没装时面板提示 + 注入配额按 0（见 _pack_builder）。
CFG_REQUIRES = {
    "record_outbound": ("qq_tools", "sugarcron"),
    "record_outbound_other": ("sugarcron", "qq_tools"),
    "outbound_cause_window": ("qq_tools", "sugarcron"),
    "outbound_dedup_seconds": ("qq_tools", "sugarcron"),
    "inject_outbound_relay_limit": ("qq_tools",),
    "inject_outbound_relay_window_hours": ("qq_tools",),
    "inject_outbound_away_limit": ("qq_tools",),
    "inject_outbound_cron_limit": ("sugarcron",),
    "inject_outbound_cron_window_hours": ("sugarcron",),
    "inject_outbound_away_cron_limit": ("sugarcron",),
    "inject_outbound_labeled_only": ("qq_tools", "sugarcron"),
    "inject_outbound_pinned": ("qq_tools", "sugarcron"),
    "tool_outbound_history": ("qq_tools", "sugarcron"),
}

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def _norm_dep_name(value) -> str:
    """插件名归一：大小写 / 全角波浪号 / 连字符统一，避免 metadata 写法差异漏判。"""
    return str(value or "").strip().lower().replace("～", "~").replace("-", "_")


def _dep_plugin_dirs() -> list[str]:
    """插件目录候选：本插件同级的 plugins 目录 + 框架 data 目录推导。"""
    dirs = [os.path.dirname(_here)]
    try:
        if get_astrbot_data_path:
            root = str(get_astrbot_data_path())
            dirs += [os.path.join(root, "plugins"),
                     os.path.join(root, "data", "plugins")]
    except Exception:
        pass
    out = []
    for d in dirs:
        if d and d not in out:
            out.append(d)
    return out


def detect_dep_state(context, dep: dict, plugin_dirs: list[str] | None = None) -> dict:
    """检测一个联动插件的真实状态，返回 {state, installed, matched, module}。

    state:
      active   = 注册表里有且已启用 → 联动确实生效
      disabled = 注册表里有但被停用（activated=False 或 star_cls=None）→ 装了，只是停用（绝不能说「未装」）
      inactive = 注册表里没有，磁盘上却有插件目录（加载失败 / 热重载 / 未启用）
      missing  = 真的没装

    三处历史坑：
      1. 只认一个插件名。QQ 全套工具的转述能力在 group_history_agent 里，
         qq_tools 只是同名系插件，必须同时认主名 + 兼容别名 + 显示名。
      2. module_path 形如 data.plugins.xxx.main，split('.')[0] 恒为 "data"，
         必须逐段比对。
      3. 把「停用」和「未安装」混成一个 False，面板就对着装好的插件报「未装」。

    目录兜底只用来区分「没装」和「装了但没跑起来」；注册表里没有启用项时，
    一律不把 installed 置 True，gating 才不会对着空池子放行配额。
    """
    main = str(dep.get("module") or "")
    aliases = [str(x) for x in (dep.get("aliases") or ()) if x]
    displays = {_norm_dep_name(x) for x in (dep.get("display_names") or ()) if x}
    main_n = _norm_dep_name(main)
    alias_ns = {_norm_dep_name(x) for x in aliases}
    alias_ns.discard(main_n)
    known = {main_n} | alias_ns

    def result(state: str, matched: str = "") -> dict:
        return {"state": state, "installed": state == "active",
                "matched": matched, "module": main}

    try:
        stars = list(context.get_all_stars() or [])
    except Exception:
        stars = []

    main_hits: list[tuple[str, bool]] = []
    alias_hits: list[tuple[str, bool]] = []
    for st in stars:
        try:
            raw: list[str] = []
            for attr in ("name", "root_dir_name", "display_name", "module_path"):
                value = getattr(st, attr, None)
                if isinstance(value, str) and value.strip():
                    raw.append(value)
                    if "." in value:
                        raw.extend(value.split("."))
            norm = {_norm_dep_name(x) for x in raw}
            norm.discard("")
            if main_n in norm or (norm & displays):
                bucket, hit_name = main_hits, main
            elif norm & alias_ns:
                hit = sorted(norm & alias_ns)[0]
                bucket = alias_hits
                hit_name = next((x for x in aliases
                                 if _norm_dep_name(x) == hit), hit)
            else:
                continue
            raw_label = str(getattr(st, "root_dir_name", "")
                            or getattr(st, "name", "") or "")
            # 只认得出显示名时别把插件自己的花名塞进面板，用识别到的模块名
            label = raw_label if _norm_dep_name(raw_label) in known else hit_name
            activated = bool(getattr(st, "activated", True))
            # 新式 @register 插件被停用时，框架只把 star_cls 置 None，
            # activated 还是默认 True（只有旧式加载路径才写 activated=False）。
            # 不看 star_cls，停用的插件又会被当成「生效中」。
            if (getattr(st, "star_cls_type", None) is not None
                    and getattr(st, "star_cls", None) is None):
                activated = False
            bucket.append((label, activated))
        except Exception:
            continue

    hits = main_hits + alias_hits
    for label, activated in hits:
        if activated:
            return result("active", label)
    if hits:
        return result("disabled", hits[0][0])

    for name in [main, *aliases]:
        if not name:
            continue
        for base in (plugin_dirs if plugin_dirs is not None else _dep_plugin_dirs()):
            folder = os.path.join(base, name)
            if not os.path.isdir(folder):
                continue
            if any(os.path.isfile(os.path.join(folder, f)) for f in
                   ("main.py", "metadata.yaml", name + ".py")):
                return result("inactive", name)
    return result("missing")


# AstrBot 热重载只清主模块，子包会留在 sys.modules 里导致改了不生效。
# 这里在每次重新执行 main.py 时把本插件的子模块全部丢掉，强制重新导入。
for _mod in [m for m in sys.modules if m == "sugarmemory_core" or m.startswith("sugarmemory_core.")]:
    sys.modules.pop(_mod, None)

from sugarmemory_core.store import (  # noqa: E402
    SugarStore, count_conversations, is_generic_name, alias_key,
    PERMANENT_PREDICATES,
    bot_of_session as _bot_of_session,
    event_sender_name as _event_sender_name, speaker_key as _speaker_key,
)
from sugarmemory_core.l3_import import (  # noqa: E402
    parse_ts as _parse_ts, parse_import_text as _parse_import_text,
    import_report as _import_report, assign_roles as _assign_roles,
    name_key as _name_key,
)
from sugarmemory_core.gate import WriteGate  # noqa: E402
from sugarmemory_core.injector import (  # noqa: E402
    build_block, estimate_tokens, inject, trim_pack)
from sugarmemory_core.pack import (  # noqa: E402
    PackBuilder, _outbound_kind, resolve_preview_scope)
from sugarmemory_core.summarize_core import (  # noqa: E402
    SUMMARY_SYSTEM_PROMPT, apply_summary, parse_summary_json, _normalize_predicate,
    make_skeleton, pick_related_l2, format_related, chunk_transcript,
    merge_summaries, number_transcript, roster_text,
)
from sugarmemory_core.alias_gen import (  # noqa: E402
    ALIAS_SYSTEM_PROMPT, build_alias_prompt, parse_alias_json,
)
from sugarmemory_core.social import (  # noqa: E402
    RELATION_RANKS, affinity_gain, normalize_relation, snapshot_text, stage_of,
    tone_for_stage, DEFAULT_STAGE_DESCS,
)

# 追加内容的段头。写死在代码里、不开放配置：它是「内置规则 vs 本机追加」的分界线，
# 换一个字节就等于换一份 prompt，而这里最不需要的就是让用户去调标点。
SUMMARY_EXTRA_HEADER = (
    "\n\n补充规则（本机主人追加，优先级高于上面所有规则；与上面冲突时以这里为准）：\n"
)
_PROMPT_SECTION_MAX = 70


def summary_prompt_sections(text: str) -> list[dict]:
    """把一份提示词按「xx规则：」这类小标题切段，给面板做只读预览。

    不写死段名：提示词以后加一段，这里自动多一段，不会像配置文件那样过期。
    判据只有两条——整行以「：」结尾、不长于 70 字、且不是列表项（- 开头）。
    开头的输出格式那几行本身就以「：」结尾但太长，会被当正文，标题回落成
    「输出格式与字段」。
    """
    out: list[dict] = []
    title, buf = "输出格式与字段", []
    for ln in str(text or "").splitlines():
        s = ln.strip()
        if (s.endswith("：") and not s.startswith("-")
                and 2 <= len(s) <= _PROMPT_SECTION_MAX):
            if buf or out:
                out.append({"title": title, "body": "\n".join(buf).strip("\n")})
            title, buf = s[:-1], []
            if not out and len(title) > 24:
                title = "输出格式与字段"
        else:
            buf.append(ln)
    if buf:
        out.append({"title": title, "body": "\n".join(buf).strip("\n")})
    return [x for x in out if x["body"].strip()]



def _docstring(text: str):
    """把动态生成的文案当 docstring 用（必须挂在 filter.llm_tool 的内层）。

    AstrBot 在 `@filter.llm_tool` 装饰的那一刻就把 `func.__doc__` 解析成工具描述
    （core/star/register/star_handler.py:632），之后再改 __doc__ 对已注册的工具无效。
    所以文案要按配置生成，就得让这个装饰器比 filter.llm_tool 先跑；
    运行中改配置则走 SugarMemory._sync_tool_docs()，直接改 FunctionTool.description。
    """
    def _apply(func):
        func.__doc__ = text
        return func
    return _apply


def _memory_search_doc(injected: bool, mode: str = "frozen") -> str:
    """memory_search 的工具描述：有没有自动注入、注入的是哪种包，路径不一样。

    inject_enabled 开着 + frozen：每轮开头有一份常驻包，话题命中时才增量刷新。
    inject_enabled 开着 + dynamic（A/B 对照 / 回退）：按本轮原话实时检索的动态块，
      不能沿用「常驻、不随话题变、可能滞后几分钟」那套说法，否则模型会按错的
      前提判断「包里没有＝库里没有」。
    inject_enabled 关着（影子模式）：根本没有注入块，必须让模型知道「不查就看不到」。
    """
    if injected and mode == "dynamic":
        head = """检索自己的记忆库：长期事实、聊天记录、以及你主动发出去的话（转述/定时任务）。

每轮对话开头系统会按**这一轮的原话**实时检索，注入一份动态记忆块：它只带和当前
话题对得上的几条，没检索到的内容本来就不在里面，块里没有 ≠ 库里没有。
它是参考，不是完整档案：
- 块里有对得上的内容 → 直接用来回答，不用重复查
- 用户问具体某件事、某条消息、某个人的细节，而块里没有或不够 → 主动调用本工具查，
  不要因为动态块里没写就断言「没有这回事」
- 用户问「你是不是发过 XX」「那个定时任务还有吗」→ scope 用 outbound
- 用户问「我们之前聊过 XX 吗」「你还记得吗」→ scope 用 all
- 需要确认某条事实的细节 → scope 用 fact"""
    elif injected:
        head = """检索自己的记忆库：长期事实、聊天记录、以及你主动发出去的话（转述/定时任务）。

每轮对话开头系统会注入一份**常驻记忆包**：它按「说话人 + 会话」固定，绝大多数
轮次不随本轮话题变（为了省缓存）；只有你真的聊到某条包外相关记忆、或追问
「那它呢 / 上次 / 刚才」这类指代时，这一轮才会临时把那几条补进来。
它也可能滞后几分钟（刚发过的转述、刚记下的新事不一定已经在里面）。
它是背景底档，不是本轮问题的完整答案：
- 包里有对得上的内容 → 直接用来回答，不用重复查
- 用户问具体某件事、某条消息、某个人的细节，而包里没有或不够 → 主动调用本工具查，
  不要因为常驻包里没写就断言「没有这回事」
- 用户问「你是不是发过 XX」「那个定时任务还有吗」→ scope 用 outbound
- 用户问「我们之前聊过 XX 吗」「你还记得吗」→ scope 用 all
- 需要确认某条事实的细节 → scope 用 fact"""
    else:
        head = """检索自己的记忆库：长期事实、聊天记录、以及你主动发出去的话（转述/定时任务）。

当前没有自动注入的记忆包，记忆库里有什么，你不查就看不到。要回忆过去，先调用本工具：
- 用户问「你是不是发过 XX」「那个定时任务还有吗」→ scope 用 outbound
- 用户问「我们之前聊过 XX 吗」「你还记得吗」→ scope 用 all，查完再回答
- 需要确认某条事实的细节 → scope 用 fact
- 查不到就如实说「没有记录」，别编，也别因为当前这段对话里没有就断言「没这回事」"""
    return head + """

Args:
    query(string): 关键词，从用户消息里提取的词（人名/事件/物品，2~12 个字）
    scope(string): 查什么：all=全部（默认），fact=长期事实，chat=聊天记录，outbound=我发出去的话
    limit(int): 最多返回几条，默认 8"""


@register("sugarmemory", "sugarcane", "蔗糖的记忆中枢~", PLUGIN_VERSION)
class SugarMemory(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        data_root = None
        try:
            data_root = get_astrbot_data_path() if get_astrbot_data_path else None
        except Exception:
            data_root = None
        # 插件配置文件的落点（框架没注入 config 时用它兜底读写）
        self._cfg_path = None
        if data_root:
            self._cfg_path = os.path.join(str(data_root), "config", f"{PLUGIN_NAME}_config.json")
        if config is None:
            loaded: dict = {}
            if self._cfg_path and os.path.exists(self._cfg_path):
                try:
                    with open(self._cfg_path, encoding="utf-8-sig") as f:
                        loaded = json.load(f) or {}
                except Exception:
                    loaded = {}
            config = loaded
        self.config = config
        data_root = data_root or os.path.join(_here, "data_fallback")
        plugin_data = os.path.join(str(data_root), "plugin_data", PLUGIN_NAME)
        os.makedirs(plugin_data, exist_ok=True)
        self.store = SugarStore(os.path.join(plugin_data, "memory.db"))
        # 错峰沉底的相位跨度：0 就算关。只影响「什么时候沉」，不影响沉不沉。
        try:
            self.store.stagger_days = (
                max(0.0, float(self._cfg("decay_stagger_days", 75) or 0))
                if self._cfg("decay_stagger_enabled", False) else 0.0)
        except (TypeError, ValueError):
            self.store.stagger_days = 0.0
        # 老数据里同一个人的几种写法（用户乙 / 用户乙（用户甲））会变成图上两个节点、
        # 看着像两条重复的单向关系。启动时折一次，之后写入都走 canonical_name。
        try:
            for f in self.store.relation_fold_aliases():
                logger.info(f"[SugarMemory] 关系重名自动合并：{f['drop']} → {f['keep']}")
        except Exception as e:
            logger.warning(f"[SugarMemory] 关系重名合并失败：{e}")
        try:
            n = self.store.group_backfill()
            if n:
                logger.info(f"[SugarMemory] 从历史会话补登记 {n} 个群（不进关系图/好感度）")
        except Exception as e:
            logger.warning(f"[SugarMemory] 群登记回填失败：{e}")
        # 人物身份：把名字登记成「一个人 = 一个 QQ + 一串称呼」，顺带把
        # 已经裂开的历史节点并回去（丙哥 → 用户丙 这类）。
        try:
            r = self.store.person_bootstrap()
            for f in r.get("folded") or []:
                logger.info(
                    f"[SugarMemory] 同一个人合并：{f['drop']} → {f['keep']}（{f.get('why','')}）")
            if r.get("created") or r.get("learned"):
                logger.info(
                    f"[SugarMemory] 人物身份：登记 {r.get('created',0)} 人、"
                    f"认出 QQ {r.get('qq_bound',0)} 个、学到称呼 {len(r.get('learned') or [])} 条")
            # 人并完了还要把记忆的归属改过来，否则面板上还是两个人
            ra = r.get("realigned") or {}
            if any(ra.values()):
                logger.info(
                    f"[SugarMemory] 记忆归属对齐：L2 {ra.get('l2',0)} 条、"
                    f"L3 {ra.get('l3',0)} 条、好感 {ra.get('affinity',0)} 个、"
                    f"关系 {ra.get('rels',0)} 条")
        except Exception as e:
            logger.warning(f"[SugarMemory] 人物身份初始化失败：{e}")
        self.gate = WriteGate()
        self._msg_count: dict[str, int] = {}
        self._pending: set[str] = set()
        self._idle_tasks: dict[str, asyncio.Task] = {}
        self._pending_since: dict[str, float] = {}
        self._summary_fail_at: dict[str, float] = {}
        self._summary_lock = asyncio.Lock()
        self._summarizing = False
        # 总结在跑时又来一次触发：以前直接丢掉，批量导入时等于丢一半数据。
        # 现在记下来，跑完补一次（只补一次，不排队堆积）。
        self._summary_pending = False
        # 补录提炼：把批量导入的原话按人分块压成长期记忆。
        # 跟会话总结共用一个锁，避免两边同时打模型。
        self._distilling = False
        self._distill_stop = False
        self._distill_state: dict = {}
        # 存量事实的问法别名补齐只踢一次（懒触发：第一次有人说话时再跑，
        # 避免插件加载期抢事件循环）
        self._alias_kicked = False
        self._last_cleanup = 0.0
        self._cleaning = False
        self._last_sender: dict[str, tuple[float, str]] = {}
        # 每个会话最近一条用户原话，仅用于校验工具参数是不是"用户真说过的"
        self._last_user_text: dict[str, tuple[float, str]] = {}
        # 每个会话最近一次「有人说话」的时刻。转述通道靠它过滤回声：
        # AI 往一个刚还有人说话的会话发的不带标记消息，基本就是当前对话自己的
        # 发送路径（分段器/send 工具），不该再记一遍「主动发送」。
        self._active_sessions: dict[str, float] = {}
        self._parked_tools: dict[str, object] = {}
        # 常驻记忆包缓存：(who, sess) -> 冻结块状态字典。
        # 2026-11-06 A 方案：注入包不再按本轮 query 实时检索，而是按 (说话人,
        # 会话) 冻结成一份 query="" 的常驻包，连同关系快照一起缓存。它落在
        # messages[0]（system），后面跟着整段历史——这里改一个字，provider
        # 的前缀缓存就从最前面整段失效。所以「内容变了也先别换」的真节流
        # 比「内容相同才复用」有用得多：后者在转述/定时这类流水记录每轮都变
        # 时等于没开。字段：block / fp / l2_sig / built_at / rounds / pack_tokens
        # / rel_tokens。
        self._inject_cache: dict[str, dict] = {}
        # 每个会话最近几轮用户原话，只给检索做指代补位（「那它呢」「刚才那个」）。
        # 不进注入块、不进问法判断、不落库，进程重启即清空，无状态残留。
        self._recent_queries: dict[str, list[str]] = {}
        # 最近一轮 L2 排序明细，供 /糖记忆 why <id> 解释「为什么排第 3」。
        self._last_why: dict[int, dict] = {}
        # on_llm_request 里这次请求挂了哪份前缀 → on_llm_response 记真实
        # cached_tokens 时一并落库。按 session_id 索引，只留最近若干条。
        # 没有它，cached 就无法归因到「同 fp 连续轮 / 重建轮」。
        self._last_inject_state: dict[str, dict] = {}
        logger.info(f"[SugarMemory] M1 已加载，数据目录：{plugin_data}")
        self._register_web()
        self._apply_tool_flags()
        self._hook_context_send()

    def _cfg(self, key, default):
        """读配置。真值在分组里（schema 用了 object 嵌套），但老用户的扁平配置也要认。

        顺序：所属分组 → 顶层扁平 → 任意分组。分组优先，这样即使升级时旧扁平键
        还残留在文件里，也不会盖掉分组里的新值。
        """
        try:
            group = CFG_GROUP_OF.get(key)
            if group:
                box = self.config.get(group)
                if isinstance(box, dict) and key in box:
                    value = box.get(key)
                    if value is not None:
                        return value
            if key in self.config:
                value = self.config.get(key, default)
                if value is not None:
                    return value
            for value in self.config.values():
                if isinstance(value, dict) and key in value:
                    got = value.get(key)
                    if got is not None:
                        return got
        except Exception:
            return default
        return default

    def _inject_mode(self) -> str:
        """当前注入架构。frozen=A 方案冻结常驻包；dynamic=旧动态包（A/B / 回退）。"""
        v = str(self._cfg("inject_mode", "frozen") or "frozen").strip().lower()
        return v if v in ("frozen", "dynamic") else "frozen"

    def _cfg_fp(self) -> str:
        """影响 system_prompt 字节的配置指纹。

        缓存项记下「我是用哪版配置生成的」。没有它，改 pack_l2_limit 之类
        会让选中的 L2 条数变化 → l2_sig 跟着变 → 被误判成「用户刚改了记忆」
        → 所有活跃分片下一轮立即重建。现在配置变化单独判，L2 内容改写才
        算 important，两者不再混为一谈。
        """
        try:
            raw = "|".join(
                f"{k}={self._cfg(k, CFG_DEFAULTS.get(k, ''))!r}"
                for k in PROMPT_CFG_KEYS)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "cfg?"

    def _provider_label(self, event=None, response=None) -> str:
        """当前实际使用的 provider 标识，用于按服务商算缓存经济账。

        框架的 provider_id 只在写主库 provider_stats 时才有，插件的
        on_llm_response 里拿不到现成字段，所以从 context 反查；任何一步
        失败都返回空串，绝不影响对话本身。
        """
        for obj in (response, event):
            if obj is None:
                continue
            v = getattr(obj, "provider_id", None)
            if v:
                return normalize_provider_label(str(v)[:64])
        try:
            umo = str(getattr(event, "unified_msg_origin", "") or "")
            getter = getattr(self.context, "get_using_provider", None)
            if not callable(getter):
                return ""
            prov = getter(umo)
            if prov is None:
                return ""
            cfg = getattr(prov, "provider_config", None)
            pid = str(cfg.get("id") or "")[:32] if isinstance(cfg, dict) else ""
            if not pid:
                meta = getattr(prov, "meta", None)
                pid = str(meta().id)[:32] if callable(meta) else ""
            model = ""
            gm = getattr(prov, "get_model", None)
            if callable(gm):
                model = str(gm() or "")[:48]
            return normalize_provider_label(f"{pid}/{model}" if model else pid)
        except Exception:
            logger.debug("[SugarMemory] 识别 provider 失败", exc_info=True)
            return ""

    def _invalidate_cache_for_config(self):
        """改了影响 prompt 的配置 → 一次性统一失效，而不是伪装成 L2 改写。

        这是用户主动改配置，前缀确实必须重写，所有分片各重建一次是应得的；
        要防的是「没改记忆却被误判成重要变更」。统一清缓存还顺带保证
        缓存项里不会残留旧 cfg_fp 的块。
        """
        n = len(self._inject_cache or {})
        self._inject_cache.clear()
        if n:
            logger.info(
                "[SugarMemory] 影响 prompt 的配置已变更：统一失效 %s 个常驻包分片，"
                "下一轮各自重建一次", n)

    def _dep_state(self, key: str) -> dict:
        """联动插件的细分状态（active / disabled / inactive / missing）。

        面板靠它把「已停用」和「未安装」分开说；检测本身放在模块级
        detect_dep_state()，脱离 AstrBot 运行时也能单测。
        """
        dep = DEP_PLUGINS.get(key)
        if not dep:
            return {"state": "missing", "installed": False,
                    "matched": "", "module": ""}
        try:
            return detect_dep_state(self.context, dep, _dep_plugin_dirs())
        except Exception:
            logger.debug("[SugarMemory] 联动插件检测失败", exc_info=True)
            return {"state": "missing", "installed": False,
                    "matched": "", "module": dep["module"]}

    def _dep_installed(self, key: str) -> bool:
        """注入端 gating 用：只有「已启用」的联动插件才算生效。

        停用 / 加载失败 / 真没装都返回 False（配额按 0），但面板走
        _integration_payload() 拿 state 区分措辞，不再对着装好的插件报「未装」。
        """
        return bool(self._dep_state(key)["installed"])

    def _integration_payload(self) -> dict:
        """主页「联动」卡和配置页提示共用的状态数据。"""
        out = {}
        for key, dep in DEP_PLUGINS.items():
            st = self._dep_state(key)
            out[key] = {
                "installed": st["installed"],
                "state": st["state"],
                "matched": st["matched"],
                "label": dep["label"],
                "module": dep["module"],
                "aliases": list(dep.get("aliases") or ()),
                "provides": dep["provides"],
                "cfg_keys": [k for k, req in CFG_REQUIRES.items() if key in req],
            }
        # 注入端的实际生效值（gating 之后），页面上能直接看到「现在到底带不带」
        relay_on = out.get("qq_tools", {}).get("installed", False)
        cron_on = out.get("sugarcron", {}).get("installed", False)
        out["effective"] = {
            "relay_limit": int(self._cfg("inject_outbound_relay_limit", 3) or 0) if relay_on else 0,
            "cron_limit": int(self._cfg("inject_outbound_cron_limit", 2) or 0) if cron_on else 0,
        }
        return out

    def _cfg_items(self) -> list[dict]:
        """给面板「总配置」用的字段清单（带当前值）。"""
        items = []
        for key, label, typ, group, hint in CFG_ITEMS:
            default = CFG_DEFAULTS.get(key, "")
            value = self._cfg(key, default)
            if typ == "bool":
                value = bool(value)
            elif typ == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = int(default)
            else:
                value = str(value or "")
            it = {"key": key, "label": label, "type": typ,
                  "group": CFG_GROUP_LABEL.get(group, group),
                  "group_key": group,
                  "group_hint": CFG_GROUP_HINT.get(group, ""),
                  "hint": hint, "value": value, "default": default,
                  "multiline": key in CFG_MULTILINE}
            # 联动提示：这项依赖的外部插件一个都没装时，配了也不生效，
            # 在配置旁挂出来，用户就不用对着「明明开了却没动静」抓瞎了
            req = CFG_REQUIRES.get(key)
            if req:
                deps = [DEP_PLUGINS[d]["label"] for d in req if d in DEP_PLUGINS]
                states = [self._dep_state(d) for d in req]
                it["requires"] = deps
                it["requires_ok"] = any(s["installed"] for s in states)
                # 停用和没装是两回事：文案要分开，别对着装好的插件说「未安装」
                it["requires_state"] = (
                    "ok" if it["requires_ok"] else
                    "disabled" if any(s["state"] in ("disabled", "inactive")
                                      for s in states) else "missing")
            items.append(it)
        return items

    def _save_cfg(self, patch: dict) -> tuple[bool, str]:
        """写回插件配置文件并立刻生效（内存 + 磁盘）。"""
        clean: dict = {}
        for key, value in (patch or {}).items():
            if key not in CFG_KEYS:
                continue
            typ = next(t for k, _l, t, _g, _h in CFG_ITEMS if k == key)
            if typ == "bool":
                value = bool(value)
            elif typ == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            else:
                value = str(value or "")
            clean[key] = value
        if not clean:
            return False, "没有可保存的字段"
        for key, value in clean.items():
            group = CFG_GROUP_OF.get(key)
            placed = False
            if group:
                try:
                    box = self.config.get(group)
                    if not isinstance(box, dict):
                        box = {}
                        self.config[group] = box
                    box[key] = value
                    placed = True
                except Exception:
                    logger.warning(f"[SugarMemory] 写入分组 {group} 失败，退回扁平", exc_info=True)
            if not placed:
                try:
                    self.config[key] = value
                except Exception:
                    pass
        saver = getattr(self.config, "save_config", None)
        if callable(saver):
            try:
                saver()
                return True, "saved"
            except Exception:
                logger.warning("[SugarMemory] save_config 失败，改走直接写文件", exc_info=True)
        if not self._cfg_path:
            return False, "找不到配置文件路径"
        try:
            data = {}
            try:
                data = dict(self.config)
            except Exception:
                data = {}
            with open(self._cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, "file"
        except Exception as e:
            return False, f"写入失败：{e}"

    def _apply_tool_flags(self):
        """按配置挂载/卸载给 AI 的工具（面板关掉就立刻从工具列表里消失）。"""
        try:
            mgr = self.context.provider_manager.llm_tools
        except Exception:
            return
        for name, key in (("relation_set", "tool_relation_write"),
                          ("relation_query", "tool_relation_query"),
                          ("memory_search", "tool_memory_search"),
                          ("outbound_history", "tool_outbound_history")):
            enabled = bool(self._cfg(key, True))
            try:
                cur = mgr.get_func(name)
                if enabled and cur is None:
                    parked = self._parked_tools.pop(name, None)
                    if parked is not None:
                        mgr.func_list.append(parked)
                elif not enabled and cur is not None:
                    self._parked_tools[name] = cur
                    mgr.remove_func(name)
            except Exception:
                logger.warning(f"[SugarMemory] 切换工具 {name} 失败", exc_info=True)
        self._sync_tool_docs(mgr)

    def _sync_tool_docs(self, mgr=None):
        """按当前开关改写工具描述（目前只有 memory_search 需要）。

        工具描述在 @filter.llm_tool 注册时就定死了，光改配置不会变。而 memory_search 的
        描述里写着「每轮开头会自动注入记忆包」——影子模式下这句话是假的，模型会去找一个
        不存在的注入块，找不到就跟用户说「我没看到记忆」。所以开关一动，这里同步一次。

        工具被面板关掉时对象在 _parked_tools 里躺着（已从 func_list 摘除），也要一并改，
        不然重新打开时会带着旧文案回来。
        """
        try:
            desc = _memory_search_doc(bool(self._cfg("inject_enabled", True)),
                                      self._inject_mode())
        except Exception:
            return
        if mgr is None:
            try:
                mgr = self.context.provider_manager.llm_tools
            except Exception:
                return
        for tool in (mgr.get_func("memory_search"),
                     getattr(self, "_parked_tools", {}).get("memory_search")):
            if tool is None:
                continue
            try:
                if tool.description != desc:
                    tool.description = desc
            except Exception:
                logger.warning("[SugarMemory] 改写 memory_search 描述失败", exc_info=True)

    @staticmethod
    def _is_group(event) -> bool:
        """群聊判定。get_group_id() 在某些适配器上可能为空，再兜一层 origin/消息类型。"""
        try:
            if event.get_group_id():
                return True
        except Exception:
            pass
        try:
            origin = str(getattr(event, "unified_msg_origin", "") or "")
            if "GroupMessage" in origin or origin.startswith("group:"):
                return True
            if "FriendMessage" in origin or origin.startswith("private:"):
                return False
        except Exception:
            pass
        try:
            mo = getattr(event, "message_obj", None)
            name = type(getattr(mo, "type", "") or "").__name__
            if "Group" in name or "Guild" in name:
                return True
        except Exception:
            pass
        return False

    def _remember_group(self, event):
        """把见过的群记下来（群号 + 群名）。面板要靠它把「群」从人名里摘出去。"""
        try:
            gid = str(event.get_group_id() or "").strip()
        except Exception:
            gid = ""
        if not gid:
            return
        name = ""
        try:
            grp = getattr(getattr(event, "message_obj", None), "group", None)
            name = str(getattr(grp, "group_name", "") or "").strip()
        except Exception:
            name = ""
        try:
            self.store.group_remember(gid, name)
        except Exception:
            logger.debug("[SugarMemory] 记群失败", exc_info=True)

    def _is_group_name(self, name: str, keys: set | None = None) -> bool:
        """这个名字是群而不是人吗？群号、群名、group: 前缀都算。"""
        s = str(name or "").strip()
        if not s:
            return False
        if s.startswith("group:"):
            return True
        try:
            ks = keys if keys is not None else self.store.group_keys()
        except Exception:
            return False
        return s.casefold() in ks

    @staticmethod
    def _session_key(event) -> str:
        """会话标识：优先 unified_msg_origin，其次群号/私聊对象。"""
        try:
            origin = getattr(event, "unified_msg_origin", "") or ""
            if origin:
                return str(origin)[:120]
        except Exception:
            pass
        try:
            gid = event.get_group_id()
            if gid:
                return f"group:{gid}"[:120]
        except Exception:
            pass
        try:
            sid = event.get_sender_id()
            if sid:
                return f"private:{sid}"[:120]
        except Exception:
            pass
        return ""

    @staticmethod
    def _bot_of(session) -> str:
        """会话属于哪个机器人：umo 第一段就是平台实例 id（一个账号一个实例）。"""
        return _bot_of_session(session)

    def _event_bot(self, event) -> str:
        """本轮事件由哪个机器人收到。

        一个平台实例 = 一个机器人账号，所以 platform_id 就是记忆命名空间。
        多账号共用一个 AstrBot 进程时，这是唯一能把「A 的定时提醒」和
        「B 的记忆包」分开的维度——umo 里的会话 id 只说明发给谁，不说明是谁发的。
        """
        try:
            pid = str(event.get_platform_id() or "").strip()
            if pid:
                return pid[:32]
        except Exception:
            logger.debug("[SugarMemory] 取 platform_id 失败", exc_info=True)
        return self._bot_of(self._session_key(event))

    def _iso(self, bot: str) -> str:
        """隔离开关：bot_isolation 关掉就退回全局共享（老行为，逃生阀）。"""
        return str(bot or "")[:32] if self._cfg("bot_isolation", True) else ""

    def _remember_sender(self, session: str, name: str):
        if not session or not name:
            return
        self._last_sender[session] = (time.time(), name[:40])
        if len(self._last_sender) > 200:
            oldest = min(self._last_sender, key=lambda k: self._last_sender[k][0])
            self._last_sender.pop(oldest, None)

    def _remember_text(self, session: str, text: str):
        """存下本轮用户原话，给 relation_set 校验参数用（只存最近一条）。"""
        if not session or not text:
            return
        self._last_user_text[session] = (time.time(), text[:500])
        if len(self._last_user_text) > 200:
            oldest = min(self._last_user_text, key=lambda k: self._last_user_text[k][0])
            self._last_user_text.pop(oldest, None)

    def _mark_active(self, session: str):
        """标记「这个会话刚有人说话」。给 _record_outbound 的回声过滤用。"""
        if not session:
            return
        self._active_sessions[session] = time.time()
        if len(self._active_sessions) > 300:
            oldest = min(self._active_sessions, key=lambda k: self._active_sessions[k])
            self._active_sessions.pop(oldest, None)

    def _quote_check(self, event, *names) -> tuple[bool, str]:
        """本轮用户原话里有没有提到这些人名。返回 (是否放行, 原话)。

        AI 读到的上下文里混着记忆包、别人的转述、贴进来的网页，那些字不是
        「这个人说的」。校验只认本轮用户亲手打的字，防 AI 把读到的内容当成授权。
        """
        session = self._session_key(event)
        hit = self._last_user_text.get(session)
        raw = hit[1] if hit and time.time() - hit[0] <= 300 else ""
        text = self._norm_name(raw)
        if len(text) < 2:
            return True, raw  # 拿不到原话（图片/语音/超时），fail-open 不挡
        me = self._sender_name(event)
        for name in names:
            key = self._norm_name(name)
            if not key:
                continue
            if me and self._same_person(me, name):
                return True, raw  # 涉及说话人自己，那是他的判断，本来就该记
            if key in text:
                return True, raw
        return False, raw

    def _l3_expiry(self) -> float:
        """按配置算 L3 到期时间；0 表示永久（与配置说明一致）。"""
        try:
            days = int(self._cfg("l3_retention_days", 14))
        except (TypeError, ValueError):
            days = 14
        return 0.0 if days <= 0 else time.time() + days * 86400

    def _gain_affinity(self, subject: str, qq: str = ""):
        """每说一条用户消息加一点好感度；越接近上限涨得越慢（见 social.affinity_gain）。"""
        if not self._cfg("affinity_enabled", True):
            return
        subject = (subject or "").strip()[:40]
        if not subject:
            return
        # 顺手把「这个昵称 = 这个 QQ」记进人物身份：同一个人换了昵称、或者被
        # 起了外号，好感度和关系都该落在同一个人身上，而不是各算一份。
        # 主流程（record_message）已经 ensure 过，这里只做认领、不再计一次 hits。
        try:
            subject = self.store.person_resolve(subject) or self.store.person_ensure(
                subject, qq=str(qq or ""), source="event") or subject
        except Exception:
            logger.debug("[SugarMemory] 绑定人物身份失败", exc_info=True)
        try:
            cap = max(1, int(self._cfg("affinity_cap", 1000) or 1000))
        except (TypeError, ValueError):
            cap = 1000
        row = self.store.affinity_get(subject) or {}
        gain = affinity_gain(float(row.get("xp") or 0), cap)
        self.store.affinity_add(subject, gain)

    # ---------- Token 用量 ----------

    def _record_tokens(self, module: str, *, usage=None, input: int = 0, cached: int = 0,
                       output: int = 0, source: str = "actual", subject: str = "",
                       session_id: str = "", provider_id: str = ""):
        """usage 有就用 provider 报的真实值，没有就用传进来的估算值。全 0 不记。

        module='chat' 时自动把本轮 system 前缀的状态（哪份 fp、是否刚重建、
        第几轮、哪版配置、重建原因）跟真实 cached_tokens 一起落库。没有这项归因，
        cached 就只是「整轮请求命中了多少」，说不清 180s/12 轮合不合理，
        也分不出「插件重载冷启动」和「记忆包真漂移」这两种完全不同的 0%。
        """
        if not self._cfg("token_stats_enabled", True):
            return
        try:
            if usage is not None:
                input = int(getattr(usage, "input_other", 0) or 0)
                cached = int(getattr(usage, "input_cached", 0) or 0)
                output = int(getattr(usage, "output", 0) or 0)
                source = "actual"
            if input + cached + output <= 0:
                return
            extra: dict = {}
            if module == "chat":
                st = (self._last_inject_state or {}).get(session_id) or {}
                extra = {
                    "arch_ver": str(st.get("arch_ver") or ""),
                    "inject_fp": str(st.get("fp") or ""),
                    "rebuilt": int(st.get("rebuilt") or 0),
                    "freeze_rounds": int(st.get("rounds") or 0),
                    "cfg_ver": str(st.get("cfg_fp") or ""),
                    "rebuild_reason": str(st.get("reason") or ""),
                    # 常驻底档自己的指纹：面板判「冻结生效没」用它，
                    # 整块 fp 换个话题就会变，拿它当故障指标会天天报警。
                    "prefix_sig": str(st.get("prefix_sig") or ""),
                }
                # 重载 / 重启后内存缓存被清空，下一轮必然写一次新缓存。
                # 在日志里明说这是固定正常，别让用户（和以后的我）拿它当故障查。
                if str(st.get("reason") or "") in COLD_REASONS and not cached:
                    logger.info(
                        "[SugarMemory] 本轮为重载冷启动（reason=%s，fp=%s）："
                        "缓存命中率 0%% 是固定正常的，不计入失败样本；"
                        "下一轮同 fp 应恢复命中",
                        str(st.get("reason") or ""), str(st.get("fp") or ""))
            self.store.token_record(module, input=input, cached=cached, output=output,
                                    source=source, subject=subject, session_id=session_id,
                                    provider_id=provider_id, **extra)
        except Exception:
            logger.debug("[SugarMemory] token 用量记录失败", exc_info=True)

    @filter.on_llm_response()
    async def record_llm_tokens(self, event: AstrMessageEvent, response):
        """对话本身一轮的真实用量（provider 没报 usage 就不记，不瞎估）。"""
        try:
            self._record_tokens("chat", usage=getattr(response, "usage", None),
                                subject=self._sender_name(event),
                                session_id=self._session_key(event),
                                provider_id=self._provider_label(event, response))
        except Exception:
            logger.debug("[SugarMemory] 对话用量记录失败", exc_info=True)

    # ---------- 注入（只进 system_prompt） ----------

    def _sender_name(self, event) -> str:
        return _event_sender_name(event)

    def _identify_quietly(self, event, sender) -> None:
        """没有正文的消息只认人，不写记忆。

        图片、表情、语音、纯引用图都拿不到 message_str，但「这个 QQ 发了东西」
        本身就是最硬的身份证据：号是活的，昵称是当前这个。以前空文本直接
        return，这些人就永远进不了通讯录，只能等总结从别人的话里把他们猜出来
        ——猜出来的那份没 QQ，面板上永远是空壳。
        不写 L3 是因为一堆「[图片]」进正文只会白烧 token、挤占总结预算，
        也加不了好感度：没说的话本来就不该算说过。
        """
        try:
            name = self._sender_name(event) or str(sender or "")
            if not name or not str(sender or ""):
                return
            if is_generic_name(name) or self._is_pseudo_sender(name, ""):
                return
            try:
                is_group = bool(event.get_group_id())
            except Exception:
                is_group = False
            # 群名片只当别名，别拿它顶掉本人的 QQ 昵称（原因见正文路径那段）
            self.store.person_ensure(name, qq=str(sender), source="event",
                                     tier=2 if is_group else 0)
        except Exception:
            logger.debug("[SugarMemory] 空消息认人失败", exc_info=True)

    def _is_pseudo_sender(self, name: str, text: str) -> bool:
        """挡掉不是人说的话。

        astrbot_plugin_qq_tools 的定时唤醒会伪造一条消息：nickname 写成「系统唤醒」，
        但 sender_id 用的是真人 QQ，所以看起来跟用户发言一模一样。不拦掉的话它会被
        记进 L3、加好感度、还会被总结当成用户亲口说的话。
        """
        who = str(name or "").strip()
        if who and who in self._ignore_senders():
            return True
        return str(text or "").lstrip().startswith(PSEUDO_SENDER_PREFIX)

    def _ignore_senders(self) -> set[str]:
        raw = self._cfg("ignore_senders", ["系统唤醒"])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return {"系统唤醒"}
        return {str(item).strip() for item in raw if str(item).strip()}

    @staticmethod
    def _norm_name(s: str) -> str:
        """去零宽字符/空白后归一，用于人名比对（不用于存储）。"""
        s = str(s or "")
        s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", s)
        return s.strip().casefold()

    def _same_person(self, a: str, b: str) -> bool:
        x, y = self._norm_name(a), self._norm_name(b)
        if not x or not y:
            return False
        if x == y:
            return True
        # 群名片常带前后缀（「张三(摸鱼)」/「张三」），短的一方被长的一方包含也算同一人
        return min(len(x), len(y)) >= 2 and (x in y or y in x)

    def _is_admin(self, event) -> bool:
        """管理员判定：平台 is_admin + 配置里的额外授权名单（按 sender_id）。"""
        try:
            if event.is_admin():
                return True
        except Exception:
            pass
        extra = str(self._cfg("tool_relation_admin_ids", "") or "")
        ids = {p.strip() for p in re.split(r"[,\s，、;；]+", extra) if p.strip()}
        if not ids:
            return False
        try:
            sid = str(event.get_sender_id() or "").strip()
        except Exception:
            return False
        return bool(sid) and sid in ids

    def _relation_block(self, subject: str) -> str:
        """「这个人跟谁什么关系」压成几十 token 的小段文本。零 LLM、零工具调用。"""
        if not subject:
            return ""
        try:
            # 同一个人可能有几种写法（「张三」/「张三（群友）」），先归一到库里存的写法，
            # 否则明明有记录也会当成没有，AI 就会再写一遍同样的关系
            try:
                subject = self.store.canonical_name(subject) or subject
            except Exception:
                pass
            rows = self.store.relations_all()
            out_edges = [r for r in rows if r.get("source") == subject]
            in_edges = [r for r in rows if r.get("target") == subject]
            aff = self.store.affinity_get(subject) or {}
            xp = float(aff.get("xp") or 0)
            stage = ""
            tone = ""
            if xp > 0:
                try:
                    stages = self.store.affinity_stages()
                    _cur = stage_of(xp, stages)
                    stage = _cur.get("name", "")
                except Exception:
                    stages, stage = [], ""
                if stage and self._cfg("affinity_tone", True):
                    # 阶段描述优先（面板逐阶段自定义的文案），空了才按名字猜
                    _desc = next((str(s.get("desc") or "") for s in (stages or [])
                                  if s.get("name") == stage), "")
                    tone = tone_for_stage(stage, _desc)
            try:
                limit = int(self._cfg("inject_relation_limit", 6) or 6)
            except (TypeError, ValueError):
                limit = 6
            return snapshot_text(subject, xp, stage, out_edges, in_edges,
                                 limit=max(1, limit), tone=tone)
        except Exception:
            logger.debug("[SugarMemory] 关系快照生成失败", exc_info=True)
            return ""

    def _relation_detail(self, subject: str) -> dict:
        """面板「注入明细」用的关系快照结构化数据，与 _relation_block 同源。

        真实注入里，关系快照是记忆块之后单独追加的一段 system_prompt，
        但面板明细之前只渲染 pack，这段就 invisibility 了——用户看注入原文
        有关系、看明细没有，以为两处对不上。这里把同一份数据拆成结构化
        行给前端，纯本地渲染，不经 LLM。
        """
        d = {"subject": subject or "", "off": False, "xp": 0.0, "stage": "",
             "out": [], "in": [], "note": ""}
        if not self._cfg("inject_relations", False):
            d["off"] = True
            d["note"] = "人际关系网总开关没开，关系不注入对话"
            return d
        if not subject:
            d["note"] = "认不出当前人物，没有关系快照"
            return d
        try:
            who = subject
            try:
                who = self.store.canonical_name(who) or who
            except Exception:
                pass
            rows = self.store.relations_all()
            try:
                limit = max(1, int(self._cfg("inject_relation_limit", 6) or 6))
            except (TypeError, ValueError):
                limit = 6
            d["subject"] = who
            d["out"] = [{"target": r.get("target"), "relation": r.get("relation")}
                        for r in rows if r.get("source") == who][:limit]
            d["in"] = [{"source": r.get("source"), "relation": r.get("relation")}
                       for r in rows if r.get("target") == who][:limit]
            aff = self.store.affinity_get(who) or {}
            d["xp"] = float(aff.get("xp") or 0)
            if d["xp"] > 0:
                try:
                    stages = self.store.affinity_stages()
                    d["stage"] = stage_of(d["xp"], stages).get("name", "")
                except Exception:
                    pass
        except Exception:
            logger.debug("[SugarMemory] 关系明细生成失败", exc_info=True)
            d["note"] = "关系明细生成失败"
        return d

    def _pack_builder(self, subject: str = "", session_id: str = "",
                      extra_subjects: list[str] | None = None,
                      l2_keep_min_override: int | None = None,
                      lite: bool = False, bot_id: str = "",
                      semantic_scores: dict | None = None) -> PackBuilder:
        """统一构造 PackBuilder。

        预算相关的四个参数只在这里接一次：注入主链路、面板预览都走它，
        免得以后加一个参数又漏掉某条调用路径（漏掉的那条会悄悄用默认值）。

        l2_keep_min_override：只给常驻冻结包用。query="" 时
        PackBuilder._filter_by_relevance 走「零命中 → 只留 l2_keep_min 条」
        的保底分支，默认 3 条会把整个人档案砍到只剩 3 条。冻结包没有
        query 相关性可用，必须改成按置信度/稳度带满 l2_limit 条，
        否则省下了缓存却丢了召回。动态统计轨仍用原值，别串过去。

        lite：frozen 模式下统计轨的结果不进 prompt，只需 L2 命中标记，
        让 PackBuilder 跳过 l3/骨架/outbound/cross 那些重查询。
        """
        def _int(key, default):
            try:
                return int(self._cfg(key, default))
            except (TypeError, ValueError):
                return int(default)

        def _bool(key, default):
            v = self._cfg(key, default)
            return default if v is None else bool(v)

        ratio = _int("pack_l2_rel_ratio", 30)
        # 联动 gating：转述/定时的记录都来自外部插件（QQ 全套工具 / 定时任务），
        # 对应插件没装时池子永远是空的，配额留着只会让配置看起来「开了但没反应」。
        # 未装 → 配额按 0（等检测到插件装上，这里自动恢复配置值，不用手动改）。
        relay_dep = self._dep_installed("qq_tools")
        cron_dep = self._dep_installed("sugarcron")
        # 转述 / 定时各自配额。以前合成一个 inject_outbound_limit 混着算，
        # 模型分不清哪句是替人带的话、哪句是闹钟发的。
        relay_lim = max(0, _int("inject_outbound_relay_limit", 3)) if relay_dep else 0
        cron_lim = max(0, _int("inject_outbound_cron_limit", 2)) if cron_dep else 0
        return PackBuilder(
            self.store,
            # L3 原话通道默认 0 是有意设计（作者定稿）：astrbot 自带最近若干轮
            # 对话上下文，原话再进记忆包是重复内容白烧 token，还挤占 L2/摘要
            # 的预算。要召回更早的一次性话（打算/转述这类被降级到 L3 的）时，
            # 在配置里手动调大即可，默认不背这个成本。
            l3_snippets=_int("pack_l3_snippets", 0),
            l2_limit=_int("pack_l2_limit", 12),
            subject=subject,
            session_id=session_id,
            extra_subjects=extra_subjects,
            rel_ratio=max(0, ratio) / 100.0,
            l2_keep_min=(max(1, int(l2_keep_min_override))
                         if l2_keep_min_override is not None
                         else max(1, _int("pack_l2_keep_min", 3))),
            l3_keep_min=max(0, _int("pack_l3_keep_min", 0)),
            # 跨会话两条通道：都是「不管当前是谁」都带上的
            # outbound_limit 只在关掉 labeled_only（连普通回声也带）时当总条数用
            outbound_limit=max(relay_lim, cron_lim),
            outbound_relay_limit=relay_lim,
            outbound_cron_limit=cron_lim,
            # 转述/定时两个池子各自的新鲜窗（小时）。过窗不常驻进包，追问照旧召回
            outbound_relay_window_hours=max(
                0, _int("inject_outbound_relay_window_hours", 24)),
            outbound_cron_window_hours=max(
                0, _int("inject_outbound_cron_window_hours", 24)),
            cross_limit=max(0, _int("inject_cross_limit", 5)),
            cross_include_self=_bool("inject_cross_include_self", False),
            outbound_labeled_only=_bool("inject_outbound_labeled_only", True),
            outbound_pinned=_bool("inject_outbound_pinned", True),
            # 被追问「你刚才跟 X 说了什么」时，补几条发到别的会话的转述；
            # 问定时任务另有一条通道，两边互不串台。同样受联动 gating。
            outbound_away_limit=(max(0, _int("inject_outbound_away_limit", 1))
                                 if relay_dep else 0),
            outbound_away_cron_limit=(max(0, _int("inject_outbound_away_cron_limit", 1))
                                      if cron_dep else 0),
            query_expand=_bool("pack_query_expand", True),
            skeleton_limit=max(0, _int("inject_skeleton_limit", 2)),
            # 语义分由 async 上下文预计算好再查表传进来：pack 层保持纯同步、
            # 零网络。没配 provider / 没开开关 / 本轮不需要时 scores 为空，
            # 整个 semantic_fn 直接是 None，一点调用都不发生。
            semantic_fn=(self._semantic_lookup(semantic_scores)
                         if _bool("pack_semantic_fallback", False)
                         and semantic_scores else None),
            candidate_limit=max(0, _int("pack_l2_candidate_limit", 200)),
            context_rounds=max(0, _int("pack_context_rounds", 3)),
            bot_id=self._iso(bot_id),
            lite=lite,
            logger=logger,
        )

    @staticmethod
    def _semantic_lookup(scores: dict | None):
        """把预计算好的 {l2_id: 分} 包成 PackBuilder 要的同步回调。"""
        frozen = dict(scores or {})
        return lambda query, rows: frozen

    def _semantic_ready(self) -> bool:
        """语义路是否通电：总开关 + 用户明确填的 Provider ID，缺一不可。

        旧实现是「开了开关就去摸当前对话 provider」，而 AstrBot 的 embedding
        接口基本都是 async，`_semantic_fn` 里一句 `if iscoroutine: return {}`
        让它永远返回空 dict —— 开关是个假的。现在没配 ID 就整条路不通电，
        零调用零开销，配了才在 async 上下文里正经跑。
        """
        if not bool(self._cfg("pack_semantic_fallback", False)):
            return False
        return bool(str(self._cfg("semantic_provider_id", "") or "").strip())

    def _semantic_provider(self):
        pid = str(self._cfg("semantic_provider_id", "") or "").strip()
        if not pid:
            return None
        try:
            prov = self.context.get_provider_by_id(pid)
            if prov is None:
                logger.warning(f"[SugarMemory] 语义兜底 Provider 不存在：{pid}，本轮跳过")
            return prov
        except Exception:
            logger.warning(f"[SugarMemory] 取语义兜底 Provider 失败：{pid}")
            return None

    async def _semantic_scores(self, query: str, rows: list[dict]) -> dict[int, float]:
        """语义补召回：query 与每条 L2 的向量余弦相似度（0~1）。

        只在字面/概念/问法别名全部零命中时由 inject_memory 预计算一次，
        pack 层拿分数纯查表。同步 / 异步 embedding 接口都认，统一套超时；
        任何一步失败都返回空 dict，由 pack 静默退回按置信度保底。
        """
        prov = self._semantic_provider()
        if prov is None or not query or not rows:
            return {}
        embed = getattr(prov, "embedding", None) or getattr(prov, "get_embedding", None)
        if not callable(embed):
            return {}
        try:
            timeout = max(0.2, float(self._cfg("semantic_timeout_ms", 1500)) / 1000.0)
        except (TypeError, ValueError):
            timeout = 1.5
        texts = [str(query)] + [
            (f"{r.get('subject', '')}{r.get('predicate', '')}"
             f"{r.get('value', '')}{r.get('ask_alias', '')}")
            for r in rows[:80]
        ]
        try:
            if inspect.iscoroutinefunction(embed):
                res = await asyncio.wait_for(embed(texts), timeout)
            else:
                # 同步接口丢进线程，别把整个事件循环卡在等网络上
                res = await asyncio.wait_for(asyncio.to_thread(embed, texts), timeout)
                if inspect.isawaitable(res):
                    res = await asyncio.wait_for(res, timeout)
            vecs = list(res)
            if len(vecs) < len(texts):
                return {}

            def _cos(a, b):
                na = sum(x * x for x in a) ** 0.5
                nb = sum(x * x for x in b) ** 0.5
                if not na or not nb:
                    return 0.0
                return sum(x * y for x, y in zip(a, b)) / (na * nb)

            qv = vecs[0]
            out: dict[int, float] = {}
            for r, v in zip(rows, vecs[1:]):
                try:
                    out[int(r.get("id", 0))] = max(0.0, min(1.0, _cos(qv, v)))
                except Exception:
                    continue
            return out
        except asyncio.TimeoutError:
            logger.debug("[SugarMemory] 语义兜底超时，本轮退回字面路")
            return {}
        except Exception:
            logger.debug("[SugarMemory] 语义兜底失败，本轮退回字面路", exc_info=True)
            return {}

    async def _run_alias_batch(self, batch: int = 10) -> int:
        """给还没有问法别名的 active L2 批量生成别名（一次 LLM 调用）。

        低优先级后台任务：失败静默放弃这一批，下次总结后自动再试；
        不参与熔断（失败不影响对话主链路，顶多旧事实晚点有别名）。
        返回本次写了几条。
        """
        if not self._cfg("alias_autogen", True):
            return 0
        items = self.store.l2_need_alias(limit=max(1, int(batch)))
        if not items:
            return 0
        provider = self._summary_provider()
        if provider is None:
            return 0
        prompt = build_alias_prompt(items)
        raw = ""
        try:
            resp = await asyncio.wait_for(
                provider.text_chat(prompt=prompt, system_prompt=ALIAS_SYSTEM_PROMPT),
                timeout=max(20, int(self._cfg("summary_timeout_seconds", 120) or 120)),
            )
            raw = str(getattr(resp, "completion_text", "") or "")
        except Exception as e:
            logger.debug(f"[SugarMemory] 问法别名生成失败（下次再试）：{e!r}")
            return 0
        if raw:
            self._record_tokens("alias", source="estimate",
                                input=estimate_tokens(ALIAS_SYSTEM_PROMPT + prompt),
                                output=estimate_tokens(raw))
        parsed = parse_alias_json(raw)
        if not parsed:
            return 0
        written = 0
        for it in items:
            asks = parsed.get(int(it["id"]))
            if asks:
                self.store.l2_set_alias(int(it["id"]), "；".join(asks))
                written += 1
        if written:
            done, total = self.store.l2_alias_stats()
            logger.info(f"[SugarMemory] 问法别名：本次补 {written} 条（{done}/{total} 已覆盖）")
        return written

    async def _kick_alias_backfill(self):
        """存量旧事实的别名补齐：每个进程生命周期只踢一次，静默跑完为止。"""
        if self._alias_kicked:
            return
        self._alias_kicked = True
        try:
            while self._cfg("alias_autogen", True) \
                    and self.store.l2_need_alias(limit=1):
                n = await self._run_alias_batch()
                if not n:
                    break
        except Exception:
            logger.debug("[SugarMemory] 别名补齐任务异常", exc_info=True)

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            # 存量旧事实还没有问法别名：第一次有人说话时后台补齐（只踢一次）
            if not self._alias_kicked and self._cfg("alias_autogen", True):
                asyncio.create_task(self._kick_alias_backfill())
            # 群聊只给查、不给写：直接从本次请求的工具集里摘掉 relation_set
            if self._is_group(event) and not self._cfg("tool_write_in_group", False):
                try:
                    if req.func_tool:
                        req.func_tool.remove_tool("relation_set")
                except Exception:
                    logger.debug("[SugarMemory] 群聊摘除 relation_set 失败", exc_info=True)
            # 作用域：默认只取当前说话人的 L2 + 本会话摘要，避免把别人的记忆喂进来
            scoped = bool(self._cfg("inject_scope_subject", True))
            # 按 QQ 认人，不是按名字。群聊里 get_sender_name() 给的是群名片，
            # 同一个人在 A 群叫「李四」、在 B 群叫「赵六」，按名字查记忆必然空包。
            # 这里也是「主人的小号能不能被认出来」唯一的落点：号认得就认得。
            who = _speaker_key(self.store, event) if scoped else ""
            # 人物归一：名字是旧昵称、外号、群名片时，落到同一个人身上
            if scoped and who:
                try:
                    resolved = self.store.person_resolve(who)
                    if resolved:
                        who = resolved
                except Exception:
                    logger.debug("[SugarMemory] 人物归一失败", exc_info=True)
            # 认不出发言人时宁可不注入，也不能退化成全库检索。
            # 这里刻意不再拿 inject_enabled 当总闸：影子模式（关着注入观察）
            # 期间也要跑一遍检索，否则提及计数整个停摆，面板上整库都是「❄ 冷」。
            # 注入默认开启；手动切成影子模式观察记录准不准时，提及计数照跑，
            # 免得几天后看到一份「这些记忆都没人提」的假报告。
            if not (scoped and not who):
                query = event.message_str or ""
                budget = int(self._cfg("inject_budget_tokens", 1200) or 1200)
                sess = self._session_key(event) if scoped else ""
                bot = self._iso(self._event_bot(event))
                mode = self._inject_mode()
                inject_on = bool(self._cfg("inject_enabled", True))
                # frozen 且开着注入时，统计轨的结果完全不进 prompt，只为面板
                # 提及计数服务 → 走 lite：只跑 L2 相关性打分，跳过 l3(60)/
                # 骨架(2000)/outbound/cross 那一堆查询。否则每轮要付两次
                # 全量 build 的钱，其中一次纯属浪费。
                # dynamic 对照 / 影子模式仍跑全量：前者要拿它去注入，后者
                # 只剩统计这一个价值，追问日志也还得看。
                lite = (mode == "frozen" and inject_on)
                # 这轮对话点到了别人，就把那人的档案一起带上（统计轨用）：
                # 「B 是不是说过他叫 X」这种问题，注入轨里没有 B 的档案，
                # 得靠 memory_search 按需查；这里带上只为正确统计提及度。
                extra: list[str] = []
                if scoped and who:
                    try:
                        extra = self.store.subjects_mentioned(query, exclude=who, limit=3)
                    except Exception:
                        logger.debug("[SugarMemory] 识别被提及的人失败", exc_info=True)
                # 最近几轮用户原话只给检索做指代补位。先取上一轮为止的历史，
                # 再把本轮追加进去（本轮自己就是 query，不重复参与）。
                hist = list(self._recent_queries.get(sess) or [])
                self._recent_queries[sess] = (hist + [query])[-6:]
                if len(self._recent_queries) > INJECT_CACHE_MAX:
                    for k in list(self._recent_queries)[:-INJECT_CACHE_MAX]:
                        self._recent_queries.pop(k, None)
                # ── 统计轨 ──────────────────────────────────────────────
                # 按本轮原话 + 最近几轮实时检索。除了提及计数，它还产出
                # 「话题命中信号」，决定这一轮要不要增量刷新常驻包（B 方案）。
                builder = self._pack_builder(subject=who, session_id=sess,
                                             extra_subjects=extra, lite=lite,
                                             bot_id=bot)
                pack = builder.build(query=query, budget_tokens=budget,
                                     recent_queries=hist)
                # 语义兜底：只在字面/概念/别名全零命中、且用户明确填了
                # provider ID 时才真的调向量接口。分数在 async 上下文算完，
                # pack 层纯查表重跑一次，绝不在同步链路里等网络。
                semantic_scores: dict = {}
                pool = pack.get("_l2_pool") or []
                if (self._semantic_ready() and pool
                        and not any(int(r.get("_rel_content") or 0) > 0
                                    for r in pool)):
                    semantic_scores = await self._semantic_scores(query, pool)
                    if semantic_scores:
                        builder = self._pack_builder(
                            subject=who, session_id=sess, extra_subjects=extra,
                            lite=lite, bot_id=bot,
                            semantic_scores=semantic_scores)
                        pack = builder.build(query=query, budget_tokens=budget,
                                             recent_queries=hist)
                try:
                    self._last_why = {
                        int(k): dict(v) for k, v in (pack.get("why") or {}).items()}
                except Exception:
                    self._last_why = {}
                # 提及计数：只数「内容真的和这轮话对上」的事实（pack 打了
                # _count_hit 标记）。点名带出的档案、预算兜底硬塞的不算。
                try:
                    hit_ids = [r.get("id") for r in pack.get("l2") or []
                               if r.get("id") and r.get("_count_hit")]
                    if hit_ids:
                        asyncio.create_task(self._bump_hits(hit_ids))
                except Exception:
                    logger.debug("[SugarMemory] 提及计数失败", exc_info=True)
                # 追问命中时留一行 INFO：这个功能全是隐式的（不进包也是静默的），
                # 出问题时只能靠这行日志判断「到底补没补、补的是哪条」。
                if pack.get("outbound_away_added"):
                    ids = "、".join(f"#{x.get('id')}" for x in pack["outbound_away_added"])
                    logger.info(f"[SugarMemory] 追问命中：补带跨会话记录 {ids} 进记忆包")
                # ── 注入轨 ──────────────────────────────────────────────
                # 影子模式只跑到统计轨为止：算计数、不拼块、不注入。
                # 注入总闸关着时关系快照也不许漏进 system_prompt
                # （inject_relations 只管功能，不是注入口子）。
                if inject_on:
                    rel = ""
                    if self._cfg("inject_relations", False):
                        rel = self._relation_block(self._sender_name(event))
                    if mode == "dynamic":
                        # 旧路径（A/B 对照 / 紧急回退）：按本轮原话实时检索，
                        # 每轮都变、前缀缓存基本不命中。铁律不变，仍只写 system_prompt。
                        mem = build_block(pack)
                        block = "\n\n".join(x for x in (mem, rel) if x)
                        pack_tok, rel_tok = estimate_tokens(mem), estimate_tokens(rel)
                        meta = {
                            "fp": hashlib.sha256(block.encode("utf-8")).hexdigest()[:16],
                            "rebuilt": True, "rounds": 0, "reason": "dynamic",
                            "cfg_fp": self._cfg_fp(), "mode": "dynamic",
                            "arch_ver": "dynamic",
                        }
                    else:
                        block, pack_tok, rel_tok, meta = self._frozen_block(
                            who=who, sess=sess, budget=budget, rel_text=rel,
                            bot_id=bot,
                            topic_ids=pack.get("topic_ids") or [],
                            topic_signal=bool(pack.get("topic_signal")),
                            topic_query=query, recent_queries=hist,
                            semantic_scores=semantic_scores)
                    if block:
                        inject(req, block)  # 铁律：只写 system_prompt
                        logger.debug(
                            f"[SugarMemory] {mode} 记忆包 ~{pack_tok + rel_tok} tokens")
                        # 把这一轮挂的是哪份前缀留给 on_llm_response，让真实
                        # cached_tokens 能和「同 fp 连续轮 / 重建轮」对上。
                        self._last_inject_state[sess] = dict(meta)
                        if len(self._last_inject_state) > INJECT_CACHE_MAX:
                            for k in list(self._last_inject_state)[:-INJECT_CACHE_MAX]:
                                self._last_inject_state.pop(k, None)
                        self._record_tokens("pack", input=pack_tok, source="estimate",
                                            subject=who, session_id=sess)
                        if rel_tok:
                            self._record_tokens("relation", input=rel_tok,
                                                source="estimate",
                                                subject=self._sender_name(event),
                                                session_id=self._session_key(event))
        except Exception:
            logger.exception("[SugarMemory] 注入异常")

    def _resident_l2_n(self) -> int:
        """常驻前缀到底放几条长期事实。

        `pack_l2_limit` 是「最多能带几条」，而这一项决定「常驻位只放几条」：
        剩下的事实不进每一轮的底档，但仍然留着，聊到相关话题时从尾部召回。

        为什么不直接用 pack_l2_limit：那个值同时决定 l2_keep_min 和裁剪保底，
        调小会连带削弱预算不足时的保护。这里只收缩常驻长度，语义更干净。

        默认 0 = 不额外限制（等同 pack_l2_limit，即现有行为）。设成 6 就是
        「常驻只带 6 条身份/习惯，其余全部改成按话题召回」。
        """
        try:
            l2_limit = max(1, int(self._cfg("pack_l2_limit", 12) or 12))
        except (TypeError, ValueError):
            l2_limit = 12
        try:
            n = int(self._cfg("inject_l2_resident_limit", 0) or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 0 or n >= l2_limit:
            return l2_limit
        return max(1, n)

    def _frozen_block(self, who: str, sess: str, budget: int,
                      rel_text: str = "", bot_id: str = "",
                      topic_ids: list | None = None,
                      topic_signal: bool = False,
                      topic_query: str = "",
                      recent_queries: list[str] | None = None,
                      semantic_scores: dict | None = None,
                      preview: bool = False
                      ) -> tuple[str, int, int, dict]:
        """按 (who, sess) 冻结的常驻记忆包。

        返回 (完整块, 包 token, 关系 token, meta)；meta 记录本轮实际使用的
        前缀 fp / 是否重建 / 连续轮数 / 重建原因 / 字节是否未变，交给
        on_llm_response 跟真实 cached_tokens 一起落库。

        2026-11-06 A 方案（用户定稿：绝不碰用户角色槽位，只写 system_prompt）：
        旧实现把 `build(query=用户原话)` 的结果拼到 system_prompt 尾部，而框架
        把 system_prompt 插成 messages[0]、后面接整段历史——等于把每轮都变的
        内容放在了**整个请求的最前面**，前缀缓存从第一个 token 就分叉。
        旧的 inject_dedup 只在「内容一模一样」时复用，可流水型记录（转述、
        定时、跨会话近况）每轮都在变，等于永远不命中（实测命中率 21.7%，
        51.8% 的调用 cached=0）。

        现在 system 里放 `query=""` 的常驻底档：L2 事实、本会话摘要、
        常驻配额内的转述/定时/骨架，按置信度与稳度排。B 方案在这份底档上
        加了一层「话题触发增量」：本轮真的聊到包外相关事实、或出现「那它呢 /
        上次 / 刚才」这类指代追问时，才重建这一轮，把那几条补进来（替换掉
        最弱的常驻条，包不撑大）；话题指纹不变就一直复用同一份，闲聊轮次
        一个字节都不动。没有信号时绝大多数对话仍然吃满前缀缓存。

        重建判据只有一条（0.16.15 起）：这一轮算出来的块，跟上一轮真正挂出去的
        是不是同一串字节。一样就原样复用（连时间都不判，聊一百轮也不动）；
        不一样就重建。没有时间 / 轮数节流——B 方案的召回是纯追加在整块最末尾，
        前缀段（前 prefix_n 条 L2 + 关系快照前段）字节不动，重建只重算尾部几条，
        硬等 180 秒对缓存没有收益，只会让新记忆迟到。
        关系快照和记忆包合成同一个块一起冻结——它单独变也会废掉整段前缀。
        """
        bot = self._iso(bot_id)
        # 缓存 key 带上机器人：同一个会话 id 在不同平台实例下可能重名，
        # 不隔离的话 A 机器人的常驻包会被 B 直接复用（连重建都不会触发）。
        key = f"{bot}|{who}|{sess}" if bot else f"{who}|{sess}"
        now = time.time()
        enabled = bool(self._cfg("inject_dedup", True))
        cfg_fp = self._cfg_fp()

        def _int_cfg(k, default):
            try:
                return max(0, int(self._cfg(k, default)))
            except (TypeError, ValueError):
                return int(default)

        l2_limit = max(1, _int_cfg("pack_l2_limit", 12))
        # 常驻前缀实际条数：默认 = l2_limit；配置项收缩后只带前几条身份/习惯，
        # 其余事实退到「聊到再召回」，不再每轮都占着 system 的固定位置。
        resident_n = self._resident_l2_n()
        # 常驻包不能沿用 pack_l2_keep_min（默认 3）：query="" 时
        # _filter_by_relevance 走零命中的保底分支，会把档案砍到 3 条。
        # 把保底抬到 l2_limit，让它按置信度/稳度带满常驻条目。
        sb = self._pack_builder(subject=who, session_id=sess,
                                extra_subjects=None,
                                l2_keep_min_override=resident_n,
                                bot_id=bot, semantic_scores=semantic_scores)
        # 常驻底档永远 query=""：不传 recent_queries，store 就走旧的置信度
        # 排序，字节级稳定。B 方案的增量只叠加在这一份之上。
        pack = sb.build(query="", budget_tokens=budget)
        # l2_sig 只认常驻底档，绝不把本轮话题增量算进去——否则话题结束切回
        # 纯常驻包时会被误判成「用户刚改了记忆」，立刻重建、缓存白费。
        base_l2 = list(pack.get("l2") or [])
        l2_sig = hashlib.sha256("|".join(
            f"{r.get('id')}:{r.get('value')}" for r in base_l2
        ).encode("utf-8")).hexdigest()[:16]

        # ---- B 方案（volatile-tail 变体，0.16.8）------------------------
        # 冻结块的 L2 分两段，前缀缓存与召回灵敏度兼得：
        #   prefix = 整份 base_l2（按置信/稳度排，字节级稳定，跨轮不变）
        #            → 前缀缓存命中的主体，占 token 绝对大头；
        #   tail   = 本轮话题召回 tail_k 条，纯追加在整块的最末尾
        #            （build_block 里 _topic_add 单独渲染到最后），前面不动。
        # 闲聊时 tail 退化回 base 尾部低分条 → 整块字节=底档 → 全命中；
        # 聊到相关事实时它出现在尾部，召回贴近动态包，且前缀照旧命中。
        # 这是用户定稿的设计：对比上一次记忆包，把低分记忆替换为新的高分
        # 召回记忆、拼在后面，其他不变——既保前缀缓存，又不再「死板召不回」。
        refresh_on = bool(self._cfg("inject_topic_refresh", True))
        tail_k = max(0, _int_cfg("inject_topic_extra", 3))
        # 纯应答语（「ok我明白了」「可以的」「哈喽」）按闲聊处理：它们没有任何
        # 话题内容，bigram 只能蹭到一两个虚词，捞回来的都不是这轮在聊的事。
        if topic_query and _is_noise_reply(topic_query):
            topic_query = ""
            recent_queries = []
            topic_ids = []
            topic_signal = False
        signal_ids = [int(i) for i in (topic_ids or []) if i][:8]
        # 底档整份进「固定前缀」。sb.build 已经按 resident_n 出条（l2_keep_min
        # 就是它），所以这里 prefix_n 直接等于 resident_n。以前是
        # prefix_n = l2_limit - tail_k，把最后 3 条底档让出来当「尾部槽位」，
        # 可拼装实际是纯追加（底档一条都不删），等于白让出 3 个前缀名额。
        prefix_n = resident_n
        base_l2_sorted = list(pack.get("l2") or [])
        # 前缀严格取底档前 prefix_n 条，一个字节都不动——这是前缀缓存命中的主体。
        prefix = base_l2_sorted[:prefix_n]
        # 底档尾部（含「自我相关 / 其他」等永久事实）全部原样保留，绝不为了塞话题
        # 增量把它们挤掉。永久事实再打上 _extra_pin，让 trim_pack 超预算时也不砍。
        tail_base = base_l2_sorted[prefix_n:]
        for r in base_l2_sorted:
            if str(r.get("predicate") or "").strip() in PERMANENT_PREDICATES:
                r["_extra_pin"] = True
        prefix_ids = {int(r.get("id") or 0) for r in prefix}
        keep_ids = prefix_ids | {int(r.get("id") or 0) for r in tail_base}
        add_ids: list[int] = []
        recalled: list[dict] = []
        # 上一轮真正挂出去的那几条：本轮没召回到新东西就原样沿用。否则块会从
        # 15 条缩回 12 条——尾巴凭空消失，等于每轮白重算一次尾部缓存。
        prev_ent = None if preview else self._inject_cache.get(key)
        if refresh_on and tail_k > 0 and (
                topic_query or rel_text or recent_queries
                or signal_ids or topic_signal):
            # 候选包给足预算：它只负责「按本轮话题捞出候选」，真正的裁剪由
            # 下面外层 trim_pack 统一做。这里若也按 budget 裁一次，低置信但
            # 本轮高度相关的事实会先被砍掉，增量永远补不进来（实测：预算 300
            # 时话题刷新 reason=same_fp、答案一条没进包）。
            tb = self._pack_builder(subject=who, session_id=sess,
                                    extra_subjects=None, bot_id=bot,
                                    semantic_scores=semantic_scores)
            topic_pack = tb.build(query=(topic_query or rel_text),
                                  budget_tokens=max(int(budget), 4096),
                                  recent_queries=recent_queries)
            # 从全量候选池挑，而不是已被 _filter_by_relevance / _pick_l2
            # 砍过的 l2：相对阈值偶尔会误杀短答案，池子里那条还在。
            cands = [r for r in (topic_pack.get("_l2_pool")
                                 or topic_pack.get("l2") or [])
                     if int(r.get("id") or 0) not in keep_ids
                     and (r.get("_sem")
                          or int(r.get("_rel_content") or 0) > 0
                          or int(r.get("_rel") or 0) > 0
                          # 指代（「那它呢」）和问法别名召回的行 _rel 恒为 0，
                          # 只认 _rel 会把它们挡在尾部之外 —— 那正是这两个通道
                          # 存在的意义（实测「那它呢」整句零 _rel，答案在 _rel_ctx）。
                          or int(r.get("_rel_ctx") or 0) > 0
                          or int(r.get("_rel_alias") or 0) >= 2)]
            cands.sort(key=lambda r: (float(r.get("_rel_w") or 0),
                                      float(r.get("_score") or 0)), reverse=True)
            # 同一件事被总结成好几条时（#9「开发了记忆插件」/#45「开发了记忆
            # 系统插件」/#73「开发并修复了多个插件」），尾部只有 3 个名额，
            # 全是近重复等于白补。复用主包那套重叠系数去重，阈值放软到 0.6
            # ——尾部本来就是给「换个说法」用的，主包那档 0.7 太严。
            if len(cands) > 1:
                cands = tb._apply_mmr(cands, lam=0.6)
            ceiling = max((float(x.get("_score") or 0) for x in prefix), default=1.0)
            recalled = []
            # 只为凑满 tail_k 抓低分噪声的坑（2026-11-08）：实盘「xulaoya 和我的
            # 关系」那轮，#24/#40 已经在底档里被 keep_ids 排除，剩下真命中的只有
            # #56（rel=6），后面两条 rel=1 的（插件对比、抖音点赞）纯粹蹭到
            # 「关系」这两个字，被凑数塞进尾部——用户一眼看出是噪声。
            # 判据：不是第一条、只蹭到 1 个词、分数又不到最高分的 55% → 不要。
            # 不设「至少几条」的保底：宁缺毋滥，本轮真没关系就只带一条。
            top_w = float(cands[0].get("_rel_w") or 0) if cands else 0.0
            for r in cands:
                if len(recalled) >= tail_k:
                    break
                w = float(r.get("_rel_w") or 0)
                if recalled and int(r.get("_rel") or 0) <= 1 and w < top_w * 0.55:
                    continue
                idx = len(recalled)
                r = dict(r)
                r["_topic_add"] = True
                # 量纲抬到常驻包之上，保证 trim_pack 不会把它当最弱裁掉；
                # build_block 按列表顺序渲染（不按分数），尾部位置由这里定。
                r["_score"] = ceiling + 0.5 + 0.01 * (tail_k - idx)
                recalled.append(r)
                add_ids.append(int(r.get("id") or 0))
            if not recalled and prev_ent:
                # 本轮没捞到新东西，但旧尾巴要不要留，得看是不是还在这条话题上。
                # 判据：这一轮有没有任何内容命中（_rel>0 只算本轮本体，不算
                # 历史词和问法别名）。
                #   零命中 = 这句是应答/寒暄/情感表达（「嘿嘿想你了」「啊？我
                #   有点没懂」），接着上一条说是对的 → 沿用，块字节不变还能
                #   直接命上轮缓存。
                #   有命中却跟旧尾巴无关（上轮聊「插件开发」，这轮「吃东西」）
                #   = 真换话题 → 清掉。尾部追加在块末，清掉不伤前缀缓存，
                #   挂着反而让 AI 以为用户这轮还在说插件。
                pool_now = topic_pack.get("_l2_pool") or []
                if not any(int(r.get("_rel") or 0) > 0 for r in pool_now):
                    recalled = [dict(r) for r in (prev_ent.get("tail_rows") or [])]
                    for r in recalled:
                        r["_topic_add"] = True
                    add_ids = [int(r.get("id") or 0) for r in recalled]
            # 纯追加：prefix（原样）+ 底档尾部（原样，含永久）+ 本轮召回。
            # 前缀零改动 → 前缀缓存命中；永久事实一条不丢；块最多长 tail_k 条，
            # 超预算时由 trim_pack 砍最弱的非保护条（永久/话题条都已打 pin）。
            pack["l2"] = prefix + tail_base + recalled
            # 不同步任何 query 驱动的字段：build_block 会拿 query 折叠
            # 多子项 value、筛昵称映射子集，probe_outbound 也随问法开关。
            # 任何一个混进来，同一话题换种说法块字节就变，前缀缓存白废。
            # 话题增量只改 L2 条目集合，其余一律保持 query="" 的底档口径。
            pack = trim_pack(pack, budget,
                             l3_keep_min=max(0, _int_cfg("pack_l3_keep_min", 0)),
                             l2_keep_min=l2_limit)
        # 指纹只认「真的补进去了什么」。统计轨给的 ids 里大半是常驻包本来
        # 就有的条目、纯指代追问也可能一条都补不到——拿它们当刷新理由会
        # 白废一次前缀缓存，还什么都没捞回来。
        topic_sig = ""
        if add_ids:
            raw = "|".join(str(i) for i in sorted(add_ids))
            topic_sig = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        mem = ""
        # 只有外发/跨会话行、没有 l2/l3/摘要时也要注入——用户跨会话提问时
        # 这两样可能就是包里唯一有信息量的东西。
        if (pack.get("l2") or pack.get("l3") or pack.get("summary")
                or pack.get("outbound") or pack.get("cross")
                or pack.get("skeleton")):
            mem = build_block(pack)
        if preview:
            # 旁路：把这一轮真正渲染的 pack 留给 _api_pack 做结构化展示
            # （面板要按「哪些是尾部追加」逐条列 L2），绝不写运行缓存。
            self._last_preview_pack = pack
        candidate = "\n\n".join(x for x in (mem, rel_text) if x)
        fp = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
        pack_tok = estimate_tokens(mem)
        rel_tok = estimate_tokens(rel_text)

        def _meta(used_fp: str, rebuilt: bool, rounds: int, reason: str,
                  unchanged: bool = False, prefix: str | None = None) -> dict:
            # fp 必须是**实际挂进 system 的那块**的指纹。节流期复用旧块时
            # 记新候选的 fp 会把「同 fp 连续轮」错算成重建轮，归因就废了。
            # unchanged：重建了内存块，但字节和上一轮完全一样（重载冷启动），
            # 记忆包并没漂移，别把它算成失败样本。
            # prefix：常驻底档自己的指纹（l2_sig）。整块 fp 含本轮话题尾部，
            # 换话题必变；面板判「冻结失效」要看这一项，而不是整块 fp。
            return {"fp": used_fp, "rebuilt": bool(rebuilt), "rounds": int(rounds),
                    "reason": reason, "cfg_fp": cfg_fp, "unchanged": bool(unchanged),
                    "topic_sig": topic_sig,
                    "prefix_sig": str(prefix if prefix is not None else l2_sig),
                    "mode": "frozen", "arch_ver": "frozen"}

        ent = self._inject_cache.get(key)
        if preview:
            # 预览模式：只读模拟，绝不读/写运行缓存（否则「看一眼预览」就会把
            # 节流状态/缓存块改掉，污染真实注入链路）。
            ent = None
        used_rounds = 0
        # 配置版本不一致 → 单独判，绝不让改 pack_l2_limit 之类的假 l2_sig
        # 变化伪装成「用户刚改了记忆」（important）而立刻重建全部分片。
        cfg_changed = bool(ent and ent.get("cfg_fp") != cfg_fp)
        if not enabled or ent is None or cfg_changed:
            rebuild = True
            reason = ("config" if cfg_changed else
                      "dedup_off" if not enabled else "cold")
        else:
            ent["rounds"] = int(ent.get("rounds", 0)) + 1
            used_rounds = ent["rounds"]
            if fp == ent["fp"]:
                # 一字未变：直接复用，连时间都不用看（聊一百轮也不动）。
                # 不刷新 built_at：它记的是「这块第一次挂出去的时刻」，面板靠它
                # 显示真实年龄；每轮刷新会让「已冻结多久」永远显示 0。
                ent["topic_sig"] = topic_sig
                ent["topic_ids"] = list(add_ids)
                return (ent["block"], ent["pack_tokens"], ent["rel_tokens"],
                        _meta(str(ent["fp"]), False, used_rounds, "same_fp",
                              prefix=ent.get("l2_sig")))
            # fp 变了 = 内容确实变了 → 重建。0.16.15 起这里不再判时间/轮数：
            # B 方案的召回是纯追加在整块最末尾，prefix 段（前 prefix_n 条 L2 +
            # 关系快照前段）一个字节都不动，重建只重算尾部几条，硬等 180 秒
            # 对缓存没有收益，只会让新记下的东西迟到。「L2 底档本身变了」单独
            # 归因（l2_changed），其余统称 fp_changed。
            reason = ("l2_changed" if l2_sig != ent.get("l2_sig") else "fp_changed")

        unchanged = False
        if reason == "cold":
            # 冻结块只在内存里：插件重载 / 进程重启会把它清空，于是下一轮
            # 走 cold 重建。但重建出来的字节常常和上一轮一模一样 ——
            # provider 侧的前缀并没变，只是我们重新算了一遍。
            # 从库里取上一轮真实挂过的 fp 比对，把这种「假重建」标出来：
            # 它必然要重新写一次缓存，0% 是固定正常，不是记忆包漂移。
            try:
                prev_fp = self.store.token_last_inject_fp(sess)
            except Exception:
                prev_fp = ""
            if prev_fp and prev_fp == fp:
                reason = "reload"
                unchanged = True
                logger.info(
                    "[SugarMemory] 重载冷启动：常驻包字节未变（fp=%s），"
                    "本轮整轮缓存命中率 0%% 属固定正常，不计入失败样本", fp)
        # 容量上限：按最久未重建淘汰，别让 (人,会话) 分片无界增长。
        if (key not in self._inject_cache
                and len(self._inject_cache) >= INJECT_CACHE_MAX):
            victim = min(self._inject_cache,
                         key=lambda k: float(
                             self._inject_cache[k].get("built_at", 0)))
            self._inject_cache.pop(victim, None)
        if not preview:
            self._inject_cache[key] = {
                "block": candidate, "fp": fp, "l2_sig": l2_sig,
                "topic_sig": topic_sig, "topic_ids": list(add_ids),
                "tail_rows": [dict(r) for r in recalled],
                "built_at": now, "rounds": 0, "cfg_fp": cfg_fp,
                "pack_tokens": pack_tok, "rel_tokens": rel_tok,
            }
        logger.debug(
            f"[SugarMemory] 常驻包重建（{reason}）：{key} ~{pack_tok + rel_tok} tokens")
        return candidate, pack_tok, rel_tok, _meta(fp, True, 0, reason, unchanged)

    # ---------- L3 事件记录 ----------

    async def _bump_hits(self, fact_ids: list):
        """后台更新 L2 提及计数，绝不抛错打断对话。"""
        try:
            self.store.bump_l2_hits(fact_ids)
        except Exception:
            logger.debug("[SugarMemory] bump_l2_hits 失败", exc_info=True)

    def _cleanup_async(self):
        """cleanup 放后台线程跑。积压场景一次能吞几万行（实测 8.6 万行 19 秒），
        同步跑会卡死整个事件循环；后台线程 + store 内部分批放锁，消息处理
        最多等一批的耗时。节流（每小时一次）和防重入都在这里。"""
        if self._cleaning:
            return
        self._cleaning = True

        async def _run():
            try:
                await asyncio.to_thread(self._cleanup)
            except Exception:
                logger.debug("[SugarMemory] cleanup 失败", exc_info=True)
            finally:
                self._cleaning = False

        asyncio.create_task(_run())

    def _cleanup(self):
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        # 原文过期先压成一行主题词骨架再删（可关：关掉 = 到期直接删的原行为）。
        # 骨架行自带更长的过期时间，到点后由后续清理自然回收。
        skeleton_fn = None
        if self._cfg("l3_skeleton_enabled", True):
            try:
                days = max(1, int(self._cfg("skeleton_retention_days", 90) or 90))
            except (TypeError, ValueError):
                days = 90
            keep_until = now + days * 86400

            def skeleton_fn(rows, _ku=keep_until, _now=now):
                return make_skeleton(rows, keep_until=_ku, now=_now)
        self.store.cleanup_expired(now=now, skeleton_fn=skeleton_fn)
        # 长期记忆的沉底归档：只改 status='archived'，物理数据一条不删。
        # 门槛很保守（保留度 <2% 且静默 90 天以上），跑一次通常一条都不动。
        try:
            res = self.store.cleanup_decay(now=now)
            if res.get("archived"):
                logger.info(
                    f"[SugarMemory] 沉底归档 {res['archived']} 条长期记忆（可恢复）："
                    f"{'、'.join(res.get('subjects') or [])}"
                )
        except Exception:
            logger.debug("[SugarMemory] 沉底归档失败", exc_info=True)
        # 称呼的「小型遗忘」：叫错的名字（tier 3）久不提就删，老外号同理。
        # 顶层两种（显示名 / 能被称呼的名）永不自动删——那是身份不是装饰。
        try:
            gdays = max(1.0, float(self._cfg("alias_guess_retention_days", 90) or 90))
            ndays = max(1.0, float(self._cfg("alias_nick_retention_days", 365) or 365))
        except (TypeError, ValueError):
            gdays, ndays = 90.0, 365.0
        if self._cfg("alias_cleanup_enabled", True):
            try:
                res = self.store.alias_cleanup(guess_days=gdays, nick_days=ndays, now=now)
                if res.get("removed"):
                    logger.info(
                        f"[SugarMemory] 忘掉 {res['removed']} 个太久没提起的称呼："
                        f"{'、'.join(res.get('aliases') or [])}"
                    )
            except Exception:
                logger.debug("[SugarMemory] 清理称呼失败", exc_info=True)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def record_message(self, event: AstrMessageEvent):
        try:
            # 原话缓存放最前面：不管记不记录、是不是群聊，工具校验都要用到它
            try:
                sess_key = self._session_key(event)
                self._remember_text(sess_key, (event.message_str or "").strip())
                self._mark_active(sess_key)
            except Exception:
                pass
            # 群也要记：就算群聊记忆是关的，也得知道这个号是群不是人
            self._remember_group(event)
            if not self._cfg("record_enabled", True):
                return
            # 群聊只读：不写 L3 → 不加好感度、不计数、不触发总结。注入钩子不受影响。
            if self._is_group(event) and not self._cfg("record_in_group", False):
                return
            text = re.sub(r"\s*\[MSG_ID:[^\]]*\]\s*$", "", (event.message_str or "").strip()).strip()
            sender = event.get_sender_id()
            if sender and str(sender) == str(event.get_self_id() or ""):
                return
            # 没有正文的消息（图片、表情、语音、纯引用图）以前在这里直接 return，
            # 连「这是谁」都不认。结果只在表情消息里出现过的人，people 表里永远
            # 没有 QQ——面板上是空壳，总结只能从别人的话里猜他是谁（Jerry/小杰
            # 就是这么变成「未绑 QQ」的）。发消息这个动作本身就是身份证据，
            # 先认人建档，正文该不该记另说。
            if not text:
                self._identify_quietly(event, sender)
                return
            if text.startswith("/"):
                return
            # 统一走 _sender_name：它做过 strip。之前这里直接用 get_sender_name()，
            # 昵称带前导空格时会存成「 李四…」，而注入端用的是 strip 过的名字，
            # 写入键和查询键对不上，同一个人被拆成两个节点、记忆还查不出来。
            name = self._sender_name(event) or str(sender)
            # 定时唤醒不是人说的话，记进去会污染总结和好感度
            if self._is_pseudo_sender(name, text):
                # 顺手清掉 _last_sender 里可能残留的伪发送者，
                # 否则紧接着的 bot_reply 会拿它当归属人，接着往库里写
                try:
                    self._last_sender.pop(self._session_key(event), None)
                except Exception:
                    pass
                logger.debug(f"[SugarMemory] 跳过系统唤醒伪消息：{text[:40]}")
                return
            msg_id = str(getattr(event.message_obj, "message_id", "") or "")
            expires = self._l3_expiry()
            scope = "群" if event.get_group_id() else "私"
            session = self._session_key(event)
            # QQ 是身份锚：同一个人改昵称、换马甲，都要落到同一个 person 上。
            # 以前这里直接拿昵称当 subject，张三把昵称改成 Moon 之后，
            # 同一个号的话被写成两个 subject，L2 分成两套、关系图上是两个人。
            person = name
            try:
                # 群里拿到的是**群名片**，那只是这个人在某一个群里的马甲，不是
                # 他自己的昵称——降成别名（tier 2），不许它覆盖私聊拿到的 QQ
                # 昵称（tier 0）。否则同一个人换几个群，面板上的名字就跟着乱跳。
                person = self.store.person_ensure(
                    name, qq=str(sender or ""), source="event",
                    tier=0 if scope == "私" else 2) or ""
            except Exception:
                logger.debug("[SugarMemory] 人物归并失败", exc_info=True)
                person = name
            # 名字是「用户」「对方」这类泛称时 person_ensure 会拒绝造人。
            # 这时退回用 QQ 当主体，免得 L3 里又长出一个认不回来的假人。
            if not person or is_generic_name(person):
                person = f"qq{sender}" if sender else ""
                name = name or person
            self._remember_sender(session, person)
            # 用户亲口说「用户丙 就是丙哥」时当场学着，不用等总结去抓。
            # 纯正则，不调模型；学到的称呼会立刻绑到那个人身上，同一个人不再裂成两个。
            try:
                learned = self.store.person_learn_aliases(text)
                for item in learned:
                    logger.info(
                        f"[SugarMemory] 学到称呼：{item.get('alias')} 也是「{item.get('keep')}」")
            except Exception:
                logger.debug("[SugarMemory] 学称呼失败", exc_info=True)
            self.store.write_l3(
                event_type="user_message",
                subject=person,
                # 正文写当次的昵称（模型看到的是真实叫法），subject 用稳定的人
                content=f"[{scope}] {name}: {text}"[:300],
                source_msg_id=msg_id,
                session_id=session,
                expires_at=expires,
            )
            self._cleanup_async()
            self._gain_affinity(person, qq=str(sender or ""))
            if self._cfg("summarize_enabled", True):
                interval = max(1, int(self._cfg("summary_interval_messages", 50) or 50))
                # 计数从库里数（"上次总结之后的用户消息条数"），不再靠内存计数器：
                # 插件重载/进程重启都不会把进度清零，只有总结成功才归零。
                try:
                    since = self.store.last_summary_at(session)
                    n = self.store.count_user_messages_since(session, since)
                except Exception:
                    n = int(self._msg_count.get(session, 0)) + 1
                self._msg_count[session] = n
                if len(self._msg_count) > 500:
                    for k in [k for k, v in self._msg_count.items() if not v]:
                        self._msg_count.pop(k, None)
                if n >= interval:
                    self._pending.add(session)
                if session in self._pending:
                    self._schedule_summary(session)
        except Exception:
            logger.exception("[SugarMemory] L3 记录异常")

    @filter.after_message_sent()
    async def record_bot_reply(self, event: AstrMessageEvent, *args, **kwargs):
        try:
            if not self._cfg("record_enabled", True):
                return
            # 群聊只读：AI 回复同样不落库，不攒总结。
            if self._is_group(event) and not self._cfg("record_in_group", False):
                return
            result = event.get_result()
            if result is None or result.chain is None:
                return
            text = "".join(
                str(getattr(c, "text", "") or "") for c in result.chain
            ).strip()
            if not text:
                return
            session = self._session_key(event)
            who = ""
            hit = self._last_sender.get(session)
            if hit and time.time() - hit[0] <= 600:
                who = hit[1]
            if not who:
                try:
                    sender_obj = getattr(getattr(event, "message_obj", None), "sender", None)
                    who = str(getattr(sender_obj, "nickname", "") or "")[:40]
                except Exception:
                    who = ""
            # 对定时唤醒的回复同样不落库：那头根本没有真人，留着只会让总结
            # 以为 AI 主动说了这些话
            if who and str(who).strip() in self._ignore_senders():
                logger.debug(f"[SugarMemory] 跳过对系统唤醒的回复：{text[:40]}")
                return
            expires = self._l3_expiry()
            self.store.write_l3(
                event_type="bot_reply", subject=who, content=f"[你] {text}"[:300],
                session_id=session, expires_at=expires,
            )
            if session in self._pending:
                self._schedule_summary(session)
        except Exception:
            pass

    # ---------- 主动发送（跨会话转述）---------- #

    def _hook_context_send(self):
        """给 context.send_message 套一层，把 AI 主动发到别的会话的话也记进记忆。

        为什么必须 hook 而不是靠 after_message_sent：xrelay 这类跨会话转述走的是
        `context.send_message`，压根不经过「回复当前事件」的发送链路，record_bot_reply
        一次都不会触发。结果就是你让 AI 在 A 会话给 B 带句话，切回 A 问「你刚才说了啥」，
        库里零记录，它只能按上下文猜——猜不着就否认。用户看到的就是
        「它以为自己转述的话是假的」。

        只加记录、不改原行为：原函数的返回值原样透传，异常也照抛。
        """
        try:
            ctx = self.context
            orig = getattr(ctx, "send_message", None)
            if orig is None or getattr(orig, "_sugarmemory_wrapped", False):
                return
            plugin = self

            async def _wrapped(session, message_chain):
                result = await orig(session, message_chain)
                try:
                    tag = getattr(message_chain, "_sugarmemory_relay", None)
                    plugin._record_outbound(
                        session, message_chain, ok=result is not False, tag=tag
                    )
                except Exception:
                    logger.debug("[SugarMemory] 记录主动发送失败", exc_info=True)
                return result

            _wrapped._sugarmemory_wrapped = True
            ctx.send_message = _wrapped
            # 留一份原函数，terminate 时还原（context 是所有插件共享的）
            self._orig_send_message = orig
            logger.info("[SugarMemory] 已挂载跨会话发送记录（转述/主动发送的话会进记忆）")
        except Exception:
            logger.warning("[SugarMemory] 挂载跨会话发送记录失败", exc_info=True)

    @staticmethod
    def _session_label(session) -> str:
        """把 unified_msg_origin 翻成人话：111:FriendMessage:123456002 → 「用户 123456002」"""
        s = str(session or "")
        parts = s.split(":")
        if len(parts) >= 3:
            kind, peer = parts[-2], parts[-1]
            if "Group" in kind:
                return f"群 {peer}"
            if "Friend" in kind or "Private" in kind:
                return f"用户 {peer}"
            return peer
        return s[:40] or "未知会话"

    def _outbound_cause(self, target_session: str, window: int = 180,
                        bot_id: str = "") -> str:
        """找出这次主动发送的起因：最近一条「不是发去那个会话」的用户原话。

        光记「我说了什么」只解决一半问题。AI 换个会话被问「你为什么发这个」时，
        它知道自己发过，但答不出为什么——因为动机在另一个会话里，人一换就断了。
        这里把「谁、什么时候、原话怎么说的」一起钉在记录上，隔几天再问也答得上来。

        为什么不取 bot_reply：AI 自己那句「好，我这就发」是它的承诺，不是起因；
        真正的动因永远在用户那句话里。

        bot_id：只在同一个机器人自己的对话里找起因。不加这一条，两个机器人
        共用一个库时，B 的定时任务会把 A 刚说的话当成因记下来。
        """
        if window <= 0:
            return ""  # 0 = 关掉起因功能，语义和配置 hint 保持一致
        try:
            rows = self.store.query_recent_l3(
                limit=12, event_type="user_message", bot_id=self._iso(bot_id))
        except Exception:
            return ""
        now = time.time()
        for r in rows:
            ts = float(r.get("created_at") or 0)
            # 结果按 id 倒序，一旦最老的那条都超窗，后面只会更老，直接收手
            if ts and window > 0 and now - ts > window:
                break
            sid = str(r.get("session_id") or "")
            if sid and sid == target_session:
                continue  # 目标会话自己刚说的话不是起因
            text = str(r.get("content") or "").strip()
            if not text:
                continue
            # "[私] 用户乙: 帮我给用户丙带句话" → "用户乙: 帮我给用户丙带句话"
            text = re.sub(r"^\[(私|群)\]\s*", "", text)
            return text[:80]
        return ""

    def _record_outbound(self, session, message_chain, ok: bool = True, tag=None) -> None:
        """记一条「AI 主动发给别人」的 L3，subject 留空让它跨人可见。

        tag 是转述工具（relay_message）打在 message_chain 上的显式标记：
        只有「用户要求的转述」才带这个标。有标 → 记成「转述」，起因直接用
        标记里带的本轮用户原话（比启发式搜索准）；无标 → 按配置决定要不要记
        （定时推送、插件自发通知这类不该被当成转述的，关掉就干净了）。
        """
        if not ok or not self._cfg("record_enabled", True):
            return
        if not self._cfg("record_outbound", True):
            return
        chain = getattr(message_chain, "chain", None) or []
        text = "".join(str(getattr(c, "text", "") or "") for c in chain).strip()
        if not text:
            return
        sess = str(session or "")[:120]
        if not sess:
            return
        # 这条话是「哪个机器人发出去的」。context.send_message 只有目标会话，
        # 没有发送方 event，所以按目标会话的 umo 第一段归属：同账号自发时就是
        # 自己；跨账号代发属于高级用法，归到接收方账号也不会污染第三个机器人。
        raw_bot = self._bot_of(sess)
        bot = self._iso(raw_bot)
        # 隔离开着、却从会话串里认不出发送方（没有平台实例前缀）时，宁可不记也
        # 不写进「共享池」：共享记录本来是留给老数据和手动补录的，新消息一旦落进
        # 去，所有机器人的记忆包都会读到它——那正好又回到「串记忆」这个老毛病上。
        if self._cfg("bot_isolation", True) and not raw_bot:
            logger.debug(f"[SugarMemory] 会话无机器人前缀，跳过外发记录：{sess[-24:]}")
            return
        is_relay = isinstance(tag, dict)
        if not is_relay and not self._cfg("record_outbound_other", True):
            return  # 非转述的主动发送，用户选择不记
        # 回声过滤：目标是「刚刚还有人说话」的会话、又没带标记 → 这是当前对话
        # 自己的发送路径（分段器的分段、send 工具的碎碎念）。这些内容要么已经
        # 以 bot_reply 落库、要么就在当前对话上下文里，再记一遍「主动发送」只会
        # 把 bot_outbound 通道堵死——真转述几秒钟就被挤出去，换会话后
        # 「你刚才发了什么」就查不到了。带标记的（转述/定时任务）永远照记。
        if not is_relay:
            try:
                dedup = max(0, int(self._cfg("outbound_dedup_seconds", 120) or 0))
            except (TypeError, ValueError):
                dedup = 120
            if dedup:
                last = self._active_sessions.get(sess)
                if last and time.time() - last <= dedup:
                    logger.debug(f"[SugarMemory] 跳过本会话回声外发：{sess[-20:]}")
                    return
        try:
            if is_relay:
                # 起因优先用标记里带的本轮原话；空了再退回启发式搜索
                cause = str(tag.get("cause") or "").strip()
                if not cause:
                    window = max(0, int(self._cfg("outbound_cause_window", 180) or 0))
                    cause = self._outbound_cause(sess, window, bot_id=bot) if window else ""
                # label 允许发送方自定义这句记录的开头（定时任务插件用它记成
                # 「你按定时任务「早安」发给群 xxx」，跟真转述区分开）。不填照旧。
                label = (str(tag.get("label") or "").strip()
                         or f"你转述给{self._session_label(sess)}")
            else:
                window = max(0, int(self._cfg("outbound_cause_window", 180) or 0))
                cause = self._outbound_cause(sess, window, bot_id=bot) if window else ""
                label = f"你主动发给{self._session_label(sess)}"
            # 正文留 180 字，起因 80 字，加起来不撑破 300 的字段上限。
            # 宁可截正文也别截掉起因：正文没了还能从对方嘴里问出来，起因没了永远补不回。
            body = f"[{label}] {text[:180]}"
            if cause:
                body += f"｜起因：{cause}"
            self.store.write_l3(
                event_type="bot_outbound",
                subject="",
                content=body[:300],
                session_id=sess,
                expires_at=self._l3_expiry(),
                bot_id=bot,
            )
        except Exception:
            logger.debug("[SugarMemory] 写跨会话发送记录失败", exc_info=True)

    # ---------- 阶段总结：单次 LLM 调用 ----------

    def _summary_prompt_extra(self) -> str:
        """本机追加在总结提示词末尾的那一段（空串 = 只用内置提示词）。"""
        return str(self._cfg("summary_prompt_extra", "") or "").strip()

    def _summary_system_prompt(self) -> str:
        """真正发给总结模型的 system prompt：内置原文 + 可选追加段。

        内置那段**只读**：面板里是预览，没有任何入口能改它一个字。追加段单独
        存在配置里，清空即回到出厂行为，不会因为编辑失误把规则弄丢。
        注意这里不参与注入缓存指纹（PROMPT_CFG_KEYS 里没有这个键）——
        它只影响总结请求，不影响每轮挂出去的 system_prompt 字节。
        """
        extra = self._summary_prompt_extra()
        if not extra:
            return SUMMARY_SYSTEM_PROMPT
        return SUMMARY_SYSTEM_PROMPT + SUMMARY_EXTRA_HEADER + extra

    def _summary_provider(self):
        """总结优先用配置里指定的 provider，留空则用当前对话 provider。"""
        pid = str(self._cfg("summary_provider_id", "") or "").strip()
        if pid:
            try:
                prov = self.context.get_provider_by_id(pid)
                if prov is not None:
                    return prov
                logger.warning(f"[SugarMemory] 总结 provider 不存在：{pid}，回退到当前对话 provider")
            except Exception:
                logger.warning(f"[SugarMemory] 取总结 provider 失败：{pid}")
        try:
            return self.context.get_using_provider()
        except Exception:
            return None

    def _roster_display(self, roster: dict) -> tuple[dict, dict]:
        """{编号: 内部键} → ({编号: 平时怎么叫}, {内部键: 平时怎么叫})。

        名单上给模型看的必须是 call_name：模型看到什么名字，写摘要就照着用。
        给它内部键（往往是一个人最早的昵称），摘要里就会留下一个谁也不认识
        的历史名字。落库仍然用内部键——那是主键，不能拿显示名去写。
        查不到映射就原样退回，这一步不该挡住总结。
        """
        if not roster:
            return {}, {}
        try:
            names = self.store.person_names(list(roster.values()))
        except Exception:
            names = {}
        show = {pid: (names.get(name) or name) for pid, name in roster.items()}
        shown_of = {name: (names.get(name) or name) for name in roster.values()}
        return show, shown_of

    async def _summary_chat(self, provider, transcript: str, timeout: int,
                            related: str = "", roster: dict | None = None):
        """跑一次总结调用，返回 (正文, (input, cached, output) 或 None)。

        配了「总结思考强度」就走 provider 的 client 直连，只给这一次请求带
        reasoning_effort。标准通道的 text_chat 不收这个参数，而 custom_extra_body
        是 provider 级共享的，临时改会波及同时在跑的聊天，所以干脆另开一条。
        直连失败（不同 provider 的 client 不一定兼容）自动退回标准通道。

        related 是这个人的「已有长期记忆」清单（带 #编号）。有了它，总结才不是
        闭卷答题：模型能看出这次说的话是在推翻哪一条，而不是只能不断新增。
        roster 是「在场的人」名单（{编号: 真名}）。模型只选编号不写名字，
        写不出名单外的名字 = 不会在人物库里凭空多一个假人。
        """
        # 日志放前面、清单放后面：日志长、清单短，模型读完日志再看清单，
        # 判断「这句是不是在改那条」比反着来准。
        sysp = self._summary_system_prompt()
        user_text = ""
        table = roster_text(roster or {})
        if table:
            user_text += f"在场的人（subject_id / source_id / target_id 只能从这里选）：{table}\n\n"
        user_text += f"事件日志：\n{transcript}"
        if related:
            user_text += f"\n\n已有长期记忆（只用于判断有没有改口，不要照抄）：\n{related}"
        effort = str(self._cfg("summary_reasoning_effort", "") or "").strip().lower()
        client = getattr(provider, "client", None)
        try:
            model = provider.get_model()
        except Exception:
            model = None
        if effort and client is not None and model:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sysp},
                            {"role": "user", "content": user_text},
                        ],
                        reasoning_effort=effort,
                    ),
                    timeout=timeout,
                )
                text = str(resp.choices[0].message.content or "")
                u = getattr(resp, "usage", None)
                triple = None
                if u is not None:
                    pt = int(getattr(u, "prompt_tokens", 0) or 0)
                    details = getattr(u, "prompt_tokens_details", None)
                    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
                    triple = (max(0, pt - cached), cached,
                              int(getattr(u, "completion_tokens", 0) or 0))
                return text, triple
            except Exception as e:
                logger.warning(
                    f"[SugarMemory] 直连总结失败（{type(e).__name__}: {e}），回退标准通道")
        resp = await asyncio.wait_for(
            provider.text_chat(prompt=user_text, system_prompt=sysp),
            timeout=timeout,
        )
        u = getattr(resp, "usage", None)
        triple = None
        if u is not None:
            triple = (int(getattr(u, "input_other", 0) or 0),
                      int(getattr(u, "input_cached", 0) or 0),
                      int(getattr(u, "output", 0) or 0))
        return str(getattr(resp, "completion_text", "") or ""), triple

    def _schedule_summary(self, session: str):
        """满 N 条后不立刻跑：等这一轮聊完、静默若干秒再跑，避开对话请求的并发额度。"""
        if session not in self._pending:
            return
        now = time.time()
        # 刚失败过的会话先退避一分钟，免得每条消息都重烧一次（一次要 1~3 分钟）
        fail_at = self._summary_fail_at.get(session, 0.0)
        if fail_at and now - fail_at < 60:
            return
        delay = max(0, int(self._cfg("summary_idle_seconds", 10) or 0))
        old = self._idle_tasks.get(session)
        if old is not None and not old.done():
            # "等这轮聊完"只在开头有效：最多顺延 max(delay*3, 30) 秒，
            # 否则聊得越热闹计时器被推得越远，永远等不到静默。
            first = self._pending_since.setdefault(session, now)
            if now - first < max(delay * 3, 30):
                old.cancel()
            else:
                return
        else:
            self._pending_since.setdefault(session, now)
        self._idle_tasks[session] = asyncio.create_task(self._delayed_summary(session, delay))

    async def _delayed_summary(self, session: str, delay: int):
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._summary_lock:
                if session not in self._pending:
                    return
                ok = await self._run_summary(session)
                # 成功才清待办。失败/熔断时保留，下次这个会话再来消息会自动重试；
                # 熔断挡着的时候只多一次 DB 查询，不会真的调 LLM。
                if ok:
                    self._pending.discard(session)
                    self._summary_fail_at.pop(session, None)
                else:
                    # 记下失败时间，_schedule_summary 会因此退避，不再每条消息都重烧
                    self._summary_fail_at[session] = time.time()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[SugarMemory] 延迟总结异常")
            self._summary_fail_at[session] = time.time()
        finally:
            try:
                if self._idle_tasks.get(session) is asyncio.current_task():
                    self._idle_tasks.pop(session, None)
                self._pending_since.pop(session, None)
            except Exception:
                pass

    def _summary_circuit_fail(self):
        """记一次总结失败；达到阈值就进冷却并打日志。"""
        try:
            st = self.store.circuit_fail(
                "summary",
                int(self._cfg("summary_circuit_threshold", 3) or 3),
                float(self._cfg("summary_circuit_cooldown", 600) or 600),
            )
            if st["fails"] >= st["threshold"]:
                logger.warning(
                    f"[SugarMemory] 总结连续失败 {st['fails']} 次，熔断 {int(st['cooldown'])}s 内不再自动跑")
            else:
                logger.info(f"[SugarMemory] 总结失败计数 {st['fails']}/{st['threshold']}")
        except Exception:
            logger.debug("[SugarMemory] 记录熔断失败状态出错", exc_info=True)

    async def _run_summary(self, session: str | None = None, manual: bool = False) -> bool:
        # 群聊只读：群会话永不总结，防止遗留 pending 或手动触发绕过记录层
        if session and not self._cfg("record_in_group", False):
            if "GroupMessage" in session or session.startswith("group:"):
                return False
        if self._summarizing:
            # 补录/导入会连续触发，丢掉就等于这批消息永远轮不到总结。
            self._summary_pending = True
            return False
        # 熔断：自动触发被挡，手动点总结不受限（但仍会给熔断计数清零/累加）
        if not manual:
            try:
                blocked, left = self.store.circuit_blocked(
                    "summary", float(self._cfg("summary_circuit_cooldown", 600) or 600))
                if blocked:
                    logger.warning(f"[SugarMemory] 总结熔断中，还剩 {int(left)}s")
                    return False
            except Exception:
                logger.debug("[SugarMemory] 熔断检查失败，放行", exc_info=True)
        self._summarizing = True
        try:
            wu = max(1, int(self._cfg("summary_window_user", 60) or 60))
            wb = max(0, int(self._cfg("summary_window_bot", 30) or 30))
            if not session:
                latest = self.store.query_recent_l3(limit=1, event_type="user_message")
                if latest and latest[0].get("session_id"):
                    session = latest[0]["session_id"]
            logs = self.store.query_recent_l3(limit=wu, event_type="user_message",
                                             session_id=session)
            if wb:
                logs += self.store.query_recent_l3(limit=wb, event_type="bot_reply",
                                                   session_id=session)
            if not logs:
                return False
            logs.sort(key=lambda r: r["id"])
            # 给每条日志打上说话人编号，模型只选编号、不写名字——写不出名单
            # 以外的人，就不会再冒出「弟弟」这种凭空多出来的档案。
            lines, roster = number_transcript(logs)
            # 给模型看的名单必须是「平时怎么叫」，不是内部键。内部键往往是一个人
            # 最早的昵称，模型照着它写摘要，库里就会多出一段用旧名字记的摘要
            # （实机：「Moon 谈论…」和「张三」并存，读起来像两个人）。
            # 落库仍然用 internal（apply_summary 要的是真名），两者不可混用。
            roster_show, shown_of = self._roster_display(roster)
            # 名单为空说明这段日志里没有一句能归到具体人身上（全是 AI 自己的话，
            # 或者是没有 subject 的历史数据）。此时严格模式会把候选全部丢掉，
            # 这行日志是唯一能解释「总结跑了却什么都没写」的线索。
            if not roster:
                logger.warning(
                    "[SugarMemory] 这段日志没有可归属的说话人（名单为空）："
                    "本次总结不会写入任何记忆，也不会新建人物")
            users = [r for r in logs
                     if r.get("event_type") == "user_message" and r.get("subject")]
            subject = users[-1]["subject"] if users else ""
            sess = session or ""
            provider = self._summary_provider()
            if provider is None:
                logger.warning("[SugarMemory] 无可用 provider，跳过阶段总结")
                return False
            # 这个人的底档：带上 #编号一起喂给模型，它才有能力判断「这次是
            # 在推翻哪一条」。不带的话总结是闭卷答题，只会不断新增。
            related = ""
            use_related = bool(subject) and self._cfg("summary_related_l2", True)
            rel_limit = 0
            if use_related:
                try:
                    rel_limit = max(0, int(self._cfg("summary_related_limit", 16) or 0))
                except (TypeError, ValueError):
                    rel_limit = 0
            timeout = max(20, int(self._cfg("summary_timeout_seconds", 120) or 120))
            chunks = chunk_transcript(lines)
            parts: list[dict] = []
            used = [0, 0, 0]
            got_usage = False
            for ci, chunk in enumerate(chunks):
                # 每段现查一次底档：前一段刚落库的新事实，后一段应该能看见，
                # 否则同一轮总结里前后两段会各记一条互相矛盾的说法。
                if use_related and rel_limit > 0:
                    try:
                        rows = self.store.query_active_l2(subject=subject, limit=200)
                        related = format_related(
                            pick_related_l2(rows, chunk[-1500:], rel_limit), shown_of)
                    except Exception:
                        logger.debug("[SugarMemory] 组装相关记忆失败", exc_info=True)
                raw, err, usage_triple = "", None, None
                for attempt in range(3):
                    try:
                        raw, usage_triple = await self._summary_chat(
                            provider, chunk, timeout, related, roster_show)
                        err = None
                        break
                    except Exception as e:
                        err = e
                        if attempt < 2:
                            await asyncio.sleep(2 * (attempt + 1))
                if err is not None:
                    logger.warning(
                        f"[SugarMemory] 总结调用失败（第 {ci + 1}/{len(chunks)} 段，"
                        f"已重试 3 次）：{err!r}")
                    self._summary_circuit_fail()
                    return False
                parsed = parse_summary_json(raw)
                if parsed is None:
                    logger.warning(f"[SugarMemory] 总结 JSON 解析失败：{raw[:150]!r}")
                    self._summary_circuit_fail()
                    return False
                if usage_triple:
                    got_usage = True
                    for i in range(3):
                        used[i] += int(usage_triple[i] or 0)
                parts.append(parsed)
            # 多段合并：摘要接起来，updates 的 by 按段偏移（否则会指到别人的候选）
            merged = merge_summaries(parts)
            # 真实 usage 优先；provider 没给就按文本估一份，标成 estimate
            if got_usage:
                self._record_tokens("summary", input=used[0], cached=used[1],
                                    output=used[2], subject=subject, session_id=sess)
            else:
                self._record_tokens(
                    "summary", source="estimate",
                    input=estimate_tokens(self._summary_system_prompt()
                                          + "\n".join(chunks) + related),
                    output=estimate_tokens(merged["summary"]),
                    subject=subject, session_id=sess)
            stats = apply_summary(self.store, self.gate, merged,
                                  subject=subject, session_id=sess, roster=roster)
            # 成功一次就清零熔断计数，冷却期间手动跑成功也能立刻恢复自动触发
            try:
                self.store.circuit_ok("summary")
            except Exception:
                pass
            logger.info(f"[SugarMemory] 阶段总结完成（{sess or '全局'}）：{stats}")
            # 总结可能刚写入新事实：顺手给缺别名的事实补问法（后台、静默、失败不重试）
            if self._cfg("alias_autogen", True):
                asyncio.create_task(self._run_alias_batch())
            return True
        except Exception:
            logger.exception("[SugarMemory] 阶段总结异常")
            self._summary_circuit_fail()
            return False
        finally:
            self._summarizing = False
            if self._summary_pending:
                self._summary_pending = False
                try:
                    asyncio.create_task(self._run_summary(session, manual=False))
                except RuntimeError:
                    # 事件循环已经在关，补跑没意义
                    pass

    # ---------- 调试命令 ----------

    @filter.command("糖记忆")
    async def sugar_cmd(self, event: AstrMessageEvent):
        """糖记忆调试：status|mem [N]|why <id>|del <id>|pack [关键词]|export|sync"""
        if not event.is_admin():
            return
        parts = (event.message_str or "").strip().split()
        sub = parts[1] if len(parts) > 1 else "status"

        if sub == "status":
            l2 = self.store.query_active_l2(limit=1000)
            l3 = self.store.query_recent_l3(limit=1000)
            audit = self.store.recent_audit(limit=1000, decisions=())
            yield event.plain_result(
                f"SugarMemory M1：L2 事实 {len(l2)} 条｜L3 事件 {len(l3)} 条｜审计 {len(audit)} 条｜"
                f"注入={'开' if self._cfg('inject_enabled', True) else '关（影子模式）'}｜"
                f"总结={'开' if self._cfg('summarize_enabled', True) else '关'}｜"
                f"计数 {max(self._msg_count.values()) if self._msg_count else 0}"
                f"/{self._cfg('summary_interval_messages', 50)}"
                f"（{len(self._pending)} 个会话待总结）｜"
                f"面板：WebUI → /api/plug/{PANEL_PREFIX}"
            )
        elif sub == "mem":
            n = int(parts[2]) if len(parts) > 2 else 10
            rows = self.store.query_active_l2(limit=n)
            if not rows:
                yield event.plain_result("L2 库为空")
                return
            lines = [
                f"#{r['id']} [{r['confidence']:.2f}|稳{r['stability']}] {r['subject']}的{r['predicate']}：{r['value']}"
                for r in rows
            ]
            yield event.plain_result("\n".join(lines))
        elif sub == "why":
            if len(parts) < 3 or not parts[2].lstrip("#").isdigit():
                yield event.plain_result("用法：/糖记忆 why <id>；被拒候选看 /糖记忆 deny")
                return
            fid = int(parts[2].lstrip("#"))
            fact = self.store.get_l2(fid)
            if fact:
                chain = self.store.l2_chain(fid)
                chain_desc = " ← ".join(f"#{c['id']}({c['status']})" for c in chain)
                w = (self._last_why or {}).get(fid) or {}
                labels = (
                    ("channel", "通道"), ("bigram", "命中"), ("idf", "IDF"),
                    ("rel_ctx", "历史"), ("rel_alias", "别名"),
                    ("norm", "归一"), ("confidence", "置信"),
                    ("retention", "保留"), ("intent", "意图"),
                    ("bonus", "加权"), ("mmr", "去重"),
                    ("consensus", "共识"), ("score", "总分"),
                )
                if w:
                    bits = []
                    for k, label in labels:
                        v = w.get(k)
                        if v in (None, "", 0, 0.0, False):
                            continue
                        bits.append(f"{label}={v}")
                    bits.append("进包=" + ("是" if w.get("in_pack") else "否（本轮落选）"))
                    rank_line = "排序明细：" + "｜".join(bits)
                else:
                    rank_line = "排序明细：本轮没进候选池（人不对 / 已沉底 / 候选上限卡住）"
                yield event.plain_result(
                    f"#{fact['id']} {fact['subject']}的{fact['predicate']}：{fact['value']}\n"
                    f"门控原因：{fact['gate_reason']}｜来源：{fact['source_type']} #{fact['source_msg_id'] or '—'}\n"
                    f"原文：{fact['source_quote'] or '—'}｜链路：{chain_desc}\n"
                    f"{rank_line}"
                )
                return
            audit = self.store.recent_audit(limit=50, decisions=())
            hit = next((a for a in audit if a["id"] == fid), None)
            if hit:
                yield event.plain_result(
                    f"审计 #{hit['id']} [{hit['decision']}] 原因：{hit['reason']}\n候选：{hit['candidate'][:100]}"
                )
            else:
                yield event.plain_result(f"未找到 #{fid}（既非 L2 也非最近审计）")
        elif sub == "deny":
            recent = self.store.recent_audit(limit=8)
            yield event.plain_result(
                "\n".join(f"审计#{a['id']} [{a['decision']}] {a['reason']}｜{a['candidate'][:50]}" for a in recent)
                or "最近无 deny/transform 记录"
            )
        elif sub == "del":
            if len(parts) < 3 or not parts[2].lstrip("#").isdigit():
                yield event.plain_result("用法：/糖记忆 del <id>")
                return
            ok = self.store.delete_l2(int(parts[2].lstrip("#")))
            yield event.plain_result(f"#{parts[2]} 已删除" if ok else "未找到该条")
        elif sub == "pack":
            query = " ".join(parts[2:]) if len(parts) > 2 else ""
            builder = self._pack_builder()
            pack = builder.build(query=query, budget_tokens=int(self._cfg("inject_budget_tokens", 1200)))
            block = build_block(pack)
            yield event.plain_result(
                f"记忆包（估 {pack.get('token_estimate')} tokens）：\n{block[:1500]}"
            )
        elif sub == "export":
            data = self.store.export_all()
            yield event.plain_result(
                "导出统计：L2 %d｜L3 %d｜审计 %d（完整导出走 Web API）"
                % (len(data["l2_facts"]), len(data["l3_events"]), len(data["gate_audit"]))
            )
        elif sub == "sync":
            if self._summarizing:
                yield event.plain_result("总结进行中，稍候")
                return
            yield event.plain_result("开始阶段总结（1 次 LLM 调用）…")
            asyncio.create_task(self._manual_summary(event))
        else:
            yield event.plain_result("子命令：status|mem [N]|why <id>|deny|del <id>|pack [关键词]|export|sync")

    async def _manual_summary(self, event: AstrMessageEvent):
        async with self._summary_lock:
            await self._run_summary(manual=True)

    # ---------- WebUI 面板 ----------

    def _register_web(self):
        try:
            register = self.context.register_web_api
        except Exception:
            return
        register(f"{PANEL_PREFIX}", self._panel_page, ["GET"], "记忆中枢面板")
        register(f"{PANEL_PREFIX}/img/<path:name>", self._panel_asset, ["GET"], "面板图片资源")
        register(f"{API_PREFIX}/overview", self._api_overview, ["GET"], "总览")
        register(f"{API_PREFIX}/subjects", self._api_subjects, ["GET"], "用户列表")
        register(f"{API_PREFIX}/user", self._api_user, ["GET"], "单个用户的记忆")
        register(f"{API_PREFIX}/l2", self._api_l2, ["GET"], "L2 事实")
        register(f"{API_PREFIX}/l3", self._api_l3, ["GET"], "L3 事件")
        register(f"{API_PREFIX}/audit", self._api_audit, ["GET"], "门控审计")
        register(f"{API_PREFIX}/outbound", self._api_outbound, ["GET"], "转述/外发记录")
        register(f"{API_PREFIX}/delete", self._api_delete, ["POST"], "删除 L2")
        register(f"{API_PREFIX}/chain", self._api_chain, ["GET"], "修订链路")
        register(f"{API_PREFIX}/pack", self._api_pack, ["POST"], "记忆包预览")
        register(f"{API_PREFIX}/summarize", self._api_summarize, ["POST"], "触发阶段总结")
        register(f"{API_PREFIX}/summary_progress", self._api_summary_progress, ["GET"], "总结进度")
        register(f"{API_PREFIX}/summary_prompt", self._api_summary_prompt, ["GET"],
                 "总结提示词预览")
        register(f"{API_PREFIX}/l2/add", self._api_l2_add, ["POST"], "新增长期记忆")
        register(f"{API_PREFIX}/l2/update", self._api_l2_update, ["POST"], "修改长期记忆")
        register(f"{API_PREFIX}/l2/pin", self._api_l2_pin, ["POST"], "钉住长期记忆")
        register(f"{API_PREFIX}/l2/archive", self._api_l2_archive, ["POST"], "归档长期记忆")
        register(f"{API_PREFIX}/l2/unarchive", self._api_l2_unarchive, ["POST"], "恢复归档记忆")
        register(f"{API_PREFIX}/l2/decay", self._api_l2_decay, ["POST"], "沉底清理")
        register(f"{API_PREFIX}/l3/add", self._api_l3_add, ["POST"], "新增短期记忆")
        register(f"{API_PREFIX}/l3/update", self._api_l3_update, ["POST"], "修改短期记忆")
        register(f"{API_PREFIX}/l3/delete", self._api_l3_delete, ["POST"], "删除短期记忆")
        register(f"{API_PREFIX}/l3/import", self._api_l3_import, ["POST"], "批量导入短期记忆")
        register(f"{API_PREFIX}/l3/distill", self._api_l3_distill, ["POST"], "补录提炼长期记忆")
        register(f"{API_PREFIX}/l3/distill/stop", self._api_l3_distill_stop, ["POST"], "停止补录提炼")
        register(f"{API_PREFIX}/l3/distill/progress", self._api_l3_distill_progress,
                 ["GET"], "补录提炼进度")
        register(f"{API_PREFIX}/affinity", self._api_affinity, ["GET"], "好感度总览")
        register(f"{API_PREFIX}/affinity/set", self._api_affinity_set, ["POST"], "设置好感度")
        register(f"{API_PREFIX}/affinity/add", self._api_affinity_add, ["POST"], "增减好感度")
        register(f"{API_PREFIX}/affinity/stages", self._api_affinity_stages, ["GET"], "好感度阶段")
        register(f"{API_PREFIX}/affinity/stages/save", self._api_affinity_stages_save, ["POST"], "保存阶段表")
        register(f"{API_PREFIX}/affinity/stages/reset", self._api_affinity_stages_reset, ["POST"], "重置阶段表")
        register(f"{API_PREFIX}/people", self._api_people, ["GET"], "人物（AI 与人）")
        register(f"{API_PREFIX}/people/qq", self._api_people_qq, ["POST"], "绑定 QQ")
        register(f"{API_PREFIX}/people/alias", self._api_people_alias, ["POST"], "加一个称呼")
        register(f"{API_PREFIX}/people/alias/remove", self._api_people_alias_remove,
                 ["POST"], "删一个称呼")
        register(f"{API_PREFIX}/people/merge", self._api_people_merge, ["POST"], "合并两个人")
        register(f"{API_PREFIX}/people/delete", self._api_people_delete, ["POST"],
                 "删除一个人及全部记忆")
        register(f"{API_PREFIX}/people/call", self._api_people_call, ["POST"], "设称呼名")
        register(f"{API_PREFIX}/people/call/clear", self._api_people_call_clear,
                 ["POST"], "删除称呼名")
        register(f"{API_PREFIX}/people/alias_cleanup", self._api_people_alias_cleanup,
                 ["POST"], "清理久不提的称呼")
        register(f"{API_PREFIX}/people/scan", self._api_people_scan, ["POST"], "重新登记与归并")
        register(f"{API_PREFIX}/relations/graph", self._api_rel_graph, ["GET"], "人际关系图")
        register(f"{API_PREFIX}/relations/user", self._api_rel_user, ["GET"], "某人的关系与证据")
        register(f"{API_PREFIX}/relations/set", self._api_rel_set, ["POST"], "设置关系")
        register(f"{API_PREFIX}/relations/delete", self._api_rel_delete, ["POST"], "删除关系")
        register(f"{API_PREFIX}/relations/aliases", self._api_rel_aliases, ["GET"], "重名分组")
        register(f"{API_PREFIX}/relations/merge", self._api_rel_merge, ["POST"], "合并重名的人")
        register(f"{API_PREFIX}/relations/evidence", self._api_rel_evidence, ["GET"], "关系证据")
        register(f"{API_PREFIX}/relations/evidence/delete", self._api_rel_evidence_delete,
                 ["POST"], "删除一条关系证据")
        register(f"{API_PREFIX}/relations/evidence/update", self._api_rel_evidence_update,
                 ["POST"], "修改一条关系证据")
        register(f"{API_PREFIX}/config", self._api_config, ["GET"], "总配置读取")
        register(f"{API_PREFIX}/config/set", self._api_config_set, ["POST"], "总配置保存")
        register(f"{API_PREFIX}/tokens", self._api_tokens, ["GET"], "Token 用量统计")
        register(f"{API_PREFIX}/tokens/hit", self._api_tokens_hit, ["GET"], "缓存命中率明细")
        register(f"{API_PREFIX}/tokens/clear", self._api_tokens_clear, ["POST"], "清空 Token 用量")
        register(f"{API_PREFIX}/guide", self._api_guide, ["GET"], "配置引导（只读）")

    async def _panel_page(self):
        page = Path(__file__).parent / "pages" / "记忆中枢" / "index.html"
        return await send_file(str(page))

    async def _panel_asset(self, name: str = ""):
        """面板的图片资源。只认 pages/记忆中枢/img/ 下面，且不许用 .. 跳出去。"""
        base = (Path(__file__).parent / "pages" / "记忆中枢" / "img").resolve()
        try:
            target = (base / str(name or "")).resolve()
            target.relative_to(base)
        except (ValueError, OSError):
            return Response("not found", status=404)
        if not target.is_file():
            return Response("not found", status=404)
        return await send_file(str(target))

    def _ok(self, data=None) -> dict:
        return {"status": "ok", "data": data or {}}

    def _err(self, msg: str) -> dict:
        return {"status": "error", "message": msg}

    def _stats(self) -> dict:
        l2 = self.store.query_active_l2(limit=5000)
        l3 = self.store.query_recent_l3(limit=5000)
        audit = self.store.recent_audit(limit=5000, decisions=())
        subjects: dict[str, int] = {}
        for r in l2:
            subjects[r["subject"]] = subjects.get(r["subject"], 0) + 1
        return {
            "l2_count": len(l2),
            "l3_count": len(l3),
            "l3_units": count_conversations(l3),
            "summary_count": sum(1 for r in l3 if r.get("event_type") == "session_summary"),
            "chat_count": sum(1 for r in l3 if r.get("event_type") != "session_summary"),
            "outbound_count": sum(1 for r in l3 if r.get("event_type") == "bot_outbound"),
            "relay_count": sum(1 for r in l3
                               if str(r.get("content") or "").startswith("[你转述给")),
            "audit_count": len(audit),
            "deny_count": sum(1 for a in audit if a["decision"] == "deny"),
            "transform_count": sum(1 for a in audit if a["decision"] == "transform"),
            "subjects": subjects,
            "inject_enabled": bool(self._cfg("inject_enabled", True)),
            "inject_relations": bool(self._cfg("inject_relations", False)),
            "record_enabled": bool(self._cfg("record_enabled", True)),
            "summarize_enabled": bool(self._cfg("summarize_enabled", True)),
            "msg_count": max(self._msg_count.values()) if self._msg_count else 0,
            "pending_summaries": len(self._pending),
            "summary_window_user": int(self._cfg("summary_window_user", 60) or 60),
            "summary_window_bot": int(self._cfg("summary_window_bot", 30) or 30),
            "summary_idle_seconds": int(self._cfg("summary_idle_seconds", 10) or 10),
            "summary_provider": str(self._cfg("summary_provider_id", "") or ""),
            "summary_interval": int(self._cfg("summary_interval_messages", 50) or 50),
            "l3_retention_days": int(self._cfg("l3_retention_days", 14) or 14),
            "summarizing": self._summarizing,
            "affinity_enabled": bool(self._cfg("affinity_enabled", True)),
            "affinity_cap": int(self._cfg("affinity_cap", 1000) or 1000),
            "affinity_avg": self._affinity_payload()["avg"],
            "relation_count": len(self.store.relations_all()),
            "panel_url": f"/api/plug/{PANEL_PREFIX}",
        }

    async def _api_overview(self):
        d = self._stats()
        d["integration"] = self._integration_payload()
        return self._ok(d)

    async def _api_subjects(self):
        gkeys = self.store.group_keys()
        # subject 为空的行是历史上没归属的 L3（转述、旧版本写进去的），
        # 它们不属于任何人，别在「人」的列表里占一行。
        items = [s for s in self.store.subject_stats()
                 if str(s.get("subject") or "").strip()
                 and not self._is_group_name(s.get("subject", ""), gkeys)]
        # 防呆：人物库里有、但一条记忆都还没有的人，也必须出现在这个列表里。
        # 用户第一眼看的就是这里，人突然不见了只会以为记忆被清了或者插件坏了。
        # 记忆条数为 0 是「还没攒到」，不是「被删了」，得让人看得见这个区别。
        known = {str(x.get("subject") or "") for x in items}
        try:
            people = self.store.person_all()
        except Exception:
            people = []
        for p in people:
            name = str(p.get("person") or "").strip()
            if not name or name in known:
                continue
            if self._is_group_name(name, gkeys):
                continue
            items.append({
                "subject": name, "l2": 0, "l3": 0, "l3_msgs": 0, "sum": 0,
                "last_at": float(p.get("updated_at") or 0),
                "empty": True, "qq": str(p.get("qq") or ""),
            })
        # 两个名字分工不同：
        #   shown  = QQ 昵称（display_name），"这人是谁"，稳定、认人用；
        #   spoken = AI 平时怎么叫（call_name），"张口叫哪个"，可以跟昵称不一样。
        # subject 是内部键（往往是最早那个昵称），直接拿去显示会出现"列表叫张三、
        # 详情叫 Moon"这种自相矛盾。群名片只是别名，两个位置都不参与。
        pmap = {str(p.get("person") or ""): p for p in people}
        for it in items:
            sub = str(it.get("subject") or "")
            p = pmap.get(sub) or {}
            nick = str(p.get("display_name") or "").strip()
            call = str(p.get("call_name") or "").strip()
            # display_name 清空（自愈把写坏的号码清掉了）时退到称呼名，
            # 别把 person 键（可能是纯 QQ 号）直接顶上显示位。
            it["shown"] = nick or call or sub
            it["spoken"] = call if call and alias_key(call) != alias_key(it["shown"]) else ""
            it["qq"] = str(p.get("qq") or it.get("qq") or "")
        # 有记忆的排前面，同档再按最后活跃时间——空档案不该把真人挤下去
        items.sort(key=lambda x: (
            -(int(x.get("l2") or 0) * 2 + int(x.get("l3") or 0)),
            -float(x.get("last_at") or 0)))
        return self._ok({"items": items})

    async def _api_user(self):
        from quart import request as quart_request
        subject = (quart_request.args.get("subject") or "").strip()
        limit = min(int(quart_request.args.get("limit") or 400), 2000)
        l2 = [] if not subject else self.store.query_active_l2(subject=subject, limit=1000)
        # 保留度/沉底倒计时统一走后端那一份（带错峰相位）——前端只负责显示。
        # 直接拿裸行算会漏掉 _stagger，面板倒计时跟清理那边差两个月。
        try:
            rmap = self.store.l2_retention_map(subject=subject)
        except Exception:
            rmap = {}
        for r in l2:
            m = rmap.get(int(r["id"]))
            if m:
                r["retention"] = m["r"]
                r["retire_days"] = m["left"]
        archived = [] if not subject else self.store.query_archived_l2(
            subject=subject, limit=200)
        l3 = self.store.query_l3(subject=subject, limit=limit)
        summaries = self.store.query_l3(
            subject=subject, event_type="session_summary", limit=200
        )
        # 骨架单独拉全量：普通 l3 受 limit 截断，骨架页要完整的按天视图
        skels = [] if not subject else self.store.query_l3(
            subject=subject, event_type="l3_skeleton", limit=1000
        )
        units = self.store.count_units(subject)
        # subject 是内部身份键（常常是这个人最早的昵称）。面板要显示他现在
        # 能被称呼的名，否则改了昵称之后整页还写着旧名字。
        shown = ""
        try:
            shown = (self.store.person_names([subject]).get(subject) or "") if subject else ""
        except Exception:
            shown = ""
        return self._ok({
            "subject": subject, "shown_name": shown or subject,
            "l2": l2, "l3": l3, "summaries": summaries,
            "archived": archived, "skels": skels,
            "stats": {"l2": len(l2), "l3": units["l3"], "l3_msgs": units["l3_msgs"],
                      "sum": len(summaries), "archived": len(archived)},
        })

    async def _api_l2(self):
        from quart import request as quart_request
        subject = (quart_request.args.get("subject") or "").strip()
        q = (quart_request.args.get("q") or "").strip().lower()
        limit = min(int(quart_request.args.get("limit") or 500), 2000)
        rows = self.store.query_active_l2(subject=subject or None, limit=limit)
        if q:
            rows = [
                r for r in rows
                if q in f"{r['subject']} {r['predicate']} {r['value']} {r['source_quote']}".lower()
            ]
        # 保留度 / 沉底倒计时统一在后端算，前端不重算公式——
        # 参数一改两头就不同步，面板上的倒计时会变成假的。
        try:
            rmap = self.store.l2_retention_map(subject=subject or None)
        except Exception:
            rmap = {}
        for r in rows:
            m = rmap.get(int(r["id"]))
            if m:
                r["retention"] = m["r"]
                r["retire_days"] = m["left"]
                r["pinned"] = m["pinned"]
        archived: list = []
        if (quart_request.args.get("archived") or "") == "1":
            try:
                archived = self.store.query_archived_l2(subject=subject or None, limit=200)
            except Exception:
                archived = []
        return self._ok({"items": rows, "archived": archived})

    async def _api_l2_pin(self):
        """钉住 / 松开。钉住的条目永不自动归档。"""
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            fid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        ok = self.store.set_l2_pinned(fid, bool(payload.get("pinned")))
        return self._ok({"pinned": bool(payload.get("pinned"))}) if ok else self._err(f"未找到 #{fid}")

    async def _api_l2_archive(self):
        """手动把一条长期记忆归档。只改 status，数据还在，能恢复。"""
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            fid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        ok = self.store.archive_l2(fid)
        return self._ok({"archived": True}) if ok else self._err(f"未找到 #{fid}")

    async def _api_l2_unarchive(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            fid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        ok = self.store.unarchive_l2(fid)
        return self._ok({"archived": False}) if ok else self._err(f"未找到归档的 #{fid}")

    async def _api_l2_decay(self):
        """手动跑一次沉底清理。dry_run 只报会收走哪些，不动数据。"""
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        dry = bool(payload.get("dry_run"))
        try:
            res = self.store.cleanup_decay(dry_run=dry)
        except Exception as exc:  # noqa: BLE001
            return self._err(f"清理失败：{exc}")
        if not dry and res.get("archived"):
            logger.info(f"[SugarMemory] 沉底归档 {res['archived']} 条长期记忆")
        return self._ok(res)

    async def _api_l3(self):
        from quart import request as quart_request
        etype = (quart_request.args.get("type") or "").strip()
        subject = quart_request.args.get("subject")
        q = (quart_request.args.get("q") or "").strip().lower()
        limit = min(int(quart_request.args.get("limit") or 200), 2000)
        rows = self.store.query_l3(
            subject=subject.strip() if subject is not None else None,
            event_type=etype or None,
            limit=limit,
        )
        if q:
            rows = [r for r in rows if q in r["content"].lower()]
        return self._ok({"items": rows})

    async def _api_audit(self):
        from quart import request as quart_request
        limit = min(int(quart_request.args.get("limit") or 100), 2000)
        rows = self.store.recent_audit(limit=limit, decisions=())
        return self._ok({"items": rows})

    async def _api_outbound(self):
        """转述/外发记录列表。带 q 搜索和 bot 筛选；顺便给出按标签分类的计数。

        bot 参数：多机器人共用一个库时，页面必须能分开看「这条是谁发的」。
        不传 = 全部（管理视角），传具体 platform_id = 只看那个机器人。
        """
        from quart import request as quart_request
        q = (quart_request.args.get("q") or "").strip().lower()
        bot = (quart_request.args.get("bot") or "").strip()[:32]
        limit = min(int(quart_request.args.get("limit") or 200), 2000)
        rows = self.store.query_recent_l3(limit=limit, event_type="bot_outbound",
                                          bot_id=bot or None)
        if q:
            rows = [r for r in rows if q in str(r.get("content") or "").lower()]
        try:
            bots = self.store.bot_ids()
        except Exception:
            bots = []
        return self._ok({"items": rows, "bots": bots, "bot": bot})

    async def _api_delete(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            fid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        ok = self.store.delete_l2(fid)
        return self._ok({"deleted": bool(ok)}) if ok else self._err(f"未找到 #{fid}")

    async def _api_l2_add(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        subject = str(payload.get("subject") or "").strip()
        predicate = str(payload.get("predicate") or "").strip()
        value = str(payload.get("value") or "").strip()
        if not subject:
            return self._err("缺少归属用户")
        if not predicate or not value:
            return self._err("谓词和内容都不能为空")
        predicate = _normalize_predicate(predicate)
        try:
            conf = float(payload.get("confidence", 0.9))
        except (TypeError, ValueError):
            conf = 0.9
        try:
            fid = self.store.add_l2_manual(
                subject=subject, predicate=predicate, value=value, confidence=conf
            )
        except Exception as exc:  # noqa: BLE001
            return self._err(str(exc))
        return self._ok({"id": fid})

    async def _api_l2_update(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            fid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        predicate = payload.get("predicate")
        value = payload.get("value")
        confidence = payload.get("confidence")
        try:
            conf = None if confidence in (None, "") else float(confidence)
        except (TypeError, ValueError):
            conf = None
        done = self.store.update_l2(
            fid,
            predicate=None if predicate is None else _normalize_predicate(predicate),
            value=None if value is None else str(value),
            confidence=conf,
        )
        return self._ok({"updated": done}) if done else self._err(f"未找到 #{fid}")

    async def _api_l3_add(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        subject = str(payload.get("subject") or "").strip()
        content = str(payload.get("content") or "").strip()
        etype = str(payload.get("event_type") or "manual_note").strip()
        if not subject:
            return self._err("缺少归属用户")
        if not content:
            return self._err("内容不能为空")
        if etype not in ("user_message", "bot_reply", "manual_note", "session_summary", "demoted_candidate"):
            etype = "manual_note"
        expires = self._l3_expiry()
        # 补录：面板选的那天就是这天。留空就按现在，跟老行为一致。
        ts = _parse_ts(payload.get("ts"))
        # 和实时录制保持同一种信封，否则总结器认不出谁说的：
        # `[你] …` 是 AI 说的，`[私] 名字: …` 是人说的
        if not content.startswith(("[你]", "[私]", "[群]")):
            scope = "群" if str(payload.get("scope") or "").strip() == "群" else "私"
            if etype == "bot_reply":
                content = f"[你] {content}"
            elif etype == "user_message":
                content = f"[{scope}] {subject}: {content}"
        eid = self.store.write_l3(
            event_type=etype, subject=subject, content=content[:300],
            session_id=str(payload.get("session_id") or "")[:120],
            expires_at=expires, now=ts,
        )
        return self._ok({"id": eid, "ts": ts or time.time()})

    async def _api_l3_update(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            eid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        content = str(payload.get("content") or "").strip()
        if not content:
            return self._err("内容不能为空")
        done = self.store.update_l3(eid, content=content)
        return self._ok({"updated": done}) if done else self._err(f"未找到 #{eid}")

    async def _api_l3_delete(self):
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        try:
            eid = int(payload.get("id"))
        except (TypeError, ValueError):
            return self._err("缺少有效的 id")
        ok = self.store.delete_l3(eid)
        return self._ok({"deleted": bool(ok)}) if ok else self._err(f"未找到 #{eid}")

    async def _api_l3_import(self):
        """批量补录聊天原话。先 dry_run 体检，确认了再真写。

        和时间戳有关的坑都在这一层拦掉：几千条挤同一天不会报错，但骨架会退化、
        时序锚失效，提炼出来的 L2 全是缝合怪——等发现时已经污染了长期记忆。

        说话人也要在这里判：总结器只认「名字: 内容」（人说的）和「[你] 内容」
        （AI 说的）两种开头，裸文本进去会把 AI 的发言提炼成用户的事实。
        """
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        dry_run = bool(payload.get("dry_run"))
        default_subject = str(payload.get("default_subject") or "").strip()
        default_session = str(payload.get("session_id") or "").strip()
        default_type = str(payload.get("event_type") or "user_message").strip()
        scope = "群" if str(payload.get("scope") or "").strip() == "群" else "私"
        allowed = ("user_message", "bot_reply", "manual_note", "session_summary", "demoted_candidate")
        if default_type not in allowed:
            default_type = "user_message"

        raw_ai = payload.get("ai_names")
        if isinstance(raw_ai, str):
            ai_names = [p.strip() for p in re.split(r"[,，、;；/\n]+", raw_ai) if p.strip()]
        elif isinstance(raw_ai, (list, tuple)):
            ai_names = [str(p).strip() for p in raw_ai if str(p).strip()]
        else:
            ai_names = []

        items = payload.get("items")
        rows: list[dict] = []
        long_cnt = 0
        if isinstance(items, list):
            if len(items) > 20000:
                return self._err(f"单次最多 20000 条，这批 {len(items)} 条，请分批导入")
            for it in items:
                if not isinstance(it, dict):
                    continue
                content = str(it.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > 300:
                    long_cnt += 1
                etype = str(it.get("event_type") or "").strip()
                rows.append({
                    "ts": _parse_ts(it.get("ts")),
                    "speaker": str(it.get("speaker") or it.get("name") or "").strip(),
                    "subject": str(it.get("subject") or default_subject).strip(),
                    "content": content,
                    # 没写 event_type 就留空，交给 assign_roles 按说话人判
                    "event_type": etype if etype in allowed else "",
                    "session_id": str(it.get("session_id") or default_session).strip(),
                })
        else:
            for r in _parse_import_text(str(payload.get("text") or ""), default_subject):
                if len(r.get("content") or "") > 300:
                    long_cnt += 1
                r["session_id"] = default_session
                rows.append(r)

        if not rows:
            return self._err("没有解析出可导入的记录，检查一下格式")

        # 名字先归一：同一个人有多个称呼时，subject 和内容里的名字必须一致，
        # 否则一条记录里 subject 是「丙哥」、内容却写着「用户丙」，总结器会当成两个人。
        resolve_cache: dict[str, str] = {}

        def _res(name: str) -> str:
            if not name:
                return ""
            # 「用户」「对方」这种泛称一进库就是个认不回来的假人，宁可不挂主语
            if is_generic_name(name):
                return ""
            k = _name_key(name)
            if k not in resolve_cache:
                try:
                    resolve_cache[k] = (self.store.person_resolve(name) or name).strip()
                except Exception:
                    resolve_cache[k] = name.strip()
            return resolve_cache[k]

        for r in rows:
            if r.get("speaker"):
                r["speaker"] = _res(r["speaker"])
            if r.get("subject"):
                r["subject"] = _res(r["subject"])

        owner = _res(default_subject)
        # 补录进库的每一条都要有会话号，否则它不属于任何会话窗口：
        # 自动总结按 session 取最近 N 条，session 为空就等于永远轮不到它。
        # 这里给每批盖一个 `import:` 前缀的戳，既让提炼能整批捞出来，
        # 又不会混进真实会话的总结窗口。
        if not default_session:
            default_session = f"import:{owner or '未归类'}:{int(time.time())}"
        for r in rows:
            if not (r.get("session_id") or "").strip():
                r["session_id"] = default_session

        stat = _assign_roles(
            rows, default_type=default_type,
            ai_names=[_res(n) or n for n in ai_names], owner=owner, scope=scope,
        )

        report = _import_report(rows, ai_names=ai_names, owner=owner)
        report["owner"] = owner
        report["session_id"] = default_session
        # 归属用户到底认不认识：认不出不会报错，但会凭空长出一个同名用户，
        # 记忆归错人这种事等发现时就晚了（L2 是永久的，不会自愈）。
        owner_exists = True
        if owner:
            try:
                owner_exists = bool(self.store.person_resolve(owner))
            except Exception:
                owner_exists = True
        report["owner_exists"] = owner_exists
        if owner and not owner_exists:
            report["warnings"].append(
                f"归属用户「{owner}」在人物库里没有记录，导入后会新建一个同名用户。"
                "如果这是某位老用户的错别字或另一种叫法，先去「人物」页加个别名再导，"
                "否则同一个人的记忆会裂成两份。"
            )
        if long_cnt:
            report["warnings"].append(
                f"{long_cnt} 条超过 300 字，只保留前 300 字（L3 是原话层，不是长文仓库）。"
            )
        report["dry_run"] = dry_run
        if dry_run:
            return self._ok(report)


        now = time.time()
        expires = self._l3_expiry()
        # 新用户在人物库里建个档，否则它只活在 l3 里，人物页/别名/关系都看不到它
        if owner and not owner_exists:
            try:
                self.store.person_ensure(owner, source="import")
            except Exception:
                logger.debug("[SugarMemory] 导入建档失败", exc_info=True)
        # 按真实时间排序落库：id 单调与时间一致，之后按 id 取最近窗口才不会串年代
        rows.sort(key=lambda r: r.get("ts") or now)
        for r in rows:
            r["created_at"] = r.get("ts") or now
        written = self.store.write_l3_bulk(rows, expires_at=expires)
        try:
            days = int(self._cfg("l3_retention_days", 14))
        except (TypeError, ValueError):
            days = 14
        report["written"] = written
        keep = f"{days} 天后自动清理" if days > 0 else "永久保留（保留期设为 0）"
        split = ""
        if report.get("ai_count") or report.get("user_count"):
            split = f"识别出 AI 说的 {report.get('ai_count', 0)} 条、人说的 {report.get('user_count', 0)} 条。"
        report["note"] = (
            f"已写入 {written} 条，{keep}。{split}"
            "这批记录不参与实时会话的自动总结（那套按会话取窗口，够不到补录数据），"
            "要变成长期记忆得跑一遍提炼。"
        )
        logger.info(
            f"[SugarMemory] 批量导入 L3：{written} 条，跨度 {report['span_days']} 天，"
            f"AI {stat.get('ai', 0)} / 人 {stat.get('user', 0)} / 其他 {stat.get('other', 0)}"
        )
        return self._ok(report)

    # ---------- 补录提炼：把导入的原话压成长期记忆 ----------
    #
    # 为什么不能靠自动总结：自动触发按「会话」取最近 N 条原话，而补录数据
    # 的 session 是 `import:*`（跟真实会话不同），窗口永远扫不到它。
    # 于是单独走一条按人分块的路径：id 升序、一块一次模型调用、处理完记水位。
    # 几千条一次塞不进上下文（_run_summary 也只截前 6000 字），分块才谈得上覆盖。

    def _distill_plan(self, subject: str, block: int) -> dict:
        try:
            cur = self.store.distill_cursor(subject)
            prog = self.store.distill_progress(subject, cur.get("last_id") or 0)
        except Exception:
            cur, prog = {"last_id": 0, "items": 0, "blocks": 0}, {}
        left = int(prog.get("left") or 0)
        chunks = (left + block - 1) // block if left else 0
        return {
            "subject": subject,
            "block": block,
            "total": int(prog.get("total") or 0),
            "left": left,
            "days": int(prog.get("days") or 0),
            "done_days": int(prog.get("done_days") or 0),
            "days_left": int(prog.get("days_left") or 0),
            "current_day": prog.get("current_day") or "",
            # 面板还在读 blocks/chunks，两个都留着，语义都是「大概还要几次调用」
            "blocks": chunks,
            "chunks": chunks,
            "done_items": int(cur.get("items") or 0),
            "done_blocks": int(cur.get("blocks") or 0),
            "cursor": int(cur.get("last_id") or 0),
        }

    async def _api_l3_distill(self):
        """dry_run 只报「要花几次调用」；真跑丢后台，面板轮询进度。"""
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        subject = str(payload.get("subject") or "").strip()
        if not subject:
            return self._err("缺少归属用户：提炼要按人取原话")
        try:
            block = int(payload.get("block") or 0)
        except (TypeError, ValueError):
            block = 0
        if block <= 0:
            block = DISTILL_BLOCK
        block = max(10, min(block, 400))
        dry_run = bool(payload.get("dry_run"))
        plan = self._distill_plan(subject, block)
        plan["running"] = self._distilling
        if dry_run:
            plan["done"] = plan["left"] <= 0
            return self._ok(plan)
        if plan["left"] <= 0:
            plan["done"] = True
            plan["note"] = "没有待提炼的补录记录了"
            return self._ok(plan)
        if self._distilling:
            return self._err("已经有一批在提炼，等它跑完或者先停掉")
        plan["running"] = True
        asyncio.create_task(self._run_distill(subject, block))
        return self._ok(plan)

    async def _api_l3_distill_stop(self):
        if not self._distilling:
            return self._ok({"stopped": False, "running": False})
        self._distill_stop = True
        return self._ok({"stopped": True, "running": True})

    async def _api_l3_distill_progress(self):
        st = dict(self._distill_state) if self._distill_state else {}
        st.setdefault("running", self._distilling)
        st.setdefault("subject", "")
        st.setdefault("total", 0)
        st.setdefault("blocks", 0)
        st.setdefault("done_blocks", 0)
        st.setdefault("items", 0)
        st.setdefault("l2", 0)
        st.setdefault("stale", 0)
        st.setdefault("error", "")
        st.setdefault("total_days", 0)
        st.setdefault("done_days", 0)
        st.setdefault("day", "")
        st.setdefault("day_n", 0)
        st.setdefault("day_items", 0)
        subject = st.get("subject") or ""
        if subject:
            try:
                cur = self.store.distill_cursor(subject)
                prog = self.store.distill_progress(subject, cur.get("last_id") or 0)
                st["left"] = int(prog.get("left") or 0)
                # 注意：done_blocks / items 是「本次运行」的计数，进度条要用它；
                # 累计值单独给，混着用会让进度条跳出 100%。
                st["done_items_total"] = int(cur.get("items") or 0)
                st["done_blocks_total"] = int(cur.get("blocks") or 0)
                st["total_days"] = int(prog.get("days") or 0)
                st["done_days"] = max(int(st.get("done_days") or 0),
                                      int(prog.get("done_days") or 0))
            except Exception:
                st["left"] = 0
        return self._ok(st)

    async def _run_distill(self, subject: str, block: int):
        """一块一次调用，按 id 升序推进。失败就停，已完成的水位保留，可以再点一次接着来。"""
        self._distilling = True
        self._distill_stop = False
        plan = self._distill_plan(subject, block)
        st = {
            "subject": subject, "running": True, "error": "", "stopped": False,
            "block": block, "total": plan["total"], "left": plan["left"],
            "blocks": plan["blocks"], "day": plan["current_day"],
            "total_days": plan["days"], "done_days": plan["done_days"],
            "day_n": 0, "day_items": 0,
            "done_blocks": 0, "items": 0, "l2": 0, "stale": 0,
            "started_at": time.time(),
            "done": False,
        }
        self._distill_state = st
        try:
            provider = self._summary_provider()
            if provider is None:
                st["error"] = "没有可用的模型 provider（总结/提炼共用同一个）"
                return
            timeout = max(20, int(self._cfg("summary_timeout_seconds", 120) or 120))
            legacy = int(plan.get("cursor") or 0)
            async with self._summary_lock:
                while not self._distill_stop:
                    chunk = self.store.distill_chunk(
                        subject, block, DISTILL_CHUNK_CHARS, legacy_after_id=legacy)
                    if not chunk:
                        break
                    rows = chunk["rows"]
                    # 不再硬截断：取多少条由 store 那边的字符预算决定，
                    # 截断会让「塞不进去的那半段」被静默丢掉却照样推进水位。
                    # 补录同样走编号协议：模型只认名单里的人。
                    n_lines, roster = number_transcript(rows)
                    roster_show, shown_of = self._roster_display(roster)
                    if not roster:
                        logger.warning(
                            "[SugarMemory] 这段补录数据没有可归属的说话人（名单为空）："
                            "本段不会写入任何记忆")
                    transcript = "\n".join(n_lines)
                    # 补录也在改口：每段现查一次底档，让前面几段刚落库的新事实
                    # 能被后面几段看见（不然同一批导入里，前一段说 21 岁、
                    # 后一段说 20 岁，两条会一起留下）。
                    related = ""
                    if self._cfg("summary_related_l2", True):
                        try:
                            limit = max(0, int(self._cfg("summary_related_limit", 16) or 0))
                            live = self.store.query_active_l2(subject=subject, limit=200)
                            related = format_related(
                                pick_related_l2(live, transcript[-1500:], limit),
                                shown_of)
                        except Exception:
                            logger.debug("[SugarMemory] 提炼组装相关记忆失败", exc_info=True)
                    raw, triple, err = "", None, None
                    # 一段最多试两次：连续失败多半是 provider 挂了，重试只会烧钱
                    for attempt in range(2):
                        try:
                            raw, triple = await self._summary_chat(
                                provider, transcript, timeout, related, roster_show)
                            err = None
                            break
                        except Exception as exc:  # noqa: BLE001
                            err = exc
                            if attempt == 0:
                                await asyncio.sleep(3)
                    if err is not None:
                        st["error"] = f"模型调用失败：{err}"
                        break
                    parsed = parse_summary_json(raw)
                    if parsed is None:
                        st["error"] = "模型返回的不是合法 JSON，这一段跳不过去，先停了"
                        break
                    if triple:
                        self._record_tokens("distill", input=triple[0], cached=triple[1],
                                            output=triple[2], subject=subject,
                                            session_id=f"import:{subject}")
                    else:
                        self._record_tokens(
                            "distill", source="estimate",
                            input=estimate_tokens(self._summary_system_prompt()
                                                  + transcript + related),
                            output=estimate_tokens(raw),
                            subject=subject, session_id=f"import:{subject}",
                        )
                    stats = apply_summary(
                        self.store, self.gate, parsed,
                        subject=subject, session_id=f"import:{subject}",
                        # 用这段原话里最新的时间：半年前的「19 岁」不能把现在的「20 岁」改回去
                        source_ts=float(rows[-1].get("created_at") or 0),
                        roster=roster,
                    )
                    self.store.distill_mark(subject, chunk["day"],
                                            int(rows[-1]["id"]), len(rows))
                    self.store.distill_tally(subject, items=len(rows))
                    st["done_blocks"] += 1
                    st["items"] += len(rows)
                    st["l2"] += int(stats.get("l2") or 0)
                    st["stale"] = int(st.get("stale") or 0) + int(stats.get("stale") or 0)
                    st["day"] = chunk["day"]
                    st["day_n"] = chunk["day_n"]
                    st["day_items"] = chunk["day_items"] + len(rows)
                    st["left"] = max(0, int(plan["left"]) - st["items"])
                    if not chunk["more_in_day"]:
                        st["done_days"] = int(st.get("done_days") or 0) + 1
                    logger.info(
                        f"[SugarMemory] 补录提炼 {subject}：{chunk['day']} "
                        f"第 {chunk['day_index']}/{chunk['total_days']} 天，"
                        f"{len(rows)} 条原话 → L2 {stats.get('l2', 0)} 条"
                    )
                    # 段间喘一口，别把对话用的额度挤干净
                    await asyncio.sleep(0.3)
            st["done"] = not st["error"] and not self._distill_stop
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SugarMemory] 补录提炼异常")
            st["error"] = str(exc)
        finally:
            st["running"] = False
            st["stopped"] = bool(self._distill_stop)
            if st.get("done") and not st.get("error"):
                try:
                    if self._cfg("alias_autogen", True):
                        asyncio.create_task(self._run_alias_batch())
                except Exception:
                    pass
            self._distilling = False
            self._distill_stop = False

    async def _api_chain(self):
        from quart import request as quart_request
        try:
            fid = int(quart_request.args.get("id") or 0)
        except ValueError:
            return self._err("无效的 id")
        if fid <= 0:
            return self._err("缺少 id")
        return self._ok({"chain": self.store.l2_chain(fid)})

    # ---------- 配置引导（只读：绝不替用户改框架主配置） ----------

    def _framework_config(self) -> tuple[dict, str, str]:
        """读「这台机器上真正生效」的 AstrBot 配置。

        优先级：运行时配置对象 context.get_config()（框架自己用的那一份）
        → 框架同源的 get_astrbot_data_path()/cmd_config.json。
        绝不猜实例根目录：launcher 迁移会在根目录留一份旧 cmd_config.json，
        之前把它当真值，得出过「压缩没触发」的错误结论。
        """
        raw: dict = {}
        source, detail = "unavailable", "没读到框架配置"
        try:
            obj = self.context.get_config()
            if isinstance(obj, dict):
                raw = dict(obj)
                source = "runtime"
                detail = "AstrBot 运行时配置对象（和框架发请求用的是同一份）"
                p = getattr(obj, "config_path", None)
                if p:
                    detail += f"｜{Path(str(p)).name}"
        except Exception as e:
            logger.debug(f"[SugarMemory] 运行时配置读取失败：{e}")
        if not raw and get_astrbot_data_path:
            try:
                p = Path(str(get_astrbot_data_path())) / "cmd_config.json"
                if p.exists():
                    raw = json.loads(p.read_text(encoding="utf-8-sig"))
                    source = "file"
                    detail = f"框架配置路径 {p.name}（运行时对象不可用时的同源兜底）"
            except Exception as e:
                logger.debug(f"[SugarMemory] 配置兜底读取失败：{e}")
        return raw, source, detail

    # ---------------------- 注入包设置体检 ----------------------
    def _pack_audit(self) -> list:
        """逐项体检注入包相关配置：现在这样设会出什么问题、健康长什么样。

        只读 self._cfg()，不改任何配置。判定区间按本机实测定的保守范围
        （51 条事实的库、常驻包 700~900 tok），不是理论上的「越大越好」。
        """
        def _i(key: str, default: int) -> int:
            try:
                return int(self._cfg(key, default))
            except (TypeError, ValueError):
                return int(default)

        def _b(key: str, default: bool) -> bool:
            return bool(self._cfg(key, default))

        def _s(key: str, default: str) -> str:
            value = self._cfg(key, default)
            return str(value if value is not None else default).strip()

        rows: list = []

        def add(key, label, value, level, risk, healthy):
            rows.append({"key": key, "label": label, "value": value,
                         "level": level, "risk": risk, "healthy": healthy})

        # ---- 1 总开关 ----
        if not _b("inject_enabled", True):
            add("inject_enabled", "记忆包注入总开关", "关闭（影子模式）", "warn",
                "注入 0 字节：记忆照样在攒，但一句都不进 system。面板上五十多条，"
                "对话里一条都用不上，表现是「它怎么什么都不记得」。",
                "开启。只有排查「是不是记忆干扰了回复」时才临时关。")
        else:
            add("inject_enabled", "记忆包注入总开关", "开启", "good",
                "没有副作用，这就是正常工作状态。", "开启。")

        mode = _s("inject_mode", "frozen") or "frozen"
        if mode != "frozen":
            add("inject_mode", "记忆包注入架构", mode, "bad",
                "dynamic 是旧版按原话实时检索：问法换个字、少打两个字，块就变，"
                "messages[0] 的第一个字节每轮都不同，前缀缓存基本不命中。",
                "frozen。切换会让所有活跃分片各重建一次，属预期。")
        else:
            add("inject_mode", "记忆包注入架构", "frozen", "good",
                "按（人, 会话）冻结常驻包，闲聊轮次一个字节都不动。", "frozen。")

        if not _b("inject_dedup", True):
            add("inject_dedup", "冻结常驻记忆包", "关闭", "bad",
                "冻结的总闸被关：每轮无条件重算常驻包，内容一模一样也算新块。"
                "症状很固定——轮次恒 0、重建原因写着 dedup_off、命中率整片 0%。",
                "开启。这是「冻结」唯一的总闸，没有第二个开关。")
        else:
            add("inject_dedup", "冻结常驻记忆包", "开启", "good",
                "内容不变就原样复用，不重算、不重发。", "开启。")

        # ---- 2 预算与条数 ----
        budget = _i("inject_budget_tokens", 1200)
        if budget < 300:
            add("inject_budget_tokens", "注入预算", f"{budget} tok", "bad",
                "预算比身份底档还小：超预算会「突破保底」继续砍，长期事实可能只剩 1 条，"
                "问什么都缺记忆。",
                "600~1600，且比面板上的实测体积多留 20~30%。")
        elif budget < 600:
            add("inject_budget_tokens", "注入预算", f"{budget} tok", "warn",
                "余量太薄：库一长（事实变多、转述进来）就开始触发裁剪，"
                "症状是「有时记得、有时不记得」，还查不出原因。",
                "600~1600，且比面板上的实测体积多留 20~30%。")
        elif budget <= 2400:
            add("inject_budget_tokens", "注入预算", f"{budget} tok", "good",
                "预算是天花板不是目标：实际多大由记忆库决定，调大不会多花一分钱。",
                "600~1600 起步，调大条数配额时同步提到 1400~1600。")
        else:
            add("inject_budget_tokens", "注入预算", f"{budget} tok", "warn",
                "远高于常见体积：不会更聪明，只会让块更容易变厚、更占上下文窗口；"
                "真正决定召回的是下面几项配额，不是这个数。",
                "600~1600。留余量靠实测体积 + 20%，不是靠猜一个大数。")

        l2_limit = max(1, _i("pack_l2_limit", 12))
        if l2_limit < 4:
            add("pack_l2_limit", "长期事实上限", f"{l2_limit} 条", "bad",
                "常驻底档太薄：连「这个人是谁、住哪、干什么」都放不下，"
                "每轮都得靠临时检索补，闲聊时几乎等于没有长期记忆。",
                "8~16 条（默认 12）。")
        elif l2_limit < 8:
            add("pack_l2_limit", "长期事实上限", f"{l2_limit} 条", "warn",
                "偏薄：身份、习惯、长期状态挤在几条里，预算一紧先掉的就是它们。",
                "8~16 条（默认 12）。")
        elif l2_limit <= 24:
            add("pack_l2_limit", "长期事实上限", f"{l2_limit} 条", "good",
                "12~16 是实测最稳的一段：够装基本盘，又不会把低相关事实顶进常驻前缀。",
                "8~16 条（默认 12）。")
        else:
            add("pack_l2_limit", "长期事实上限", f"{l2_limit} 条", "warn",
                "条数太多：低相关事实会挤进常驻前缀，白占 token 还稀释注意力。",
                "8~16 条（默认 12）。")

        keep = _i("pack_l2_keep_min", 3)
        if keep < 1:
            add("pack_l2_keep_min", "长期事实保底条数", f"{keep} 条", "bad",
                "超预算时一条事实都不保，长期事实段可能整段消失。",
                "3~5 条，且不大于长期事实上限。")
        elif keep > l2_limit:
            add("pack_l2_keep_min", "长期事实保底条数",
                f"{keep} 条（上限只有 {l2_limit}）", "warn",
                "保底比上限还大：先按上限取材，再谈保底，所以这一项实际不生效。",
                "3~5 条，且不大于长期事实上限。")
        elif keep < 3:
            add("pack_l2_keep_min", "长期事实保底条数", f"{keep} 条", "warn",
                "只保 1~2 条：预算一紧就只剩名字，问什么都是「你是……」开头。",
                "3~5 条，且不大于长期事实上限。")
        elif keep <= 8:
            add("pack_l2_keep_min", "长期事实保底条数", f"{keep} 条", "good",
                "再挤也保住「这个人是谁」的基本盘。",
                "3~5 条，且不大于长期事实上限。")
        else:
            add("pack_l2_keep_min", "长期事实保底条数", f"{keep} 条", "warn",
                "保底太高：预算紧时保底反而压过别的段，聊天原文和摘要先被挤空。",
                "3~5 条，且不大于长期事实上限。")

        ratio = _i("pack_l2_rel_ratio", 30)
        if ratio <= 0:
            add("pack_l2_rel_ratio", "相关性门槛", "0（关闭过滤）", "warn",
                "等于关掉相关性过滤：一句「在吗」也能把十几条无关事实拽进包，"
                "AI 会在打招呼里强行背资料卡。",
                "20~40（默认 30）。")
        elif ratio < 15:
            add("pack_l2_rel_ratio", "相关性门槛", f"{ratio}%", "warn",
                "门槛太松：命中数不到头名一成多的都留着，闲聊轮次会垫进一堆不相关记忆。",
                "20~40（默认 30）。")
        elif ratio <= 45:
            add("pack_l2_rel_ratio", "相关性门槛", f"{ratio}%", "good",
                "只留命中数达到头名这个比例的事实，噪声和召回兼顾。",
                "20~40（默认 30）。")
        elif ratio <= 70:
            add("pack_l2_rel_ratio", "相关性门槛", f"{ratio}%", "warn",
                "偏严：冷门问题容易只剩保底那几条，看着有记忆却答不上细节。",
                "20~40（默认 30）。")
        else:
            add("pack_l2_rel_ratio", "相关性门槛", f"{ratio}%", "bad",
                "过滤过狠：多数轮次只剩保底条数，等于关掉了按话题召回，"
                "只剩那几条身份卡反复出现。",
                "20~40（默认 30）。")

        # ---- 3 召回通道 ----
        if not _b("pack_query_expand", True):
            add("pack_query_expand", "问法归一（改写补召回）", "关闭", "warn",
                "「多大了」「搞什么创作」这类改写问法字面撞不上任何一条，"
                "召回直接掉一半以上（实测改写问法 90%+ → 20%）。",
                "开启。关掉等于只认原话里的字。")
        else:
            add("pack_query_expand", "问法归一（改写补召回）", "开启", "good",
                "口语问法先映射成概念词再检索，补住字面匹配的盲区。", "开启。")

        rounds = _i("pack_context_rounds", 3)
        if rounds <= 0:
            add("pack_context_rounds", "最近几轮参与检索", "0（关闭）", "warn",
                "「那它呢 / 后来呢」接不上上一轮：本轮字面零命中，包要么空、要么只剩保底，"
                "追问等于重新开始。",
                "2~4 轮（默认 3）。")
        elif rounds <= 6:
            add("pack_context_rounds", "最近几轮参与检索", f"{rounds} 轮", "good",
                "指代追问能接上，又不会把上一个话题拖太远。", "2~4 轮（默认 3）。")
        else:
            add("pack_context_rounds", "最近几轮参与检索", f"{rounds} 轮", "warn",
                "轮数太多：上一话题的记忆会被继续拽进新话题，"
                "表现是「都聊别的了还挂着前面的事」。",
                "2~4 轮（默认 3）。")

        cand = _i("pack_l2_candidate_limit", 200)
        if cand < 50:
            add("pack_l2_candidate_limit", "长期事实候选池", f"{cand} 条", "warn",
                "候选池太小：低置信但本轮正好相关的事实进不了候选，漏召回会重新出现。",
                "150~300（默认 200）。")
        elif cand <= 400:
            add("pack_l2_candidate_limit", "长期事实候选池", f"{cand} 条", "good",
                "先把够多候选捞出来，再按本轮相关度挑，低置信但相关的才有机会被想起。",
                "150~300（默认 200）。")
        else:
            add("pack_l2_candidate_limit", "长期事实候选池", f"{cand} 条", "warn",
                "候选太多：每轮都要打分，200 条约 10ms 级，再往上收益不明显、CPU 白烧。",
                "150~300（默认 200）。")

        refresh = _b("inject_topic_refresh", True)
        if not refresh:
            add("inject_topic_refresh", "话题命中时刷新常驻包", "关闭", "warn",
                "纯冻结：常驻包外的答案（某段经历、某条包外事实）永远进不来，"
                "只能靠常驻那几条答，问「我以前干过什么」就答不全。",
                "开启。闲聊轮次它不会动包，只在真聊到包外记忆时才重建。")
        else:
            add("inject_topic_refresh", "话题命中时刷新常驻包", "开启", "good",
                "平时一个字节都不动，真聊到包外记忆才补一次，命中率和召回都不亏。",
                "开启。")

        extra = _i("inject_topic_extra", 3)
        if not refresh and extra > 0:
            add("inject_topic_extra", "话题刷新最多补几条", f"{extra} 条", "info",
                "上面的刷新总闸关着，这一项不生效。", "跟随话题刷新开关。")
        elif extra == 0:
            add("inject_topic_extra", "话题刷新最多补几条", "0 条", "warn",
                "话题触发时只刷新、不补事实——包重建了一次，答案还是没进来，等于白重建。",
                "3~5 条（默认 3）。")
        elif extra <= 6:
            add("inject_topic_extra", "话题刷新最多补几条", f"{extra} 条", "good",
                "用「替换最弱常驻条」的方式补，不撑大包、不额外烧预算。",
                "3~5 条（默认 3）。")
        else:
            add("inject_topic_extra", "话题刷新最多补几条", f"{extra} 条", "warn",
                "补太多：替换最弱常驻条时动的前缀更多，这一次重建的代价变大。",
                "3~5 条（默认 3）。")

        # ---- 4 其余段 ----
        l3s = _i("pack_l3_snippets", 0)
        l3k = _i("pack_l3_keep_min", 0)
        if l3s == 0:
            add("pack_l3_snippets", "附带聊天原文", "0 条", "good",
                "框架上下文里本来就有最近几条原文，再带一遍是二次花 token。",
                "0；只有在需要「原话怎么说的」时才加。")
        else:
            add("pack_l3_snippets", "附带聊天原文", f"{l3s} 条", "info",
                "带原文会和框架已有的历史重复，只在追问原话时才值这几百 token。",
                "0；需要时 1~2 条。")
        if l3k > l3s:
            add("pack_l3_keep_min", "聊天原文保底条数",
                f"{l3k} 条（上限只有 {l3s}）", "warn",
                "保底比上限还大：实际按上限取，这一项不生效。",
                "不大于「附带聊天原文」的条数。")
        else:
            add("pack_l3_keep_min", "聊天原文保底条数", f"{l3k} 条", "good",
                "超预算时按这个数给原文留底，砍到 0 就是允许整段消失。",
                "0~1，且不大于「附带聊天原文」。")

        skel = _i("inject_skeleton_limit", 2)
        if skel == 0:
            add("inject_skeleton_limit", "带几条旧事骨架", "0 条", "info",
                "骨架不进包：问「咱们以前聊过啥」没有索引可查，只能凭常驻事实。",
                "1~3 条（一行约 30 token）。")
        else:
            add("inject_skeleton_limit", "带几条旧事骨架", f"{skel} 条", "good",
                "一行约 30 token 换一个话题索引，问「以前聊过啥」时至少有的指。",
                "1~3 条。")

        cross = _i("inject_cross_limit", 5)
        if cross == 0:
            add("inject_cross_limit", "带几条其他会话的近况", "0 条", "info",
                "只认本会话：别处发生的事不会串进来，保守但安全。",
                "0~5 条（默认 5）。")
        elif cross <= 8:
            add("inject_cross_limit", "带几条其他会话的近况", f"{cross} 条", "good",
                "按本轮话题筛过才给，还会标出是谁说的，不会乱串。",
                "0~5 条（默认 5）。")
        else:
            add("inject_cross_limit", "带几条其他会话的近况", f"{cross} 条", "warn",
                "跨会话带太多：别处的事会串台，AI 容易把别人说的按到当前人头上。",
                "0~5 条（默认 5）。")

        relay = _i("inject_outbound_relay_limit", 3)
        if relay <= 0:
            add("inject_outbound_relay_limit", "带几条转述过的话", "0 条", "warn",
                "转述段整段不进包：问「你刚才替我带了什么话」时 AI 只能凭记忆，"
                "而它没有这段记忆——容易把真发过的话当幻觉否认。",
                "2~4 条（默认 3）。")
        elif relay <= 6:
            add("inject_outbound_relay_limit", "带几条转述过的话", f"{relay} 条", "good",
                "发到当前会话的转述排在最前，被追问时有据可查。", "2~4 条（默认 3）。")
        else:
            add("inject_outbound_relay_limit", "带几条转述过的话", f"{relay} 条", "warn",
                "带太多：转述段会挤占事实和摘要的预算，而真正常被追问的只有最近那两三条。",
                "2~4 条（默认 3）。")

        cron = _i("inject_outbound_cron_limit", 2)
        if cron <= 0:
            add("inject_outbound_cron_limit", "带几条定时消息", "0 条", "warn",
                "定时任务不进包：问「你提醒过我什么」时答不上，也容易否认自己发过提醒。",
                "1~3 条（默认 2）。")
        elif cron <= 4:
            add("inject_outbound_cron_limit", "带几条定时消息", f"{cron} 条", "good",
                "只带当前会话最近的定时消息，和转述分开计数，不会被说成「替你带的话」。",
                "1~3 条（默认 2）。")
        else:
            add("inject_outbound_cron_limit", "带几条定时消息", f"{cron} 条", "warn",
                "带太多：闹钟/提醒会淹没真正该答的事实，还占预算。", "1~3 条（默认 2）。")

        if not _b("inject_scope_subject", True):
            add("inject_scope_subject", "记忆包按人过滤", "关闭", "bad",
                "全库检索：别人说过的事会按到当前说话人头上（问「我有什么病」"
                "答出别人的病）。多人共用一个机器人时基本必错。",
                "开启。")
        else:
            add("inject_scope_subject", "记忆包按人过滤", "开启", "good",
                "只注入当前说话人的事实和本会话摘要，别人说的不会按到他头上。",
                "开启。")

        sem_on = _b("pack_semantic_fallback", False)
        sem_pid = _s("semantic_provider_id", "")
        if sem_on and not sem_pid:
            add("pack_semantic_fallback", "语义兜底",
                "开着 · Provider 没填", "warn",
                "开关开着但没接线：整条语义路静默跳过——零调用、零开销，也零效果。"
                "以为自己开了语义兜底，其实一直靠字面在跑。",
                "要开就两个都填：开关 + 支持 embedding 的 Provider ID。")
        elif sem_on and sem_pid:
            add("pack_semantic_fallback", "语义兜底", f"开着 · {sem_pid}", "good",
                "字面/概念/别名三层全零命中时才调一次向量接口，超时或报错自动退回。",
                "开关 + Provider 都填，超时 1500ms 足够。")
        elif sem_pid and not sem_on:
            add("pack_semantic_fallback", "语义兜底",
                f"关着 · Provider 已填", "info",
                "填了 Provider 但开关关着，一次都不会调用，一分钱不花。",
                "不用就两个都空着；要用就两个都填。")
        else:
            add("pack_semantic_fallback", "语义兜底", "未启用", "info",
                "改写问法靠「问法归一」兜；语义类比（打字很响的设备 → 机械键盘）"
                "这种完全不同字的，兜不住。",
                "零依赖用法可以一直关着。")

        resident = _i("inject_l2_resident_limit", 0)
        if resident == 0:
            add("inject_l2_resident_limit", "常驻事实条数",
                f"0（等于上限 {l2_limit} 条）", "info",
                "全部长期事实都常驻，前缀最长但也最全。",
                "12 条以内可以保持 0；库大时设 6~8 更省。")
        elif resident > l2_limit:
            add("inject_l2_resident_limit", "常驻事实条数",
                f"{resident} 条（上限只有 {l2_limit}）", "warn",
                "大于长期事实上限：实际按上限取，这一项不生效。",
                "6~8 条，且不大于长期事实上限。")
        elif resident >= l2_limit:
            add("inject_l2_resident_limit", "常驻事实条数",
                f"{resident} 条（= 上限）", "info",
                "等于长期事实上限，也就是全部常驻——这一项没起到收缩作用，"
                "只是把默认值写明了。",
                "6~8 条比上限小，才真的把经历类退到按话题召回。")
        elif resident >= 6:
            add("inject_l2_resident_limit", "常驻事实条数", f"{resident} 条", "good",
                "只常驻身份和习惯，经历类退到按话题召回，闲聊时不再白占 token。",
                "6~8 条，且不大于长期事实上限。")
        else:
            add("inject_l2_resident_limit", "常驻事实条数", f"{resident} 条", "warn",
                "常驻面太窄：包里每轮只剩这几条，稍微问偏一点就得靠临时补，"
                "而临时补会改前缀。",
                "6~8 条，或 0（不额外限制）。")

        if _b("inject_relations", False):
            add("inject_relations", "关系快照进包",
                f"开启 · 最多 {_i('inject_relation_limit', 6)} 条", "info",
                "关系快照参与「包变没变」的判定：关系一更新就重建一次，"
                "前缀不动、只重算尾部。不是错误，只是多一次重建。",
                "要用就开着；介意多几次重建就关掉。")
        else:
            add("inject_relations", "关系快照进包", "关闭", "good",
                "判定只看常驻底档，重建次数更少；关系仍可在「人际关系」页查。",
                "要用就开着；介意多几次重建就关掉。")

        return rows

    async def _api_guide(self):
        """配置引导：自动体检宿主的上下文压缩策略 + 讲清记忆包上下限 + 预览词典。

        这里只读，不写 AstrBot 主配置，也不写插件配置。别人装了这个插件，
        打开引导页就能看到他自己那台机器上的真实值，而不是我们写死的示例。
        """
        raw, source, detail = self._framework_config()
        ar = raw.get("agent_runner") if isinstance(raw, dict) else None
        ar = ar if isinstance(ar, dict) else {}
        acfg = ar.get("config") if isinstance(ar.get("config"), dict) else {}
        comp = acfg.get("compression") if isinstance(acfg.get("compression"), dict) else {}
        ps = raw.get("provider_settings") if isinstance(raw.get("provider_settings"), dict) else {}
        runner = str(ar.get("runner_type") or "unknown")
        supported = bool(comp)

        def _ci(key: str, default: int) -> int:
            try:
                return int(comp.get(key, default))
            except (TypeError, ValueError):
                return default

        max_turns = _ci("max_turns", -1)
        trim_turns = max(1, _ci("trim_turns", 1))
        strategy = str(comp.get("overflow_strategy") or "")
        fallback = _ci("fallback_max_tokens", 128000)
        dt_on = bool(ps.get("datetime_system_prompt", True))
        prefix = str(ps.get("prompt_prefix") or "{{prompt}}")
        keep_ratio = comp.get("keep_recent_ratio", 0.15)

        window = 0
        providers = raw.get("provider") if isinstance(raw, dict) else None
        if isinstance(providers, list):
            for pv in providers:
                if isinstance(pv, dict):
                    try:
                        w = int(pv.get("max_context_tokens") or 0)
                    except (TypeError, ValueError):
                        w = 0
                    if w > 0:
                        window = w
                        break
        if window <= 0:
            window = fallback

        def ck(key, label, path, cur, cur_label, rec, rec_label,
               level, why, effect):
            return {"key": key, "label": label, "path": path,
                    "current": cur, "current_label": cur_label,
                    "recommended": rec, "recommended_label": rec_label,
                    "level": level, "why": why, "effect": effect}

        checks: list[dict] = []
        where = "AstrBot 管理面板 → 设置 → 上下文管理策略（agent_runner.config.compression）"

        if not supported:
            checks.append(ck(
                "compression", "上下文管理策略", "agent_runner.config.compression",
                "未检测到", "这个版本没有这组配置", "—", "—", "info",
                "通常是 AstrBot 版本较旧、或用了非 local runner；框架仍可能按旧字段裁剪历史。"
                "下面几项能读到的检查照常有效，缺失的项目请按自己版本的设置页对照。",
                "无法自动判断压缩策略，请手动确认历史轮数上限。"))
        else:
            if max_turns == -1:
                mt_level, mt_rec, mt_rec_label = "good", -1, "-1（不按轮数截断）"
                mt_why = ("已关闭按轮数截断：历史只追加，messages 前缀不会逐轮位移，"
                          "对前缀缓存最友好。代价是历史会一直长，最后靠 token 窗口兜底压缩。")
            elif max_turns >= 30:
                mt_level, mt_rec, mt_rec_label = "good", max_turns, f"{max_turns}（够用）或 -1"
                mt_why = (f"常见会话长度一般到不了 {max_turns} 轮，这段区间里不会发生轮数裁剪，"
                          "前缀是稳定的。但一旦聊过这个数，之后每轮都会开始位移。")
            elif max_turns >= 20:
                mt_level, mt_rec, mt_rec_label = "warn", -1, "-1（或 ≥30）"
                mt_why = (f"超过 {max_turns} 轮就开始每轮裁掉最老一轮，长一点的连续聊天很容易够到；"
                          "前缀一旦开始位移，provider 缓存就会整段作废。")
            else:
                mt_level, mt_rec, mt_rec_label = "bad", -1, "-1（或 ≥30）"
                mt_why = (f"{max_turns} 轮太小：普通连续聊天十几句就会触发轮数裁剪，"
                          "之后每轮历史都位移一段，前缀缓存基本留不住。")
            checks.append(ck(
                "max_turns", "压缩前最多保留对话轮数", "agent_runner.config.compression.max_turns",
                max_turns, "-1 = 不限制" if max_turns == -1 else f"{max_turns} 轮",
                mt_rec, mt_rec_label, mt_level, mt_why,
                "决定「会话会不会开始逐轮改写历史前缀」，是影响命中率最大的一项。"))

            if max_turns > 1:
                eff_keep = max(1, max_turns - min(trim_turns, max_turns - 1) + 1)
                trim_why = (
                    "这里要纠一个常见误解：它不是「每 N 轮才裁一次」。超限时框架把历史裁成 "
                    f"max_turns − trim_turns + 1 = {eff_keep} 轮，之后每聊一轮仍然会位移一次。"
                    "调大它只会让超限后的上下文更短，并不会让缓存间歇命中。")
            else:
                trim_why = ("当前没有启用轮数截断，这个值暂时不生效；"
                            "等哪天把 max_turns 从 -1 改回有限值，它才会参与「裁完剩几轮」的计算。")
            checks.append(ck(
                "trim_turns", "轮次超限时一次丢弃轮数", "agent_runner.config.compression.trim_turns",
                trim_turns, f"{trim_turns} 轮", "按需", "1（对缓存无影响，按上下文长度取舍）",
                "info", trim_why,
                "只影响超限后保留多少轮；不决定是否位移，也不改善前缀缓存。"))

            if strategy == "llm_compress":
                st_level = "bad"
                st_why = ("每次让 LLM 重写历史摘要，摘要文本每轮都不同、还会多烧一次模型调用；"
                          "而且本插件已经有会话摘要，功能重复。")
            elif strategy == "truncate_by_turns":
                st_level = "good"
                st_why = ("按轮数截断是确定性的：同一批历史每次裁出来一样，"
                          "至少不会像 LLM 摘要那样无端改写字节。")
            else:
                st_level = "warn"
                st_why = "不是框架内置的两种策略之一，建议改成「按对话轮数截断」。"
            checks.append(ck(
                "overflow_strategy", "历史超限时的处理方式",
                "agent_runner.config.compression.overflow_strategy",
                strategy or "未设置",
                {"llm_compress": "由 LLM 压缩上下文",
                 "truncate_by_turns": "按对话轮数截断"}.get(strategy, strategy or "未设置"),
                "truncate_by_turns", "按对话轮数截断", st_level, st_why,
                "LLM 压缩会让历史前缀每轮都变，且额外产生一次 LLM 请求。"))

            fb_level = "warn" if 0 < fallback < 32000 else "good"
            checks.append(ck(
                "fallback_max_tokens", "上下文窗口兜底值",
                "agent_runner.config.compression.fallback_max_tokens",
                fallback, f"{fallback:,} tok", fallback,
                "按模型真实窗口填写", fb_level,
                (f"当前 token 压缩触发线约 {int(window * 0.82):,} tok（窗口 × 0.82）。"
                 "兜底值设得比模型真实窗口小很多，会在远没到上限时就频繁触发压缩。"),
                "只在模型没写 max_context_tokens、也不在内置元数据里时生效。"))
            if runner and runner != "local":
                checks.append(ck(
                    "runner_type", "Agent 运行器", "agent_runner.runner_type",
                    runner, runner, "local", "local", "info",
                    "上面这组 compression 配置是 local runner 专用的；当前运行器可能不读它。",
                    "非 local 时请以该运行器自己的上下文策略为准。"))

        dt_check = ck(
            "datetime_system_prompt", "当前时间注入", "provider_settings.datetime_system_prompt",
            dt_on, "开启" if dt_on else "关闭", "保持现状", "保持现状",
            "good" if dt_on else "info",
            "框架把「Current datetime」挂在每条 user 消息的尾部，不在 system 前缀里，"
            "不会破坏前缀缓存。不用为了缓存专门关掉它。",
            "对缓存无害；关掉只是少一个时间锚。")
        dyn_prefix = any(t in prefix for t in ("{{datetime}}", "{{date}}", "{{time}}", "{{now}}"))
        prefix_check = ck(
            "prompt_prefix", "系统提示词前缀模板", "provider_settings.prompt_prefix",
            prefix, prefix, "{{prompt}}", "{{prompt}}（纯占位，不带动态变量）",
            "bad" if dyn_prefix else "good",
            ("这个模板排在所有 system 内容的最前面。里面只要塞了按时间变化的变量，"
             "每一轮第一个 token 就不同，后面整段前缀缓存全部作废。" if dyn_prefix else
             "当前是纯占位写法，system 最前面是静态的，前缀缓存可以正常复用。"),
            "决定整段 system 的第一个字节是否稳定，是缓存生死线。")
        # 配置完全读不到时，不能把「默认 True / 默认 {{prompt}}」当成这台机器的
        # 实测值报出去 —— 那等于猜。只保留「未读到」这一条明确提示。
        if raw:
            checks.extend([dt_check, prefix_check])
        else:
            checks.append(ck(
                "config_unavailable", "框架配置", "cmd_config.json",
                "未读到", "运行时配置对象和同源文件都没读到", "—", "检查插件运行环境",
                "warn", "无法自动判断上下文策略；请对照 AstrBot 版本和 WebUI 里的实际设置。",
                "这一页的逐项建议暂时无法自动校核，只保留通用说明。"))

        bad = sum(1 for c in checks if c["level"] == "bad")
        warn = sum(1 for c in checks if c["level"] == "warn")
        if not raw:
            verdict = "连框架配置对象都没读到（运行时对象与同源配置文件都不可用）。这一页只给通用原理，请对照你的 AstrBot 版本和 WebUI 里的实际设置手动检查。"
        elif not supported:
            verdict = "这台机器的上下文策略没读到完整配置，先按下面能读到的项检查；其余请对照自己的 AstrBot 版本。"
        elif bad:
            verdict = f"有 {bad} 项会直接破坏前缀缓存（另有 {warn} 项需留意）。插件侧已经把记忆包冻结住了，剩下这些在框架层，按建议改完再连续聊 20~30 轮看有效命中率。"
        elif warn:
            verdict = f"没有致命项，但有 {warn} 项偏保守。常见会话长度内基本稳定，长会话仍可能触发逐轮裁剪。"
        else:
            verdict = "上下文策略对前缀缓存已经很友好。记忆包本身也是冻结的，剩下的 0% 多半是冷启动、长间隔自然过期或 provider 上报口径。"

        lines = ["【AstrBot 上下文策略建议 · 由蔗糖的记忆中枢生成】"]
        for c in checks:
            if c["level"] in ("bad", "warn"):
                lines.append(f"· {c['label']}（{c['path']}）：当前 {c['current_label']} → 建议 {c['recommended_label']}")
        if len(lines) == 1:
            lines.append("· 当前没有需要改动的项，保持现状即可。")
        lines.append("")
        lines.append("【本次检测到的当前值】")
        lv_label = {"good": "良好", "warn": "建议调整", "bad": "会破坏缓存", "info": "说明"}
        for c in checks:
            lines.append(f"· {c['label']}（{c['path']}）= {c['current_label']}［{lv_label.get(c['level'], c['level'])}］")
        lines.append("· 改完请连续聊 20~30 轮，到「用量 → 缓存命中率」看记忆包命中率与有效命中率，不要看单轮 0%/99%。")
        lines.append("· 本页只读，不会替你修改 AstrBot 主配置。")

        def _pint(key: str, default: int) -> int:
            try:
                return max(0, int(self._cfg(key, default)))
            except (TypeError, ValueError):
                return default

        def _pbool(key: str, default: bool) -> bool:
            return bool(self._cfg(key, default))

        # 体检失败不能把整页带下去：最坏情况是这一块空着，其余照常显示。
        try:
            audit_rows = self._pack_audit()
        except Exception as e:
            logger.debug(f"[SugarMemory] 注入包设置体检失败：{e}")
            audit_rows = []

        limits = {
            "budget": _pint("inject_budget_tokens", 1200),
            "mode": self._inject_mode(),
            "enabled": _pbool("inject_enabled", True),
            "dedup": _pbool("inject_dedup", True),
            "l2_limit": max(1, _pint("pack_l2_limit", 12)),
            "l2_keep_min": max(1, _pint("pack_l2_keep_min", 3)),
            "rel_ratio": max(0, _pint("pack_l2_rel_ratio", 30)),
            "l3_keep_min": max(0, _pint("pack_l3_keep_min", 0)),
            "summary_floor_chars": 60,
            "window": window,
            "token_trigger": int(window * 0.82),
            "upper": [
                "上限 = inject_budget_tokens，是「最多允许塞多少」的天花板，不是目标体积。"
                "实际多大由记忆库和各项配额决定，预算调大不会多花一分钱。",
                "各段还有自己的条数上限：L2 事实 pack_l2_limit、转述 / 定时 / L3 摘录配额等，"
                "先按这些上限取材，再按总预算裁剪。",
                "超过预算会按「信息价值从低到高」依次让位，并把裁剪原因写进日志与预览的 trim_notes。",
            ],
            "trim_order": [
                "1 · L3 原话事件（框架上下文里本来就有，属于重复花 token）→ 退到 pack_l3_keep_min",
                "2 · 会话摘要（逐步压短，压到 60 字停手）",
                "3 · 不带标签的外发回声 → 4 · 旧事索引（可砍到 0）",
                "5 · L2 长期事实（退到 pack_l2_keep_min，砍了补不回来，所以排得靠后）",
                "6 · 跨会话杂闻 → 7 · 带标签的定时任务 / 转述（最后的凭据）",
                "8 · 兜底：仍超预算就突破保底，L2 最少可能只剩 1 条——这就是「预算调太低会毁召回」的原因",
            ],
            "lower": [
                "下限首先是「可以完全没有」：inject_enabled 默认开，手动关成影子模式时"
                "注入 0 字节；没有任何记忆时，块里只剩写死的头尾标签。",
                "裁剪保底：L3 至少 pack_l3_keep_min 条、L2 至少 pack_l2_keep_min 条、"
                "摘要压到 60 字就不再压、被点名的人至少留 1 条档案。",
                "复用规则：frozen 模式下这一轮算出来的块，跟上一轮真正挂出去的"
                "一模一样就原样复用，没有「必须等 N 秒 / N 轮」的限制；只要新写入 / 改写 "
                "L2、本轮补进了包外相关事实，或任何一段渲染结果变了，下一轮立即重建。"
                "重建时 prefix 段（前几条 L2）字节不动，所以前缀缓存照旧命中，"
                "代价只有尾部那几条。",
                "缓存侧还有 provider 自己的下限：必须支持前缀缓存，且常见 TTL 5~10 分钟、"
                "按块对齐（例如 64 token）。低于这些条件，块再稳也报不出命中。",
            ],
            "iron_rules": [
                "注入路径只有一条：追加到 system_prompt 末尾（messages[0]），不写 extra_user_content_parts、"
                "不碰 user 角色槽位——否则等于把记忆伪装成用户发言，模型可能拿它当指令越权。",
                "frozen 模式按（人,会话）冻结 query=\"\" 的常驻包：本轮原话默认不进 system，"
                "只影响提及统计；但话题命中 / 指代追问时会触发一次增量刷新，把答案补进这一轮。",
                "dynamic 模式是旧版按原话实时检索，每轮都变、前缀基本不命中，只用于 A/B 对照或紧急回退。",
            ],
            # 逐项体检：读的是这台机器的真实配置，不是示例。前端只负责渲染。
            "audit": audit_rows,
        }
        return self._ok({
            # 判定是这次请求现算的；带上时间，面板上就能一眼确认「这份体检是不是刚读的」。
            "read_at": time.strftime("%H:%M:%S"),
            "read_ts": int(time.time()),
            "framework": {
                "ok": bool(raw), "source": source, "source_detail": detail,
                "runner_type": runner, "compression_supported": supported,
                "compression": {"max_turns": max_turns, "trim_turns": trim_turns,
                                "overflow_strategy": strategy,
                                "fallback_max_tokens": fallback,
                                "keep_recent_ratio": keep_ratio},
                "settings": {"datetime_system_prompt": dt_on, "prompt_prefix": prefix},
                "checks": checks, "bad": bad, "warn": warn,
                "verdict": verdict, "recommend_text": "\n".join(lines),
                "where": where, "read_only": True,
                "mechanism": (
                    "框架每次发请求前都会检查历史轮数。超过 max_turns 后，它把历史裁成 "
                    "max_turns − trim_turns + 1 轮；此后每聊一轮就挤掉最老一条、追加新一条，"
                    "整个 messages 序列从头开始逐轮位移，前缀缓存从分叉点往后全部作废。"
                    "trim_turns 只决定「裁完剩几轮」，不决定「会不会位移」；"
                    "真正决定命中率的是会话轮数有没有够到 max_turns，以及 token 用量有没有到窗口的 "
                    f"0.82 倍（当前约 {int(window * 0.82):,} tok）。"),
            },
            "limits": limits,
            "sections": GUIDE_SECTIONS,
            "notes": [
                "配置引导只读：不修改 AstrBot 主配置，也不修改插件配置。所有建议都要你自己去框架设置页改。",
                "「记忆包命中率」和「整轮请求命中率」是两个数：前者看插件把块冻得稳不稳，"
                "后者还包含框架 system、历史消息、工具 schema；框架裁剪历史只会拉低后者。",
                "首轮、插件重载后的第一轮、距上轮超过 10 分钟的轮次，0% 是固定正常的，"
                "用量页已把这些单独打标并剔除出有效样本。",
            ],
        })

    async def _api_pack(self):
        """面板「注入预览」。

        2026-11-06 A 方案后这里必须分两种模式：真实注入的是按 (who,sess)
        冻结的 query="" 常驻底档。0.16.0 起 B 方案还会在这份底档上叠加
        「本轮话题补进来的几条事实」，所以它并非跟用户这轮完全无关——
        但注入块的渲染口径始终是 query=""（字节稳定），统计轨那次按原话
        实时检索不会进 prompt。旧实现直接拿 query 跑 build() 再展示，会把
        「模拟出来的动态包」当成 system_prompt 里真正的内容。

        live（默认）：展示运行时冻结缓存里真正在用的块（可能含话题增量）
        + 节流状态，另附一份「即时候选」＝不含话题增量的纯底档用于对照。
        cache.topic_active 为真时，两者 fp 不同是预期现象，不能报成
        「节流中有新内容待换入」。
        query：保留旧的按原话检索，仅用于对比 / 模拟 memory_search 按需查。

        这里刻意**不调用** _frozen_block：那会递增 rounds、刷新 built_at，
        等于「看一眼预览就把缓存节流状态改了」。只读 self._inject_cache。
        """
        from quart import request as quart_request
        payload = await quart_request.get_json(silent=True) or {}
        query = str(payload.get("query") or "")
        subject = str(payload.get("subject") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        bot_hint = str(payload.get("bot") or "").strip()[:32]
        budget = int(payload.get("budget") or self._cfg("inject_budget_tokens", 1200) or 1200)
        mode = str(payload.get("mode") or "live").strip().lower()
        # 预览时前端不传会话，sid 为空 → 记忆包里所有转述都会被判成「发到别的
        # 会话」，再被 labeled_only 整段滤掉，预览里永远看不到转述（实测现象）。
        # 所以按 subject 反推一个「他最近说话的会话」当默认，让预览和真实注入一致。
        #
        # subject 为空时**绝不允许悄悄退化成「全库」**：query_active_l2(subject="")
        # 走的是不分人的分支，把全库所有人的 L2 按置信度混在一起（实测某个人的
        # 包里混进两条别人的隐私事实），而面板/配置引导都会把这块标成「真正注入
        # 的字节」。规则集中写在 resolve_preview_scope 里，连同单元测试一起。
        scope_all = str(payload.get("scope") or "").strip().lower() == "all"
        recent = []
        if not subject and not scope_all:
            try:
                recent = self.store.query_recent_l3(
                    limit=1, event_type="user_message", bot_id=bot_hint or None)
            except Exception:
                recent = []
        subject, session_id, subject_auto = resolve_preview_scope(
            subject, session_id, list((self._inject_cache or {}).items()),
            (recent[0] if recent else None), scope_all)
        auto_sid = ""
        resolved = subject
        if not session_id and subject:
            try:
                # 跟 inject_memory 一样先做人物归一：同一 QQ 换昵称（用户乙→用户甲）
                # 后，L3 记在旧名下，直接按新昵称查一条都查不到。
                try:
                    resolved = self.store.person_resolve(subject) or subject
                except Exception:
                    resolved = subject
                rows = self.store.query_recent_l3(
                    limit=1, event_type="user_message", subject=resolved,
                    bot_id=bot_hint or None)
                if rows:
                    auto_sid = str(rows[0].get("session_id") or "")
            except Exception:
                auto_sid = ""
        eff_sid = session_id or auto_sid
        # 预览必须和真实注入看到同一份数据：bot 不传就按会话第一段推。
        bot = self._iso(bot_hint or self._bot_of(eff_sid))
        relations_on = bool(self._cfg("inject_relations", False))
        if relations_on:
            rel_display = self._relation_block(subject) if subject else ""
        else:
            # 总开关关着时明说，别让面板「注入预览」里那段空白让人以为坏了。
            # 但这句「关闭说明」只是给面板看的，真实注入轨里关系段是空串，
            # 不能拼进候选指纹，否则每次预览都误报「节流中有新内容」。
            rel_display = "（当前此功能为关闭：人际关系网总开关没开，关系不进记忆包、不注入对话）"
        rel_real = rel_display if relations_on else ""

        def _int_cfg(k, d):
            try:
                return max(0, int(self._cfg(k, d)))
            except (TypeError, ValueError):
                return int(d)

        cur_mode = self._inject_mode()
        freeze = {"dedup": bool(self._cfg("inject_dedup", True)),
                  "budget": budget, "mode": cur_mode,
                  "cfg_fp": self._cfg_fp(),
                  "topic_refresh": bool(self._cfg("inject_topic_refresh", True)),
                  "topic_extra": _int_cfg("inject_topic_extra", 3),
                  "inject_enabled": bool(self._cfg("inject_enabled", True))}
        common = {"subject": subject, "session_id": eff_sid,
                  # subject_auto：预览没指定人时自动锁定的（不是用户在面板里选的）。
                  # scope=all 才是真正的全库混合，界面必须自己标「会串台」。
                  "subject_auto": subject_auto,
                  "scope": "all" if scope_all else "one",
                  "relation_block": rel_display,
                  "freeze": freeze, "relation_detail": self._relation_detail(subject)}

        # 旧动态包：只作对比，不代表现在真正注入的内容。
        if mode == "query":
            builder = self._pack_builder(subject=subject, session_id=eff_sid,
                                         bot_id=bot)
            pack = builder.build(query=query, budget_tokens=budget)
            block = build_block(pack)
            mem_tokens = int(pack.get("token_estimate") or 0)
            rel_tokens = estimate_tokens(rel_real) if rel_real else 0
            return self._ok(dict(common, mode="query", block=block, pack=pack,
                                 memory_tokens=mem_tokens, relation_tokens=rel_tokens,
                                 token_estimate=mem_tokens + rel_tokens, cache=None,
                                 candidate_block="", note=(
                                     "按本轮原话实时检索的旧动态包，只用于对比 / 模拟 "
                                     "memory_search 按需查，不会进入 system_prompt。")))

        # ── 真实注入模式里直接叠「本轮尾部增量」─────────────────────────
        # 不再单开一个「冻结尾部模拟」分类：面板里填了句子，就等于在**同一个
        # 冻结包**上模拟这一轮 —— 固定前缀一字不动，尾部追加本轮召回到的 L2。
        # preview=True 只读：不写运行缓存、不推进轮数、不动节流状态。
        if query and mode != "query" and not scope_all:
            who_subj = resolved or subject
            if not who_subj:
                # 冻结模拟必须锁到具体的人：subject="" 会走 query_active_l2 的
                # 不分人分支，把全库记忆混在一起，面板会把它当成「你的注入」。
                return self._err("请先在「预览谁」里选一个人；或在聊天里说句话，"
                                 "让系统自动锁定身份后再试。冻结模拟不会在全库混合下进行。")
            sb_stat = self._pack_builder(subject=who_subj, session_id=eff_sid,
                                        lite=True, bot_id=bot)
            stat_pack = sb_stat.build(query=query, budget_tokens=budget,
                                     recent_queries=[])
            self._last_preview_pack = None
            block, _pt, _rt, _meta = self._frozen_block(
                who=who_subj, sess=eff_sid, budget=budget, rel_text=rel_real,
                bot_id=bot,
                topic_ids=stat_pack.get("topic_ids") or [],
                topic_signal=bool(stat_pack.get("topic_signal")),
                topic_query=query, recent_queries=[], preview=True)
            # 结构化 L2：面板要逐条区分「常驻前缀」与「尾部追加」，并显示
            # 这一轮到底补了哪几条。tail=True 的即本轮话题增量。
            pg = getattr(self, "_last_preview_pack", None) or {}
            l2_rows = []
            for r in (pg.get("l2") or []):
                val = str(r.get("value") or "")
                l2_rows.append({
                    "id": int(r.get("id") or 0),
                    "subject": str(r.get("subject") or ""),
                    "predicate": str(r.get("predicate") or ""),
                    "value": val,
                    "confidence": float(r.get("confidence") or 0),
                    "tail": bool(r.get("_topic_add")),
                    "tokens": estimate_tokens(val),
                })
            tail_rows = [x for x in l2_rows if x["tail"]]
            tail_ids = [x["id"] for x in tail_rows]
            # 固定前缀的真实条数＝整份常驻底档（0.16.8 起不再让出 tail_k 个槽位：
            # 拼装是纯追加，底档一条都不删，让出来的名额只会白丢前缀缓存）。
            base_count = len(l2_rows) - len(tail_rows)
            prefix_count = base_count
            # 纯底档（不含本轮尾部增量）一起带出去：面板的「读取明细」和
            # 「纯底档对照」都要它，而且它正是上面那块**前缀的来源**。
            try:
                base_pb = self._pack_builder(
                    subject=who_subj, session_id=eff_sid, extra_subjects=None,
                    l2_keep_min_override=self._resident_l2_n(),
                    bot_id=bot)
                base_pack = base_pb.build(query="", budget_tokens=budget)
                base_block = build_block(base_pack)
            except Exception:
                base_pack, base_block = {}, ""
            return self._ok(dict(
                common, mode="live", simulated=True, block=block,
                pack=base_pack, cache=None,
                l2=l2_rows, tail_rows=tail_rows, tail_ids=tail_ids,
                prefix_count=prefix_count, base_count=base_count,
                tail_count=len(tail_rows),
                # 面板要显示「当前注入 N 条 / 上限 M 条」和「块 tok / 预算」，
                # 让「追加会不会滚雪球」一眼可见：M = 常驻条数 + tail_k 恒定，
                # budget 是 trim_pack 的硬顶，两者都不随轮次增长。
                l2_limit=self._resident_l2_n(),
                tail_k=max(0, _int_cfg("inject_topic_extra", 3)),
                budget=int(budget),
                block_tokens=estimate_tokens(block),
                memory_tokens=estimate_tokens(block), relation_tokens=0,
                token_estimate=estimate_tokens(block),
                candidate_block=base_block,
                candidate_tokens=estimate_tokens(base_block),
                trim_notes=list((base_pack or {}).get("trim_notes") or []),
                note=(f"这是用你输入的句子「{query}」在**同一个冻结包**上模拟的这一轮："
                      f"固定前缀一字不动，尾部追加本轮召回到的 L2。仅预览，"
                      f"不写运行缓存、不推进节流轮数。想看进程里此刻真正挂着的"
                      f"那一块，把输入框清空再生成。")))

        # ── live：即时候选 = 不节流时该分片会生成的冻结包 ──
        # 用常驻条数（可能被 inject_l2_resident_limit 收缩过），不是 pack_l2_limit：
        # 面板显示的「上限」必须和真正会挂出去的那一块一致。
        l2_limit = self._resident_l2_n()
        sb = self._pack_builder(subject=subject, session_id=eff_sid,
                                extra_subjects=None,
                                l2_keep_min_override=self._resident_l2_n(),
                                bot_id=bot)
        pack = sb.build(query="", budget_tokens=budget)
        mem = build_block(pack)
        candidate = "\n\n".join(x for x in (mem, rel_real) if x)
        fp = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]

        # 运行时缓存的 key 里 who 是身份键（认不出人时会是 QQ 号），未必等于
        # 面板下拉里的 subject（昵称）。按 sess 匹配 + 身份键多重候选来找。
        # key 有两种形态：`bot|who|sess`（开了隔离）和老的 `who|sess`。
        ent, ent_key, ent_who = None, "", ""
        for k, e in (self._inject_cache or {}).items():
            parts = str(k).split("|")
            if len(parts) >= 3:
                kb, w, s = parts[0], parts[1], "|".join(parts[2:])
            else:
                kb, w, s = "", parts[0], "|".join(parts[1:])
            if s != eff_sid:
                continue
            if bot and kb and kb != bot:
                continue
            if not w or not (w == subject or w == resolved or w in eff_sid):
                continue
            if ent is None or float(e.get("built_at", 0)) > float(ent.get("built_at", 0)):
                ent, ent_key, ent_who = e, k, w
        live_block = (ent or {}).get("block") or candidate
        now = time.time()
        runtime_who = ent_who
        mismatch = bool(ent and runtime_who and runtime_who != subject)
        # pending 必须和运行时用**同一套口径**比：身份键不同时，拿面板 subject
        # 构建出来的候选去比缓存指纹会永远不相等，面板就会一直谎报「节流中」。
        runtime_fp = fp
        if mismatch:
            try:
                rb = self._pack_builder(subject=runtime_who, session_id=eff_sid,
                                        extra_subjects=None,
                                        l2_keep_min_override=self._resident_l2_n(),
                                        bot_id=bot)
                rp = rb.build(query="", budget_tokens=budget)
                rc = "\n\n".join(x for x in (build_block(rp), rel_real) if x)
                runtime_fp = hashlib.sha256(rc.encode("utf-8")).hexdigest()[:16]
            except Exception:
                runtime_fp = fp
        if ent:
            age = now - float(ent.get("built_at", 0))
            rounds = int(ent.get("rounds", 0))
            cache = {
                "exists": True, "key": ent_key, "runtime_who": runtime_who,
                "identity_mismatch": mismatch,
                # 只报现实：这块从什么时候起挂着、连续复用了多少轮。
                # 0.16.15 起没有「冻结秒数/轮数」这种倒计时——重建只由内容
                # 决定，所以不再返回 seconds_left / rounds_left 那类假承诺。
                "age_seconds": int(age), "rounds": rounds,
                # 话题增量激活时，runtime_fp 是「纯常驻底档」的指纹，跟真正
                # 挂着的（底档 + 本轮话题补进来的几条）必然不等。这不是
                # 「有新内容待换入」，别让面板一直谎报 pending_change。
                "topic_active": bool(ent.get("topic_sig")),
                "topic_ids": list(ent.get("topic_ids") or []),
                # 库里有新东西、下一轮就会重建（fp 一变即重建，没有等待期）。
                "pending_change": bool(ent.get("fp") != runtime_fp)
                                  and not bool(ent.get("topic_sig")),
                "block_tokens": estimate_tokens(str(ent.get("block") or "")),
            }
        else:
            cache = {"exists": False,
                     "pending_change": False, "identity_mismatch": False,
                     "hint": ("当前是 dynamic 模式，本来就没有冻结缓存；"
                              "下面显示的是按当前配置即时生成的块。"
                              if cur_mode == "dynamic" else
                              "进程刚启动或配置刚变更，这个分片还没生成过常驻包；"
                              "发一句话即可生成，不是故障。")}
        note = ("上面「实际注入」是 AI 这一轮真正看到的字节：按 (人,会话) 冻结的"
                "常驻底档，聊到包外相关记忆时会额外补进几条话题事实（B 方案）；"
                "「即时候选」只是不含话题增量的纯底档，用来对照，不代表待换入的新内容。"
                "本轮原话还会跑一遍实时检索做提及统计 / memory_search 按需查，"
                "那条统计轨和这里显示的注入块是两回事。"
                "预览里改 budget 只影响「即时候选」，不改变当前真实注入的那一块。")
        if scope_all:
            note = ("【全库模式】这一份是所有人的 L2/L3 混在一起按置信度排出来的，"
                    "真实注入永远不会这样——只用于排查「为什么某条没进包」。"
                    "要看待人的包，请在人物下拉里选一个。" + note)
        elif subject_auto and subject:
            note = (f"预览没指定人，已自动锁定到「{subject}」（运行时最近生成常驻包的那个"
                    f"身份键），这样看到的才是真正挂在 system_prompt 上的那块。" + note)
        if cur_mode == "dynamic":
            note = ("【inject_mode=dynamic】当前走旧的按原话实时检索路径，每轮都变、"
                    "不冻结、前缀缓存基本不命中，仅用于 A/B 对照或回退。" + note)
        return self._ok(dict(
            common, mode="live", block=live_block, candidate_block=candidate,
            pack=pack, cache=cache,
            # 纯底档：没有尾部追加，给面板标「上限/预算」用同一套字段。
            l2=[{"id": int(r.get("id") or 0),
                 "subject": str(r.get("subject") or ""),
                 "predicate": str(r.get("predicate") or ""),
                 "value": str(r.get("value") or ""),
                 "confidence": float(r.get("confidence") or 0),
                 "tail": False,
                 "tokens": estimate_tokens(str(r.get("value") or ""))}
                for r in (pack.get("l2") or [])],
            tail_rows=[], tail_ids=[], tail_count=0,
            prefix_count=len(pack.get("l2") or []),
            base_count=len(pack.get("l2") or []),
            l2_limit=l2_limit,
            tail_k=max(0, _int_cfg("inject_topic_extra", 3)),
            budget=int(budget),
            memory_tokens=estimate_tokens(mem),
            relation_tokens=estimate_tokens(rel_real) if rel_real else 0,
            token_estimate=estimate_tokens(live_block),
            candidate_tokens=estimate_tokens(candidate),
            trim_notes=list(pack.get("trim_notes") or []), note=note))

    async def _api_summarize(self):
        if self._summarizing:
            return self._err("总结正在进行中")
        asyncio.create_task(self._manual_run_summary())
        return self._ok({"started": True})

    async def _manual_run_summary(self):
        try:
            async with self._summary_lock:
                await self._run_summary(manual=True)
        except Exception:
            logger.exception("[SugarMemory] 手动总结异常")

    # ---------- 总结进度 API（面板显示「还差多少条触发」） ----------

    def _summary_progress(self) -> dict:
        """每个会话距离触发阶段总结还差多少条。计数与触发逻辑同源（从库里数）。"""
        interval = max(1, int(self._cfg("summary_interval_messages", 50) or 50))
        enabled = bool(self._cfg("summarize_enabled", True))
        sessions: set[str] = set()
        try:
            sessions.update(self.store.active_session_ids())
        except Exception:
            pass
        sessions.update(k for k in self._msg_count if k)
        sessions.update(self._pending)
        items = []
        for sess in sessions:
            try:
                since = self.store.last_summary_at(sess)
                n = self.store.count_user_messages_since(sess, since)
            except Exception:
                n = int(self._msg_count.get(sess, 0) or 0)
            pending = sess in self._pending
            # 没有新增消息也没在等待总结的老会话不值得占一行
            if n <= 0 and not pending:
                continue
            items.append({
                "session": sess,
                "count": min(n, interval),
                "remaining": max(0, interval - n),
                "pending": pending,
            })
        items.sort(key=lambda x: (-x["count"], x["session"]))
        return {
            "enabled": enabled,
            "interval": interval,
            "pending_summaries": len(self._pending),
            "summarizing": self._summarizing,
            "items": items[:50],
        }

    async def _api_summary_progress(self):
        return self._ok(self._summary_progress())

    async def _api_summary_prompt(self):
        """总结提示词预览：内置原文（只读）+ 本机追加段 + 实际发送的全文。

        面板据此渲染两栏：左边只读、右边可编辑追加。这里绝不返回「一个可以
        原地改的完整 prompt」——内置规则一旦可改，升级时就没法保证语义一致，
        用户也很难分清哪句是自己写的。
        """
        base = SUMMARY_SYSTEM_PROMPT
        extra = self._summary_prompt_extra()
        effective = self._summary_system_prompt()
        return self._ok({
            "base": base,
            "extra": extra,
            "header": SUMMARY_EXTRA_HEADER if extra else "",
            "effective": effective,
            "sections": summary_prompt_sections(base),
            "chars": {"base": len(base), "extra": len(extra),
                      "effective": len(effective)},
            "tokens": {"base": estimate_tokens(base),
                       "extra": estimate_tokens(extra) if extra else 0,
                       "effective": estimate_tokens(effective)},
            "hash": hashlib.sha1(effective.encode("utf-8")).hexdigest()[:12],
        })

    # ---------- 好感度 API ----------

    def _affinity_payload(self) -> dict:
        stages = self.store.affinity_stages()
        try:
            cap = max(1, int(self._cfg("affinity_cap", 1000) or 1000))
        except (TypeError, ValueError):
            cap = 1000
        rows = {r["subject"]: r for r in self.store.affinity_all()}
        mem = {s["subject"]: s for s in self.store.subject_stats()}
        gkeys = self.store.group_keys()
        names = {n for n in (set(rows) | set(mem)) if not self._is_group_name(n, gkeys)}
        users = []
        for name in names:
            xp = float((rows.get(name) or {}).get("xp") or 0)
            st = stage_of(xp, stages)
            info = mem.get(name) or {}
            users.append({
                "subject": name,
                "xp": round(xp, 1),
                "stage": st["name"],
                "next": st["next"],
                "next_threshold": st["next_threshold"],
                "progress": round(st["progress"], 4),
                "last_at": info.get("last_at", 0),
                "l2": info.get("l2", 0),
                "l3": info.get("l3", 0),
                "has_memory": bool(info),
            })
        users.sort(key=lambda u: (-u["xp"], u["subject"]))
        counts: dict[str, int] = {}
        for u in users:
            counts[u["stage"]] = counts.get(u["stage"], 0) + 1
        dist = [{"name": s["name"], "threshold": s["threshold"],
                 "count": counts.get(s["name"], 0)} for s in stages]
        total = sum(u["xp"] for u in users)
        return {
            "enabled": bool(self._cfg("affinity_enabled", True)),
            "cap": cap,
            "stages": dist,
            "users": users,
            "total": round(total, 1),
            "avg": round(total / len(users), 1) if users else 0,
            "relation_count": len(self.store.relations_all()),
            "ranks": list(RELATION_RANKS),
        }

    async def _api_affinity(self):
        return self._ok(self._affinity_payload())

    async def _api_affinity_set(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        subject = str(data.get("subject") or "").strip()[:40]
        if not subject:
            return self._err("缺少用户名")
        try:
            xp = float(data.get("xp"))
        except (TypeError, ValueError):
            return self._err("数值不合法")
        row = self.store.affinity_set(subject, max(0.0, xp))
        return self._ok({"row": row, "affinity": self._affinity_payload()})

    async def _api_affinity_add(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        subject = str(data.get("subject") or "").strip()[:40]
        try:
            delta = float(data.get("delta") or 0)
        except (TypeError, ValueError):
            return self._err("数值不合法")
        if not subject:
            return self._err("缺少用户名")
        cur = float((self.store.affinity_get(subject) or {}).get("xp") or 0)
        row = self.store.affinity_set(subject, max(0.0, cur + delta))
        return self._ok({"row": row, "affinity": self._affinity_payload()})

    async def _api_affinity_stages(self):
        return self._ok({"stages": self.store.affinity_stages(),
                         "ranks": list(RELATION_RANKS)})

    async def _api_affinity_stages_save(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        stages = data.get("stages")
        if not isinstance(stages, list) or not stages:
            return self._err("至少要留一档")
        saved = self.store.affinity_stages_save(stages)
        return self._ok({"stages": saved, "affinity": self._affinity_payload()})

    async def _api_affinity_stages_reset(self):
        return self._ok({"stages": self.store.affinity_stages_reset(),
                         "affinity": self._affinity_payload()})

    # ---------- 人物身份 API（AI 与人） ----------

    def _people_payload(self) -> dict:
        """面板「AI 与人」要的全部数据：一个人 = QQ + 一串称呼 + 好感 + 关系。"""
        stages = self.store.affinity_stages()
        try:
            cap = max(1, int(self._cfg("affinity_cap", 1000) or 1000))
        except (TypeError, ValueError):
            cap = 1000
        aff = {r["subject"]: r for r in self.store.affinity_all()}
        mem = {s["subject"]: s for s in self.store.subject_stats()}
        edges = self.store.relations_all()
        out_map: dict[str, list] = {}
        in_map: dict[str, list] = {}
        for e in edges:
            out_map.setdefault(e["source"], []).append(
                {"target": e["target"], "relation": e["relation"]})
            in_map.setdefault(e["target"], []).append(
                {"source": e["source"], "relation": e["relation"]})
        people = []
        known = set()
        for p in self.store.person_all():
            name = p["person"]
            known.add(name)
            xp = float((aff.get(name) or {}).get("xp") or 0)
            st = stage_of(xp, stages)
            info = mem.get(name) or {}
            people.append({
                "person": name,
                "qq": p.get("qq") or "",
                # 有没有 QQ 是「确认过的真人」和「只是被提到过」唯一可靠的分界。
                # 有 QQ：这个人真的在这个号上说过话；没有：只从关系边/总结里
                # 出现过，可能是真人（比如群友的室友），也可能是模型写岔的名字。
                "bound": bool(p.get("qq")),
                # 显示名 / 能被称呼的名是顶层两个，其余是绰号外号、以及叫错的
                # call_name 清空后必须返回空串，不能回退成 person 键——回退的话
                # 面板「AI 平时叫 TA『用户甲』」永远都在，删除看起来像没生效。
                "display_name": p.get("display_name") or name,
                "call_name": p.get("call_name") or "",
                "aliases": [{"key": a.get("alias_key"), "raw": a.get("raw"),
                             "source": a.get("source") or "",
                             "tier": int(a.get("tier") or 2),
                             "last_seen": float(a.get("last_seen") or 0),
                             "hits": int(a.get("hits") or 0)}
                            for a in (p.get("aliases") or [])],
                "xp": round(xp, 1),
                "stage": st["name"],
                "next": st["next"],
                "next_threshold": st["next_threshold"],
                "progress": round(st["progress"], 4),
                "l2": info.get("l2", 0),
                "l3": info.get("l3", 0),
                "last_at": info.get("last_at", 0),
                "out": out_map.get(name, []),
                "in": in_map.get(name, []),
            })
        # 只出现在好感度/关系里、还没登记身份的名字（理论上 bootstrap 后不会有，
        # 但数据库是别人也能改的，漏了就在面板上补一次）
        for name in set(aff) | set(out_map) | set(in_map):
            if name in known:
                continue
            xp = float((aff.get(name) or {}).get("xp") or 0)
            st = stage_of(xp, stages)
            people.append({
                "person": name, "qq": "", "aliases": [], "bound": False,
                "xp": round(xp, 1), "stage": st["name"], "next": st["next"],
                "next_threshold": st["next_threshold"],
                "progress": round(st["progress"], 4), "l2": 0, "l3": 0,
                "last_at": 0, "out": out_map.get(name, []), "in": in_map.get(name, []),
                "unregistered": True,
            })
        # 确认过的真人排前面，仅提及的靠后：不会因为一个路人关系多就挤掉正主
        people.sort(key=lambda u: (not u.get("bound"), -u["xp"],
                                   -(len(u["out"]) + len(u["in"])), u["person"]))
        return {
            "people": people,
            "suggestions": self.store.person_suggestions(),
            "stages": [{"name": s["name"], "threshold": s["threshold"]} for s in stages],
            "cap": cap,
            "total": round(sum(u["xp"] for u in people), 1),
            "avg": round(sum(u["xp"] for u in people) / len(people), 1) if people else 0,
            "with_qq": sum(1 for u in people if u["qq"]),
            "relation_count": len(edges),
        }

    async def _api_people(self):
        return self._ok(self._people_payload())

    async def _api_people_qq(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        person = str(data.get("person") or "").strip()[:40]
        if not person:
            return self._err("缺少人名")
        qq = re.sub(r"\D", "", str(data.get("qq") or ""))[:20]
        if qq and not (5 <= len(qq) <= 12):
            return self._err("QQ 号看着不对（应该是 5~12 位数字）")
        # 换号时顺手把撞号的人记下来：面板要提示「这两个号是一个人」
        res = self.store.person_set_qq(person, qq)
        return self._ok({"result": res, "people": self._people_payload()})

    async def _api_people_call(self):
        """指定「能被称呼的名」。一个人只有一个，设新的会把旧的降成外号。"""
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        person = str(data.get("person") or "").strip()[:40]
        name = str(data.get("name") or "").strip()[:40]
        if not person or not name:
            return self._err("人名和称呼都要填")
        res = self.store.person_set_call_name(person, name)
        if not res.get("ok"):
            if res.get("reason") == "taken":
                return self._err(f"「{name}」已经是「{res.get('owner')}」的称呼了，先合并再改")
            return self._err("没改成功")
        return self._ok({"result": res, "people": self._people_payload()})

    async def _api_people_call_clear(self):
        """删掉「能被称呼的名」：AI 的称呼退回显示名，原写法降回外号档。"""
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        person = str(data.get("person") or "").strip()[:40]
        if not person:
            return self._err("缺少人名")
        res = self.store.person_call_name_clear(person)
        if not res.get("ok"):
            return self._err("没删成功（人可能已不在库里）")
        return self._ok({"result": res, "people": self._people_payload()})

    async def _api_people_alias_cleanup(self):
        """手动跑一次称呼清理。dry_run=true 只算不动，面板用它做预览。"""
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        dry = bool(data.get("dry_run"))
        try:
            gdays = max(1.0, float(self._cfg("alias_guess_retention_days", 90) or 90))
            ndays = max(1.0, float(self._cfg("alias_nick_retention_days", 365) or 365))
        except (TypeError, ValueError):
            gdays, ndays = 90.0, 365.0
        res = self.store.alias_cleanup(guess_days=gdays, nick_days=ndays, dry_run=dry)
        return self._ok({"result": res, "people": self._people_payload()})

    async def _api_people_alias(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        person = str(data.get("person") or "").strip()[:40]
        raw = str(data.get("alias") or "").strip()[:40]
        if not person or not raw:
            return self._err("人名和称呼都要填")
        if raw == person:
            return self._err("这已经是本名了")
        res = self.store.person_bind(
            raw, person, source="manual", force=bool(data.get("force")))
        if not res.get("ok"):
            if res.get("reason") == "taken":
                return self._ok({"need_confirm": True, "owner": res.get("owner"),
                                 "people": self._people_payload()})
            return self._err("加不上，参数不对")
        return self._ok({"result": res, "people": self._people_payload()})

    async def _api_people_alias_remove(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        key = str(data.get("key") or "").strip()
        person = str(data.get("person") or "").strip()[:40]
        if not key:
            return self._err("缺少称呼")
        ok = self.store.person_alias_remove(key, person)
        return self._ok({"removed": ok, "people": self._people_payload()})

    async def _api_people_merge(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        keep = str(data.get("keep") or "").strip()[:40]
        drop = str(data.get("drop") or "").strip()[:40]
        if not keep or not drop or keep == drop:
            return self._err("要合并的两个名字不对")
        # force：这是人亲手点的合并，他知道这俩是同一个人。自动流程没有这个权限。
        res = self.store.person_merge(keep=keep, drop=drop, force=True)
        if not res.get("ok"):
            if res.get("reason") == "qq_conflict":
                return self._err(
                    f"这两个人绑的 QQ 不一样（{res.get('keep_qq')} / {res.get('drop_qq')}），"
                    "不像是同一个人。确认的话先解绑其中一个的 QQ 再合。")
            return self._err("合并失败")
        logger.info(f"[SugarMemory] 手动合并人物：{drop} → {keep}")
        # 内存里还在拿 drop 当归属人的会话，立刻改指 keep；否则下一条回复会把
        # 刚并掉的名字又写成一条独立记录，等于白合。
        moved = self._retarget_sender_cache(drop, keep)
        if moved:
            logger.info(f"[SugarMemory] {moved} 个会话的归属人改指 {keep}")
        return self._ok({"result": res, "people": self._people_payload()})

    def _retarget_sender_cache(self, old: str, new: str = "") -> int:
        """删人 / 合人之后修一遍内存里的 session→person 映射。

        这是在修一个「删了又长回来」的隐蔽路径：record_bot_reply 会拿
        _last_sender 里的归属人当 subject 写 L3。人删了、映射还在，AI 下次
        在那条会话里回一句，库里立刻又出现一个同名 subject——人物页没有档案
        （people 已删），记忆列表里却多出一格空的。只有重载插件才会消失，
        看起来就像「删不掉」。
        """
        n = 0
        for sess, hit in list(self._last_sender.items()):
            try:
                if len(hit) < 2 or hit[1] != old:
                    continue
                if new:
                    self._last_sender[sess] = (hit[0], new[:40])
                else:
                    self._last_sender.pop(sess, None)
                n += 1
            except Exception:
                continue
        return n

    async def _api_people_delete(self):
        """删除一个人和名下全部记忆。物理删除，面板上两次点击确认后才到这。"""
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        person = str(data.get("person") or "").strip()[:40]
        if not person:
            return self._err("缺少人名")
        res = self.store.person_delete(person)
        if not res.get("ok"):
            if res.get("reason") == "not_found":
                return self._err("库里没有这个人，刷新一下再看")
            return self._err("删除失败")
        # 同一毫秒内会话还拿着这个名字：不摘掉，下一条 bot_reply 会把它写回去
        self._retarget_sender_cache(person, "")
        left = self.store.person_residue(person)
        logger.info(
            f"[SugarMemory] 手动删除人物：{person}"
            f"（L2 {res.get('l2', 0)} 条 / L3 {res.get('l3', 0)} 条 / "
            f"关系 {res.get('edges', 0)} 条 / 好感 {res.get('xp', 0)} / "
            f"用量归属 {res.get('tokens', 0)} 条，残留 {left.get('total', 0)}）")
        if left.get("total"):
            logger.warning(f"[SugarMemory] 删除后仍有残留 {person}: {left.get('detail')}")
        return self._ok({"result": res, "residue": left,
                         "people": self._people_payload()})

    async def _api_people_scan(self):
        """重新扫一遍：登记新名字、从原话里学称呼、把裂开的节点并回去。"""
        res = self.store.person_bootstrap()
        return self._ok({"result": res, "people": self._people_payload()})

    # ---------- 人际关系 API ----------

    async def _api_rel_graph(self):
        edges = self.store.relations_all()
        aff = {r["subject"]: float(r["xp"] or 0) for r in self.store.affinity_all()}
        known = {s["subject"] for s in self.store.subject_stats()}
        deg: dict[str, int] = {}
        for e in edges:
            deg[e["source"]] = deg.get(e["source"], 0) + 1
            deg[e["target"]] = deg.get(e["target"], 0) + 1
        # 有记忆的人即使还没有关系也画出来，方便直接给 Ta 补一条；
        # 有关系的人排在前面，环上相邻，连线不会穿过整个圆。
        # 群不进关系图：群号/群名一旦被当成 subject 存进来，会变成“一个人”，这里摘掉。
        gkeys = self.store.group_keys()
        cand = set(deg) | set(aff) | known
        # 空 subject 是历史残留（没归属的转述），画出来就是一个没名字的点。
        names = sorted((n for n in cand
                        if str(n).strip() and not self._is_group_name(n, gkeys)),
                       key=lambda n: (-deg.get(n, 0), n))
        hidden_groups = len(cand) - len(names)
        truncated = len(names) > 60
        # 节点名同样不能直接拿内部键：图上写着「张三」、人物页写着「Moon」，
        # 用户会以为是两个人。有 display_name 就用它（本人昵称优先）。
        pmap = {}
        try:
            pmap = {str(p.get("person") or ""): p for p in self.store.person_all()}
        except Exception:
            pass
        nodes = []
        for n in names[:60]:
            p = pmap.get(n) or {}
            nick = str(p.get("display_name") or "").strip() or n
            call = str(p.get("call_name") or "").strip()
            nodes.append({
                "subject": n, "degree": deg.get(n, 0), "xp": aff.get(n, 0),
                "shown": nick,
                "call": call if call and alias_key(call) != alias_key(nick) else "",
            })
        keep = {n["subject"] for n in nodes}
        shown = [e for e in edges if e["source"] in keep and e["target"] in keep]
        return self._ok({"nodes": nodes, "edges": shown, "ranks": list(RELATION_RANKS),
                         "truncated": truncated, "edge_total": len(edges),
                         "hidden_groups": hidden_groups})

    async def _api_rel_user(self):
        from quart import request as quart_request
        subject = (quart_request.args.get("subject") or "").strip()
        if not subject:
            return self._err("缺少用户名")
        edges = self.store.relations_all()
        out = []
        for e in edges:
            if e["source"] == subject:
                e = dict(e)
                e["evidence"] = self.store.relation_evidence(e["source"], e["target"], limit=30)
                out.append(e)
        inc = [dict(e) for e in edges if e["target"] == subject]
        aff_row = self.store.affinity_get(subject) or {}
        stages = self.store.affinity_stages()
        xp = float(aff_row.get("xp") or 0)
        return self._ok({
            "subject": subject, "out": out, "in": inc,
            "affinity": {"xp": round(xp, 1), "stage": stage_of(xp, stages)["name"]},
            "ranks": list(RELATION_RANKS),
        })

    async def _api_rel_set(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        src = str(data.get("source") or "").strip()[:40]
        dst = str(data.get("target") or "").strip()[:40]
        if not src or not dst:
            return self._err("两个人名都要填")
        if src == dst:
            return self._err("不能跟自己建关系")
        rank = normalize_relation(data.get("relation"))
        if not rank:
            return self._err("关系只能是：" + " / ".join(RELATION_RANKS))
        res = self.store.relation_upsert(
            source=src, target=dst, relation=rank,
            quote=str(data.get("quote") or "").strip()[:200],
            confidence=1.0, source_type="manual",
        )
        if not res.get("ok"):
            if res.get("reason") == "same_person":
                return self._err("这两个名字是同一个人，不能建关系")
            return self._err("没存上：两个人名和档位都要填")
        return self._ok({"result": res, "edges": self.store.relations_all()})

    async def _api_rel_delete(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        src = str(data.get("source") or "").strip()
        dst = str(data.get("target") or "").strip()
        if not src or not dst:
            return self._err("缺少 source/target")
        n = self.store.relation_delete_edge(src, dst, drop_evidence=True)
        return self._ok({"deleted": n, "edges": self.store.relations_all()})

    async def _api_rel_aliases(self):
        return self._ok({"groups": self.store.relation_alias_groups()})

    async def _api_rel_merge(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        keep = str(data.get("keep") or "").strip()[:40]
        drop = str(data.get("drop") or "").strip()[:40]
        if not keep or not drop or keep == drop:
            return self._err("要指定保留谁、并进谁")
        res = self.store.relation_merge(keep, drop)
        if not res.get("ok"):
            return self._err("合并失败：名字不对")
        return self._ok({"result": res,
                         "groups": self.store.relation_alias_groups(),
                         "edges": self.store.relations_all()})

    async def _api_rel_evidence(self):
        from quart import request as quart_request
        src = (quart_request.args.get("source") or "").strip()
        dst = (quart_request.args.get("target") or "").strip()
        if not src or not dst:
            return self._err("缺少 source/target")
        return self._ok({"items": self.store.relation_evidence(src, dst)})

    async def _api_rel_evidence_delete(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        try:
            ev_id = int(data.get("id") or 0)
        except (TypeError, ValueError):
            return self._err("无效的 id")
        if ev_id <= 0:
            return self._err("缺少 id")
        ok = self.store.relation_evidence_delete(ev_id)
        return self._ok({"deleted": bool(ok)}) if ok else self._err(f"未找到 #{ev_id}")

    async def _api_rel_evidence_update(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        try:
            ev_id = int(data.get("id") or 0)
        except (TypeError, ValueError):
            return self._err("无效的 id")
        quote = str(data.get("quote") or "").strip()[:200]
        if ev_id <= 0:
            return self._err("缺少 id")
        if not quote:
            return self._err("证据不能改成空的，想删就点 ×")
        ok = self.store.relation_evidence_update(ev_id, quote)
        return self._ok({"updated": bool(ok)}) if ok else self._err(f"未找到 #{ev_id}")

    # ---------- 总配置 ----------

    def _tool_active(self, name: str) -> bool:
        try:
            return self.context.provider_manager.llm_tools.get_func(name) is not None
        except Exception:
            return False

    async def _api_config(self):
        return self._ok({
            "items": self._cfg_items(),
            "path": self._cfg_path or "",
            "tools": {n: self._tool_active(n) for n in ("relation_set", "relation_query")},
        })

    async def _api_config_set(self):
        from quart import request as quart_request
        data = await quart_request.get_json(silent=True) or {}
        patch = data.get("patch") if isinstance(data.get("patch"), dict) else data
        fp_before = self._cfg_fp()
        ok, info = self._save_cfg(patch)
        if not ok:
            return self._err(info)
        # 只有真正影响 prompt 字节的配置变了才失效缓存：改 token 统计、
        # 总结间隔这类无关项不该把一天的命中率一波带走。
        if self._cfg_fp() != fp_before:
            self._invalidate_cache_for_config()
        self._apply_tool_flags()
        return self._ok({
            "saved": True, "mode": info,
            "items": self._cfg_items(), "path": self._cfg_path or "",
            "tools": {n: self._tool_active(n) for n in ("relation_set", "relation_query")},
        })

    # ---------- Token 用量 API ----------

    TOKEN_MODULE_LABELS = {
        "chat": "对话回复",
        "summary": "生成摘要",
        "pack": "记忆包注入",
        "relation": "关系快照",
        "memory_search": "AI 查记忆",
        "outbound_history": "AI 翻外发历史",
        "relation_query": "AI 查关系",
        "alias": "问法别名生成",
        "distill": "补录提炼",
    }

    async def _api_tokens(self):
        from quart import request as quart_request
        rng = (quart_request.args.get("range") or "all").strip().lower()
        now = time.time()
        span, step = {
            "24h": (86400, 3600),
            "7d": (7 * 86400, 86400),
            "30d": (30 * 86400, 86400),
        }.get(rng, (0, 86400))
        since = now - span if span else 0.0
        if not span:
            first = self.store.token_first_ts() or now
            if first < now - 90 * 86400:
                step = 7 * 86400          # 太久远的按周合并，别画 300 根柱子
        labels = dict(self.TOKEN_MODULE_LABELS)

        total_all = self.store.token_totals(0)
        h24 = self.store.token_totals(now - 86400)
        lt = time.localtime(now)
        today_start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        today = self.store.token_totals(today_start)

        by_mod_all = self.store.token_by_module(0)
        by_mod = self.store.token_by_module(since) if span else by_mod_all
        for row in by_mod + by_mod_all:
            row["label"] = labels.get(row["module"], row["module"])

        mods = sorted({r["module"] for r in by_mod_all})
        tz = -(time.altzone if time.localtime(now).tm_isdst else time.timezone)
        off = -tz
        raw = self.store.token_series(since, step, offset=off)
        grid: dict[float, dict] = {}

        def align(t: float) -> float:
            return float(int((t - off) / step) * step + off)

        b0 = align(since) if span else (align(raw[0]["bucket"]) if raw else align(now))
        b = b0
        while b <= now:
            grid[float(b)] = {"t": float(b), "total": 0, "by": {m: 0 for m in mods}}
            b += int(step)
        for r in raw:
            key = align(r["bucket"])
            cell = grid.get(key)
            if cell is None:
                cell = grid.setdefault(key, {"t": key, "total": 0, "by": {m: 0 for m in mods}})
            cell["by"][r["module"]] = cell["by"].get(r["module"], 0) + int(r["total"])
            cell["total"] += int(r["total"])
        series = [grid[k] for k in sorted(grid)][-120:]

        recent = []
        for r in self.store.token_recent(limit=30):
            recent.append({"ts": r["ts"], "module": r["module"],
                           "label": labels.get(r["module"], r["module"]),
                           "source": r["source"], "input": r["input"], "cached": r["cached"],
                           "output": r["output"], "total": r["total"],
                           "subject": r["subject"]})
        return self._ok({
            "range": rng, "step": step,
            "totals": {"all": total_all, "h24": h24, "today": today},
            "by_module": by_mod, "by_module_all": by_mod_all,
            "series": series, "recent": recent,
            "labels": labels, "modules": mods,
            "enabled": bool(self._cfg("token_stats_enabled", True)),
        })

    async def _api_tokens_hit(self):
        """缓存命中率明细。这里给的是**两个不同的数**，别混着看：

        1. 记忆包命中率（pack）：本轮注入块的 fp 与同会话上一轮一致的比例。
           这是插件自己能保证的部分，衡量冻结策略有没有生效；frozen 架构下
           它应该接近 100%。dynamic 旧版没有常驻包，此项标为不适用。
        2. 整轮请求命中率（request/summary）：provider 报的
           cached / (input + cached)，包含框架 system、历史、工具 schema。
           前缀被框架改写时它会掉，但记忆包可能一个字节没变。

        另外把首轮、重载冷启动、长间隔自然过期单独标记并从「有效命中率」
        里剔除 —— 这三类 0% 都是固定正常，混进失败样本只会误导调参。
        只统计 module='chat' AND source='actual'：provider 上报的真实用量。
        工具循环里一次用户消息会打多次 LLM，这里只记对话轮，与主库
        provider_stats 的粒度不是 1:1。
        """
        from quart import request as quart_request
        try:
            q = quart_request.args
            rng = (q.get("range") or "all").strip().lower()
            arch = (q.get("arch") or "all").strip().lower()
            provider = (q.get("provider") or "").strip()
            session = (q.get("session") or "").strip()
            limit = max(1, min(500, int(q.get("limit") or 100)))
            offset = max(0, int(q.get("offset") or 0))

            now = time.time()
            span = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}.get(rng, 0)
            since = now - span if span else 0.0
            all_rows = self.store.token_chat_rows(since, 20000)

            def _arch(r):
                return str(r.get("arch_ver") or "unknown").strip().lower() or "unknown"

            def _provider_raw(r):
                return str(r.get("provider_id") or "").strip()

            def _provider(r):
                # 老数据里可能有 `5.3/glm-5.3-flash/glm-5.3-flash` 这种重复
                # 拼段，读的时候统一收敛，分组卡和下拉才不会冒出假服务商。
                return normalize_provider_label(_provider_raw(r))

            plow, slow = provider.lower(), session.lower()
            rows = [r for r in all_rows
                    if (arch in ("", "all") or _arch(r) == arch)
                    and (not plow or plow in _provider(r).lower()
                         or plow in _provider_raw(r).lower())
                    and (not slow or slow in str(r.get("session_id") or "").lower()
                         or slow in str(r.get("subject") or "").lower())]

            def _sum(items):
                inp = sum(int(r.get("input") or 0) for r in items)
                cac = sum(int(r.get("cached") or 0) for r in items)
                out = sum(int(r.get("output") or 0) for r in items)
                prompt = inp + cac
                return {"calls": len(items), "input": inp, "cached": cac,
                        "output": out, "prompt": prompt,
                        "hit_rate": round(cac / prompt, 4) if prompt else 0.0,
                        "zero_cached": sum(1 for r in items if not int(r.get("cached") or 0)),
                        "reported": sum(1 for r in items if int(r.get("cached") or 0) > 0)}

            def _int_cfg(key, default):
                try:
                    return int(self._cfg(key, default))
                except (TypeError, ValueError):
                    return int(default)

            def _group(items, keyfn, label=None):
                buckets: dict[str, list] = {}
                for r in items:
                    buckets.setdefault(str(keyfn(r) or "—"), []).append(r)
                out = []
                for k, g in buckets.items():
                    d = _sum(g)
                    d["key"] = k
                    d["label"] = (label(k, g) if label else k)
                    out.append(d)
                return sorted(out, key=lambda x: (-x["calls"], x["key"]))

            frozen = [r for r in rows if _arch(r) == "frozen"]
            by_arch = _group(rows, _arch)
            by_provider = _group(rows, lambda r: _provider(r) or "未记录")
            by_session = _group(
                rows,
                lambda r: str(r.get("session_id") or "未知会话"),
                label=lambda k, g: k + (" · " + "、".join(sorted({
                    str(r.get("subject") or "") for r in g if r.get("subject")})[:2])
                    if any(r.get("subject") for r in g) else ""))
            by_state = _group(
                frozen,
                lambda r: "重建轮" if int(r.get("rebuilt") or 0) else "同 fp 连续轮")
            # 按「常驻前缀指纹」分组（老数据退回整块 fp）。用 inject_fp 分组会
            # 把正常的话题尾部刷新也算成一份新包：同一会话聊三个话题就是三份，
            # 看起来像「一直在重建」，其实前缀一个字节没动。
            by_fp = _group(
                [r for r in frozen if r.get("prefix_sig") or r.get("inject_fp")],
                lambda r: str(r.get("prefix_sig") or r.get("inject_fp") or "")[:12])
            by_reason = _group(
                [r for r in frozen if str(r.get("rebuild_reason") or "")],
                lambda r: str(r.get("rebuild_reason") or ""))

            # ── 逐轮归因 ────────────────────────────────────────────────
            # 按 (会话, 架构) 时间正序走一遍，拿到上一轮的 fp 和距上轮间隔。
            # 这三个标签的 0% 都是「本来就该 0%」，不该混进失败样本：
            #   first   新会话 / 首次记录，要先写一次缓存
            #   cold    插件重载把内存冻结块清空了，字节常常根本没变
            #   expired 隔太久，provider 侧前缀缓存已经自然过期
            expire_gap = float(CACHE_EXPIRE_GAP)
            prev_by_key: dict[tuple, dict] = {}
            meta_by_id: dict[int, dict] = {}
            flag_labels = {
                "": "正常轮", "first": "首轮（写缓存）",
                "cold": "重载冷启动", "expired": "自然过期",
            }
            for r in sorted(rows, key=lambda x: (
                    float(x.get("ts") or 0), int(x.get("id") or 0))):
                k = (str(r.get("session_id") or ""), _arch(r))
                prev = prev_by_key.get(k)
                gap = (float(r.get("ts") or 0) - float(prev.get("ts") or 0)) if prev else None
                prev_fp = str(prev.get("inject_fp") or "") if prev else ""
                cur_fp = str(r.get("inject_fp") or "")
                reason = str(r.get("rebuild_reason") or "").strip().lower()
                long_gap = gap is not None and gap >= expire_gap
                if prev is None:
                    flag = "first"
                elif reason in COLD_REASONS:
                    flag = "cold"
                elif long_gap:
                    flag = "expired"
                else:
                    flag = ""
                # 一轮可能既是重载冷启动、又隔了很久。主标记只留一个（冷启动
                # 的提示更重要），但 long_gap 单独记下来，别把自然过期的信息吞掉。
                pack_stable = None
                if cur_fp and prev_fp:
                    pack_stable = (cur_fp == prev_fp)
                elif cur_fp and reason == "reload":
                    # 上一轮没记 fp（老数据），但本轮已确认是「字节未变的重载」
                    pack_stable = True
                # 常驻底档指纹单独比：整块 fp 含「本轮话题尾部」，换个话题、
                # 多召一条就变。拿它判漂移，正常的话题刷新会被全算成故障。
                cur_prefix = str(r.get("prefix_sig") or "")
                prev_prefix = str(prev.get("prefix_sig") or "") if prev else ""
                prefix_stable = None
                if cur_prefix and prev_prefix:
                    prefix_stable = (cur_prefix == prev_prefix)
                meta_by_id[int(r.get("id") or 0)] = {
                    "flag": flag, "flag_label": flag_labels.get(flag, flag),
                    "gap_s": round(gap, 1) if gap is not None else None,
                    "long_gap": bool(long_gap),
                    "pack_stable": pack_stable, "prev_fp": prev_fp,
                    "prefix_stable": prefix_stable, "prefix_sig": cur_prefix,
                    "rebuild_reason": reason,
                }
                prev_by_key[k] = r

            # 有 prefix_sig 的轮次按它判；老数据（加列之前）只能退回整块 fp。
            def _stable_of(m):
                return m["prefix_stable"] if m["prefix_stable"] is not None else m["pack_stable"]

            comparable = [m for m in meta_by_id.values()
                          if _stable_of(m) is not None]
            stable = sum(1 for m in comparable if _stable_of(m))
            prefix_based = sum(1 for m in meta_by_id.values()
                               if m["prefix_stable"] is not None)
            pack = {
                "comparable": len(comparable),
                "stable": stable,
                "drift": len(comparable) - stable,
                "uncomparable": len(meta_by_id) - len(comparable),
                "rate": round(stable / len(comparable), 4) if comparable else 0.0,
                # basis=prefix 说明这批数据是按常驻底档判的（准）；pack 表示
                # 还是老数据/老轮次，只能按整块 fp 判，会被话题尾部带偏。
                "basis": "prefix" if prefix_based else "pack",
                "prefix_based": prefix_based,
                # dynamic 旧版每轮重新拼包，没有「常驻包」这个概念，别拿它算稳定性
                "applicable": arch != "dynamic",
            }
            effective_rows = [r for r in rows
                              if meta_by_id.get(int(r.get("id") or 0), {}).get("flag") == ""]
            effective = _sum(effective_rows)
            effective["excluded"] = len(rows) - len(effective_rows)
            flags = {k: sum(1 for m in meta_by_id.values() if m["flag"] == k)
                     for k in ("first", "cold", "expired", "")}
            flags["long_gap"] = sum(1 for m in meta_by_id.values() if m["long_gap"])
            by_flag = _group(
                [r for r in rows
                 if meta_by_id.get(int(r.get("id") or 0), {}).get("flag")],
                lambda r: meta_by_id[int(r.get("id") or 0)]["flag_label"])

            detail = []
            for r in rows:
                inp = int(r.get("input") or 0)
                cac = int(r.get("cached") or 0)
                prompt = inp + cac
                m = meta_by_id.get(int(r.get("id") or 0), {})
                detail.append({
                    "id": r.get("id"), "ts": float(r.get("ts") or 0),
                    "arch_ver": _arch(r), "provider_id": _provider(r),
                    "provider_id_raw": _provider_raw(r),
                    "subject": str(r.get("subject") or ""),
                    "session_id": str(r.get("session_id") or ""),
                    "input": inp, "cached": cac,
                    "output": int(r.get("output") or 0), "prompt": prompt,
                    "hit_rate": round(cac / prompt, 4) if prompt else 0.0,
                    "inject_fp": str(r.get("inject_fp") or ""),
                    "rebuilt": int(r.get("rebuilt") or 0),
                    "freeze_rounds": int(r.get("freeze_rounds") or 0),
                    "cfg_ver": str(r.get("cfg_ver") or ""),
                    "rebuild_reason": str(m.get("rebuild_reason") or ""),
                    "flag": str(m.get("flag") or ""),
                    "flag_label": str(m.get("flag_label") or "正常轮"),
                    "gap_s": m.get("gap_s"),
                    "long_gap": bool(m.get("long_gap")),
                    "pack_stable": m.get("pack_stable"),
                    "prefix_stable": m.get("prefix_stable"),
                    "prefix_sig": str(r.get("prefix_sig") or ""),
                })

            return self._ok({
                "range": rng, "arch": arch, "provider": provider, "session": session,
                "limit": limit, "offset": offset, "total_rows": len(rows),
                "rows": detail[offset:offset + limit],
                "summary": _sum(rows),          # = 整轮请求命中率（兼容旧前端）
                "request": _sum(rows),          # 同一份，语义更清楚
                "pack": pack,                  # 记忆包命中率：优先看这个
                "effective": effective,        # 剔除首轮/冷启动/自然过期后
                "flags": flags, "by_flag": by_flag, "by_reason": by_reason,
                "expire_gap_seconds": int(expire_gap),
                "by_arch": by_arch, "by_provider": by_provider,
                "by_session": by_session, "by_state": by_state, "by_fp": by_fp,
                "frozen_calls": len(frozen),
                "dynamic_calls": sum(1 for r in rows if _arch(r) == "dynamic"),
                "last_ts": max((float(r.get("ts") or 0) for r in rows), default=0.0),
                "providers": sorted({_provider(r) for r in all_rows if _provider(r)}),
                "arches": sorted({_arch(r) for r in all_rows}),
                "current_mode": str(self._cfg("inject_mode", "frozen") or "frozen"),
                "enabled": bool(self._cfg("token_stats_enabled", True)),
                # 疑难解答页要显示「你现在实际配的是多少」，别让用户自己翻配置面板猜。
                "settings": {
                    "inject_mode": str(self._cfg("inject_mode", "frozen") or "frozen"),
                    "inject_enabled": bool(self._cfg("inject_enabled", True)),
                    "inject_dedup": bool(self._cfg("inject_dedup", True)),
                    "inject_budget_tokens": _int_cfg("inject_budget_tokens", 1200),
                    "pack_l2_limit": _int_cfg("pack_l2_limit", 12),
                    "token_stats_enabled": bool(self._cfg("token_stats_enabled", True)),
                    "cache_expire_gap_seconds": int(expire_gap),
                },
                "notes": [
                    "【两个指标，优先看常驻前缀命中率】它 = 常驻底档（prefix 那几条 L2）的指纹与同会话上一轮一致的比例，只衡量插件自己挂进 system 的那块有没有变；frozen 架构下它接近 100% 才说明冻结生效。整块 fp 还含本轮话题尾部，换个话题就会变，拿它判漂移会把正常的话题刷新全算成故障（2026-11-08 起分开记 prefix_sig）。",
                    "整轮请求命中率 = 缓存输入 / (普通输入 + 缓存输入)，只统计 provider 上报真实用量的对话轮（chat / actual）；注入块、工具结果那些按文本估的行不算，否则会算低。",
                    "cached_tokens 是整轮请求的前缀缓存，包含框架 system_prompt、历史消息、工具 schema。所以整轮命中率低 ≠ 记忆包失效：包字节没变时，miss 出在框架历史裁剪 / 工具集变化 / provider 缓存 TTL 这些外层。",
                    f"【0% 有三种是固定正常】首轮要先写一次缓存；插件重载 / 重启会清空内存里的冻结块（日志会提示「重载冷启动」，字节往往一个没变）；距上轮间隔超过 {int(expire_gap)} 秒时 provider 缓存已自然过期。这三类都单独打标，不算失败样本，「有效命中率」也把它们剔除了。",
                    "cached = 0 还可能是这个 provider 没把命中数上报（部分框架只认 OpenAI 格式的 prompt_tokens_details.cached_tokens）。别把零命中一律当成缓存失效。",
                    "工具循环里一次用户消息会打多次 LLM，插件只记「对话轮」，与主库 provider_stats 的请求级记录不是 1:1，跨表对比前先对齐口径。",
                    "arch_ver=frozen / dynamic 才能做 A/B；旧数据在加字段时已统一锁成 dynamic，作为改造前的 baseline。「按重建原因」能看出 cold/reload、config、l2_changed、fp_changed 各占多少，比只看 rebuilt=1 准。",
                ],
            })
        except Exception as e:
            logger.warning("[SugarMemory] 缓存命中率统计失败：%s", e, exc_info=True)
            return self._err(f"缓存命中率统计失败：{e}")

    async def _api_tokens_clear(self):
        n = self.store.token_clear()
        return self._ok({"deleted": n})

    # ---------- 给 AI 的工具 ----------

    @filter.llm_tool(name="relation_set")
    async def tool_relation_set(self, event: AstrMessageEvent, source: str,
                                target: str, relation: str, quote: str = ""):
        """记录两个人之间的关系（谁认为谁是什么关系）。

        当对话里明确出现「A 是 B 的某某」「B 是我朋友」这类话时调用。
        只记真人之间的关系，不要把机器人自己算进去。
        source 和 target 只填用户本轮消息里出现过的名字，别从记忆或别人转述里挑。

        Args:
            source(string): 说出这个判断的人
            target(string): 被描述的那个人
            relation(string): 关系档位，必须是 认识、朋友、亲友、恋人 之一
            quote(string): 用户表达这层关系的原话
        """
        if not self._cfg("inject_relations", False):
            return "人际关系功能当前为关闭，我这边记不了关系。"
        if self._is_group(event) and not self._cfg("tool_write_in_group", False):
            return "群聊里没有写关系的权限，只能在私聊里改。"
        if self._cfg("tool_relation_write_admin_only", False) and not self._is_admin(event):
            return "只有主人能让我改关系，你没有这个权限。"
        src = (source or "").strip()[:40]
        dst = (target or "").strip()[:40]
        if not src or not dst:
            return "失败：source 和 target 都要填"
        if src == dst:
            return "失败：不能跟自己建关系"
        rank = normalize_relation(relation)
        if not rank:
            return "失败：关系只能选 " + " / ".join(RELATION_RANKS)
        # 身份门控：非管理员只能记跟自己有关的边，防止路人让 AI 改别人的关系
        if self._cfg("tool_relation_self_only", True) and not self._is_admin(event):
            me = self._sender_name(event)
            if not me:
                return "失败：认不出当前说话人，不能改关系。"
            if not (self._same_person(me, src) or self._same_person(me, dst)):
                return (f"失败：当前说话人是「{me}」，只能记跟他本人有关的关系，"
                        f"不能改 {src} 和 {dst} 之间的。"
                        f"如果要记的是他自己的判断，source 请填「{me}」。")
        # 防间接注入：只认说话人本轮亲手打的字。上下文里读到的内容（记忆包、
        # 别人转述、贴进来的网页）不代表他授权，必须两者都不在本轮原话里才拒。
        if self._cfg("tool_relation_quote_check", True):
            ok, _raw = self._quote_check(event, src, dst)
            if not ok:
                return (f"没有记录：本轮消息里没有提到「{src}」或「{dst}」，"
                        f"这层关系看起来不是说话人自己讲的，可能是从别处读到的内容。"
                        f"先跟他确认一句再记。")
        # 幂等锁：同一个改动在窗口内只落一次，防模型重试/tool loop 重复写证据
        try:
            window = float(self._cfg("tool_idem_seconds", 120) or 0)
            if window > 0 and self.store.idem_hit(
                    _idem_key("relation_set", src, dst, rank), window):
                return f"「{src} 认为 {dst} 是{rank}」刚刚已经记过了，这次没有重复写。"
        except Exception:
            logger.debug("[SugarMemory] 幂等检查失败，放行", exc_info=True)
        res = self.store.relation_upsert(
            source=src, target=dst, relation=rank,
            quote=(quote or "").strip()[:200], confidence=1.0, source_type="ai_tool",
        )
        if not res.get("ok"):
            if res.get("reason") == "same_person":
                return f"失败：{src} 和 {dst} 是同一个人，不能建关系"
            if res.get("reason") == "kinship":
                nm = res.get("name") or dst
                return (f"没有记：「{nm}」是关系称呼（弟弟/妈妈/老板这类），不是名字。"
                        f"要么换成这个人的真名再调一次，要么先去人物页给「{nm}」"
                        f"绑一个真名——直接拿称呼建关系会凭空多出一个人。")
            return "失败：参数不合法"
        # 归并后落库的名字可能和模型说的不一样（同一个人有几种写法），按落库的报回去
        real_src = res.get("source") or src
        real_dst = res.get("target") or dst
        if res.get("changed") and res.get("old"):
            return f"已更新：{real_src} 认为 {real_dst} 是{rank}（原来记的是{res['old']}）"
        return f"已记下：{real_src} 认为 {real_dst} 是{rank}"

    @filter.llm_tool(name="relation_query")
    async def tool_relation_query(self, event: AstrMessageEvent, subject: str = ""):
        """查询某个人和别人之间的关系。想知道「我跟谁是什么关系」时调用。

        Args:
            subject(string): 要查的人名；留空则查当前正在对话的人
        """
        if not self._cfg("inject_relations", False):
            return "人际关系功能当前为关闭，我这边查不了关系。"
        if self._cfg("tool_relation_query_admin_only", False) and not self._is_admin(event):
            return "只有主人能让我查关系，你没有这个权限。"
        who = (subject or "").strip()
        if not who:
            who = (self._last_sender.get(self._session_key(event)) or ("", ""))[1]
        if not who:
            return "没有指定人名，也判断不出当前对话的是谁"
        try:
            who = self.store.canonical_name(who) or who
        except Exception:
            pass
        edges = self.store.relations_all()
        out = [f"{e['target']}：{e['relation']}" for e in edges if e["source"] == who]
        inc = [f"{e['source']} 认为 {who} 是{e['relation']}"
               for e in edges if e["target"] == who]
        if not out and not inc:
            return f"没有关于 {who} 的关系记录"
        parts = []
        if out:
            parts.append(f"{who} 认为：" + "，".join(out))
        if inc:
            parts.append("别人认为：" + "，".join(inc))
        result = "；".join(parts)
        # 工具结果会作为 tool 消息进下一轮对话输入，照实估一笔（同 memory_search）
        self._record_tokens("relation_query", source="estimate",
                            input=estimate_tokens(result))
        return result

    @filter.llm_tool(name="alias_lookup")
    async def tool_alias_lookup(self, event: AstrMessageEvent, name: str):
        """查一个称呼/外号指的是谁，以及这个人在库里登记的全部称呼。

        对话里出现不确定的称呼时调用：用户提到一个你没见过的外号/昵称/
        群名片，想确认它是不是某个已认识的人；或者想确认某人都能被怎么叫。
        返回这个人的显示名、QQ号和所有登记过的称呼（按 显示名>常用名>外号
        分层，数量不多）。查不到时给出相近的候选称呼。

        Args:
            name(string): 要查的称呼，用用户消息里出现的原样写法
        """
        q = str(name or "").strip()
        if not q:
            return "称呼为空，请填用户提到的原样称呼。"
        hits = self.store.alias_owner(q)
        if not hits:
            return f"库里没有叫「{q}」的人，也没查到相近称呼。可能是新外号，可以直接问用户。"
        exact = [h for h in hits if h.get("exact")]
        if exact:
            persons = list(dict.fromkeys(h["person"] for h in exact))
        else:
            near = "、".join(f"「{h['alias']}」" for h in hits[:5])
            return (f"没有精确叫「{q}」的人。相近的登记称呼有：{near}。"
                    f"用更接近的那个原样称呼再查一次。")
        idmap = self.store.person_identity(persons, alias_cap=12)
        parts = []
        tier_cn = {0: "显示名", 1: "常用名", 2: "外号", 3: "叫错的"}
        for p in persons:
            info = idmap.get(p)
            if not info:
                parts.append(f"{p}（库里没有档案）")
                continue
            who = info["shown"]
            bits = []
            if info["nick"] and info["nick"] != who:
                bits.append(f"QQ昵称 {info['nick']}")
            if info["qq"]:
                bits.append(f"QQ {info['qq']}")
            als = [f"{nm}({tier_cn.get(t, '外号')})"
                   for nm, t in (info["aliases"] or []) if nm and nm != who][:10]
            if als:
                bits.append("登记称呼：" + "、".join(als))
            tail = f"：{'；'.join(bits)}" if bits else ""
            parts.append(f"「{q}」= {who}{tail}")
        return "\n".join(parts)

    @filter.llm_tool(name="memory_search")
    @_docstring(_memory_search_doc(True))
    async def tool_memory_search(self, event: AstrMessageEvent, query: str,
                                 scope: str = "all", limit: int = 8):
        # docstring 由上面的 _docstring 塞进来（注册时先给「有注入」那版），
        # 运行中再按 inject_enabled 改写，见 _sync_tool_docs。文案只有
        # _memory_search_doc 一处，别在这里再写一份，否则两份迟早对不上。
        q = self._norm_name(query)
        if len(q) < 2:
            return "关键词太短，至少 2 个字。请从用户消息里提取具体的人名/事件再查。"
        try:
            limit = max(1, min(int(limit or 8), 15))
        except (TypeError, ValueError):
            limit = 8
        scope = str(scope or "all").strip().lower()
        now = time.time()
        # 只搜「这台机器人自己的」对话与外发记录：多账号共用一个库时，
        # 不加这条会把 A 机器人的外发记录当成 B 说过的，回答「你发过什么」直接张冠李戴。
        bot = self._iso(self._event_bot(event))

        def _score(text: str) -> float:
            """这条内容和关键词有多像：2 = 整串命中，0~1 = 2 字切片的命中比例。

            为什么不能只做整串子串匹配：那等于要求用户的问法和库里的写法一字不差。
            实测翻车：库里明明躺着「你按定时任务「早安」发给…」，
            AI 拿用户原话「定时回复」去查，整整四个字差一个，结果是零命中，
            它转头就跟用户说「没有定时任务」——记录白写了，人还以为插件坏了。
            """
            t = self._norm_name(text)
            if q in t:
                return 2.0
            if len(q) <= 2:
                return 0.0
            grams = list(dict.fromkeys(q[i:i + 2] for i in range(len(q) - 1)))
            if not grams:
                return 0.0
            ratio = sum(1 for g in grams if g in t) / len(grams)
            if ratio >= 0.5:
                return ratio
            # 切片命中率不够、但至少有一个 2 字切片对上，且字几乎都出现在
            # 记录里时也算命中——「同一件事换了个说法」的问法自救通道。
            # 实测：库里躺着「你按定时任务「早安」…不用回复」，
            # AI 拿用户原话「定时回复」去查，切片只有「定时」对上（1/3），
            # 结果是零命中，它转头跟用户说「没有定时任务」。
            if any(g in t for g in grams):
                chars = list(dict.fromkeys(q))
                if chars:
                    cover = sum(1 for ch in chars if ch in t) / len(chars)
                    if cover >= 0.7:
                        return 0.6
            return ratio

        def _hit(text: str) -> bool:
            return _score(text) >= 0.5

        def _fmt_ts(ts) -> str:
            try:
                return time.strftime("%m-%d %H:%M", time.localtime(float(ts or 0)))
            except (TypeError, ValueError, OSError):
                return "?"

        parts: list[str] = []
        if scope in ("all", "fact"):
            rows = [r for r in self.store.query_active_l2(limit=400)
                    if not r.get("expires_at") or r["expires_at"] > now]
            scored = []
            for r in rows:
                # 原话(source_quote)也要参与打分：原子拆分后，一条长话被切成
                # 多条断言，每条 value 只是片段；只有带上 source_quote 才能按
                # 「用户当年说的整句」召回回来。面板检索早就这么做了，这里对齐。
                s = _score(f"{r.get('subject','')}{r.get('predicate','')}"
                           f"{r.get('value','')}{r.get('source_quote','')}")
                if s >= 0.5:
                    scored.append((s, r))
            scored.sort(key=lambda x: (-x[0], -int(x[1].get("id") or 0)))
            for _s, r in scored[:limit]:
                sq = r.get("source_quote") or ""
                tail = f"｜原话：{sq}" if sq and sq != r.get("value") else ""
                parts.append(f"[事实#{r['id']}] {r['subject']}的{r['predicate']}：{r['value']}{tail}")
        if scope in ("all", "chat"):
            msgs = (self.store.query_recent_l3(limit=300, event_type="user_message",
                                               bot_id=bot)
                    + self.store.query_recent_l3(limit=300, event_type="bot_reply",
                                                 bot_id=bot))
            msgs.sort(key=lambda r: r.get("id", 0), reverse=True)
            scored = []
            for r in msgs:
                s = _score(r.get("content", ""))
                if s >= 0.5:
                    scored.append((s, r))
            scored.sort(key=lambda x: (-x[0], -int(x[1].get("id") or 0)))
            n = 0
            for _s, r in scored:
                if n >= limit:
                    break
                parts.append(f"[聊天#{r['id']}|{_fmt_ts(r.get('created_at'))}] {r['content']}")
                n += 1
            # 「最近其他会话来信」兜底。为什么必须有：关键词检索是字面匹配，
            # 用户问「有没有人给你发消息」时，AI 会拿「发消息」当 query——
            # 对方刚发的「你好」「请求添加你为好友」里没有这三个字，永远查不到，
            # AI 就会信誓旦旦说「没有人找你」（实测 2026-10-28 00:59 翻车现场，
            # 注入包里明明有 用户丁 的两条 fresh 消息，被工具结果压过去了）。
            # 所以查聊天记录时，不管命中多少，都附上最近 30 分钟其他会话的来信。
            here_sid = str(getattr(event, "unified_msg_origin", "") or "")
            recent_other = [
                r for r in self.store.query_recent_l3(limit=60, event_type="user_message",
                                                     bot_id=bot)
                if str(r.get("session_id") or "") != here_sid
                and now - float(r.get("created_at") or 0) <= 1800
            ]
            recent_other.sort(key=lambda r: r.get("id", 0), reverse=True)
            if recent_other:
                lines = "；".join(
                    f"[{_fmt_ts(r.get('created_at'))}] {str(r.get('content', ''))[:40]}"
                    for r in recent_other[:3]
                )
                parts.append(f"[最近30分钟其他会话来信（关键词查不到的也在这）] {lines}")
            elif n == 0:
                parts.append("[最近30分钟其他会话来信] 无——最近半小时没有任何其他会话发来消息。")
        if scope in ("all", "outbound"):
            rows = self.store.query_recent_l3(limit=100, event_type="bot_outbound",
                                              bot_id=bot)
            try:
                nick_map = self.store.recent_nick_by_session()
            except Exception:
                nick_map = {}
            def _ob_line(r) -> str:
                c = str(r.get("content") or "")
                # 「用户 123456003」是冷号码，AI 看了也认不出是谁；补上昵称
                nick = nick_map.get(str(r.get("session_id") or ""), "")
                if nick:
                    c = re.sub(r"(用户\s?\d{5,12})(?!\()", rf"\g<1>（{nick}）", c, count=1)
                return f"[外发#{r['id']}|{_fmt_ts(r.get('created_at'))}] {c}"

            scored = []
            for r in rows:
                s = _score(r.get("content", ""))
                if s >= 0.5:
                    scored.append((s, r))
            scored.sort(key=lambda x: (-x[0], -int(x[1].get("id") or 0)))
            n = 0
            for _s, r in scored:
                if n >= limit:
                    break
                parts.append(_ob_line(r))
                n += 1
            # 一个都没命中时别直接回「没有」。外发记录本来就少（一天几条），
            # 关键词稍微偏一点就全漏，而 AI 拿到「没查到」会当众否认自己刚发过的话
            # ——记录明明在库里，用户看到的却是「它失忆了」。所以兜个底，
            # 把最近几条摆出来让它自己看，并明确禁止它据此说「没有」。
            if not scored and rows:
                # 兜底优先摆「发到当前会话」的：跨会话提问时，AI 要证明的是
                # 「我有没有跟眼前这个人说过」，别的会话的外发反而会把它带偏。
                here_sid = str(getattr(event, "unified_msg_origin", "") or "")
                here_rows = [r for r in rows
                             if str(r.get("session_id") or "") == here_sid][:5]
                picked = here_rows if here_rows else rows[:5]
                where = "本会话" if here_rows else "所有会话"
                parts.append(
                    f"「{query}」在外发记录里没精确命中。以下是最近 {len(picked)} 条"
                    f"你发到{where}的话，自己看有没有相关的——"
                    "**不许因为关键词没命中就说「没有/不记得」**："
                )
                for r in picked:
                    parts.append(_ob_line(r))
        if not parts:
            where = {"fact": "长期事实", "chat": "聊天记录", "outbound": "外发记录"}.get(scope, "记忆库")
            return f"「{query}」在{where}里没有查到相关记录。可以换个关键词，或如实告诉用户你不记得。"
        self._record_tokens("memory_search", source="estimate", input=estimate_tokens("\n".join(parts)))
        return "\n".join(parts[: limit * 3])

    @filter.llm_tool(name="outbound_history")
    async def tool_outbound_history(self, event: AstrMessageEvent, kind: str = "all",
                                    hours: int = 168, limit: int = 10):
        """翻阅你过往主动发出去的话的历史记录：转述（替人带的话）和定时任务发的话。

        和 memory_search 的分工：那是拿关键词「搜」某一条；这个是按时间「翻」
        一段时期内的全部外发记录。使用时机：
        - 用户问「你上周都发了什么」「列一下定时任务发过的消息」「我让你带过哪些话」
        - 用户想核对某条转述/定时消息的原文、收件人或发送时间
        - 记忆包里没有的老转述/老定时任务（过了新鲜窗不常驻注入），只有这里能翻到

        Args:
            kind(string): 查哪类：all=转述+定时（默认），relay=只看转述，cron=只看定时任务
            hours(int): 往前翻多久：24=一天，168=一周（默认），720=30天，0=全部历史
            limit(int): 最多返回几条，默认 10，最大 20
        """
        if not bool(self._cfg("tool_outbound_history", True)):
            return "外发历史翻阅功能当前为关闭。"
        try:
            hours = max(0.0, float(hours if hours is not None else 168))
        except (TypeError, ValueError):
            hours = 168.0
        try:
            limit = max(1, min(int(limit or 10), 20))
        except (TypeError, ValueError):
            limit = 10
        fetch = 500 if hours <= 0 else 400
        try:
            rows = self.store.query_recent_l3(
                limit=fetch, event_type="bot_outbound",
                bot_id=self._iso(self._event_bot(event)))
        except Exception:
            rows = []
        try:
            nick_map = self.store.recent_nick_by_session()
        except Exception:
            nick_map = {}
        lines = outbound_history_lines(rows, kind=kind, hours=hours, limit=limit,
                                       nick_map=nick_map)
        self._record_tokens("outbound_history", source="estimate",
                            input=estimate_tokens("\n".join(lines)))
        return "\n".join(lines)

    async def terminate(self):
        # 先把 context.send_message 还原，再关库：否则插件停用/重载后这个壳还挂在
        # 共享的 context 上，别家插件一发送就会往已关闭的 store 里写，抛一串异常。
        try:
            ctx = self.context
            cur = getattr(ctx, "send_message", None)
            orig = getattr(self, "_orig_send_message", None)
            if orig is not None and getattr(cur, "_sugarmemory_wrapped", False):
                ctx.send_message = orig
                self._orig_send_message = None
        except Exception:
            logger.debug("[SugarMemory] 还原 send_message 失败", exc_info=True)
        try:
            self.store.close()
        except Exception:
            pass
