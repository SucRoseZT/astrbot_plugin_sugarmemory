"""一次性数据清洗：把 L2 里用「；」拼成一坨的 value 拆成独立记录。

背景：
    旧版把「自我相关」「评价认为」等分类误当成单值，多件独立的事被
    `_merge_value` 用「；」硬拼进同一条 value（例：「现在20岁，在加拿大留学，
    和哥哥一起；情绪极度迷茫绝望，疑似有自杀倾向」）。
    现已统一为 7 类全多值，本脚本把存量脏数据按「；」拆开，每件独立的事
    单独成一条 active 记录，原记录标 replaced、replaced_by 指向新首条。

用法：
    python split_l2_data.py --db <path/to/memory.db> [--dry-run]

- 只拆「；」/「;」分隔的多子项 value；单子项不动。
- 拆分片段沿用原记录的 subject / predicate / speaker / speech_act /
  source_type / source_msg_id / source_quote / source_summary_id /
  confidence / gate_reason / expires_at，重新计算 fingerprint。
- 默认 dry-run 只打印报告；加 --apply 才真正写入。
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sugarmemory_core.store import _fp_norm

_SPLIT_RE = __import__("re").compile(r"[；;]")


def split_value(value: str) -> list[str]:
    """按「；」/「;」拆 value，去掉空白与空片段。单片段返回原样（不拆）。"""
    v = (value or "").strip()
    if not v:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(v) if p.strip()]
    # 只有一个片段 = 本来就没拼，原样返回（含未命中的情况）
    if len(parts) <= 1:
        return [v]
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    rows = [dict(r) for r in db.execute(
        "SELECT * FROM l2_facts WHERE status='active' ORDER BY subject, id"
    )]

    # 找出需要拆的条目
    to_split = []
    for r in rows:
        parts = split_value(r["value"])
        if len(parts) > 1:
            to_split.append((r, parts))

    print(f"active L2 共 {len(rows)} 条，其中需要拆分的「；」拼接条目 {len(to_split)} 条\n")
    print("=" * 90)

    for r, parts in to_split:
        print(f"#{r['id']:>3} [{r['subject']}] {r['predicate']}（拆成 {len(parts)} 条）")
        for i, p in enumerate(parts, 1):
            print(f"    [{i}] {p[:60]}")

    if not to_split:
        print("\n（没有需要拆分的条目）")
        return 0

    print("\n" + "=" * 90)
    if not args.apply:
        print("\n[预览模式] 未写入。确认无误后加 --apply 真正执行。")
        return 0

    now = time.time()
    inserted = 0
    with db:
        for r, parts in to_split:
            first_new_id = None
            for p in parts:
                fp = _fp_norm(r["subject"], r["predicate"], p)
                cur = db.execute(
                    "INSERT INTO l2_facts (subject, predicate, value, speaker, speech_act,"
                    " source_type, source_msg_id, source_quote, source_summary_id, actionable,"
                    " confidence, status, stability, expires_at, replaced_by, gate_reason,"
                    " fingerprint, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,0,?,'active',?,?,0,?,?,?,?)",
                    (
                        r["subject"], r["predicate"], p,
                        r["speaker"], r["speech_act"], r["source_type"],
                        r["source_msg_id"], r["source_quote"], r["source_summary_id"],
                        r["confidence"], r["stability"], r["expires_at"],
                        r["gate_reason"], fp, r["created_at"], now,
                    ),
                )
                new_id = int(cur.lastrowid)
                if first_new_id is None:
                    first_new_id = new_id
                inserted += 1
            # 原记录标 replaced，指向拆出的第一条（保留历史可追溯）
            db.execute(
                "UPDATE l2_facts SET status='replaced', replaced_by=?, updated_at=? WHERE id=?",
                (first_new_id, now, r["id"]),
            )
    db.commit()
    print(f"\n[已执行] 拆分 {len(to_split)} 条 → 新插入 {inserted} 条独立记录，原记录已标 replaced。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
