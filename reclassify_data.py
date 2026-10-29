"""一次性数据清洗：把旧 14 类谓词迁移到新 6 类 + 去重。

用法：
    python reclassify_data.py --db <path/to/memory.db> [--dry-run]

- 重新分类：按 value 内容语义重新判断每条的 predicate（不看旧标签）。
- 去重：同 subject 下 value 归一化指纹相同/包含的条目合并，旧的标 replaced_by。
- 默认 dry-run，只打印报告不动数据；加 --apply 才真正写入。
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sugarmemory_core.summarize_core import reclassify_value, is_single_valued
from sugarmemory_core.store import _norm_fingerprint, _fp_norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    rows = [dict(r) for r in db.execute(
        "SELECT * FROM l2_facts WHERE status='active'"
        " ORDER BY subject, id"
    )]

    print(f"共 {len(rows)} 条 active L2\n")
    print("=" * 80)
    print("【1】重新分类结果")
    print("=" * 80)

    reclass = []
    for r in rows:
        old = r["predicate"]
        new = reclassify_value(r["value"], old)
        reclass.append((r, old, new))
        flag = "" if old == new else "  <-- 变更"
        print(f"#{r['id']:>3} [{r['subject']}] {old!r:12} -> {new!r:12}{flag}")
        print(f"        value: {r['value'][:70]}")

    print("\n" + "=" * 80)
    print("【2】去重分析（同 subject 下 value 归一化指纹相同/包含）")
    print("=" * 80)

    by_subject = {}
    for r, old, new in reclass:
        by_subject.setdefault(r["subject"], []).append(r)

    merges = []  # (keep_id, drop_id, reason)
    for subject, items in by_subject.items():
        # 按 id 顺序，逐个与前面已保留的比对
        kept = []  # list of (id, norm_value, value)
        for r in items:
            nv = _norm_fingerprint(r["value"])
            dup = None
            for kid, knv, kval in kept:
                # 完全相同 或 一者包含另一者（换描述后更短/更长）
                if nv == knv or (nv and knv and (nv in knv or knv in nv)):
                    dup = (kid, kval)
                    break
            if dup:
                merges.append((dup[0], r["id"], dup[1], r["value"]))
                print(f"[{subject}] #{dup[0]} 吞并 #{r['id']}")
                print(f"    保留: {dup[1][:60]}")
                print(f"    去掉: {r['value'][:60]}")
            else:
                kept.append((r["id"], nv, r["value"]))

    if not merges:
        print("（未发现可安全合并的重复）")

    print("\n" + "=" * 80)
    print(f"【3】汇总：{len(reclass)} 条 · 分类变更 {sum(1 for _,o,n in reclass if o!=n)} 条 · 去重合并 {len(merges)} 组")
    print("=" * 80)

    if not args.apply:
        print("\n[预览模式] 未写入。确认无误后加 --apply 真正执行。")
        return 0

    now = time.time()
    with db:
        # 1. 重新分类
        for r, old, new in reclass:
            if old != new:
                fp = _fp_norm(r["subject"], new, r["value"])
                db.execute(
                    "UPDATE l2_facts SET predicate=?, fingerprint=?, updated_at=? WHERE id=?",
                    (new, fp, now, r["id"]),
                )
        # 2. 去重：把被吞并的标 replaced，replaced_by 指向保留的那条
        for keep_id, drop_id, _, _ in merges:
            db.execute(
                "UPDATE l2_facts SET status='replaced', replaced_by=? WHERE id=?",
                (keep_id, drop_id),
            )
    db.commit()
    print(f"\n[已执行] 分类变更 {sum(1 for _,o,n in reclass if o!=n)} 条，去重合并 {len(merges)} 组完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
