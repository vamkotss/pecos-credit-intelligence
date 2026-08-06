"""Build and work the human review queue.

    python scripts/review_queue.py build --all
    python scripts/review_queue.py list
    python scripts/review_queue.py list --deal PCP-0004
    python scripts/review_queue.py decide PCP-0004-RECO-00 --accept --by "V Kota"
    python scripts/review_queue.py status

State lives in `reports/review/` as two files: `items.jsonl` for current state
and `events.jsonl` as an append-only log of every decision with its reviewer and
timestamp. A credit file is a regulated artefact, and "who approved this" is a
question asked years later.

Exit code 7 from `status` while any blocking item is unreviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pecos.chunking import load_chunks  # noqa: E402
from pecos.config import REPO_ROOT  # noqa: E402
from pecos.drafting import AnthropicDrafter, TemplateDrafter  # noqa: E402
from pecos.guardrails import check_memo  # noqa: E402
from pecos.memo import MEMO_QUESTIONS, MemoWriter, extract_figures  # noqa: E402
from pecos.retrieval import HybridRetriever  # noqa: E402
from pecos.review import (  # noqa: E402
    Decision,
    ReviewQueue,
    build_queue_for_memo,
    format_queue,
)


def _queue_dir(args) -> Path:
    return args.queue or (REPO_ROOT / "reports" / "review")


def _contexts(retriever, deal_id: str, k: int = 6) -> list[dict]:
    from pecos.answering import contexts_from_hits

    seen: set[str] = set()
    contexts: list[dict] = []
    for _, question in MEMO_QUESTIONS:
        for context in contexts_from_hits(retriever.retrieve(question, deal_id, k=k)):
            if context["chunk_id"] not in seen:
                seen.add(context["chunk_id"])
                contexts.append(context)
    return contexts


def cmd_build(args) -> int:
    chunks_dir = args.chunks or (REPO_ROOT / "data" / "interim" / "chunks")
    if not chunks_dir.is_dir() or not any(chunks_dir.glob("PCP-*.jsonl")):
        print(f"No chunks at {chunks_dir}.")
        return 1

    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("PCP-*.jsonl")):
        chunks.extend(load_chunks(path))
    retriever = HybridRetriever()
    retriever.build(chunks)

    deals = sorted(retriever.indexes) if args.all else [args.deal]
    drafter = TemplateDrafter() if args.drafter == "template" else AnthropicDrafter()
    writer = MemoWriter(retriever=retriever, drafter=drafter, k=6)

    queue = ReviewQueue.load(_queue_dir(args))
    existing = {item.item_id for item in queue.items}
    added = 0

    for deal_id in deals:
        result = writer.write(deal_id)
        contexts = _contexts(retriever, deal_id)
        guard = check_memo(
            deal_id,
            result.text,
            result.computations,
            contexts,
            figures=extract_figures(contexts),
        )
        for item in build_queue_for_memo(deal_id, result, guard):
            # Rebuilding must not duplicate or silently overwrite a decision
            # somebody already made.
            if item.item_id in existing:
                continue
            queue.add(item)
            existing.add(item.item_id)
            added += 1
        status = "releasable" if queue.releasable(deal_id) else "HELD"
        print(f"{deal_id}  {status:<11} pending={len(queue.pending(deal_id))}")

    queue.save(_queue_dir(args))
    print(f"\nadded {added} new item(s) -> {_queue_dir(args)}")
    return 0


def cmd_list(args) -> int:
    queue = ReviewQueue.load(_queue_dir(args))
    print(format_queue(queue, args.deal))
    return 0


def cmd_decide(args) -> int:
    queue = ReviewQueue.load(_queue_dir(args))
    decision = (
        Decision.ACCEPTED
        if args.accept
        else Decision.REJECTED
        if args.reject
        else Decision.ESCALATED
    )
    try:
        item = queue.decide(args.item_id, decision, args.by, args.note)
    except (KeyError, ValueError) as error:
        print(error)
        return 1
    queue.save(_queue_dir(args))
    print(item.render())
    if not queue.releasable(item.deal_id):
        print(
            f"\n{item.deal_id} still held: "
            f"{len(queue.blocking(item.deal_id))} blocking item(s) remain."
        )
    else:
        print(f"\n{item.deal_id} is releasable.")
    return 0


def cmd_status(args) -> int:
    queue = ReviewQueue.load(_queue_dir(args))
    counts = queue.counts()
    if not counts:
        print("review queue is empty")
        return 0
    for key in sorted(counts):
        print(f"  {key:<48} {counts[key]}")
    blocking = queue.blocking()
    print(f"\npending  {len(queue.pending())}")
    print(f"blocking {len(blocking)}")
    if blocking:
        held = sorted({item.deal_id for item in blocking})
        print(f"held     {', '.join(held)}")
        return 7
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pecos human review queue.")
    parser.add_argument("--queue", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write memos and queue what needs a human")
    build.add_argument("--deal", default=None)
    build.add_argument("--all", action="store_true")
    build.add_argument("--chunks", type=Path, default=None)
    build.add_argument(
        "--drafter", default="template", choices=["template", "anthropic"]
    )
    build.set_defaults(func=cmd_build)

    listing = sub.add_parser("list", help="show pending items, most serious first")
    listing.add_argument("--deal", default=None)
    listing.set_defaults(func=cmd_list)

    decide = sub.add_parser("decide", help="record a decision")
    decide.add_argument("item_id")
    group = decide.add_mutually_exclusive_group(required=True)
    group.add_argument("--accept", action="store_true")
    group.add_argument("--reject", action="store_true")
    group.add_argument("--escalate", action="store_true")
    decide.add_argument("--by", required=True, help="reviewer name")
    decide.add_argument("--note", default="")
    decide.set_defaults(func=cmd_decide)

    status = sub.add_parser("status", help="counts, and exit 7 if anything is held")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if args.command == "build" and not args.all and not args.deal:
        print("Pass --deal PCP-0001 or --all.")
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
