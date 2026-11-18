"""Prepare or generate a fresh profile and diary using the latest local source."""

import argparse
from pathlib import Path

from mockingbird.config import CONFIG_PATH, PROJECT_ROOT, load_config
from mockingbird.llm.client import LLM
from mockingbird.persona.refresh import (
    bootstrap_local_persona,
    plan_refresh,
    prepare_refresh,
    refresh_persona,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--local",
        action="store_true",
        help="Create a new evidence-only starter profile locally, without API calls or replacing existing files",
    )
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Save two exact requests locally; no model calls",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Make two building-model requests and replace validated artifacts",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/previews/persona-refresh"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.local:
        metadata = bootstrap_local_persona(config)
        print(
            f"Local starter profile saved: {metadata['profile']['example_count']} verbatim examples; no inferred biography or seed diary entries"
        )
    elif args.prepare:
        plan = plan_refresh(config)
        prepare_refresh(plan, args.output)
        print(
            f"Prepared locally: {plan.source['messages']} messages, {plan.source['first']} — {plan.source['last']}. Requests: {args.output}"
        )
    else:
        settings = config["building_model"]
        model = LLM(
            settings["host"],
            settings["model"],
            settings.get("api_key"),
            settings.get("temperature", 0.3),
        )
        refresh_persona(config, model, audit_dir=args.output)


if __name__ == "__main__":
    main()
