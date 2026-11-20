import argparse
import logging
from pathlib import Path

from mockingbird.config import load_config
from mockingbird.conversation.history import History
from mockingbird.llm.client import LLM
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.inner.inner import InnerPrompt
from mockingbird.persona.life import LifePrompt
from mockingbird.persona.source import (
    load_manifest,
    save_manifest,
    select_persona_history,
)
from mockingbird.preprocessing.factory import make_processor
from mockingbird.preprocessing.telegram.source import participant_ids
from mockingbird.preprocessing.vision import VisionModel

logger = logging.getLogger(__name__)


def _build_history(config: dict) -> History:
    logger.info("Building history")

    user = config["user"]
    other = config["other"]

    vision_config = config["vision_model"]
    vision_model = (
        VisionModel(
            vision_config["host"],
            vision_config["model"],
            vision_config["api_key"],
            vision_config["temperature"],
        )
        if vision_config["enabled"]
        else None
    )

    preprocessing_config = config["preprocessing"]
    latest = preprocessing_config.get("latest_messages")
    processor = make_processor(
        preprocessing_config["messenger"],
        vision_model,
        participant_ids(config),
    )

    history = processor.process(
        user,
        other,
        config["paths"]["raw"],
        0 if latest else preprocessing_config.get("message_start", 0),
        None if latest else preprocessing_config.get("message_end"),
    )
    if latest is not None:
        history = History.latest_history(
            history,
            latest,
            config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
        )

    logger.info("History built: %d messages", len(history.messages))
    return history


def _build_inner_prompt(config: dict, history: History) -> InnerPrompt:
    history = select_persona_history(history, config)
    logger.info("Building inner prompt")

    building_config = config["building_model"]
    model = LLM(
        building_config["host"],
        building_config["model"],
        building_config["api_key"],
        building_config["temperature"],
    )

    return InnerPrompt.from_history(model, history)


def _build_life_prompt(config: dict, history: History) -> LifePrompt:
    history = select_persona_history(history, config)
    logger.info("Building life prompt")

    building_config = config["building_model"]
    model = LLM(
        building_config["host"],
        building_config["model"],
        building_config["api_key"],
        building_config["temperature"],
    )

    return LifePrompt.from_history(model, history)


def _build_diary(config: dict, history: History) -> Diary:
    history = select_persona_history(history, config)
    logger.info("Building diary")

    building_config = config["building_model"]
    model = LLM(
        building_config["host"],
        building_config["model"],
        building_config["api_key"],
        building_config["temperature"],
    )

    diary_config = config["diary"]

    return Diary.from_history(
        model,
        history,
        first_n=diary_config["first_n"],
        batch_size=diary_config["batch_size"],
    )


def build() -> None:
    parser = argparse.ArgumentParser(
        description="Build Mockingbird artifacts. No args to run the entire pipeline, args to run selected."
    )

    parser.add_argument(
        "--history", action="store_true", help="Build conversation history"
    )
    parser.add_argument(
        "--inner-prompt", action="store_true", help="Build inner prompt"
    )
    parser.add_argument(
        "--life-prompt", action="store_true", help="Build life prompt"
    )
    parser.add_argument("--diary", action="store_true", help="Build diary")

    args = parser.parse_args()

    build_all = not any(vars(args).values())

    config = load_config()

    output_dir = Path(config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / config["paths"]["history_filename"]
    if args.history or build_all:
        logger.info("Building history")
        history = _build_history(config)
        history.save(history_path)
    elif args.inner_prompt or args.life_prompt or args.diary:
        logger.info("Loading history")
        history = History.load(history_path)

    inner_prompt_path = output_dir / config["paths"]["inner_prompt_filename"]
    if args.inner_prompt or build_all:
        logger.info("Building inner prompt")
        inner_prompt = _build_inner_prompt(config, history)
        inner_prompt.save(inner_prompt_path)
        manifest = load_manifest(output_dir)
        manifest["profile"] = History.history_info(
            select_persona_history(history, config),
            config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
        )
        save_manifest(output_dir, manifest)

    life_prompt_path = output_dir / config["paths"].get(
        "life_prompt_filename", "life_prompt.md"
    )
    if args.life_prompt or build_all:
        logger.info("Building life prompt")
        life_prompt = _build_life_prompt(config, history)
        life_prompt.save(life_prompt_path)
        manifest = load_manifest(output_dir)
        manifest["life"] = History.history_info(
            select_persona_history(history, config),
            config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
        )
        save_manifest(output_dir, manifest)

    diary_path = output_dir / config["paths"]["diary_filename"]
    if args.diary or build_all:
        logger.info("Building diary")
        diary = _build_diary(config, history)
        diary.save(diary_path)
        manifest = load_manifest(output_dir)
        manifest["diary"] = History.history_info(
            select_persona_history(history, config),
            config.get("scheduler", {}).get("timezone", "Europe/Moscow"),
        )
        save_manifest(output_dir, manifest)

    logger.info("Mockingbird build complete")


def main() -> None:
    build()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
