import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import mockingbird.llm.message as llm_message
from mockingbird.conversation.history import History
from mockingbird.conversation.message import Message as ConversationMessage
from mockingbird.llm.client import LLM
from mockingbird.llm.message import Role
from mockingbird.persona.diary import Diary
from mockingbird.persona.inner import build_inner_prompt
from mockingbird.preprocessing.factory import make_processor
from mockingbird.prompts.imitation import IMITATION
from mockingbird.prompts.supress_llm_behavior import SUPRESS_LLM_BEHAVIOR

logger = logging.getLogger(__name__)


class InnerPromptPipeline:
    """Build a full-history profile in resumable chronological chunks."""

    def __init__(self, user: str, other: str, model: LLM, chunk_size: int = 500):
        if chunk_size < 1:
            raise ValueError("profile_build.chunk_size must be at least 1")
        self.user = user
        self.other = other
        self.model = model
        self.chunk_size = chunk_size

    @classmethod
    def from_config(cls, config: dict) -> "InnerPromptPipeline":
        building_config = config["building_model"]
        model = LLM(
            building_config["host"],
            building_config["model"],
            building_config["api_key"],
            building_config["temperature"],
        )
        return cls(
            config["user"],
            config["other"],
            model,
            config.get("profile_build", {}).get("chunk_size", 500),
        )

    def build(
        self,
        history: History,
        output_path: Path,
        force: bool = False,
    ) -> str:
        messages = sorted(history.messages, key=lambda message: message.timestamp)
        if not messages:
            raise ValueError("No participant messages are available for profiling.")

        fingerprint = self._source_fingerprint(messages)
        checkpoint_path = output_path.with_name("profile_build.json")
        chunk_count = (len(messages) + self.chunk_size - 1) // self.chunk_size
        state = self._load_state(checkpoint_path, fingerprint, force)

        if state.get("complete") and output_path.exists() and not force:
            logger.info("Full-history profile is already built; reusing it")
            return output_path.read_text(encoding="utf-8")

        profiles: list[dict[str, Any]] = state["chunk_profiles"]
        for index in range(len(profiles), chunk_count):
            start = index * self.chunk_size
            chunk = messages[start : start + self.chunk_size]
            chunk_history = History(self.user, self.other, chunk)
            profile = build_inner_prompt(
                self.user, self.other, chunk_history, self.model
            )
            if not profile.strip():
                raise ValueError(f"Profile chunk {index + 1} returned empty output.")
            profiles.append(
                {
                    "index": index,
                    "message_count": len(chunk),
                    "start": chunk[0].timestamp.isoformat(),
                    "end": chunk[-1].timestamp.isoformat(),
                    "profile": profile,
                }
            )
            state["chunk_profiles"] = profiles
            self._save_state(checkpoint_path, fingerprint, state)
            logger.info("Built profile chunk %d/%d", index + 1, chunk_count)

        final_profile = self._consolidate(profiles, len(messages))
        if not final_profile:
            raise ValueError("Full-history profile synthesis returned empty output.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(output_path, final_profile)
        state["complete"] = True
        state["final_profile"] = final_profile
        self._save_state(checkpoint_path, fingerprint, state)
        logger.info(
            "Saved full-history profile from %d messages in %d chunks",
            len(messages),
            chunk_count,
        )
        return final_profile

    def _consolidate(self, profiles: list[dict[str, Any]], message_count: int) -> str:
        sections = []
        for profile in profiles:
            sections.append(
                f"CHUNK {profile['index'] + 1} "
                f"({profile['start']} to {profile['end']}, "
                f"{profile['message_count']} messages):\n{profile['profile']}"
            )
        prompt = f"""Create one complete communication profile for {self.user}, the modeled person, in conversation with {self.other}.

The material below covers all {message_count} selected messages in chronological order. Each section is an evidence-based profile of one part of the history.

Synthesize the sections into one useful inner prompt for future conversation. Keep recurring traits and interests, preserve concrete facts that remain useful, and distinguish stable patterns from temporary or time-specific states. If the sections show that the person changed over time, describe that carefully. Do not infer facts that are not supported, do not turn a single event into a permanent trait, and do not repeat the same guidance. Preserve distinctive wording in its original language where useful. Do not mention chunking or this synthesis process.

Write a concise, information-dense instruction document in English, following the supplied profile sections' style. It must guide another model to communicate as {self.user}; use {self.other}'s messages only for context about the relationship.

FULL-HISTORY PROFILE SECTIONS:
{chr(10).join(sections)}
"""
        return self.model.complete(
            [
                {
                    "role": Role.SYSTEM,
                    "content": "Synthesize only evidence-supported persona guidance.",
                },
                {"role": Role.USER, "content": prompt},
            ]
        ).strip()

    def _load_state(self, path: Path, fingerprint: str, force: bool) -> dict[str, Any]:
        if not force and path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if (
                state.get("source_fingerprint") == fingerprint
                and state.get("chunk_size") == self.chunk_size
            ):
                return state
        return {
            "source_fingerprint": fingerprint,
            "chunk_size": self.chunk_size,
            "chunk_profiles": [],
            "complete": False,
        }

    def _save_state(self, path: Path, fingerprint: str, state: dict[str, Any]) -> None:
        state["source_fingerprint"] = fingerprint
        state["chunk_size"] = self.chunk_size
        self._atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2))

    @staticmethod
    def _source_fingerprint(messages: list[ConversationMessage]) -> str:
        source = [
            {
                "message_id": message.message_id,
                "author": message.author,
                "timestamp": message.timestamp.isoformat(),
                "text": message.text,
                "picture_description": message.picture_description,
                "has_picture": message.picture_path is not None,
            }
            for message in messages
        ]
        serialized = json.dumps(source, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)


class Pipeline:
    """Owns the persona prompt, conversation context, model calls, and diary."""

    def __init__(
        self,
        user: str,
        other: str,
        model: LLM,
        inner_prompt: str,
        diary: Diary | None = None,
    ) -> None:
        self.user = user
        self.other = other
        self.model = model
        self.diary = diary or Diary.disabled(user, other)
        self.base_system_prompt = (
            IMITATION.format(user=user, other=other)
            + inner_prompt
            + SUPRESS_LLM_BEHAVIOR.format(user=user)
        )
        self.conversation_history: list[llm_message.Message] = []
        self._refresh_system_prompt()

    @classmethod
    def from_config(
        cls,
        config: dict,
        diary_update_every_messages: int | None = None,
        diary_path_root: Path | None = None,
        load_recent_context: bool = False,
    ) -> "Pipeline":
        user = config["user"]
        other = config["other"]

        inference_config = config["inference_model"]
        model = LLM(
            inference_config["host"],
            inference_config["model"],
            inference_config["api_key"],
            inference_config["temperature"],
        )

        prompt_path: Path = (
            config["paths"]["processed"]
            / f"{user}_{other}".replace(" ", "_")
            / "inner_prompt.txt"
        )
        inner_prompt = prompt_path.read_text(encoding="utf-8")
        diary = Diary.from_config(
            config,
            user,
            other,
            model,
            inner_prompt,
            diary_update_every_messages,
            diary_path_root,
        )
        logger.info("Loaded persona profile and persistent diary state")
        logger.info("Markdown journal and update log are enabled")
        pipeline = cls(user, other, model, inner_prompt, diary)
        runtime_context = config.get("runtime_context", {})
        recent_message_count = runtime_context.get("recent_messages", 0)
        if recent_message_count < 0:
            raise ValueError("runtime_context.recent_messages cannot be negative")
        if load_recent_context and recent_message_count:
            processor = make_processor(config["preprocessing"]["messenger"])
            history = processor.process(user, other, config["paths"]["raw"])
            participants = {user, other}
            recent_messages = sorted(
                (
                    message
                    for message in history.messages
                    if message.author in participants
                ),
                key=lambda message: message.timestamp,
            )[-recent_message_count:]
            pipeline.feed(recent_messages, record_diary=False)
            logger.info(
                "Loaded %d recent conversation messages as response context",
                len(recent_messages),
            )
        return pipeline

    def feed(
        self,
        messages: list[ConversationMessage],
        record_diary: bool = True,
    ) -> None:
        self.conversation_history.extend(
            llm_message.Message(
                role=(Role.ASSISTANT if message.author == self.user else Role.USER),
                content=message.text,
            )
            for message in messages
        )
        if record_diary:
            self._record_diary_messages(messages)
        self._refresh_system_prompt()
        logger.info("Added %d message(s) to persona context", len(messages))

    def respond(self) -> str:
        response = self.model.complete(self.conversation_history)
        self.conversation_history.append(
            llm_message.Message(role=Role.ASSISTANT, content=response)
        )
        logger.info("Generated a persona response")
        return response

    def respond_to(self, text: str) -> str:
        self.conversation_history.append(
            llm_message.Message(role=Role.USER, content=text)
        )
        request_messages = [
            {"role": str(message["role"]), "content": message["content"]}
            for message in self.conversation_history
        ]
        response = self.respond()
        self.diary.record_chat_turn(text, response, request_messages)
        self._record_diary_input(text)
        self._refresh_system_prompt()
        return response

    def _record_diary_messages(self, messages: list[ConversationMessage]) -> None:
        previous_count = len(self.diary.updates)
        self.diary.record_messages(messages)
        self._log_new_diary_updates(previous_count)

    def _record_diary_input(self, text: str) -> None:
        previous_count = len(self.diary.updates)
        self.diary.record_incoming(text)
        self._log_new_diary_updates(previous_count)

    def _log_new_diary_updates(self, previous_count: int) -> None:
        for update in self.diary.updates[previous_count:]:
            logger.info(
                "Diary review: %s after %d message(s)",
                update["decision"],
                update["message_count"],
            )

    def starve(self) -> None:
        self.conversation_history = [self.conversation_history[0]]

    def _refresh_system_prompt(self) -> None:
        system_prompt = self.diary.add_to_prompt(self.base_system_prompt)
        system_message = llm_message.Message(role=Role.SYSTEM, content=system_prompt)
        if self.conversation_history:
            self.conversation_history[0] = system_message
        else:
            self.conversation_history.append(system_message)
