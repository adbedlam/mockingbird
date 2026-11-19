import json
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import pytest
from test_memory_and_freshness import NOW, incoming
from test_pipeline import Model, bird_for

from mockingbird.chat.pipeline import ChatPipeline
from mockingbird.persona.diary.diary import Diary
from mockingbird.persona.diary.entry import Entry
from mockingbird.persona.diary.runtime import DiaryMemory


def memory_at(path, base=None, **kwargs):
    return DiaryMemory(path, "Persona", "Partner", base or Diary(), **kwargs)


def operation(batch, op="add", target=None, **values):
    row = json.loads(batch.messages[-1]["content"])["incoming_evidence"][0]
    result = {
        "op": op,
        "timestamp": row["timestamp"],
        "source_ids": [row["id"]],
        "reason": "Evidence-grounded decision",
    }
    if target:
        result.update(target_id=target.memory_id, target_revision=target.revision)
    if op == "add":
        result.update(
            content="Partner told me they plan to finish the project tests",
            subject="partner",
            topic="project.tests",
            state="planned",
        )
    result.update(values)
    return result


def apply(memory, batch, operations, now=NOW):
    return memory.apply(
        batch,
        json.dumps({"operations": operations, "reason": "Processed new evidence"}),
        now,
    )


def seed(memory):
    memory.ingest([incoming("I plan to finish the project tests")])
    batch = memory.prepare(NOW)
    apply(memory, batch, [operation(batch)])
    return memory.combined().entries[0]


def test_paraphrase_confirmation_then_completion_keep_one_memory_and_all_versions(
    tmp_path,
):
    memory = memory_at(tmp_path)
    original = seed(memory)
    memory.ingest([incoming("I'm going to wrap up the tests", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    result = apply(
        memory,
        batch,
        [operation(batch, "confirm", original)],
        NOW + timedelta(seconds=10),
    )
    assert (result.added, result.confirmed) == (0, 1)
    confirmed = memory.combined().entries[0]
    assert confirmed.memory_id == original.memory_id
    assert confirmed.content == original.content
    assert len(confirmed.source_ids) == 2
    memory.ingest([incoming("The tests are finished", seconds=20)])
    batch = memory.prepare(NOW + timedelta(seconds=20))
    result = apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                confirmed,
                content="Partner told me the tests are finished",
                state="completed",
            )
        ],
        NOW + timedelta(seconds=20),
    )
    assert (result.added, result.updated) == (0, 1)
    restored = memory_at(tmp_path)
    latest = restored.combined().entries
    assert len(latest) == 1
    assert latest[0].memory_id == original.memory_id
    assert latest[0].state == "completed" and latest[0].revision == 2
    assert len(latest[0].source_ids) == 3
    chain = restored.versions[str(original.memory_id)]
    assert [e.state for e in chain] == ["planned", "planned", "completed"]
    assert chain[0].source_ids == original.source_ids
    assert "plan to finish" not in (tmp_path / "diary.md").read_text(encoding="utf-8")
    assert "plan to finish" in (tmp_path / "diary_history.md").read_text(
        encoding="utf-8"
    )
    assert "confirm" in (tmp_path / "diary_updates.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("state", ["completed", "cancelled"])
def test_closed_task_and_explicit_reopening_are_versions_not_new_records(
    tmp_path, state
):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    for offset, next_state in ((10, state), (20, "planned")):
        memory.ingest([incoming(f"Explicit task state: {next_state}", seconds=offset)])
        batch = memory.prepare(NOW + timedelta(seconds=offset))
        apply(
            memory,
            batch,
            [
                operation(
                    batch,
                    "update",
                    entry,
                    content=f"Partner reported {next_state}",
                    state=next_state,
                )
            ],
            NOW + timedelta(seconds=offset),
        )
        entry = memory.combined().entries[0]
    assert len(memory.entries) == 1
    assert entry.revision == 2 and entry.state == "planned"
    assert memory.versions[str(entry.memory_id)][1].state == state


def test_older_evidence_cannot_reopen_a_newer_closed_task(tmp_path):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("Finished", seconds=30)])
    batch = memory.prepare(NOW + timedelta(seconds=30))
    apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                entry,
                content="Partner finished the task",
                state="completed",
            )
        ],
        NOW + timedelta(seconds=30),
    )
    entry = memory.combined().entries[0]
    memory.ingest([incoming("Earlier still planned", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=40))
    result = apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                entry,
                content="Partner still plans it",
                state="planned",
            )
        ],
        NOW + timedelta(seconds=40),
    )
    assert result.skipped == 1 and result.updated == 0
    assert memory.combined().entries[0] == entry
    assert not memory.pending
    assert "Источник старше" in result.changes[0]["reason"]


def test_conflicting_claims_stay_uncertain_and_keep_previous_fact(tmp_path):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("This conflicts with the earlier report", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    result = apply(
        memory,
        batch,
        [
            operation(
                batch,
                "dispute",
                entry,
                content="Partner's two claims conflict; I need to clarify",
                state="uncertain",
            )
        ],
        NOW + timedelta(seconds=10),
    )
    assert result.disputed == 1
    assert memory.combined().entries[0].state == "uncertain"
    assert memory.versions[str(entry.memory_id)][0].content == entry.content


def test_reused_key_cannot_silently_add_or_overwrite_an_existing_fact(tmp_path):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("Different claim under the same topic", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    result = apply(
        memory,
        batch,
        [operation(batch, content="Different fact with reused key")],
        NOW + timedelta(seconds=10),
    )
    assert len(memory.entries) == 1
    assert result.added == 0 and result.disputed == 1
    assert entry.content in memory.entries[0].content
    assert "Different fact" in memory.entries[0].content
    assert memory.entries[0].state == "uncertain"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(target_id="invented"),
        lambda o: o.update(target_revision=42),
        lambda o: o.update(target_revision=True),
        lambda o: o.update(source_ids=["invented"]),
        lambda o: o.update(timestamp="2030-01-01T00:00:00+03:00"),
        lambda o: o.update(subject="persona"),
        lambda o: o.update(topic="another.task"),
        lambda o: o.update(state="mystery"),
        lambda o: o.update(content=""),
        lambda o: o.update(reason=""),
        lambda o: o.update(op="delete_history"),
    ],
)
def test_invalid_update_preserves_pending_and_full_revision_history(tmp_path, mutate):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("new information", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    update = operation(
        batch, "update", entry, content="New supported content", state="completed"
    )
    mutate(update)
    before = (tmp_path / "memory.json").read_bytes()
    with pytest.raises(ValueError):
        apply(memory, batch, [update], NOW + timedelta(seconds=10))
    assert (tmp_path / "memory.json").read_bytes() == before
    assert len(memory.pending) == 1
    assert memory.entries[0] == entry
    assert len(memory.versions[str(entry.memory_id)]) == 1


def test_confirmation_cannot_rewrite_state_and_dispute_cannot_assert_certainty(
    tmp_path,
):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("new message", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    for op in [
        operation(batch, "confirm", entry, state="completed"),
        operation(batch, "dispute", entry, content="A conflict", state="fact"),
    ]:
        with pytest.raises(ValueError):
            apply(memory, batch, [op], NOW + timedelta(seconds=10))
    assert memory.entries[0] == entry


def test_valid_confirmation_can_use_the_models_batch_note_without_inventing_a_reason(
    tmp_path,
):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("Same intent in different words", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    item = operation(batch, "confirm", entry)
    item.pop("reason")
    result = apply(memory, batch, [item], NOW + timedelta(seconds=10))
    assert result.confirmed == 1
    assert result.changes[0]["reason"] == result.reason
    assert result.changes[0]["reason_scope"] == "batch"


def test_unreferenced_incoming_is_visible_as_skip_and_all_sources_remain(tmp_path):
    memory = memory_at(tmp_path)
    memory.ingest([incoming(), incoming("Transient greeting", seconds=1)])
    batch = memory.prepare(NOW + timedelta(seconds=1))
    result = apply(memory, batch, [operation(batch)], NOW + timedelta(seconds=1))
    assert result.added == 1 and result.skipped == 1
    assert result.changes[-1]["reason_scope"] == "application"
    assert len(memory.processed) == len(memory.sources) == 2
    assert not memory.pending


def test_same_topic_for_different_people_is_not_deduplicated(tmp_path):
    memory = memory_at(tmp_path)
    seed(memory)
    memory.ingest([incoming("You also plan to finish tests", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    result = apply(
        memory,
        batch,
        [
            operation(
                batch,
                subject="persona",
                content="Partner says I also plan to finish tests",
            )
        ],
        NOW + timedelta(seconds=10),
    )
    assert result.added == 1
    assert {e.subject for e in memory.combined().entries} == {"partner", "persona"}


def test_base_diary_is_overlaid_without_modifying_archive_and_has_history(tmp_path):
    base = Diary(
        [
            Entry(
                NOW - timedelta(days=1),
                "Partner planned a task",
                ["message:archive"],
                state="planned",
                subject="partner",
                topic="project.tests",
            )
        ]
    )
    original = asdict(base.entries[0])
    memory = memory_at(tmp_path, base)
    entry = memory.combined().entries[0]
    memory.ingest([incoming("Task is canceled")])
    batch = memory.prepare(NOW)
    apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                entry,
                content="Partner canceled the task",
                state="cancelled",
            )
        ],
    )
    assert asdict(base.entries[0]) == original
    restored = memory_at(tmp_path, base)
    assert len(restored.combined().entries) == 1
    assert restored.combined().entries[0].state == "cancelled"
    assert restored.versions[str(entry.memory_id)][0].source_ids == ["message:archive"]


def test_atomic_batch_and_failed_disk_write_preserve_all_prior_versions(
    tmp_path, monkeypatch
):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("Task completed", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    valid = operation(
        batch, "update", entry, content="Partner completed task", state="completed"
    )
    invalid = operation(batch, source_ids=["not-in-batch"], topic="another")
    disk = (tmp_path / "memory.json").read_bytes()
    with pytest.raises(ValueError):
        apply(memory, batch, [valid, invalid], NOW + timedelta(seconds=10))
    assert (tmp_path / "memory.json").read_bytes() == disk
    assert memory.entries[0] == entry

    def fail():
        raise OSError("simulated disk error")

    monkeypatch.setattr(memory, "save", fail)
    with pytest.raises(OSError):
        apply(memory, batch, [valid], NOW + timedelta(seconds=10))
    assert memory.entries[0] == entry
    assert len(memory.versions[str(entry.memory_id)]) == 1
    assert len(memory.pending) == 1


def test_unseen_or_changed_target_cannot_be_updated_from_stale_snapshot(tmp_path):
    memory = memory_at(tmp_path, context_entries=1)
    entry = seed(memory)
    memory.ingest([incoming("Other task", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    apply(
        memory,
        batch,
        [operation(batch, topic="another.task", content="Partner's other task")],
        NOW + timedelta(seconds=10),
    )
    memory.ingest([incoming("Update the first task", seconds=20)])
    batch = memory.prepare(NOW + timedelta(seconds=20))
    with pytest.raises(ValueError, match="unseen"):
        apply(
            memory,
            batch,
            [operation(batch, "update", entry, content="changed", state="completed")],
            NOW + timedelta(seconds=20),
        )
    latest = memory.entries[-1]
    memory.entries[-1] = replace(latest, content="Changed outside this snapshot")
    with pytest.raises(ValueError, match="stale"):
        apply(
            memory,
            batch,
            [operation(batch, "confirm", latest)],
            NOW + timedelta(seconds=20),
        )


def test_version_one_state_migrates_without_dropping_evidence(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "memory.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pair": ["Persona", "Partner"],
                "entries": [
                    {
                        "timestamp": (NOW - timedelta(days=1)).isoformat(),
                        "content": "Partner planned a task",
                        "source_ids": ["old-source"],
                        "recorded_at": None,
                    }
                ],
                "pending": [],
                "sources": {},
                "processed": ["old-source"],
                "updates": [],
                "retry_at": None,
            }
        ),
        encoding="utf-8",
    )
    memory = memory_at(tmp_path)
    entry = memory.entries[0]
    memory.ingest([incoming("Task finished")])
    batch = memory.prepare(NOW)
    apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                entry,
                content="Partner finished task",
                state="completed",
                topic="project.tests",
            )
        ],
    )
    restored = memory_at(tmp_path)
    assert (
        json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))["version"]
        == 2
    )
    assert restored.entries[0].memory_id == entry.memory_id
    assert set(restored.entries[0].source_ids) >= {"old-source"}
    assert restored.versions[str(entry.memory_id)][0].recorded_at is None


def test_reply_context_contains_latest_state_not_superseded_pending_memory(tmp_path):
    memory = memory_at(tmp_path)
    entry = seed(memory)
    memory.ingest([incoming("Finished", seconds=10)])
    batch = memory.prepare(NOW + timedelta(seconds=10))
    apply(
        memory,
        batch,
        [
            operation(
                batch,
                "update",
                entry,
                content="Partner has completed the tests",
                state="completed",
            )
        ],
        NOW + timedelta(seconds=10),
    )
    bird = bird_for(Model())
    bird.memory, bird.diary = memory, memory.combined()
    context = bird.context(
        bird.scheduler.tick(NOW + timedelta(seconds=20)), NOW + timedelta(seconds=20)
    )
    diary_message = next(
        m["content"] for m in context if m["content"].startswith("Dated diary")
    )
    assert "state=completed" in diary_message
    assert "plan to finish" not in diary_message
    assert "completed and cancelled are closed" in diary_message


def test_pipeline_displays_add_confirm_and_update_with_current_revision(tmp_path):
    class MemoryModel:
        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            try:
                body = json.loads(messages[-1]["content"])
            except ValueError:
                return json.dumps({"send": True, "text": "понял", "reason": "ответ"})
            if "incoming_evidence" not in body:
                return json.dumps({"send": True, "text": "понял", "reason": "ответ"})
            self.calls += 1
            row = body["incoming_evidence"][0]
            item = {
                "source_ids": [row["id"]],
                "timestamp": row["timestamp"],
                "reason": "durable evidence",
            }
            if self.calls == 1:
                item.update(
                    op="add",
                    content="Partner plans the tests",
                    subject="partner",
                    topic="project.tests",
                    state="planned",
                )
            else:
                target = body["recent_existing_diary"][0]
                item.update(
                    op="confirm" if self.calls == 2 else "update",
                    target_id=target["memory_id"],
                    target_revision=target["revision"],
                )
                if self.calls == 3:
                    item.update(content="Partner finished the tests", state="completed")
            return json.dumps({"operations": [item], "reason": "batch processed"})

    model = MemoryModel()
    bird = bird_for(model)
    bird.memory = memory_at(tmp_path / "memory", update_every=1)
    bird.diary = bird.memory.combined()
    clock = [NOW]
    events = []
    pipeline = ChatPipeline(
        bird,
        tmp_path / "runtime",
        events.append,
        clock=lambda: clock[0],
        max_generations=6,
    )
    pipeline.start()
    for index, text in enumerate(
        ("I plan tests", "Still want to finish tests", "Finished tests"), 1
    ):
        pipeline.receive(text)
        clock[0] += timedelta(seconds=31)
        deadline = time.monotonic() + 3
        while (
            sum(e.stage == "DIARY_RESULT" for e in events) < index
            and time.monotonic() < deadline
        ):
            pipeline.step()
            time.sleep(0.005)
        assert sum(e.stage == "DIARY_RESULT" for e in events) == index
    pipeline.close()
    assert [
        e.stage
        for e in events
        if e.stage in ("DIARY_ADDED", "DIARY_CONFIRMED", "DIARY_UPDATED")
    ] == ["DIARY_ADDED", "DIARY_CONFIRMED", "DIARY_UPDATED"]
    assert bird.memory.entries[0].state == "completed"
    assert bird.memory.entries[0].revision == 2
    assert pipeline.generations == 6
