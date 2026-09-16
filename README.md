# Mockingbird

> **mockingbird**
> *noun*
> **mock·ing·bird** · /ˈmä-kiŋ-ˌbərd, ˈmȯ-/
>
> *a common grayish North American bird* (*Mimus polyglottos*) *related to the thrashers that is remarkable for its exact imitations of the notes of other birds.*
>
> — *Merriam-Webster*

## Goal

The goal of this project is to explore whether a large language model can simulate the communication style, behavioral patterns, personality traits, long-term memory, and autonomous behavior of a real person based on their conversation history.

The project aims to build a persistent digital agent that can:

* communicate in a way that resembles the target person's writing style and behavioral patterns;
* maintain long-term memories without relying on a continuously growing context window;
* keep a personal diary and use it as part of its long-term state;
* develop and update an internal representation of the person it interacts with;
* make autonomous decisions about when to initiate a conversation;
* adapt its behavior based on previous interactions and accumulated memories;
* allow controlled experimentation with different personality, memory, and behavior configurations.

The project is primarily an engineering and research experiment focused on long-term memory, personality simulation, autonomous agents, and human-agent interaction.

## Development

The project targets **Python 3.13** and uses [uv](https://docs.astral.sh/uv/) for the virtualenv and dependencies.

```powershell
uv sync
```

This creates `.venv` in the project root and installs the package in editable mode together with the dev tools (`pytest`, `ruff`). Cursor is configured to select that interpreter and activate the virtualenv in new terminals.

| Command | Action |
| --- | --- |
| `make install` / `make dev` | `uv sync` |
| `make test` | run pytest |
| `make lint` | `ruff check` |
| `make format` | `ruff format` |
