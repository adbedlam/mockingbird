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

## Prompt Building

Mockingbird uses a generated inner prompt to capture the target person's stable communication style, personality, interests, and other behavioral characteristics. The prompt is assembled from modular components and generated from conversation history using an LLM.

Rebuild the inner prompt whenever the underlying prompt components or source conversation history change. The generated prompt is an artifact, so it should be regenerated rather than edited manually.

## Development

The project targets **Python 3.13** and uses [uv](https://docs.astral.sh/uv/) for the virtualenv and dependencies.

| Command | Action |
| --- | --- |
| `make install` / `make dev` | `uv sync` |
| `make test` | run pytest |
| `make lint` | `ruff check` |
| `make format` | `ruff format` |
| `uv run python -m mockingbird.scripts.build_inner` | Generate inner prompt |
| `uv run python -m mockingbird.cli` | Start Mockingbird |

### Configuration

The project uses two configuration files: `.env` for secrets and `config.yaml` for application settings.

#### Environment variables

Create a `.env` file in the project root based on `.env.example`:

```dotenv
LLM_API_KEY=
```

`LLM_API_KEY` contains the API key used to authenticate with the configured LLM provider. Set it to your OpenRouter API key when using the default model configuration.

Keep `.env` private and do not commit it to version control. `.env.example` documents the required environment variables without containing actual credentials.

#### config.yaml

The YAML configuration controls the data source, conversation participants, model settings, message limits, and data paths.

- `filename`: Name of the output file for the processed result, such as `result.json`.
- `messenger`: Source messaging platform. Currently configured as `telegram`.
- `user`: Person whose communication style Mockingbird imitates.
- `other`: Conversation partner whose messages provide relational context.
- `llm.host`: Base URL of the LLM API.
- `llm.model`: Model identifier sent to the API.
- `llm.building_temperature`: Sampling temperature used when generating the inner prompt.
- `llm.inference_temperature`: Sampling temperature used for conversational responses.
- `max_messages`: Maximum number of messages to process from the source history.
- `paths.raw`: Directory containing raw input data.
- `paths.processed`: Directory for processed data and generated artifacts, including `inner_prompt.txt`.
