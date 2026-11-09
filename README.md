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
| `uv run python -m mockingbird.scripts.seed_diary` | Process the latest configured message batch from the Telegram export into the diary |
| `uv run python -m mockingbird.mockingbird` | Start the local conversation loop |
| `uv run python -m mockingbird.mockingbird --preview-diary --preview-name chat-preview-clean --diary-every 4` | Chat using an isolated preview diary updated every four incoming messages |

### Configuration

The project uses two configuration files: `.env` for secrets and `config.yaml` for application settings.

Use [`config.example.yaml`](config.example.yaml) as a starting point for `config.yaml`. Set `user` to the modeled person and `other` to the conversation partner. The diary uses these configured names when it distinguishes speakers.

#### Environment variables

Create a `.env` file in the project root based on `.env.example`:

```dotenv
BUILDING_MODEL_API_KEY=
INFERENCE_MODEL_API_KEY=
VISION_MODEL_API_KEY=
```

Set the key used by each configured model when its host requires authentication. Keep `.env` private and do not commit it to version control.

`.env.example` documents these variables without containing actual credentials.

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
- `paths.diary`: Directory for persistent diary state, separate from source conversation data. Mockingbird stores one `diary.json` per configured participant pair under this directory.
- `diary.update_every_messages`: Number of new messages to accumulate before asking the inference model to add a diary passage. The example default is 100 messages (about 50 exchanges); messages below the threshold are saved and carried over to the next run.
- `profile_build.chunk_size`: Number of messages analyzed per profile-building request. The profile pipeline processes the full configured pair history in chunks of this size, saves progress, then synthesizes the chunk profiles into one `inner_prompt.txt`.
- `runtime_context.recent_messages`: Number of latest messages from the configured pair to preload into the conversational context at startup. These original exchanges provide concrete style and relationship examples; they are context only and are not re-added to the diary.

## Runtime Pipeline and Diary Inspection

`src/mockingbird/pipeline.py` contains the full-history profile builder and coordinates conversation history, model responses, and diary updates. `uv run python -m mockingbird.scripts.build_inner` processes every message from the configured pair, not only `preprocessing.message_end`; with the default 500-message chunk size and the current export it makes about 36 chunk-profile requests and one consolidation request. Progress is checkpointed in `profile_build.json` so an interrupted build can resume. The terminal loop and future Telegram transport call into the runtime pipeline; transport-specific sending stays outside it.

For a safe chat rehearsal, use `--preview-diary --preview-name chat-preview-clean --diary-every 4`. This stores test exchanges separately under `paths.diary/chat-preview-clean/`, and after four incoming messages you can inspect that preview's `diary.md`, `diary_updates.md`, and `chat_transcript.md` without changing the main diary. Diary updates use only incoming messages as evidence; generated replies are not fed back as facts about the modeled person.

For the configured participant pair, the pipeline creates these files under `paths.diary/<user>_<other>/`:

- `diary.md`: accepted first-person diary passages. This is the file to open to read the modeled person's accumulated diary.
- `diary_updates.md`: an audit of each update attempt, including the exact prompt and message batch supplied to the model, its append/skip decision, a brief evidence-based note, and the raw structured response.
- `chat_transcript.md`: local chat inputs and model replies, useful for preview sessions.
- Each chat transcript turn also includes the exact message list sent to the inference model, including the system prompt, loaded recent history, and preview dialogue. The files are UTF-8; in Windows PowerShell 5, read them with `Get-Content -Encoding utf8 <path>` to avoid displaying valid UTF-8 as garbled text.
- `diary.json`: persistent state, including entries, pending messages, and fingerprints used to skip already-seen messages.

The update note is a short explanation, not hidden reasoning. These diary and audit files contain personal conversation context, so keep the diary directory private. Console logs show pipeline stages and message counts but do not print message text or API keys.
