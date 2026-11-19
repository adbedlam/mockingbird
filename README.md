# Mockingbird

Mockingbird is a research project that uses a Telegram conversation export to model how a person writes and when they tend to reply or start a conversation. It builds a persona profile from the archive, keeps a diary as new messages arrive, and uses a scheduler to time replies and new conversations. You can run it in a local terminal or connect it to one private chat through Telegram Business.

## Setup

Run these commands from the repository root. The project uses Python 3.13 and uv.

```powershell
uv sync --locked
if (!(Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

1. In `config.yaml`, set `user` to the person you want to model and `other` to their conversation partner. Set `preprocessing.user_id` and `preprocessing.other_id` if you know their Telegram IDs. This lets the importer recognize them after a name change.
2. Place the Telegram JSON export at `data/raw/result.json`. Set `paths.raw` to use a different input directory.
3. Set the endpoint and model name in `building_model` and `inference_model`. For remote providers, set `BUILDING_MODEL_API_KEY` and `INFERENCE_MODEL_API_KEY` in `.env`. Image descriptions are optional. Enable them with `vision_model.enabled` and set `VISION_MODEL_API_KEY` if the provider requires it.

Local configs, keys, chat exports, generated artifacts, and runtime state are kept out of Git. For a second persona, create a separate `config.<name>.yaml` with distinct data paths and pass `--config` to commands that support it. If you configure a remote model, the application sends the selected history and memory excerpts to that provider.

## Build a persona and start a local chat

```powershell
uv run python -m mockingbird.mockingbird.build --history
uv run python scripts/refresh_persona.py --local
uv run mockingbird --fast-replies
```

The first command imports the export and saves the conversation history. It runs locally when image descriptions are disabled. The second initializes a new persona with examples copied from the archive and an empty diary, without calling a model. The third starts the chat and calls the configured inference model for replies. Omit `--fast-replies` to use the scheduler's reply delays.

Run `uv run python -m mockingbird.mockingbird.build` to build the complete persona with the configured models, or select stages with `--history`, `--inner-prompt`, and `--diary`. The default output directory is `data/processed/`. Use `uv run python scripts/refresh_persona.py --prepare` to save the refresh requests locally for inspection without sending them. Use `uv run mockingbird --refresh-persona` to rebuild the profile and diary before opening the chat. Set `persona_build.auto_refresh_on_start` to enable automatic refresh when the source changes.

In the local terminal, use `/status`, `/context`, `/memory`, `/diary`, `/life`, and `/quit` to inspect or stop the session.

## Main modules

The build stage turns an export into conversation history, a persona prompt, and a diary. At runtime, the scheduler decides when to act, the agent prepares a draft, and the pipeline coordinates generation and delivery through the selected interface.

### Data and persona

- [Configuration](src/mockingbird/config.py) loads YAML settings and environment variables, resolves data paths, and saves a discovered Telegram Business connection ID to the local `.env`.
- [Preprocessing](src/mockingbird/preprocessing/) imports the Telegram export, selects messages from the configured pair, and can describe images through the configured vision model. The Telegram source adapter in [preprocessing/telegram/source.py](src/mockingbird/preprocessing/telegram/source.py) selects the latest raw export or cached history.
- [Conversation](src/mockingbird/conversation/) defines shared message and history types. [history.py](src/mockingbird/conversation/history.py) handles persistence, chronological selection, and source fingerprints; [identity.py](src/mockingbird/conversation/identity.py) provides stable conversation-pair IDs.
- [Artifact builder](src/mockingbird/mockingbird/build.py) runs the export import, profile generation, and diary generation as separate stages or as a full build.
- [Persona profile](src/mockingbird/persona/inner/) builds the prompt that describes how the person writes and relates to the other participant. [Persona refresh](src/mockingbird/persona/refresh.py) prepares, initializes, and updates the stored profile and diary; [persona/source.py](src/mockingbird/persona/source.py) applies input limits and tracks artifact provenance.
- [Diary and memory](src/mockingbird/persona/diary/) stores entries tied to source messages, incorporates new conversations, and retrieves relevant entries for the next model request.
- [Daily life](src/mockingbird/persona/life.py) manages the persona's simulated activities. It uses the daily plan to inform availability and can update future activities based on agreements made in the conversation.

### Runtime and interfaces

- [Scheduler](src/mockingbird/scheduler/) estimates active hours, response delays, and initiation patterns from history. Its runtime uses those patterns and current state to decide whether to wait, reply, or start a conversation.
- [Agent](src/mockingbird/mockingbird/mockingbird.py) combines the profile, recent conversation, selected memories, and current activity into model context. It parses generated drafts and updates conversation state when delivery is recorded.
- [Prompts](src/mockingbird/prompts/) holds shared instructions for how the agent should express itself in the conversation.
- [Model client](src/mockingbird/llm/) defines the structure of model messages and sends completion requests to an OpenAI-compatible endpoint.
- [Chat pipeline](src/mockingbird/chat/) connects the scheduler, agent, and delivery interface. It tracks pending work, coordinates model calls, and publishes events for the terminal and runtime logs.
- [Interfaces](src/mockingbird/interfaces/) adapt the pipeline to local terminal and Telegram Business sessions. The [terminal interface](src/mockingbird/interfaces/terminal.py) handles local commands and displays pipeline events. [Telegram Business](src/mockingbird/interfaces/telegram/) receives updates, checks who can receive messages, and delivers replies through the Bot API; its `business.py`, `cli.py`, and `state.py` modules contain transport, operator commands, and durable state. `mockingbird-telegram` is an alias for `mockingbird-business`.
- [Observability](src/mockingbird/observability.py) writes per-run logs, redacts secrets in formatted output, and packages diagnostics for export. These local files can contain conversation data.
- [Dashboard](src/mockingbird/dashboard/) serves a read-only localhost view of connection status, scheduling decisions, delivery history, diary entries, and model requests using its bundled [HTML interface](src/mockingbird/dashboard/dashboard.html).

## Telegram Business

Connect the bot in Telegram Business and grant it access to the intended private chat and permission to reply. Put the bot token in `TELEGRAM_BOT_TOKEN` in `.env`. In `config.yaml`, set `telegram_business.bot_username`, the account owner's `preprocessing.user_id`, and the partner's `preprocessing.other_id`. Set `telegram_business.allowed_chat_ids` to a list containing that same partner ID. Start with `PERSONA_SEND_ENABLED=false`.

```powershell
uv run mockingbird-business observe --seconds 60
uv run mockingbird-business status --live
```

Ask the partner to message the account owner while `observe` is running. This command reads updates without calling a model or sending messages. When it discovers the connection, it saves `TELEGRAM_BUSINESS_CONNECTION_ID` to `.env`. If the ID is unknown and no event arrives, reconnect the bot while observation is running. Use one polling process per bot token. Before sending a test, check the connection owner, reply permission, and verified recipient in the status output.

Prepare a test with the actual chat ID. Once the account owner has approved the recipient and exact text, set `PERSONA_SEND_ENABLED=true` and send the approved test. Check how the sender appears in Telegram before confirming it and starting the agent.

```powershell
$recipientId = Read-Host "Recipient chat ID"
uv run mockingbird-business prepare-test --recipient $recipientId --text "Mockingbird test"
$approvalId = Read-Host "Approval ID"
uv run mockingbird-business send-test --approval-id $approvalId --confirm-text "Mockingbird test"
$messageId = Read-Host "Delivered message ID"
uv run mockingbird-business confirm-appearance --message-id $messageId
uv run mockingbird-business run --fast-replies
```

Enter `/pause`, `/resume`, `/status`, `/delivery`, and `/quit` in the local operator terminal. An outgoing message written by the account owner pauses the agent by default. Before delivery, the transport checks reply permission and whether the chat received a message within the last 24 hours. If a delivery outcome is unknown, check the chat before recording the result with `mockingbird-business resolve-delivery`. Use `mockingbird-business export` to save the journal and diagnostics.

## Dashboard and previews

```powershell
uv run mockingbird-dashboard --open
uv run mockingbird-dashboard --status
uv run python scripts/preview_scheduler.py --days 7 --seed 7
uv run python scripts/preview_memory_search.py
```

On Windows, you can also launch the dashboard with `start_dashboard.cmd`. The two preview commands above run without model calls. [Scripts](scripts/) also contains tools for persona refresh, diary updates, daily plans, and plan changes. Each script lists its options with `--help`. Preview output goes to `data/previews/` by default.

## Development

The [test suite](tests/) covers import, persona and memory, scheduling, the runtime pipeline, Telegram delivery and recovery, and the dashboard.

```powershell
uv run pytest
uv run mypy
uv run isort --check-only src scripts tests
uv run black --check src scripts tests
```
