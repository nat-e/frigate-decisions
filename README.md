# Frigate Decisions

Ask your [Frigate](https://frigate.video) cameras yes/no questions in plain language, and turn the
answers into Frigate alerts and detections, with their recordings.

> [!WARNING]
> Frigate Decisions is in early development. Expect bugs and breaking changes between releases until 1.0.

```yaml
cameras:
  living_room:
    rules:
      dog_on_sofa:
        question: Is a dog lying or sitting on the sofa (not on the floor or the rug)?
        min_duration: 10
```

The dog has been on the sofa for 10 seconds: a `dog_on_sofa` alert shows up in Frigate's Review
page, with the clip from the moment it jumped on, until it gets off.

No model to train, no detector needed. Frigate's object detector only knows a fixed list of
objects, and only from usual angles. A decision model reads your question and looks at the
picture, so it can answer things a detector can't: *is the dog **on the sofa**?*, *is the garage
door **open**?*, *is there a parcel **on the doorstep**?*, *is the cat **inside** the litter box,
seen from above?*

## How it works

1. Every second, it grabs the camera's latest frame from Frigate and checks for movement (a cheap
   local pixel comparison).
2. While something moves, it asks a decision model all the camera's questions every few seconds.
   A decision model answers with a probability per question, e.g. `dog_on_sofa=0.93`.
3. When an answer stays "yes" (above `threshold`) for `min_duration` seconds, it creates a Frigate
   event with the rule's label, starting a few seconds before the first "yes". The event ends once
   the answer has been "no" for `end_after` seconds.

When nothing moves, it asks nothing, so it costs nothing.

## Decision models

| Provider | `provider.name` | API key |
| :- | :- | :- |
| [TypeSafe Jev](https://docs.typesafe.ai) | `jev` | `TYPESAFE_API_KEY` |

Jev costs $0.042 per million input tokens; one camera frame with a few questions is a couple of
thousand tokens. Image input is still experimental at TypeSafe.

Other providers (Cloudflare Clef-flash, OpenAI Decisions…) fit the same interface: a class with an
`ask(image, context, questions)` method in [`providers.py`](src/frigate_decisions/providers.py).

## Setup

Run it next to Frigate, on the same Docker network: it uses Frigate's internal API (port 5000),
which needs no login.

```yaml
# docker-compose.yml, next to your frigate service
services:
  frigate-decisions:
    image: ghcr.io/nat-e/frigate-decisions:latest
    restart: unless-stopped
    depends_on: [frigate]
    environment:
      TYPESAFE_API_KEY: ${TYPESAFE_API_KEY}
      TZ: Europe/Paris
    volumes:
      - ./frigate-decisions.yml:/config/config.yml:ro
```

Write `frigate-decisions.yml`, starting from [`examples/config.yml`](examples/config.yml).

To upgrade, run `docker compose pull frigate-decisions && docker compose up -d frigate-decisions`.

### Alert or detection?

Frigate decides, from your Frigate config. An event is an **alert** unless its label is listed
under `review.detections.labels` for that camera:

```yaml
# Frigate config
cameras:
  front_door:
    review:
      detections:
        labels: [person, visitor] # visitor events are detections, parcel_delivered ones alerts
```

## Configuration

```yaml
frigate:
  url: http://frigate:5000 # default

provider:
  name: jev # default
  model: jev-1.14.0 # optional, the provider's default model otherwise

cameras:
  <frigate camera name>:
    context: <optional description of what the camera shows>
    ask_every: 2 # seconds between questions while something moves
    idle_after: 20 # seconds without movement before questions stop
    motion_threshold: 0.003 # share of pixels that must change to count as movement
    rules:
      <rule name>:
        question: <yes/no question about the image> # required
        threshold: 0.6 # "yes" when the probability is at least this
        min_duration: 0 # seconds of uninterrupted "yes" before the event starts
        end_after: 10 # seconds without a "yes" before the event ends
        label: <Frigate label> # the rule name by default
        sub_label: <optional Frigate sub label>
```

## OpenTelemetry

Telemetry is opt-in. The app uses the OpenTelemetry API; the official Python launcher
configures the SDK and exporters. The Docker image includes the launcher. For a Python
installation, install `frigate-decisions[otel]` or run `uv sync --extra otel` from this repo.

```sh
export OTEL_SERVICE_NAME=frigate-decisions
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
opentelemetry-instrument frigate-decisions --config config.yml
```

With Docker Compose, add this to the service's configuration:

```yaml
command: [opentelemetry-instrument, frigate-decisions, --config, /config/config.yml]
environment:
  TYPESAFE_API_KEY: ${TYPESAFE_API_KEY}
  OTEL_SERVICE_NAME: frigate-decisions
  OTEL_EXPORTER_OTLP_ENDPOINT: http://collector:4318
  OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf
```

The launcher exports traces, metrics and Python logs. It also instruments Frigate HTTP
requests and collects process/system metrics. Configure endpoints, authentication, TLS,
resource attributes, sampling and export intervals with the standard
[`OTEL_*` settings](https://opentelemetry.io/docs/zero-code/python/configuration/).
Set `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER` or `OTEL_LOGS_EXPORTER` to `none` to
disable a signal; `OTEL_SDK_DISABLED=true` disables all telemetry. Running the app without
the launcher leaves its API instrumentation inactive unless another OTel agent configures it.

### GenAI telemetry

Model calls follow the [OTel GenAI inference conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md),
which are in development. TypeSafe calls use `gen_ai.provider.name=typesafe` and
`gen_ai.operation.name=system_one`, since this API answers structured questions rather than chat messages.
Each span includes requested/actual models, server address, reported token counts and
`error.type` on failure. A call includes the SDK's retries; missing token counts stay absent.

| Metric | Meaning |
| :- | :- |
| `gen_ai.client.inference.duration` | Call duration in seconds; its count gives the number of calls, split by `error.type` for failures |
| `gen_ai.client.inference.usage.input_tokens` / `output_tokens` | Total token consumption, with modality `unknown` because TypeSafe reports combined counts |
| `gen_ai.client.inference.operation.input_tokens` / `output_tokens` | Token distribution per call |

Inputs and outputs are excluded by default. Set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`
to `SPAN_ONLY`, `EVENT_ONLY` or `SPAN_AND_EVENT` to capture question text, camera context and
answer probabilities in standard GenAI message attributes. `NO_CONTENT` is the default.
Events use the logs pipeline. Camera images are excluded in every mode.

### Langfuse and other backends

Any compatible OTLP backend can receive the telemetry. For Langfuse, use `http/protobuf`,
set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to your Langfuse URL followed by
`/api/public/otel/v1/traces`, and configure `OTEL_EXPORTER_OTLP_TRACES_HEADERS` with its
authentication headers. See [Langfuse's OTel setup](https://langfuse.com/integrations/native/opentelemetry)
for authentication and ingestion-version headers.

Langfuse maps the standard model and token usage span attributes. Send metrics and logs to
a collector or another backend using their signal-specific endpoints, or disable those
exporters. To send traces to several backends, route them through an OpenTelemetry Collector.

## Writing good questions

- Ask about one thing you can see in a single frame: *Is a dog on the sofa?*, not *Did the dog
  sleep well?*
- Say what doesn't count. *Is a cat inside the litter box (the cat itself, not droppings)?*
  stopped false visits caused by what a cat left behind.
- Use `context` for what's always true about the scene, so each question can stay short.
- Watch the logs: every answer is printed with its probability. Raise `threshold` or
  `min_duration` if you get false events, lower them if you miss some.

## Limits

- Frigate's API can't create events in the past: an event appears in Frigate once
  `min_duration` has passed, with its recording starting a few seconds before the first "yes".
- Events created through the API aren't published on Frigate's `frigate/events` MQTT topic.
- Every camera is watched in full frames, at the resolution of Frigate's `detect` stream.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
