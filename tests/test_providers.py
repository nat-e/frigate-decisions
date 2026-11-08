import base64
import json
from collections.abc import Iterator
from functools import partial
from typing import cast

import httpx2
import pytest
from conftest import Telemetry
from openai import AuthenticationError, InternalServerError
from openai import OpenAI as SDKOpenAI
from opentelemetry.sdk.metrics.export import Histogram, Sum
from opentelemetry.trace import StatusCode

from frigate_decisions.providers import OpenAIDecisions, make_provider

QUESTIONS = {"dog_on_sofa": "Is a dog on the sofa?", "door_open": "Is the door open?"}


class FakeOpenAI:
    def __init__(self):
        self.requests: list[httpx2.Request] = []
        self.answers: list[dict[str, object]] = [
            {"type": "predicate", "name": "dog_on_sofa", "probability": 0.93},
            {"type": "predicate", "name": "door_open", "probability": 0.12},
        ]
        self.status: int = 200

    def respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(
                self.status, json={"error": {"message": "API error", "type": "invalid_request_error"}}
            )
        return httpx2.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": self.answers,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 100,
                },
            },
        )


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenAI]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake = FakeOpenAI()
    with httpx2.Client(transport=httpx2.MockTransport(fake.respond)) as http:
        monkeypatch.setattr(
            "openai.OpenAI", partial(SDKOpenAI, api_key="test-key", base_url="https://openai.test/v1", http_client=http)
        )
        yield fake


def test_openai_sends_image_context_and_named_questions(api: FakeOpenAI) -> None:
    provider = make_provider("openai", None)
    image = b"camera JPEG"
    assert provider.ask(image, "Camera above a grey sofa.", QUESTIONS) == {"dog_on_sofa": 0.93, "door_open": 0.12}

    (request,) = api.requests
    assert request.method == "POST"
    assert str(request.url) == "https://openai.test/v1/decisions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Camera above a grey sofa."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}",
                    },
                ],
            }
        ],
        "questions": [{"type": "predicate", "name": name, "instructions": text} for name, text in QUESTIONS.items()],
    }


@pytest.mark.parametrize("context", [None, ""])
def test_openai_without_context(api: FakeOpenAI, context: str | None) -> None:
    _ = make_provider("openai", "gpt-6-luna").ask(b"JPEG", context, QUESTIONS)
    body = cast("dict[str, object]", json.loads(api.requests[0].content))
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,SlBFRw=="}]}
    ]


def test_openai_unsupported_model_is_a_config_error(api: FakeOpenAI) -> None:
    with pytest.raises(ValueError, match=r"jev-1\.14\.0"):
        _ = make_provider("openai", "jev-1.14.0")


def test_openai_refused_question_is_left_out(api: FakeOpenAI) -> None:
    api.answers[1] = {"type": "refusal", "name": "door_open"}
    assert make_provider("openai", None).ask(b"JPEG", None, QUESTIONS) == {"dog_on_sofa": 0.93}


def test_openai_accepts_answers_without_names(api: FakeOpenAI) -> None:
    for answer in api.answers:
        answer["name"] = None
    assert make_provider("openai", None).ask(b"JPEG", None, QUESTIONS) == {"dog_on_sofa": 0.93, "door_open": 0.12}


@pytest.mark.parametrize("count", [1, 3])
def test_openai_rejects_missing_or_extra_answers(api: FakeOpenAI, count: int) -> None:
    api.answers = (api.answers * 2)[:count]
    with pytest.raises(ValueError):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_openai_rejects_invalid_probability(api: FakeOpenAI, probability: float) -> None:
    api.answers[1]["probability"] = probability
    with pytest.raises(ValueError, match=r"invalid answer.*door_open"):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)


def test_openai_rejects_mismatched_question_names(api: FakeOpenAI) -> None:
    api.answers.reverse()
    with pytest.raises(ValueError, match="answered 'door_open' for question 'dog_on_sofa'"):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)


@pytest.mark.parametrize("status, error", [(401, AuthenticationError), (503, InternalServerError)])
def test_openai_api_error_propagates_without_retry(api: FakeOpenAI, status: int, error: type[Exception]) -> None:
    api.status = status
    with pytest.raises(error):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)
    assert len(api.requests) == 1


def test_openai_missing_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "admin-key")  # the SDK accepts this alone, but decisions need an API key
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = OpenAIDecisions()


def test_openai_genai_span_and_usage(api: FakeOpenAI, telemetry: Telemetry) -> None:
    _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)
    (span,) = telemetry.spans.get_finished_spans()
    assert span.name == "decisions gpt-6-luna"
    assert span.attributes is not None
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-6-luna"
    assert span.attributes["gen_ai.response.model"] == "gpt-6-luna"
    assert span.attributes["server.address"] == "openai.test"
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.input.messages" not in span.attributes
    data = telemetry.collect()["gen_ai.client.inference.usage.input_tokens"].data
    assert isinstance(data, Sum)
    assert sum(point.value for point in data.data_points) == 100


def test_openai_incomplete_response_still_reports_usage(api: FakeOpenAI, telemetry: Telemetry) -> None:
    api.answers = []
    with pytest.raises(ValueError):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)
    (span,) = telemetry.spans.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["error.type"] == "ValueError"
    assert span.status.status_code == StatusCode.ERROR


def test_openai_failure_reports_genai_error(api: FakeOpenAI, telemetry: Telemetry) -> None:
    api.status = 503
    with pytest.raises(InternalServerError):
        _ = make_provider("openai", None).ask(b"JPEG", None, QUESTIONS)
    data = telemetry.collect()["gen_ai.client.inference.duration"].data
    assert isinstance(data, Histogram)
    point = next(point for point in data.data_points if point.count)
    assert point.attributes is not None
    assert point.attributes["gen_ai.provider.name"] == "openai"
    assert point.attributes["error.type"] == "InternalServerError"
