"""Decision models: they answer yes/no questions about an image with a probability.

To add a provider, write a class with the Provider attributes and `ask` method, and add it to PROVIDERS.
"""

import base64
import os
from typing import Protocol

from .telemetry import inference


class Provider(Protocol):
    image_sides: tuple[int, int]  # smallest and largest side, in pixels, of the images `ask` gets
    max_questions: int | None  # most questions per request, None for no limit

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        """Answer each question about the JPEG `image`: {question name: probability of "yes"}.

        A question the model won't answer is left out.
        """
        ...


class Jev:
    """TypeSafe Jev (https://docs.typesafe.ai). Reads the API key from TYPESAFE_API_KEY."""

    image_sides: tuple[int, int] = (280, 1568)  # the size range Jev accepts
    max_questions: int | None = None

    def __init__(self, model: str | None = None):
        from typesafe_sdk import TypeSafeClient  # imported here: only needed when Jev is used

        self.client: TypeSafeClient = TypeSafeClient(model=model)

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        from typesafe_sdk import Image, Noul

        state: dict[str, Image | str] = {"frame": Image(image)}
        if context:
            state["context"] = context
        # The SDK resolves the model and URL from its arguments and environment, and exposes them only here
        config = self.client._config  # pyright: ignore[reportPrivateUsage]
        with inference(
            "typesafe", "system_one", config.default_model, config.base_url, context=context, questions=questions
        ) as observed:
            # The SDK's types for JSON values are partly unknown
            result = self.client.system_one(  # pyright: ignore[reportUnknownMemberType]
                state=state,
                questions={name: Noul(instructions=text) for name, text in questions.items()},
                extra_headers={"TypeSafe-Danger-Zone-Images": "true"},  # image input is still experimental
            )
            observed.response(result.model, result.usage.input_tokens, result.usage.output_tokens)
            answers = {name: result.nouls[name].noul for name in questions}
            observed.output(answers)
            return answers


class OpenAIDecisions:
    """OpenAI Decisions. Reads the API key from OPENAI_API_KEY."""

    MODELS: tuple[str, ...] = ("gpt-6-luna",)  # the models the Decisions API supports
    # OpenAI documents no size limits; this keeps uploads and image tokens small.
    image_sides: tuple[int, int] = (280, 1568)
    max_questions: int | None = 400

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.model: str = model or self.MODELS[0]
        if self.model not in self.MODELS:
            raise ValueError(f"OpenAI Decisions doesn't support model {self.model!r}, use one of {list(self.MODELS)}")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("set OPENAI_API_KEY to use OpenAI Decisions")
        # Skip failed frames promptly; the watcher asks again with a fresh image.
        self.client: OpenAI = OpenAI(timeout=20.0, max_retries=0)

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        from openai.types.decision_input_part_union_param import DecisionInputPartUnionParam

        content: list[DecisionInputPartUnionParam] = []
        if context:
            content.append({"type": "input_text", "text": context})
        encoded = base64.b64encode(image).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"})
        with inference(
            "openai", "decisions", self.model, str(self.client.base_url), context=context, questions=questions
        ) as observed:
            result = self.client.decisions.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                questions=[
                    {"type": "predicate", "name": name, "instructions": text} for name, text in questions.items()
                ],
            )
            observed.response(
                result.model,
                result.usage.input_tokens,
                result.usage.output_tokens,
                cache_read_count=result.usage.input_tokens_details.cached_tokens,
                cache_write_count=result.usage.input_tokens_details.cache_write_tokens,
                reasoning_count=result.usage.output_tokens_details.reasoning_tokens,
            )
            answers: dict[str, float] = {}
            for name, answer in zip(questions, result.answers, strict=True):  # answers come in question order
                if answer.name not in (None, name):
                    raise ValueError(f"OpenAI answered {answer.name!r} for question {name!r}")
                if answer.type == "refusal":
                    continue
                if answer.type != "predicate" or not 0 <= answer.probability <= 1:
                    raise ValueError(f"OpenAI returned an invalid answer for question {name!r}")
                answers[name] = answer.probability
            observed.output(answers)
            return answers


PROVIDERS = {"jev": Jev, "openai": OpenAIDecisions}


def make_provider(name: str, model: str | None) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}, available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model)
