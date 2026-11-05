"""Decision models: they answer yes/no questions about an image with a probability.

To add a provider, write a class with an `ask` method and add it to PROVIDERS.
"""

import io
from typing import Protocol

from PIL import Image as PILImage


class Provider(Protocol):
    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        """Answer each question about the JPEG `image`: {question name: probability of "yes"}."""
        ...


class Jev:
    """TypeSafe Jev (https://docs.typesafe.ai). Reads the API key from TYPESAFE_API_KEY."""

    # Image size Jev accepts, in pixels per side
    MIN_SIDE: int = 280
    MAX_SIDE: int = 1568

    def __init__(self, model: str | None = None):
        from typesafe_sdk import TypeSafeClient  # imported here: only needed when Jev is used

        self.client: TypeSafeClient = TypeSafeClient(model=model) if model else TypeSafeClient()

    def ask(self, image: bytes, context: str | None, questions: dict[str, str]) -> dict[str, float]:
        from typesafe_sdk import Image, Noul

        state: dict[str, Image | str] = {"frame": Image(self._fit(image))}
        if context:
            state["context"] = context
        # The SDK's types for JSON values are partly unknown
        result = self.client.system_one(  # pyright: ignore[reportUnknownMemberType]
            state=state,
            questions={
                name: Noul(instructions=f"Looking at the camera image `frame`: {text}")
                for name, text in questions.items()
            },
            extra_headers={"TypeSafe-Danger-Zone-Images": "true"},  # image input is still experimental
        )
        return {name: result.nouls[name].noul for name in questions}

    def _fit(self, jpeg: bytes) -> bytes:
        """Scale the image into the size range Jev accepts."""
        img = PILImage.open(io.BytesIO(jpeg))
        scale = min(1.0, self.MAX_SIDE / max(img.size))
        scale = max(scale, self.MIN_SIDE / min(img.size))
        if scale == 1.0:
            return jpeg
        out = io.BytesIO()
        size = (round(img.width * scale), round(img.height * scale))
        # Pillow's type for `size` includes numpy arrays, unknown without numpy installed
        resized = img.convert("RGB").resize(size)  # pyright: ignore[reportUnknownMemberType]
        resized.save(out, "JPEG", quality=90)
        return out.getvalue()


PROVIDERS = {"jev": Jev}


def make_provider(name: str, model: str | None) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}, available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model)
