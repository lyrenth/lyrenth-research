"""A research agent that reads its sources.

    from lyrenth_research import gather, answer

    gathered = gather(["https://example.com/a", "https://example.com/b"])
    result = answer("What does the page say about X?", gathered.sources)
    print(result.text)

Reading goes through Lyrenth (LYRENTH_API_KEY, free key at
https://lyrenth.com/signup). The answer comes from any OpenAI-compatible
model endpoint you configure with LLM_BASE_URL and LLM_MODEL.
"""

__version__ = "0.1.0"

from .answer import Answer, ModelError, answer, build_context, check_citations  # noqa: E402
from .sources import Gathered, Skipped, Source, gather  # noqa: E402

__all__ = [
    "Answer",
    "Gathered",
    "ModelError",
    "Skipped",
    "Source",
    "answer",
    "build_context",
    "check_citations",
    "gather",
    "__version__",
]
