from unittest.mock import AsyncMock, patch

import pytest

from src.dialectic.core import DialecticAgent
from src.llm import HonchoLLMCallResponse


@pytest.mark.asyncio
async def test_answer_prefetches_with_semantic_query_but_prompts_with_query() -> None:
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name=None,
        observer="observer",
        observed="observed",
        reasoning_level="minimal",
    )

    mock_response = HonchoLLMCallResponse(
        content="answer",
        input_tokens=1,
        output_tokens=1,
        finish_reasons=["stop"],
    )

    with (
        patch.object(DialecticAgent, "_initialize_session_history", new=AsyncMock()),
        patch.object(
            DialecticAgent,
            "_prefetch_relevant_observations",
            new=AsyncMock(return_value="semantic hit"),
        ) as mock_prefetch,
        patch("src.dialectic.core.create_tool_executor", new=AsyncMock(return_value=AsyncMock())),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ),
    ):
        result = await agent.answer(
            "expanded reasoning prompt", semantic_query="raw retrieval query"
        )

    assert result == "answer"
    mock_prefetch.assert_awaited_once_with("raw retrieval query")
    assert agent.messages[-1] == {
        "role": "user",
        "content": (
            "Query: expanded reasoning prompt\n\n"
            "## Relevant Observations (prefetched)\n"
            "The following observations were found to be semantically relevant to your query. "
            "Use these as primary context. You may still use tools to find additional information if needed.\n\n"
            "semantic hit"
        ),
    }


@pytest.mark.asyncio
async def test_answer_uses_query_for_prefetch_when_semantic_query_omitted() -> None:
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name=None,
        observer="observer",
        observed="observed",
        reasoning_level="minimal",
    )

    mock_response = HonchoLLMCallResponse(
        content="answer",
        input_tokens=1,
        output_tokens=1,
        finish_reasons=["stop"],
    )

    with (
        patch.object(DialecticAgent, "_initialize_session_history", new=AsyncMock()),
        patch.object(
            DialecticAgent,
            "_prefetch_relevant_observations",
            new=AsyncMock(return_value=None),
        ) as mock_prefetch,
        patch("src.dialectic.core.create_tool_executor", new=AsyncMock(return_value=AsyncMock())),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ),
    ):
        result = await agent.answer("legacy query")

    assert result == "answer"
    mock_prefetch.assert_awaited_once_with("legacy query")
    assert agent.messages[-1] == {"role": "user", "content": "Query: legacy query"}
