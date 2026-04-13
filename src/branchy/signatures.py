from __future__ import annotations

import dspy


class SingleTraceRelevance(dspy.Signature):
    """Evaluate how relevant a conversation trace is for a new user message."""

    user_message = dspy.InputField()
    conversation_trace = dspy.InputField()

    relevance_score = dspy.OutputField(desc="Float between 0.0 and 1.0")
    reasoning = dspy.OutputField(desc="Brief explanation")


class ConnectionDecision(dspy.Signature):
    """Decide where to attach the user's message in the conversation tree."""

    user_message = dspy.InputField()
    best_trace_score = dspy.InputField(
        desc="Relevance score of best matching trace (0-1)")
    best_trace_text = dspy.InputField(
        desc="The conversation trace with highest score")

    action = dspy.OutputField(
        desc="One of: 'ATTACH', 'NEW_BRANCH', 'NEW_ROOT'")
    explanation = dspy.OutputField(desc="Why you chose this action")


class ResponseGenerator(dspy.Signature):
    """Generate a helpful response using the selected conversation context."""

    context = dspy.InputField(desc="Previous conversation history")
    user_message = dspy.InputField()

    response = dspy.OutputField(desc="Helpful, natural response")
