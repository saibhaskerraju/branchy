from __future__ import annotations

from typing import Any, Optional

import dspy

from .signatures import ConnectionDecision, ResponseGenerator, SingleTraceRelevance
from .tree import ConversationTree


class BranchingChat:
    """Branch-aware chat manager that routes messages into a conversation tree."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **lm_kwargs: Any,
    ) -> None:
        """Create a chat instance.

        Initializes a DSPy LM from LiteLLM-compatible parameters.
        """

        llm = dspy.LM(
            model=model,
            api_key=api_key,
            api_base=api_base,
            **lm_kwargs,
        )

        dspy.settings.configure(lm=llm)
        self.tree = ConversationTree()
        self.llm = llm

        self.evaluator = dspy.ChainOfThought(SingleTraceRelevance)
        self.decider = dspy.ChainOfThought(ConnectionDecision)
        self.responder = dspy.ChainOfThought(ResponseGenerator)

    def add_message(self, user_message: str) -> str:
        """Add a user message, route it in the tree, and return the new message id."""

        best_match = self._find_best_trace(user_message)
        decision = self._decide_action(user_message, best_match)
        response = self._generate_response(user_message, decision)

        message_id = self.tree.add_message(
            user=user_message,
            assistant=response,
            parent_message_id=decision["target_message_id"],
        )
        return message_id

    def show_tree(self, print_output: bool = True) -> str:
        """Render the current user-message tree and optionally print it."""

        tree_output = self.tree.render_tree()
        if print_output:
            print(tree_output)
        return tree_output

    def _find_best_trace(self, user_message: str) -> Optional[dict[str, Any]]:
        all_traces = self.tree.get_all_traces(max_window=3)
        if not all_traces:
            return None

        best_score = -1.0
        best_trace: Optional[dict[str, Any]] = None
        best_reasoning = ""

        for trace in all_traces:
            result = self.evaluator(
                user_message=user_message,
                conversation_trace=trace["formatted_text"],
            )

            try:
                score = float(result.relevance_score)
                score = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                score = 0.0

            if score > best_score:
                best_score = score
                best_trace = trace
                best_reasoning = result.reasoning
            elif score == best_score and best_trace:
                if trace["window_size"] > best_trace["window_size"]:
                    best_trace = trace
                    best_reasoning = result.reasoning

        return {"trace": best_trace, "score": best_score, "reasoning": best_reasoning}

    def _decide_action(
        self,
        user_message: str,
        best_match: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if best_match is None:
            return {
                "action": "NEW_ROOT",
                "explanation": "No existing conversations, starting new root",
                "target_message_id": None,
            }

        decision_result = self.decider(
            user_message=user_message,
            best_trace_score=str(best_match["score"]),
            best_trace_text=best_match["trace"]["formatted_text"],
        )

        action = decision_result.action.strip().upper()
        if action == "ATTACH":
            return {
                "action": "ATTACH",
                "explanation": decision_result.explanation,
                "target_message_id": best_match["trace"]["end_message_id"],
            }
        if action == "NEW_BRANCH":
            root_of_trace = best_match["trace"]["messages"][0]
            return {
                "action": "NEW_BRANCH",
                "explanation": decision_result.explanation,
                "target_message_id": root_of_trace.message_id,
            }

        return {
            "action": "NEW_ROOT",
            "explanation": decision_result.explanation,
            "target_message_id": None,
        }

    def _generate_response(self, user_message: str, decision: dict[str, Any]) -> str:
        if decision["action"] in ("ATTACH", "NEW_BRANCH") and decision["target_message_id"]:
            path = self.tree.get_path_to_root(decision["target_message_id"])
            context_lines = []
            for message in path:
                context_lines.append(f"User: {message.user}")
                context_lines.append(f"Assistant: {message.assistant}")
            context = "\n".join(context_lines)
        else:
            context = "Starting a completely new conversation"

        result = self.responder(context=context, user_message=user_message)
        return result.response
