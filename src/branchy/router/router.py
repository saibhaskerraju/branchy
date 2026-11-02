from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..tree import ConversationTree
from .lexical import (
    DEFAULT_CUES,
    DEFAULT_REQUEST_MARKERS,
    bm25_scores,
    has_cue,
    is_fragment,
    normalize,
    tokenize,
)
from .sticker import build_sticker


@dataclass(frozen=True)
class RouterConfig:
    k: int = 3
    w_rec: float = 0.3
    recency_half_life: float = 3.0
    new_sim_max: float = 0.30
    firm_margin: float = 0.15
    fragment_cues: frozenset = DEFAULT_CUES
    request_markers: frozenset = DEFAULT_REQUEST_MARKERS


class CandidateScore(BaseModel):
    node_id: str
    root_id: str
    kind: Literal["leaf", "root"]
    bm25: float
    recency: float
    s1: float
    evidence: int = 0


class RoutingDecision(BaseModel):
    action: Literal["NEW", "ATTACH_FIRM", "ATTACH_TENTATIVE"]
    target_message_id: Optional[str]
    candidates: list[CandidateScore] = Field(default_factory=list)
    shortlisted_ids: list[str] = Field(default_factory=list)
    dominated_root_ids: list[str] = Field(default_factory=list)
    margin: Optional[float] = None
    reason: str


class ThreadRouter:
    """Deterministic thread router: BM25 + recency."""

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        self.config = config

    def route(self, tree: ConversationTree, message: str) -> RoutingDecision:
        cfg = self.config

        if not tree.messages:
            return RoutingDecision(
                action="NEW",
                target_message_id=None,
                reason="empty tree; starting a new thread",
            )

        # candidates: leaf assistants union root assistants, deduped; leaf wins.
        candidates: dict[str, str] = {}  # node_id -> kind
        for node in tree.root_assistants():
            candidates[node.message_id] = "root"
        for node in tree.leaf_assistants():
            candidates[node.message_id] = "leaf"
        node_ids = sorted(
            candidates, key=lambda nid: tree.messages[nid].created_seq
        )

        docs_tokens = []
        scored: list[CandidateScore] = []
        for node_id in node_ids:
            node = tree.messages[node_id]
            root_id = tree.root_of(node_id)
            sticker = tree.stickers.get(root_id)
            sticker_text = sticker.text if sticker else build_sticker(tree, root_id)
            user, assistant = tree.last_turn(node)
            doc = f"{sticker_text} {user} {assistant}"
            docs_tokens.append(tokenize(doc))

            age = tree.seq - (sticker.last_used_seq if sticker else 0)
            recency = 0.5 ** (age / cfg.recency_half_life)
            scored.append(
                CandidateScore(
                    node_id=node_id,
                    root_id=root_id,
                    kind=candidates[node_id],
                    bm25=0.0,
                    recency=recency,
                    s1=0.0,
                )
            )

        message_terms = set(tokenize(message))
        raw_bm25 = normalize(bm25_scores(tokenize(message), docs_tokens))
        sims: dict[str, float] = {}
        matched: dict[str, set] = {}
        for cand, bm, doc in zip(scored, raw_bm25, docs_tokens):
            cand.bm25 = bm
            sims[cand.node_id] = bm
            cand.s1 = bm + cfg.w_rec * cand.recency
            matched[cand.node_id] = message_terms & set(doc)
            cand.evidence = len(matched[cand.node_id])

        # A fork at the root is only justified when the message hits content
        # specific to the root that the later turns do not have; if a leaf in
        # the same tower already covers every matched term, continue the
        # thread instead of forking. Empty ⊆ empty, so fragments never fork.
        leaf_sets: dict[str, list[set]] = {}
        for c in scored:
            if c.kind == "leaf":
                leaf_sets.setdefault(c.root_id, []).append(matched[c.node_id])
        dominated_root_ids = [
            c.node_id
            for c in scored
            if c.kind == "root"
            and any(matched[c.node_id] <= s for s in leaf_sets.get(c.root_id, []))
        ]
        dominated = set(dominated_root_ids)

        sticker_seq = {
            nid: (
                tree.stickers[tree.root_of(nid)].last_used_seq
                if tree.root_of(nid) in tree.stickers
                else 0
            )
            for nid in node_ids
        }
        created_seq = {nid: tree.messages[nid].created_seq for nid in node_ids}

        def order(c: CandidateScore, key: float) -> tuple:
            kind_rank = 0 if c.kind == "leaf" else 1
            return (-key, -sticker_seq[c.node_id], kind_rank, -created_seq[c.node_id])

        ranked = sorted(scored, key=lambda c: order(c, c.s1))
        eligible = [c for c in ranked if c.node_id not in dominated]
        shortlist = eligible[: cfg.k]

        # the active tower's leaf assistant (the fragment fallback target)
        active_leaf_id: Optional[str] = None
        if tree.active_root_id:
            active_leaves = [
                n
                for n in tree.leaf_assistants()
                if tree.root_of(n.message_id) == tree.active_root_id
            ]
            if active_leaves:
                active_leaf_id = active_leaves[-1].message_id

        best = shortlist[0]
        margin = (
            best.s1 - shortlist[1].s1 if len(shortlist) > 1 else None
        )

        best_sim = sims[best.node_id]
        if best_sim < cfg.new_sim_max:
            if is_fragment(message, cfg.fragment_cues, cfg.request_markers):
                return self._route_fragment(
                    shortlist, ranked, eligible, active_leaf_id,
                    margin, dominated_root_ids,
                )
            return RoutingDecision(
                action="NEW",
                target_message_id=None,
                candidates=ranked,
                shortlisted_ids=[c.node_id for c in shortlist],
                dominated_root_ids=dominated_root_ids,
                margin=margin,
                reason=f"new thread: best sim={best_sim:.3f} < {cfg.new_sim_max}",
            )

        if margin is None or margin >= cfg.firm_margin:
            action = "ATTACH_FIRM"
        else:
            action = "ATTACH_TENTATIVE"
        downgrade_note = ""
        if (
            action == "ATTACH_FIRM"
            and best.evidence <= 1
            and not has_cue(message, cfg.fragment_cues)
        ):
            # One shared word ("recipe") is not enough evidence to firmly
            # attach; a cue word ("it", "that") means the user is referring
            # back, so firm is allowed.
            action = "ATTACH_TENTATIVE"
            downgrade_note = "; single-term match -> tentative"

        return RoutingDecision(
            action=action,
            target_message_id=best.node_id,
            candidates=ranked,
            shortlisted_ids=[c.node_id for c in shortlist],
            dominated_root_ids=dominated_root_ids,
            margin=margin,
            reason=(
                f"{action.lower().replace('_', ' ')}: best s1={best.s1:.3f}, "
                f"margin={'n/a' if margin is None else f'{margin:.3f}'}"
                f"{downgrade_note}"
            ),
        )

    def _route_fragment(
        self,
        shortlist: list[CandidateScore],
        ranked: list[CandidateScore],
        eligible: list[CandidateScore],
        active_leaf_id: Optional[str],
        margin: Optional[float],
        dominated_root_ids: list[str],
    ) -> RoutingDecision:
        """Referential fragment: never start a thread — attach tentatively."""
        by_id = {c.node_id: c for c in eligible}

        # (b) the active tower's leaf assistant, even if not shortlisted
        if active_leaf_id and active_leaf_id in by_id:
            cand = by_id[active_leaf_id]
            return RoutingDecision(
                action="ATTACH_TENTATIVE",
                target_message_id=cand.node_id,
                candidates=ranked,
                shortlisted_ids=[c.node_id for c in shortlist],
                dominated_root_ids=dominated_root_ids,
                margin=margin,
                reason=(
                    f"fragment: fallback to active thread (s1={cand.s1:.3f}); "
                    f"tentative attach"
                ),
            )

        # (c) best by s1
        best = shortlist[0]
        return RoutingDecision(
            action="ATTACH_TENTATIVE",
            target_message_id=best.node_id,
            candidates=ranked,
            shortlisted_ids=[c.node_id for c in shortlist],
            dominated_root_ids=dominated_root_ids,
            margin=margin,
            reason=(
                f"fragment: fallback to best s1={best.s1:.3f}; "
                f"tentative attach"
            ),
        )
