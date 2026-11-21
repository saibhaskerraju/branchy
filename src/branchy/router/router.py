"""Deterministic message router: where does a new user message go?

The answer is one parent for the message — an existing assistant message, or
none (a new topic). Specified in decisions/0002-routing-algorithm.md; the
step numbers in comments refer to that document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..tree import ConversationTree
from .embed import RouterData, Vec, cosine, load_router_data
from .text import (
    ANAPHORA_WORDS,
    FOLLOW_UP_PHRASES,
    NEW_TOPIC_PHRASES,
    REQUEST_MARKERS,
    RETURN_PHRASES,
    Cues,
    asks_question,
    detect_cues,
    term_set,
    topic_text,
    topic_words,
)
from .porter import stem

Action = Literal["NEW", "CONTINUE", "SWITCH", "FORK"]
Confidence = Literal["FIRM", "TENTATIVE"]
# why a decision is TENTATIVE: the uncertain rel band, a reference cue with
# weak content evidence, or a deciding check that passed by < m_slack
Uncertainty = Literal["band", "reference", "margin"]
_ROUND = 6


@dataclass(frozen=True)
class RouterConfig:
    """Every threshold the router uses. Scores are on an absolute 0..1 scale."""

    s_full: float = 0.25          # cosine lift above background that counts as sem = 1
    bg_percentile: float = 0.95   # background similarity = this percentile
    tau_new: float = 0.25         # best rel below this -> NEW
    tau_attach: float = 0.35      # best rel at/above this -> attach
    tau_override: float = 0.60    # NEW_TOPIC cue loses only to a match this strong
    m_switch: float = 0.15        # lead needed to move to another branch
    delta0: float = 0.10          # fork evidence bar for dropping 1 turn ...
    delta_step: float = 0.03      # ... plus this per extra dropped turn ...
    delta_max: float = 0.30       # ... capped here
    w_act: float = 0.15           # weight of the thread activity prior (ranking only)
    activity_decay: float = 0.8408964152537145  # 0.5 ** (1/4): half-life 4 messages
    m_slack: float = 0.05         # every deciding check needs this slack for FIRM
    idf_rare: int = 7000          # idf_milli of a "rare" term: in < ~1/1100 background prompts
    fragment_max_terms: int = 3
    anaphora_max_terms: int = 6
    return_phrases: tuple[str, ...] = RETURN_PHRASES
    new_topic_phrases: tuple[str, ...] = NEW_TOPIC_PHRASES
    follow_up_phrases: tuple[str, ...] = FOLLOW_UP_PHRASES
    anaphora_words: frozenset = ANAPHORA_WORDS
    request_markers: frozenset = REQUEST_MARKERS
    max_candidates_reported: int = 12
    data_dir: Optional[str] = None  # None -> bundled data


class CandidateScore(BaseModel):
    node_id: str
    root_id: str
    rel: float
    sem: float
    lex: float
    matched: list[str] = Field(default_factory=list)


class ThreadScore(BaseModel):
    root_id: str
    best_rel: float
    prior: float
    score: float
    eligible: bool


class GateCheck(BaseModel):
    name: str
    value: float
    threshold: float
    passed: bool
    slack: float
    note: str = ""


class LLMReview(BaseModel):
    """What the judge did, stored on the decision for the viewer and eval."""

    router_action: Action
    router_target: Optional[str]
    options: list[Optional[str]] = Field(default_factory=list)  # targets, by letter
    answer: str = ""          # raw model text (trimmed)
    choice: Optional[str] = None  # the letter used, None on failure
    agreed: bool = True
    error: str = ""
    latency_ms: int = 0
    model: str = ""


class RoutingDecision(BaseModel):
    action: Action
    target_message_id: Optional[str]
    confidence: Confidence
    decided_by: Literal["rules", "explicit", "llm"]  # llm: the LLM review overrode the rules
    reason: str
    uncertainty: Optional[Uncertainty] = None  # set when TENTATIVE
    llm: Optional[LLMReview] = None  # set when a RoutingJudge reviewed it
    cues: list[str] = Field(default_factory=list)
    best_rel: Optional[float] = None
    background: Optional[float] = None
    thread_id: Optional[str] = None
    threads: list[ThreadScore] = Field(default_factory=list)
    candidates: list[CandidateScore] = Field(default_factory=list)
    gates: list[GateCheck] = Field(default_factory=list)
    data_id: str = ""

    @property
    def tentative(self) -> bool:
        return self.confidence == "TENTATIVE"


@dataclass
class RouterState:
    """Derived from the tree alone (nothing extra is persisted): replaying the
    user messages in creation order reproduces it exactly."""

    root_of: dict[str, str]
    active_leaf: Optional[str]
    active_thread: Optional[str]
    thread_leaf: dict[str, str]
    activity: dict[str, float]


def derive_state(tree: ConversationTree, decay: float) -> RouterState:
    nodes = sorted(tree.messages.values(), key=lambda n: n.created_seq)
    root_of: dict[str, str] = {}
    for n in nodes:  # parents are always created before children
        p = n.parent_message_id
        root_of[n.message_id] = root_of.get(p, p) if p else n.message_id
    activity: dict[str, float] = {}
    thread_leaf: dict[str, str] = {}
    active_leaf = active_thread = None
    for n in nodes:
        if n.node_type != "user":
            continue
        root = root_of[n.message_id]
        for key in activity:
            activity[key] *= decay
        activity[root] = activity.get(root, 0.0) + 1.0
        active_thread = root
        reply = next(
            (c for c in n.child_message_ids
             if c in tree.messages and tree.messages[c].node_type == "assistant"),
            None,
        )
        if reply:
            thread_leaf[root] = reply
            active_leaf = reply
    return RouterState(root_of, active_leaf, active_thread, thread_leaf, activity)


@dataclass(frozen=True)
class _NodeFeatures:
    terms: frozenset
    user_vec: Vec
    assistant_vec: Vec


def _r(x: float) -> float:
    return round(x, _ROUND)


def _gate(name: str, value: float, threshold: float, passed: bool, note: str = "") -> GateCheck:
    return GateCheck(
        name=name, value=_r(value), threshold=_r(threshold), passed=passed,
        slack=_r(abs(value - threshold)), note=note,
    )


class ThreadRouter:
    """Places each new user message under one assistant message, or starts a
    new topic. Pure, deterministic, no LLM."""

    def __init__(
        self,
        config: RouterConfig = RouterConfig(),
        data: Optional[RouterData] = None,
    ) -> None:
        self.config = config
        self.data = data or load_router_data(config.data_dir)
        self._features: dict[str, _NodeFeatures] = {}
        self._bg_cache: dict[str, float] = {}

    # ------------------------------------------------------------ features

    def _node_features(self, tree: ConversationTree, assistant_id: str) -> _NodeFeatures:
        """Terms of the turn (user + assistant) and both embeddings. Message
        content never changes, so this is cached by message id."""
        cached = self._features.get(assistant_id)
        if cached is not None:
            return cached
        a = tree.messages[assistant_id]
        u = tree.messages.get(a.parent_message_id or "")
        u_text = u.content if u else ""
        emb = self.data.embedder
        feats = _NodeFeatures(
            terms=term_set(u_text) | term_set(a.content),
            user_vec=emb.embed(topic_text(u_text)),
            assistant_vec=emb.embed(topic_text(a.content)),
        )
        self._features[assistant_id] = feats
        return feats

    def background_similarity(self, vec: Vec) -> float:
        """bg_percentile of the message's cosine to the background prompts."""
        sims = sorted(cosine(vec, b) for b in self.data.background)
        n = len(sims)
        if n == 0:
            return 0.0
        p_milli = round(self.config.bg_percentile * 1000)
        idx = max(0, min(n - 1, (p_milli * n + 999) // 1000 - 1))
        return sims[idx]

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _classify(
        tree: ConversationTree, target: Optional[str], state: RouterState
    ) -> Action:
        if target is None:
            return "NEW"
        if target == state.active_leaf:
            return "CONTINUE"
        if not tree.messages[target].child_message_ids:
            return "SWITCH"
        return "FORK"

    @staticmethod
    def _assistant_path(tree: ConversationTree, node_id: str) -> list[str]:
        """Assistant nodes from the thread root down to node_id (inclusive)."""
        out = []
        cur = tree.messages.get(node_id)
        while cur is not None:
            if cur.node_type == "assistant":
                out.append(cur.message_id)
            cur = tree.messages.get(cur.parent_message_id or "")
        return out[::-1]

    @staticmethod
    def _newest_leaf_under(tree: ConversationTree, node_id: str) -> str:
        best = tree.messages[node_id]
        stack = [node_id]
        while stack:
            n = tree.messages[stack.pop()]
            if n.node_type == "assistant" and not n.child_message_ids:
                if n.created_seq > best.created_seq or best.child_message_ids:
                    best = n
            stack.extend(n.child_message_ids)
        return best.message_id

    def _explicit(self, tree, state, target, reason) -> RoutingDecision:
        return RoutingDecision(
            action=self._classify(tree, target, state),
            target_message_id=target,
            confidence="FIRM",
            decided_by="explicit",
            reason=reason,
            thread_id=state.root_of.get(target) if target else None,
            data_id=self.data.data_id,
        )

    # ------------------------------------------------------------ route

    def route(
        self,
        tree: ConversationTree,
        message: str,
        *,
        reply_to: Optional[str] = None,
        new_topic: bool = False,
    ) -> RoutingDecision:
        cfg = self.config
        state = derive_state(tree, cfg.activity_decay)

        # Step 3: explicit overrides, empty tree
        if reply_to is not None:
            node = tree.messages.get(reply_to)
            if node is None or node.node_type != "assistant":
                raise ValueError(
                    f"reply_to must be an existing assistant message id: {reply_to!r}"
                )
            return self._explicit(tree, state, reply_to, "explicit reply_to")
        if new_topic:
            return self._explicit(tree, state, None, "explicit new topic")

        assistants = sorted(
            (n for n in tree.messages.values() if n.node_type == "assistant"),
            key=lambda n: n.created_seq,
        )
        if not assistants:
            return RoutingDecision(
                action="NEW", target_message_id=None, confidence="FIRM",
                decided_by="rules", reason="empty tree: new topic",
                data_id=self.data.data_id,
            )

        # Step 1: read the message
        cues: Cues = detect_cues(
            message,
            fragment_max_terms=cfg.fragment_max_terms,
            anaphora_max_terms=cfg.anaphora_max_terms,
            return_phrases=cfg.return_phrases,
            new_topic_phrases=cfg.new_topic_phrases,
            follow_up_phrases=cfg.follow_up_phrases,
            anaphora_words=cfg.anaphora_words,
            request_markers=cfg.request_markers,
        )
        m_terms = term_set(message)
        m_words = topic_words(message)
        m_vec = self.data.embedder.embed(topic_text(message))
        bg = self.background_similarity(m_vec)
        rest_cache: dict[frozenset, tuple[Vec, float]] = {}

        def rest_of_message(matched: frozenset) -> tuple[Vec, float]:
            """Embedding of the message words NOT shared exactly with the
            node, and its background similarity. lex already counts the shared
            words; sem must judge the rest, or one shared word ("container",
            "index") is counted twice."""
            if not matched:
                return m_vec, bg
            hit = rest_cache.get(matched)
            if hit is None:
                rest = [w for w in m_words if stem(w) not in matched]
                if rest:
                    v = self.data.embedder.embed(" ".join(rest))
                    hit = (v, self.background_similarity(v))
                else:
                    hit = (m_vec, bg)  # every word matched: lex is 1.0 anyway
                rest_cache[matched] = hit
            return hit
        idf = self.data.idf
        idf_total = sum(idf(t) for t in m_terms)

        # Step 2: score every assistant node
        cands: dict[str, CandidateScore] = {}
        seq: dict[str, int] = {}
        for a in assistants:
            f = self._node_features(tree, a.message_id)
            shared = m_terms & f.terms
            matched = sorted(shared)
            lex = sum(idf(t) for t in matched) / idf_total if idf_total else 0.0
            r_vec, r_bg = rest_of_message(frozenset(shared))
            sim = max(cosine(r_vec, f.user_vec), cosine(r_vec, f.assistant_vec))
            sem = min(1.0, max(0.0, (sim - r_bg) / cfg.s_full)) if r_vec.sq else 0.0
            cands[a.message_id] = CandidateScore(
                node_id=a.message_id,
                root_id=state.root_of[a.message_id],
                # matched words count fully; the rest counts as far as it is
                # related in meaning
                rel=_r(_r(lex) + (1 - _r(lex)) * _r(sem)),
                sem=_r(sem),
                lex=_r(lex),
                matched=matched,
            )
            seq[a.message_id] = a.created_seq

        def rel(nid: str) -> float:
            return cands[nid].rel

        # thread table (Step 5 inputs)
        by_thread: dict[str, list[str]] = {}
        for nid in cands:
            by_thread.setdefault(cands[nid].root_id, []).append(nid)
        max_act = max(state.activity.values(), default=0.0) or 1.0
        threads: dict[str, ThreadScore] = {}
        for root, nids in by_thread.items():
            best_t = max(rel(n) for n in nids)
            prior = _r(cfg.w_act * state.activity.get(root, 0.0) / max_act)
            threads[root] = ThreadScore(
                root_id=root, best_rel=best_t, prior=prior,
                score=_r(best_t + prior), eligible=best_t >= cfg.tau_attach,
            )

        best_node = max(cands, key=lambda n: (rel(n), seq[n]))
        best = rel(best_node)
        gates: list[GateCheck] = []
        deciding: list[tuple[str, float]] = []  # (check, slack) that decided the outcome
        forced_tentative: list[str] = []
        uncertainty: Optional[Uncertainty] = None
        why: list[str] = []

        def pick_thread(eligible_only: bool) -> Optional[str]:
            pool = [t for t in threads.values() if t.eligible or not eligible_only]
            if not pool:
                return None
            pool.sort(
                key=lambda t: (t.score, state.activity.get(t.root_id, 0.0),
                               tree.messages[t.root_id].created_seq),
                reverse=True,
            )
            if len(pool) > 1:
                margin = _r(pool[0].score - pool[1].score)
                g = _gate("thread_margin", margin, cfg.m_slack, margin >= cfg.m_slack,
                          f"{pool[0].root_id[:8]} vs {pool[1].root_id[:8]}")
                gates.append(g)
                deciding.append(("thread_margin", margin))
            return pool[0].root_id

        # Step 4: NEW gate
        thread: Optional[str] = None
        if cues.new_topic and best < cfg.tau_override:
            g = _gate("new_topic_override", best, cfg.tau_override, True,
                      "NEW_TOPIC cue; best rel below override")
            gates.append(g)
            deciding.append((g.name, g.slack))
            why.append(f"new topic: NEW_TOPIC cue and best rel {best:.3f} < {cfg.tau_override}")
        elif (cues.refers_back and not cues.return_) or (
            cues.return_ and best < cfg.tau_attach
        ):
            # never NEW: a return ("the first one", "going back to") also
            # points at something already said, even when its words do not
            # match it. A reference points at what was just said: the active
            # thread, unless another thread leads it on content by m_switch
            # AND has attach-level evidence of its own. A lead over a weak
            # active thread is not enough: a generic answer ("I can help with
            # coding, writing...") is loosely similar to almost anything.
            thread = state.active_thread
            active_best = threads[thread].best_rel if thread in threads else 0.0
            rival = max(
                (t for t in threads.values() if t.root_id != thread),
                key=lambda t: (t.best_rel, tree.messages[t.root_id].created_seq),
                default=None,
            )
            if rival is not None:
                lead = _r(rival.best_rel - active_best)
                g = _gate("thread_switch", lead, cfg.m_switch, lead >= cfg.m_switch,
                          f"best other thread vs active thread")
                gates.append(g)
                deciding.append((g.name, g.slack))
                if g.passed:
                    g_ev = _gate("thread_switch_evidence", rival.best_rel,
                                 cfg.tau_attach, rival.eligible,
                                 "other thread needs attach-level evidence")
                    gates.append(g_ev)
                    deciding.append((g_ev.name, g_ev.slack))
                    if g_ev.passed:
                        thread = rival.root_id
            t_best = threads[thread].best_rel
            g = _gate("attach", t_best, cfg.tau_attach, t_best >= cfg.tau_attach,
                      "reference cue: never NEW")
            gates.append(g)
            answers_question = (
                (cues.affirmation or cues.fragment)
                and thread == state.active_thread
                and state.active_leaf is not None
                and asks_question(tree.messages[state.active_leaf].content)
            )
            if g.passed:
                deciding.append((g.name, g.slack))
                why.append(f"reference cue, matched thread (rel {t_best:.3f})")
            elif answers_question:
                # "yes" / "the refund please" right after the assistant asked
                # something: the answer belongs to that question. No content
                # evidence is expected, so this is not a weak attach.
                gates.append(GateCheck(
                    name="answers_question", value=1.0, threshold=1.0, passed=True,
                    slack=1.0, note="short reply right after the assistant asked a question",
                ))
                why.append("short reply answers the assistant's question")
            else:
                uncertainty = "reference"
                forced_tentative.append(
                    f"attached by reference cue; content evidence {t_best:.3f} < {cfg.tau_attach}"
                )
                why.append("reference cue: never NEW; best thread by content + activity")
        elif best >= cfg.tau_attach:
            g = _gate("attach", best, cfg.tau_attach, True)
            gates.append(g)
            deciding.append((g.name, g.slack))
            thread = pick_thread(eligible_only=True)
            why.append(f"attach: best rel {best:.3f} >= {cfg.tau_attach}")
        elif best < cfg.tau_new:
            g = _gate("new", best, cfg.tau_new, True, "best rel below tau_new")
            gates.append(g)
            deciding.append((g.name, g.slack))
            why.append(f"new topic: best rel {best:.3f} < {cfg.tau_new}")
        else:
            gates.append(_gate("attach", best, cfg.tau_attach, False, "uncertain band"))
            gates.append(_gate("new", best, cfg.tau_new, False, "uncertain band"))
            uncertainty = "band"
            forced_tentative.append(
                f"best rel {best:.3f} in uncertain band [{cfg.tau_new}, {cfg.tau_attach})"
            )
            # a wrong attach only adds context; a wrong NEW loses all of it
            rare = [t for t in cands[best_node].matched if idf(t) >= cfg.idf_rare]
            if state.root_of[best_node] == state.active_thread:
                thread = state.active_thread
                why.append("uncertain: best match is in the active thread")
            elif rare:
                # a rare shared term ("fastapi", "alembic") names the topic;
                # lex undersells it because the generic words around it share
                # the idf total. Common shared words ("container", "index")
                # are not enough: those are the false matches NEW guards.
                thread = state.root_of[best_node]
                why.append(
                    "uncertain: best match is in another thread and shares a rare "
                    f"term ({', '.join(rare)})"
                )
            else:
                why.append("uncertain: best match is not in the active thread -> new topic")

        target: Optional[str] = None
        if thread is not None:
            target = self._place(
                tree, state, cands, seq, thread, cues, gates, deciding, why
            )

        action = self._classify(tree, target, state)
        confidence: Confidence = "FIRM"
        if forced_tentative or any(s < cfg.m_slack for _, s in deciding):
            confidence = "TENTATIVE"
        if confidence == "TENTATIVE" and not forced_tentative:
            uncertainty = "margin"
            name, slack = min(deciding, key=lambda d: d[1])
            forced_tentative.append(f"{name} slack {slack:.3f} < {cfg.m_slack}")
        reason = f"{action.lower()}: " + "; ".join(why)
        if forced_tentative:
            reason += " | tentative: " + "; ".join(forced_tentative)

        ranked = sorted(cands.values(), key=lambda c: (-c.rel, -seq[c.node_id]))
        thread_rows = sorted(
            threads.values(),
            key=lambda t: (-t.score, -tree.messages[t.root_id].created_seq),
        )
        return RoutingDecision(
            action=action,
            target_message_id=target,
            confidence=confidence,
            decided_by="rules",
            reason=reason,
            uncertainty=uncertainty if confidence == "TENTATIVE" else None,
            cues=cues.names(),
            best_rel=best,
            background=_r(bg),
            thread_id=thread if target else None,
            threads=thread_rows,
            candidates=ranked[: cfg.max_candidates_reported],
            gates=gates,
            data_id=self.data.data_id,
        )

    # ------------------------------------------------------------ Step 6/7

    def _place(self, tree, state, cands, seq, thread, cues, gates, deciding, why) -> str:
        cfg = self.config

        def rel(nid: str) -> float:
            return cands[nid].rel

        base = state.thread_leaf.get(thread) or self._newest_leaf_under(tree, thread)
        nodes = [n for n in cands if cands[n].root_id == thread]
        x = max(nodes, key=lambda n: (rel(n), seq[n]))
        path = self._assistant_path(tree, base)

        if x == base:
            why.append("best node is the branch leaf -> continue")
            return base
        if x in path:
            if self._fork_gate(tree, cands, x, base, cues, gates, deciding):
                why.append(f"fork at earlier answer (rel {rel(x):.3f})")
                return x
            why.append("earlier answer not distinct enough -> continue at leaf")
            return base

        # x is on another branch of this thread. Compare it only with the part
        # of the current branch that x's branch does not share: the common
        # ancestors are in the history either way.
        x_path = set(self._assistant_path(tree, x))
        own = [n for n in path if n not in x_path]
        lead = _r(rel(x) - max(rel(n) for n in own))
        if cues.return_:
            gates.append(_gate("switch", lead, cfg.m_switch, True, "RETURN cue"))
            moved = True
        else:
            g = _gate("switch", lead, cfg.m_switch, lead >= cfg.m_switch)
            gates.append(g)
            deciding.append((g.name, g.slack))
            moved = g.passed
        if not moved:
            why.append("other branch does not lead enough -> continue at leaf")
            return base
        leaf_x = self._newest_leaf_under(tree, x)
        if leaf_x == x:
            why.append("best node is the leaf of another branch -> switch")
            return x
        if self._fork_gate(tree, cands, x, leaf_x, cues, gates, deciding):
            why.append(f"fork at answer on another branch (rel {rel(x):.3f})")
            return x
        why.append("resume the other branch at its leaf")
        return leaf_x

    def _fork_gate(self, tree, cands, x, leaf, cues, gates, deciding) -> bool:
        """G1 distinct evidence, G2 intent, G3 not a fragment."""
        cfg = self.config
        path = self._assistant_path(tree, leaf)
        below = path[path.index(x) + 1:]
        k = len(below)
        delta = _r(min(cfg.delta_max, cfg.delta0 + cfg.delta_step * (k - 1)))
        diff = _r(cands[x].rel - max(cands[d].rel for d in below))

        g1 = _gate("fork_G1_distinct", diff, delta, diff >= delta,
                   f"k={k} turn(s) would be dropped")
        gates.append(g1)
        g2 = GateCheck(
            name="fork_G2_intent", value=1.0 if cues.return_ else 0.0,
            threshold=1.0, passed=cues.return_, slack=1.0,
            note="a fork needs a RETURN cue" + ("" if cues.return_ else " (none)"),
        )
        gates.append(g2)
        g3 = GateCheck(name="fork_G3_not_fragment", value=0.0 if cues.fragment else 1.0,
                       threshold=1.0, passed=not cues.fragment, slack=1.0)
        gates.append(g3)

        passed = g1.passed and g2.passed and g3.passed
        if passed:
            deciding.append(("fork_gate", min(g1.slack, g2.slack)))
        else:
            failing = [g.slack for g in (g1, g2, g3) if not g.passed]
            deciding.append(("fork_gate", max(failing)))
        return passed
