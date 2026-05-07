"""Conversation builder spec (v1) shared by API, DB, and LangGraph compiler.

Router rules (graph mode):
- After an agent node speaks, the runtime picks the next node from **outgoing edges**
  of the current node.
- Optional per-node ``static_message``: fixed spoken text when that turn uses ``static``
  in ``reply_turn_modes`` (or legacy: whenever the user has already spoken and no modes set).
- Optional per-node ``reply_turn_modes``: ordered list (``static`` / ``llm``) for successive
  agent utterances **at this node** (including the opener before any user speech). If omitted
  with ``static_message`` set, legacy behavior applies: opener uses the LLM (or static only
  when modes explicitly request it); after the user speaks, replies use ``static_message``.
- **0 outgoing edges** → the current node keeps handling the conversation.
- **1 outgoing edge** → that target is chosen without an extra LLM call.
- **2+ outgoing edges** → ``ConversationBrain.classify`` picks one label from the
  allowed target node ids (instruction summarizes the last user turn and options).
  Destination **labels** from the graph are included when set; optional per-node
  ``classify_hint`` adds routing rules for ambiguous intents.
- Optional per-node ``loop_min_turns`` / ``loop_max_turns`` constrain self-loops:
  min blocks leaving until enough completed stays; max forces exit to a non-self
  neighbor after enough stays (only when those edges exist).
- To end a graph conversation, add an explicit edge to the terminal ``complete`` node.

Single mode:
- One internal node id (``__single__``) runs every turn with ``system_prompt`` until
  max turns, exit phrases handled in the service layer, or ``complete``.

Template ``sdr``:
- Ignores ``mode``/``nodes``/``edges`` for execution; the app uses the legacy
  SDR LangGraph (tools, stages) while still storing optional spec metadata.
"""

from __future__ import annotations

import re
from typing import Any, Literal

ReplyTurnMode = Literal["static", "llm"]

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONVERSATION_SPEC_VERSION = 1
SINGLE_AGENT_NODE_ID = "__single__"
# Agent node ids must not collide with LangGraph internal node names.
RESERVED_NODE_IDS = frozenset({"route_turn", "complete", SINGLE_AGENT_NODE_ID})


class SpecNode(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    label: str | None = Field(default=None, max_length=120)
    system_prompt: str = Field(..., min_length=1)
    tool_ids: list[str] = Field(default_factory=list)
    # Consecutive "stay" turns on this node (self-loop) for custom graph routing.
    # Applied only when the node has 2+ outgoing edges including a self-loop (classify path).
    loop_min_turns: int | None = Field(
        default=None,
        ge=0,
        description="Minimum completed stay cycles before classifier may leave this node.",
    )
    loop_max_turns: int | None = Field(
        default=None,
        ge=1,
        description="After this many completed stays, force exit to a non-self neighbor if one exists.",
    )
    classify_hint: str | None = Field(
        default=None,
        max_length=2000,
        description="When this node has multiple outbound edges, appended to the router LLM prompt.",
    )
    static_message: str | None = Field(
        default=None,
        max_length=8000,
        description=(
            "Fixed spoken text when this node's reply mode is ``static`` (see reply_turn_modes). "
            "Supports the same {placeholder} variables as system_prompt."
        ),
    )
    reply_turn_modes: list[ReplyTurnMode] | None = Field(
        default=None,
        description=(
            "Order maps to successive agent utterances at this node (0 = opener before user speaks, "
            "then each reply while on this node). Each entry chooses static_message vs LLM (system_prompt). "
            "Indices beyond this list default to llm. When omitted and static_message is set, legacy "
            "behavior applies for utterances after the user has spoken."
        ),
    )

    @field_validator("reply_turn_modes", mode="before")
    @classmethod
    def empty_reply_modes(cls, v: object) -> object:
        if v == []:
            return None
        return v

    @field_validator("id")
    @classmethod
    def id_alnum(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                "node id must start with a letter and contain only letters, digits, underscores"
            )
        if v.lower() == "complete":
            raise ValueError("node id 'complete' is reserved for the terminal state")
        if v in RESERVED_NODE_IDS:
            raise ValueError(f"node id {v!r} is reserved")
        return v

    @model_validator(mode="after")
    def loop_min_max_consistent(self):
        if self.loop_min_turns is not None and self.loop_max_turns is not None:
            if self.loop_max_turns < self.loop_min_turns:
                raise ValueError("loop_max_turns must be >= loop_min_turns when both are set")
        return self


class SpecEdge(BaseModel):
    from_: str = Field(..., alias="from", min_length=1, max_length=64)
    to: str = Field(..., min_length=1, max_length=64)

    model_config = {"populate_by_name": True}

    @field_validator("from_", "to")
    @classmethod
    def edge_node_ref(cls, v: str) -> str:
        if v == "complete":
            return v
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError("edge endpoint must be a valid node id or 'complete'")
        return v


class ConversationSpecV1(BaseModel):
    """Wire format stored in ``bot_configs.conversation_spec``."""

    model_config = ConfigDict(extra="allow")

    conversation_spec_version: Literal[1] = 1
    mode: Literal["single", "graph"]
    template: Literal["custom", "sdr"] = "custom"
    system_prompt: str | None = None
    tool_ids: list[str] = Field(default_factory=list)
    entry_node_id: str | None = None
    nodes: list[SpecNode] = Field(default_factory=list)
    edges: list[SpecEdge] = Field(default_factory=list)
    variables_hint: str | None = None
    # Single-mode only: opener / turn scheduling without graph nodes.
    single_static_message: str | None = Field(
        default=None,
        max_length=8000,
        description="Fixed text for single-mode turns that use reply mode static.",
    )
    single_reply_turn_modes: list[ReplyTurnMode] | None = Field(
        default=None,
        description="Same semantics as per-node reply_turn_modes for the virtual single agent.",
    )

    @field_validator("single_reply_turn_modes", mode="before")
    @classmethod
    def empty_single_reply_modes(cls, v: object) -> object:
        if v == []:
            return None
        return v

    @model_validator(mode="after")
    def validate_mode_shape(self) -> ConversationSpecV1:
        if self.template == "sdr":
            return self
        if self.mode == "single":
            if not (self.system_prompt and self.system_prompt.strip()):
                raise ValueError("single mode requires non-empty system_prompt")
            if self.nodes or self.edges:
                raise ValueError("single mode must not include nodes or edges")
            if self.entry_node_id is not None:
                raise ValueError("single mode must not set entry_node_id")
            return self
        # graph
        if self.single_static_message is not None or self.single_reply_turn_modes is not None:
            raise ValueError("graph mode must not set single_static_message or single_reply_turn_modes")
        if not self.nodes:
            raise ValueError("graph mode requires at least one node")
        if not self.entry_node_id:
            raise ValueError("graph mode requires entry_node_id")
        ids = {n.id for n in self.nodes}
        if self.entry_node_id not in ids:
            raise ValueError("entry_node_id must match a node id")
        if self.system_prompt is not None:
            raise ValueError("graph mode must not set system_prompt on the root spec")
        for e in self.edges:
            if e.from_ not in ids and e.from_ != "complete":
                raise ValueError(f"edge from unknown node: {e.from_!r}")
            if e.to not in ids and e.to != "complete":
                raise ValueError(f"edge to unknown node: {e.to!r}")
        return self


def parse_conversation_spec(raw: Any | None) -> ConversationSpecV1 | None:
    """Parse DB JSON into a spec, or None if absent."""
    if raw is None:
        return None
    if isinstance(raw, ConversationSpecV1):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("conversation_spec must be a JSON object")
    return ConversationSpecV1.model_validate(raw)


def graph_execution_kind(bot_config: dict[str, Any]) -> Literal["sdr", "single", "graph"]:
    """Which LangGraph pipeline to run for this bot."""
    spec = parse_conversation_spec(bot_config.get("conversation_spec"))
    if spec is None or spec.template == "sdr":
        return "sdr"
    if spec.mode == "single":
        return "single"
    return "graph"


def build_adjacency(spec: ConversationSpecV1) -> dict[str, list[str]]:
    assert spec.mode == "graph"
    adj: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    for e in spec.edges:
        adj.setdefault(e.from_, [])
        if e.to not in adj[e.from_]:
            adj[e.from_].append(e.to)
    return adj


def prompt_for_node(spec: ConversationSpecV1, node_id: str) -> str:
    if spec.mode == "graph":
        for n in spec.nodes:
            if n.id == node_id:
                return n.system_prompt
        raise KeyError(node_id)
    raise ValueError("prompt_for_node only applies to graph mode")


def static_message_for_node(spec: ConversationSpecV1, node_id: str) -> str | None:
    """Non-empty static spoken text for this node when reply mode is ``static``."""
    if spec.mode != "graph":
        return None
    for n in spec.nodes:
        if n.id == node_id:
            raw = (n.static_message or "").strip()
            return raw if raw else None
    return None


def reply_turn_modes_for_node(spec: ConversationSpecV1, node_id: str) -> list[ReplyTurnMode] | None:
    if spec.mode != "graph":
        return None
    for n in spec.nodes:
        if n.id == node_id:
            return n.reply_turn_modes
    return None
