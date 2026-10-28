"""A bounded, read-only Agent loop over one pinned knowledge snapshot.

The rule policy is the default. Optional model predictions are never silently
rewritten into the rule answer: invalid actions produce an explicit blocked run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

from .engine import Query
from .evidence import EvidenceBindingError, ReferenceRegistry
from .learning_data import decide, prompt_for
from .provenance import execution_binding
from .store import canonical

REFERENCE_PROMPT_VERSION = "bound-observation-reference-v2"
REFERENCE_NOTE = """Reference response protocol v2: observation.selection.answer_ref and observation.citations are opaque transport identifiers, not package versions or factual proof. Copy the observation's status, selection and citations exactly when its contract permits answer; the host separately expands the bound observation. Optional full_evidence is read-only factual context, not the response-copy target. The publication/freshness/ask/refresh/abstain rules below still apply."""
_FULL_ANSWER_RULE = '5. A valid observation with status resolved or conflict requires action answer and arguments containing exactly status, selection, citations copied from that observation. Preserve their values. Example: {"action":"answer","arguments":{"status":"resolved","selection":{"demo":"2.0"},"citations":["m7"]}}. For conflict, selection is {}.'
_REFERENCE_ANSWER_RULE = '5. A valid observation with status resolved or conflict requires action answer and arguments containing exactly status, selection, citations copied from observation. For BOTH resolved and conflict, selection must remain {"answer_ref": the exact offered handle} and citations must remain the exact offered ["proof:" + handle] list. Do not replace a conflict reference with an empty object. Do not copy package selections or citations from full_evidence. Example transport shape: {"action":"answer","arguments":{"status":"conflict","selection":{"answer_ref":"h0123456789abcdef"},"citations":["proof:h0123456789abcdef"]}}. This example handle is not evidence; copy the handle actually offered in observation. The host, not the model, separately expands that bound reference into the full resolved/conflict payload.'


def reference_prompt_for(state):
    """Replace the full-payload answer rule; never leave conflicting instructions.

    All non-answer rules and the state serialization remain from prompt_for.
    Fail explicitly if that public full contract changes rather than silently
    applying a reference adapter written against different answer semantics.
    """
    prompt = prompt_for(state)
    old = "\n" + _FULL_ANSWER_RULE + "\n"
    if prompt.count(old) != 1:
        raise EvidenceBindingError("Full answer contract changed; reference prompt needs an explicit review")
    return REFERENCE_NOTE + "\n" + prompt.replace(old, "\n" + _REFERENCE_ANSWER_RULE + "\n", 1)


def normalize_state(state):
    """Erase absolute revision nonces while preserving equality-only semantics.

    Real revision/provenance stays in the host receipt. Under this interface,
    current versus old is semantic; the absolute global commit number is not.
    This does not erase explicit historical query dates or solver constraints.
    """
    state = deepcopy(state)
    revision = state["target_revision"]
    state["target_revision"] = 100
    for key in ("observation", "full_evidence"):
        obs = state.get(key)
        if obs is not None:
            obs["validated_at"] = 100 if obs["validated_at"] == revision else 99
            # Absolute generation time stays in the host receipt, including the
            # full-context control arm; it cannot poison semantic cache equality.
            obs["computed_at"] = 98
    return state


def environment_node(ctx, args):
    rows = ctx.query(Query("environment", where={"environment": args["environment"], "field": "python_full_version"}, select=("value",)))
    return rows[0]["value"] if len(rows) == 1 else None


class KnowledgeSession:
    def __init__(self, engine, *, policy=None, policy_identity=None, semantic_cache=True,
                 evidence_view="full"):
        if evidence_view not in {"full", "reference", "reference_context"}:
            raise ValueError("evidence_view must be full, reference, or reference_context")
        if engine.mode == "positive":
            raise ValueError("Positive-row heuristics cannot certify session observations")
        if policy is not None and (not callable(policy) or not isinstance(policy_identity, str)
                                   or not policy_identity.strip() or policy_identity == "rule-v1"):
            raise ValueError("Custom policies require an explicit non-rule identity binding model/adapter/decoding/seed")
        if policy is None and policy_identity not in (None, "rule-v1"):
            raise ValueError("The built-in rule policy has identity rule-v1")
        self.engine = engine
        self.engine.register("controller_python", environment_node, version="1")
        self._policy = policy
        self._policy_identity = "rule-v1" if policy is None else policy_identity
        self.semantic_cache = semantic_cache
        self._model_cache = {}
        self._observations = {}
        self._evidence_view = evidence_view
        self._references = ReferenceRegistry()

    @property
    def evidence_view(self):
        return self._evidence_view

    @property
    def policy(self):
        return self._policy

    @property
    def policy_identity(self):
        return self._policy_identity

    def _model_action(self, state):
        semantic = normalize_state(state)
        prompt = prompt_for(semantic) if self.evidence_view == "full" else reference_prompt_for(semantic)
        key = hashlib.sha256(canonical({"policy": self.policy_identity, "prompt": prompt,
            "evidence_view": self.evidence_view, "reference_protocol": REFERENCE_PROMPT_VERSION}).encode()).hexdigest()
        if self.policy is None:
            return decide(semantic), {"kind": "rule", "cache_hit": False, "prompt_sha256": key}
        if self.semantic_cache and key in self._model_cache:
            action, original = self._model_cache[key]
            return deepcopy(action), {"kind": "model", "cache_hit": True,
                "completed": original.get("completed", False), "prompt_sha256": key,
                "origin_receipt": deepcopy(original)}
        action, details = self.policy(prompt)
        receipt = {**details, "kind": "model", "cache_hit": False, "prompt_sha256": key}
        # Ordinary completed-generation memoization: mistakes are cached too and
        # are blocked by the same downstream guard, never rejection-sampled away.
        if self.semantic_cache and details.get("completed", False):
            self._model_cache[key] = (deepcopy(action), deepcopy(receipt))
        return action, receipt

    def request(self, query, *, max_actions=3, prevalidate=False, publication_ready=True):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be positive")
        started = time.perf_counter()
        snapshot = self.engine.store.snapshot
        query = json.loads(canonical(query))
        query_key = canonical(query)
        binding = execution_binding(self.engine, snapshot)
        cached = deepcopy(self._observations.get(query_key))
        # A same-revision code/source change must not pass numeric validity checks.
        if cached is not None and (cached["binding"]["registry_fingerprint"] != binding["registry_fingerprint"]
                or (cached["binding"]["snapshot_revision"] == snapshot.revision
                    and cached["binding"]["snapshot_sha256"] != binding["snapshot_sha256"])):
            cached = None
            self._observations.pop(query_key, None)
        observation = None if cached is None else cached["observation"]
        observation_binding = None if cached is None else cached["binding"]
        proof = None if cached is None else cached.get("proof")
        tool_receipts = []

        def check_registry():
            if self.engine.registry_fingerprint != binding["registry_fingerprint"]:
                raise _RuntimeBindingChanged

        def refresh():
            nonlocal observation_binding, proof
            check_registry()
            result, receipt = self.engine.run_with_receipt("resolve", query, snapshot=snapshot)
            tool_receipts.append(receipt)
            check_registry()
            observation_binding = deepcopy(binding)
            if self.evidence_view != "full":
                proof = self._references.create(result=result, query=query, binding=binding, receipt=receipt)
            return {"computed_at": receipt["computed_at"],
                    "validated_at": snapshot.revision, "status": result["status"],
                    "selection": result["selection"], "citations": result["citations"],
                    "issues": result["issues"]}

        steps, final = [], None
        status = "budget_exhausted"
        try:
            check_registry()
            python = self.engine.run("controller_python", {"environment": query["environment"]}, snapshot=snapshot)
            check_registry()
            if prevalidate and observation is not None and publication_ready:
                observation = refresh()
            for _ in range(max_actions):
                check_registry()
                visible_observation = observation
                if self.evidence_view != "full" and observation is not None:
                    if proof is None:
                        raise EvidenceBindingError("Reference observation lacks its host proof")
                    visible_observation = proof.project_observation(observation)
                state = {"request": "Resolve " + ", ".join(query["requirements"]) + " within the selected catalog.",
                         "python": python, "target_revision": snapshot.revision, "publication_ready": publication_ready,
                         "observation": visible_observation, "cached_answer": None,
                         "catalog_scope": "selected universal-wheel files only; declared constraints, not runtime compatibility"}
                if self.evidence_view == "reference_context":
                    state["full_evidence"] = deepcopy(observation)
                action, receipt = self._model_action(state)
                expected = decide(state)
                accepted = canonical(action) == canonical(expected) and (receipt["kind"] == "rule" or receipt.get("completed", False))
                step = {"state": state, "action": action, "raw_action": deepcopy(action),
                        "expanded_answer": None, "accepted": accepted, "receipt": receipt,
                        "proof_record": None if proof is None else proof.export()}
                steps.append(step)
                check_registry()
                if not accepted:
                    status = "blocked_model_contract"; break
                if action["action"] == "refresh":
                    observation = refresh()
                    continue
                final = action
                if self.evidence_view != "full" and action["action"] == "answer":
                    final = proof.expand(action, query=query, binding=binding)
                    step["expanded_answer"] = deepcopy(final)
                status = "completed"; break
        except _RuntimeBindingChanged:
            status, final = "blocked_runtime_binding", None
        except EvidenceBindingError:
            status, final = "blocked_evidence_binding", None
        if observation is not None and status not in {"blocked_runtime_binding", "blocked_evidence_binding"}:
            self._observations[query_key] = {"observation": deepcopy(observation),
                                             "binding": deepcopy(observation_binding), "proof": proof}
        fresh_calls = sum(step["receipt"]["kind"] == "model" and not step["receipt"]["cache_hit"]
                          for step in steps)
        cache_hits = sum(step["receipt"]["kind"] == "model" and step["receipt"]["cache_hit"]
                         for step in steps)
        return {"status": status, "answer": final, "snapshot_revision": snapshot.revision,
                "evidence_view": self.evidence_view,
                "reference_prompt_version": None if self.evidence_view == "full" else REFERENCE_PROMPT_VERSION,
                "raw_action": None if not steps else deepcopy(steps[-1]["raw_action"]),
                "expanded_answer": None if not steps else deepcopy(steps[-1]["expanded_answer"]),
                "proof_record": None if not steps else deepcopy(steps[-1]["proof_record"]),
                "snapshot_sha256": binding["snapshot_sha256"], "runtime_binding": binding,
                "query": deepcopy(query), "policy_identity": self.policy_identity, "steps": steps,
                "tool_receipts": tool_receipts, "seconds": time.perf_counter()-started,
                "fresh_model_calls": fresh_calls, "model_cache_hits": cache_hits,
                "closed_loop_contract_success": status == "completed",
                "raw_model_success": status == "completed" if fresh_calls else None,
                "scope": "Finite declared constraints; a blocked prediction is not counted as a correct model answer."}


class _RuntimeBindingChanged(Exception):
    """Do not complete a request across two registered-code configurations."""
