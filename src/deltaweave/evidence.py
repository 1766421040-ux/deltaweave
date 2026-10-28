"""Bound-reference transport for an already computed, verified tool result.

This module does not infer or repair model answers. A raw action must exactly
name the reference projected into its input. The host then returns the full
answer from that request's immutable proof record. Short handles identify
semantic answer/query bytes; snapshot and code bindings stay in the host record
so unrelated commits need not change the model's semantic input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import RLock

from .store import canonical


class EvidenceBindingError(ValueError):
    """A reference is stale, corrupted, ambiguous, or not the offered reference."""


def _digest(encoded):
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _answer(result):
    return {"action": "answer", "arguments": {
        key: result[key] for key in ("status", "selection", "citations")}}


def short_handle(content_sha256):
    return "h" + content_sha256[:16]


@dataclass(frozen=True)
class ReferenceRecord:
    """Only immutable JSON strings/scalars are retained; exports are copies."""
    handle: str
    content_sha256: str
    proof_sha256: str
    _semantic: str
    _proof: str

    def _verified(self):
        if _digest(self._semantic) != self.content_sha256:
            raise EvidenceBindingError("Reference semantic content digest mismatch")
        if self.handle != short_handle(self.content_sha256):
            raise EvidenceBindingError("Reference handle does not match its full digest")
        if _digest(self._proof) != self.proof_sha256:
            raise EvidenceBindingError("Reference host proof digest mismatch")
        semantic, proof = json.loads(self._semantic), json.loads(self._proof)
        if canonical(semantic) != self._semantic or canonical(proof) != self._proof:
            raise EvidenceBindingError("Reference record is not canonical")
        result, receipt, binding = proof["resolver_result"], proof["solver_receipt"], proof["runtime_binding"]
        expected_semantic = {"query": proof["query"], "answer": _answer(result)}
        if canonical(expected_semantic) != self._semantic:
            raise EvidenceBindingError("Host proof and semantic answer differ")
        if (receipt.get("name") != "resolve" or receipt.get("status") != "valid"
                or receipt.get("validation_contract") != "conservative_observation"
                or canonical(receipt.get("args")) != canonical(proof["query"])
                or receipt.get("validated_at") != binding["snapshot_revision"]
                or receipt.get("version") != binding["registry_versions"].get("resolve")
                or receipt.get("value_sha256") != _digest(canonical(result))):
            raise EvidenceBindingError("Solver receipt does not certify this result/binding")
        return semantic, proof

    def project_observation(self, observation):
        semantic, proof = self._verified()
        answer = semantic["answer"]["arguments"]
        if (canonical({key: observation[key] for key in answer}) != canonical(answer)
                or observation["validated_at"] != proof["runtime_binding"]["snapshot_revision"]
                or observation["computed_at"] != proof["solver_receipt"]["computed_at"]):
            raise EvidenceBindingError("Observation differs from its immutable host proof")
        projected = json.loads(canonical(observation))
        projected["selection"] = {"answer_ref": self.handle}
        projected["citations"] = ["proof:" + self.handle]
        return projected

    def expand(self, raw_action, *, query, binding):
        semantic, proof = self._verified()
        if (canonical(query) != canonical(proof["query"])
                or canonical(binding) != canonical(proof["runtime_binding"])):
            raise EvidenceBindingError("Reference is not bound to this request snapshot/configuration")
        status = semantic["answer"]["arguments"]["status"]
        if status not in {"resolved", "conflict"}:
            raise EvidenceBindingError("Incomplete evidence cannot be expanded into an answer")
        expected = {"action": "answer", "arguments": {"status": status,
                    "selection": {"answer_ref": self.handle}, "citations": ["proof:" + self.handle]}}
        if canonical(raw_action) != canonical(expected):
            raise EvidenceBindingError("Raw action does not exactly identify the offered proof")
        return json.loads(canonical(semantic["answer"]))

    def export(self):
        self._verified()
        return {"schema_version": 1, "handle": self.handle,
                "content_sha256": self.content_sha256, "proof_sha256": self.proof_sha256,
                "semantic": json.loads(self._semantic), **json.loads(self._proof),
                "scope": "Bound tool observation and execution receipt, not a proof of factual truth. Host expansion supplies the full payload; the model did not generate it."}


class ReferenceRegistry:
    """Collision checking only; current snapshot bindings are never stored here."""
    def __init__(self):
        self._semantic_by_handle = {}
        self._lock = RLock()

    def create(self, *, result, query, binding, receipt):
        # Canonicalization detaches caller-owned nested dictionaries immediately.
        semantic = canonical({"query": query, "answer": _answer(result)})
        content_sha256 = _digest(semantic)
        handle = short_handle(content_sha256)
        proof = canonical({"query": query, "runtime_binding": binding,
                           "resolver_result": result, "solver_receipt": receipt})
        record = ReferenceRecord(handle, content_sha256, _digest(proof), semantic, proof)
        record._verified()
        with self._lock:
            prior = self._semantic_by_handle.get(handle)
            if prior is not None and prior != semantic:
                raise EvidenceBindingError("Short reference collision; refusing ambiguous evidence")
            self._semantic_by_handle[handle] = semantic
        return record
