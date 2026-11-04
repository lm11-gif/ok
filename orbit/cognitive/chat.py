#!/usr/bin/env python3
"""Interactive inference for ORBIT Cognitive v1."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch

try:
    from .core import encode_text, encode_facts
    from .models import PerceptionNet, AssociationNet, MemoryNet
except ImportError:
    from core import encode_text, encode_facts
    from models import PerceptionNet, AssociationNet, MemoryNet

ANSWER_FORMS = {
    "is_a": "{s} is a {o}.", "made_of": "{s} is made of {o}.",
    "falls_from": "{s} falls from {o}.", "located_in": "{s} is found in {o}.",
    "produces": "{s} produces {o}.", "used_for": "{s} is used for {o}.",
    "powered_by": "{s} is powered by {o}.", "contains": "{s} contains {o}.",
    "performs": "{s} performs {o}.", "part_of": "{s} is part of {o}.",
    "stores": "{s} stores {o}.", "needs": "{s} needs {o}.",
    "absorbs": "{s} absorbs {o}.", "orbits": "{s} orbits {o}.",
    "causes": "{s} can cause {o}.", "flows_into": "{s} flows into {o}.",
    "has_property": "One property of {s} is that it is {o}.",
    "connects": "{s} connects {o}.",
}

def load_checkpoint(path: Path, model, device):
    state = torch.load(path, map_location=device)
    model_state = state["state_dict"]

    # Compatibility with the first v1 release, whose graph scorer shared one
    # entity embedding table. New AssociationNet has separate subject/object
    # tables; initialise both from the legacy table when loading an old zip.
    if isinstance(model, AssociationNet) and "entity.weight" in model_state:
        legacy = model_state
        model_state = {
            "subject.weight": legacy["entity.weight"],
            "object.weight": legacy["entity.weight"].clone(),
            "relation.weight": legacy["relation.weight"],
            "object_bias.weight": legacy["object_bias.weight"],
        }

    model.load_state_dict(model_state)
    model.eval()
    return state.get("metadata", {})

class OrbitCognitive:
    def __init__(self, model_dir: Path, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.entities = json.loads((model_dir / "entities.json").read_text(encoding="utf-8"))
        self.relations = json.loads((model_dir / "relations.json").read_text(encoding="utf-8"))
        self.facts = json.loads((model_dir / "facts.json").read_text(encoding="utf-8"))
        dim = int(self.config["dim"])
        self.perception = PerceptionNet(len(self.vocab), len(self.entities), len(self.relations), dim).to(self.device)
        self.association = AssociationNet(len(self.entities), len(self.relations), dim).to(self.device)
        self.memory = MemoryNet(len(self.vocab), dim).to(self.device)
        load_checkpoint(model_dir / "perception.pt", self.perception, self.device)
        load_checkpoint(model_dir / "association.pt", self.association, self.device)
        load_checkpoint(model_dir / "memory.pt", self.memory, self.device)
        fact_ids = encode_facts(self.facts, self.vocab, int(self.config["max_length"]))
        self.fact_tokens = torch.as_tensor(fact_ids, dtype=torch.long, device=self.device)
        self.fact_subject_ids = [f["subject_id"] for f in self.facts]
        self.fact_relation_ids = [f["relation_id"] for f in self.facts]

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().replace("_", " ").split())

    def _mentioned_entities(self, question: str):
        normalised = self._normalise(question)
        return [
            idx for idx, entity in enumerate(self.entities)
            if re.search(r"\b" + re.escape(entity.replace("_", " ")) + r"\b", normalised)
        ]

    @torch.inference_mode()
    def answer(self, question: str) -> str:
        matches = self._mentioned_entities(question)
        if not matches:
            return ("I don't have a recognised concept in that question yet. "
                    "My current knowledge base is small, so I won't guess.")
        ids = encode_text(question, self.vocab, int(self.config["max_length"]))
        x = torch.as_tensor(ids[None, :], dtype=torch.long, device=self.device)
        subject_logits, relation_logits = self.perception(x)
        subject_probs = subject_logits.softmax(dim=-1)[0]
        relation_probs = relation_logits.softmax(dim=-1)[0]
        subject_id = max(matches, key=lambda idx: float(subject_probs[idx].item()))
        subject_conf = float(subject_probs[subject_id].item())
        relation_id = int(relation_probs.argmax().item())
        relation_conf = float(relation_probs[relation_id].item())
        candidate_ids = [i for i, sid in enumerate(self.fact_subject_ids) if sid == subject_id]
        if not candidate_ids:
            return f"I recognise {self.entities[subject_id].replace('_', ' ')}, but I have no stored facts about it."
        scores = self.memory(x, self.fact_tokens)[0]
        relation_candidates = [i for i in candidate_ids if self.fact_relation_ids[i] == relation_id]
        if relation_conf >= 0.18 and not relation_candidates:
            relation_name = self.relations[relation_id].replace("_", " ")
            return (f"I recognise {self.entities[subject_id].replace('_', ' ')}, "
                    f"but my current memory has no reliable fact about {relation_name}.")
        if relation_candidates:
            candidate_ids = relation_candidates
        association_logits = self.association(
            torch.tensor([subject_id], device=self.device),
            torch.tensor([relation_id], device=self.device),
        )[0]
        association_log_probs = association_logits.log_softmax(dim=-1)
        def combined_score(fact_id: int) -> float:
            object_id = int(self.facts[fact_id]["object_id"])
            return float(scores[fact_id].item()) + 0.15 * float(association_log_probs[object_id].item())
        best_id = max(candidate_ids, key=combined_score)
        cosine_score = float(scores[best_id].item()) * float(self.memory.temperature)
        if cosine_score < 0.12 or subject_conf < 0.05:
            return (f"I recognise {self.entities[subject_id].replace('_', ' ')}, "
                    "but I couldn't retrieve a sufficiently relevant fact to answer confidently.")
        fact = self.facts[best_id]
        subject, obj = fact["subject"].replace("_", " "), fact["object"].replace("_", " ")
        answer = ANSWER_FORMS.get(fact["relation"], "{s} is related to {o}.").format(s=subject, o=obj)
        return answer + f" [retrieved fact similarity: {cosine_score:.2f}]"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/orbit-cognitive-v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    bot = OrbitCognitive(Path(args.model), args.device)
    print("ORBIT Cognitive v1. Enter 'exit' to quit.")
    print("This is an early knowledge-graph system, not a general assistant.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("ORBIT:", bot.answer(question))

if __name__ == "__main__":
    main()
