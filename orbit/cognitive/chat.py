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
    "is_a": "{s} is {o}.", "made_of": "{s} is made of {o}.",
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

RELATION_CUES = [
    ("falls_from", ("falls from", "fall from", "descends from")),
    ("made_of", ("made of", "consists of", "consist of", "composition")),
    ("flows_into", ("flows into", "flow into", "drains into", "drain into", "flows", "flow")),
    ("powered_by", ("powered by", "power source", "what powers", "what runs")),
    ("used_for", ("used for", "purpose", "what can i do with")),
    ("has_property", ("property", "characteristic", "feature", "what is like")),
    ("needs", ("need", "needs", "require", "requires", "necessary", "essential", "depend on")),
    ("absorbs", ("absorb", "absorbs", "takes in", "soaks up")),
    ("stores", ("store", "stores", "stored", "saved in", "holds")),
    ("contains", ("contain", "contains", "inside", "components", "parts")),
    ("performs", ("perform", "performs", "operation", "function")),
    ("part_of", ("part of", "belongs to", "included in")),
    ("orbits", ("orbit", "orbits", "revolve around", "goes around")),
    ("causes", ("cause", "causes", "effect", "result from")),
    ("connects", ("connect", "connects", "linked", "joins", "join")),
    ("produces", ("produce", "produces", "generates", "creates", "create", "output")),
    ("located_in", ("where", "located", "found in", "environment", "which place")),
    ("is_a", ("what is", "what are", "define", "classified", "classify", "category", "what kind", "what type")),
]

UNCOUNTABLE_IS_A_OBJECTS = {
    "precipitation", "furniture", "water", "oxygen", "computation",
    "transportation", "persistent data", "temporary data", "software development",
}

def entity_aliases(entity: str) -> set[str]:
    """Return a concept's exact and simple plural surface forms."""
    phrase = " ".join(entity.lower().replace("_", " ").split())
    aliases = {phrase}
    words = phrase.split()
    if not words:
        return aliases
    last = words[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        plural = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        plural = last + "es"
    else:
        plural = last + "s"
    aliases.add(" ".join(words[:-1] + [plural]))
    return aliases

def find_mentioned_entities(question: str, entities: list[str]) -> list[int]:
    normalised = " ".join(question.lower().replace("_", " ").split())
    matches = []
    for idx, entity in enumerate(entities):
        for alias in entity_aliases(entity):
            if re.search(r"\b" + re.escape(alias) + r"\b", normalised):
                matches.append(idx)
                break
    return matches

def relation_hint(question: str, relations: list[str]) -> str | None:
    """Use explicit wording as a symbolic routing hint; neural perception remains the fallback."""
    normalised = " ".join(question.lower().replace("_", " ").split())
    available = set(relations)
    for relation, cues in RELATION_CUES:
        if relation in available and any(
            re.search(r"\b" + re.escape(cue) + r"\b", normalised) for cue in cues
        ):
            return relation
    return None

def compose_is_a(subject: str, object_text: str) -> str:
    if object_text in UNCOUNTABLE_IS_A_OBJECTS:
        return f"{subject} is {object_text}."
    first_word = object_text.split()[0] if object_text else ""
    article = "an" if first_word[:1] in "aeiou" else "a"
    return f"{subject} is {article} {object_text}."

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
        return find_mentioned_entities(question, self.entities)

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
        predicted_relation_id = int(relation_probs.argmax().item())
        predicted_relation_conf = float(relation_probs[predicted_relation_id].item())
        hinted_relation = relation_hint(question, self.relations)
        if hinted_relation is not None:
            relation_id = self.relations.index(hinted_relation)
            relation_conf = 1.0
        else:
            relation_id = predicted_relation_id
            relation_conf = predicted_relation_conf
        candidate_ids = [i for i, sid in enumerate(self.fact_subject_ids) if sid == subject_id]
        if not candidate_ids:
            return f"I recognise {self.entities[subject_id].replace('_', ' ')}, but I have no stored facts about it."
        if relation_conf < 0.15:
            return (
                f"I recognise {self.entities[subject_id].replace('_', ' ')}, "
                "but I am not confident enough about the relationship being asked."
            )
        candidate_ids = [
            i for i in candidate_ids if self.fact_relation_ids[i] == relation_id
        ]
        if not candidate_ids:
            relation_name = self.relations[relation_id].replace("_", " ")
            return (f"I recognise {self.entities[subject_id].replace('_', ' ')}, "
                    f"but my current memory has no reliable fact about {relation_name}.")

        scores = self.memory(x, self.fact_tokens)[0]
        association_logits = self.association(
            torch.tensor([subject_id], device=self.device),
            torch.tensor([relation_id], device=self.device),
        )[0]
        association_log_probs = association_logits.log_softmax(dim=-1)

        # Rank all valid records for this subject/relation. Multiple facts are
        # returned together rather than treating one valid object as incorrect.
        def combined_score(fact_id: int) -> float:
            object_id = int(self.facts[fact_id]["object_id"])
            return float(scores[fact_id].item()) + 0.15 * float(association_log_probs[object_id].item())

        ranked_ids = sorted(candidate_ids, key=combined_score, reverse=True)
        best_similarity = float(scores[ranked_ids[0]].item()) * float(self.memory.temperature)
        if best_similarity < 0.12 or subject_conf < 0.05:
            return (
                f"I recognise {self.entities[subject_id].replace('_', ' ')}, "
                "but I couldn't retrieve a sufficiently relevant fact to answer confidently."
            )

        fact = self.facts[ranked_ids[0]]
        subject = fact["subject"].replace("_", " ")
        objects = [self.facts[fid]["object"].replace("_", " ") for fid in ranked_ids]
        if len(objects) == 1:
            object_text = objects[0]
        else:
            object_text = ", ".join(objects[:-1]) + " and " + objects[-1]
        template = ANSWER_FORMS.get(fact["relation"], "{s} is related to {o}.")
        if fact["relation"] == "is_a":
            answer = compose_is_a(subject, object_text)
        else:
            answer = template.format(s=subject, o=object_text)
        return answer + f" [retrieved fact similarity: {best_similarity:.2f}]"


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
