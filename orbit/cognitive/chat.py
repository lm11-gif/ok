#!/usr/bin/env python3
"""Run ORBIT Cognitive v2.1: specialist retrieval plus a trained subword language generator."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import torch

try:
    from .core import encode_text, encode_facts, QUESTION_TEMPLATES
    from .models import PerceptionNet, AssociationNet, MemoryNet
    from .language import VocabularyTokenizer, NeuralLanguageGenerator
except ImportError:
    from core import encode_text, encode_facts, QUESTION_TEMPLATES
    from models import PerceptionNet, AssociationNet, MemoryNet
    from language import VocabularyTokenizer, NeuralLanguageGenerator

LANGUAGE_SYSTEM_PROMPT = (
    "You are ORBIT, an experimental AI assistant. Respond naturally and directly. "
    "Use retrieved memory facts as evidence, but do not claim that memory is complete. "
    "When evidence is missing, be honest rather than inventing a factual answer."
)
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
    return [
        idx for idx, entity in enumerate(entities)
        if any(re.search(r"\b" + re.escape(alias) + r"\b", normalised)
               for alias in entity_aliases(entity))
    ]

def canonical_relation_query(subject: str, relation: str) -> str:
    templates = QUESTION_TEMPLATES.get(relation)
    if not templates:
        raise ValueError(f"Unsupported relation: {relation}")
    return templates[0].format(s=subject.replace("_", " "))

def relation_hint(question: str, relations: list[str]) -> str | None:
    normalised = " ".join(question.lower().replace("_", " ").split())
    available = set(relations)
    for relation, cues in RELATION_CUES:
        if relation in available and any(
            re.search(r"\b" + re.escape(cue) + r"\b", normalised) for cue in cues
        ):
            return relation
    return None

def compose_is_a(subject: str, object_text: str) -> str:
    """Legacy helper retained for regression tests; runtime chat uses neural generation."""
    if object_text in UNCOUNTABLE_IS_A_OBJECTS:
        return f"{subject} is {object_text}."
    first_word = object_text.split()[0] if object_text else ""
    article = "an" if first_word[:1].lower() in "aeiou" else "a"
    return f"{subject} is {article} {object_text}."

def load_checkpoint(path: Path, model, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    model_state = state["state_dict"]
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
    def __init__(self, model_dir: Path, device: str = "auto",
                 max_new_tokens: int = 240, temperature: float = 0.8):
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

        generator_config = self.config.get("language_generator")
        if not generator_config or not (model_dir / "generator.pt").is_file():
            raise RuntimeError(
                "This chat runner needs an ORBIT Cognitive v2 archive containing "
                "generator.pt. The older v1 archive contains retrieval networks only."
            )
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise RuntimeError(
                "This ORBIT Cognitive v2.1 archive is missing tokenizer.json. "
                "Use the matching v2.1 model ZIP rather than the older byte-level v2 ZIP."
            )
        self.language_tokenizer = VocabularyTokenizer.load(tokenizer_path)
        if int(generator_config["vocab_size"]) != self.language_tokenizer.vocab_size:
            raise RuntimeError("Generator checkpoint and tokenizer vocabulary do not match.")
        self.generator = NeuralLanguageGenerator(generator_config).to(self.device)
        load_checkpoint(model_dir / "generator.pt", self.generator, self.device)
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = float(temperature)
        self.history: list[tuple[str, str]] = []
        fact_ids = encode_facts(self.facts, self.vocab, int(self.config["max_length"]))
        self.fact_tokens = torch.as_tensor(fact_ids, dtype=torch.long, device=self.device)
        self.fact_subject_ids = [fact["subject_id"] for fact in self.facts]
        self.fact_relation_ids = [fact["relation_id"] for fact in self.facts]

    @torch.inference_mode()
    def _retrieve_facts(self, question: str) -> list[dict]:
        matches = find_mentioned_entities(question, self.entities)
        if not matches:
            return []
        encoded = encode_text(question, self.vocab, int(self.config["max_length"]))
        query = torch.as_tensor(encoded[None, :], dtype=torch.long, device=self.device)
        subject_logits, relation_logits = self.perception(query)
        subject_probs = subject_logits.softmax(dim=-1)[0]
        relation_probs = relation_logits.softmax(dim=-1)[0]
        subject_id = max(matches, key=lambda idx: float(subject_probs[idx].item()))
        subject_conf = float(subject_probs[subject_id].item())
        predicted_relation_id = int(relation_probs.argmax().item())
        predicted_relation_conf = float(relation_probs[predicted_relation_id].item())
        hinted_relation = relation_hint(question, self.relations)
        used_relation_hint = hinted_relation is not None
        if hinted_relation is not None:
            relation_id = self.relations.index(hinted_relation)
            relation_conf = 1.0
        else:
            relation_id = predicted_relation_id
            relation_conf = predicted_relation_conf

        subject_fact_ids = [
            i for i, sid in enumerate(self.fact_subject_ids) if sid == subject_id
        ]
        if not subject_fact_ids or subject_conf < 0.05:
            return []
        scores = self.memory(query, self.fact_tokens)[0].float()
        if relation_conf >= 0.15:
            candidate_ids = [
                i for i in subject_fact_ids if self.fact_relation_ids[i] == relation_id
            ]
        else:
            candidate_ids = list(subject_fact_ids)
        if not candidate_ids:
            candidate_ids = list(subject_fact_ids)
        association_logits = self.association(
            torch.tensor([subject_id], dtype=torch.long, device=self.device),
            torch.tensor([relation_id], dtype=torch.long, device=self.device),
        )[0].float().log_softmax(dim=-1)

        def rank(ids: list[int]) -> list[int]:
            return sorted(
                ids,
                key=lambda fid: float(scores[fid].item())
                + 0.15 * float(association_logits[int(self.facts[fid]["object_id"])].item()),
                reverse=True,
            )

        ranked = rank(candidate_ids)
        best_similarity = float(scores[ranked[0]].item()) * float(self.memory.temperature)
        if best_similarity < 0.12 and used_relation_hint:
            subject = self.entities[subject_id].replace("_", " ")
            canonical = canonical_relation_query(subject, self.relations[relation_id])
            canonical_ids = encode_text(canonical, self.vocab, int(self.config["max_length"]))
            canonical_query = torch.as_tensor(canonical_ids[None, :], dtype=torch.long, device=self.device)
            scores = self.memory(canonical_query, self.fact_tokens)[0].float()
            ranked = rank(candidate_ids)
            best_similarity = float(scores[ranked[0]].item()) * float(self.memory.temperature)
        if best_similarity < 0.12:
            return []
        return [self.facts[fid] for fid in ranked[:4]]

    @torch.inference_mode()
    def answer(self, question: str) -> str:
        facts = self._retrieve_facts(question)
        context = (
            "; ".join(
                f"{fact['subject'].replace('_', ' ')} {fact['relation'].replace('_', ' ')} "
                f"{fact['object'].replace('_', ' ')}"
                for fact in facts
            )
            if facts else "No verified memory facts were retrieved."
        )
        history_text = "".join(
            f"<|user|> {old_question}\n<|assistant|> {old_answer}\n"
            for old_question, old_answer in self.history[-4:]
        )
        prompt = (
            f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n{history_text}"
            f"<|user|> {question}\n<|context|> {context}\n<|assistant|>"
        )
        response = self.generator.generate(
            prompt, self.language_tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_k=40, top_p=0.92,
        )
        if not response:
            response = "I do not have enough information to give a useful answer yet."
        self.history.append((question, response))
        self.history = self.history[-6:]
        return response

    def reset_conversation(self):
        self.history.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/orbit-cognitive-v2")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    bot = OrbitCognitive(
        Path(args.model), args.device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    parameters = sum(p.numel() for p in bot.generator.parameters())
    print(f"ORBIT Cognitive v2.1 | neural subword generator | {parameters:,} generator parameters")
    print("Type /reset to clear conversation history, or /exit to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if question.lower() == "/reset":
            bot.reset_conversation()
            print("ORBIT: Conversation history cleared.")
            continue
        if question:
            print("\nORBIT:", bot.answer(question))


if __name__ == "__main__":
    main()
