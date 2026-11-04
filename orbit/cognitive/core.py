"""Data definitions and deterministic dataset generation for ORBIT Cognitive v1.

The initial corpus is a small, inspectable seed knowledge graph. Training examples
are generated from it with multiple question templates. This is a bootstrapping
benchmark, not broad world knowledge; the report records that limitation.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_LEN = 32

# subject, relation, object, provenance
RAW_FACTS = [
    ("rain", "is_a", "precipitation", "seed_knowledge_v1"),
    ("rain", "made_of", "water", "seed_knowledge_v1"),
    ("rain", "falls_from", "clouds", "seed_knowledge_v1"),
    ("rain", "located_in", "atmosphere", "seed_knowledge_v1"),
    ("cloud", "is_a", "weather_phenomenon", "seed_knowledge_v1"),
    ("cloud", "made_of", "water_droplets", "seed_knowledge_v1"),
    ("cloud", "located_in", "sky", "seed_knowledge_v1"),
    ("cloud", "produces", "rain", "seed_knowledge_v1"),
    ("water", "is_a", "substance", "seed_knowledge_v1"),
    ("water", "has_property", "liquid_at_room_temperature", "seed_knowledge_v1"),
    ("water", "made_of", "hydrogen_and_oxygen", "seed_knowledge_v1"),
    ("car", "is_a", "vehicle", "seed_knowledge_v1"),
    ("car", "used_for", "transportation", "seed_knowledge_v1"),
    ("car", "powered_by", "engine", "seed_knowledge_v1"),
    ("car", "made_of", "metal_glass_and_plastic", "seed_knowledge_v1"),
    ("desk", "is_a", "furniture", "seed_knowledge_v1"),
    ("desk", "used_for", "working_and_studying", "seed_knowledge_v1"),
    ("desk", "made_of", "wood_or_metal", "seed_knowledge_v1"),
    ("computer", "is_a", "electronic_device", "seed_knowledge_v1"),
    ("computer", "used_for", "computation", "seed_knowledge_v1"),
    ("computer", "contains", "cpu", "seed_knowledge_v1"),
    ("computer", "contains", "ram", "seed_knowledge_v1"),
    ("computer", "contains", "storage", "seed_knowledge_v1"),
    ("cpu", "is_a", "processor", "seed_knowledge_v1"),
    ("cpu", "performs", "instructions", "seed_knowledge_v1"),
    ("cpu", "part_of", "computer", "seed_knowledge_v1"),
    ("ram", "is_a", "volatile_memory", "seed_knowledge_v1"),
    ("ram", "stores", "temporary_data", "seed_knowledge_v1"),
    ("ram", "part_of", "computer", "seed_knowledge_v1"),
    ("storage", "stores", "persistent_data", "seed_knowledge_v1"),
    ("storage", "part_of", "computer", "seed_knowledge_v1"),
    ("tree", "is_a", "plant", "seed_knowledge_v1"),
    ("tree", "needs", "sunlight", "seed_knowledge_v1"),
    ("tree", "needs", "water", "seed_knowledge_v1"),
    ("tree", "produces", "oxygen", "seed_knowledge_v1"),
    ("tree", "absorbs", "carbon_dioxide", "seed_knowledge_v1"),
    ("plant", "is_a", "living_organism", "seed_knowledge_v1"),
    ("plant", "needs", "sunlight", "seed_knowledge_v1"),
    ("plant", "needs", "water", "seed_knowledge_v1"),
    ("plant", "produces", "oxygen", "seed_knowledge_v1"),
    ("dog", "is_a", "mammal", "seed_knowledge_v1"),
    ("dog", "has_property", "domesticated_animal", "seed_knowledge_v1"),
    ("cat", "is_a", "mammal", "seed_knowledge_v1"),
    ("cat", "has_property", "domesticated_animal", "seed_knowledge_v1"),
    ("earth", "is_a", "planet", "seed_knowledge_v1"),
    ("earth", "orbits", "sun", "seed_knowledge_v1"),
    ("moon", "orbits", "earth", "seed_knowledge_v1"),
    ("sun", "is_a", "star", "seed_knowledge_v1"),
    ("sun", "produces", "light_and_heat", "seed_knowledge_v1"),
    ("gravity", "is_a", "physical_force", "seed_knowledge_v1"),
    ("gravity", "causes", "objects_to_accelerate_toward_each_other", "seed_knowledge_v1"),
    ("photosynthesis", "is_a", "biological_process", "seed_knowledge_v1"),
    ("photosynthesis", "needs", "sunlight", "seed_knowledge_v1"),
    ("photosynthesis", "needs", "water", "seed_knowledge_v1"),
    ("photosynthesis", "absorbs", "carbon_dioxide", "seed_knowledge_v1"),
    ("photosynthesis", "produces", "oxygen", "seed_knowledge_v1"),
    ("ocean", "is_a", "body_of_water", "seed_knowledge_v1"),
    ("river", "is_a", "flowing_body_of_water", "seed_knowledge_v1"),
    ("river", "flows_into", "ocean", "seed_knowledge_v1"),
    ("fire", "needs", "fuel", "seed_knowledge_v1"),
    ("fire", "needs", "oxygen", "seed_knowledge_v1"),
    ("python", "is_a", "programming_language", "seed_knowledge_v1"),
    ("python", "used_for", "software_development", "seed_knowledge_v1"),
    ("python", "used_for", "automation", "seed_knowledge_v1"),
    ("python", "used_for", "data_analysis", "seed_knowledge_v1"),
    ("internet", "is_a", "global_network", "seed_knowledge_v1"),
    ("internet", "connects", "computers_and_networks", "seed_knowledge_v1"),
]

QUESTION_TEMPLATES = {
    "is_a": [
        "what is {s}", "define {s}", "what kind of thing is {s}",
        "what category does {s} belong to", "explain what {s} is",
        "tell me what {s} is", "describe {s}", "how would you classify {s}",
        "give me a definition of {s}", "which type of thing is {s}",
    ],
    "made_of": [
        "what is {s} made of", "what materials make up {s}",
        "what are the components of {s}", "which substances form {s}",
        "what does {s} consist of", "what makes up {s}",
        "which materials are used to make {s}", "explain the composition of {s}",
        "what ingredients or materials form {s}", "tell me the composition of {s}",
    ],
    "falls_from": [
        "what does {s} fall from", "where does {s} fall from",
        "from what does {s} fall", "what is the source that {s} falls from",
        "where does falling {s} come from", "what does {s} descend from",
        "tell me where {s} falls from", "what is above {s} when it falls",
        "which thing does {s} fall out of", "identify the source of {s}",
    ],
    "located_in": [
        "where is {s} found", "where does {s} exist",
        "where can i find {s}", "in what place is {s} located",
        "what is the location of {s}", "where is {s} usually found",
        "tell me where {s} is", "what environment contains {s}",
        "in which place would i find {s}", "where does {s} occur",
    ],
    "produces": [
        "what does {s} produce", "what comes from {s}",
        "what does {s} create", "what output does {s} make",
        "what is made by {s}", "which things are produced by {s}",
        "tell me what {s} produces", "what does {s} generate",
        "what is an output of {s}", "what results from {s}",
    ],
    "used_for": [
        "what is {s} used for", "why do people use {s}",
        "what purpose does {s} serve", "how is {s} useful",
        "what can someone do with {s}", "what is the purpose of {s}",
        "tell me the uses of {s}", "which tasks use {s}",
        "what job does {s} help with", "how do people use {s}",
    ],
    "powered_by": [
        "what powers {s}", "what supplies power to {s}",
        "what makes {s} run", "what is the power source for {s}",
        "what provides energy to {s}", "how is {s} powered",
        "tell me what powers {s}", "which component drives {s}",
        "what energy source runs {s}", "what runs {s}",
    ],
    "contains": [
        "what does {s} contain", "which components are inside {s}",
        "what is inside {s}", "what parts does {s} have",
        "which things are included in {s}", "list a component of {s}",
        "what can be found inside {s}", "tell me what {s} contains",
        "what does {s} include", "name something contained in {s}",
    ],
    "performs": [
        "what does {s} do", "which operations does {s} perform",
        "what actions are carried out by {s}", "what task does {s} perform",
        "what work does {s} do", "tell me what {s} performs",
        "what operation is handled by {s}", "what is the function of {s}",
        "which job does {s} carry out", "describe the work of {s}",
    ],
    "part_of": [
        "what is {s} part of", "where does {s} belong",
        "which larger thing contains {s}", "what system includes {s}",
        "what does {s} belong to", "tell me what {s} is part of",
        "which whole contains {s}", "what larger object includes {s}",
        "where would {s} be found as a component", "identify the whole that {s} belongs to",
    ],
    "stores": [
        "what does {s} store", "what information is stored by {s}",
        "which data does {s} keep", "what is saved in {s}",
        "what kind of data does {s} hold", "tell me what {s} stores",
        "what can {s} remember", "which information does {s} retain",
        "what data is kept by {s}", "what does {s} hold",
    ],
    "needs": [
        "what does {s} need", "what does {s} require",
        "which things are necessary for {s}", "what is needed by {s}",
        "what resources does {s} need", "tell me what {s} requires",
        "what must {s} have", "what is essential for {s}",
        "which input does {s} need", "what does {s} depend on",
    ],
    "absorbs": [
        "what does {s} absorb", "which substance does {s} take in",
        "what is absorbed by {s}", "what does {s} take from its surroundings",
        "tell me what {s} absorbs", "which material is taken in by {s}",
        "what enters {s} through absorption", "what does {s} soak up",
        "which gas does {s} absorb", "what substance goes into {s}",
    ],
    "orbits": [
        "what does {s} orbit", "which object does {s} go around",
        "what body does {s} travel around", "what is orbited by {s}",
        "tell me what {s} orbits", "around what does {s} move",
        "which object is at the center of {s}'s orbit",
        "what does {s} revolve around", "which body does {s} circle",
        "identify the object that {s} orbits",
    ],
    "causes": [
        "what does {s} cause", "what happens because of {s}",
        "what effect does {s} have", "what can {s} make happen",
        "tell me what {s} causes", "which result can come from {s}",
        "what is an effect of {s}", "what outcome is caused by {s}",
        "what does {s} lead to", "what can result from {s}",
    ],
    "flows_into": [
        "where does {s} flow", "what does {s} flow into",
        "where does the water from {s} go", "which body of water receives {s}",
        "tell me where {s} flows", "where does {s} empty into",
        "what is downstream from {s}", "where does {s} run into",
        "which place does {s} drain into", "identify where {s} flows",
    ],
    "has_property": [
        "what property does {s} have", "describe a property of {s}",
        "what is one characteristic of {s}", "how can {s} be described",
        "tell me a feature of {s}", "what characteristic does {s} have",
        "name a property of {s}", "what is {s} like",
        "give me one attribute of {s}", "what feature is associated with {s}",
    ],
    "connects": [
        "what does {s} connect", "which things are connected by {s}",
        "what is linked together by {s}", "what does {s} join",
        "tell me what {s} connects", "which systems does {s} link",
        "what can communicate through {s}", "what does {s} tie together",
        "what is joined using {s}", "which things does {s} connect together",
    ],
}

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())

def build_vocabulary(facts: list[dict[str, Any]]) -> dict[str, int]:
    words = set()
    for fact in facts:
        words.update(tokenize(fact["text"]))
    for relation, templates in QUESTION_TEMPLATES.items():
        for template in templates:
            words.update(tokenize(template.format(s="example_concept")))
        words.update(tokenize(relation))
    ordered = [PAD_TOKEN, UNK_TOKEN] + sorted(words - {PAD_TOKEN, UNK_TOKEN})
    return {word: idx for idx, word in enumerate(ordered)}

def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)[:max_len]]
    result = np.zeros(max_len, dtype=np.int64)
    if ids:
        result[:len(ids)] = ids
    return result

def build_seed_data() -> dict[str, Any]:
    facts = []
    entities = sorted({x for s, _, o, _ in RAW_FACTS for x in (s, o)})
    relations = sorted({r for _, r, _, _ in RAW_FACTS})
    entity_to_id = {name: i for i, name in enumerate(entities)}
    relation_to_id = {name: i for i, name in enumerate(relations)}
    for idx, (subject, relation, obj, source) in enumerate(RAW_FACTS):
        facts.append({
            "id": idx,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "subject_id": entity_to_id[subject],
            "relation_id": relation_to_id[relation],
            "object_id": entity_to_id[obj],
            "source": source,
            "confidence": 0.95,
            "text": f"{subject} {relation} {obj}",
        })

    vocab = build_vocabulary(facts)

    # Group all facts that share a subject/relation. A single question such
    # as "what does a tree need?" can have multiple correct fact records.
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for fact in facts:
        key = (fact["subject_id"], fact["relation_id"])
        grouped.setdefault(key, []).append(fact)

    examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    association_rows = []
    for (subject_id, relation_id), group_facts in sorted(grouped.items()):
        subject = group_facts[0]["subject"]
        relation = group_facts[0]["relation"]
        templates = QUESTION_TEMPLATES[relation]
        fact_ids = sorted(fact["id"] for fact in group_facts)
        object_ids = sorted({fact["object_id"] for fact in group_facts})
        association_rows.append({
            "subject_id": subject_id,
            "relation_id": relation_id,
            "object_ids": object_ids,
            "fact_ids": fact_ids,
        })
        for split, indices in (
            ("train", range(0, 6)),
            ("validation", range(6, 8)),
            ("test", range(8, 10)),
        ):
            for template_index in indices:
                question = templates[template_index].format(s=subject.replace("_", " "))
                examples[split].append({
                    "text": question,
                    "subject_id": subject_id,
                    "relation_id": relation_id,
                    "fact_id": fact_ids[0],
                    "fact_ids": fact_ids,
                })

    # AssociationNet is a compact learned cache of the complete known graph.
    # It is therefore trained and measured on all known subject/relation groups.
    # Its graph-reconstruction score is explicitly NOT a held-out link-prediction
    # or generalisation benchmark.
    assoc = {
        "train": list(association_rows),
        "validation": list(association_rows),
        "test": list(association_rows),
    }
    return {
        "facts": facts,
        "entities": entities,
        "relations": relations,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "vocab": vocab,
        "examples": examples,
        "association_splits": assoc,
        "max_len": MAX_LEN,
    }

def encode_examples(rows: list[dict[str, Any]], vocab: dict[str, int], max_len: int = MAX_LEN):
    if not rows:
        return (
            np.zeros((0, max_len), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    x = np.stack([encode_text(row["text"], vocab, max_len) for row in rows])
    subjects = np.asarray([row["subject_id"] for row in rows], dtype=np.int64)
    relations = np.asarray([row["relation_id"] for row in rows], dtype=np.int64)
    fact_ids = np.asarray([row["fact_id"] for row in rows], dtype=np.int64)
    return x, subjects, relations, fact_ids

def encode_facts(facts: list[dict[str, Any]], vocab: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    return np.stack([encode_text(fact["text"], vocab, max_len) for fact in facts])
