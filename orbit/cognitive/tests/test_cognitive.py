import unittest
import torch
from orbit.cognitive.core import build_seed_data, encode_examples, encode_text
from orbit.cognitive.models import PerceptionNet, AssociationNet, MemoryNet
from orbit.cognitive.chat import entity_aliases, find_mentioned_entities, relation_hint, compose_is_a

class CognitiveDataTests(unittest.TestCase):
    def test_seed_data_is_nonempty_and_split(self):
        data = build_seed_data()
        self.assertGreater(len(data["facts"]), 40)
        self.assertGreater(len(data["entities"]), 20)
        for name in ("train", "validation", "test"):
            self.assertTrue(data["examples"][name])

    def test_vocab_and_encoding(self):
        data = build_seed_data()
        encoded = encode_text("What is rain?", data["vocab"])
        self.assertEqual(encoded.shape[0], data["max_len"])
        self.assertGreater(int(encoded.max()), 0)
        self.assertEqual(encoded.dtype.name, "int64")

    def test_each_fact_has_ten_question_variants(self):
        data = build_seed_data()
        for fact in data["facts"]:
            examples = [row for split in data["examples"].values()
                        for row in split if fact["id"] in row["fact_ids"]]
            self.assertEqual(len(examples), 10)

    def test_memory_queries_group_all_valid_facts(self):
        data = build_seed_data()
        tree_id = data["entity_to_id"]["tree"]
        needs_id = data["relation_to_id"]["needs"]
        rows = [
            row for row in data["examples"]["train"]
            if row["subject_id"] == tree_id and row["relation_id"] == needs_id
        ]
        self.assertEqual(len(rows), 6)
        expected = {
            fact["id"] for fact in data["facts"]
            if fact["subject_id"] == tree_id and fact["relation_id"] == needs_id
        }
        self.assertEqual(set(rows[0]["fact_ids"]), expected)
        self.assertTrue(all(set(row["fact_ids"]) == expected for row in rows))

    def test_association_groups_allow_multiple_correct_objects(self):
        data = build_seed_data()
        subject_id = data["entity_to_id"]["tree"]
        relation_id = data["relation_to_id"]["needs"]
        groups = [
            group for group in data["association_splits"]["train"]
            if group["subject_id"] == subject_id and group["relation_id"] == relation_id
        ]
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["object_ids"]), 2)
        expected_pairs = {
            (row["subject_id"], row["relation_id"])
            for row in data["association_splits"]["train"]
        }
        for split in ("validation", "test"):
            actual_pairs = {
                (row["subject_id"], row["relation_id"])
                for row in data["association_splits"][split]
            }
            self.assertEqual(actual_pairs, expected_pairs)

    def test_plural_concepts_are_recognised(self):
        entities = ["cat", "tree", "country", "bus"]
        self.assertEqual(find_mentioned_entities("How are cats classified?", entities), [0])
        self.assertEqual(find_mentioned_entities("What do trees need?", entities), [1])
        self.assertEqual(find_mentioned_entities("Which countries have trees?", entities), [1, 2])
        self.assertIn("buses", entity_aliases("bus"))

    def test_relation_cues_route_common_question_forms(self):
        relations = ["is_a", "made_of", "needs", "located_in", "flows_into"]
        self.assertEqual(relation_hint("How are cats classified?", relations), "is_a")
        self.assertEqual(relation_hint("What is rain made of?", relations), "made_of")
        self.assertEqual(relation_hint("What does a tree need?", relations), "needs")
        self.assertEqual(relation_hint("Where does a cloud form?", relations), "located_in")
        self.assertEqual(relation_hint("Where does a river flow?", relations), "flows_into")

    def test_is_a_answers_use_natural_articles(self):
        self.assertEqual(compose_is_a("rain", "precipitation"), "rain is precipitation.")
        self.assertEqual(compose_is_a("car", "vehicle"), "car is a vehicle.")
        self.assertEqual(compose_is_a("computer", "electronic device"), "computer is an electronic device.")
        self.assertEqual(compose_is_a("desk", "furniture"), "desk is furniture.")

    def test_neural_module_shapes(self):
        data = build_seed_data()
        x, subjects, relations, _ = encode_examples(data["examples"]["train"][:4], data["vocab"])
        x = torch.as_tensor(x, dtype=torch.long)
        perception = PerceptionNet(len(data["vocab"]), len(data["entities"]), len(data["relations"]))
        s_logits, r_logits = perception(x)
        self.assertEqual(tuple(s_logits.shape), (4, len(data["entities"])))
        self.assertEqual(tuple(r_logits.shape), (4, len(data["relations"])))
        association = AssociationNet(len(data["entities"]), len(data["relations"]))
        scores = association(torch.as_tensor(subjects), torch.as_tensor(relations))
        self.assertEqual(tuple(scores.shape), (4, len(data["entities"])))
        memory = MemoryNet(len(data["vocab"]))
        logits = memory(x, x)
        self.assertEqual(tuple(logits.shape), (4, 4))

if __name__ == "__main__":
    unittest.main()
