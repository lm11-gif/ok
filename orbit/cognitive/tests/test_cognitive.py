import unittest
import torch
from orbit.cognitive.core import build_seed_data, encode_examples, encode_text
from orbit.cognitive.models import PerceptionNet, AssociationNet, MemoryNet

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
                        for row in split if row["fact_id"] == fact["id"]]
            self.assertEqual(len(examples), 10)

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
