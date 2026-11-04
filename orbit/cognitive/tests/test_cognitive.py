import unittest
import torch
from orbit.cognitive.core import build_seed_data, encode_examples, encode_text
from orbit.cognitive.models import PerceptionNet, AssociationNet, MemoryNet
from orbit.cognitive.language import VocabularyTokenizer, NeuralLanguageGenerator
from orbit.cognitive.train import assistant_target_mask, _format_daily_dialogue, _normalise_ultrachat_messages, _load_ultrachat_dialogues, _generate_conversation_rollouts, _language_batch, _masked_language_loss, _prepare_language_sampler
from unittest.mock import patch
from orbit.cognitive.chat import entity_aliases, find_mentioned_entities, relation_hint, compose_is_a, canonical_relation_query, build_conversation_prompt

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

    def test_canonical_relation_query_uses_training_template(self):
        self.assertEqual(canonical_relation_query("cat", "is_a"), "what is cat")
        self.assertEqual(canonical_relation_query("tree", "needs"), "what does tree need")
        self.assertEqual(canonical_relation_query("river", "flows_into"), "where does river flow")

    def test_is_a_answers_use_natural_articles(self):
        self.assertEqual(compose_is_a("rain", "precipitation"), "rain is precipitation.")
        self.assertEqual(compose_is_a("car", "vehicle"), "car is a vehicle.")
        self.assertEqual(compose_is_a("computer", "electronic device"), "computer is an electronic device.")
        self.assertEqual(compose_is_a("desk", "furniture"), "desk is furniture.")


    def test_runtime_prompt_keeps_system_latest_message_and_whole_recent_turns(self):
        examples = [
            "<|system|>You are ORBIT. "
            "<|user|> older question marker. <|context|> older context marker. "
            "<|assistant|> older reply marker. "
            "<|user|> latest question marker. <|context|> latest context marker. "
            "<|assistant|>"
        ] * 5
        tokenizer = VocabularyTokenizer.build(examples, max_vocab=1024, min_frequency=1)
        history = [
            (f"old question {index} " + "details " * 10,
             f"old answer {index} " + "explanation " * 10,
             f"old context {index}")
            for index in range(8)
        ]
        prompt = build_conversation_prompt(
            "LATEST_USER_MARKER what should I do next?",
            "LATEST_CONTEXT_MARKER",
            history,
            tokenizer,
            block_size=180,
            system_prompt="You are ORBIT. Be helpful.",
        )
        self.assertLessEqual(len(tokenizer.encode(prompt)), 180)
        self.assertIn("<|system|>", prompt)
        self.assertIn("LATEST_USER_MARKER", prompt)
        self.assertIn("LATEST_CONTEXT_MARKER", prompt)
        self.assertIn("<|assistant|>", prompt)
        self.assertNotIn("old question 0", prompt)
        self.assertNotIn("old answer 0", prompt)

    def test_runtime_prompt_rejects_current_question_larger_than_context(self):
        examples = ["<|system|>You are ORBIT. <|user|> hello <|context|> facts <|assistant|>"] * 4
        tokenizer = VocabularyTokenizer.build(examples, max_vocab=512, min_frequency=1)
        with self.assertRaises(ValueError):
            build_conversation_prompt(
                "word " * 100,
                "No retrieved facts.",
                [],
                tokenizer,
                block_size=8,
                system_prompt="You are ORBIT.",
            )

    def test_conversation_rollout_uses_generated_reply_in_follow_up_history(self):
        class FakeGenerator:
            def __init__(self):
                self.prompts = []
            def generate(self, prompt, tokenizer, **kwargs):
                self.prompts.append(prompt)
                return f"generated response {len(self.prompts)}"

        dialogue = {
            "dataset": "test dialogue",
            "rollout_messages": [
                {"role": "user", "text": "I am building a small game."},
                {"role": "assistant", "text": "REFERENCE FIRST ANSWER"},
                {"role": "user", "text": "What should I implement first?"},
                {"role": "assistant", "text": "REFERENCE SECOND ANSWER"},
                {"role": "user", "text": "How do I test that?"},
                {"role": "assistant", "text": "REFERENCE THIRD ANSWER"},
            ],
        }
        model = FakeGenerator()
        samples = _generate_conversation_rollouts(
            model, tokenizer=None, dialogues=[dialogue], max_dialogues=1, max_turns=3
        )
        self.assertEqual(len(samples), 3)
        self.assertIn("generated response 1", model.prompts[1])
        self.assertNotIn("REFERENCE FIRST ANSWER", model.prompts[1])
        self.assertIn("generated response 2", model.prompts[2])
        self.assertNotIn("REFERENCE SECOND ANSWER", model.prompts[2])
        self.assertEqual(samples[1]["evaluation_mode"], "generated-history rollout")

    def test_ultrachat_message_normalisation_preserves_conversation_order(self):
        raw = [
            {"role": "user", "content": "I am building a small game."},
            {"role": "assistant", "content": "That sounds fun."},
            {"role": "assistant", "content": "What kind of game is it?"},
            {"role": "user", "content": "A puzzle game."},
            {"role": "assistant", "content": "Start with one simple puzzle mechanic."},
        ]
        messages = _normalise_ultrachat_messages(raw)
        self.assertEqual([item["role"] for item in messages],
                         ["user", "assistant", "user", "assistant"])
        self.assertIn("That sounds fun.\n\nWhat kind of game is it?", messages[1]["text"])
        self.assertEqual(messages[-1]["text"], "Start with one simple puzzle mechanic.")

    def test_ultrachat_loader_splits_conversations_and_keeps_test_separate(self):
        def make_row(index, split):
            return {
                "prompt_id": f"{split}-{index}",
                "messages": [
                    {"role": "user", "content": f"Question number {index} in {split}-{index}."},
                    {"role": "assistant", "content": f"First reply {split}-{index}."},
                    {"role": "user", "content": f"Follow-up {split}-{index}."},
                    {"role": "assistant", "content": f"Second reply {split}-{index}."},
                ],
            }
        train_rows = [make_row(index, "train") for index in range(30)]
        test_rows = [make_row(index, "test") for index in range(5)]
        def fake_download(url, label):
            if "test_sft" in label:
                return test_rows, "test-sha", 200
            return train_rows, "train-sha", 1000
        with patch("orbit.cognitive.train._download_parquet_rows", side_effect=fake_download):
            train_docs, validation_docs, eval_examples, metadata = _load_ultrachat_dialogues(
                seed=29, max_train_conversations=30
            )
        self.assertTrue(train_docs)
        self.assertTrue(validation_docs)
        self.assertTrue(eval_examples)
        self.assertTrue(all("train-" in doc for doc in train_docs + validation_docs))
        self.assertTrue(all("test-" in item["reference"] for item in eval_examples))
        self.assertEqual(metadata["license"], "MIT")
        self.assertEqual(metadata["sha256"]["train_shard_0000"], "train-sha")
        self.assertNotIn("test-", "\n".join(train_docs + validation_docs))

    def test_daily_dialogue_formatter_preserves_alternating_multi_turn_history(self):
        row = {
            "turns": [
                {"speaker": "A", "text": "I want to learn programming."},
                {"speaker": "B", "text": "What would you like to build?"},
                {"speaker": "A", "text": "A small puzzle game."},
                {"speaker": "B", "text": "Start with one simple puzzle mechanic."},
            ]
        }
        formatted = _format_daily_dialogue(row)
        self.assertIsNotNone(formatted)
        document, messages = formatted
        self.assertEqual([item["role"] for item in messages],
                         ["user", "assistant", "user", "assistant"])
        self.assertIn("<|user|> I want to learn programming.", document)
        self.assertIn("<|assistant|> What would you like to build?", document)
        self.assertIn("<|user|> A small puzzle game.", document)
        self.assertIn("<|assistant|> Start with one simple puzzle mechanic.", document)
        self.assertTrue(document.endswith("<|end|>"))

    def test_daily_dialogue_formatter_rejects_broken_turn_order(self):
        row = {
            "turns": [
                {"speaker": "A", "text": "Hello."},
                {"speaker": "A", "text": "I have another thought."},
                {"speaker": "B", "text": "Go ahead."},
                {"speaker": "A", "text": "Thanks."},
            ]
        }
        self.assertIsNone(_format_daily_dialogue(row))

    def test_assistant_target_mask_ignores_user_and_context_tokens(self):
        text = (
            "<|user|> Hello there. <|context|> No retrieved facts. "
            "<|assistant|> I can help with that. <|user|> Another question. "
            "<|context|> No retrieved facts. <|assistant|> Here is the answer. <|end|>"
        )
        tokenizer = VocabularyTokenizer.build([text] * 4, max_vocab=1024, min_frequency=2)
        ids = tokenizer.encode(text)
        mask = assistant_target_mask(ids, tokenizer)
        user_id = tokenizer.token_id("<|user|>")
        assistant_id = tokenizer.token_id("<|assistant|>")
        context_id = tokenizer.token_id("<|context|>")
        hello_id = tokenizer.token_id("Hello")
        help_id = tokenizer.token_id("help")
        user_positions = [i for i, value in enumerate(ids) if value == user_id]
        assistant_positions = [i for i, value in enumerate(ids) if value == assistant_id]
        context_positions = [i for i, value in enumerate(ids) if value == context_id]
        self.assertEqual(len(user_positions), 2)
        self.assertEqual(len(assistant_positions), 2)
        self.assertFalse(bool(mask[ids.index(hello_id)]))
        self.assertTrue(bool(mask[ids.index(help_id)]))
        self.assertTrue(bool(mask[user_positions[1]]))
        self.assertFalse(bool(mask[context_positions[0]]))
        self.assertGreater(int(mask.sum()), 5)

    def test_language_batches_always_include_assistant_targets(self):
        text = (
            "<|system|>You are helpful. "
            "<|user|> I need help with a project. "
            "<|context|> No verified facts. "
            "<|assistant|> Start with one small, testable step. "
            "<|user|> What should I do next? "
            "<|context|> No verified facts. "
            "<|assistant|> Tell me which part is most confusing. <|end|>"
        )
        tokenizer = VocabularyTokenizer.build([text] * 4, max_vocab=1024, min_frequency=2)
        ids = tokenizer.encode(text) + [tokenizer.EOS_ID]
        sequence = torch.tensor(ids, dtype=torch.long)
        mask = assistant_target_mask(ids, tokenizer)
        eligible = _prepare_language_sampler([sequence], [mask], block_size=24)
        inputs, targets, loss_masks = _language_batch(
            [sequence], [mask], block_size=24, batch_size=8,
            pad_id=tokenizer.PAD_ID, device=torch.device("cpu"), eligible=eligible,
        )
        self.assertEqual(tuple(inputs.shape), (8, 24))
        self.assertEqual(tuple(targets.shape), (8, 24))
        self.assertTrue(bool(loss_masks.any(dim=1).all()))
        self.assertTrue(bool((targets[loss_masks] != tokenizer.PAD_ID).all()))

    def test_assistant_only_loss_is_finite(self):
        logits = torch.randn(2, 5, 20, requires_grad=True)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.zeros((2, 5), dtype=torch.bool)
        mask[:, 2:] = True
        loss = _masked_language_loss(logits, targets, mask, pad_id=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

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

    def test_subword_tokenizer_round_trips_unseen_words(self):
        training_texts = [
            "<|assistant|> Hello there. This is a small language model.",
            "<|user|> Can you explain a small language model?",
        ] * 3
        tokenizer = VocabularyTokenizer.build(training_texts, max_vocab=1024, min_frequency=2)
        sample = "<|user|> A never-seen flibbertigibbet walks in."
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
        self.assertGreater(tokenizer.vocab_size, 259)

    def test_neural_language_generator_shape_and_sampling(self):
        tokenizer = VocabularyTokenizer.build(
            ["<|assistant|> Hello there. This is a small language model."] * 4,
            max_vocab=1024, min_frequency=2,
        )
        config = {
            "block_size": 24, "dim": 32, "layers": 1, "heads": 2,
            "dropout": 0.0, "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.PAD_ID, "eos_id": tokenizer.EOS_ID,
        }
        generator = NeuralLanguageGenerator(config)
        tokens = torch.randint(0, tokenizer.vocab_size, (2, 16))
        logits = generator(tokens)
        self.assertEqual(tuple(logits.shape), (2, 16, tokenizer.vocab_size))
        generated = generator.generate(
            "<|assistant|>", tokenizer, max_new_tokens=12, temperature=0.0
        )
        self.assertIsInstance(generated, str)

if __name__ == "__main__":
    unittest.main()
