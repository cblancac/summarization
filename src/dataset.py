from transformers import AutoTokenizer


class FormatDataset():
    """Class representing the dataset"""
    def __init__(
            self,
            dataset,
            model_ckpt,
    ):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)

    def get_tokenized_dataset(self):
        """Method to convert string to features"""
        dataset_tokenized = self.dataset.map(self._convert_examples_to_features,
                                             batched=True)

        columns = ["input_ids", "labels", "attention_mask"]
        dataset_tokenized.set_format(type="torch", columns=columns)
        return dataset_tokenized

    def _convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch["dialogue"], max_length=1024,
                                    truncation=True)
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch["summary"], max_length=128,
                                        truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}
