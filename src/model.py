from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

from datasets import load_metric
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer
)


class SequenceToSequence(ABC):
    """Class representing a conversor between sequences"""
    @abstractmethod
    def train_model(self):
        """Method to get the dataset"""

class SequenceSummarizerTrainer(SequenceToSequence):
    """Class representing a summarizer"""
    def __init__(
            self,
            training_args,
            dataset,
            dataset_tokenized,
            model_ckpt: str,
            model_output: str,
    ) -> None:
        self.training_args = training_args
        self.dataset = dataset
        self.dataset_tokenized = dataset_tokenized
        self.model_ckpt = model_ckpt
        self.model_output = model_output

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(self.device)

    def train_model(self):
        rouge_metric = load_metric("rouge")
        seq2seq_data_collator = self._get_data_collator()
        trainer = Trainer(model=self.model, args=self.training_args,
        tokenizer=self.tokenizer, data_collator=seq2seq_data_collator,
        train_dataset=self.dataset_tokenized["train"],
        eval_dataset=self.dataset_tokenized["validation"])

        trainer.train()
        trainer.save_model(self.model_output)
        score = self.evaluate_summaries_pegasus(
        self.dataset["test"], rouge_metric, trainer.model, self.tokenizer,
        batch_size=2, column_text="dialogue", column_summary="summary")

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        metrics = pd.DataFrame(rouge_dict, index=["pegasus"])
        metrics.to_csv(self.model_output+"metrics.csv", index = False)

    def evaluate_summaries_pegasus(self, dataset, metric, model, tokenizer,
                                batch_size=16, column_text="article",column_summary="highlights"):
        article_batches = list(self._chunks(dataset[column_text], batch_size))
        target_batches = list(self._chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

            inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                               padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(self.device),
                                       attention_mask=inputs["attention_mask"].to(self.device),
                                       length_penalty=0.8, num_beams=8, max_length=128)

            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True) for s in summaries]

            decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    def _get_data_collator(self):
        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        return seq2seq_data_collator
    
    def _chunks(self, list_of_elements, batch_size):
        """Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]