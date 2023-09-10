from transformers import TrainingArguments

from datasets import load_dataset
from src.dataset import FormatDataset
from src.model import SequenceSummarizerTrainer


training_args = TrainingArguments(
    output_dir="pegasus-samsum",
    num_train_epochs=1,
    warmup_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16,
)

DATASET_PATH = "samsum"
MODEL_CKPT = "google/pegasus-cnn_dailymail"
MODEL_OUTPUT = "pegasus-finetuned"


def train_pipeline():
    data = load_dataset(DATASET_PATH)
    data["train"] = data["train"].shuffle(seed=42).select(range(10))
    data["test"] = data["test"].shuffle(seed=42).select(range(2))
    data["validation"] = data["validation"].shuffle(seed=42).select(range(2))

    dataset = FormatDataset(data, MODEL_CKPT)
    dataset_tokenized = dataset.get_tokenized_dataset()

    _train(data, dataset_tokenized, MODEL_CKPT, MODEL_OUTPUT)


def _train(data, dataset_tokenized, model_ckpt, model_output):
    summarizer = SequenceSummarizerTrainer(
        training_args, data, dataset_tokenized, model_ckpt, model_output
    )
    summarizer.train_model()
