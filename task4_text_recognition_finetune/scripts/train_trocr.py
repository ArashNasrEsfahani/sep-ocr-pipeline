import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from Levenshtein import distance as levenshtein_distance  # type: ignore[import]

TRANSLATION_TABLE = str.maketrans({
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ئ": "ی",
    "أ": "ا",
    "إ": "ا",
    "آ": "آ",
    "ؤ": "و",
    "ة": "ه",
    "ۀ": "ه",
    "ﻻ": "لا",
    "٠": "۰",
    "١": "۱",
    "٢": "۲",
    "٣": "۳",
    "٤": "۴",
    "٥": "۵",
    "٦": "۶",
    "٧": "۷",
    "٨": "۸",
    "٩": "۹",
})


def _remove_tatweel(text: str) -> str:
    return text.replace("ـ", "")


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.translate(TRANSLATION_TABLE)
    text = _remove_tatweel(text)
    text = " ".join(text.split())
    return text.strip()


def extract_charset(dataset: Dataset, limit: int | None = None) -> List[str]:
    chars: Set[str] = set()
    iterable: Iterable[Dict] = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    for sample in iterable:
        for ch in normalize_text(sample["text"]):
            if ch:
                chars.add(ch)
    return sorted(chars)


def ensure_tokenizer_vocab(processor: TrOCRProcessor, charset: List[str]) -> List[str]:
    tokenizer = processor.tokenizer
    new_tokens: List[str] = []
    for ch in charset:
        if not ch.strip():
            continue
        # Consider a token new if it maps to under-specified representation
        encoded = tokenizer.encode(ch, add_special_tokens=False)
        if not encoded or tokenizer.unk_token_id in encoded:
            new_tokens.append(ch)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
    return new_tokens


class PixelLabelCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([torch.tensor(feature["pixel_values"], dtype=torch.float32) for feature in features])
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        labels_padded[labels_padded == self.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels_padded}


def build_datasets(args: argparse.Namespace, processor: TrOCRProcessor) -> Tuple[Dataset, Dataset]:
    raw_train = load_dataset(args.dataset_name, split="train")
    if args.max_train_samples:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))

    split = raw_train.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    def preprocess(batch: Dict[str, List]) -> Dict[str, np.ndarray]:
        images = [img.convert("RGB") for img in batch["image_path"]]
        texts = [normalize_text(txt) for txt in batch["text"]]
        model_inputs = processor(
            images=images,
            text=texts,
            padding="longest",
            max_length=args.max_label_length,
            return_tensors="np",
        )
        labels = model_inputs["labels"]
        labels[labels == processor.tokenizer.pad_token_id] = -100
        pixel_values = [pv.astype(np.float32) for pv in model_inputs["pixel_values"]]
        labels_list = [lb.astype(np.int64) for lb in labels]
        return {
            "pixel_values": pixel_values,
            "labels": labels_list,
        }

    remove_columns = raw_train.column_names
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        remove_columns=remove_columns,
        num_proc=args.preprocessing_workers,
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        remove_columns=remove_columns,
        num_proc=args.preprocessing_workers,
    )

    return train_dataset, eval_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Parsynth OCR dataset")
    parser.add_argument("--dataset-name", default="hezarai/parsynth-ocr-200k", help="HuggingFace dataset identifier")
    parser.add_argument("--model-name", default="microsoft/trocr-small-printed", help="Pretrained TrOCR checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--max-train-samples", type=int, help="Limit number of training samples")
    parser.add_argument("--max-eval-samples", type=int, help="Limit number of evaluation samples")
    parser.add_argument("--eval-ratio", type=float, default=0.05, help="Validation ratio from training split")
    parser.add_argument("--preprocessing-batch-size", type=int, default=32, help="Batch size for dataset.map preprocessing")
    parser.add_argument("--preprocessing-workers", type=int, default=4, help="Number of workers for preprocessing map")
    parser.add_argument("--max-label-length", type=int, default=64, help="Maximum decoder sequence length")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--charset-out", type=Path, help="Optional path to store extracted charset JSON")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze vision encoder to speed up fine-tuning")
    args = parser.parse_args()

    set_seed(args.seed)

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    charset = extract_charset(load_dataset(args.dataset_name, split="train"), limit=args.max_train_samples)
    new_tokens = ensure_tokenizer_vocab(processor, charset)
    if new_tokens:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = len(processor.tokenizer)
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = args.max_label_length
    model.config.early_stopping = False
    processor.tokenizer.model_max_length = args.max_label_length

    model.generation_config.max_length = args.max_label_length
    model.generation_config.early_stopping = False
    model.generation_config.num_beams = 1

    if args.charset_out:
        args.charset_out.parent.mkdir(parents=True, exist_ok=True)
        args.charset_out.write_text(json.dumps({"charset": charset, "new_tokens": new_tokens}, ensure_ascii=False, indent=2))

    train_dataset, eval_dataset = build_datasets(args, processor)

    data_collator = PixelLabelCollator(processor.tokenizer.pad_token_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.max_label_length,
        fp16=args.fp16,
    report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
    )

    def compute_metrics(eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_str = processor.batch_decode(predictions, skip_special_tokens=True)

        labels = np.array(labels)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        total_chars = 0
        char_errors = 0
        total_words = 0
        word_errors = 0
        exact_matches = 0
        for ref, hyp in zip(label_str, pred_str):
            ref_norm = normalize_text(ref)
            hyp_norm = normalize_text(hyp)
            if ref_norm == hyp_norm:
                exact_matches += 1
            total_chars += len(ref_norm)
            char_errors += levenshtein_distance(ref_norm, hyp_norm)
            ref_words = [w for w in ref_norm.split() if w]
            hyp_words = [w for w in hyp_norm.split() if w]
            total_words += len(ref_words)
            if ref_words:
                word_errors += _word_edit_distance(ref_words, hyp_words)

        cer = (char_errors / total_chars) if total_chars else 0.0
        wer = (word_errors / total_words) if total_words else 0.0
        em = exact_matches / len(label_str) if label_str else 0.0
        return {"cer": cer, "wer": wer, "em": em}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)


def _word_edit_distance(ref: List[str], hyp: List[str]) -> int:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


if __name__ == "__main__":
    main()
