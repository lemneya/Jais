#!/usr/bin/env python3
"""
Jais Fine-Tuning Script for Mac (Apple Silicon MPS)
====================================================
Fine-tune Jais-13b on Hassaniya dialect data using LoRA with MPS backend.
Optimized for MacBook Pro M4 with 36GB Memory.

Usage:
    python3 jais_mps_finetune.py --epochs 3
"""

import os
import sys
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Jais on Hassaniya (Mac MPS)")
    parser.add_argument("--model-path", type=str,
                        default="/Users/mohiyidinecheikh/models/jais/jais-13b",
                        help="Path to Jais model")
    parser.add_argument("--data-path", type=str,
                        default="/Users/mohiyidinecheikh/models/jais/hassania-qwen-finetune/hdrp/data/processed/exports/sft/sft_hassaniya_v2.jsonl",
                        help="Path to training data")
    parser.add_argument("--output-dir", type=str,
                        default="/Users/mohiyidinecheikh/models/jais/jais-hassaniya-lora",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size (keep small for memory)")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples (for testing)")
    return parser.parse_args()


JAIS_SYSTEM_PROMPT = "اسمك مساعد حسانية، متخصص في اللهجة الحسانية الموريتانية."


def format_jais_prompt(user_message: str, assistant_response: str = None) -> str:
    """Format conversation in Jais prompt format."""
    prompt = f"### Instruction: {JAIS_SYSTEM_PROMPT}\n\n"
    prompt += f"أكمل المحادثة:\n"
    prompt += f"### Input: [|Human|] {user_message}\n"
    prompt += f"### Response: [|AI|]"
    if assistant_response:
        prompt += f" {assistant_response}"
    return prompt


def load_and_prepare_data(data_path: str, max_samples: int = None):
    """Load HDRP data and convert to Jais format."""
    print(f"Loading data from {data_path}...")

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                messages = item.get('messages', [])

                user_msg = ""
                assistant_msg = ""

                for msg in messages:
                    if msg['role'] == 'user':
                        user_msg = msg['content']
                    elif msg['role'] == 'assistant':
                        assistant_msg = msg['content']

                if user_msg and assistant_msg:
                    text = format_jais_prompt(user_msg, assistant_msg)
                    data.append({"text": text})

            except json.JSONDecodeError:
                continue

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} training examples")
    return data


def main():
    args = parse_args()

    print("=" * 60)
    print("JAIS FINE-TUNING FOR HASSANIYA (Mac MPS)")
    print("=" * 60)

    # Check dependencies
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install with: pip install torch transformers peft datasets")
        sys.exit(1)

    # Use CPU for training (MPS has memory buffer limits for large models)
    # M4 with 36GB RAM can handle 13B model in float16 on CPU
    device = "cpu"
    print(f"Using CPU with unified memory (36GB)")
    print("Training will be slower but memory-efficient")

    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Max length: {args.max_length}")

    # Load data
    train_data = load_and_prepare_data(args.data_path, args.max_samples)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float16 to save memory
    print(f"\nLoading model in float16 (this may take a few minutes)...")
    print("Memory required: ~26GB")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    print(f"Model loaded. Parameters: {model.num_parameters() / 1e9:.1f}B")

    # Configure LoRA
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj", "c_fc"],  # Jais attention/MLP modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = Dataset.from_list(train_data)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length',
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
    )

    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,  # MPS doesn't support fp16 training well
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=2,
        dataloader_pin_memory=False,  # Required for MPS
        use_mps_device=True if device == "mps" else False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("This will take several hours on M4 Max...")
    print("Monitor memory usage in Activity Monitor")

    # Train
    trainer.train()

    # Save
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nLoRA adapter saved to: {final_dir}")
    print("\nTo test:")
    print(f"  python3 test_jais_hassaniya.py --model {args.model_path} --adapter {final_dir}")


if __name__ == "__main__":
    main()
