import os
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

class MultiTaskDistilBert(nn.Module):
    def __init__(
        self,
        num_ideology_labels: int,
        num_factuality_labels: int,
        class_weights_ideo: torch.Tensor,
        class_weights_fact: torch.Tensor,
        alpha: float = 0.5848983419474759,
        beta: float = 0.4151016580525241
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        self.ideology_head   = nn.Linear(hidden_size, num_ideology_labels)
        self.factuality_head = nn.Linear(hidden_size, num_factuality_labels)
        self.dropout = nn.Dropout(0.1)

        self.loss_fct_ideo = nn.CrossEntropyLoss(weight=class_weights_ideo)
        self.loss_fct_fact = nn.CrossEntropyLoss(weight=class_weights_fact)

        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        input_ids,
        attention_mask,
        ideology_label=None,
        factuality_label=None
    ):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])

        ideology_logits   = self.ideology_head(pooled)
        factuality_logits = self.factuality_head(pooled)

        loss = None
        if ideology_label is not None and factuality_label is not None:
            loss_ideo = self.loss_fct_ideo(ideology_logits, ideology_label)
            loss_fact = self.loss_fct_fact(factuality_logits, factuality_label)
            loss = self.alpha * loss_ideo + self.beta * loss_fact

        return {
            "loss": loss,
            "ideology_logits": ideology_logits,
            "factuality_logits": factuality_logits
        }

def evaluate(model, loader, device):
    model.eval()
    all_ideo_preds, all_ideo_labels = [], []
    all_fact_preds, all_fact_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, ideo_lbl, fact_lbl in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            out = model(input_ids, attention_mask)
            preds_ideo = out["ideology_logits"].argmax(dim=-1).cpu().tolist()
            preds_fact = out["factuality_logits"].argmax(dim=-1).cpu().tolist()

            all_ideo_preds  += preds_ideo
            all_fact_preds  += preds_fact
            all_ideo_labels += ideo_lbl.tolist()
            all_fact_labels += fact_lbl.tolist()

    return {
        "ideo_acc": accuracy_score(all_ideo_labels, all_ideo_preds),
        "fact_acc": accuracy_score(all_fact_labels, all_fact_preds),
        "ideo_f1":  f1_score(all_ideo_labels, all_ideo_preds, average="macro"),
        "fact_f1":  f1_score(all_fact_labels, all_fact_preds, average="macro"),
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load & inspect data
    df = pd.read_csv(args.data_file)
    print("Loaded DataFrame:", df.shape)
    print(df.head(), "\n")
    df = df.dropna(subset=['text', 'ideology', 'factuality']).reset_index(drop=True)
    print("After dropna:", df.shape)
    print("Ideology distribution:\n", df['ideology'].value_counts(normalize=True))
    print("Factuality distribution:\n", df['factuality'].value_counts(normalize=True), "\n")

    # 2) Split: train/val/test stratified on ideology
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df['ideology'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1765, stratify=train_val_df['ideology'], random_state=42
    )
    print(f"Splits → Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}\n")

    # 3) Label encoding
    ideology_enc   = LabelEncoder().fit(train_df['ideology'])
    factuality_enc = LabelEncoder().fit(train_df['factuality'])

    train_df['ideo_lbl'] = ideology_enc.transform(train_df['ideology'])
    val_df['ideo_lbl']   = ideology_enc.transform(val_df['ideology'])
    test_df['ideo_lbl']  = ideology_enc.transform(test_df['ideology'])

    train_df['fact_lbl'] = factuality_enc.transform(train_df['factuality'])
    val_df['fact_lbl']   = factuality_enc.transform(val_df['factuality'])
    test_df['fact_lbl']  = factuality_enc.transform(test_df['factuality'])

    num_ideo_labels   = len(ideology_enc.classes_)
    num_fact_labels   = len(factuality_enc.classes_)

    # 4) Compute class weights
    ideo_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_ideo_labels),
        y=train_df['ideo_lbl'].to_numpy()
    )
    fact_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_fact_labels),
        y=train_df['fact_lbl'].to_numpy()
    )
    ideo_weights = torch.tensor(ideo_weights, dtype=torch.float).to(device)
    fact_weights = torch.tensor(fact_weights, dtype=torch.float).to(device)
    print("Computed class weights.")

    # 5) Tokenizer & DataLoaders
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def make_dataset(df_split, lbl_cols):
        enc = tokenizer(
            df_split['text'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return TensorDataset(
            enc['input_ids'],
            enc['attention_mask'],
            torch.tensor(df_split[lbl_cols[0]].tolist(), dtype=torch.long),
            torch.tensor(df_split[lbl_cols[1]].tolist(), dtype=torch.long)
        )

    train_ds = make_dataset(train_df, ['ideo_lbl', 'fact_lbl'])
    val_ds   = make_dataset(val_df,   ['ideo_lbl', 'fact_lbl'])
    test_ds  = make_dataset(test_df,  ['ideo_lbl', 'fact_lbl'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # 6) Model, optimizer, scheduler
    model = MultiTaskDistilBert(
        num_ideology_labels   = num_ideo_labels,
        num_factuality_labels = num_fact_labels,
        class_weights_ideo    = ideo_weights,
        class_weights_fact    = fact_weights,
        alpha=args.alpha,
        beta=1.0-args.alpha
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 7) Training loop with progress bar
    print("\nStarting training...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in loop:
            input_ids, attention_mask, ideo_lbl, fact_lbl = [t.to(device) for t in batch]
            optimizer.zero_grad()
            out = model(input_ids, attention_mask, ideology_label=ideo_lbl, factuality_label=fact_lbl)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg training loss: {avg_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(
            f" → Val Ideo Acc {val_metrics['ideo_acc']:.4f}, F1 {val_metrics['ideo_f1']:.4f} |"
            f" Val Fact Acc {val_metrics['fact_acc']:.4f}, F1 {val_metrics['fact_f1']:.4f}\n"
        )

    # 8) Final test evaluation
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test Ideo Acc {test_metrics['ideo_acc']:.4f}, F1 {test_metrics['ideo_f1']:.4f} |"
        f" Test Fact Acc {test_metrics['fact_acc']:.4f}, F1 {test_metrics['fact_f1']:.4f}"
    )

    # 9) Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "model")
    tok_dir   = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    tokenizer.save_pretrained(tok_dir)
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, "model_state_dict.pt")
    )
    # Also save the DistilBERT config so you can rehydrate the encoder
    model.bert.config.save_pretrained(model_dir)

    pickle.dump(ideology_enc,   open(os.path.join(args.output_dir, "ideo_encoder.pkl"), "wb"))
    pickle.dump(factuality_enc, open(os.path.join(args.output_dir, "fact_encoder.pkl"), "wb"))

    print(f"\nArtifacts saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Moderately")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to babe_polibias_combined_labeled.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model & tokenizers"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=7,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2.16411857404266e-05,
        help="Learning rate"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5848983419474759,
        help="Task weight for ideology loss (beta = 1-alpha)"
    )
    args = parser.parse_args()
    main(args)
