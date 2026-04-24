# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase


def get_dataset(
    name: str, tokenizer: PreTrainedTokenizerBase | None = None
) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4", "alpaca", "gsm8k", or "olmo_if".
        tokenizer: Optional tokenizer. Used to render chat-formatted datasets (e.g. olmo_if)
            via apply_chat_template; if omitted or the tokenizer has no chat template,
            a plain "role: content" format is used as a fallback.

    Returns:
        The dataset.
    """
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "allenai--c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
        "gsm8k": {"path": "openai/gsm8k", "config_name": "main"},
        "olmo_if": {"path": "allenai/tulu-3-sft-mixture"},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    # if alpaca, create a test and validation set from the training set
    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    # if gsm8k, combine question+answer into a single text column and create a validation set from the test set
    if name == "gsm8k":
        ds = ds.map(
            lambda x: {"text": x["question"] + "\n" + x["answer"]},
            remove_columns=["question", "answer"],
        )
        temp_ds = ds["test"].train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    # if olmo_if (Tülu-3 SFT mixture), flatten the messages list into a single text column
    # and carve small test/validation splits out of train (dataset ships with train-only).
    if name == "olmo_if":
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            def _render(example):
                return {
                    "text": tokenizer.apply_chat_template(
                        example["messages"], tokenize=False
                    )
                }
        else:
            if tokenizer is not None:
                logging.warning(
                    "olmo_if: tokenizer has no chat_template; falling back to plain 'role: content' format."
                )
            def _render(example):
                return {
                    "text": "\n\n".join(
                        f"{m['role']}: {m['content']}" for m in example["messages"]
                    )
                }
        ds = ds.map(_render, remove_columns=ds["train"].column_names)
        temp_ds = ds["train"].train_test_split(test_size=0.01, seed=42)
        ds["train"] = temp_ds["train"]
        temp_ds = temp_ds["test"].train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds


def prepare_test_dataloader(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048, batch_size: int = 1
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            data_name = ds.column_names[0]
            tokenized_ds = tokenizer("\n\n".join(ds[data_name]), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    logging.info(f"Preparing test dataloader done")
    return loader


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    ntokens: int | None = None,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce. Ignored if ntokens is set.
        ntokens: Total token budget. If set, takes precedence over nsamples: for fixed-seqlen
            packing the number of samples is ntokens // max_seqlen; for varied_seqlen it
            selects rows (in random order) until the cumulative token count reaches ntokens.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """
    logging.info(f"Preparing dataloader")

    if ntokens is not None and not varied_seqlen:
        nsamples = max(1, ntokens // max_seqlen)
        logging.info(
            f"ntokens={ntokens} budget at max_seqlen={max_seqlen} -> nsamples={nsamples}"
        )

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        if len(new_data_list) < nsamples:
            logging.warning(
                f"Could only pack {len(new_data_list)} samples of {max_seqlen} tokens "
                f"(requested {nsamples}). The token budget exceeds what this dataset can provide."
            )
            nsamples = len(new_data_list)

        ds = datasets.Dataset.from_dict({data_name: new_data_list})
    elif ntokens is not None:
        # varied-seqlen path with a token budget: select whole rows in random order
        # until the cumulative token count reaches ntokens.
        torch.manual_seed(seed)
        shuffled = torch.randperm(len(ds)).tolist()
        selected = []
        cum_tokens = 0
        for idx in shuffled:
            selected.append(idx)
            cum_tokens += len(tokenizer.tokenize(ds[idx][data_name]))
            if cum_tokens >= ntokens:
                break
        if cum_tokens < ntokens:
            logging.warning(
                f"Token budget {ntokens} not met with varied_seqlen=True: only {cum_tokens} "
                f"tokens available across {len(selected)} rows."
            )
        ds = ds.select(selected)
        nsamples = len(selected)

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader
