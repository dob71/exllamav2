import torch
import pandas, fastparquet
import os
from safetensors.torch import save_file
import random
from exllamav2.conversion.bot_status import print_stage

def get_tokens(num_rows, length, filename, tokenizer):

    min_tokens = num_rows * length

    df = pandas.read_parquet(filename, engine = "fastparquet")
    df['concatenated'] = df.apply(lambda r: ' '.join([str(v) for v in r.values]), axis = 1)

    all_tokens = torch.empty((1,0), dtype = torch.long)

    for _, row in df['concatenated'].items():
        tokens = tokenizer.encode(row)
        all_tokens = torch.cat((all_tokens, tokens), dim = -1)
        if all_tokens.shape[-1] >= min_tokens: break

    if all_tokens.shape[-1] < min_tokens:
        print(f" ** Warning: Not enough sample data in {filename}")

    all_tokens = all_tokens.flatten()[:min_tokens]
    all_tokens = all_tokens.view((num_rows, length))

    num_print_tokens = 50
    data_sample = all_tokens[0, :num_print_tokens]
    print(f" -- First {num_print_tokens} tokens of dataset:")
    print(f"    {repr(tokenizer.decode(data_sample))}")
    data_sample = all_tokens[-1, -num_print_tokens:]
    print(f" -- Last {num_print_tokens} tokens of dataset:")
    print(f"    {repr(tokenizer.decode(data_sample))}")

    return all_tokens


def tokenize(job, save_fn, tokenizer, measure = False, noise_rows = None, code_factor = 1.0):

    print_stage(job, "Tokenizing (1)" if measure else "Tokenizing (2)", 0, 1)

    cal_ds = job["cal_dataset"]

    if cal_ds is not None:
        rows = job["measurement_rows"] if measure else job["dataset_rows"]
        length = job["measurement_length"] if measure else job["length"]
        cal_tokens = get_tokens(rows, length, cal_ds, tokenizer)
    else:
        cal_tokens = get_standard_calibration(job, measure, tokenizer, noise_rows, code_factor = code_factor)
        if measure:
            job["measurement_rows"] = cal_tokens.shape[0]
        else:
            job["dataset_rows"] = cal_tokens.shape[0]

    cal_filename = os.path.join(job["out_dir"], "cal_data.safetensors")
    cal_dict = { "input_ids": cal_tokens }
    save_file(cal_dict, cal_filename)
    job["cal_filename"] = cal_filename

    print_stage(job, "Tokenizing (1)" if measure else "Tokenizing (2)", 1, 1)


def get_standard_calibration(job, measure, tokenizer, noise_rows = None, code_factor = 1.0):

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_cal_data")
    file_c4 =os.path.join(data_dir, "c4.utf8")
    file_code =os.path.join(data_dir, "code.utf8")
    code_samples_dir =os.path.join(data_dir, "code_samples")
    file_multilingual =os.path.join(data_dir, "multilingual.utf8")
    file_technical =os.path.join(data_dir, "technical.utf8")
    file_wiki = os.path.join(data_dir, "wiki.utf8")
    file_tiny = os.path.join(data_dir, "tiny.utf8")

    # Determine the padding token
    pad_token = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 0

    rows = []
    rows_c4 = 2 if measure else 10
    rows_wiki = 4 if measure else 48
    rows_code = round(3 * code_factor) if measure else round(15 * code_factor)
    rows_tiny = 2 if measure else 10
    rows_multilingual = 3 if measure else 15
    rows_multilingual_s = 1 if measure else 5
    rows_technical = round(2 * code_factor) if measure else round(10 * code_factor)
    rows_random = 2
    if noise_rows is not None:
        rows_noise = noise_rows[0] if measure else noise_rows[1]
    else:
        rows_noise = 0

    ctx = job["measurement_length"] if measure else job["length"]
    # For datasets that support long context use it as is, for the rest use 2048
    short_ctx = ctx if ctx <= 2048 else 2048

    # C4: 10 rows

    r_prev = 0
    with open(file_c4, "r", encoding="utf8") as f:
        lines = f.readlines()

    text = "\n\n".join(lines)
    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % short_ctx)]
    tokenized_rows = tokens.view(-1, short_ctx)

    for i in range(rows_c4):
        rows.append(tokenized_rows[i:i+1])
    print(f"C4: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Wiki: 24 aligned rows + 24 aligned rows with BOS

    with open(file_wiki, "r", encoding="utf8") as f:
        text = f.read()

    articles = [a[a.find("\n") + 1:] for a in text.split("</doc>\n")]
    tokenized_articles = [tokenizer.encode(a, add_bos = True, add_eos = True) for a in articles]

    idx = 0
    r = 0
    while r < rows_wiki:
        length = 0
        idx0 = idx
        while length < short_ctx + 1 and idx < len(tokenized_articles):
            length += tokenized_articles[idx].shape[-1]
            idx += 1
        if idx0 == idx:
            print(f"-- Run out of rows in {file_wiki} at row {idx}, repeating")
            idx = 0
            continue
        row = torch.cat(tokenized_articles[idx0 : idx], dim = -1)
        if r < rows_wiki // 2: row = row[:, 1:short_ctx+1]
        else: row = row[:, :short_ctx]
        rows.append(row)
        r += 1

    print(f"Wiki: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Code: 15 rows

    if ctx <= 2048: # for short context use the built in set
        with open(file_code, "r", encoding="utf8") as f:
            text = f.read()
    
        tokens = tokenizer.encode(text)
        tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % ctx)]
        tokenized_rows = tokens.view(-1, ctx)
        if len(tokenized_rows) < rows_code:
             print(f"-- Run out of rows in {file_code} at row {len(tokenized_rows)}, repeating")
    
        for i in range(rows_code):
            r = i % len(tokenized_rows)
            rows.append(tokenized_rows[r:r+1])
    else:
        txt_files = [f for f in os.listdir(code_samples_dir) if f.endswith('.txt')]
        for i in range(rows_code):
            selected_file = random.choice(txt_files)
            file_path = os.path.join(code_samples_dir, selected_file)
            with open(file_path, "r", encoding="utf8") as f:
                text = f.read()
            tokens = tokenizer.encode(text)
            if tokens.shape[-1] > ctx:
                tokens = tokens[:,:ctx]
            rows.append(tokens)

    print(f"Code {ctx}: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Tinystories: 5 aligned rows + 5 aligned rows with BOS

    with open(file_tiny, "r", encoding="utf8") as f:
        text = f.read()

    articles = text.split("<|endoftext|>")
    tokenized_articles = [tokenizer.encode(a.strip(), add_bos = True, add_eos = True) for a in articles]

    idx = 0
    r = 0
    while r < rows_tiny:
        length = 0
        idx0 = idx
        while length < short_ctx + 1 and idx < len(tokenized_articles):
            length += tokenized_articles[idx].shape[-1]
            idx += 1
        if idx0 == idx:
            print(f"-- Run out of rows in {file_tiny} at row {idx}, repeating")
            idx = 0
            continue
        row = torch.cat(tokenized_articles[idx0 : idx], dim = -1)
        if r < rows_tiny // 2: row = row[:, 1:short_ctx+1]
        else: row = row[:, :short_ctx]
        rows.append(row)
        r += 1

    print(f"Tiny stories: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Multilingual: 15 rows + 5 shuffled rows

    with open(file_multilingual, "r", encoding="utf8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % short_ctx)]
    tokenized_rows = tokens.view(-1, short_ctx)

    for i in range(rows_multilingual):
        rows.append(tokenized_rows[i:i+1])

    tokenized_rows = tokens.view(-1, 128)
    random.seed(69420)
    for i in range(rows_multilingual_s):
        row = []
        for j in range(short_ctx // 128):
            k = random.randint(0, tokenized_rows.shape[0] - 1)
            row.append(tokenized_rows[k].unsqueeze(0))
        rows.append(torch.cat(row, dim = -1))

    print(f"Multilingual: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Randomized: 2 rows

    vocab_size = tokenizer.get_vocab_size()
    random.seed(69420)
    for i in range(rows_random):
        row = torch.randint(0, vocab_size, (1, short_ctx), dtype = torch.long)
        rows.append(row)

    print(f"Random tokens: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Technical: 10 rows

    with open(file_technical, "r", encoding="utf8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = tokens[:, : tokens.shape[-1] - (tokens.shape[-1] % short_ctx)]
    tokenized_rows = tokens.view(-1, short_ctx)

    rows_count = min(tokenized_rows.shape[0], rows_technical)
    if rows_count < rows_technical:
        print(f"-- WARNING: Run out of rows in {file_technical}, wanted {rows_technical}, got {rows_count}")
    for i in range(rows_count):
        rows.append(tokenized_rows[i:i+1])

    print(f"Technical: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # Noise: 30 rows

    for i in range(rows_noise):
        rows.append(torch.neg(torch.ones_like(rows[-1])))

    print(f"Noise: {r_prev} - {len(rows)}")
    r_prev = len(rows)

    # for idx, r in enumerate(rows):
    #     print("------------------------------------------------------------------------------")
    #     print(idx)
    #     print("--------")
    #     print(tokenizer.decode(r))

    # Pad rows that are shorter than ctx
    for i in range(len(rows)):
        row = rows[i]
        if row.shape[-1] < ctx:
            pad_length = ctx - row.shape[-1]
            pad_tensor = torch.full((1, pad_length), pad_token, dtype=row.dtype)
            rows[i] = torch.cat([row, pad_tensor], dim=-1)

    return torch.cat(rows, dim = 0)
