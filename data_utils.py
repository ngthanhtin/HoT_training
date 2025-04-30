from datasets import load_dataset
import re

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def qa_formatter(example):
    # choose a delimiter that was present during SFT pre-processing
    return f"<question>{example['input']}</question>\n<answer>{example['output']}</answer>"

def load_and_preprocess_dataset(no_test_split=True):
    # ds = load_dataset("groundingauburn/HoT")
    ds = load_dataset("groundingauburn/HoT", split="train").shuffle(seed=42)

    # 1️⃣  turn the text column into a ClassLabel
    ds = ds.class_encode_column("dataset_name")        # ⇐ key line

    # 2️⃣  80 / 20 split, still stratified
    split1   = ds.train_test_split(
                    test_size=0.20,
                    stratify_by_column="dataset_name",
                    seed=42)

    train_ds = split1["train"]     # 80 %
    rest_ds  = split1["test"]      # 20 %

    if no_test_split:
        return train_ds, rest_ds, None
    
    # 3️⃣  split that 20 % into val / test  (=10 % each)
    split2   = rest_ds.train_test_split(
                    test_size=0.50,
                    stratify_by_column="dataset_name",
                    seed=42)

    val_ds   = split2["train"]     # 10 %
    test_ds  = split2["test"]      # 10 %
    
    return train_ds, val_ds, test_ds

def process_examples_for_localization_training(batch):
    TAG_RE = re.compile(r"<fact(\d+)>(.*?)</fact\1>", re.DOTALL)
    processed = []

    # i=0
    for question, raw_ans, gt in zip(batch["question"], batch["answer"], batch["gt"]):
        # i += 1
        # if i == 10:
        #     break
        # ---------- pull the clean answer text ------------------------------
        m = re.search(r"Answer:(.*)", raw_ans, re.DOTALL)
        if not m:
            continue
        answer = m.group(1).strip()

        q_words = question.split()
        used_mask = [False]*len(q_words)          # keeps track of tokens already claimed
        fact_pos  = {}                            # fact_id -> (x1,x2)

        def locate_next_span(span_words):
            """Return (x1,x2) of first *unclaimed* occurrence of span_words."""
            L = len(span_words)
            for i in range(len(q_words)-L+1):
                if any(used_mask[i:i+L]):               # skip tokens already assigned
                    continue
                if all(
                    q_words[i+k].strip('.,!?:;()"\'').lower() ==
                    span_words[k].strip('.,!?:;()"\'').lower()
                    for k in range(L)
                ):
                    return i, i+L-1
            return -1, -1

        # --------- rewrite every tag ----------------------------------------
        def repl(match):
            fid, content = match.groups()
            if fid not in fact_pos:                     # first time we see this fact id
                x1, x2 = locate_next_span(content.split())
                if x1 == -1:
                    return match.group(0)              # leave tag unchanged if not found
                fact_pos[fid] = (x1, x2)
                for i in range(x1, x2+1):              # mark tokens as used
                    used_mask[i] = True
            else:
                x1, x2 = fact_pos[fid]

            return f"<fact{fid} x1={x1} x2={x2}>{content}</fact{fid}>"

        new_answer = TAG_RE.sub(repl, answer)

        processed.append({"input": question,
                          "output": new_answer,
                          "gt": gt})
    
    # save the processed data to a file
    # with open("hot_processed_examples.jsonl", "w") as f:
    #     for ex in processed:
    #         f.write(f"{ex}\n")
    
    # exit()

    return processed

def process_examples_for_repeating_training(batch):
    processed = []

    for question, raw_ans, gt in zip(batch["question"], batch["answer"], batch["gt"]):
        processed.append({"input": question,
                          "output": raw_ans,
                          "gt": gt})
    

    return processed