import re, json, pandas as pd, torch
from collections import Counter
TAG = re.compile(r"<fact(\d+)\s+x1=(\d+)\s+x2=(\d+)>(.*?)</fact\1>")

def extract_tags(text):
    tags = []
    for fid, x1, x2, span in TAG.findall(text):
        tags.append(dict(fid=fid, x1=int(x1), x2=int(x2), span=span.strip()))
    return tags

def span_f1(pred_tags, gold_tags):
    p = Counter((t["fid"], t["x1"], t["x2"]) for t in pred_tags)
    g = Counter((t["fid"], t["x1"], t["x2"]) for t in gold_tags)
    tp = sum((p & g).values())
    fp = sum((p - g).values())
    fn = sum((g - p).values())
    if tp == 0: return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2*prec*rec / (prec+rec)

def numeric_exact(pred_text, gold_text):
    # your tasks end with "{50}" — pull the token in braces
    m_pred = re.search(r"\{(.*?)\}", pred_text)
    m_gold = re.search(r"\{(.*?)\}", gold_text)
    return int(m_pred.group(1)) == int(m_gold.group(1)) if (m_pred and m_gold) else False