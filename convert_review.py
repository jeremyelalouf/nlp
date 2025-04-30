#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def _read_jsonl(path: Path, docs: List[Dict]):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts: List[str] = []
            if obj.get("summary"):
                parts.append(str(obj["summary"]))
            if obj.get("reviewText"):
                parts.append(str(obj["reviewText"]))
            if not parts:
                continue
            obj["_fulltext"] = " ".join(parts)
            docs.append(obj)

def _read_dir(path: Path, docs: List[Dict]):
    for p in path.rglob("*.txt"):
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            docs.append({"reviewerName": p.stem, "_fulltext": txt})

def load_documents(src: Path) -> List[Dict]:
    docs: List[Dict] = []
    if src.is_dir():
        _read_dir(src, docs)
    else:
        _read_jsonl(src, docs)
    if not docs:
        raise RuntimeError("no usable documents")
    return docs

def build_tfidf_and_index(docs: List[Dict], max_vocab: int = 20000) -> Tuple[np.ndarray, Dict[str, int], np.ndarray, Dict[str, Set[int]]]:
    doc_tokens: List[Counter] = []
    total_counts: Counter = Counter()
    for doc in docs:
        counts = Counter(tokenize(doc["_fulltext"]))
        doc_tokens.append(counts)
        total_counts.update(counts)
    vocab_tokens = [t for t, _ in total_counts.most_common(max_vocab)]
    vocab = {t: i for i, t in enumerate(vocab_tokens)}
    N = len(docs)
    V = len(vocab)
    df = np.zeros(V, dtype=int)
    for counts in doc_tokens:
        seen = set()
        for t in counts:
            if t in vocab and t not in seen:
                df[vocab[t]] += 1
                seen.add(t)
    idf = np.log((1 + N) / (1 + df)) + 1.0
    tfidf = np.zeros((N, V))
    for doc_id, counts in enumerate(doc_tokens):
        total = sum(counts.values()) or 1
        for t, c in counts.items():
            if t in vocab:
                i = vocab[t]
                tfidf[doc_id, i] = (c / total) * idf[i]
    index: Dict[str, Set[int]] = defaultdict(set)
    for doc_id, counts in enumerate(doc_tokens):
        for t in counts:
            index[t].add(doc_id)
    return tfidf, vocab, idf, index

def _tokenize_query(expr: str) -> List[str]:
    expr = expr.replace("(", " ( ").replace(")", " ) ")
    raw = expr.split()
    out: List[str] = []
    for tok in raw:
        up = tok.upper()
        if up in ("AND", "OR", "NOT", "(", ")"):
            out.append(up)
        else:
            out.append(tok.lower())
    return out

_PREC = {"NOT": 3, "AND": 2, "OR": 1}

def _infix_to_postfix(tokens: List[str]) -> List[str]:
    output: List[str] = []
    stack: List[str] = []
    for tok in tokens:
        if tok in _PREC:
            while stack and stack[-1] != "(" and _PREC.get(stack[-1], 0) >= _PREC[tok]:
                output.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
        else:
            output.append(tok)
    while stack:
        output.append(stack.pop())
    return output

def _evaluate_postfix(postfix: List[str], index: Dict[str, Set[int]], total_docs: int) -> Set[int]:
    universe = set(range(total_docs))
    st: List[Set[int]] = []
    for tok in postfix:
        if tok == "AND":
            b, a = st.pop(), st.pop()
            st.append(a & b)
        elif tok == "OR":
            b, a = st.pop(), st.pop()
            st.append(a | b)
        elif tok == "NOT":
            a = st.pop()
            st.append(universe - a)
        else:
            st.append(index.get(tok, set()))
    return st[-1] if st else set()

def _apply_lsi(tfidf: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0 or k >= min(tfidf.shape):
        return tfidf, None, None
    U, S, Vt = np.linalg.svd(tfidf, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = Vt[:k, :].T
    return U_k * S_k, S_k, V_k

def _project_query(qvec: np.ndarray, S_k: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    if S_k is None:
        return qvec
    return (qvec @ V_k) / S_k

def _cosine(mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    num = mat @ q
    denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom == 0, 0, num / denom)
    return sims

def _build_qvec(terms: List[str], vocab: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    counts = Counter(terms)
    total = sum(counts.values()) or 1
    v = np.zeros(len(vocab))
    for t, c in counts.items():
        if t in vocab:
            i = vocab[t]
            v[i] = (c / total) * idf[i]
    return v

def search(query: str, docs: List[Dict], tfidf: np.ndarray, vocab: Dict[str, int], idf: np.ndarray, index: Dict[str, Set[int]], k: int, topics: int) -> List[Tuple[int, float]]:
    tokens = _tokenize_query(query)
    postfix = _infix_to_postfix(tokens)
    mids = _evaluate_postfix(postfix, index, len(docs))
    if not mids:
        return []
    terms = [t for t in tokens if t not in ("AND", "OR", "NOT", "(", ")")]
    qvec = _build_qvec(terms, vocab, idf)
    mat = tfidf
    if topics > 0:
        mat, S_k, V_k = _apply_lsi(tfidf, topics)
        qvec = _project_query(qvec, S_k, V_k)
    mids_sorted = sorted(mids)
    sims = _cosine(mat[mids_sorted], qvec) if qvec.any() else np.zeros(len(mids_sorted))
    pairs = sorted(zip(mids_sorted, sims), key=lambda x: x[1], reverse=True)[:k]
    return pairs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="jsonl file or directory")
    ap.add_argument("--query", default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--topics", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=20000)
    args = ap.parse_args()
    docs = load_documents(Path(args.source))
    tfidf, vocab, idf, index = build_tfidf_and_index(docs, args.max_vocab)
    q = args.query or input("query: ")
    res = search(q, docs, tfidf, vocab, idf, index, args.top_k, args.topics)
    for doc_id, score in res:
        doc = docs[doc_id]
        name = doc.get("reviewerName", "-")
        snippet = doc["_fulltext"][:120].replace("
", " ")
        print(f"{score:.4f}	{name}	{snippet}")

if __name__ == "__main__":
    main()

