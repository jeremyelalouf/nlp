# ⚡ Boolean & LSI Information-Retrieval Toolkit

Tiny, dependency-light search utilities that build a TF-IDF (optionally LSI-reduced) index over **Amazon review data** and perform Boolean filtering with cosine-similarity ranking.

| Script | Purpose |
| ------ | ------- |
| `convert_reviews.py` | Convert a single `*.jsonl` dump of Amazon reviews into a neat folder tree of plain-text files (*one review = one `.txt`*). |
| `main.py` | Build the index **on-the-fly** (folder **or** jsonl-file input) and answer Boolean queries. |

---

## ✨ Features

* **Boolean operators** `AND`, `OR`, `NOT` + parentheses – case-insensitive  
* **TF-IDF + Cosine** ranking (🡒 relevant docs first)  
* **Optional LSI** (`--topics K`) via truncated SVD for latent-semantic search  
* **Zero heavy deps** – only NumPy ✔️  
* Designed for one-shot experiments; no persistent index files to manage  

---

## 🛠️ Installation

```bash
git clone https://github.com/<your-handle>/<repo>.git
cd <repo>
python3 -m venv .venv          # optional
source .venv/bin/activate      # optional
pip install numpy python-docx  # docx needed only to regenerate the report
```

*Tested on Python 3.9 + 3.10 (Linux, macOS).*

---

## 🚀 Quick start

### 1 Convert the dataset (once)

```bash
python convert_reviews.py --input reviews.jsonl --outdir reviews_folder
```

Resulting layout (example):

```
reviews_folder/
└── 0449819906/
    ├── 0000000.txt
    ├── 0000001.txt
    └── …
```

*Each sub-directory is an **ASIN**; each file contains `summary + reviewText`.*

### 2 Query

```bash
python main.py reviews_folder        --query "(knit OR stitches) AND NOT (index)"        --top_k 10
```

Sample output:

```
0.4227  Zelmira, Ph.D.   Contains some interesting stitches.
0.3765  Dangerous when Cooking …reversible stitch patterns offered and shown reverse and obverse…
```

---

## 🔧 Command‑line reference

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| **positional** | – | directory (created by *convert*) **OR** `*.jsonl` file |
| `--query` | *stdin* | Boolean expression with `AND`, `OR`, `NOT`, parentheses |
| `--top_k` | `5` | number of results to print |
| `--topics` | `0` | if > 0, apply LSI with *K* latent topics |
| `--max_vocab` | `20000` | keep N most frequent tokens (speed/RAM trade‑off) |

---

## 📊 Performance

| Dataset size | Build + search (M1 Air) |
| ------------ | ----------------------- |
| 5 reviews (sample) | < 0.1 s |
| 100 k reviews | ≈ 10 s |

No RAM‑hungry giant matrices are stored on disk; everything lives in memory for the duration of the run.

---

## 📝 Report

A concise project report (`IR_Toolkit_Report.docx`) lives in the repo root, summarising design choices, usage instructions and a miniature evaluation.

---

## 🤝 Contributing

1. Fork the repo  
2. Create your feature branch (`git checkout -b feat/my-idea`)  
3. Commit your changes (`git commit -am 'Add amazing feature'`)  
4. Push to the branch (`git push origin feat/my-idea`)  
5. Open a Pull Request  

Please keep the dependency list lean and the CLI self-contained.

---

## ⚖️ License

MIT – see [`LICENSE`](LICENSE) for details.

---

## 💬 Contact

Questions, ideas, bug reports → open an issue or ping **@<your-handle>** on GitHub.
