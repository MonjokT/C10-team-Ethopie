# African Folktales SLM — Team Ethopie

**Programme:** TRI AI Saturdays, Cohort 10 (built on Google DeepMind's AI Research Foundations curriculum, in partnership with AI Saturdays Lagos and UCL)
**Competition:** African Folktales SLM Challenge (Kaggle, private community prediction competition)

## Problem statement

The preservation of African folktales, and the cultural memory, values, and oral traditions they carry, is a growing challenge. Manual transcription and scattered anthropological archives are slow and incomplete, and mainstream language models are trained mostly on Western text, so they struggle to generate African stories that feel authentic in dialect and structure. This project explores a small, domain-specific language model that can generate narratives reflecting regional themes and storytelling structures, as a step toward tools for cultural preservation, education, and creative storytelling.

## Dataset

We use the competition-provided synthetic folktale corpus:

- `data/documents.csv` — 24 source folktales, each with `document_id`, `title`, `theme`, `culture_region`, `text`, `origin`, and `license`.
- `data/train_prompts.csv` — 38 prompt → reference-story pairs, each linked to a source document via `document_id`, spanning themes (trickster, origin_myth, moral_tale, hero_journey, animal_fable, community_wisdom) and regions (west, east, southern, central Africa, diaspora).
- `data/test_prompts.csv` — 10 held-out prompts we must generate stories for.
- `data/sample_submission.csv`, `data/baseline_submission.csv` — required output format and a worked example.

The corpus is v1, original synthetic narratives created for this benchmark, not transcriptions of real oral histories. The dataset card in `docs/data_card.pdf` documents provenance and licensing in full.

## Training pipeline

Our notebook (`notebooks/african-folktales-slm.ipynb`) proceeds in stages:

1. **EDA** — load all three CSVs, inspect theme/region balance, word-count distributions, and missing values; drop the unused `source_url` column.
2. **Retrieval baseline** — a TF-IDF (unigram + bigram) + cosine-similarity retriever that, for each test prompt, returns the closest training reference story within the same theme. This needs no training and gives a non-parametric lower bound.
3. **Prompt formatting** — every `train_prompts.csv` row is turned into a Gemma-2 chat-formatted example (`<start_of_turn>user ... <start_of_turn>model ...`), using the matching source document's text as a style cue.
4. **LoRA fine-tuning** — Gemma-2-2B-IT loaded in 4-bit/fp16 on a single Kaggle GPU. Two configurations are compared:
   - *Baseline LoRA*: `r=8`, attention-only targets (`q_proj`, `v_proj`), full-sequence cross-entropy loss (loss computed over prompt tokens too).
   - *Optimized LoRA (used for final submission)*: `r=16`, `alpha=32`, targets attention **and** MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`), and masks prompt tokens with `-100` so gradients only flow through the story tokens. Trained 5 epochs, batch size 1 with gradient accumulation 4, cosine LR schedule, `MAX_SEQ_LENGTH=512`.
5. **Generation** — greedy decoding (`do_sample=False`) with `MAX_NEW_TOKENS=180`, chosen deliberately: the competition's character-level Levenshtein metric penalizes paraphrasing and length inflation, so low-variance decoding and tight length control matter more than creative diversity (see literature-review section in the notebook for the full rationale).

## Evaluation

The competition scores submissions on **mean character-level Levenshtein distance** against hidden reference stories (lower is better). Before every submission we run an integrity check (`Section 8` in the notebook) asserting:
- columns are exactly `PromptId, Story`
- row count and `PromptId` order match `test_prompts.csv`
- no missing or empty-string stories

We compare the retrieval-only baseline against both LoRA configurations on this metric; the masked, higher-rank LoRA setup produced substantially lower training loss (~0.007 vs ~2.19 for the baseline configuration) and is the version used to produce the final `submission.csv`.

## Reproduction

1. Create an environment with the packages in `requirements.txt` (or open the notebook directly on Kaggle, where these are preinstalled).
2. Attach the competition dataset (`african-folktales-slm`) and, for the LoRA sections, the Gemma-2-2B-IT model as a Kaggle Input.
3. Run `notebooks/african-folktales-slm.ipynb` top to bottom:
   - Sections 1–3 load data and run EDA — no GPU required.
   - Section 4 produces the TF-IDF retrieval baseline submission.
   - Section 5 builds the SFT/LoRA training examples.
   - Sections 6–7 load Gemma-2, fine-tune with LoRA (set `USE_LORA = True`), and regenerate `submission.csv` on the fine-tuned model.
   - Section 8 validates the final `submission.csv` before it is submitted.
4. On Kaggle: **Save Version → Save & Run All**, then submit the committed notebook's `submission.csv` via the competition page.

## Appendix

**Team:** Ethopie
**Contributors:** *(add full names / GitHub handles here)*
**Mentors:** *(add mentor name(s) here)*

## References

- Wagner, R. A., and Fischer, M. J. (1974). The string to string correction problem. *Journal of the ACM*, 21(1), 168–173.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707–710.
- Snover, M., Dorr, B., Schwartz, R., Micciulla, L., and Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. *Proceedings of AMTA*, 223–231.
- Devatine, N., and Abraham, L. (2024). Assessing human editing effort on LLM generated texts via compression based edit distance. arXiv preprint. https://arxiv.org/abs/2412.17321
