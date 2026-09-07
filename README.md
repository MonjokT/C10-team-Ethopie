# **📖 African Folktales SLM-Preserving Oral Tradition with a Domain-Specific Language Model**

**🌍 Project Overview**
This project fine-tunes a small language model to generate African folktale narratives that stay faithful to regional themes, dialects, and storytelling structure. Using a synthetic folktale benchmark corpus (documents, prompts, and reference stories spanning trickster tales, origin myths, moral tales, hero journeys, animal fables, and community wisdom across West, East, Southern, Central Africa, and the diaspora), we generate short stories from prompts and score them against hidden reference stories using character-level Levenshtein distance.

Mainstream LLMs are trained mostly on Western text and often flatten African dialect and narrative voice. This project is a small-scale proof of concept toward domain-specific models that can support cultural preservation, education, and creative storytelling for communities often left out of mainstream AI development.

**🎯 Objectives**
- Explore and clean the competition's folktale corpus (documents, train and test prompts).
- Build a non-parametric TF-IDF retrieval baseline as a lower bound, with no training required.
- Fine-tune Gemma-2-2B-IT with LoRA adapters to generate stories that closely match reference style and phrasing.
- Compare a standard LoRA setup against an optimized, loss-masked configuration.
- Validate every submission against the competition's required output schema before scoring.

**⚙️ Tools & Libraries**
- Python
- Pandas, Scikit-learn
- PyTorch, Hugging Face Transformers, PEFT, Datasets
- Kaggle Notebooks (GPU runtime)

**📊 Dataset**
Source: TRI AI's African Folktales SLM Challenge (Kaggle), a synthetic benchmark corpus (v1, original synthetic narratives, not real transcriptions).

Key files (in data/):
- documents.csv — 24 source folktales with theme, culture_region, text, origin, and license.
- train_prompts.csv — 38 prompt → reference-story pairs, each linked to a source document.
- test_prompts.csv — 10 held-out prompts to generate stories for.
- sample_submission.csv, baseline_submission.csv — required output format and worked example.

**🧩 Approach Highlights**
- Retrieval baseline: TF-IDF (unigram + bigram) vectorization with cosine similarity, matching each test prompt to the closest training reference story within the same theme.
- Prompt formatting: every training example is converted into a Gemma-2 chat-style prompt (theme, region, prompt, and a style cue drawn from the matching source document).
- Loss masking: prompt tokens are masked with -100 during training so gradients focus entirely on the story text rather than the instruction header.
- Length and decoding control: since Levenshtein distance penalizes both paraphrasing and extra length, generation uses greedy decoding (do_sample=False) with output length capped close to the observed reference-story lengths.

**🤖 Modeling Approach**
Two LoRA configurations were compared:
- Baseline LoRA — rank r=8, attention-only targets (q_proj, v_proj), standard full-sequence cross-entropy loss. Training loss plateaued around ~2.19.
- Optimized LoRA (used for final submission) — rank r=16, alpha=32, targets both attention and MLP projections (q/k/v/o_proj, gate/up/down_proj), with prompt-loss masking, 512-token context, and a cosine learning-rate schedule. Training loss dropped to ~0.0073.

Evaluation metric: mean character-level Levenshtein distance against hidden reference stories (lower is better).

**🛠️ How to Use the Code**
- Attach the competition dataset (african-folktales-slm) and, for the LoRA sections, the Gemma-2-2B-IT model as Kaggle Inputs, or install the packages in requirements.txt to run locally.
- Open notebooks/african-folktales-slm.ipynb and run top to bottom:
  - Sections 1–3 load the data and run EDA (no GPU needed).
  - Section 4 produces the TF-IDF retrieval baseline submission.
  - Sections 5–7 build training examples, load Gemma-2, and fine-tune with LoRA (set USE_LORA = True).
  - Section 8 validates the final submission.csv (correct columns, row count, prompt order, no missing values) before submitting.
- On Kaggle: Save Version → Save & Run All, then submit the committed notebook's submission.csv.

**🧠 Insights & Expected Outcomes**
- A working comparison between a zero-training retrieval baseline and LoRA fine-tuning for closely matching reference narrative style.
- Evidence that prompt-loss masking and full attention+MLP LoRA targeting substantially improve training loss over a narrower baseline LoRA setup.
- A reproducible pipeline that could extend to real oral-history recordings and additional African languages and regions in future work.

**🧭 Repository Structure**
- 📁 data/ — documents.csv, train_prompts.csv, test_prompts.csv, sample/baseline submissions
- 📁 docs/ — Cohort Challenge deliverables (problem statement, data card, impact statement, stakeholder engagement)
- 📁 notebooks/ — african-folktales-slm.ipynb (EDA, baseline, LoRA fine-tuning, submission validation)
- 📁 scripts/ — notes on where standalone scripts would live if the notebook is later split up
- 📄 README.md
- 📄 requirements.txt

**👥 Contributors**
- Team: Ethopie
- Team members:
  1). Iyinoluwa Don-Taiwo(Team Lead)
  2).Monjok Joseph Terem
- Program: TRI AI Saturdays, Cohort 10 (Google DeepMind AI Research Foundations curriculum, in partnership with AI Saturdays Lagos)

**📜 Acknowledgment**
This project was developed as part of TRI AI Saturdays Cohort 10. Thanks to the team, weekly guests,and cohort peers for their guidance throughout the programme.

**🔗 References**
- Wagner, R. A., and Fischer, M. J. (1974). The string to string correction problem. Journal of the ACM, 21(1), 168–173.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707–710.
- Snover, M., Dorr, B., Schwartz, R., Micciulla, L., and Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. Proceedings of AMTA, 223–231.
- Devatine, N., and Abraham, L. (2024). Assessing human editing effort on LLM generated texts via compression based edit distance. arXiv preprint. https://arxiv.org/abs/2412.17321
