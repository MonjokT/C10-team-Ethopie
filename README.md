# **📖 African Folktales SLM-Preserving Oral Tradition with a Domain-Specific Language Model**

## **🌍 Project Overview**

This project fine-tunes a small language model to generate African folktale narratives that stay faithful to regional themes, dialects, and storytelling structure. Using a synthetic folktale benchmark corpus (documents, prompts, and reference stories spanning trickster tales, origin myths, moral tales, hero journeys, animal fables, and community wisdom across West, East, Southern, Central Africa, and the diaspora), we generate short stories from prompts and score them against hidden reference stories using character-level Levenshtein distance.

Mainstream LLMs are trained mostly on Western text and often flatten African dialect and narrative voice. This project is a small-scale proof of concept toward domain-specific models that can support cultural preservation, education, and creative storytelling for communities often left out of mainstream AI development.



## **🎯 Objectives**

- Explore and clean the competition's folktale corpus (documents, train and test prompts).
- Build a non-parametric TF-IDF retrieval baseline as a lower bound, with no training required.
- Fine-tune Gemma-2-2B-IT with LoRA adapters to generate stories that closely match reference style and phrasing.
- Compare a standard LoRA setup against an optimized, loss-masked configuration.
- Validate every submission against the competition's required output schema before scoring.

## **⚙️ Tools & Libraries**

- Python
- Pandas, Scikit-learn
- PyTorch, Hugging Face Transformers, PEFT, Datasets
- Kaggle Notebooks (GPU runtime)

## **📊 Dataset**

Source: TRI AI's African Folktales SLM Challenge (Kaggle), a synthetic benchmark corpus (v1, original synthetic narratives, not real transcriptions).

Key files (in data/):
- documents.csv — 24 source folktales with theme, culture_region, text, origin, and license.
- train_prompts.csv — 38 prompt → reference-story pairs, each linked to a source document.
- test_prompts.csv — 10 held-out prompts to generate stories for.
- sample_submission.csv, baseline_submission.csv — required output format and worked example.



## **📌 Scope Note:


Our original plan — documented in `docs/problem_statement.docx`, `docs/Data_card.pdf`, `docs/Stakeholder_Engagement_Plan.pdf`, and `docs/impact_statement.pdf` — was to build this model on real-world oral history recordings, community-contributed folktales, and published regional literature, with active stakeholder governance (a Community Review Board of elders, educators, and cultural guardians).

Given the timeline and lack of access to community partners during Cohort 10, we used the **TRI AI Saturdays / Kaggle-provided synthetic benchmark corpus** instead (see [Dataset](#-dataset) above). The stakeholder engagement plan, data sourcing/labeling strategy, and governance model in `docs/` reflect the team's discussions about how this project *should* be run if scaled with real community data — they were not carried out in this phase.

In short:
- **What we planned:** real-world data sourcing, community labeling and verification, elder/educator governance with veto power over outputs.
- **What we built:** a fine-tuning exercise on a synthetic, pre-labeled competition dataset, evaluated automatically via Levenshtein distance — no human or community review step.

The `docs/` folder should be read as the target design for a future, fully community-grounded version of this project, not as a description of this submission's dataset or process.



## **🧩 Approach Highlights**

- Retrieval baseline: TF-IDF (unigram + bigram) vectorization with cosine similarity, matching each test prompt to the closest training reference story within the same theme.
- Prompt formatting: every training example is converted into a Gemma-2 chat-style prompt (theme, region, prompt, and a style cue drawn from the matching source document).
- Loss masking: prompt tokens are masked with -100 during training so gradients focus entirely on the story text rather than the instruction header.
- Length and decoding control: since Levenshtein distance penalizes both paraphrasing and extra length, generation uses greedy decoding (do_sample=False) with output length capped close to the observed reference-story lengths.

## **🤖 Modeling Approach**

Two LoRA configurations were compared:
- Baseline LoRA — rank r=8, attention-only targets (q_proj, v_proj), standard full-sequence cross-entropy loss. Training loss plateaued around ~2.19.
- Optimized LoRA (used for final submission) — rank r=16, alpha=32, targets both attention and MLP projections (q/k/v/o_proj, gate/up/down_proj), with prompt-loss masking, 512-token context, and a cosine learning-rate schedule. Training loss dropped to ~0.0073.

Evaluation metric: mean character-level Levenshtein distance against hidden reference stories (lower is better).

## **🛠️ How to Use the Code**
- Attach the competition dataset (african-folktales-slm) and, for the LoRA sections, the Gemma-2-2B-IT model as Kaggle Inputs, or install the packages in requirements.txt to run locally.
- Open notebooks/african-folktales-slm.ipynb and run top to bottom:
  - Sections 1–3 load the data and run EDA (no GPU needed).
  - Section 4 produces the TF-IDF retrieval baseline submission.
  - Sections 5–7 build training examples, load Gemma-2, and fine-tune with LoRA (set USE_LORA = True).
  - Section 8 validates the final submission.csv (correct columns, row count, prompt order, no missing values) before submitting.
- On Kaggle: Save Version → Save & Run All, then submit the committed notebook's submission.csv.

## **🧠 Insights & Expected Outcomes**

- A working comparison between a zero-training retrieval baseline and LoRA fine-tuning for closely matching reference narrative style.
- Evidence that prompt-loss masking and full attention+MLP LoRA targeting substantially improve training loss over a narrower baseline LoRA setup.
- A reproducible pipeline that could extend to real oral-history recordings and additional African languages and regions in future work.

## **🧭 Repository Structure**

- 📁 data/ — documents.csv, train_prompts.csv, test_prompts.csv, sample/baseline submissions
- 📁 docs/ — Cohort Challenge deliverables (problem statement, data card, impact statement, stakeholder engagement)
- 📁 notebooks/ — african-folktales-slm.ipynb (EDA, baseline, LoRA fine-tuning, submission validation)
- 📁 scripts/ — notes on where standalone scripts would live if the notebook is later split up
- 📄 README.md
- 📄 requirements.txt

## **🧗 Debugging & Engineering Notes**

Getting from a working notebook to a working *Kaggle submission* surfaced several environment issues worth documenting for future cohorts attempting similar fine-tuning tasks on Kaggle's free GPU tier:

- **Multi-GPU silently breaks single-GPU training code.** Kaggle's T4 x2 accelerator caused repeated CUDA out-of-memory errors during `Trainer.train()`, even after correctly setting `device_map={"": 0}` at model load time. The fix required setting `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` as the very first line executed in the session — `device_map` alone doesn't prevent `Trainer` from initializing multi-GPU data-parallel training.
- **GPU memory doesn't clear on a failed run.** Re-running a cell after a CUDA OOM error fails identically every time, since the crashed allocation stays resident. A full session restart is required before every retry, not just re-executing the failing cell.
- **Model variant selection matters.** Kaggle's model search surfaces near-identical listings (e.g. `gemma-2-2b-it` vs. `gemma-2-2b-jpn-it`, a Japanese-tuned variant) that load without error but are wrong for the task — worth explicitly verifying the attached model path before training.
- **Offline-mode constraints.** Competition notebooks run with internet disabled during the committed "Save & Run All" pass, which ruled out `pip install`-ing extra dependencies (e.g. `trl`, `bitsandbytes`) at submission time — we adapted training code to rely only on packages preinstalled in Kaggle's base image.
- **Retrieval-matching granularity affects output diversity.** An early version of the generation loop matched style-cue documents by theme only, causing several test prompts sharing a theme to receive identical style cues and, under greedy decoding, identical output text. Switching to per-prompt TF-IDF similarity matching (theme + region first, falling back to theme alone) resolved the duplication.

## **👥 Contributors**

- Team: Ethopie
  
- Mentor: Mr. Patrick Owor
  
- Team members:
  
  1). Iyinoluwa Don-Taiwo(Team Lead)
  2).Monjok Joseph Terem
  3).Cheikh kandji
  4).Hammad
  5).Christian Obichukwu
  6). Umar Adamu Hussaini
  7). Emmanuel Chinecherem Nwankwo
  8).Tijani Fawaz
  9).Odey Divine

- Program: TRI AI Saturdays, Cohort 10 (Google DeepMind AI Research Foundations curriculum, in partnership with AI Saturdays Lagos)


## **PROJECT SUMMARY**

In this project, two approaches were used to see the effect of fine-tuning on the Levenshtein distance score: a simple retrieval baseline(i.e,no training at all) and a LoRA fine-tuned model trained on our 38 example stories. A baseline score of 189.75 was achieved on the retrieval approach because it works by picking the most similar existing story we already had and reusing it wholesale, rather than writing something new for the specific prompt given.

To make this concrete using our own dataset: for the test prompt "Hyena jumps at the moon in water.", the baseline searched train_prompts.csv for the closest matching prompt and simply copied over its reference_story:

"Hyena saw the moon in a still pond and leapt to seize it, soaking his muzzle in mud. Owl hooted that some lights are for watching, not eating. Hyena laughed, washed, and guarded the pond so calves could drink by starlight. Knowing what you cannot catch is also a kind of feast."

This story is a real, complete folktale — but it wasn't written for this exact prompt, it was just the closest thing we already had lying around. Since Levenshtein distance counts every letter-level difference between our answer and the hidden correct one, reusing an unrelated (if similar) story naturally racks up a high edit count, hence the 189.75 average across our 10 test prompts.

This relates directly to our topic because it shows why fine-tuning matters here: a model that only retrieves existing text has no way to adjust its wording to match a new, unseen prompt. Fine-tuning, even on a small set of 38 examples, gives the model a chance to actually generate text tailored to each specific prompt instead of recycling something similar, which is the whole point of trying to preserve and generate authentic folktale narratives rather than just retrieving old ones.

## **📜 Acknowledgment**

This project was developed as part of TRI AI Saturdays Cohort 10. Thanks to the team, weekly guests,and cohort peers for their guidance throughout the programme.

## **🔗 References**

- Wagner, R. A., and Fischer, M. J. (1974). The string to string correction problem. Journal of the ACM, 21(1), 168–173.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707–710.
- Snover, M., Dorr, B., Schwartz, R., Micciulla, L., and Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. Proceedings of AMTA, 223–231.
- Devatine, N., and Abraham, L. (2024). Assessing human editing effort on LLM generated texts via compression based edit distance. arXiv preprint. https://arxiv.org/abs/2412.17321
  
-TRI AI Saturdays (2026). Cohort 10 Project Requirements and Structure. AI Saturdays Lagos, in partnership with Google DeepMind's AI Research Foundations curriculum and University College London. https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html
