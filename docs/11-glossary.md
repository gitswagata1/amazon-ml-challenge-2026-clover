# 11 · Glossary

Every technical term used in these docs, in one line each, alphabetical.

| Term | Meaning |
|---|---|
| **A/B test** | Run two versions (A = control, B = change) on the same data and compare scores; keep B only if it wins clearly. |
| **AUC** | Area under the ROC curve: the probability a model ranks a random true match above a random non-match (0.5 = guessing, 1.0 = perfect). |
| **Bagging fraction** | Share of training rows each LightGBM tree sees; randomness that reduces overfitting. |
| **Band (uncertainty band)** | Pairs with stage-1 probability between 0.02 and 0.98, the only ones the expensive re-ranker reads. |
| **Bi-encoder** | A model that turns each text into a vector separately; good for fast nearest-neighbour search. |
| **Binary cross-entropy** | The standard loss for yes/no prediction; punishes confident wrong answers heavily. |
| **Blocking** | Cheaply shortlisting plausible candidates so the expensive matcher never compares all pairs. |
| **Blocking key** | A short string (e.g. `t:trdrs`, `x:srm 12`); two records "meet" if they share one. |
| **Calibration** | How well predicted probabilities match real frequencies (of pairs scored 0.9, are 90 % true?). |
| **Candidate recall** | Share of true matches that appear among the candidates; a hard ceiling on the final score. |
| **Candidate set / pairs** | The (S1, S2/S3) pairs the matcher actually scores; submitted as `candidate_pairs.tsv`. |
| **Consonant skeleton** | A word with vowels removed and similar consonants merged (`marketing` → `mrktng`); robust to typos and transliteration. |
| **Contrastive learning** | Training vectors so that matching items end up close and non-matching items far apart. |
| **Core name** | A business name with legal words (Pvt, Ltd, LLC, SARL, the, and…) removed. |
| **Cosine similarity** | Similarity of two vectors by the angle between them: 1 = same direction, 0 = unrelated. |
| **Cross-encoder** | A transformer that reads both records together and outputs one match score; accurate but slow. |
| **Cross-validation (CV)** | Splitting training data into folds, training on some and scoring the rest, rotating; an honest estimate of performance. |
| **Data augmentation** | Creating extra training examples by modifying real ones (here: synthetic French look-alikes). |
| **DataParallel** | PyTorch's way of splitting each batch across several GPUs. |
| **Decoy** | A different business deliberately made to look like a real one (same name + one word, nearby house number). |
| **Dense retriever** | A bi-encoder fine-tuned so that a query's nearest neighbours are its true matches. |
| **Domain adaptation** | Making a model trained on one data distribution (US/India) work on another (France). |
| **Early stopping** | Stopping training when the validation score stops improving. |
| **EDA** | Exploratory data analysis: studying the data to find patterns that shape the design. |
| **Embedding** | A list of numbers (a vector) representing a piece of text's meaning. |
| **Ensemble / seed averaging** | Averaging several models' predictions (here, 3 LightGBM seeds) for a steadier result. |
| **Entity resolution (ER)** | Deciding which records refer to the same real-world entity. |
| **Epoch** | One full pass of a model over its training data. |
| **F0.5 / F-beta** | A single score combining precision and recall; β = 0.5 weights precision twice as much as recall. |
| **False negative (FN)** | A true match we missed. |
| **False positive (FP)** | A pair we matched that is actually two different businesses (a "false merge"). |
| **Feature** | One number describing a pair (e.g. name similarity) that the model learns from. |
| **Feature engineering** | Designing those numbers from raw data. |
| **Feature importance (gain)** | How much a feature improved the model's splits in total. |
| **Fine-tuning** | Continuing to train a pre-trained model on your own task data. |
| **fp16 / mixed precision** | Using 16-bit numbers on the GPU for speed and memory, with safeguards for accuracy. |
| **Fuzzy matching** | String similarity that tolerates typos and reordering (RapidFuzz `ratio`, `token_set_ratio`…). |
| **Gradient boosting** | Building many small decision trees in sequence, each correcting the previous ones' errors. |
| **GroupKFold** | Cross-validation that keeps all rows of one group (here one S1 business) in the same fold. |
| **Hard negative** | A non-match that looks very much like a match; the most useful kind of training example. |
| **Hyperparameter** | A setting chosen before training (learning rate, number of leaves, trees), not learned from data. |
| **IDF (inverse document frequency)** | `log(N / df)`: rare tokens or keys get high weight, common ones low. |
| **InfoNCE** | The contrastive loss: a softmax that asks the model to pick the true match out of a batch of candidates. |
| **In-batch negatives** | Using other examples in the same training batch as negatives, for free. |
| **Jaccard similarity** | Shared items ÷ all items, for two sets of words. |
| **Jaro-Winkler** | A string similarity that rewards matching prefixes; good for names. |
| **k (top-k)** | How many candidates are kept per S1 (here 20 from keys). |
| **kNN (nearest neighbours)** | Finding the k most similar vectors to a query. |
| **L1 / L2 regularisation** | Penalties that keep a model simpler to reduce overfitting. |
| **LambdaRank** | A learning-to-rank method; tried as a blocking re-ranker and rejected. |
| **Leakage** | Answer information sneaking into training or validation, making scores look better than reality. |
| **Leaderboard (public / private)** | The competition's scoring of part (public) or the rest (private, final) of the test set. |
| **LightGBM** | A fast gradient-boosted-tree library; our main matcher. |
| **Logit** | `log(p / (1 − p))`: a probability stretched onto the whole number line; easier for models to combine. |
| **LOCO** | Leave-one-country-out validation: train on one country, test on another, to simulate an unseen country. |
| **Macro average** | Compute the score per item (per S1), then average, so every item counts equally. |
| **Mean pooling** | Averaging a transformer's per-token vectors into one vector for the text. |
| **Meta-model / stacking** | A small model that learns how to combine other models' predictions. |
| **Normalisation** | Converting text into one standard form (lowercase, expand abbreviations, remove punctuation…). |
| **One-to-one assignment** | Each S2/S3 record goes to at most one S1, its best-scoring one. |
| **Optuna** | A hyperparameter search library (TPE sampler, pruning). |
| **Out-of-fold (OOF)** | A prediction made by a model that did not train on that row; honest for validation and stacking. |
| **Overfitting** | Learning quirks of the training (or public leaderboard) data that don't hold elsewhere. |
| **Parquet** | A compressed, column-based file format that allows reading only the needed columns or rows. |
| **Polars** | A fast DataFrame library (like pandas) with lazy, streaming queries. |
| **Precision** | Of the pairs we predicted, the share that are true. |
| **Probe upload** | A leaderboard submission designed to measure one change (e.g. only French rows differ). |
| **Pseudo-labelling / self-training** | Using a model's own confident predictions on unlabelled data as training labels. |
| **Recall** | Of the true pairs, the share we found. |
| **Re-ranking** | Re-scoring an existing shortlist with a better (more expensive) model. |
| **Singleton** | A Source 1 business with no matches; scores 1.0 only if we predict nothing. |
| **Synthetic data** | Artificially generated examples (here, French look-alikes mimicking the data generator). |
| **TF-IDF** | Vectors where each word or character trigram is weighted by frequency in the text × rarity overall. |
| **Threshold** | The probability above which a pair is accepted as a match (0.58 US/India, 0.91 France). |
| **Token** | A unit of text: a word, or for transformers a word-piece. |
| **Transfer learning** | Starting from a model pre-trained on huge text and adapting it to a specific task. |
| **Transformer** | The neural-network architecture behind modern language models; reads text with attention. |
| **Transliteration** | Writing one script in another's letters (राम → ram). |
| **TSV** | Tab-separated values; like CSV but with tabs. |
| **Validation set** | Data held out from training to measure performance honestly. |

Next: [12 · 10-minute team debrief script →](12-debrief-script.md)
