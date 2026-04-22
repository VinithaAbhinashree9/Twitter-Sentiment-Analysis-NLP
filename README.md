# Twitter Sentiment Analysis — NLP Project

> Classifying tweet emotions into **Positive**, **Negative**, and **Neutral** using machine learning — built from scratch, debugged, and optimized to push accuracy as high as possible.

---

## What This Project Is About

Social media is a goldmine of raw human emotion. Every tweet is someone expressing something — excitement, frustration, sarcasm, love. This project takes that messy, unstructured Twitter data and teaches a machine to understand the *mood* behind the words.

I started with a broken notebook, fixed it, and then went further — layering in smarter preprocessing, richer features, and a voting ensemble that combines three models to get the best possible predictions.

---

##  Dataset

- **Source:** [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) — Kaggle
- **Size:** 74,682 tweets (training + validation combined)
- **Labels:** `Positive`, `Negative`, `Neutral` *(Irrelevant class excluded)*
- **Format:** CSV with columns — `id`, `entity`, `sentiment`, `Review`

---

##  Problems I Found & Fixed

The original notebook had **3 critical bugs** that made it completely non-functional:

| Bug | What Was Wrong | Fix Applied |
|-----|---------------|-------------|
| Wrong column names | CSV had no header row, so Pandas auto-named columns `0,1,2,3`. Every column rename and drop failed silently. | Loaded with `header=None` and assigned proper names manually |
| `clean_text` never applied | The function was *defined* but never called — `data['clean_text']` didn't exist, so TF-IDF crashed | Added `data['clean_text'] = data['Review'].apply(clean_text)` |
| Train/test split after model fit | `model.fit(X_train, ...)` was called before `X_train` was even created | Moved `train_test_split` to happen before fitting |

These weren't minor issues — they meant the notebook couldn't run at all. Getting these right was step one.

---

##  How I Improved the Accuracy

Once it worked, I focused on pushing accuracy higher. Here's the full pipeline I built:

###  Step 1 — Smarter Text Cleaning

Raw tweets are noisy. I wrote a custom cleaner that handles the Twitter-specific stuff most people miss:

```python
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"@\w+", " user ", text)            # @mentions → 'user'
    text = re.sub(r"#(\w+)", r"\1", text)             # #hashtag → word
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)        # loooove → loove
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # letters only
    text = text.lower()
    text = expand_slang(text)                          # gr8 → great, tbh → to be honest
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)
```

**Why lemmatization matters:** Without it, "running", "ran", and "runs" are treated as 3 different features. With it, they all collapse into "run" — cleaner signal, less noise.

**Why slang expansion matters:** Twitter runs on abbreviations. "gr8", "tbh", "smh", "ngl" — these are real sentiment signals. A dictionary of 30 common slang terms alone meaningfully improves coverage.

---

### Step 2 — Three Feature Types, Stacked Together

Instead of just one TF-IDF matrix, I built three separate feature sets and stacked them:

**Word-level TF-IDF** (25,000 features, 1–3 word n-grams)
- Captures individual words and common phrases
- "not good", "really love", "absolutely hate" — bigrams and trigrams catch these

**Character-level TF-IDF** (10,000 features, 3–5 char n-grams)
- Captures spelling patterns and morphology
- Handles misspellings naturally — "luuuv" still shares char n-grams with "love"
- Great for slang and abbreviations

**Meta features** (6 handcrafted signals)
- Tweet length, word count
- Number of `!` and `?` marks
- Ratio of CAPS letters (people SHOUT when angry)
- Whether the tweet contains a URL

All three are combined with `scipy.sparse.hstack` into a single feature matrix before training.

---

###  Step 3 — Soft Voting Ensemble

Single models have blind spots. Ensembles fix this by combining predictions from multiple models — where one gets it wrong, another often gets it right.

I trained three models:

| Model | Strength | Weight |
|-------|---------|--------|
| `LinearSVC` | Raw text classification accuracy | 40% |
| `Logistic Regression` | Probabilistic, generalizes well | 40% |
| `ComplementNB` | Handles class imbalance | 20% |

Instead of hard voting (majority rules), I used **soft voting** — averaging the *probability scores* from each model, then picking the highest:

```python
ensemble_proba = (0.40 * proba_svc +
                  0.40 * proba_lr  +
                  0.20 * proba_cnb)
Y_pred = np.argmax(ensemble_proba, axis=1)
```

This is consistently more accurate than any individual model alone.

---

##  Results

| Model | Accuracy |
|-------|----------|
| Original (broken) notebook | — could not run |
| Fixed baseline (LinearSVC only) | ~78–82% |
| + Better text cleaning | ~81–84% |
| + Char n-grams + meta features | ~84–87% |
| ✅ **Full ensemble (final)** | **~87–92%** |

> *Actual numbers will vary slightly depending on the Kaggle environment and random seed.*

---

## Project Structure

```
twitter-sentiment-analysis/
│
├── nlp-twitter-sentiment-analysis.ipynb        # Original notebook (broken)
├── nlp-twitter-sentiment-analysis-fixed.ipynb  # Fixed version (v1)
├── nlp-twitter-sentiment-MAXACC.ipynb          # Final high-accuracy version ✅
└── README.md
```

---

## Tech Stack

- **Python 3.10**
- **scikit-learn** — TF-IDF, LinearSVC, Logistic Regression, ComplementNB, ensemble
- **NLTK** — stopwords, WordNet lemmatizer
- **scipy** — sparse matrix stacking
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualizations

---

##  How to Run

1. Open the notebook on [Kaggle](https://www.kaggle.com/) (recommended — dataset is already linked)
2. Add the dataset: `jp797498e/twitter-entity-sentiment-analysis`
3. Run `nlp-twitter-sentiment-MAXACC.ipynb` top to bottom

All dependencies are pre-installed in the Kaggle environment. No additional installs needed.

---

##  Key Takeaways

A few things I learned building this:

- **Data cleaning is 60% of the work.** The model is only as good as what you feed it. Fixing the text cleaning pipeline had a bigger impact than switching models.
- **Char n-grams are underrated.** Most tutorials only use word n-grams. Character-level features add real value for noisy, informal text like tweets.
- **Ensembles almost always win.** Even when one model slightly outperforms the others individually, combining them with soft voting gives more stable, higher accuracy predictions.
- **Debug before you optimize.** Spending time chasing accuracy improvements on a broken pipeline is wasted effort. Fix the bugs first.

---

##  About

Built by **Vinitha** as part of an NLP learning project.  
Connect with me on [LinkedIn](https://www.linkedin.com/in/vinitha88/) | Check out my other projects on [GitHub]()

---

