# Vaccination Sentiment Analysis: A Comparative ML Study

An NLP-based multi-class sentiment classifier designed to categorize social media discourse regarding vaccinations. This project focuses on the trade-offs between different text vectorization strategies and optimizing model generalization through feature engineering.

## 🛠️ Tech Stack
* **ML Framework:** Scikit-Learn
* **NLP Tools:** NLTK (Tokenization, Lemmatization)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib

---

## 🏗️ Methodology & Vectorization
I conducted a comparative analysis of three primary vectorization techniques to transform raw text into numerical features for the **Softmax Regression** (Multinomial Logistic Regression) model.



### 1. Vectorizers Tested
* **CountVectorizer:** Baseline approach counting token frequency.
* **HashingVectorizer:** Memory-efficient vectorization utilizing the hashing trick.
* **TF-IDF:** Normalized counts by inverse document frequency. **Selected as the optimal choice** for maintaining high scores while minimizing the generalization gap.

### 2. Hyperparameter Tuning
* **min_df:** Optimized to `0.001` to filter noise while retaining 1,469 core features.
* **N-grams:** Evaluated `ngram_range=(1,1)` to maintain a manageable feature space.
* **Convergence:** Increased `max_iter` to 5000 to ensure stable gradients in high-dimensional space.

---

## 📊 Experimental Results & Evolution

### Phase 1: Initial Baseline (High Overfitting)
With a raw vocabulary of **32,656 words**, the model exhibited significant overfitting.

<p align="center">
  <img src="images/first_model_with_countvectorizer_scores.png" width="350" />
  <img src="images/first_model_with_countvectorizer.png" width="350" /> 
</p>

### Phase 2: Vectorizer Comparison
Comparing Hashing vs. TF-IDF performance:

**Hashing Vectorizer:**
<p align="center">
  <img src="images/hashingvectorizer_scores.png" width="350" />
  <img src="images/hashingvectorizer.png" width="350" /> 
</p>

**TF-IDF Vectorizer (Top Performer):**
<p align="center">
  <img src="images/tfidfvectorizer_scores.png" width="350" />
  <img src="images/tfidfvectorizer.png" width="350" /> 
</p>

### Phase 3: Final Optimized Model
By applying `min_df=0.001` and custom preprocessing, the vocabulary was pruned to **1,469 words**, significantly stabilizing the learning curve.

<p align="center">
  <img src="images/final_scores.png" width="350" />
  <img src="images/final.png" width="350" /> 
</p>



---

## 🚀 Key Insights & Remarks
* **Generalization:** I observed that smaller, highly-relevant vocabularies reduce the gap between training and validation scores, effectively mitigating overfitting.
* **Preprocessing Limitations:** While URL and punctuation removal cleaned the data, they provided marginal improvements to the F1 score, suggesting that the model relies heavily on core keyword sentiment.
* **Learning Curves:** As the dataset size increased, training scores converged downward—a classic indicator of the model moving from memorization to general pattern recognition.

---
## 🎓 Academic Context
Developed as part of the Artificial Intelligence I course at the National and Kapodistrian University of Athens (UoA). Based on the UC Berkeley CS188 framework.

## 🚦 Execution
```bash
# Install dependencies
pip install pandas scikit-learn nltk numpy matplotlib

# Run the analysis
python main.py
