# 🧠 Fake vs Real News Classifier — NLP Project

## �� Executive Summary
This project focuses on **text classification** using **Natural Language Processing (NLP)** techniques to detect whether a news headline is **real or fake**.
The dataset contains news articles labeled according to their authenticity.
The objective was to build a model capable of **accurately distinguishing between true and false news**.
After testing multiple models, the **best-performing approach** combined **TF-IDF Vectorization** with a **Naive Bayes classifier**, achieving a **validation accuracy of 93%**.

---

## �� Procedure for model creation :

 <li>Loading the training dataset "training_data_lowercase.csv"</li>
 <li>Preprocesses text (tokenization, stopword removal, stemming, lemmatization)
    <ul>
      <li>Tokenization:  Splits the text into individual words (tokens) so each word can be analyzed separately</li>
      <li>Stopword Removal: Removes common words like “the”, “is”, “and” that don’t carry meaningful information for classification.</li>
      <li>Cleaning:  Eliminates punctuation, numbers, and other non-letter characters to reduce noise.</li>
      <li>Lemmatization: Converts words to their base or dictionary form (e.g., “running” → “run”) to treat similar words the same.</li>
       <li>Stemming:  Reduces words to their root form (e.g., “played” → “play”), further simplifying the vocabulary.</li>
    </ul>
  </li>
  <li>Spliting the data : split the data into training and validation sets.</li>
  <li>Feature Extraction : We prepare two types of features: Bag-of-Words (CountVectorizer) and TF-IDF.</li>
  <li>Trains all four models separately. Four models were developed and compared: Each model was evaluated using training accuracy, validation accuracy, confusion matrices, and F1-scores.
    <ul>
      <li>CountVectorizer + Multinomial Naive Bayes (CV + NB)</li>
      <li>CountVectorizer + Random Forest (CV + RF)</li>
      <li>TF-IDF + Random Forest (TFIDF + RF)</li>
      <li>TF-IDF + Multinomial Naive Bayes (TFIDF + NB)</li>
    </ul>
  </li>
  <li>Model Training</li>
  <li>Evaluation</li>
  <li>Selecting the best performing model</li>
  <li>Evaluation with test dataset</li>
  <li>Generates a final CSV (title, label) predicting 0 = fake and 1 = real</li>
</ul>

---

---

<!-- Badges (images linked) -->
<h3>🧩 Dependencies & Libraries</h3>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img alt="pandas" src="https://img.shields.io/badge/pandas-1.5.3-green?logo=pandas&logoColor=white" />
  <img alt="numpy" src="https://img.shields.io/badge/numpy-1.24.2-blueviolet?logo=numpy&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.2.2-orange?logo=scikitlearn&logoColor=white" />
  <img alt="matplotlib" src="https://img.shields.io/badge/matplotlib-3.7.1-9cf?logo=plotly&logoColor=white" />
  <img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.12.2-lightblue?logo=seaborn&logoColor=white" />
  <img alt="nltk" src="https://img.shields.io/badge/nltk-3.8.1-yellow?logo=python&logoColor=black" />
</p>

<ul>
  <li>Python 3.8 or higher</li>
  <li>pandas == 1.5.3</li>
  <li>numpy == 1.24.2</li>
  <li>scikit-learn == 1.2.2</li>
  <li>matplotlib == 3.7.1</li>
  <li>seaborn == 0.12.2</li>
  <li>nltk == 3.8.1 (for tokenization, stopword removal, and lemmatization)</li>
</ul>

<h3> 🧩 Requirements </h3>
<ul>
  <li>training_data_lowercase.csv</li>
  <li>testing_data_lowercase_nolabels.csv</li>
</ul>

---

<h3> ⚙️  Models & Learning Process </h3>
<li> :one: CountVectorizer + Naive Bayes </li>
Baseline bag-of-words representation.
Fast to train and test, strong initial performance (~92%).
<li> :two: TF-IDF + Random Forest </li>
Weighted term representation using TF-IDF.
Random Forest ensemble for non-linear classification.
Validation accuracy ≈ 92.1%.
<li> :three: TF-IDF + Naive Bayes (Winning Model) </li>
Balanced simplicity and effectiveness.
Captured important term frequency-inverse frequency relationships.
Validation accuracy 93%, F1-score showed strong class balance.

---

## Result Summary 

<section>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Train accuracy</th>
          <th>Validation accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Count vect + Multinomial</td><td>0.94</td><td>0.92</td></tr>
        <tr><td>Count vect + Random Forest</td><td>1.0</td><td>0.9177</td></tr>
        <tr><td>TF-IDF + Random Forest</td><td>1.0</td><td>0.9215</td></tr>
        <tr><td>TF-IDF + Naive Bayes</td><td>0.9418</td><td>0.9308</td></tr>
       <tr><td>Embedding + Logistic regression</td><td>0.88</td><td>0.87</td></tr>
      </tbody>
    </table>
    <p><strong>Note:</strong> Models 2 and 3 show perfect training accuracy, which indicates overfitting. Models 1 and 4 generalize better and were selected.</p>
  </section>
  
---

## Graphs and Visualization (suggested)

> **Count vect + Multinomial**:
> **Count vect + Random Forest**:
> **TF-IDF + Random Forest**:
> **TF-IDF + Naive Bayes**:
> **test_prediction file sample output**:
![image_alt](https://github.com/IshuDhana/Project_2_G5_Natural_Langugae_Processing/blob/c97622298fdf70ea68ba0821f3ce52ca09db66b3/screenshots/confusion_matrix_of_models.png)

![image_alt](https://github.com/IshuDhana/Project_2_G5_Natural_Langugae_Processing/blob/c97622298fdf70ea68ba0821f3ce52ca09db66b3/screenshots/test_prediction_csv_sample_pic.png)

---

<h3> :bar_chart: Evaluation & Learnings </h3>

<h5> Metrics used: </h5>
<li> Accuracy </li>
<li> Confusion Matrix </li>
<li> F1-Score </li>

<h5> What worked: </h5> 
<li> TF-IDF significantly improved text representation quality. </li>
<li> Naive Bayes performed best for short text data. </li>
<li> Proper preprocessing (lemmatization, stopword removal) improved consistency. </li>

<h5> What didn’t: </h5> 

<li> Random Forest required heavy computation and it was overfitting. </li>
<li> Over-cleaning (aggressive stemming) sometimes removed key semantic cues. (u.s)</li>

<h5> Key Learnings: </h5>
<li> Simplicity often wins in text classification tasks. </li>
<li> Vectorization choice (TF-IDF vs Count) impacts model accuracy significantly. </li>
<li> Proper text preprocessing is critical for model performance. </li>

---

<h3>Why we selected Model 1 and Model 4 </h3>

Selected models: Model 1 (CV + NB) and Model 4 (TFIDF + NB) were preferred because they show strong and stable validation performance while avoiding obvious overfitting.

<h3> Why not Model 2 & Model 3: </h3>

Both Random Forest variants achieved perfect (or near-perfect) training accuracy (1.0) while their validation accuracy dropped — a clear sign of overfitting to the training set.

Because Random Forests (as configured) memorized the training data more than they generalized to unseen validation data, they were not chosen as the final models.

<h3> Why NB models were preferred: </h3>

Naive Bayes with bag-of-words or TF-IDF often generalizes better for short headline text and is less prone to overfitting in this setting. They also provide faster training and interpretable behaviour.

---

<h3> ⚙️  Embedding Process </h3>

<li>While embeddings capture richer semantic meaning, they often perform best in context-heavy or numerical tasks (e.g., similarity, clustering, ranking)</li>
<li>For short headline classification, the dataset benefits more from frequency-based representations (TF-IDF), where Naïve Bayes leverages direct term occurrence patterns more effectively than a linear boundary learned from dense embeddings.</li>

<!-- ===========================
     Embedding + Logisctic Regression
     =========================== -->
<h4 id="files"> Embedding + Logisctic Regression</h4>
<table>
  <tr><th>Path</th><th>Description</th></tr>
  <tr><td><code>Training Value</code></td><td>Validation Value</td></tr>
  <tr><td><code>0.88</code></td><td>0.87</td></tr>
</table>
![image_alt](https://github.com/IshuDhana/Project_2_G5_Natural_Langugae_Processing/blob/fb867b9b5681ff13129ef2e18b49cabc6dc47124/screenshots/Embedding_logisticalRegretion.png)


---

<!-- ===========================
     Key Files
     =========================== -->
<h2 id="files">Key Files</h2>
<table>
  <tr><th>Path</th><th>Description</th></tr>
  <tr><td><code>nlp_model_g5_final.ipynb</code></td><td>Main training & evaluation notebook</td></tr>
  <tr><td><code>training_data_lowercase.csv</code></td><td>Training input data</td></tr>
  <tr><td><code>testing_data_lowercase_nolabels.csv</code></td><td>validation dataset</td></tr>
  <tr><td><code>testing_predictions.csv</code></td><td>Prediction generated by validation dataset</td></tr>
 <tr><td><code>Presentation template.pptx</code></td><td>Project Presentation</td></tr>
 <tr><td><code>Embedding </code></td><td>Embedding+LR.ipynb</td></tr>
 <tr><td><code>Graphs & Visualization </code></td><td>Screenshots</td></tr>
</table>

---

<h2> Team Members </h2>
<li> Cristina Insignares </li>
<li> Gulmehak Dutta </li>
<li> Iswarya Malayamaan </li>
<li> Tiago Borges </li>




