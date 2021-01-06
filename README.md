# hypernymy extraction
Code for msbd5014/Fa20-Independent project

> "Hypernymy Extraction: Extensive Exploration of Several Methods" ([report](https://github.com/WillaFan/hypernymy-extraction/blob/master/_FILES_/5014report.pdf))

This project is conducted under the topic of taxonomy learning in Natural Language Processing (NLP). In the project, I mainly made an attempt to extract hypernymy relation via two directions, pattern based (mostly unsupervised) and learning based (mostly supervised). Pattern based methods start from manual hearst patterns, so that techniques like PPMI and SVD are applied for further detection. At the same time, distributional methods based on several hypernymy similarity measures are also explored to compare with those pattern based ones. In supervised learning models, I mainly explored term embedings for SVM hypernymy classification, projection learning and Bi-LSTM sequence labeling models. Evaluation results of those learning models all achieve an f1 score higher than 0.7.

# Requirements
Python 3 is required (3.7 is preferred). <p>
**Dependent packages** include `numpy`, `pandas`, `matplotlib`, `sklearn`, `scipy`, `pytorch`, `gensim`, `nltk`. <p>
`commands.txt` gives the list of commands about how to start.

# Demo
See major hypernymy extraction results under ./Results folder, but only evaluation results are shown in report. <p>
Part of evaluation scores are shown as follows. <p>
  <img src="https://github.com/WillaFan/hypernymy-extraction/blob/master/_FILES_/pic_dih.png" width="550" alt="dih"/>
  <img src="https://github.com/WillaFan/hypernymy-extraction/blob/master/_FILES_/pic_termEmbed.png" width="550" alt="termEmbed"/>
  <img src="https://github.com/WillaFan/hypernymy-extraction/blob/master/_FILES_/pic_proj.png" width="550" alt="proj"/>
  
# Performance
The code covers six parts: a. **Manual hearst patterns**;  b. **Pattern-based methods**;  c. **Distributional methods**;
			      d. **Term embeddings**;  e. **Projection learning**;  f. **Bi-LSTM sequence labeling**.

Each module mainly consists of data, preprocess, model, evaluator part. An *Untitled.ipynb* under each part gives a simple check how each module works.

## Training
Here are some selected training procedure.
- Projection learning training loss (left)
- Bi-LSTM sequence labeling model training (right)

<div align="centert">
<img src="https://github.com/WillaFan/hypernymy-extraction/blob/master/projection/loss.png" width="400" alt="proj_loss" >
<img src="https://github.com/WillaFan/hypernymy-extraction/blob/master/BiLSTM/f1.png" width="400" alt="bi_f1" >
</div>
