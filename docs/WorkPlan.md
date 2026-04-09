### Performance Evaluation of Lightweight Classifiers for Early-Stage Parkinson’s Disease Detection using Acoustic Features

<span style="color:red; font-size:18px">Deadline:  April 20</span>

#### Use Supervised Machine Learning
---

#### Day 01:
- setup environment 
- install libraries (pip install pandas scikit-learn matplotlib seaborn)
- Download UCI Parkinson Data set


#### Day 02:
- Learn Pandas From Bro Code
- Developed the data cleaning program(data_clean.py)
- create the cleaned Data Set (except name and status) 

#### Day 03:
- Learning Scikit-Learn basics

#### Day 04 (23/03/2026):
- Learning Scikit-Learn Basics
- Split Data
- Data Scalling

#### Day 04 (24/03/2026):
- Learn K-Cross Validation
- Basic Info About Classifiers

### 30/03/2026
- Start making paper
- Methodology Part

## Paper Parts

- Title - Done
- Abstract 
- Keywords
- Introduction
- Methodology - Done
- Results - DOne
- Conclusion - Done
- References


01.

[1]Costa HN, Esteves AR, Empadinhas N, Cardoso SM. Parkinson's Disease: A Multisystem Disorder. Neurosci Bull. 2023 Jan;39(1):113-124. doi: 10.1007/s12264-022-00934-6. Epub 2022 Aug 22. PMID: 35994167; PMCID: PMC9849652.

Rizzo G, Copetti M, Arcuti S, Martino D, Fontana A, Logroscino G. Accuracy of clinical diagnosis of Parkinson disease: A systematic review and meta-analysis. Neurology. 2016 Feb 9;86(6):566-76. doi: 10.1212/WNL.0000000000002350. Epub 2016 Jan 13. PMID: 26764028.





---

. The "Settings" (Hyperparameters)
Inside the brackets of the algorithms, you will see some specific settings. In Machine Learning, these are called Hyperparameters. You are essentially turning the dials on the machine before turning it on.

class_weight='balanced': Remember the 75/25 imbalance between Parkinson's and healthy patients? If you don't include this, the AI might just guess "Parkinson's" every time to cheat and get a 75% score. This setting forces the math to penalize the AI heavily if it gets a "Healthy" patient wrong, forcing it to pay attention to both.

random_state=42: The random seed. It locks the randomness so that if you run this code today, and then run it again tomorrow to show your university panel, you will get the exact same numbers.

kernel='linear' (SVM only): Tells the Support Vector Machine to draw a perfectly straight line through the data to separate the sick from the healthy.

n_neighbors=5 (KNN only): Tells the K-Nearest Neighbors algorithm to look at the 5 patients closest to the current one to make a guess.

scale_pos_weight=3 (XGBoost only): XGBoost doesn't understand the word 'balanced', so we literally tell it: "Parkinson's cases are roughly 3 times more common here, adjust your math accordingly."