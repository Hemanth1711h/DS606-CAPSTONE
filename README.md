# 🏏 Umpire Decision Accuracy Prediction (Capstone Project)

  ![WhatsApp Image 2025-05-07 at 6 09 34 PM](https://github.com/user-attachments/assets/07473981-8264-4622-bd0b-a9a83e196d5a)



## 📍 Project Overview

This project explores the use of machine learning to predict the accuracy of cricket umpire decisions under various match conditions.
Why does this matter? Because a single umpire decision can shape the outcome of a cricket match, affecting fairness, player morale, and fan trust.

Goal: Build models that help assess the likelihood of umpire decision correctness, offering data-driven support for cricket authorities to enhance fairness and accountability.

## 🎯 Primary Objective

The primary objective of this capstone project is to analyze decision-making patterns in professional cricket umpiring using machine learning. Specifically, the project aims to predict the correctness of an umpire's decision based on historical match data, ball characteristics, player behavior, and contextual game features. The goal is not to replace umpires but to provide a supportive analytical tool that identifies potential biases or inconsistencies and enables data-driven insights for cricket governing bodies.

## 📌 Literature Review

The accuracy of umpire decisions in cricket has been a prominent topic in sports analytics research, with various studies studying the components that influence such judgments. Earlier research by Sacheti, Gregory-Smith, & Paton (2015) studied home bias in LBW rulings in 1000 test matches utilizing negative binomial regression models. Their analysis found that home referees have historically preferred home teams, with bias growing more obvious in the later rounds of matches. Nonetheless, the adoption of neutral umpires considerably lessened this impact. Although their investigation produced statistical insights, it stressed descriptive analysis over machine learning. 

(Sacheti et al., 2014).Ramachandran et al. (2022) evaluated the impact of task-switching on the decision-making of umpires regarding LBW calls. Using eye-tracking technology, it was observed that umpires alternating between judging no-balls and LBW calls demonstrated shorter fixation durations and a spike in inaccuracies regarding pitch judgments. Nonetheless, judges adjusted by employing adaptive gaze techniques like fixing their gaze on the stumps to enhance precision. The results revealed that educating umpires on improving gaze fixation tactics and handling task-switching issues could enhance decision accuracy 

(Ramachandran et al., n.d.).Despite extensive study on umpire judgment influences, machine learning applications in umpire decision validation remain largely unexplored. Our endeavor tackles this gap by presenting a data-driven approach to assess umpire judgment accuracy. our approach aims to forecast whether an umpire’s decision was right or wrong. This predictive strategy extends on earlier discoveries by applying machine learning techniques which will suite categorical data and complex interactions between components.

Through this technique, our endeavor provides fresh insights to the field of sports analytics, improving the study of aspects that influence umpire performance. This research not only expands past work but also has practical implications for cricket regulating bodies to improve umpire training, minimize judgment bias, and promote the general fairness of the sport.


---

## 🔍 Research Questions

While moving forward through the project these questions were brought into the picture:

1. Can we build a machine learning model that predicts umpire decision accuracy based on match conditions?

   We aim to develop predictive models using machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost) to assess whether match conditions, player behavior, and in-game events can accurately forecast the correctness of umpire decisions.


3. What are the key factors influencing umpire decision accuracy in ODI matches?

   We analyzed feature importance using models such as Random Forest and Logistic Regression to identify which factors (e.g., over number, batter, bowler, venue, type of delivery) most impact the likelihood of an umpire making a correct or incorrect decision.



---

## 💾 Data Source

- *Primary Source:* [CricSheet](https://cricsheet.org/matches/)
- *Data format:* JSON → cleaned into CSV
- *Files:*
  - ball_by_ball_data.csv: granular ball-level match data
  - match_info.csv: match context (teams, umpires, venue, date)
- *Sample size:* ~18,000 matches, filtered to ~2,400 DRS-reviewed decisions

---

## 📊 Data Overview 

- *Numerical features: ['Match ID', 'Over', 'Runs by Batter', 'Extras', 'Total Runs','Current Wickets', 'Win Margin', 'Event Match Number']
- *Categorical features: ['Inning Team', 'Batter', 'Bowler', 'Wicket Kind', 'Player Out','Fielder', 'Review By', 'Umpire', 'Review Decision', 'Date', 'Season', 'Team 1', 'Team 2', 'Venue', 'Winner',     'win_margin_type','Event Name', 'Event Stage','Umpire 1','Umpire 2', Umpire Decision']
- *Target variable:* Umpire_Decision  
- ‘Struck down’ → correct decision  
- ‘Upheld’ → wrong decision

---

## 📌 Tools & Technologies:
- [Python Latest version](https://www.python.org/downloads/)
- Pandas, NumPy (data manipulation & analysis)
- Scikit-learn (traditional ML models, pipelines, feature selection)
- XGBoost (advanced boosting models)
- TensorFlow / Keras (deep learning models like FCNN)
- Matplotlib, Seaborn (visualizations)
- (Anaconda Navigator for Jupyter Notebook)[https://www.anaconda.com/products/navigator] (Deployment Environment)
- (Google Colab)[https://colab.research.google.com/] (Development Environment)
- Git, GitHub (version control & collaboration)

--- 

## 📈 Implementation and steps for how to run the project:

✅ Step 1: Open Anaconda Navigator
→ Launch the Anaconda Navigator from your desktop or start menu.

✅ Step 2: Launch Jupyter Notebook
-   Inside Anaconda Navigator
-   Click on Jupyter Notebook
-   It will open Jupyter in your web browser.

  <img width="1110" alt="image" src="https://github.com/user-attachments/assets/ca885130-c411-43ed-bcab-3bae1f5c7ced" />

-   While you click launch its going to display like:
  <img width="1102" alt="image" src="https://github.com/user-attachments/assets/91879c00-9a0a-4333-8172-40d53edce317" />



✅ Step 3: Create a new notebook
-   In Jupyter, navigate to your project folder.
-   Click New 
-   select Python 3 (ipykernel)
-   It will create a new .ipynb file.

  <img width="1120" alt="image" src="https://github.com/user-attachments/assets/90ff2003-b263-4da1-9921-96ab7967fc56" />




✅ Step 4: Write your code
→ Import the necessary libraries in your notebook, for example:

  import pandas as pd  
  import numpy as np  
  import matplotlib.pyplot as plt  
  import seaborn as sns  
  import sklearn  
  import xgboost  
  import tensorflow  

✅ Step 5: Execute your code
→ Run each code cell by pressing Shift + Enter or clicking the Run button.

✅ Step 6: Save the notebook
-   Go to File 
-   Save and Checkpoint
-   The .ipynb file will be saved to your local machine.

✅ Step 7: Upload to GitHub
-   Open your GitHub repository in the browser.
-   Click Add file → Upload files → drag and drop or select your .ipynb file.
-   Write a commit message → click Commit changes → your file is now updated in GitHub.


---

## 🛠️ Methodology

✅Data Collection:
We collected ball-by-ball and match-level cricket data from official databases and match logs, including umpire decisions and review outcomes.

✅Data Preprocessing:
We cleaned the data by removing duplicates, handling missing values, dropping irrelevant columns (e.g., toss decision, TV umpire), and correcting data types. We also one-hot encoded categorical variables and removed constant features.

✅Exploratory Data Analysis (EDA):
We visualized the data using histograms, count plots, pie charts (for target distribution), and correlation matrices to understand patterns, class imbalance and relationships between variables.

✅Feature Engineering:
We engineered new features and selected the top 15 important variables using feature importance from RandomForest and SelectKBest. We also applied domain knowledge to retain relevant cricket-specific variables.

✅Feature Scaling:
We standardized continuous features using StandardScaler to ensure uniform scale across features before feeding them into machine learning models.

✅Model Selection:
We selected both traditional and ensemble models:

![image](https://github.com/user-attachments/assets/a52c1df8-f053-4e2d-b0c0-714377f62acd)


✅Traditional Models: Logistic Regression, Decision Tree

Ensemble Models: Random Forest, AdaBoost, Gradient Boosting, XGBoost

Stacking using Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, XGBoost as Meta Model

✅Model Training:
We split the data into training and test sets (with stratification) and trained models using hyperparameter tuning techniques like GridSearchCV and RandomizedSearchCV.

✅Model Evaluation:
We evaluated models using metrics such as accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices on both training and test sets. We also visualized FCNN training history (accuracy, loss) over epochs.

✅Visualization and Communication:
We created visual summaries, including pie charts, bar plots for feature importance, and training vs. validation curves, to communicate insights and explain model decisions.




---


## 💥 Key Insights

✅ Top predictors: Over, Wicket Kind, Review By, Inning Team  
✅ Correlation heatmap revealed weak direct correlation, but models captured non-linear patterns  
✅ Feature importance varied across models — RF(Random Forest and FCNN(Fully Concoluted Neural Network) picked slightly different dominant variables

---

## 📈 Visual Highlights

- Feature Importance Graphs
<img width="780" alt="image" src="https://github.com/user-attachments/assets/8d6f1b99-ded6-4380-96db-6c089816077a" />

Over, Wicket Kind _Caught, Total Runs, Win Margin, Review By_England, Match ID, Review By_Australia, Current Wickets, Inning Team England, Even Match Number, Wicket Kind _LBW

Each of these features contributes context, whether it's the match situation (like total runs, over, or current wickets), the type of appeal (like LBW or caught), or team-specific behavior (like who reviewed). These features consistently showed up in our different model plots, meaning they truly influence whether a decision is likely to be right or wrong.

- Correlation Heatmap
<img width="611" alt="image" src="https://github.com/user-attachments/assets/21ba2b26-3bb6-4063-ac44-fc0f469b3b3d" />

This heatmap shows how numerical features relate to each other and to umpire decision accuracy.
We expected to find strong predictors — but surprisingly, most correlations with the target were weak.
Total Runs and Extras had a high correlation, while Over and Wickets showed game flow patterns.
 
The umpire decision variable barely correlated linearly, signaling hidden patterns beneath the surface.
This insight pushed us toward non-linear models like Random Forest and Neural Networks.
It taught us that in cricket, as in life, the surface often hides the real story

- FCNN Accuracy/Loss Curves
<img width="755" alt="image" src="https://github.com/user-attachments/assets/57632cbe-ea4e-4de3-bc58-1a028d46ab43" />

These plots show how our neural network learned over 80 epochs.
On the left, we see training and validation accuracy steadily improving and stabilizing around ~76-77%.
On the right, the loss curves show a smooth, consistent decrease without divergence.
Importantly, the train and validation curves stay close, indicating minimal overfitting.
This tells us the model generalizes well and has learned meaningful patterns.
Overall, these curves gave us confidence to select this as one of our best-performing models.






## 🚀 Future Work

1. Expand the Dataset Across More Matches and Formats
Collect and include more ODI, T20, and Test match data.
Include women’s cricket and franchise leagues (e.g., IPL and BBL).
This improves model generalizability and fairness across formats and conditions.

2. Analyze Umpire-Level and Player-Level Bias
Build models specifically for each umpire to detect patterns or potential biases.
Examine whether certain players or teams consistently get favorable/unfavorable decisions.
This helps address fairness, accountability, and even policy recommendations.

3. Incorporate Real-Time Data for Live Decision Support
Integrate ball-tracking, player movement, weather conditions, and crowd noise into models.
Develop a system to provide live decision support or post-match analytics.
This pushes the project from retrospective analysis into a real-time application. Focus on critical review moments

4. Apply Unsupervised Learning for Pattern Discovery
Use clustering to group umpires by decision styles.
Use anomaly detection to spot unusual decision patterns or rare events.
This can reveal hidden structures in umpire behavior that supervised models miss.

5. Develop Dashboard or Web App for Stakeholders
Build an interactive dashboard for analysts, commentators, or fans.
Allow users to explore predictions, important features, and accuracy trends.
This enhances transparency and puts data insights into the hands of end-users.


---

## 💻Final Presentation Link

https://docs.google.com/presentation/d/1y2Sj3eoIU7yoPSCkBxLuHfijoEmk0bN1/edit?usp=sharing&ouid=115928439987088123968&rtpof=true&sd=true
