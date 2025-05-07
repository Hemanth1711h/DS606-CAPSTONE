# 🏏 Umpire Decision Accuracy Prediction (Capstone Project)

## 📍 Project Overview

Cricket is one of the most popular sports globally, where a single umpire decision can change the course of a match.  
In this project, we explore whether machine learning can predict the *accuracy of umpire decisions* under different match conditions.  
Our goal: *improve fairness, trust, and accountability in cricket decision-making.*

## 🎯 Objective

The primary objective of this capstone project is to analyze decision-making patterns in professional cricket umpiring using machine learning. Specifically, the project aims to predict the correctness of an umpire's decision based on historical match data, ball characteristics, player behavior, and contextual game features. The goal is not to replace umpires but to provide a supportive analytical tool that identifies potential biases or inconsistencies and enables data-driven insights for cricket governing bodies.


---

## 🔍 Research Questions

1. Can we build a machine learning model that predicts umpire decision accuracy based on match conditions?
2. What are the key factors influencing umpire decision accuracy in ODI matches?

---

## 💾 Data Source

- *Primary Source:* [CricSheet](https://cricsheet.org/matches/)
- *Data format:* JSON → cleaned into CSV
- *Files:*
  - ball_by_ball_data.csv: granular ball-level match data
  - match_info.csv: match context (teams, umpires, venue, date)
- *Sample size:* ~18,000 matches, filtered to ~2,400 DRS-reviewed decisions

---

## 📊 Data Overview and Tools to Use

- *Numerical features: ['Match ID', 'Over', 'Runs by Batter', 'Extras', 'Total Runs','Current Wickets', 'Win Margin', 'Event Match Number']
- *Categorical features: ['Inning Team', 'Batter', 'Bowler', 'Wicket Kind', 'Player Out','Fielder', 'Review By', 'Umpire', 'Review Decision', 'Date', 'Season', 'Team 1', 'Team 2', 'Venue', 'Winner',     'win_margin_type','Event Name', 'Event Stage','Umpire 1','Umpire 2', Umpire Decision']
- *Target variable:* Umpire_Decision  
- ‘Struck down’ → correct decision  
- ‘Upheld’ → wrong decision
  
-  Tools & Technologies:
-  Python 3.x
-  Pandas, NumPy (data manipulation & analysis)
-  Scikit-learn (traditional ML models, pipelines, feature selection)
-  XGBoost (advanced boosting models)
-  TensorFlow / Keras (deep learning models like FCNN)
-  Matplotlib, Seaborn (visualizations)
-  Jupyter Notebook / Google Colab (development environment)
-  Git, GitHub (version control & collaboration)

---

## 🛠️ Methodology

- Preprocessing: handled missing values, one-hot encoding, feature selection
- Feature selection: Random Forest, SHAP, SelectKBest
- Tasks:
  - *Supervised:* Binary classification


---

## 🤖 Models and Results

<img width="731" alt="image" src="https://github.com/user-attachments/assets/c86b1a8a-5148-4d10-bd29-de6af6873fb1" />

---

## 💥 Key Insights

✅ Top predictors: Over, Wicket Kind, Review By, Inning Team  
✅ Correlation heatmap revealed weak direct correlation, but models captured non-linear patterns  
✅ Feature importance varied across models — RF and FCNN picked slightly different dominant variables

---

## 📈 Visual Highlights

- Feature Importance Graphs
<img width="780" alt="image" src="https://github.com/user-attachments/assets/8d6f1b99-ded6-4380-96db-6c089816077a" />

- FCNN Architecture
<img width="448" alt="image" src="https://github.com/user-attachments/assets/aa6e7b3b-0185-401f-9877-b25944111132" />


- Correlation Heatmap
<img width="611" alt="image" src="https://github.com/user-attachments/assets/21ba2b26-3bb6-4063-ac44-fc0f469b3b3d" />

- FCNN Accuracy/Loss Curves
<img width="755" alt="image" src="https://github.com/user-attachments/assets/57632cbe-ea4e-4de3-bc58-1a028d46ab43" />

- Confusion Matrix for Best Model
<img width="749" alt="image" src="https://github.com/user-attachments/assets/9b17d131-cefb-491b-9086-94127b4bd81e" />


---

## 🚀 Future Work

- Expand to T20/Test match formats  
- Model umpire-specific bias patterns  
- Build a live prediction dashboard for commentators  
- Explore sentiment analysis from commentary  
- Deploy with MLOps practices (monitoring, auto-retraining)

---

## 💻Final Presentation Link

https://docs.google.com/presentation/d/1y2Sj3eoIU7yoPSCkBxLuHfijoEmk0bN1/edit?usp=sharing&ouid=115928439987088123968&rtpof=true&sd=true
