# 🏏 Umpire Decision Accuracy Prediction (Capstone Project)

## 📍 Project Overview

Cricket is one of the most popular sports globally, where a single umpire decision can change the course of a match.  
In this project, we explore whether machine learning can predict the *accuracy of umpire decisions* under different match conditions.  
Our goal: *improve fairness, trust, and accountability in cricket decision-making.*

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

## 📊 Data Overview

- *Numerical features: ['Match ID', 'Over', 'Runs by Batter', 'Extras', 'Total Runs','Current Wickets', 'Win Margin', 'Event Match Number']
- *Categorical features: ['Inning Team', 'Batter', 'Bowler', 'Wicket Kind', 'Player Out','Fielder', 'Review By', 'Umpire', 'Review Decision', 'Date', 'Season', 'Team 1', 'Team 2', 'Venue', 'Winner',     'win_margin_type','Event Name', 'Event Stage','Umpire 1','Umpire 2', Umpire Decision']
- *Target variable:* Umpire_Decision  
- ‘Struck down’ → correct decision  
- ‘Upheld’ → wrong decision

---

## 🛠️ Methodology

- Preprocessing: handled missing values, one-hot encoding, feature selection
- Feature selection: Random Forest, SHAP, SelectKBest
- Tasks:
  - *Supervised:* Binary classification
  - *Unsupervised:* Clustering umpire behavior (KMeans)

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
- Correlation Heatmap  
- FCNN Accuracy/Loss Curves  
- Confusion Matrix  
- Clustering (Umpire Behavior)

---

## 🚀 Future Work

- Expand to T20/Test match formats  
- Model umpire-specific bias patterns  
- Build a live prediction dashboard for commentators  
- Explore sentiment analysis from commentary  
- Deploy with MLOps practices (monitoring, auto-retraining)

---

## 💻 Repository Structure
