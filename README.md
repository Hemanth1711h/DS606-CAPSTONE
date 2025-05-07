# 🏏 Umpire Decision Accuracy Prediction (Capstone Project)

  ![WhatsApp Image 2025-05-07 at 6 09 34 PM](https://github.com/user-attachments/assets/07473981-8264-4622-bd0b-a9a83e196d5a)



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
    - Python 3.x
    - Pandas, NumPy (data manipulation & analysis)
    - Scikit-learn (traditional ML models, pipelines, feature selection)
    - XGBoost (advanced boosting models)
    - TensorFlow / Keras (deep learning models like FCNN)
    - Matplotlib, Seaborn (visualizations)
    - Jupyter Notebook / Google Colab (development environment)
    - Git, GitHub (version control & collaboration)

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
