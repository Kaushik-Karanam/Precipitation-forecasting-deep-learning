# 🌦️ Precision Precipitation Forecasting for the Great Lakes Region

This project presents a hybrid deep learning framework for short-term precipitation forecasting over the Lake Michigan area. By fusing meteorological time-series data with GOES satellite imagery, the model predicts precipitation intensity for 24, 48, and 72-hour windows, with a special focus on addressing the severe class imbalance inherent in weather data.

***

## 🎯 Objective
To develop a reliable and scalable forecasting system that accurately classifies precipitation into four categories (*No Rain, Light Rain, Medium Rain, Heavy Rain*) up to 72 hours in the future. The core aim is to improve the prediction accuracy of rare but critical weather events, such as heavy rain, which traditional models often miss.

***

## ⚠️ Problem Statement
Precipitation forecasting is a complex task due to two primary challenges addressed in this project:

1.  **Severe Class Imbalance:** Weather data is overwhelmingly dominated by "No Rain" events (over 93% in this dataset), causing standard models to achieve high accuracy by simply predicting the majority class, while failing completely on minority classes like "Heavy Rain".
2.  **Multi-Modal & Multi-Resolution Data:** The project integrates two distinct data sources with different temporal resolutions:
    * **Meteorological Data:** High-frequency tabular data (e.g., 24 records per day).
    * **GOES Satellite Imagery:** Lower-frequency image data (e.g., 8 records per day).
    Fusing these sources effectively requires a sophisticated architecture and a thoughtful data preprocessing strategy to align the features in time.

***

## ⚙️ System Architecture
A two-branch hybrid deep learning model was designed to process each data type optimally before fusing them for a final prediction.

#### 📊 Meteorological Branch (LSTM with Attention)
* Processes tabular time-series data (e.g., temperature, pressure, humidity).
* **Layers:** Stacked LSTMs capture temporal dependencies, while an **Attention Layer** allows the model to focus on the most influential time steps in the input sequence.
* **Input Shape:** `(None, T, 10)` where `T` is the number of time steps and `10` is the number of meteorological features.

#### 🛰️ Satellite Imagery Branch (ConvLSTM2D)
* Processes sequences of satellite images to learn spatio-temporal patterns (e.g., cloud movement and formation).
* **Layers:** **ConvLSTM2D** layers are ideal for learning from image sequences. These are followed by Time-Distributed Convolutional layers to apply convolutions across each time step.
* **Input Shape:** `(None, T, 64, 64, 1)` where `T` is the number of time steps and `(64, 64, 1)` is the dimension of the grayscale satellite images.

#### 🔗 Fusion and Classification
* The feature vectors from both branches are concatenated.
* The fused vector is passed through a series of Dense layers with Dropout for regularization.
* A final `softmax` activation layer outputs the probability for each of the four precipitation classes.

***

## 🛤️ Methodology & Workflow
The project workflow is broken down into four key stages:

1.  **🧹 Data Preprocessing:**
    * Handled missing values in meteorological data.
    * Resized GOES satellite images to a uniform `64x64` resolution.
    * Engineered a **daily summary strategy** to align the 24-record meteorological data with the 8-record satellite data by averaging features.
    * Utilized a **sliding window approach** to generate sequences of `(past_data, future_target)` for training the time-series models.

2.  **🔎 Exploratory Data Analysis (EDA):**
    * Analyzed data distributions and temporal patterns to inform modeling decisions.
    * Visualized the severe class imbalance, which motivated the use of class weighting.

3.  **🤖 Model Training:**
    * Trained separate models optimized for 24, 48, and 72-hour forecast windows.
    * Implemented **class weighting** during training to penalize the model more for misclassifying minority classes (Light, Medium, Heavy Rain). This was critical to overcoming the class imbalance.

4.  **✅ Evaluation:**
    * Assessed model performance using classification reports (Precision, Recall, F1-Score) and confusion matrices.
    * Compared baseline RNN models against the more complex hybrid architectures to quantify the benefit of data fusion.

***

## 🏆 Results & Key Findings
The hybrid models demonstrated a significant performance improvement over baseline models.

| Model Type | Window Size | Weight Fusion | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Baseline RNN | - | - | 66% | 0.64 |
| Hybrid | 24H | Manual | 70% | 0.68 |
| Hybrid | 48H | Manual | 74% | 0.73 |
| **Hybrid (Best)** | **72H** | **Manual+Extended** | **77%** | **0.76** |

* **Key Insight:** ✨ The best hybrid model showed an **11% absolute improvement** in accuracy over a standard RNN that used only meteorological data.
* **Class Imbalance Success:** 💡 By implementing class weights, the model's accuracy on the critical "Heavy Rain" class **increased from 0% to 64%**, proving the effectiveness of the strategy.

***

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Git

### Run the Notebooks ▶️
The notebooks are numbered in the recommended order of execution:

1_Full_analysis_and_development.ipynb: Contains the complete exploratory data analysis and initial model experiments.
2_Model_24hr.ipynb: Trains and evaluates the final 24-hour forecast model.
3_Model_48hr.ipynb: Trains and evaluates the final 48-hour forecast model.
4_Model_72hr.ipynb: Trains and evaluates the final 72-hour forecast model.


##🔭 Future Work
Incorporate Advanced Architectures: Experiment with Transformer-based models like Time-Series Transformers for potentially better long-range dependency capture.
Expand Feature Set: Integrate more weather variables (e.g., wind direction, upper-atmosphere data) to enhance predictive accuracy.
Extend Forecast Horizon: Work towards extending the reliable forecasting window beyond 72 hours.
