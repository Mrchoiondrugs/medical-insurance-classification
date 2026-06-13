# Medical Insurance Cost Prediction using ANN

An elegant and short deep learning pipeline designed to predict annual medical insurance expenses (`charges`) using an Artificial Neural Network regression architecture.

## Dataset Insights
The dataset contains key patient diagnostic parameters:
* **Numerical features**: `age`, `bmi`, and number of `children`.
* **Categorical features**: `sex` (gender attributes), `smoker` (tobacco consumption status), and `region` (US geographic residency zone).
* **Target column**: `charges` (continuous dollar values representing medical insurance overheads).

## Core Preprocessing Workflow
1.  **Dummification Encoding**: Instead of manual label replacements, used pandas `get_dummies(drop_first=True)` to convert categorical strings into optimized structural binary flags, cleanly dodging structural multi-collinearity.
2.  **Dataset Segmentation**: Divided rows into an **80/20 train-test configuration** to rigorously evaluate predictions.
3.  **Z-Score Feature Standardization**: Standardized numeric inputs via `StandardScaler`. Deep learning convergence risks destabilizing when processing varying input data distributions (such as age spans 18–64 vs. multi-thousand-dollar target values).

## Neural Network Structure
Built dynamically inside a standard Keras Sequential object:
* **Layer 1 (Input/Hidden)**: 64 hidden nodes using non-linear `ReLU` activations.
* **Layer 2 (Hidden)**: 32 hidden nodes utilizing `ReLU` activations.
* **Layer 3 (Output Layer)**: 1 solitary neuron with an implicit **Linear Activation**. Since regression outcomes span an infinite continuum of values, bounds like Sigmoid or Softmax must not be used here.

## Execution Metrics & Performance Observations
* **Loss Tracking**: Evaluated optimization steps directly against Mean Absolute Error (`mae`).
* **Optimization Framework**: Handled via the default `adam` engine over 100 iterations.
* **Key Behavior Insights**: In medical underwriting data, the `smoker_yes` flag typically exerts massive weight variations on prediction slopes. The ANN handles these interactions with high fidelity, achieving strong predictive accuracy while holding overall validation error gaps close to baseline dataset noise.
