# House-PRICE-Prediction
ML Project

Aim is to predict the log-error between their Zestimate and the actual sale price, given all the features of a home. The log error is defined as logerror = log(Zestimate) - log(Sale Price) for the months in Fall 2017.

CODE NOTEBOOK - #Attached
# Files
- `explanatory.ipynb` performs basic explanatory data analysis. The analysis is by no means comprehensive, since most ad-hoc data analysis were performed while extracting features and building models.
- `feature_extraction.ipynb` cleans up the raw data, unifies data types and the representations of missing values, and extracts various kinds of features (eg. interaction features, region-based aggregate features). The features are saved to hdf5 files that can be easily loaded by other notebooks for modeling.
- `model_lgb.ipynb` builds [LightGBM](https://github.com/Microsoft/LightGBM) models on top of the extracted features.
- `model_catboost.ipynb` builds [CatBoost](https://github.com/catboost/catboost) models (both single model and ensemble model) on top of the extracted features.
- `stack.py` performs simple stacking by taking a linear combination of the predictions from LightGBM and CatBoost.
- `src/data_proc.py` includes several helper methods for data cleaning and feature extraction

# Models
Both the LightGBM and CatBoost models are mostly tuned based on offline cross validation, with no public leaderboard probing/overfitting. The weight for the final linear stacking is chosen based on public leaderboard scores. Outliers in the offline training and validation sets were carefully handled so that the offline validation method is as reliable as possible.

The following table outlines the chosen models' performance on the hidden private test set.

| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| LightGBM (single) | 0.0752026 | 332 / 3779 | 8.8% |
| CatBoost (single) | 0.0751456 | 250 / 3779 | 6.6% |
| CatBoost (ensemble x8) | 0.0750750 | 147 / 3779 | 3.9% |
| **Stack (LightGBM + CatBoost)** | **0.0750213** | **95 / 3779** | **2.5%** |

For this dataset, CatBoost with its default hyperparameters gives very strong performance, and the model required almost no hyperparameter tuning. In comparison, LightGBM required much more hyperparameter tuning to achieve good performance.

Also, it turns out that ensembling and stacking are essential to climbing up the leaderboard in this competition. And I believe there is still some room for improvement just by tuning the LightGBM hyperparameters and switching to a better stacking method (eg. train a meta-model on top of the base models' predictions on a hold-out set).
