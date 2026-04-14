# =========================================
# Netflix Movie Rating Prediction
# =========================================

# 1. Import libraries
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
# =========================================
# 2. Unzip dataset
# =========================================

zip_path = "../data/dataSet.zip"   # adjust path if needed
extract_path = "../data/netflix_data"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)

print("Files extracted:")
print(os.listdir(extract_path))
# =========================================
# 3. File paths
# =========================================

ratings_path = os.path.join(extract_path, "data.txt")
titles_path = os.path.join(extract_path, "movieTitles.csv")

print("Ratings file exists:", os.path.exists(ratings_path))
print("Movie titles file exists:", os.path.exists(titles_path))
# =========================================
# 4. Load movie titles
# =========================================

movie_titles = pd.read_csv(
    titles_path,
    header=None,
    names=["movie_id", "release_year", "title"],
    encoding="latin1"
)

movie_titles.head()
# =========================================
# 5. Parse ratings data.txt
# =========================================
# Format:
# movie_id:
# user_id,rating,date
# ...
# next_movie_id:
#
# To keep notebook manageable, we parse the full file but you can
# later subsample if runtime is too long.

records = []
current_movie_id = None

with open(ratings_path, "r", encoding="latin1") as f:
    for line in f:
        line = line.strip()
        
        if not line:
            continue
        
        # Movie ID line
        if line.endswith(":"):
            current_movie_id = int(line[:-1])
        else:
            user_id, rating, date = line.split(",")
            records.append([
                int(user_id),
                current_movie_id,
                int(rating),
                date
            ])

ratings_df = pd.DataFrame(records, columns=["user_id", "movie_id", "rating", "date"])

print("Parsed ratings shape:", ratings_df.shape)
ratings_df.head()
# =========================================
# 6. Merge movie titles
# =========================================

df = ratings_df.merge(movie_titles, on="movie_id", how="left")

print("Merged dataset shape:", df.shape)
df.head()
# =========================================
# 7. Basic data cleaning
# =========================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# drop bad dates if any
df = df.dropna(subset=["date"])

# make sure release year numeric
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

print(df.info())
# =========================================
# 8. Exploratory analysis
# =========================================

print("Number of users:", df["user_id"].nunique())
print("Number of movies:", df["movie_id"].nunique())
print("Number of ratings:", len(df))

plt.figure(figsize=(8, 5))
sns.countplot(x="rating", data=df)
plt.title("Distribution of Movie Ratings")
plt.show()
# =========================================
# 9. Feature engineering
# =========================================

# Extract year and month of rating
df["rating_year"] = df["date"].dt.year
df["rating_month"] = df["date"].dt.month

# Movie age when rated
df["movie_age_at_rating"] = df["rating_year"] - df["release_year"]

# Fill missing movie ages if release_year missing
df["movie_age_at_rating"] = df["movie_age_at_rating"].fillna(df["movie_age_at_rating"].median())

# Simple aggregate features
user_avg_rating = df.groupby("user_id")["rating"].mean().rename("user_avg_rating")
movie_avg_rating = df.groupby("movie_id")["rating"].mean().rename("movie_avg_rating")
movie_rating_count = df.groupby("movie_id")["rating"].count().rename("movie_rating_count")

df = df.merge(user_avg_rating, on="user_id", how="left")
df = df.merge(movie_avg_rating, on="movie_id", how="left")
df = df.merge(movie_rating_count, on="movie_id", how="left")

df.head()
# =========================================
# 10. Train / Test split
# =========================================
# The assignment says:
# use all ratings for a given movie in training,
# except one randomly picked rating per movie for the test set.
#
# We'll implement exactly that.

np.random.seed(42)

test_indices = []

for movie_id, group in df.groupby("movie_id"):
    sampled_idx = group.sample(n=1, random_state=42).index[0]
    test_indices.append(sampled_idx)

test_df = df.loc[test_indices].copy()
train_df = df.drop(index=test_indices).copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Unique movies in test:", test_df["movie_id"].nunique())
# =========================================
# 11. Feature selection
# =========================================

feature_cols = [
    "user_id",
    "movie_id",
    "release_year",
    "rating_year",
    "rating_month",
    "movie_age_at_rating",
    "user_avg_rating",
    "movie_avg_rating",
    "movie_rating_count"
]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df["rating"]

X_test = test_df[feature_cols].fillna(0)
y_test = test_df["rating"]

print(X_train.shape, X_test.shape)
# =========================================
# 12. Baseline model: Linear Regression
# =========================================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print("Linear Regression RMSE:", round(lr_rmse, 4))
# =========================================
# 13. Ridge Regression
# =========================================

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
print("Ridge Regression RMSE:", round(ridge_rmse, 4))
# =========================================
# 14. Random Forest Regressor
# =========================================

# You can reduce n_estimators if runtime is long
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print("Random Forest RMSE:", round(rf_rmse, 4))
# =========================================
# 15. Compare model performance
# =========================================

results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Random Forest"],
    "RMSE": [lr_rmse, ridge_rmse, rf_rmse]
}).sort_values("RMSE")

results
# =========================================
# 16. Plot model comparison
# =========================================

plt.figure(figsize=(8, 5))
sns.barplot(data=results, x="Model", y="RMSE")
plt.title("Model Comparison by RMSE")
plt.xticks(rotation=15)
plt.show()
# =========================================
# 17. Feature importance (Random Forest)
# =========================================

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance
# =========================================
# 18. Plot feature importance
# =========================================

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x="importance", y="feature")
plt.title("Random Forest Feature Importance")
plt.show()
# =========================================
# 19. Save best model
# =========================================

os.makedirs("../model", exist_ok=True)

best_model = rf_model
joblib.dump(best_model, "../model/rating_regression_model.pkl")

print("Best model saved to ../model/rating_regression_model.pkl")
# =========================================
# 20. Example prediction
# =========================================

sample_input = X_test.iloc[[0]]
sample_prediction = best_model.predict(sample_input)[0]

print("Predicted rating:", round(sample_prediction, 2))
print("Actual rating:", y_test.iloc[0])
