import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Google Play Store Analysis Project
# ---------------------------------

# Step 1: Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully!\n")
    print(df.head())
    return df

# Step 2: Clean Data
def clean_data(df):
    """Clean and preprocess dataset"""
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Clean Installs column
    if 'Installs' in df.columns:
        df['Installs'] = (
            df['Installs']
            .astype(str)
            .str.replace('[+,]', '', regex=True)
            .astype(int)
        )

    # ✅ Clean Price column safely
    if 'Price' in df.columns:
        df['Price'] = (
            df['Price']
            .astype(str)                      # ensure string
            .str.replace('$', '', regex=True) # remove $
            .replace('Everyone', '0')         # fix wrong entries
        )
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

    # Convert Size column
    def size_to_mb(x):
        try:
            if 'M' in str(x):
                return float(x.replace('M', ''))
            elif 'k' in str(x).lower():
                return float(x.replace('k', '')) / 1024
        except:
            return np.nan
        return np.nan

    if 'Size' in df.columns:
        df['Size'] = df['Size'].apply(size_to_mb)

    return df


# Step 3: Exploratory Data Analysis
def eda(df):
    # a) Top 10 Categories
    plt.figure(figsize=(10,6))
    df['Category'].value_counts().head(10).plot(kind='bar', color='skyblue')
    plt.title("Top 10 App Categories")
    plt.ylabel("Count")
    plt.show()

    # b) Rating Distribution
    sns.histplot(df['Rating'], bins=20, kde=True, color='green')
    plt.title("App Ratings Distribution")
    plt.show()

    # c) Installs vs Rating
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="Rating", y="Installs", data=df, alpha=0.5)
    plt.title("Installs vs Ratings")
    plt.yscale("log")
    plt.show()

    # d) Free vs Paid Apps Ratings
    sns.boxplot(x="Type", y="Rating", data=df)
    plt.title("Free vs Paid Apps Ratings")
    plt.show()

    # e) Correlation Heatmap
    corr = df[['Reviews','Installs','Rating']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Step 4: Extra Insights
def insights(df):
    print("\nTop 10 Installed Apps:")
    top_installed = df.sort_values(by="Installs", ascending=False).head(10)
    print(top_installed[['App','Category','Installs']])

# Main Execution
def main():
    file_path = "C:\Google Playstore\googleplaystore.csv"  # <-- Update with your dataset path
    df = load_data(file_path)
    df = clean_data(df)
    eda(df)
    insights(df)

if __name__ == "__main__":
    main()
