import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Path to data
data_dir = r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# 1. Target Distributions
plt.figure(figsize=(15, 10))
for i, target in enumerate(train_df['target_name'].unique()):
    plt.subplot(2, 3, i+1)
    sns.histplot(train_df[train_df['target_name'] == target]['target'], kde=True)
    plt.title(f"Distribution of {target}")
plt.tight_layout()
os.makedirs("outputs/eda_plots", exist_ok=True)
plt.savefig("outputs/eda_plots/target_distributions.png")
plt.close()

# 2. Categorical Variable analysis
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df.drop_duplicates(subset=['image_path']), x='State')
plt.title("Sample Count by State")
plt.savefig("outputs/eda_plots/state_counts.png")
plt.close()

# 3. Metadata vs Target Correlation
# Let's pivot the data to wide format for correlation analysis
wide_train = train_df.pivot_table(index=['image_path', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
                                   columns='target_name', 
                                   values='target').reset_index()

plt.figure(figsize=(10, 8))
sns.heatmap(wide_train[['Pre_GSHH_NDVI', 'Height_Ave_cm', 'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("outputs/eda_plots/correlation_matrix.png")
plt.close()

# 4. Image Visualization
def visualize_samples(df, n=5):
    plt.figure(figsize=(20, 10))
    samples = df.drop_duplicates(subset=['image_path']).sample(n)
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(data_dir, row['image_path'])
        img = Image.open(img_path)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"State: {row['State']}\nSpecies: {row['Species'][:20]}...")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/sample_images.png")
    plt.close()

visualize_samples(train_df)

print("EDA plots saved.")
