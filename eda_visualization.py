import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def create_plot_directory():
    """Create a directory for saving plots if it doesn't exist"""
    plots_dir = Path('plots')
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)
    return plots_dir

def load_data():
    """Load and return the training and test datasets"""
    print("Loading data...")
    try:
        train_data = pd.read_csv('train_v9rqX0R.csv')
        test_data = pd.read_csv('test_AbJTz2l.csv')
        print("✓ Data loaded successfully!")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def save_plot(fig, filename):
    try:
        fig.savefig(f'plots/{filename}', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        plt.close(fig)

# Set the style for all plots
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create directory and load data
create_plot_directory()
train_data, test_data = load_data()

print("\nGenerating visualizations...")

# 1. Distribution of Item_Outlet_Sales with KDE
fig, ax = plt.subplots()
sns.histplot(data=train_data, x='Item_Outlet_Sales', bins=50, kde=True, ax=ax)
ax.set_title('Distribution of Item Outlet Sales')
plt.tight_layout()
save_plot(fig, 'sales_distribution.png')

# 2. Sales by Outlet Type with violin plot
fig, ax = plt.subplots()
sns.violinplot(data=train_data, x='Outlet_Type', y='Item_Outlet_Sales', ax=ax)
ax.set_title('Sales Distribution by Outlet Type')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_plot(fig, 'sales_by_outlet_type.png')

# 3. Sales by Item Type with swarm plot overlay
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=train_data, x='Item_Type', y='Item_Outlet_Sales', ax=ax, whis=1.5)
sns.swarmplot(data=train_data, x='Item_Type', y='Item_Outlet_Sales', ax=ax, size=4, alpha=0.3, color='red')
ax.set_title('Sales Distribution by Item Type')
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
save_plot(fig, 'sales_by_item_type.png')

# 4. Correlation between MRP and Sales with regression line
fig, ax = plt.subplots()
sns.regplot(data=train_data, x='Item_MRP', y='Item_Outlet_Sales', scatter_kws={'alpha':0.5}, ax=ax)
ax.set_title('Item MRP vs Sales (with Regression Line)')
plt.tight_layout()
save_plot(fig, 'mrp_vs_sales.png')

# 5. Average Sales by Outlet Size with error bars
fig, ax = plt.subplots()
sns.barplot(data=train_data, x='Outlet_Size', y='Item_Outlet_Sales', ci=95, ax=ax)
ax.set_title('Average Sales by Outlet Size (with 95% CI)')
plt.tight_layout()
save_plot(fig, 'sales_by_outlet_size.png')

# 6. Sales Trends by Outlet Location Type with violin plot
fig, ax = plt.subplots()
sns.violinplot(data=train_data, x='Outlet_Location_Type', y='Item_Outlet_Sales', ax=ax)
ax.set_title('Sales Distribution by Location Type')
plt.tight_layout()
save_plot(fig, 'sales_by_location.png')

# 7. Item Visibility vs Sales with hexbin plot
fig, ax = plt.subplots()
plt.hexbin(train_data['Item_Visibility'], train_data['Item_Outlet_Sales'], gridsize=30, cmap='YlOrRd')
plt.colorbar(label='Count')
ax.set_xlabel('Item Visibility')
ax.set_ylabel('Item Outlet Sales')
ax.set_title('Item Visibility vs Sales (Density Plot)')
plt.tight_layout()
save_plot(fig, 'visibility_vs_sales.png')

# 8. Sales by Item Fat Content with violin plot
fig, ax = plt.subplots()
sns.violinplot(data=train_data, x='Item_Fat_Content', y='Item_Outlet_Sales', ax=ax)
ax.set_title('Sales Distribution by Fat Content')
plt.tight_layout()
save_plot(fig, 'sales_by_fat_content.png')

# Create correlation matrix heatmap with improved styling
fig, ax = plt.subplots(figsize=(12, 10))
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
correlation_matrix = train_data[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Correlation Matrix of Numeric Features')
plt.tight_layout()
save_plot(fig, 'correlation_matrix.png')

print("\n✓ All visualizations have been generated and saved in the 'plots' directory.")

# Create a markdown summary of the analysis
try:
    with open('EDA_Summary.md', 'w') as f:
        f.write("""# Exploratory Data Analysis Summary

## Key Findings

1. **Missing Values**:
   - Item_Weight has missing values in both train and test sets
   - Outlet_Size has missing values that need to be handled

2. **Distribution Patterns**:
   - Item_MRP shows a multi-modal distribution
   - Item_Outlet_Sales is right-skewed
   - Item Types are not evenly distributed

3. **Sales Patterns**:
   - Strong correlation between Item_MRP and Sales
   - Different outlet types show distinct sales patterns
   - Outlet size impacts sales performance

4. **Categorical Variables**:
   - Item_Fat_Content has inconsistent labeling
   - Outlet_Type significantly influences sales
   - Establishment Year shows historical sales trends

## Visualization Improvements

1. `sales_distribution.png`: Added KDE curve for better distribution visualization
2. `sales_by_outlet_type.png`: Used violin plots for better distribution insight
3. `sales_by_item_type.png`: Added swarm plots to show data distribution
4. `mrp_vs_sales.png`: Added regression line to show trend
5. `sales_by_outlet_size.png`: Added 95% confidence intervals
6. `sales_by_location.png`: Used violin plots for distribution visualization
7. `visibility_vs_sales.png`: Used hexbin plot for better density visualization
8. `sales_by_fat_content.png`: Used violin plots for distribution insight
9. `correlation_matrix.png`: Improved styling with masked upper triangle

## Implications for Feature Engineering

1. Need to handle missing values in Item_Weight and Outlet_Size
2. Standardize Item_Fat_Content categories
3. Consider creating price range categories based on Item_MRP
4. Potential for outlet age feature from establishment year
5. Possibility of grouping similar item types
6. Consider log transformation for right-skewed sales data
7. Investigate zero-visibility items as potential data quality issue
""")
    print("Enhanced EDA summary has been saved in 'EDA_Summary.md'")
except Exception as e:
    print(f"Error saving EDA summary: {e}") 