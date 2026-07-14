# Apple Products Pricing Analysis & Preprocessing Pipeline

An end-to-end data cleaning, preprocessing, feature engineering, and data balancing pipeline for a dataset of 80,000 Apple product listings on major platforms (Amazon, Flipkart) spanning 2020 to 2026. This project prepares the raw datasets for robust downstream machine learning models (e.g., predicting product rating, sale probability, or stock availability).

---

## 📁 Repository Structure

| File Name | Type | Description |
| :--- | :--- | :--- |
| **[Apple_Products_Pricing.ipynb](file:///c:/Users/gouth/Projects/AIML/Apple_Products_Pricing.ipynb)** | Jupyter Notebook | Complete step-by-step EDA, data cleaning, feature scaling/encoding, feature engineering, and class balancing pipeline. |
| **[apple_products_pricing_2020_2026.csv](file:///c:/Users/gouth/Projects/AIML/apple_products_pricing_2020_2026.csv)** | CSV Dataset | Raw, unprocessed dataset containing 80,000 product records and 14 features. |
| **[apple_products_pricing_preprocessed.csv](file:///c:/Users/gouth/Projects/AIML/apple_products_pricing_preprocessed.csv)** | CSV Dataset | Preprocessed and cleaned dataset (80,000 rows, 28 features) ready for ML models. |
| **[LICENSE](file:///c:/Users/gouth/Projects/AIML/LICENSE)** | Text | License information (Apache License 2.0). |

---

## 📊 Dataset Features (Raw)

The raw dataset contains **80,000 records** with the following 14 variables:
* `Date`: The date of the product record (YYYY-MM-DD).
* `Platform`: E-commerce channel where the product is listed (Amazon, Flipkart).
* `Product_Category`: Type of Apple product (iPhone, iPad, Mac, Watch).
* `Model_Name`: Precise model name/marketing name of the Apple product.
* `Condition`: Condition of the product listing (New, Renewed/Refurbished).
* `Launch_Price_USD`: Official USD price of the product at launch time.
* `Launch_Price_INR`: Official INR price of the product at launch time.
* `Current_Price_USD`: Current listing price of the product in USD.
* `Current_Price_INR`: Current listing price of the product in INR.
* `Discount_Pct`: Applied discount percentage.
* `Sale_Event`: Active promotional event (e.g., Black Friday, Big Billion Days, Prime Day, Great Indian Festival, or null).
* `Stock_Status`: Product availability category (In Stock, Out of Stock, Low Stock).
* `Rating`: Average customer rating (scale 1.0 to 5.0).
* `Reviews_Count`: Total number of customer reviews for the listing.

---

## ⚙️ Data Preprocessing & Engineering Pipeline

The Jupyter notebook implements the following preprocessing steps in order:

### 1. Data Cleaning
* **Missing Values**: Identifies that `Sale_Event` contains missing values (91.69% missing). Since a missing value indicates that no promotional event was active at the time, NaNs are filled with the explicit label `"No Sale"`.
* **Duplicates**: Scan for and drops exact duplicate records (`.drop_duplicates()`), ensuring data integrity.

### 2. Outlier Management
* **Detection**: Outliers are inspected in the continuous numerical variables (`Current_Price_USD` and `Reviews_Count`) using **IQR (Interquartile Range)** and **Z-score** methods.
* **Capping (Winsorization)**: To preserve extreme but valid popular listings, the `Reviews_Count` feature is capped at the IQR upper and lower bounds rather than deleting rows.

### 3. Categorical Encoding
* **Label Encoding**: Converted the binary variable `Condition` (`New` -> `0`, `Renewed/Refurbished` -> `1`).
* **One-Hot Encoding**: Converted nominal columns `Platform` and `Product_Category` into dummy variables (`Platform_Amazon`, `Platform_Flipkart`, `Category_Mac`, `Category_Watch`, `Category_iPad`, `Category_iPhone`).

### 4. Feature Scaling
* **Min-Max Scaling**: Applied to `Current_Price_USD` to scale values between `[0, 1]`.
* **Z-score Standardization**: Applied to `Reviews_Count_capped` to normalize feature distribution with a mean of 0 and a standard deviation of 1.

### 5. Feature Engineering
Four new domain-relevant features are constructed:
* **`Price_Drop_USD`**: Implied dollar price difference from launch (`Launch_Price_USD - Current_Price_USD`).
* **`Price_Ratio`**: Current price divided by launch price, representing price preservation/retention.
* **`On_Sale`**: Binary flag indicating if any promotional sale event was active (`Sale_Event != 'No Sale'`).
* **`Implied_FX_Rate`**: Implied USD to INR exchange rate (`Current_Price_INR / Current_Price_USD`).

### 6. Data Splitting & Balancing
* **Train-Test Split**: The data is split into an 80% training set (64,000 rows) and a 20% testing set (16,000 rows), stratified on the target `Stock_Status` to maintain class distribution.
* **Target Imbalance**: Predicting `Stock_Status` presents a class imbalance problem (e.g., ~68.8% In Stock).
* **Data Balancing Techniques**:
  * **Random Oversampling (ROS)**: Replicates minority class instances to match the majority class.
  * **SMOTE (Synthetic Minority Over-sampling Technique)**: Synthesizes new minority instances mathematically based on k-nearest neighbors to enhance model generalization.

---

## 📈 Preprocessing Transformation Summary

| Metric | Before Preprocessing | After Preprocessing |
| :--- | :---: | :---: |
| **Dataset Rows** | 80,000 | 80,000 |
| **Dataset Columns** | 14 | 28 |
| **Missing Values (Total)** | 73,351 (in `Sale_Event`) | 0 |
| **Duplicate Rows** | 0 | 0 |
| **Reviews_Count Outliers (IQR)** | 2,628 | 0 (Capped/Winsorized) |
| **Categorical Columns Encoded** | 0 | 3 (Condition, Platform, Product_Category) |
| **Engineered Features** | 0 | 4 (`Price_Drop_USD`, `Price_Ratio`, `On_Sale`, `Implied_FX_Rate`) |
| **Continuous Features Scaled** | 0 | 2 (MinMax on Price, Z-score on Reviews) |

---

## 🚀 How to Run and Reproduce

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required data science packages using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy jupyter
```

### Steps to Run
1. Clone this repository to your local machine.
2. Open the directory in your terminal or Jupyter environment.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open **`Apple_Products_Pricing.ipynb`** and run all cells sequentially to regenerate the visualizations and the preprocessed dataset `apple_products_pricing_preprocessed.csv`.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the **[LICENSE](file:///c:/Users/gouth/Projects/AIML/LICENSE)** file for details.