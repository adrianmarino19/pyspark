import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_ecommerce_data(num_records=100000, start_date='2023-01-01', end_date='2024-12-31'):
    """
    Generate synthetic e-commerce transaction data for PySpark practice

    Parameters:
    - num_records: Number of transactions to generate
    - start_date: Start date for transactions
    - end_date: End date for transactions
    """

    # Configuration
    num_customers = max(100, num_records // 50)  # Approximately 50 transactions per customer
    num_products = max(500, num_records // 200)   # Variety of products

    # Reference data
    countries = [
        'United States', 'United Kingdom', 'Germany', 'France', 'Canada',
        'Australia', 'Japan', 'Brazil', 'India', 'China', 'Mexico', 'Spain',
        'Italy', 'Netherlands', 'Sweden', 'Singapore', 'South Korea', 'UAE'
    ]

    # Add some inconsistent country names for data cleaning practice
    countries_variations = {
        'United States': ['USA', 'US', 'United States', 'United States of America'],
        'United Kingdom': ['UK', 'United Kingdom', 'Great Britain', 'GB']
    }

    customer_segments = ['Premium', 'Regular', 'New']
    segment_weights = [0.15, 0.65, 0.20]  # Premium customers are fewer

    product_categories = {
        'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras', 'Gaming'],
        'Clothing': ['Men Apparel', 'Women Apparel', 'Kids Clothing', 'Shoes', 'Accessories'],
        'Books': ['Fiction', 'Non-Fiction', 'Technical', 'Comics', 'Educational'],
        'Home & Garden': ['Furniture', 'Kitchen', 'Decor', 'Garden Tools', 'Lighting'],
        'Sports': ['Fitness Equipment', 'Outdoor Gear', 'Team Sports', 'Water Sports'],
        'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrances', 'Tools'],
        'Toys': ['Educational', 'Action Figures', 'Board Games', 'Puzzles', 'Dolls']
    }

    payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer', 'Crypto']
    payment_weights = [0.45, 0.25, 0.20, 0.08, 0.02]

    shipping_methods = ['Standard', 'Express', 'Same Day', 'International']
    shipping_weights = [0.60, 0.25, 0.05, 0.10]

    device_types = ['Desktop', 'Mobile', 'Tablet']
    device_weights = [0.35, 0.55, 0.10]

    marketing_channels = ['Direct', 'Social Media', 'Email', 'Paid Search', 'Organic']
    marketing_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    return_reasons = ['Defective', 'Wrong Item', 'Not as Described', 'Changed Mind', 'Damaged in Shipping']

    # Generate customer IDs
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, num_customers + 1)]

    # Generate product IDs with categories
    products = []
    for category, subcategories in product_categories.items():
        for subcat in subcategories:
            # Generate multiple products per subcategory
            num_products_in_subcat = max(10, num_products // (len(product_categories) * 5))
            for i in range(num_products_in_subcat):
                products.append({
                    'product_id': f"PROD_{category[:3].upper()}_{str(len(products)).zfill(5)}",
                    'category': category,
                    'subcategory': subcat,
                    'base_price': np.random.lognormal(3.5, 1.2)  # Log-normal distribution for prices
                })

    # Generate transactions
    transactions = []

    # Convert date strings to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_range = (end_dt - start_dt).days

    # Track customer purchase history for realistic patterns
    customer_history = {cust_id: {
        'first_purchase': None,
        'segment': np.random.choice(customer_segments, p=segment_weights),
        'preferred_category': np.random.choice(list(product_categories.keys())),
        'purchase_count': 0
    } for cust_id in customer_ids}

    for i in range(num_records):
        # Select customer (some customers buy more frequently)
        if i % 10 == 0:  # 10% chance to be a frequent buyer
            customer_id = np.random.choice(customer_ids[:num_customers//10])  # Top 10% of customers
        else:
            customer_id = np.random.choice(customer_ids)

        customer_info = customer_history[customer_id]

        # Generate transaction date (with some seasonal patterns)
        days_offset = random.randint(0, date_range)
        transaction_date = start_dt + timedelta(days=days_offset)

        # Add hourly patterns (more transactions during day)
        hour_weights = np.concatenate([
            np.ones(6) * 0.3,   # 00:00-06:00 (low)
            np.ones(3) * 0.8,   # 06:00-09:00 (medium)
            np.ones(3) * 1.2,   # 09:00-12:00 (high)
            np.ones(3) * 1.5,   # 12:00-15:00 (highest)
            np.ones(3) * 1.3,   # 15:00-18:00 (high)
            np.ones(3) * 1.0,   # 18:00-21:00 (medium-high)
            np.ones(3) * 0.5    # 21:00-00:00 (medium-low)
        ])
        hour = np.random.choice(24, p=hour_weights/hour_weights.sum())
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        transaction_datetime = transaction_date.replace(hour=hour, minute=minute, second=second)

        # Update customer first purchase
        if customer_info['first_purchase'] is None:
            customer_info['first_purchase'] = transaction_datetime
            customer_info['segment'] = 'New'
        elif (transaction_datetime - customer_info['first_purchase']).days > 90:
            if customer_info['purchase_count'] > 5:
                customer_info['segment'] = np.random.choice(['Premium', 'Regular'], p=[0.3, 0.7])

        customer_info['purchase_count'] += 1

        # Select country (with some inconsistencies for data cleaning practice)
        country = np.random.choice(countries)
        if country in countries_variations and random.random() < 0.1:  # 10% chance of inconsistent naming
            country = np.random.choice(countries_variations[country])

        # Select product (with preference for customer's preferred category)
        if random.random() < 0.6:  # 60% chance to buy from preferred category
            category_products = [p for p in products if p['category'] == customer_info['preferred_category']]
        else:
            category_products = products

        product = np.random.choice(category_products)

        # Quantity (most orders are 1-3 items)
        quantity = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   p=[0.5, 0.25, 0.10, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])

        # Add some invalid quantities for data cleaning practice
        if random.random() < 0.001:  # 0.1% chance
            quantity = np.random.choice([0, -1])

        # Price (with some variation from base price)
        unit_price = product['base_price'] * np.random.uniform(0.8, 1.3)
        unit_price = round(unit_price, 2)

        # Add some invalid prices for data cleaning practice
        if random.random() < 0.001:  # 0.1% chance
            unit_price = np.random.choice([0, -10])

        # Discount (higher for Premium customers, seasonal)
        if customer_info['segment'] == 'Premium':
            discount_prob = 0.4
            discount_range = (0.05, 0.30)
        elif transaction_date.month in [11, 12, 6, 7]:  # Black Friday, Christmas, Summer sale
            discount_prob = 0.3
            discount_range = (0.10, 0.50)
        else:
            discount_prob = 0.15
            discount_range = (0.05, 0.20)

        if random.random() < discount_prob:
            discount_applied = round(np.random.uniform(*discount_range), 2)
        else:
            discount_applied = 0.0

        # Payment method
        payment_method = np.random.choice(payment_methods, p=payment_weights)

        # Shipping method (Same Day more likely for Premium customers)
        if customer_info['segment'] == 'Premium':
            shipping_weights_adj = [0.40, 0.35, 0.15, 0.10]
        else:
            shipping_weights_adj = shipping_weights
        shipping_method = np.random.choice(shipping_methods, p=shipping_weights_adj)

        # Shipping cost
        shipping_costs = {
            'Standard': np.random.uniform(5, 15),
            'Express': np.random.uniform(15, 30),
            'Same Day': np.random.uniform(25, 50),
            'International': np.random.uniform(20, 60)
        }
        shipping_cost = round(shipping_costs[shipping_method], 2)

        # Device type (mobile more likely for younger products like gaming)
        if product['subcategory'] in ['Gaming', 'Smartphones']:
            device_weights_adj = [0.20, 0.70, 0.10]
        else:
            device_weights_adj = device_weights
        device_type = np.random.choice(device_types, p=device_weights_adj)

        # Marketing channel
        marketing_channel = np.random.choice(marketing_channels, p=marketing_weights)

        # Returns (higher probability for certain categories)
        return_prob = {
            'Clothing': 0.15,
            'Electronics': 0.05,
            'Books': 0.02,
            'Home & Garden': 0.08,
            'Sports': 0.10,
            'Beauty': 0.12,
            'Toys': 0.06
        }

        is_returned = random.random() < return_prob.get(product['category'], 0.05)
        return_reason = np.random.choice(return_reasons) if is_returned else None

        # Create transaction record
        transaction = {
            'transaction_id': f"TRX_2024_{str(i+1).zfill(7)}",
            'customer_id': customer_id,
            'customer_country': country,
            'customer_segment': customer_info['segment'],
            'transaction_date': transaction_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'product_id': product['product_id'],
            'product_category': product['category'],
            'product_subcategory': product['subcategory'],
            'quantity': quantity,
            'unit_price': unit_price,
            'discount_applied': discount_applied,
            'payment_method': payment_method,
            'shipping_method': shipping_method,
            'shipping_cost': shipping_cost,
            'device_type': device_type,
            'marketing_channel': marketing_channel,
            'is_returned': is_returned,
            'return_reason': return_reason
        }

        transactions.append(transaction)

        # Add some duplicate transactions for data cleaning practice
        if random.random() < 0.002:  # 0.2% chance of duplicate
            transactions.append(transaction.copy())

    # Convert to DataFrame
    df = pd.DataFrame(transactions)

    # Add some null values for data cleaning practice
    null_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[null_indices, 'customer_country'] = None

    null_indices = np.random.choice(df.index, size=int(0.002 * len(df)), replace=False)
    df.loc[null_indices, 'product_category'] = None

    return df

def generate_supplementary_data(main_df):
    """
    Generate supplementary datasets for join operations
    """

    # Customer demographics (for enrichment)
    unique_customers = main_df['customer_id'].unique()
    customer_demographics = []

    for customer_id in unique_customers:
        age = np.random.randint(18, 75)
        gender = np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04])
        registration_date = pd.to_datetime('2020-01-01') + timedelta(days=random.randint(0, 1460))

        customer_demographics.append({
            'customer_id': customer_id,
            'age': age,
            'gender': gender,
            'registration_date': registration_date.strftime('%Y-%m-%d'),
            'email_subscribed': np.random.choice([True, False], p=[0.7, 0.3]),
            'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'],
                                            p=[0.50, 0.30, 0.15, 0.05])
        })

    # Product details (for enrichment)
    unique_products = main_df[['product_id', 'product_category', 'product_subcategory']].drop_duplicates()
    product_details = []

    for _, row in unique_products.iterrows():
        avg_rating = round(np.random.uniform(3.0, 5.0), 1)
        review_count = np.random.randint(5, 5000)
        weight_kg = round(np.random.uniform(0.1, 25.0), 2)

        product_details.append({
            'product_id': row['product_id'],
            'product_name': f"{row['product_subcategory']} Item {row['product_id'][-4:]}",
            'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']),
            'average_rating': avg_rating,
            'review_count': review_count,
            'weight_kg': weight_kg,
            'is_eco_friendly': np.random.choice([True, False], p=[0.3, 0.7])
        })

    customer_df = pd.DataFrame(customer_demographics)
    product_df = pd.DataFrame(product_details)

    return customer_df, product_df

# Generate the main dataset
print("Generating main transaction dataset...")
main_df = generate_ecommerce_data(num_records=100000)  # 100K records for practice

# Generate supplementary datasets
print("Generating supplementary datasets...")
customer_df, product_df = generate_supplementary_data(main_df)

# Save to CSV files
print("\nSaving datasets to CSV files...")
main_df.to_csv('ecommerce_transactions.csv', index=False)
customer_df.to_csv('customer_demographics.csv', index=False)
product_df.to_csv('product_details.csv', index=False)

# Create a smaller sample for quick testing
sample_df = main_df.sample(n=1000, random_state=42)
sample_df.to_csv('ecommerce_transactions_sample.csv', index=False)

# Display statistics
print("\n" + "="*60)
print("DATASET GENERATION COMPLETE!")
print("="*60)
print(f"\n📊 Main Dataset Statistics:")
print(f"   - Total transactions: {len(main_df):,}")
print(f"   - Date range: {main_df['transaction_date'].min()} to {main_df['transaction_date'].max()}")
print(f"   - Unique customers: {main_df['customer_id'].nunique():,}")
print(f"   - Unique products: {main_df['product_id'].nunique():,}")
print(f"   - Product categories: {main_df['product_category'].nunique()}")
print(f"   - Countries: {main_df['customer_country'].nunique()}")
print(f"   - Return rate: {(main_df['is_returned'].sum() / len(main_df) * 100):.2f}%")
print(f"   - Avg transaction value: ${(main_df['quantity'] * main_df['unit_price']).mean():.2f}")

print(f"\n📊 Supplementary Datasets:")
print(f"   - Customer demographics: {len(customer_df):,} records")
print(f"   - Product details: {len(product_df):,} records")

print("\n📁 Files created:")
print("   1. ecommerce_transactions.csv (main dataset - 100K records)")
print("   2. ecommerce_transactions_sample.csv (sample - 1K records for testing)")
print("   3. customer_demographics.csv (customer enrichment data)")
print("   4. product_details.csv (product enrichment data)")

print("\n✅ Data Quality Issues Included (for practice):")
print("   - Inconsistent country names (US vs USA vs United States)")
print("   - Some null values in various columns")
print("   - Duplicate transactions (~0.2%)")
print("   - Invalid quantities and prices (~0.1%)")
print("   - Missing return reasons when is_returned=False")

print("\n🚀 Next Steps:")
print("   1. Upload these CSV files to Google Colab")
print("   2. Install PySpark: !pip install pyspark")
print("   3. Start with the practice questions!")

# Show sample of the data
print("\n📋 Sample of main dataset:")
print(main_df.head(10))

print("\n📋 Data types:")
print(main_df.dtypes)
