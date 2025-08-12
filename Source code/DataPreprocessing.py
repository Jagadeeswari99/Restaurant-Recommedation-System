import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RestaurantDataPreprocessor:
    def __init__(self):
        # Store original data
        self.train_customers = None
        self.train_locations = None
        self.train_orders = None
        self.test_customers = None
        self.test_locations = None
        self.vendors = None
        
        # Store processed features
        self.customer_features = None
        self.vendor_features = None
        self.location_clusters = None
        self.customer_order_history = None
        
        # Store encoders and scalers
        self.scalers = {}
        self.encoders = {}
    
    def load_data(self):
        """Load the datasets using the same paths from exploration"""
        try:
            base_train_path = r"C:\JAGA\SoulpageIT PROJECT1\Assignment\Train"
            base_test_path = r"C:\JAGA\SoulpageIT PROJECT1\Assignment\Test"
            
            self.train_customers = pd.read_csv(f"{base_train_path}\\train_customers.csv")
            self.train_locations = pd.read_csv(f"{base_train_path}\\train_locations.csv")
            self.train_orders = pd.read_csv(f"{base_train_path}\\orders.csv")
            self.test_customers = pd.read_csv(f"{base_test_path}\\test_customers.csv")
            self.test_locations = pd.read_csv(f"{base_test_path}\\test_locations.csv")
            self.vendors = pd.read_csv(f"{base_train_path}\\vendors.csv")
            
            print("✅ Data loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def clean_customer_data(self):
        """Clean and preprocess customer demographics"""
        print("\n🧹 CLEANING CUSTOMER DATA...")
        
        # Combine train and test customers for consistent processing
        all_customers = pd.concat([
            self.train_customers.assign(dataset='train'),
            self.test_customers.assign(dataset='test')
        ], ignore_index=True)
        
        # 1. Clean gender data
        print("  • Standardizing gender values...")
        gender_mapping = {
            'Male': 'Male', 'male': 'Male',
            'Female': 'Female', 'Female': 'Female', 
            '?????': 'Unknown'
        }
        all_customers['gender_clean'] = all_customers['gender'].map(gender_mapping).fillna('Unknown')
        
        # 2. Clean age data (remove outliers)
        print("  • Cleaning age data...")
        current_year = datetime.now().year
        all_customers['age'] = current_year - all_customers['dob']
        
        # Remove impossible ages (keep 13-100 years old)
        all_customers['age_clean'] = all_customers['age'].clip(13, 100)
        all_customers['age_clean'] = all_customers['age_clean'].fillna(all_customers['age_clean'].median())
        
        # 3. Create age groups
        all_customers['age_group'] = pd.cut(
            all_customers['age_clean'], 
            bins=[0, 25, 35, 45, 55, 100], 
            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder']
        )
        
        # 4. Clean account status
        all_customers['is_verified'] = all_customers['verified'].fillna(0)
        all_customers['is_active'] = all_customers['status'].fillna(1)
        
        # 5. Create account age (days since creation)
        all_customers['created_at'] = pd.to_datetime(all_customers['created_at'], errors='coerce')
        reference_date = pd.to_datetime('2024-10-15')  # Based on data inspection
        all_customers['account_age_days'] = (reference_date - all_customers['created_at']).dt.days
        all_customers['account_age_days'] = all_customers['account_age_days'].fillna(
            all_customers['account_age_days'].median()
        ).clip(0, 3650)  # Max 10 years
        
        # Split back into train and test
        train_mask = all_customers['dataset'] == 'train'
        self.train_customers_clean = all_customers[train_mask].drop('dataset', axis=1)
        self.test_customers_clean = all_customers[~train_mask].drop('dataset', axis=1)
        
        print(f"  ✅ Cleaned {len(all_customers)} customer records")
    
    def process_location_data(self):
        """Process and cluster location data"""
        print("\n📍 PROCESSING LOCATION DATA...")
        
        # Combine all locations
        all_locations = pd.concat([
            self.train_locations.assign(dataset='train'),
            self.test_locations.assign(dataset='test')
        ], ignore_index=True)
        
        # 1. Clean location types
        all_locations['location_type_clean'] = all_locations['location_type'].fillna('Unknown')
        
        # 2. Remove outlier coordinates (based on exploration)
        print("  • Removing coordinate outliers...")
        lat_q1, lat_q99 = all_locations['latitude'].quantile([0.01, 0.99])
        lon_q1, lon_q99 = all_locations['longitude'].quantile([0.01, 0.99])
        
        coord_mask = (
            (all_locations['latitude'].between(lat_q1, lat_q99)) &
            (all_locations['longitude'].between(lon_q1, lon_q99))
        )
        all_locations = all_locations[coord_mask].copy()
        
        # 3. Create location clusters using K-means
        print("  • Creating location clusters...")
        valid_coords = all_locations[['latitude', 'longitude']].dropna()
        
        if len(valid_coords) > 0:
            # Determine optimal number of clusters (10-20 seems reasonable for delivery areas)
            n_clusters = min(15, max(5, len(valid_coords) // 1000))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            all_locations.loc[valid_coords.index, 'location_cluster'] = kmeans.fit_predict(valid_coords)
            all_locations['location_cluster'] = all_locations['location_cluster'].fillna(-1).astype(int)
            
            # Store cluster centers for test data
            self.location_clusters = {
                'model': kmeans,
                'n_clusters': n_clusters
            }
        
        # 4. Calculate customer location features
        print("  • Calculating customer location features...")
        location_features = []
        
        for dataset in ['train', 'test']:
            dataset_locations = all_locations[all_locations['dataset'] == dataset]
            
            customer_loc_features = dataset_locations.groupby('customer_id').agg({
                'location_number': ['count', 'max'],
                'location_cluster': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1,
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            
            # Flatten column names
            customer_loc_features.columns = [
                'customer_id', 'num_locations', 'max_location_num', 
                'primary_cluster', 'avg_latitude', 'avg_longitude'
            ]
            
            # Add location type distribution
            location_type_dist = dataset_locations.groupby('customer_id')['location_type_clean'].apply(
                lambda x: x.value_counts(normalize=True).to_dict()
            ).reset_index()
            
            for loc_type in ['Home', 'Work', 'Other', 'Unknown']:
                customer_loc_features[f'pct_{loc_type.lower()}'] = location_type_dist['location_type_clean'].apply(
                    lambda x: x.get(loc_type, 0) if isinstance(x, dict) else 0
                )
            
            customer_loc_features['dataset'] = dataset
            location_features.append(customer_loc_features)
        
        self.customer_location_features = pd.concat(location_features, ignore_index=True)
        print(f"  ✅ Processed {len(all_locations)} location records")
    
    def analyze_order_patterns(self):
        """Analyze customer ordering patterns and create features"""
        print("\n🛒 ANALYZING ORDER PATTERNS...")
        
        # Customer-level order aggregations
        print("  • Creating customer order history features...")
        customer_stats = self.train_orders.groupby('customer_id').agg({
            'order_id': 'count',
            'grand_total': ['mean', 'std', 'sum', 'min', 'max'],
            'item_count': ['mean', 'sum'],
            'vendor_id': ['nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1],
            'deliverydistance': ['mean', 'std'],
            'preparationtime': ['mean', 'std'],
            'vendor_rating': ['mean', 'count'],
            'driver_rating': ['mean'],
            'LOCATION_NUMBER': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = [
            'customer_id', 'total_orders', 'avg_order_value', 'std_order_value', 
            'total_spent', 'min_order_value', 'max_order_value', 'avg_items_per_order',
            'total_items', 'unique_vendors', 'favorite_vendor', 'avg_delivery_distance',
            'std_delivery_distance', 'avg_prep_time', 'std_prep_time', 
            'avg_vendor_rating', 'num_ratings_given', 'avg_driver_rating', 'primary_location'
        ]
        
        # Fill missing values
        numeric_cols = customer_stats.select_dtypes(include=[np.number]).columns
        customer_stats[numeric_cols] = customer_stats[numeric_cols].fillna(0)
        
        # Create derived features
        print("  • Creating derived order features...")
        customer_stats['order_frequency'] = customer_stats['total_orders'] / customer_stats['total_orders'].max()
        customer_stats['avg_order_value_normalized'] = customer_stats['avg_order_value'] / customer_stats['avg_order_value'].median()
        customer_stats['is_high_value_customer'] = (customer_stats['total_spent'] > customer_stats['total_spent'].quantile(0.8)).astype(int)
        customer_stats['vendor_loyalty'] = customer_stats['total_orders'] / (customer_stats['unique_vendors'] + 1)
        customer_stats['rating_activity'] = (customer_stats['num_ratings_given'] > 0).astype(int)
        
        # Analyze favorite patterns
        print("  • Analyzing favorite and rating patterns...")
        favorite_stats = self.train_orders.groupby('customer_id')['is_favorite'].apply(
            lambda x: (x == 'Yes').sum() / len(x) if len(x) > 0 else 0
        ).reset_index()
        favorite_stats.columns = ['customer_id', 'favorite_rate']
        
        rating_stats = self.train_orders.groupby('customer_id')['is_rated'].apply(
            lambda x: (x == 'Yes').sum() / len(x) if len(x) > 0 else 0
        ).reset_index()
        rating_stats.columns = ['customer_id', 'rating_rate']
        
        # Merge all customer order features
        self.customer_order_history = customer_stats.merge(
            favorite_stats, on='customer_id', how='left'
        ).merge(
            rating_stats, on='customer_id', how='left'
        )
        
        self.customer_order_history['favorite_rate'] = self.customer_order_history['favorite_rate'].fillna(0)
        self.customer_order_history['rating_rate'] = self.customer_order_history['rating_rate'].fillna(0)
        
        print(f"  ✅ Created order features for {len(self.customer_order_history)} customers")
    
    def process_vendor_features(self):
        """Create comprehensive vendor features"""
        print("\n🏪 PROCESSING VENDOR FEATURES...")
        
        vendor_features = self.vendors.copy()
        
        # 1. Basic vendor characteristics
        print("  • Processing basic vendor features...")
        vendor_features['is_restaurant'] = (vendor_features['vendor_category_en'] == 'Restaurants').astype(int)
        vendor_features['has_delivery_charge'] = (vendor_features['delivery_charge'] > 0).astype(int)
        vendor_features['is_open'] = vendor_features['is_open'].fillna(0)
        vendor_features['is_verified_vendor'] = vendor_features['verified'].fillna(0)
        vendor_features['vendor_rating_clean'] = vendor_features['vendor_rating'].fillna(vendor_features['vendor_rating'].median())
        
        # 2. Parse vendor tags into features
        print("  • Processing vendor tags...")
        if 'vendor_tag_name' in vendor_features.columns:
            # Create binary features for common tags
            all_tags = []
            for tags in vendor_features['vendor_tag_name'].dropna():
                all_tags.extend([tag.strip() for tag in str(tags).split(',')])
            
            # Get most common tags
            from collections import Counter
            tag_counts = Counter(all_tags)
            common_tags = [tag for tag, count in tag_counts.most_common(20)]
            
            for tag in common_tags:
                vendor_features[f'tag_{tag.replace(" ", "_").lower()}'] = vendor_features['vendor_tag_name'].apply(
                    lambda x: 1 if isinstance(x, str) and tag in x else 0
                )
        
        # 3. Calculate vendor popularity from orders
        print("  • Calculating vendor popularity metrics...")
        vendor_popularity = self.train_orders.groupby('vendor_id').agg({
            'order_id': 'count',
            'customer_id': 'nunique',
            'grand_total': ['mean', 'sum'],
            'vendor_rating': 'mean',
            'is_favorite': lambda x: (x == 'Yes').sum(),
            'deliverydistance': 'mean'
        }).reset_index()
        
        # Flatten columns
        vendor_popularity.columns = [
            'id', 'total_orders', 'unique_customers', 'avg_order_value', 
            'total_revenue', 'avg_customer_rating', 'total_favorites', 'avg_delivery_distance'
        ]
        
        # Fill missing values
        vendor_popularity = vendor_popularity.fillna(0)
        
        # Create popularity scores
        vendor_popularity['popularity_score'] = (
            vendor_popularity['total_orders'] * 0.4 + 
            vendor_popularity['unique_customers'] * 0.3 + 
            vendor_popularity['total_favorites'] * 0.3
        )
        
        vendor_popularity['revenue_rank'] = vendor_popularity['total_revenue'].rank(pct=True)
        vendor_popularity['order_rank'] = vendor_popularity['total_orders'].rank(pct=True)
        
        # Merge with vendor features
        self.vendor_features = vendor_features.merge(
            vendor_popularity, on='id', how='left'
        )
        
        # Fill missing values for vendors with no orders
        popularity_cols = ['total_orders', 'unique_customers', 'avg_order_value', 
                          'total_revenue', 'avg_customer_rating', 'total_favorites', 
                          'avg_delivery_distance', 'popularity_score', 'revenue_rank', 'order_rank']
        
        self.vendor_features[popularity_cols] = self.vendor_features[popularity_cols].fillna(0)
        
        print(f"  ✅ Created features for {len(self.vendor_features)} vendors")
    
    def create_customer_features(self):
        """Combine all customer features into final dataset"""
        print("\n👤 CREATING COMPREHENSIVE CUSTOMER FEATURES...")
        
        # Start with cleaned customer demographics
        customer_features = self.train_customers_clean[['customer_id', 'gender_clean', 'age_clean', 
                                                       'age_group', 'is_verified', 'is_active', 
                                                       'account_age_days']].copy()
        
        # Add location features
        customer_features = customer_features.merge(
            self.customer_location_features[self.customer_location_features['dataset'] == 'train'],
            on='customer_id', how='left'
        )
        
        # Add order history features
        customer_features = customer_features.merge(
            self.customer_order_history, on='customer_id', how='left'
        )
        
        # Fill missing values for customers with no orders
        order_cols = self.customer_order_history.columns.drop('customer_id')
        customer_features[order_cols] = customer_features[order_cols].fillna(0)
        
        # Fill missing location features
        location_cols = ['num_locations', 'max_location_num', 'primary_cluster', 
                        'pct_home', 'pct_work', 'pct_other', 'pct_unknown']
        customer_features[location_cols] = customer_features[location_cols].fillna(0)
        
        # Encode categorical variables
        print("  • Encoding categorical variables...")
        le_gender = LabelEncoder()
        customer_features['gender_encoded'] = le_gender.fit_transform(customer_features['gender_clean'].fillna('Unknown'))
        
        le_age_group = LabelEncoder()
        customer_features['age_group_encoded'] = le_age_group.fit_transform(customer_features['age_group'].astype(str))
        
        self.encoders['gender'] = le_gender
        self.encoders['age_group'] = le_age_group
        
        # Create final feature set (numeric only for modeling)
        feature_cols = [col for col in customer_features.columns 
                       if customer_features[col].dtype in ['int64', 'float64']]
        
        self.customer_features = customer_features[['customer_id'] + feature_cols].copy()
        
        print(f"  ✅ Created {len(feature_cols)} features for {len(self.customer_features)} customers")
        
        # Create test customer features
        self.create_test_customer_features()
    
    def create_test_customer_features(self):
        """Create features for test customers (cold start scenario)"""
        print("\n🧪 CREATING TEST CUSTOMER FEATURES...")
        
        # Start with cleaned test customer demographics
        test_features = self.test_customers_clean[['customer_id', 'gender_clean', 'age_clean', 
                                                  'age_group', 'is_verified', 'is_active', 
                                                  'account_age_days']].copy()
        
        # Add location features
        test_features = test_features.merge(
            self.customer_location_features[self.customer_location_features['dataset'] == 'test'],
            on='customer_id', how='left'
        )
        
        # Fill missing location features
        location_cols = ['num_locations', 'max_location_num', 'primary_cluster', 
                        'pct_home', 'pct_work', 'pct_other', 'pct_unknown']
        test_features[location_cols] = test_features[location_cols].fillna(0)
        
        # For order history features, set to 0 (cold start)
        order_cols = self.customer_order_history.columns.drop('customer_id')
        for col in order_cols:
            test_features[col] = 0
        
        # Encode categorical variables using fitted encoders
        test_features['gender_encoded'] = self.encoders['gender'].transform(test_features['gender_clean'].fillna('Unknown'))
        test_features['age_group_encoded'] = self.encoders['age_group'].transform(test_features['age_group'].astype(str))
        
        # Create final feature set (same columns as training)
        feature_cols = [col for col in test_features.columns 
                       if test_features[col].dtype in ['int64', 'float64']]
        
        # Ensure same columns as training data
        train_feature_cols = [col for col in self.customer_features.columns if col != 'customer_id']
        
        for col in train_feature_cols:
            if col not in test_features.columns:
                test_features[col] = 0
        
        self.test_customer_features = test_features[['customer_id'] + train_feature_cols].copy()
        
        print(f"  ✅ Created features for {len(self.test_customer_features)} test customers")
    
    def scale_features(self):
        """Scale numerical features for modeling"""
        print("\n📏 SCALING FEATURES...")
        
        # Get numeric columns (exclude customer_id)
        numeric_cols = [col for col in self.customer_features.columns 
                       if col != 'customer_id' and self.customer_features[col].dtype in ['int64', 'float64']]
        
        # Scale training features
        scaler = StandardScaler()
        self.customer_features[numeric_cols] = scaler.fit_transform(self.customer_features[numeric_cols])
        
        # Scale test features with same scaler
        self.test_customer_features[numeric_cols] = scaler.transform(self.test_customer_features[numeric_cols])
        
        # Scale vendor features
        vendor_numeric_cols = [col for col in self.vendor_features.columns 
                              if col != 'id' and self.vendor_features[col].dtype in ['int64', 'float64']]
        
        vendor_scaler = StandardScaler()
        self.vendor_features[vendor_numeric_cols] = vendor_scaler.fit_transform(self.vendor_features[vendor_numeric_cols])
        
        self.scalers['customer'] = scaler
        self.scalers['vendor'] = vendor_scaler
        
        print(f"  ✅ Scaled {len(numeric_cols)} customer features and {len(vendor_numeric_cols)} vendor features")
    
    def create_interaction_matrix(self):
        """Create customer-vendor interaction matrix"""
        print("\n🔗 CREATING INTERACTION MATRIX...")
        
        # Create interaction matrix from orders
        interactions = self.train_orders.groupby(['customer_id', 'vendor_id']).agg({
            'order_id': 'count',
            'grand_total': 'sum',
            'is_favorite': lambda x: (x == 'Yes').sum(),
            'vendor_rating': 'mean'
        }).reset_index()
        
        interactions.columns = ['customer_id', 'vendor_id', 'order_count', 'total_spent', 'favorite_count', 'avg_rating']
        interactions = interactions.fillna(0)
        
        # Create interaction score
        interactions['interaction_score'] = (
            interactions['order_count'] * 0.4 +
            interactions['total_spent'] / interactions['total_spent'].max() * 0.3 +
            interactions['favorite_count'] * 0.2 +
            interactions['avg_rating'] / 5.0 * 0.1
        )
        
        self.interaction_matrix = interactions
        print(f"  ✅ Created interaction matrix with {len(interactions)} customer-vendor pairs")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🔧 STARTING DATA PREPROCESSING AND FEATURE ENGINEERING")
        print("=" * 70)
        
        if not self.load_data():
            return False
        
        # Run all preprocessing steps
        self.clean_customer_data()
        self.process_location_data()
        self.analyze_order_patterns()
        self.process_vendor_features()
        self.create_customer_features()
        self.scale_features()
        self.create_interaction_matrix()
        
        print(f"\n{'='*70}")
        print("✅ DATA PREPROCESSING COMPLETE!")
        print(f"{'='*70}")
        
        self.print_feature_summary()
        return True
    
    def print_feature_summary(self):
        """Print summary of created features"""
        print(f"\n📊 FEATURE ENGINEERING SUMMARY:")
        print("-" * 40)
        
        print(f"Customer Features:")
        print(f"  • Training customers: {len(self.customer_features):,}")
        print(f"  • Test customers: {len(self.test_customer_features):,}")
        print(f"  • Features per customer: {len(self.customer_features.columns)-1}")
        
        print(f"\nVendor Features:")
        print(f"  • Total vendors: {len(self.vendor_features):,}")
        print(f"  • Features per vendor: {len(self.vendor_features.columns)-1}")
        
        print(f"\nInteraction Data:")
        print(f"  • Customer-vendor interactions: {len(self.interaction_matrix):,}")
        
        print(f"\nLocation Clusters:")
        print(f"  • Number of clusters: {self.location_clusters['n_clusters']}")
        
        print(f"\n🎯 READY FOR MODELING!")
        print("   • Features are scaled and ready")
        print("   • Categorical variables encoded")
        print("   • Cold start problem addressed for test customers")
        print("   • Geographic clustering completed")
    
    def save_processed_data(self, output_dir="processed_data"):
        """Save processed datasets for modeling"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.customer_features.to_csv(f"{output_dir}/train_customer_features.csv", index=False)
        self.test_customer_features.to_csv(f"{output_dir}/test_customer_features.csv", index=False)
        self.vendor_features.to_csv(f"{output_dir}/vendor_features.csv", index=False)
        self.interaction_matrix.to_csv(f"{output_dir}/interaction_matrix.csv", index=False)
        
        print(f"💾 Processed data saved to {output_dir}/ directory")

# Usage
preprocessor = RestaurantDataPreprocessor()

print("🔧 Data Preprocessor Ready!")
print("Run: preprocessor.run_preprocessing() to start feature engineering!")

# Actually run the preprocessing
if __name__ == "__main__":
    success = preprocessor.run_preprocessing()
    
    if success:
        # Save the processed data
        preprocessor.save_processed_data()
        print("\n🎯 PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("📁 Processed files saved in 'processed_data/' directory")
        print("🚀 Ready for Step 3: Model Development!")
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.")