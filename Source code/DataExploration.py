import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")

class RestaurantDataExplorer:
    def __init__(self):
        self.train_customers = None
        self.train_locations = None
        self.train_orders = None
        self.test_customers = None
        self.test_locations = None
        self.vendors = None
    
    def load_data(self, file_paths=None):
        """
        Load all datasets
        file_paths: dict with keys: 'train_customers', 'train_locations', 'train_orders', 
                   'test_customers', 'test_locations', 'vendors'
        If file_paths is None, will use default paths based on your directory structure
        """
        try:
            # If no file_paths provided, use your directory structure
            if file_paths is None:
                base_train_path = r"C:\JAGA\SoulpageIT PROJECT1\Assignment\Train"
                base_test_path = r"C:\JAGA\SoulpageIT PROJECT1\Assignment\Test"
                
                file_paths = {
                    'train_customers': f"{base_train_path}\\train_customers.csv",
                    'train_locations': f"{base_train_path}\\train_locations.csv",
                    'train_orders': f"{base_train_path}\\orders.csv",
                    'test_customers': f"{base_test_path}\\test_customers.csv",
                    'test_locations': f"{base_test_path}\\test_locations.csv",
                    'vendors': f"{base_train_path}\\vendors.csv"
                }
            
            self.train_customers = pd.read_csv(file_paths['train_customers'])
            self.train_locations = pd.read_csv(file_paths['train_locations'])
            self.train_orders = pd.read_csv(file_paths['train_orders'])
            self.test_customers = pd.read_csv(file_paths['test_customers'])
            self.test_locations = pd.read_csv(file_paths['test_locations'])
            self.vendors = pd.read_csv(file_paths['vendors'])
            
            print("✅ All datasets loaded successfully!")
            self.print_dataset_shapes()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Please check your file paths and ensure all CSV files exist.")
    
    def print_dataset_shapes(self):
        """Print basic information about dataset sizes"""
        datasets = {
            'Train Customers': self.train_customers,
            'Train Locations': self.train_locations,
            'Train Orders': self.train_orders,
            'Test Customers': self.test_customers,
            'Test Locations': self.test_locations,
            'Vendors': self.vendors
        }
        
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        
        for name, df in datasets.items():
            if df is not None:
                print(f"{name:.<20} {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    def explore_dataset_structure(self):
        """Explore the structure of each dataset"""
        datasets = {
            'Train Customers': self.train_customers,
            'Train Locations': self.train_locations,
            'Train Orders': self.train_orders,
            'Test Customers': self.test_customers,
            'Test Locations': self.test_locations,
            'Vendors': self.vendors
        }
        
        for name, df in datasets.items():
            if df is not None:
                print(f"\n{'='*60}")
                print(f"{name.upper()} - STRUCTURE ANALYSIS")
                print(f"{'='*60}")
                
                print(f"\nColumns ({len(df.columns)}):")
                print("-" * 30)
                for i, col in enumerate(df.columns, 1):
                    print(f"{i:2d}. {col}")
                
                print(f"\nData Types:")
                print("-" * 30)
                print(df.dtypes)
                
                print(f"\nBasic Statistics:")
                print("-" * 30)
                print(df.describe(include='all'))
                
                print(f"\nMissing Values:")
                print("-" * 30)
                missing = df.isnull().sum()
                missing_pct = (missing / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Missing Count': missing,
                    'Missing %': missing_pct
                }).sort_values('Missing Count', ascending=False)
                print(missing_df[missing_df['Missing Count'] > 0])
    
    def analyze_customer_data(self):
        """Analyze customer demographics and patterns"""
        print(f"\n{'='*60}")
        print("CUSTOMER DATA ANALYSIS")
        print(f"{'='*60}")
        
        # Train customers analysis
        if self.train_customers is not None:
            print("\nTRAIN CUSTOMERS:")
            print("-" * 20)
            
            # Gender distribution
            if 'gender' in self.train_customers.columns:
                gender_dist = self.train_customers['gender'].value_counts()
                print(f"\nGender Distribution:")
                print(gender_dist)
                print(f"Gender Distribution (%):")
                print((gender_dist / len(self.train_customers) * 100).round(2))
            
            # Age analysis (if dob available)
            if 'dob' in self.train_customers.columns:
                current_year = datetime.now().year
                ages = current_year - pd.to_numeric(self.train_customers['dob'], errors='coerce')
                ages = ages.dropna()
                
                print(f"\nAge Statistics:")
                print(f"Mean age: {ages.mean():.1f}")
                print(f"Median age: {ages.median():.1f}")
                print(f"Age range: {ages.min():.0f} - {ages.max():.0f}")
            
            # Status and verification
            if 'status' in self.train_customers.columns:
                print(f"\nAccount Status:")
                print(self.train_customers['status'].value_counts())
            
            if 'verified' in self.train_customers.columns:
                print(f"\nVerification Status:")
                print(self.train_customers['verified'].value_counts())
            
            # Language preferences
            if 'language' in self.train_customers.columns:
                print(f"\nTop 10 Languages:")
                print(self.train_customers['language'].value_counts().head(10))
        
        # Compare train vs test customers
        if self.train_customers is not None and self.test_customers is not None:
            print(f"\nTRAIN vs TEST CUSTOMERS COMPARISON:")
            print("-" * 35)
            print(f"Train customers: {len(self.train_customers):,}")
            print(f"Test customers: {len(self.test_customers):,}")
            
            # Check if test customers are subset of train customers
            if 'customer_id' in self.train_customers.columns and 'customer_id' in self.test_customers.columns:
                overlap = set(self.test_customers['customer_id']).intersection(
                    set(self.train_customers['customer_id']))
                print(f"Customer ID overlap: {len(overlap):,} ({len(overlap)/len(self.test_customers)*100:.1f}%)")
        else:
            print(f"\n❌ Cannot compare customers - data not loaded properly")
    
    def analyze_location_data(self):
        """Analyze location patterns"""
        print(f"\n{'='*60}")
        print("LOCATION DATA ANALYSIS")
        print(f"{'='*60}")
        
        # Location counts per customer
        if self.train_locations is not None:
            locations_per_customer = self.train_locations.groupby('customer_id')['location_number'].count()
            
            print(f"\nLocations per Customer (Train):")
            print("-" * 30)
            print(f"Mean: {locations_per_customer.mean():.2f}")
            print(f"Median: {locations_per_customer.median():.2f}")
            print(f"Max: {locations_per_customer.max()}")
            print(f"\nDistribution:")
            print(locations_per_customer.value_counts().sort_index())
            
            # Location types
            if 'location_type' in self.train_locations.columns:
                print(f"\nLocation Types (Train):")
                print(self.train_locations['location_type'].value_counts())
        
        if self.test_locations is not None:
            test_locations_per_customer = self.test_locations.groupby('customer_id')['location_number'].count()
            
            print(f"\nLocations per Customer (Test):")
            print("-" * 30)
            print(f"Mean: {test_locations_per_customer.mean():.2f}")
            print(f"Median: {test_locations_per_customer.median():.2f}")
            print(f"Max: {test_locations_per_customer.max()}")
            
            if 'location_type' in self.test_locations.columns:
                print(f"\nLocation Types (Test):")
                print(self.test_locations['location_type'].value_counts())
    
    def analyze_order_patterns(self):
        """Analyze order history and patterns"""
        print(f"\n{'='*60}")
        print("ORDER PATTERNS ANALYSIS")
        print(f"{'='*60}")
        
        if self.train_orders is not None:
            print(f"\nTotal Orders: {len(self.train_orders):,}")
            print(f"Unique Customers: {self.train_orders['customer_id'].nunique():,}")
            print(f"Unique Vendors: {self.train_orders['vendor_id'].nunique():,}")
            
            # Orders per customer
            orders_per_customer = self.train_orders.groupby('customer_id').size()
            print(f"\nOrders per Customer:")
            print(f"Mean: {orders_per_customer.mean():.2f}")
            print(f"Median: {orders_per_customer.median():.2f}")
            print(f"Max: {orders_per_customer.max()}")
            
            # Order value analysis
            if 'grand_total' in self.train_orders.columns:
                print(f"\nOrder Value Statistics:")
                print(f"Mean: ${self.train_orders['grand_total'].mean():.2f}")
                print(f"Median: ${self.train_orders['grand_total'].median():.2f}")
                print(f"Max: ${self.train_orders['grand_total'].max():.2f}")
            
            # Popular vendors
            vendor_popularity = self.train_orders['vendor_id'].value_counts()
            print(f"\nTop 10 Most Popular Vendors:")
            print(vendor_popularity.head(10))
            
            # Favorite and rating patterns
            if 'is_favorite' in self.train_orders.columns:
                favorite_rate = self.train_orders['is_favorite'].value_counts(normalize=True) * 100
                print(f"\nFavorite Status:")
                print(favorite_rate.round(2))
            
            if 'is_rated' in self.train_orders.columns:
                rating_rate = self.train_orders['is_rated'].value_counts(normalize=True) * 100
                print(f"\nRating Status:")
                print(rating_rate.round(2))
    
    def analyze_vendor_data(self):
        """Analyze vendor characteristics"""
        print(f"\n{'='*60}")
        print("VENDOR DATA ANALYSIS")
        print(f"{'='*60}")
        
        if self.vendors is not None:
            print(f"\nTotal Vendors: {len(self.vendors):,}")
            
            # Vendor tags
            if 'vendor_tag_name' in self.vendors.columns:
                print(f"\nTop 15 Vendor Tags:")
                print(self.vendors['vendor_tag_name'].value_counts().head(15))
            
            # Geographic distribution
            if 'latitude' in self.vendors.columns and 'longitude' in self.vendors.columns:
                lat_stats = self.vendors['latitude'].describe()
                lon_stats = self.vendors['longitude'].describe()
                
                print(f"\nGeographic Distribution:")
                print(f"Latitude range: {lat_stats['min']:.4f} to {lat_stats['max']:.4f}")
                print(f"Longitude range: {lon_stats['min']:.4f} to {lon_stats['max']:.4f}")
    
    def create_summary_visualizations(self):
        """Create key visualizations for data understanding"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Restaurant Recommendation Data Overview', fontsize=16, fontweight='bold')
        
        # 1. Customer Gender Distribution
        if self.train_customers is not None and 'gender' in self.train_customers.columns:
            gender_counts = self.train_customers['gender'].value_counts()
            axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Customer Gender Distribution')
        
        # 2. Location Types
        if self.train_locations is not None and 'location_type' in self.train_locations.columns:
            location_counts = self.train_locations['location_type'].value_counts()
            axes[0, 1].bar(location_counts.index, location_counts.values)
            axes[0, 1].set_title('Location Types Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Orders per Customer Distribution
        if self.train_orders is not None:
            orders_per_customer = self.train_orders.groupby('customer_id').size()
            axes[0, 2].hist(orders_per_customer, bins=30, edgecolor='black')
            axes[0, 2].set_title('Orders per Customer Distribution')
            axes[0, 2].set_xlabel('Number of Orders')
            axes[0, 2].set_ylabel('Number of Customers')
        
        # 4. Top Vendors by Order Count
        if self.train_orders is not None:
            top_vendors = self.train_orders['vendor_id'].value_counts().head(10)
            axes[1, 0].barh(range(len(top_vendors)), top_vendors.values)
            axes[1, 0].set_yticks(range(len(top_vendors)))
            axes[1, 0].set_yticklabels([f'Vendor {v}' for v in top_vendors.index])
            axes[1, 0].set_title('Top 10 Vendors by Order Count')
            axes[1, 0].set_xlabel('Number of Orders')
        
        # 5. Order Value Distribution
        if self.train_orders is not None and 'grand_total' in self.train_orders.columns:
            # Remove outliers for better visualization
            q99 = self.train_orders['grand_total'].quantile(0.99)
            filtered_totals = self.train_orders[self.train_orders['grand_total'] <= q99]['grand_total']
            axes[1, 1].hist(filtered_totals, bins=50, edgecolor='black')
            axes[1, 1].set_title('Order Value Distribution (99th percentile)')
            axes[1, 1].set_xlabel('Order Total ($)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Vendor Tags Word Cloud (Top 10)
        if self.vendors is not None and 'vendor_tag_name' in self.vendors.columns:
            top_tags = self.vendors['vendor_tag_name'].value_counts().head(10)
            axes[1, 2].barh(range(len(top_tags)), top_tags.values)
            axes[1, 2].set_yticks(range(len(top_tags)))
            axes[1, 2].set_yticklabels(top_tags.index)
            axes[1, 2].set_title('Top 10 Vendor Tags')
            axes[1, 2].set_xlabel('Number of Vendors')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, file_paths=None):
        """Run the complete data exploration pipeline"""
        print("🔍 STARTING COMPREHENSIVE DATA EXPLORATION")
        print("=" * 60)
        
        # Always try to load data first
        self.load_data(file_paths)
        
        # Check if data was loaded successfully
        if not self._data_loaded():
            print("❌ Cannot proceed with analysis - data loading failed!")
            return
        
        # Run all analyses
        self.explore_dataset_structure()
        self.analyze_customer_data()
        self.analyze_location_data()
        self.analyze_order_patterns()
        self.analyze_vendor_data()
        
        print(f"\n{'='*60}")
        print("CREATING SUMMARY VISUALIZATIONS")
        print(f"{'='*60}")
        self.create_summary_visualizations()
        
        print(f"\n{'='*60}")
        print("✅ DATA EXPLORATION COMPLETE!")
        print(f"{'='*60}")
        
        # Summary insights
        self.print_key_insights()
    
    def _data_loaded(self):
        """Check if essential data has been loaded"""
        essential_data = [
            self.train_customers, 
            self.train_locations, 
            self.train_orders, 
            self.test_customers, 
            self.test_locations, 
            self.vendors
        ]
        return all(df is not None for df in essential_data)
    
    def print_key_insights(self):
        """Print key insights discovered during exploration"""
        print(f"\n🔑 KEY INSIGHTS DISCOVERED:")
        print("-" * 30)
        
        insights = []
        
        if self.train_customers is not None and self.test_customers is not None:
            insights.append(f"• Dataset has {len(self.train_customers):,} train and {len(self.test_customers):,} test customers")
        
        if self.train_orders is not None:
            avg_orders = self.train_orders.groupby('customer_id').size().mean()
            insights.append(f"• Average {avg_orders:.1f} orders per customer")
            insights.append(f"• {self.train_orders['vendor_id'].nunique()} unique vendors in training data")
        
        if self.train_locations is not None:
            avg_locations = self.train_locations.groupby('customer_id')['location_number'].count().mean()
            insights.append(f"• Average {avg_locations:.1f} locations per customer")
        
        if self.vendors is not None:
            insights.append(f"• {len(self.vendors)} total vendors available for recommendations")
        
        for insight in insights:
            print(insight)
        
        print(f"\n📋 NEXT STEPS:")
        print("• Proceed to Step 2: Data Preprocessing and Feature Engineering")
        print("• Focus on handling missing values and creating meaningful features")
        print("• Consider geographic clustering for location-based features")
        print("• Analyze customer-vendor interaction patterns for collaborative filtering")

# Usage Example:
# ===============
# Initialize the explorer
explorer = RestaurantDataExplorer()

# Now you can simply run the analysis - it will use your file paths automatically!
explorer.run_complete_analysis()

# Alternative: If you want to specify different paths
# file_paths = {
#     'train_customers': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Train\train_customers.csv',
#     'train_locations': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Train\train_locations.csv', 
#     'train_orders': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Train\orders.csv',
#     'test_customers': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Test\test_customers.csv',
#     'test_locations': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Test\test_locations.csv',
#     'vendors': r'C:\JAGA\SoulpageIT PROJECT1\Assignment\Train\vendors.csv'
# }
# explorer.run_complete_analysis(file_paths)

print("📊 Data Explorer Ready!")
print("Run: explorer.run_complete_analysis() to start the analysis!")