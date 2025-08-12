import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class SimpleRestaurantRecommender:
    def __init__(self):
        self.model = None
        self.vendor_popularity = None
        self.customer_profiles = None
        self.vendor_pop_dict = None
        self.vendor_ids_list = None
        self.vendor_features_array = None

    def load_data(self, data_dir="processed_data"):
        """Load preprocessed data from CSV files"""
        try:
            print("📂 Loading data...")
            self.train_customers = pd.read_csv(f"{data_dir}/train_customer_features.csv")
            self.test_customers = pd.read_csv(f"{data_dir}/test_customer_features.csv")
            self.vendors = pd.read_csv(f"{data_dir}/vendor_features.csv")
            self.interactions = pd.read_csv(f"{data_dir}/interaction_matrix.csv")
            print("✅ Data loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def create_simple_features(self):
        """Create vendor popularity and customer profile features"""
        print("🔧 Creating simple features...")

        # 1. Vendor popularity table
        self.vendor_popularity = self.interactions.groupby('vendor_id').agg({
            'order_count': 'sum',
            'interaction_score': 'mean',
            'total_spent': 'sum'
        }).reset_index()

        self.vendor_popularity['popularity_rank'] = self.vendor_popularity['order_count'].rank(pct=True)
        self.vendor_popularity['quality_score'] = self.vendor_popularity['interaction_score'].rank(pct=True)

        # 2. Customer profile table
        self.customer_profiles = self.interactions.groupby('customer_id').agg({
            'order_count': 'sum',
            'total_spent': 'mean',
            'interaction_score': 'mean'
        }).reset_index()

        # 3. Precompute vendor dict, ID list, and feature matrix for fast predictions
        self.vendor_pop_dict = self.vendor_popularity.set_index('vendor_id').to_dict(orient='index')
        self.vendor_ids_list = list(self.vendor_pop_dict.keys())
        self.vendor_features_array = np.array([
            [info['popularity_rank'], 1.0, 10.0] for info in self.vendor_pop_dict.values()
        ])

        print("✅ Features created!")

    def build_model(self):
        """Train RandomForest model"""
        print("🤖 Training model...")

        training_data = []
        for _, row in self.interactions.iterrows():
            if row['vendor_id'] in self.vendor_pop_dict:
                info = self.vendor_pop_dict[row['vendor_id']]
                features = [info['popularity_rank'], row['order_count'], row['total_spent']]
                training_data.append(features + [row['interaction_score']])

        training_data = np.array(training_data)
        if len(training_data) == 0:
            print("❌ No training data available!")
            return False

        X, y = training_data[:, :-1], training_data[:, -1]
        self.model = RandomForestRegressor(
            n_estimators=30,      # fewer trees for speed
            max_depth=15,         # limit depth
            max_features='sqrt',  # faster at split
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(X, y)

        print("✅ Model trained!")
        return True

    def predict_for_customer(self, customer_id, location_num):
        """Predict the best vendor for a customer-location pair"""
        if self.model:
            scores = self.model.predict(self.vendor_features_array)
        else:
            scores = [info['popularity_rank'] for info in self.vendor_pop_dict.values()]
        best_idx = int(np.argmax(scores))
        return self.vendor_ids_list[best_idx]

    def generate_predictions(self, test_locations):
        """Generate predictions for all customer-location pairs"""
        predictions = []
        for _, row in test_locations.iterrows():
            cust_id = row['customer_id']
            loc_num = row['location_number']
            best_vendor = self.predict_for_customer(cust_id, loc_num)
            predictions.append(f"{cust_id} X {loc_num} X {best_vendor}")
        return predictions

    def save_predictions(self, predictions, filename="restaurant_recommendations_final.txt"):
        """Save predictions in CID X LOC_NUM X VENDOR target format"""
        with open(filename, 'w') as f:
            f.write("CID X LOC_NUM X VENDOR target\n")
            for pred in predictions:
                f.write(f"{pred} 0\n")
        print(f"💾 Saved predictions to {filename} ({len(predictions)} rows)")


if __name__ == "__main__":
    start_time = time.time()
    recommender = SimpleRestaurantRecommender()

    # Load data
    if recommender.load_data():
        recommender.create_simple_features()
        if recommender.build_model():
            # Load or create test locations file
            try:
                test_locations = pd.read_csv(r"C:\JAGA\SoulpageIT PROJECT1\Assignment\Test\test_locations.csv")
            except:
                print("⚠️ test_locations.csv not found, generating dummy test locations...")
                locs = []
                for cid in recommender.test_customers['customer_id']:
                    for l in range(7):  # 0 to 6
                        locs.append({'customer_id': cid, 'location_number': l})
                test_locations = pd.DataFrame(locs)

            # Generate predictions
            preds = recommender.generate_predictions(test_locations)
            recommender.save_predictions(preds)

    print(f"⏱ Total runtime: {time.time() - start_time:.2f} seconds")
