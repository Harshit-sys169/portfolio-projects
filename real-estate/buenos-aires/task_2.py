import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


class BuenosAiresAnalyzer:
    """
    Geographic-based price prediction for Buenos Aires real estate.
    This analysis explores how location (latitude/longitude) affects apartment prices.
    """
    
    def __init__(self):
        self.model = None
        self.data = None
    
    def wrangle(self, filepath):
        """
        Load and prepare real estate data.
        Applies filters specific to Capital Federal apartments.
        """
        df = pd.read_csv(filepath)
        
        mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
        mask_apt = df["property_type"] == "apartment"
        mask_price = df["price_aprox_usd"] < 400_000
        df = df[mask_ba & mask_apt & mask_price]
        
        low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
        df = df[df["surface_covered_in_m2"].between(low, high)]
        
        df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True)
        df["lat"] = pd.to_numeric(df["lat"].str.strip(), errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"].str.strip(), errors="coerce")
        
        df = df.drop(columns="lat-lon")
        
        return df
    
    def load_data(self, filepath1, filepath2):
        """
        Load and concatenate multiple data sources.
        """
        frame1 = self.wrangle(filepath1)
        frame2 = self.wrangle(filepath2)
        self.data = pd.concat([frame1, frame2], ignore_index=True)
        return self.data
    
    def build_model(self, X_train, y_train):
        """
        Create pipeline with imputation and linear regression.
        Imputation handles any remaining missing geographic data.
        """
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", LinearRegression())
        ])
        
        self.model.fit(X_train, y_train)
        check_is_fitted(self.model["regressor"])
        
        return self.model
    
    def evaluate_model(self, X_train, y_train, X_test):
        """
        Evaluate model performance and generate predictions.
        """
        y_pred_train = self.model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred_train)
        r2 = r2_score(y_train, y_pred_train)
        
        print(f"Model Performance:")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  R2 Score: {r2:.4f}")
        
        intercept = self.model["regressor"].intercept_
        coefs = self.model["regressor"].coef_
        print(f"\nModel Equation:")
        print(f"  price = {intercept:,.2f} + ({coefs[0]:,.2f} * lon) + ({coefs[1]:,.2f} * lat)")
        
        y_pred_test = self.model.predict(X_test)
        return pd.Series(y_pred_test)
    
    def plot_geographic_distribution(self):
        """
        Create interactive map showing apartment distribution and prices.
        """
        fig = px.scatter_mapbox(
            self.data,
            lat="lat",
            lon="lon",
            color="price_aprox_usd",
            hover_data=["price_aprox_usd", "surface_covered_in_m2"],
            title="Buenos Aires Apartments: Price Distribution",
            color_continuous_scale="Viridis",
            zoom=11,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map")
        return fig
    
    def plot_3d_surface(self):
        """
        Create 3D scatter plot with regression surface.
        Visualizes how latitude and longitude jointly predict price.
        """
        fig = px.scatter_3d(
            self.data,
            x="lon",
            y="lat",
            z="price_aprox_usd",
            labels={"lon": "Longitude", "lat": "Latitude", "price_aprox_usd": "Price (USD)"},
            title="3D Price Surface: Geographic Location",
            height=700
        )
        
        x_plane = np.linspace(self.data["lon"].min(), self.data["lon"].max(), 10)
        y_plane = np.linspace(self.data["lat"].min(), self.data["lat"].max(), 10)
        xx, yy = np.meshgrid(x_plane, y_plane)
        
        z_plane = self.model.predict(pd.DataFrame({"lon": x_plane, "lat": y_plane}))
        zz = np.tile(z_plane, (10, 1))
        
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.7, name="Regression Surface"))
        
        return fig


if __name__ == "__main__":
    print("Buenos Aires Real Estate Analysis - Task 2\n")
    print("="*60)
    
    analyzer = BuenosAiresAnalyzer()
    
    try:
        analyzer.load_data(
            "data/buenos-aires-real-estate-1.csv",
            "data/buenos-aires-real-estate-2.csv"
        )
        
        print(f"Total apartments loaded: {len(analyzer.data)}")
        print(f"Price range: ${analyzer.data['price_aprox_usd'].min():,.0f} - ${analyzer.data['price_aprox_usd'].max():,.0f}")
        print(f"Geographic bounds (lon): {analyzer.data['lon'].min():.4f} to {analyzer.data['lon'].max():.4f}")
        print(f"Geographic bounds (lat): {analyzer.data['lat'].min():.4f} to {analyzer.data['lat'].max():.4f}\n")
        
        X_train = analyzer.data[["lon", "lat"]]
        y_train = analyzer.data["price_aprox_usd"]
        
        analyzer.build_model(X_train, y_train)
        
        X_test = pd.read_csv("data/buenos-aires-test-features.csv")[["lon", "lat"]]
        y_pred = analyzer.evaluate_model(X_train, y_train, X_test)
        
        print(f"\nTest set predictions (first 5):")
        print(y_pred.head())
        
    except Exception as e:
        print(f"Error: {e}")