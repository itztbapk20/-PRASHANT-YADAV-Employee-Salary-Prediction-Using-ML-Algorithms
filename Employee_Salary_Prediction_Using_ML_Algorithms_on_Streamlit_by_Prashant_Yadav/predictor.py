import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')


class EmployeeSalaryPredictor:
    """
    Employee Salary Prediction System for Streamlit
    """

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.original_df = None

    def create_sample_data(self, n_samples=2000, save_to_file=True):
        """Create comprehensive sample dataset with realistic salary patterns"""
        np.random.seed(42)

        # Generate realistic data
        ages = np.random.normal(35, 8, n_samples).astype(int)
        ages = np.clip(ages, 22, 65)

        experience = np.maximum(0, ages - np.random.randint(22, 28, n_samples))
        experience = np.clip(experience, 0, 40)

        education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
        education_weights = [0.15, 0.15, 0.4, 0.25, 0.05]
        education = np.random.choice(education_levels, n_samples, p=education_weights)

        job_titles = ['Software Engineer', 'Data Scientist', 'Manager', 'Senior Manager',
                      'Analyst', 'Senior Analyst', 'Designer', 'Developer', 'Consultant',
                      'Director', 'Vice President']

        # Fixed job weights that sum to 1.0
        job_weights = [0.2, 0.15, 0.15, 0.08, 0.12, 0.08, 0.1, 0.15, 0.05, 0.03, 0.01]
        job_weights = np.array(job_weights) / np.sum(job_weights)  # Normalize to sum to 1.0
        job_titles_assigned = np.random.choice(job_titles, n_samples, p=job_weights)

        industries = ['Technology', 'Finance', 'Healthcare', 'Manufacturing',
                      'Retail', 'Education', 'Government', 'Consulting']
        industry = np.random.choice(industries, n_samples)

        cities = ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston',
                  'Seattle', 'Austin', 'Denver', 'Atlanta', 'Dallas',
                  'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
                  'Pune', 'Kolkata', 'Ahmedabad', 'Noida', 'Gurgaon']
        city = np.random.choice(cities, n_samples)

        gender = np.random.choice(['Male', 'Female'], n_samples)

        # Calculate realistic salaries based on multiple factors
        base_salary = 40000

        # Education multipliers
        edu_multiplier = {'High School': 1.0, 'Associate': 1.1, 'Bachelor': 1.3,
                          'Master': 1.5, 'PhD': 1.7}

        # Job title multipliers
        job_multiplier = {
            'Software Engineer': 1.4, 'Data Scientist': 1.6, 'Manager': 1.5,
            'Senior Manager': 1.8, 'Analyst': 1.1, 'Senior Analyst': 1.4,
            'Designer': 1.2, 'Developer': 1.3, 'Consultant': 1.4,
            'Director': 2.0, 'Vice President': 2.5
        }

        # Industry multipliers
        industry_multiplier = {
            'Technology': 1.3, 'Finance': 1.4, 'Healthcare': 1.1,
            'Manufacturing': 1.0, 'Retail': 0.9, 'Education': 0.8,
            'Government': 1.0, 'Consulting': 1.2
        }

        # City multipliers (cost of living)
        city_multiplier = {
            'New York': 1.3, 'San Francisco': 1.4, 'Los Angeles': 1.2,
            'Chicago': 1.1, 'Boston': 1.2, 'Seattle': 1.25, 'Austin': 1.0,
            'Denver': 1.0, 'Atlanta': 0.95, 'Dallas': 0.95,
            # Indian cities (lower multipliers due to lower cost of living)
            'Mumbai': 0.4, 'Delhi': 0.35, 'Bangalore': 0.4, 'Hyderabad': 0.3,
            'Chennai': 0.3, 'Pune': 0.35, 'Kolkata': 0.25, 'Ahmedabad': 0.25,
            'Noida': 0.3, 'Gurgaon': 0.35
        }

        # Calculate salaries
        salaries = []
        for i in range(n_samples):
            salary = (base_salary +
                      experience[i] * 2500 +  # Experience impact
                      ages[i] * 800 +  # Age impact
                      edu_multiplier[education[i]] * 25000 +
                      job_multiplier[job_titles_assigned[i]] * 30000 +
                      industry_multiplier[industry[i]] * 15000 +
                      city_multiplier[city[i]] * 20000 +
                      np.random.normal(0, 8000))  # Random variation

            # Gender pay gap simulation (unfortunately realistic)
            if gender[i] == 'Female':
                salary *= 0.95

            salaries.append(max(30000, int(salary)))  # Minimum salary floor

        # Create DataFrame
        data = {
            'Age': ages,
            'Years_of_Experience': experience,
            'Education_Level': education,
            'Job_Title': job_titles_assigned,
            'Industry': industry,
            'City': city,
            'Gender': gender,
            'Salary': salaries
        }

        df = pd.DataFrame(data)

        if save_to_file:
            df.to_csv('Salary_Data.csv', index=False)

        return df

    def load_data(self, file_path='Salary_Data.csv'):
        """Load dataset from file"""
        self.original_df = pd.read_csv(file_path)
        return self.original_df

    def preprocess_data(self, df=None):
        """Comprehensive data preprocessing"""
        if df is None:
            df = self.original_df.copy()

        # Handle missing values
        df = df.dropna()

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle outliers using IQR method
        Q1 = df['Salary'].quantile(0.25)
        Q3 = df['Salary'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]

        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'Salary']

        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.encoders[col] = le

        return df

    def prepare_features(self, df):
        """Prepare feature matrix and target vector"""
        feature_columns = []

        # Add numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_columns.extend([col for col in numeric_cols if col != 'Salary'])

        # Add encoded categorical columns
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        feature_columns.extend(encoded_cols)

        self.feature_columns = feature_columns
        X = df[feature_columns]
        y = df['Salary']

        return X, y

    def train_models(self, X, y, use_cross_validation=True):
        """Train multiple models with hyperparameter tuning"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1, 0.01]
                }
            }
        }

        results = {}

        for name, config in models_config.items():
            # Hyperparameter tuning
            if config['params']:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = config['model']
                best_model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = best_model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation score
            cv_scores = None
            if use_cross_validation:
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')

            results[name] = {
                'model': best_model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_scores': cv_scores,
                'y_test': y_test,
                'y_pred': y_pred
            }

        # Find best model
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[self.best_model_name]['model']
        self.results = results

        return results

    def predict_salary(self, employee_data):
        """Predict salary for new employee with currency conversion"""

        if self.best_model is None:
            return {"error": "No trained model available. Please train a model first."}

        if not self.feature_columns:
            return {"error": "No feature columns defined. Please train a model first."}

        try:
            # Create a feature vector that matches the training feature order exactly
            feature_vector = []

            # Process each feature column in the exact same order as training
            for feature_name in self.feature_columns:

                if feature_name == 'Age':
                    feature_vector.append(float(employee_data.get('Age', 30)))

                elif feature_name == 'Years_of_Experience':
                    feature_vector.append(float(employee_data.get('Years_of_Experience', 0)))

                elif feature_name.endswith('_encoded'):
                    # This is an encoded categorical feature
                    original_col = feature_name.replace('_encoded', '')

                    if original_col in self.encoders and original_col in employee_data:
                        try:
                            encoded_value = self.encoders[original_col].transform([employee_data[original_col]])[0]
                            feature_vector.append(float(encoded_value))
                        except ValueError:
                            # Handle unknown categories
                            encoded_value = \
                            self.encoders[original_col].transform([self.encoders[original_col].classes_[0]])[0]
                            feature_vector.append(float(encoded_value))
                    else:
                        # Missing encoder or data
                        feature_vector.append(0.0)

                else:
                    # Any other numeric feature
                    feature_vector.append(float(employee_data.get(feature_name, 0)))

            # Verify length matches
            if len(feature_vector) != len(self.feature_columns):
                return {
                    "error": f"Feature vector length mismatch. Expected {len(self.feature_columns)} features, got {len(feature_vector)}."}

            # Convert to numpy array and scale
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Make prediction (in USD)
            predicted_salary_usd = self.best_model.predict(feature_vector_scaled)[0]

            # Currency conversion rate (USD to INR)
            usd_to_inr_rate = 83.0  # Approximate rate - you can update this

            # Convert to INR
            predicted_salary_inr = predicted_salary_usd * usd_to_inr_rate

            # Calculate confidence intervals for both currencies
            if self.results and self.best_model_name in self.results:
                rmse_usd = self.results[self.best_model_name]['rmse']
                confidence_lower_usd = predicted_salary_usd - 1.96 * rmse_usd
                confidence_upper_usd = predicted_salary_usd + 1.96 * rmse_usd

                # Convert confidence intervals to INR
                confidence_lower_inr = confidence_lower_usd * usd_to_inr_rate
                confidence_upper_inr = confidence_upper_usd * usd_to_inr_rate

                model_r2 = self.results[self.best_model_name]['r2']
            else:
                confidence_lower_usd = predicted_salary_usd * 0.8
                confidence_upper_usd = predicted_salary_usd * 1.2
                confidence_lower_inr = confidence_lower_usd * usd_to_inr_rate
                confidence_upper_inr = confidence_upper_usd * usd_to_inr_rate
                model_r2 = 0.0

            return {
                'predicted_salary_usd': float(predicted_salary_usd),
                'predicted_salary_inr': float(predicted_salary_inr),
                'confidence_lower_usd': float(confidence_lower_usd),
                'confidence_upper_usd': float(confidence_upper_usd),
                'confidence_lower_inr': float(confidence_lower_inr),
                'confidence_upper_inr': float(confidence_upper_inr),
                'exchange_rate': usd_to_inr_rate,
                'model_used': self.best_model_name,
                'model_r2': float(model_r2)
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction failed: {str(e)}"}

    def save_model(self, filepath='salary_prediction_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'results': self.results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        joblib.dump(model_data, filepath)

    def generate_report(self, save_to_file=True):
        """Generate comprehensive project report"""
        if not self.results:
            return "No results available. Train models first."

        report = []
        report.append("=" * 80)
        report.append("EMPLOYEE SALARY PREDICTION PROJECT - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset Summary
        if self.original_df is not None:
            report.append("1. DATASET SUMMARY")
            report.append("-" * 40)
            report.append(f"   Total Records: {len(self.original_df):,}")
            report.append(f"   Features: {len(self.original_df.columns) - 1}")
            report.append(f"   Target Variable: Salary")
            report.append(
                f"   Salary Range: ${self.original_df['Salary'].min():,} - ${self.original_df['Salary'].max():,}")
            report.append(f"   Average Salary: ${self.original_df['Salary'].mean():,.0f}")
            report.append("")

        # Model Performance
        report.append("2. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Model':<25} {'R²':<10} {'RMSE':<15} {'MAE':<15}")
        report.append("-" * 65)

        for name, result in self.results.items():
            indicator = "🏆 " if name == self.best_model_name else "   "
            report.append(
                f"{indicator}{name:<23} {result['r2']:<10.4f} ${result['rmse']:<14,.0f} ${result['mae']:<14,.0f}")

        report.append("")
        report.append(f"Best Model: {self.best_model_name}")
        report.append(f"Best R² Score: {self.results[self.best_model_name]['r2']:.4f}")

        return "\n".join(report)
