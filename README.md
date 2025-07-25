Employee Salary Prediction System


Project Description

This project is a comprehensive machine learning system designed to predict employee salaries. It is implemented as an interactive web application using Streamlit. The system leverages a multi-algorithm approach to train, evaluate, and select the best regression model for providing accurate salary estimates.
A key feature of this application is its dual-currency functionality, offering salary predictions in both US Dollars (USD) and Indian Rupees (INR). It also incorporates location intelligence, with specific data and insights for major metropolitan areas in both the United States and India, making it a globally relevant tool for HR professionals, recruiters, and job seekers.

Key Features                
•	Multi-Algorithm Training: Trains and compares five different regression models: Linear Regression, Random Forest, Gradient Boosting, Support Vector Regression (SVR), and a Multi-layer Perceptron (MLP) Neural Network.       
•	Dual Currency Prediction: Provides salary estimates in both USD and INR, with transparent exchange rate conversion.                        
•	Location Intelligence: Includes major US and Indian cities as location features, with data generation reflecting regional salary variations.                  
•	Interactive Data Exploration: Allows users to upload their own datasets or generate synthetic data, with comprehensive exploratory data analysis (EDA) tools and interactive visualizations.                
•	Comprehensive Performance Analytics: Features detailed dashboards for comparing model performance using R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).               
•	Dynamic Prediction Interface: An intuitive form for users to input employee profiles and receive instant salary predictions with confidence intervals.                 
•	Model Persistence: Functionality to save the best-performing trained model and its associated preprocessors (scaler, encoders) for future use.                        
•	Automated Reporting: Generates and exports detailed project summary reports.

System Architecture and Workflow

The application follows a standard machine learning pipeline: 
1.	Data Ingestion: User uploads a CSV file or generates a synthetic dataset.
2.	Data Preprocessing: The system handles missing values, removes duplicates, detects and removes outliers, and encodes categorical features.
3.	Feature Scaling: Numerical features are standardized using StandardScaler.
4.	Model Training: The selected algorithms are trained on the preprocessed data. Hyperparameter tuning is performed using GridSearchCV.
5.	Model Evaluation: Models are evaluated based on performance metrics, and the best model is selected for predictions.
6.	Prediction: The user provides new employee data, which is transformed using the saved preprocessors and fed into the best model to generate a salary prediction.
7.	Visualization: Results, analytics, and predictions are displayed through interactive charts and tables.
   
Installation and Setup

Prerequisites                   
•	Python 3.8 or higher     
•	pip (Python package installer)                     
•	Git (for cloning the repository)

Setup Instructions                      

1.	Clone the repository:                                   
git clone <your-repository-url>                                    
cd employee-salary-streamlit

2.	Create a virtual environment (recommended):                                   
python -m venv venv                              
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`                                                

3.	Install the required dependencies:                                
pip install -r requirements.txt

4.	Run the Streamlit application:                                  
streamlit run app.py

The application will be accessible in your web browser at http://localhost:8501.

Usage Guide                            
1.	Data Explorer: Navigate to this page to either upload your own salary dataset in CSV format or generate a sample dataset. Use the interactive plots to explore the data's characteristics.                
2.	Model Training: Select the machine learning algorithms you wish to train. Click "Start Training" to begin the process. The system will display the performance of each model and select the best one.           
3.	Salary Predictor: Once a model is trained, go to this page. Fill in the employee profile details in the form and click "Predict Salary" to get an estimate in both USD and INR.            
4.	Results & Analytics: Review detailed performance metrics, feature importance charts, and other analytical visualizations for the trained models.                  
5.	Reports: Generate and download a comprehensive report of the project execution or save the trained model for later use.
   
Technologies Used                  
•	Backend & ML: Python, Scikit-learn, Pandas, NumPy                       
•	Web Framework: Streamlit                                      
•	Data Visualization: Plotly, Matplotlib, Seaborn                               
•	Model Persistence: Joblib                             

File Structure                                              
employee_salary_streamlit/                    
│                                         
├── app.py                             # The main Streamlit application file                                                  
├── predictor.py        # Core class for data processing, model training, and prediction                                   
├── requirements.txt    # List of Python dependencies for the project                                        
└── README.md           # Project documentation file   
