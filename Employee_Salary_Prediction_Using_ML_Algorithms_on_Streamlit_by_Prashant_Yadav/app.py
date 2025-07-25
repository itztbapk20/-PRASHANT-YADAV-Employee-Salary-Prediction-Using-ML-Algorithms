import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

from predictor import EmployeeSalaryPredictor

# Configure page
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        color: white;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = EmployeeSalaryPredictor()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def main():
    # Main title
    st.markdown('<h1 class="main-header">💰 Employee Salary Prediction System</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "🔮 Salary Predictor", "📈 Results & Analytics", "📑 Reports"]
    )

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔮 Salary Predictor":
        show_salary_predictor()
    elif page == "📈 Results & Analytics":
        show_results_analytics()
    elif page == "📑 Reports":
        show_reports()


def show_home_page():
    """Home page with project overview"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This application predicts employee salaries using machine learning algorithms based on various factors like education, experience, job title, and location.</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown('<h2 class="sub-header">✨ Key Features</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Data Analysis</h4>
        <p>Comprehensive exploratory data analysis with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🤖 ML Models</h4>
        <p>Multiple algorithms comparison with hyperparameter tuning</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🔮 Predictions</h4>
        <p>Real-time salary predictions for new employees</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Analytics</h4>
        <p>Detailed performance metrics and feature importance</p>
        </div>
        """, unsafe_allow_html=True)

    # Getting started section
    st.markdown('<h2 class="sub-header">🚀 Getting Started</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Step 1: Load Data** 📊
        - Upload your own dataset or use sample data
        - Explore data characteristics and distributions

        **Step 2: Train Models** 🤖
        - Select algorithms to train
        - Optimize hyperparameters
        - Compare model performance
        """)

    with col2:
        st.markdown("""
        **Step 3: Make Predictions** 🔮
        - Input employee information
        - Get instant salary predictions
        - View confidence intervals

        **Step 4: Analyze Results** 📈
        - Review model performance
        - Understand feature importance
        - Generate comprehensive reports
        """)

    # Sample data section
    if st.button("🎲 Generate Sample Data", type="primary"):
        with st.spinner("Creating sample dataset..."):
            df = st.session_state.predictor.create_sample_data(n_samples=1000, save_to_file=False)
            st.session_state.predictor.original_df = df
            st.session_state.data_loaded = True
            st.success("✅ Sample dataset created successfully!")
            st.balloons()


def show_data_explorer():
    """Data exploration page"""
    st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)

    # Data loading section
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Choose data source:",
            ["📁 Upload CSV File", "🎲 Generate Sample Data", "💾 Load Existing Data"]
        )

    with col2:
        if data_source == "🎲 Generate Sample Data":
            n_samples = st.slider("Number of samples:", 500, 5000, 2000, 500)

    # Load data based on selection
    if data_source == "📁 Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.predictor.original_df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    elif data_source == "🎲 Generate Sample Data":
        if st.button("Generate Data", type="primary"):
            with st.spinner("Creating sample dataset..."):
                df = st.session_state.predictor.create_sample_data(n_samples=n_samples, save_to_file=False)
                st.session_state.predictor.original_df = df
                st.session_state.data_loaded = True
                st.success("✅ Sample dataset created!")

    elif data_source == "💾 Load Existing Data":
        if st.button("Load Data", type="primary"):
            try:
                df = st.session_state.predictor.load_data()
                st.session_state.data_loaded = True
                st.success("✅ Data loaded from file!")
            except:
                st.error("❌ No existing data file found. Please generate or upload data.")

    # Display data if loaded
    if st.session_state.data_loaded and st.session_state.predictor.original_df is not None:
        df = st.session_state.predictor.original_df

        # Dataset overview
        st.markdown('<h3 class="sub-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
        with col4:
            st.metric("Salary Range", f"${df['Salary'].max() - df['Salary'].min():,}")

        # Data preview
        st.markdown('<h3 class="sub-header">👁️ Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)


def show_model_training():
    """Model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Data Explorer page.")
        return

    # Training configuration
    st.markdown('<h3>⚙️ Training Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        algorithms = st.multiselect(
            "Select algorithms to train:",
            ["Linear Regression", "Random Forest", "Gradient Boosting",
             "Support Vector Regression", "Neural Network"],
            default=["Linear Regression", "Random Forest", "Gradient Boosting"]
        )

    with col2:
        use_cv = st.checkbox("Use Cross-Validation", value=True)
        test_size = st.slider("Test Size (%)", 10, 40, 20)

    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not algorithms:
            st.error("❌ Please select at least one algorithm to train.")
            return

        with st.spinner("🔄 Training models... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Preprocess data
            status_text.text("📊 Preprocessing data...")
            progress_bar.progress(20)
            processed_df = st.session_state.predictor.preprocess_data()

            # Prepare features
            status_text.text("🔧 Preparing features...")
            progress_bar.progress(40)
            X, y = st.session_state.predictor.prepare_features(processed_df)

            # Train models
            status_text.text("🤖 Training models...")
            progress_bar.progress(60)
            results = st.session_state.predictor.train_models(X, y, use_cross_validation=use_cv)

            # Complete
            status_text.text("✅ Training completed!")
            progress_bar.progress(100)
            st.session_state.model_trained = True

            st.success("🎉 Model training completed successfully!")
            st.balloons()

        # Display results
        display_training_results(results)


def display_training_results(results):
    """Display training results"""
    st.markdown('<h3>📊 Training Results</h3>', unsafe_allow_html=True)

    # Create results DataFrame
    results_df = []
    for name, result in results.items():
        results_df.append({
            'Model': name,
            'R² Score': f"{result['r2']:.4f}",
            'RMSE': f"${result['rmse']:,.0f}",
            'MAE': f"${result['mae']:,.0f}"
        })

    results_df = pd.DataFrame(results_df)

    # Highlight best model
    best_model = max(results.keys(), key=lambda k: results[k]['r2'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4>🏆 Model Comparison</h4>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)

    with col2:
        st.markdown('<h4>🥇 Best Model</h4>', unsafe_allow_html=True)
        st.success(f"**{best_model}**")
        st.metric("R² Score", f"{results[best_model]['r2']:.4f}")
        st.metric("RMSE", f"${results[best_model]['rmse']:,.0f}")


def show_salary_predictor():
    """Salary prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Salary Predictor</h2>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first from the Model Training page.")
        st.info("Go to **Model Training** → Load data → Train models")
        return

    st.markdown("### Enter employee information to predict salary:")

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Basic Information**")
            age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)

        with col2:
            st.markdown("**🎓 Education & Demographics**")
            education = st.selectbox("Education Level",
                                     ["High School", "Associate", "Bachelor", "Master", "PhD"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col3:
            st.markdown("**💼 Professional Details**")
            job_title = st.selectbox("Job Title",
                                     ["Software Engineer", "Data Scientist", "Manager", "Senior Manager",
                                      "Analyst", "Senior Analyst", "Designer", "Developer", "Consultant",
                                      "Director", "Vice President"])
            industry = st.selectbox("Industry",
                                    ["Technology", "Finance", "Healthcare", "Manufacturing",
                                     "Retail", "Education", "Government", "Consulting"])

        st.markdown("**🏙️ Location**")
        city = st.selectbox("City",
                            ["New York", "San Francisco", "Los Angeles", "Chicago", "Boston",
                             "Seattle", "Austin", "Denver", "Atlanta", "Dallas",
                             "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
                             "Pune", "Kolkata", "Ahmedabad", "Noida", "Gurgaon"])

        predict_button = st.form_submit_button("🔮 Predict Salary", type="primary")

    if predict_button:
        # Create employee data dictionary
        employee_data = {
            'Age': age,
            'Years_of_Experience': experience,
            'Education_Level': education,
            'Job_Title': job_title,
            'Industry': industry,
            'City': city,
            'Gender': gender
        }

        # Make prediction
        with st.spinner("🔄 Making prediction..."):
            try:
                prediction_result = st.session_state.predictor.predict_salary(employee_data)

                # Check for errors first
                if prediction_result is None:
                    st.error("❌ Prediction failed: No result returned")
                    st.info("💡 **Troubleshooting Steps:**")
                    st.write("1. Go to **Data Explorer** and generate sample data")
                    st.write("2. Go to **Model Training** and train at least one model")
                    st.write("3. Return here and try prediction again")

                elif 'error' in prediction_result:
                    st.error(f"❌ {prediction_result['error']}")
                    st.info("💡 **Troubleshooting Steps:**")
                    st.write("1. Go to **Data Explorer** and generate sample data")
                    st.write("2. Go to **Model Training** and train at least one model")
                    st.write("3. Return here and try prediction again")

                elif prediction_result.get('predicted_salary_usd') is not None:
                    # Display prediction results - SUCCESS CASE with dual currency
                    st.markdown("---")

                    # Main prediction display - Dual Currency
                    col1, col2 = st.columns(2)

                    # USD Display
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                            color: white;
                            margin: 10px;
                        ">
                            <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                                💵 USD Prediction
                            </h3>
                            <div style="
                                font-size: 2.5em; 
                                font-weight: bold; 
                                margin: 15px 0;
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                            ">
                                ${prediction_result['predicted_salary_usd']:,.0f}
                            </div>
                            <p style="font-size: 1em; opacity: 0.9; margin: 0;">
                                US Dollars (Annual)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # INR Display
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                            color: #333;
                            margin: 10px;
                        ">
                            <h3 style="margin: 0; font-size: 1.5em; font-weight: bold;">
                                ₹ INR Prediction
                            </h3>
                            <div style="
                                font-size: 2.5em; 
                                font-weight: bold; 
                                margin: 15px 0;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
                            ">
                                ₹{prediction_result['predicted_salary_inr']:,.0f}
                            </div>
                            <p style="font-size: 1em; opacity: 0.8; margin: 0;">
                                Indian Rupees (Annual)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Exchange rate info
                    st.info(f"💱 **Exchange Rate Used:** 1 USD = ₹{prediction_result['exchange_rate']}")

                    # Additional details
                    st.markdown("### 📊 Prediction Details")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("🎯 Model Used", prediction_result['model_used'])

                    with col2:
                        st.metric("📈 Model Accuracy (R²)", f"{prediction_result['model_r2']:.4f}")

                    with col3:
                        st.metric("💱 USD to INR Rate", f"₹{prediction_result['exchange_rate']}")

                    # Dual currency confidence intervals
                    st.markdown("### 🎯 95% Confidence Intervals")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(f"""
                        **💵 USD Range:**
                        ${prediction_result['confidence_lower_usd']:,.0f} - ${prediction_result['confidence_upper_usd']:,.0f}
                        """)

                    with col2:
                        st.success(f"""
                        **₹ INR Range:**
                        ₹{prediction_result['confidence_lower_inr']:,.0f} - ₹{prediction_result['confidence_upper_inr']:,.0f}
                        """)

                    # Dual currency visualization
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('USD Salary Range', 'INR Salary Range'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )

                    # USD range
                    fig.add_trace(
                        go.Scatter(
                            x=[prediction_result['confidence_lower_usd'],
                               prediction_result['predicted_salary_usd'],
                               prediction_result['confidence_upper_usd']],
                            y=[1, 1, 1],
                            mode='markers+lines',
                            marker=dict(size=[12, 20, 12], color=['lightblue', 'red', 'lightblue']),
                            line=dict(color='blue', width=3),
                            name='USD Range',
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                    # INR range
                    fig.add_trace(
                        go.Scatter(
                            x=[prediction_result['confidence_lower_inr'],
                               prediction_result['predicted_salary_inr'],
                               prediction_result['confidence_upper_inr']],
                            y=[1, 1, 1],
                            mode='markers+lines',
                            marker=dict(size=[12, 20, 12], color=['lightcoral', 'darkred', 'lightcoral']),
                            line=dict(color='red', width=3),
                            name='INR Range',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                    fig.update_layout(height=300, showlegend=False)
                    fig.update_yaxes(visible=False)
                    fig.update_xaxes(title_text="Salary (USD $)", row=1, col=1)
                    fig.update_xaxes(title_text="Salary (INR ₹)", row=1, col=2)

                    st.plotly_chart(fig, use_container_width=True)

                    # Employee profile summary
                    with st.expander("👤 Employee Profile Summary", expanded=False):
                        profile_df = pd.DataFrame([employee_data]).T
                        profile_df.columns = ['Value']
                        profile_df.index.name = 'Attribute'
                        st.dataframe(profile_df, use_container_width=True)

                    # Add insights and recommendations section
                    st.markdown("### 💡 Insights & Recommendations")

                    insights = []

                    # Education impact
                    if education in ['Master', 'PhD']:
                        insights.append("✅ Your advanced degree positively impacts your salary potential")
                    elif education in ['High School', 'Associate']:
                        insights.append("💡 Consider pursuing higher education to increase salary potential")

                    # Experience impact
                    if experience < 2:
                        insights.append(
                            "⏳ As you gain more experience, your salary potential will increase significantly")
                    elif experience > 10:
                        insights.append("✅ Your extensive experience is a valuable asset in salary negotiations")

                    # Industry impact
                    high_paying_industries = ['Technology', 'Finance']
                    if industry in high_paying_industries:
                        insights.append(f"✅ {industry} is typically a high-paying industry")

                    # Location impact - NEW VERSION WITH INDIAN CITIES
                    location_insights = {
                        'US_cities': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston',
                                      'Seattle', 'Austin', 'Denver', 'Atlanta', 'Dallas'],
                        'Indian_cities': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
                                          'Pune', 'Kolkata', 'Ahmedabad', 'Noida', 'Gurgaon']
                    }

                    if city in location_insights['Indian_cities']:
                        insights.append(f"🇮🇳 {city} is a major tech hub in India with growing opportunities")
                        insights.append("💡 Consider the lower cost of living advantage in Indian cities")
                    elif city in location_insights['US_cities']:
                        high_cost_us_cities = ['New York', 'San Francisco', 'Seattle', 'Boston']
                        if city in high_cost_us_cities:
                            insights.append(f"🇺🇸 {city} offers high salaries but has higher cost of living")
                        else:
                            insights.append(f"🇺🇸 {city} offers competitive salaries with reasonable cost of living")

                    # Display all insights
                    for insight in insights:
                        st.write(f"• {insight}")

                else:
                    st.error("❌ Unexpected prediction result format")
                    st.write("Debug info:", prediction_result)

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.error("Please ensure you have:")
                st.write("1. ✅ Generated sample data")
                st.write("2. ✅ Trained at least one model")
                st.write("3. ✅ Selected valid input values")

    else:
        st.info("👆 Please fill in all the required information above to get your salary prediction!")


def show_results_analytics():
    """Results and analytics page"""
    st.markdown('<h2 class="sub-header">📈 Results & Analytics</h2>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first from the Model Training page.")
        return

    results = st.session_state.predictor.results

    # Model performance overview
    st.markdown('<h3>🏆 Model Performance Overview</h3>', unsafe_allow_html=True)

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    best_model_name = st.session_state.predictor.best_model_name
    best_result = results[best_model_name]

    with col1:
        st.metric("🥇 Best Model", best_model_name)
    with col2:
        st.metric("📊 R² Score", f"{best_result['r2']:.4f}")
    with col3:
        st.metric("💲 RMSE", f"${best_result['rmse']:,.0f}")
    with col4:
        st.metric("📉 MAE", f"${best_result['mae']:,.0f}")

    # Model comparison chart
    st.markdown('<h3>📊 Model Comparison</h3>', unsafe_allow_html=True)

    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]

    fig = px.bar(
        x=model_names,
        y=r2_scores,
        title='Model R² Score Comparison',
        labels={'x': 'Models', 'y': 'R² Score'},
        color=r2_scores,
        color_continuous_scale='viridis'
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def show_reports():
    """Reports page"""
    st.markdown('<h2 class="sub-header">📑 Reports & Export</h2>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first to generate reports.")
        return

    # Report generation options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4>📊 Available Reports</h4>', unsafe_allow_html=True)

        if st.button("📋 Generate Comprehensive Report", type="primary"):
            with st.spinner("📝 Generating report..."):
                report = st.session_state.predictor.generate_report(save_to_file=False)

                st.markdown('<h3>📄 Comprehensive Project Report</h3>', unsafe_allow_html=True)
                st.text_area("Report Content:", report, height=400)

                # Download button
                st.download_button(
                    label="💾 Download Report as TXT",
                    data=report,
                    file_name=f"salary_prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    with col2:
        st.markdown('<h4>💾 Model Export</h4>', unsafe_allow_html=True)

        if st.button("💾 Save Model", type="secondary"):
            try:
                st.session_state.predictor.save_model('salary_prediction_model.pkl')
                st.success("✅ Model saved successfully!")
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")


if __name__ == "__main__":
    main()
