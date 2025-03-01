import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

def safe_encode(encoder, value, default_value='Unknown'):
    try:
        return encoder.transform([value])[0]
    except:
        # If value not in encoder, add it to the encoder with a default value
        all_classes = list(encoder.classes_)
        if default_value not in all_classes:
            all_classes.append(default_value)
            encoder.classes_ = np.array(all_classes)
        return encoder.transform([default_value])[0]

st.title("🚛 Overweight Incident Cost Prediction")

# File Upload
incident_file = st.file_uploader("Upload Incident Data (Excel)", type="xlsx")

if incident_file:
    try:
        # Read Data
        df = pd.read_excel(incident_file)
        
        # Data Validation
        required_columns = ['Ship From City', 'Ship From State', 'Ship To City', 
                          'Ship To State', 'Liable Party Name', 'Total Expense']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
            
        # Data Cleaning and Validation
        df = df.dropna(subset=['Total Expense'])  # Remove rows with NaN in Total Expense
        
        # Convert Total Expense to float and handle any currency formatting
        df['Total Expense'] = df['Total Expense'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        
        # Remove any negative or zero expenses
        df = df[df['Total Expense'] > 0]
        
        # Display detailed statistics
        #st.subheader("Dataset Information")
        #st.write(f"Total records: {len(df)}")
        #st.write("Expense Statistics:")
        #expense_stats = {
        #    "Average Cost": f"${df['Total Expense'].mean():.2f}",
        #    "Median Cost": f"${df['Total Expense'].median():.2f}",
        #    "Min Cost": f"${df['Total Expense'].min():.2f}",
        #    "Max Cost": f"${df['Total Expense'].max():.2f}"
        #}
        #st.write(expense_stats)
        
        # Updated histogram creation
        #fig = px.histogram(
        #    df, 
        #    x='Total Expense',
        #    title='Distribution of Return Costs',
        #    labels={'Total Expense': 'Cost ($)', 'count': 'Number of Incidents'},
        #    nbins=30
        #)
        #fig.update_layout(showlegend=False)
        #st.plotly_chart(fig)
        
        # Data Preprocessing
        location_encoder = LabelEncoder()
        party_encoder = LabelEncoder()
        
        # Fit encoders on all unique combinations
        all_locations = pd.concat([
            df['Ship From City'] + ', ' + df['Ship From State'],
            df['Ship To City'] + ', ' + df['Ship To State']
        ]).unique()
        location_encoder.fit(all_locations)
        party_encoder.fit(df['Liable Party Name'].unique())
        
        # Transform the data
        df['Ship_From_Encoded'] = location_encoder.transform(df['Ship From City'] + ', ' + df['Ship From State'])
        df['Ship_To_Encoded'] = location_encoder.transform(df['Ship To City'] + ', ' + df['Ship To State'])
        df['Liable_Party_Encoded'] = party_encoder.transform(df['Liable Party Name'])
        
        # Features and Target
        X = df[['Ship_From_Encoded', 'Ship_To_Encoded', 'Liable_Party_Encoded']]
        y = df['Total Expense']

        # Train Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.subheader(f"✅ Model R² Score: {r2:.2f}")

        # Mathematical Components Explanation:
        """
        1. Label Encoding: 
           - Converts categorical text data into numbers
           - Example: ['New York', 'Los Angeles'] → [0, 1]

        2. Linear Regression Model:
           - Formula: y = β₀ + β₁x₁ + β₂x₂ + β₃x₃
           - Where:
             * y = Predicted Total Expense
             * β₀ = Intercept (base cost)
             * β₁ = Weight for Ship From Location
             * β₂ = Weight for Ship To Location
             * β₃ = Weight for Liable Party
             * x₁, x₂, x₃ = Encoded input values

        3. R² Score (Coefficient of Determination):
           - Measures model accuracy from 0 to 1
           - Formula: R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
           - Example: R² of 0.75 means model explains 75% of variance
        """

        # User Input for Prediction
        col1, col2 = st.columns(2)
        with col1:
            ship_from = st.text_input(
                "Ship From (City, State):", 
                value="Fort Worth, TX",
                help="Format: City, ST (e.g. Fort Worth, TX)"
            ).strip()
        with col2:
            ship_to = st.text_input(
                "Ship To (City, State):", 
                value="Bloomfield, MO",
                help="Format: City, ST (e.g. Bloomfield, MO)"
            ).strip()
        liable_party = st.text_input("Liable Party Name:", value="")

        if st.button("Predict Cost"):
            try:
                # Validate input format
                if not all(',' in x for x in [ship_from, ship_to]):
                    st.error("Please use format: City, ST for both locations")
                    st.stop()
                
                # Transform input data safely
                from_encoded = safe_encode(location_encoder, ship_from)
                to_encoded = safe_encode(location_encoder, ship_to)
                party_encoded = safe_encode(party_encoder, liable_party)
                
                # Predict
                prediction = model.predict([[from_encoded, to_encoded, party_encoded]])[0]
                
                st.subheader("Prediction Results:")
                st.info(f"💰 Estimated return cost: ${prediction:.2f}")
                st.write(f"Average return cost: ${y.mean():.2f}")
                if prediction > y.mean():
                    st.warning("⚠️ This route has higher than average return costs!")

                # Explain the prediction math
                st.subheader("How the prediction works:")
                st.write("""
                1. Location encoding: Convert cities and states to numbers
                2. Linear combination: Multiply each encoded value by its learned weight
                3. Add base cost (intercept)
                4. Result: Predicted return cost in dollars
                """)
                
                model_weights = pd.DataFrame({
                    'Feature': ['Ship From Location', 'Ship To Location', 'Liable Party'],
                    'Weight': model.coef_
                })
                st.write("Model Weights:", model_weights)
                st.write(f"Base Cost (Intercept): ${model.intercept_:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
