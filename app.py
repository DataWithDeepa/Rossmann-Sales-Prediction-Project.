import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ---------------------------
# Load trained model and data
# ---------------------------
model_path = r"C:\Users\DEEPA\data\PROJECT 6 SHEET\models\rossmann_model.pkl"
model = joblib.load(model_path)

store_df = pd.read_csv(r"C:\Users\DEEPA\data\PROJECT 6 SHEET\Store.csv", low_memory=False)
train_df_for_features = pd.read_csv(r"C:\Users\DEEPA\data\PROJECT 6 SHEET\train.csv", low_memory=False)

# Get trained model feature names
trained_features = model.feature_names_in_

# ---------------------------
# Helper functions
# ---------------------------
def prepare_data(data_df, store_df):
    # Ensure Store column present
    if 'Store' not in data_df.columns:
        if 'store' in data_df.columns:
            data_df.rename(columns={'store': 'Store'}, inplace=True)
        elif 'Store ID' in data_df.columns:
            data_df.rename(columns={'Store ID': 'Store'}, inplace=True)
    
    merged_df = pd.merge(data_df, store_df, on='Store', how='left')

    # Handle missing competition/promo columns
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                'Promo2SinceWeek', 'Promo2SinceYear']:
        if col not in merged_df.columns:
            merged_df[col] = 0
        else:
            merged_df[col] = merged_df[col].fillna(0)
    return merged_df

def align_and_predict(processed_df, model, trained_features):
    for c in trained_features:
        if c not in processed_df.columns:
            processed_df[c] = 0
    processed_df = processed_df[trained_features]
    prediction = model.predict(processed_df)
    return max(0, prediction[0])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Rossmann Sales Prediction")
st.markdown("Enter details for a single prediction or upload a CSV for bulk prediction.")

# --- Single Prediction ---
st.header("Single Store Prediction")
store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
prediction_date = st.date_input("Date", date.today())

holiday_type = st.selectbox(
    "State Holiday",
    options=["None (0)", "Public Holiday (a)", "Easter Holiday (b)", "Christmas (c)"]
)
school_holiday = st.checkbox("Is School Holiday?")
is_promo = st.checkbox("Is there a promotion running?")
is_weekend = prediction_date.weekday() >= 5
st.write(f"**Weekend automatically detected:** {'Yes' if is_weekend else 'No'}")

if st.button("Predict Sales"):
    selected_store = store_df[store_df['Store'] == store_id]
    if selected_store.empty:
        st.error(f"Store ID {store_id} not found.")
    else:
        input_data = pd.DataFrame([{
            'Store': store_id,
            'Date': prediction_date.strftime('%d-%m-%Y'),
            'Open': 1,
            'DayOfWeek': prediction_date.weekday() + 1,
            'Promo': 1 if is_promo else 0,
            'StateHoliday': '0' if holiday_type.startswith("None") else holiday_type.split()[1].strip("()"),
            'SchoolHoliday': 1 if school_holiday else 0
        }])
        prepared_data = prepare_data(input_data, store_df)
        predicted_sales = align_and_predict(prepared_data, model, trained_features)
        st.success(f"Predicted Sales for Store {store_id} on {prediction_date.strftime('%d-%m-%Y')}: ₹ {predicted_sales:,.2f}")

# --- Bulk Prediction ---
st.header("Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with Store and Date columns", type=['csv'])
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    if 'Store' not in user_df.columns:
        if 'store' in user_df.columns:
            user_df.rename(columns={'store': 'Store'}, inplace=True)
        elif 'Store ID' in user_df.columns:
            user_df.rename(columns={'Store ID': 'Store'}, inplace=True)

    results = []
    for idx, row in user_df.iterrows():
        row_df = pd.DataFrame([row])
        if 'Store' not in row_df.columns:
            if 'store' in row_df.columns:
                row_df.rename(columns={'store': 'Store'}, inplace=True)
            elif 'Store ID' in row_df.columns:
                row_df.rename(columns={'Store ID': 'Store'}, inplace=True)

        prepared = prepare_data(row_df, store_df)
        sales = align_and_predict(prepared, model, trained_features)
        results.append(sales)
    user_df['PredictedSales'] = results

    st.write("### Prediction Results")
    st.dataframe(user_df)
    st.download_button(
        label="Download Predictions as CSV",
        data=user_df.to_csv(index=False).encode('utf-8'),
        file_name="predictions.csv",
        mime="text/csv"
    )
    st.line_chart(user_df[['PredictedSales']])
