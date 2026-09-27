
# Import library yang diperlukan
import streamlit as st
import pandas as pd
import joblib

# Memuat model, feature columns, dan scaler
@st.cache_resource
def load_resources():
    model = joblib.load('random_forest_model.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    scaler = joblib.load('min_max_scaler.pkl')
    return model, feature_cols, scaler

# Memuat sumber daya
model, feature_columns, scaler = load_resources()

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Prediksi Tujuan Keuangan")

# Judul aplikasi
st.title("Aplikasi Prediksi Tujuan Keuangan")
st.markdown("Isi formulir di bawah ini untuk memprediksi tujuan keuangan seseorang berdasarkan karakteristik mereka.")

# Deskripsi kolom dalam Bahasa Indonesia
column_descriptions = {
    "Age": "Usia responden dalam tahun.",
    "Monthly_Income": "Pendapatan bulanan responden.",
    "Monthly_Expenses": "Pengeluaran bulanan responden.",
    "Islamic_Financial_Literacy": "Tingkat literasi keuangan syariah (skala 1-10).",
    "Sharia_Awareness": "Tingkat kesadaran akan syariah (skala 1-10).",
    "Conventional_Bank_Usage": "Tingkat penggunaan bank konvensional (skala 1-10).",
    "Digital_Banking_Usage": "Tingkat penggunaan layanan perbankan digital (skala 1-10).",
    "Religious_Knowledge": "Tingkat pengetahuan agama (skala 1-10).",
    "Halal_Finance_Awareness": "Tingkat kesadaran akan keuangan halal (skala 1-10).",
    "Gender": "Jenis kelamin responden (Pria/Wanita).",
    "Education": "Tingkat pendidikan responden (Bachelor, Diploma, High School, Postgraduate).",
    "Sharia_Banking_Interest": "Tingkat minat terhadap perbankan syariah (High, Medium, Low)."
}

# Header untuk bagian input
st.header("Informasi Responden")

# Mengatur tata letak kolom input
col1, col2, col3 = st.columns(3)

# Input di kolom 1
with col1:
    age = st.number_input("Usia (Age)", min_value=18, max_value=100, value=30, help=column_descriptions["Age"])
    monthly_income = st.number_input("Pendapatan Bulanan (Monthly_Income)", min_value=0, value=5000000, step=100000, help=column_descriptions["Monthly_Income"])
    monthly_expenses = st.number_input("Pengeluaran Bulanan (Monthly_Expenses)", min_value=0, value=3000000, step=100000, help=column_descriptions["Monthly_Expenses"])
    islamic_financial_literacy = st.slider("Literasi Keuangan Syariah (Islamic_Financial_Literacy)", min_value=1, max_value=10, value=5, help=column_descriptions["Islamic_Financial_Literacy"])

# Input di kolom 2
with col2:
    sharia_awareness = st.slider("Kesadaran Syariah (Sharia_Awareness)", min_value=1, max_value=10, value=5, help=column_descriptions["Sharia_Awareness"])
    conventional_bank_usage = st.slider("Penggunaan Bank Konvensional (Conventional_Bank_Usage)", min_value=1, max_value=10, value=5, help=column_descriptions["Conventional_Bank_Usage"])
    digital_banking_usage = st.slider("Penggunaan Perbankan Digital (Digital_Banking_Usage)", min_value=1, max_value=10, value=5, help=column_descriptions["Digital_Banking_Usage"])
    religious_knowledge = st.slider("Pengetahuan Agama (Religious_Knowledge)", min_value=1, max_value=10, value=5, help=column_descriptions["Religious_Knowledge"])

# Input di kolom 3
with col3:
    halal_finance_awareness = st.slider("Kesadaran Keuangan Halal (Halal_Finance_Awareness)", min_value=1, max_value=10, value=5, help=column_descriptions["Halal_Finance_Awareness"])
    gender = st.selectbox("Jenis Kelamin (Gender)", options=["Male", "Female"], help=column_descriptions["Gender"])
    education = st.selectbox("Pendidikan (Education)", options=["Bachelor", "Diploma", "High School", "Postgraduate"], help=column_descriptions["Education"])
    sharia_banking_interest = st.selectbox("Minat Perbankan Syariah (Sharia_Banking_Interest)", options=["High", "Medium", "Low"], help=column_descriptions["Sharia_Banking_Interest"])


# Tombol untuk prediksi
if st.button("Prediksi Tujuan Keuangan"):
    # Menyiapkan data input
    input_data = {
        "Age": age,
        "Monthly_Income": monthly_income,
        "Monthly_Expenses": monthly_expenses,
        "Islamic_Financial_Literacy": islamic_financial_literacy,
        "Sharia_Awareness": sharia_awareness,
        "Conventional_Bank_Usage": conventional_bank_usage,
        "Digital_Banking_Usage": digital_banking_usage,
        "Religious_Knowledge": religious_knowledge,
        "Halal_Finance_Awareness": halal_finance_awareness,
        # Menggunakan 0/1 untuk fitur biner (Gender_1: Female=1, Male=0)
        "Gender_1": 1 if gender == "Female" else 0, 
        # Menggunakan 0/1 untuk fitur Education (drop_first=True)
        "Education_Diploma": 1 if education == "Diploma" else 0,
        "Education_High School": 1 if education == "High School" else 0,
        "Education_Postgraduate": 1 if education == "Postgraduate" else 0,
        # Menggunakan 0/1 untuk fitur Sharia_Banking_Interest (one-hot encoding)
        "Sharia_Banking_Interest_High": 1 if sharia_banking_interest == "High" else 0,
        "Sharia_Banking_Interest_Low": 1 if sharia_banking_interest == "Low" else 0,
        "Sharia_Banking_Interest_Medium": 1 if sharia_banking_interest == "Medium" else 0
    }

    # Membuat DataFrame dari input
    input_df = pd.DataFrame([input_data])

    # Mendefinisikan fitur numerik yang akan diskalakan (sesuai dengan training model)
    numeric_features_in_model = ['Age', 'Monthly_Income', 'Monthly_Expenses', 'Islamic_Financial_Literacy',
                               'Sharia_Awareness', 'Conventional_Bank_Usage', 'Digital_Banking_Usage',
                               'Religious_Knowledge', 'Halal_Finance_Awareness']

    # Menerapkan scaling pada fitur numerik menggunakan scaler yang sudah dilatih
    input_df[numeric_features_in_model] = scaler.transform(input_df[numeric_features_in_model])

    # Memastikan urutan kolom sesuai dengan feature_columns yang digunakan saat training
    processed_input = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in input_df.columns:
            processed_input[col] = input_df[col]
        else:
            processed_input[col] = 0  # Mengisi dengan 0 untuk variabel dummy yang hilang (jika ada)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(processed_input)

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.success(f"Tujuan Keuangan yang Diprediksi adalah: **{prediction[0]}**")

# Bagian deskripsi kolom
st.subheader("Deskripsi Kolom")
for col, desc in column_descriptions.items():
    st.markdown(f"**{col}**: {desc}")

