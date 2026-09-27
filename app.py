
# Import library yang diperlukan
import streamlit as st
import pandas as pd
import joblib

# Memuat model, feature columns, dan scaler
@st.cache_resource
def load_resources():
    model = joblib.load('random_forest_model.pkl')
    # feature_columns.pkl berisi daftar akhir kolom fitur yang digunakan oleh model dan scaler.
    feature_columns = joblib.load('feature_columns.pkl')
    # Scaler ini dilatih pada SEMUA feature_columns, bukan hanya numerik.
    scaler = joblib.load('min_max_scaler.pkl')
    return model, feature_columns, scaler

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
    "Gender": "Jenis kelamin responden (Male/Female).", # Opsi telah diperbaiki
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
    # Membuat kamus untuk menyimpan semua fitur, termasuk variabel dummy
    # Menginisialisasi semua variabel dummy ke 0 terlebih dahulu
    processed_input_data = {col: 0 for col in feature_columns}

    # Mengisi fitur numerik
    processed_input_data["Age"] = age
    processed_input_data["Monthly_Income"] = monthly_income
    processed_input_data["Monthly_Expenses"] = monthly_expenses
    processed_input_data["Islamic_Financial_Literacy"] = islamic_financial_literacy
    processed_input_data["Sharia_Awareness"] = sharia_awareness
    processed_input_data["Conventional_Bank_Usage"] = conventional_bank_usage
    processed_input_data["Digital_Banking_Usage"] = digital_banking_usage
    processed_input_data["Religious_Knowledge"] = religious_knowledge
    processed_input_data["Halal_Finance_Awareness"] = halal_finance_awareness

    # Mengisi fitur kategori yang sudah di-one-hot encode secara manual
    # Gender (Gender_1 untuk Female, Male adalah base case 0)
    if gender == "Female":
        processed_input_data["Gender_1"] = 1
    # Education (sesuai dengan drop_first=True saat preprocessing)
    if education == "Diploma":
        processed_input_data["Education_Diploma"] = 1
    elif education == "High School":
        processed_input_data["Education_High School"] = 1
    elif education == "Postgraduate":
        processed_input_data["Education_Postgraduate"] = 1
    # Sharia_Banking_Interest (sesuai dengan one-hot encoding)
    if sharia_banking_interest == "High":
        processed_input_data["Sharia_Banking_Interest_High"] = 1
    elif sharia_banking_interest == "Low":
        processed_input_data["Sharia_Banking_Interest_Low"] = 1
    elif sharia_banking_interest == "Medium":
        processed_input_data["Sharia_Banking_Interest_Medium"] = 1

    # Membuat DataFrame dari data input yang telah diproses, memastikan urutan kolom sesuai
    input_df_processed = pd.DataFrame([processed_input_data], columns=feature_columns)

    # Menerapkan scaling pada SELURUH DataFrame input yang telah diproses
    # Scaler dilatih pada semua feature_columns, jadi ia mengharapkan semuanya.
    scaled_input_array = scaler.transform(input_df_processed)
    scaled_input_df = pd.DataFrame(scaled_input_array, columns=feature_columns)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(scaled_input_df)

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.success(f"Tujuan Keuangan yang Diprediksi adalah: **{prediction[0]}**")

# Bagian deskripsi kolom
st.subheader("Deskripsi Kolom")
for col, desc in column_descriptions.items():
    st.markdown(f"**{col}**: {desc}")


