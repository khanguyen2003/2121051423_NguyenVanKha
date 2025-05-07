from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Tải mô hình và scaler
with open('scaler.pkl', 'rb') as s:
    scaler = pickle.load(s)

with open('best_diabetes_model.pkl', 'rb') as m:
    model = pickle.load(m)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            gender = request.form['Gender']
            gender_num = 1 if gender == 'Male' else 0

            glucose = float(request.form['Glucose'])
            bp = float(request.form['BloodPressure'])
            skin = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            age = float(request.form['Age'])

            # Đưa vào mảng và chuẩn hóa
            input_data = np.array([[gender_num, glucose, bp, skin, insulin, bmi, age]])
            input_scaled = scaler.transform(input_data)

            result = model.predict(input_scaled)[0]
            prediction = " Người này Có khả năng mắc bệnh tiểu đường." if result == 1 else " Người này Không mắc bệnh tiểu đường."

        except Exception as e:
            prediction = f"Lỗi xử lý đầu vào: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
