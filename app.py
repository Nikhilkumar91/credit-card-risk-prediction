from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained pickle model
with open("creditcard.pkl", "rb") as f:
    model = pickle.load(f)
with open("standard_scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract values
    #DebitRatio = float(request.form["DebitRatio"])
    OpenCreditLoans = int(request.form["OpenCreditLoans"])
    RealEstateLoans = int(request.form["RealEstateLoans"])
    MonthlyIncome = float(request.form["MonthlyIncome"])
    Dependents = int(request.form["Dependents"])
    Gender = int(request.form["Gender"])
    Region = request.form["Region"]
    Rented_OwnHouse = request.form["Rented_OwnHouse"]
    Occupation = float(request.form["Occupation"])    # already numeric mapping in dropdown
    Education = float(request.form["Education"])      # already numeric mapping in dropdown

    # One-hot encode Region
    region_east = 1 if Region == "East" else 0
    region_north = 1 if Region == "North" else 0
    region_south = 1 if Region == "South" else 0
    region_west = 1 if Region == "West" else 0

    # Encode Rented/Own
    rented_ownhouse = 1 if Rented_OwnHouse == "OwnHouse" else 0

    # Arrange features in the same order as training
    features = np.array([[
                          OpenCreditLoans,
                          RealEstateLoans,
                          MonthlyIncome,
                          Dependents,
                          Gender,
                          region_east, region_north, region_south, region_west,
                          rented_ownhouse,
                          Occupation,
                          Education]])

    # Prediction
    #features_array = np.array([features])  # make it 2D
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    # Assuming 1=Good, 0=Bad (adjust if needed)
    result = "Good Customer" if prediction == 1 else "Bad Customer"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)