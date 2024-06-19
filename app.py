from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'






from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
def load_model():
    return joblib.load('model/3Bus_Trained_model.pkl')

def load_train_model():
    return joblib.load('model/Train_trained_model2.pkl')

model = load_model()
# train_model = load_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    SeatCount = input_data['SeatCount']
    ACNonAC = input_data['ACNonAC']
    TicketPrice = input_data['TicketPrice']
    MonthlyBookedSeats = input_data['MonthlyBookedSeats']
    NumberOfRides = input_data['NumberOfRides']
    WorkingDays = input_data['WorkingDays']
    PreviousMonthIncome = input_data['PreviousMonthIncome']
    BaseMonthlyIncome=input_data['BaseMonthlyIncome']
    Ac = input_data['Ac']
    SemiLux = input_data['SemiLux']
    Normal = input_data['Normal']
    Luxury = input_data['Luxury']

    input_features = np.array([
        SeatCount, ACNonAC, TicketPrice, MonthlyBookedSeats, NumberOfRides,
        WorkingDays, PreviousMonthIncome,BaseMonthlyIncome, Ac, SemiLux, Normal, Luxury
    ]).reshape(1, -1)

    prediction = model.predict(input_features)

    response = {
        'predicted_income': prediction[0],
        # 'input_data': {
        #     'SeatCount': SeatCount,
        #     'ACNonAC': ACNonAC,
        #     'TicketPrice': TicketPrice,
        #     'MonthlyBookedSeats': MonthlyBookedSeats,
        #     'NumberOfRides': NumberOfRides,
        #     'WorkingDays': WorkingDays,
        #     'PreviousMonthIncome': PreviousMonthIncome,
        #     'BaseMonthlyIncome': BaseMonthlyIncome,
        #     'Ac': Ac,
        #     'SemiLux': SemiLux,
        #     'Normal': Normal,
        #     'Luxury': Luxury
        # }

    }

    return jsonify(response)

train_model = load_train_model()

# Define a route for predicting train income
@app.route('/trainpredict', methods=['POST'])
def predict_train_income():
   
        input_data = request.get_json()

        # Extract input features from JSON
        TrainName = input_data['TrainName']
        AcSeatCount = input_data['AcSeatCount']
        NonAcSeatCount = input_data['NonAcSeatCount']
        TicketPrice = input_data['TicketPrice']
        MonthlyBookedAcSeats = input_data['MonthlyBookedAcSeats']
        MonthlyBookedNonAcSeats = input_data['MonthlyBookedNonAcSeats']
        NumberOfRides = input_data['NumberOfRides']
        WorkingDays = input_data['WorkingDays']
        PreviousMonthIncome = input_data['PreviousMonthIncome']
        TrainType = input_data['TrainType']  # Assuming TrainType is also provided

        # Convert to numpy array and reshape for prediction
        input_features = np.array([
            TicketPrice, AcSeatCount, NonAcSeatCount,
            MonthlyBookedAcSeats, MonthlyBookedNonAcSeats,
            NumberOfRides, WorkingDays, PreviousMonthIncome,
            TrainType
        ]).reshape(1, -1)

        # Make prediction
        prediction = train_model.predict(input_features)

        # Prepare JSON response
        response = {
            'predicted_income': float(prediction[0]),
            # 'input_data': {
            #     'TrainName': TrainName,
            #     'AcSeatCount': AcSeatCount,
            #     'NonAcSeatCount': NonAcSeatCount,
            #     'TicketPrice': TicketPrice,
            #     'MonthlyBookedAcSeats': MonthlyBookedAcSeats,
            #     'MonthlyBookedNonAcSeats': MonthlyBookedNonAcSeats,
            #     'NumberOfRides': NumberOfRides,
            #     'WorkingDays': WorkingDays,
            #     'PreviousMonthIncome': PreviousMonthIncome,
            #     'TrainType': TrainType,
            # }
        }

        return jsonify(response)



            # @app.route('/predict-web', methods=['POST', 'GET'])
            # def predict_web():
            #     if request.method == 'POST':
            #         input_data = request.form.to_dict()

            #         # Extract input features
            #         SeatCount = float(input_data['SeatCount'])  # Ensure numeric data types
            #         ACNonAC = int(input_data['ACNonAC'])
            #         TicketPrice = float(input_data['TicketPrice'])
            #         MonthlyBookedSeats = int(input_data['MonthlyBookedSeats'])
            #         NumberOfRides = int(input_data['NumberOfRides'])
            #         WorkingDays = int(input_data['WorkingDays'])
            #         PreviousMonthIncome = float(input_data['PreviousMonthIncome'])
            #         BaseMonthlyIncome = float(input_data['BaseMonthlyIncome'])
            #         Ac = int(input_data['Ac'])
            #         SemiLux = int(input_data['SemiLux'])
            #         Normal = int(input_data['Normal'])
            #         Luxury = int(input_data['Luxury'])

            #         # Create numpy array with input features
            #         input_features = np.array([
            #             SeatCount, ACNonAC, TicketPrice, MonthlyBookedSeats, NumberOfRides,
            #             WorkingDays, PreviousMonthIncome, BaseMonthlyIncome, Ac, SemiLux, Normal, Luxury
            #         ]).reshape(1, -1)

            #         # Make prediction
            #         prediction = model.predict(input_features)

            #         # Prepare context for rendering HTML template
            #         context = {
            #             'predicted_income': prediction[0],
            #             'input_data': input_data
            #         }

            #         # Render HTML template with predicted income
            #         return render_template('index.html', **context)

            #     # If GET request, render form to input data
            #     return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)






