from flask import Flask, render_template ,request , url_for, redirect
import pickle
import jsonify
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model2.pkl'
classifier = pickle.load(open(filename, 'rb'))


app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # sl = float(request.form['sl'])
        # sw = float(request.form['sw'])
        # pl = float(request.form['pl'])
        # pw = float(request.form['pw'])

        # data = np.array([[sl,sw,pl,pw]])

        # Y_pred = classifier.predict(data)
        # print(Y_pred)
        Location_Score = float(request.form['location_score'])
        Internal_Audit_Score = float(request.form['interal_audit_score'])
        External_Audit_Score  = float(request.form['external_audit_score'])
        Fin_Score = float(request.form['fin_score'])
        Loss_score = float(request.form['loss_score'])
        Past_Results = float(request.form['past_results'])

        Total_Audit_Score = (float(Internal_Audit_Score) + float(External_Audit_Score))/2
        Internal_Total  = float(Internal_Audit_Score/Total_Audit_Score)
        External_Total = float(External_Audit_Score/Total_Audit_Score)

        data = np.array([[Location_Score,Internal_Audit_Score,External_Audit_Score,Fin_Score,Loss_score,Past_Results,Total_Audit_Score,Internal_Total,External_Total]])
        # data = np.array([[0.07,0.67,0.73,0.93,0.30,0.1,0.7, 0.95,1.047]])
        Y_pred = classifier.predict(data)
        # print(Y_pred)
        # print(int(Y_pred))


        return render_template('result.html',result=int(Y_pred))

@app.route('/go_back')
def go_back():
    return redirect(url_for('home'))





if __name__ == "__main__":
    app.run(debug=True)
