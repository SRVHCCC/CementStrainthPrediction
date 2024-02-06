from flask import Flask,render_template,request
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline



application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return "Hello"

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            cement = float(request.form.get('cement')),
            blast_furnace_slag = float(request.form.get('blast_furnace_slag')),
            fly_ash = float(request.form.get('fly_ash')),
            water = float(request.form.get('water')),
            superplasticizer = float(request.form.get('superplasticizer')),
            coarse_aggregate = float(request.form.get('coarse_aggregate')),
            fine_aggregate = float(request.form.get('fine_aggregate')),
            age = int(request.form.get('age'))
            )


    df1=data.get_data_as_dataframe()

    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(df1)

    results=round(pred[0],2)

    return render_template('result.html',result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
