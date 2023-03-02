from flask import Flask,render_template,request,jsonify

from main import chat

app=Flask(__name__,template_folder="static/template")

@app.get("/")
def get_index():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    # TODO: check if text is valid
    response=chat(text)
    message={"answer":response}
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)