import gradio as gr
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("model.joblib")

def recommend_stack():
    
    input_data = np.array()
    pred_proba = model.predict_proba(input_data)[1]
    return f"🔧 Addiction probability: {pred_proba}"

demo = gr.Interface(
    fn=recommend_stack,
    inputs=
    [
        gr.Slider(1, 50, step=1, label="Age"),
        gr.Slider(0, 20, step=1, label="Daily screen hours"),
        gr.Slider(0, 20, step=1, label="Social media hours"),
        gr.Slider(0, 10, step=1, label="Work study hours"),
        gr.Slider(1, 300, step=1, label="Notification per day"),
        gr.Slider(1, 20, step=1, label="Weekend screen time"),
        gr.Slider(1, 10, step=1, label="Team Size"),
        gr.Radio(["Male", "Female", "Other"], label="Gender"),
        gr.Radio(["Low", "Medium", "High"], label="Stress level"),
        gr.Radio(["True", "False"], label="Academic work impact")
    ],
    outputs="text",
    title="Phone addiction prediction",
    description="Get a probability score of phone addiction based on your inputs."
)

demo.launch(server_name="0.0.0.0", server_port=7860)