import gradio as gr
import joblib
import numpy as np
import pandas as pd


model = joblib.load("model.joblib")
def predict_addiction(age, daily_screen, social_media, work_study, notifications, weekend_screen, gaming_hours, gender, stress_level, academic_impact, sleep_hours):
    
    input_data = pd.DataFrame([{
        "id": 0,
        "age": age, 
        "daily_screen_time_hours": daily_screen, 
        "social_media_hours": social_media, 
        "work_study_hours": work_study, 
        "gaming_hours": gaming_hours,
        "notifications_per_day": notifications, 
        "weekend_screen_time": weekend_screen, 
        "sleep_hours": sleep_hours, 
        "gender": gender, 
        "stress_level": stress_level, 
        "academic_work_impact": academic_impact
    }])
    
    pred_proba = model.predict_proba(input_data)[0][1]
    return f"🔧 Addiction probability: {pred_proba:.2%}"

demo = gr.Interface(
    fn=predict_addiction,
    inputs=[
        gr.Slider(1, 50, step=1, label="Age"),
        gr.Slider(0, 20, step=1, label="Daily screen hours"),
        gr.Slider(0, 20, step=1, label="Social media hours"),
        gr.Slider(0, 5, step=1, label="Gaming hours"),
        
        gr.Slider(0, 10, step=1, label="Work study hours"),
        gr.Slider(1, 300, step=1, label="Notification per day"),
        gr.Slider(1, 20, step=1, label="Weekend screen time"),
        gr.Slider(1, 10, step=1, label="Sleep hours"),
        gr.Radio(["Male", "Female", "Other"], label="Gender"),
        gr.Radio(["Low", "Medium", "High"], label="Stress level"),
        gr.Radio(["True", "False"], label="Academic work impact")
    ],
    outputs="text",
    title="Phone addiction prediction",
    description="Get a probability score of phone addiction based on your inputs."
)

demo.launch(server_name="0.0.0.0", server_port=7860)