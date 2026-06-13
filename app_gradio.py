import gradio as gr
import joblib
import os
import sys
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

src_path = str(BASE_DIR / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from preprocessing import clean_text
    print("✅ Preprocessing module loaded from src/")
except ImportError as e:
    print(f"❌ Error: Could not find preprocessing.py in src folder. {e}")
    sys.exit()

BEST_MODEL_PATH = r'models\best_RF_model.pkl' 
TFIDF_PATH = r'models\tfidf.pkl'
LE_PATH = r'models\label_encoder.pkl'

model = joblib.load(BEST_MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
le = joblib.load(LE_PATH)


# os.path.basename بتعرف تجيب اسم الملف لوحدها من أي مسار
print(f"✅ Loaded: {os.path.basename(BEST_MODEL_PATH)}")
#print(f"✅ Loaded: {BEST_MODEL_PATH.split('\\')[-1]}")

def predict_ticket_category(subject, description):
    if not subject or not description:
        return "Please enter both Subject and Description."
    
    full_text = f"{subject} {description}"
 
    processed_text = clean_text(full_text)
    
    vectorized_text = tfidf.transform([processed_text])
    prediction = model.predict(vectorized_text)
    
    try:
        category = le.inverse_transform(prediction)[0]
    except:
        category = str(prediction[0])
        
    return category

# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎫 Customer Support Ticket Classifier")
    gr.Markdown(f"Powered by: **{os.path.basename(str(BEST_MODEL_PATH))}**")
    
    with gr.Row():
        with gr.Column():
            sub_input = gr.Textbox(
                lines=2, 
                label="Ticket Subject", 
                placeholder="e.g., VPN not connecting"
            )
            desc_input = gr.Textbox(
                lines=5, 
                label="Ticket Description", 
                placeholder="Enter the issue details here..."
            )
            submit_btn = gr.Button("🚀 Classify Ticket", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Predicted Category", 
                placeholder="Result will appear here...",
                interactive=False
            )
    
    gr.Examples(
        examples=[
            ["Internet connection issue", "My router is not connecting to the internet despite several restarts."],
            ["Overcharged on invoice", "I noticed an extra $20 on my billing statement this month."],
            ["Can't access my account", "I forgot my password and the reset link is not arriving."]
        ],
        inputs=[sub_input, desc_input]
    )
    
    submit_btn.click(
        fn=predict_ticket_category, 
        inputs=[sub_input, desc_input], 
        outputs=output_text
    )

# Running Application
if __name__ == "__main__":
    demo.launch(share=True)