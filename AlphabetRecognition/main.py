import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH='alphabet_classifier.h5'
try:
    model=tf.keras.models.load_model(MODEL_PATH)
    print("Model lOADED")
except Exception as e:
    model=None
    print(e)

alph_map={i:chr(65+i) for i in range(26)}
def predict(img:Image.Image) -> str:
    if model is None:
        return "Error loading model"
    if img is None:
        return "Pls uplaod image"
    try:
        img=img.convert('L').resize((28,28))
        img_array=np.array(img)
        img_array=img_array/255.0
        img_array=img_array.reshape(1,28,28,1)
        preds=model.predict(img_array)
        pred_cls=np.argmax(preds,axis=1)[0]
        pred_char=alph_map.get(pred_cls,"Unknown")
        return f"Predicted Alphabet : {pred_char}"
    except Exception as e:
        return f"Error occured during prediction :{e}"
    
iface=gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type='pil',
        label='Upload an Alphabet Image',
    ),
    outputs=gr.Textbox(label='Prediction Result'),
    title="Handwritten Alphabet Recognition",
    description="Upload an image of a single handwritten English Letter",
    theme=gr.themes.Soft(),
    flagging_mode="never",
    
)


if __name__=="__main__":
    port=int(os.environ.get("PORT",7860))
    iface.launch(server_name="0.0.0.0",server_port=port,share=True)