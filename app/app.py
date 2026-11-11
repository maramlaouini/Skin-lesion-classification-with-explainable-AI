"""Prototype de recherche (Gradio). RUN_DIR=runs/efficientnet_b0_wce_s0 python app/app.py"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, gradio as gr
from PIL import Image
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.common import CLASSES, FULL_NAMES, MEAN, STD, softmax
from src.models import load_run, target_layer

RUN = os.environ.get("RUN_DIR", "runs/efficientnet_b0_ce_s0")
dev = torch.device("cpu")
model, cfg = load_run(RUN, dev)
TEMP = json.load(open(Path(RUN) / "temperature.json"))["T"]
SIZE = cfg["size"]
tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

DISCLAIMER = ("⚠️ **PROTOTYPE DE RECHERCHE UNIQUEMENT.** Cet outil n'est **pas un dispositif médical** et "
              "**ne doit pas être utilisé pour établir un diagnostic**. Entraîné sur HAM10000 (peaux claires "
              "majoritairement, 2 centres) : généralisation non garantie. Consultez un professionnel de santé "
              "pour toute lésion cutanée.")


def predict(img):
    if img is None: return None, None, ""
    img = img.convert("RGB"); x = tf(img)[None]
    with torch.no_grad(): logits = model(x).numpy()
    p = softmax(logits / TEMP)[0]; top = int(p.argmax())
    with GradCAM(model=model, target_layers=[target_layer(model, cfg["model"])]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    rgb = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32) / 255
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    note = f"Classe la plus probable : **{FULL_NAMES[CLASSES[top]]}** ({p[top]:.1%}, probabilités calibrées, T={TEMP:.2f})."
    if p[top] < 0.5: note += "\n\n⚠️ Confiance faible : prédiction très incertaine."
    note += ("\n\nLa carte indique les zones qui influencent le modèle, **pas** une preuve de causalité "
             "clinique ; elle peut montrer des artefacts (règle, bords, marqueurs).")
    return {f"{c} – {FULL_NAMES[c]}": float(v) for c, v in zip(CLASSES, p)}, overlay, note


with gr.Blocks(title="Skin lesion XAI – prototype de recherche") as demo:
    gr.Markdown("# Classification de lésions cutanées avec IA explicable"); gr.Markdown(DISCLAIMER)
    with gr.Row():
        inp = gr.Image(type="pil", label="Image dermatoscopique")
        with gr.Column():
            lbl = gr.Label(num_top_classes=7, label="Scores par classe")
            cam_out = gr.Image(label="Carte Grad-CAM")
    txt = gr.Markdown()
    gr.Button("Analyser", variant="primary").click(predict, inp, [lbl, cam_out, txt])
    gr.Markdown(DISCLAIMER)

if __name__ == "__main__":
    demo.launch()
