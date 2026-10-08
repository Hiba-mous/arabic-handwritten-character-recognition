import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------
# Configuration de la page & thème
# ----------------------------------------
st.set_page_config(
    page_title="Reconnaissance de Lettres Arabes",
    page_icon="ض",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    /* Appliquer la police et le fond global */
    body, .block-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        background-color: #f4f6f8;
    }
    /* En-tête principal */
    .main-header {
        color: #1E3A8A;
        text-align: center;
        padding: 20px 0;
        font-size: 2.5rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        margin-bottom: 30px;
    }
    /* Style du bouton prédire */
    div.stButton > button {
        background-color: #3B82F6;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
    }
    /* Pied de page */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #6B7280;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E5E7EB;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------------------
# Chargement du modèle (mis en cache)
# ----------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(path: str):
    return load_model(path)

model = load_trained_model("best.keras")

# ----------------------------------------
# Labels des classes
# ----------------------------------------
CLASSES = [
    'ألف','باء','تاء','ثاء','جيم','حاء','خاء','دال','ذال',
    'راء','زاي','سين','شين','صاد','ضاد','طاء','ظاء','عين',
    'غين','فاء','قاف','كاف','لام','ميم','نون','هاء','واو','ياء'
]

# ----------------------------------------
# Barre latérale
# ----------------------------------------
st.sidebar.markdown("**FST Mohammedia — Université Hassan II**")
st.sidebar.title("À propos")
st.sidebar.markdown(
    """
    **Binôme :** Hiba Moussadek et Drissi Marwane  
    **Encadré par :** Prof. N Azouagh  
    **Université :** Hassan II - Casablanca  
    **Département :** Mathématiques  
    **Année :** 2024-2025
    """
)
st.sidebar.divider()
st.sidebar.title("Paramètres de dessin")
stroke_width = st.sidebar.slider("Épaisseur du trait", 1, 40, 15)
background_color = st.sidebar.color_picker("Couleur de fond", "#000000")
stroke_color = st.sidebar.color_picker("Couleur du trait", "#FFFFFF")

# Ajouter un bouton pour réinitialiser le canvas
if st.sidebar.button("Réinitialiser le cadre"):
    st.experimental_rerun()

st.sidebar.divider()
st.sidebar.title("Options")
uploaded = st.sidebar.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])

# ----------------------------------------
# Contenu principal
# ----------------------------------------
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("<div class='main-header'>Reconnaissance de Lettres Arabes</div>", unsafe_allow_html=True)
    st.markdown(
        "Dessinez ou importez une lettre arabe manuscrite pour la reconnaissance."
    )

    # Préparation du canvas ou affichage de l'image importée
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Image importée", use_column_width=True)
        img_arr = np.array(img)
        process_src = img_arr
    else:
        canvas_result = st_canvas(
            fill_color=background_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=background_color,
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
        if canvas_result.image_data is not None:
            process_src = canvas_result.image_data
        else:
            process_src = None

    # Bouton de prédiction
enable_predict = process_src is not None
if st.button("Prédire", disabled=not enable_predict):
    with st.spinner("Prédiction en cours..."):
        # Prétraitement
        img_in = cv2.resize(process_src.astype(np.uint8), (32, 32))
        gray = cv2.cvtColor(img_in, cv2.COLOR_RGBA2GRAY) if process_src.shape[2]==4 else cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
        gray = cv2.flip(gray, 1)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        inp = gray / 255.0
        inp = inp.reshape(1, 32, 32, 1)

        # Prédiction
        preds = model.predict(inp)
        idx = int(np.argmax(preds))
        label = CLASSES[idx]
        conf = preds[0][idx] * 100

    # Affichage des résultats
    with col2:
        st.subheader("Résultat :")
        st.metric(label="Lettre reconnue", value=label, delta=f"Confiance: {conf:.1f}%")

        # Afficher le top 5
        st.subheader("Top 5 prévisions")
        top5 = np.argsort(preds[0])[-5:][::-1]
        probs = preds[0][top5] * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([CLASSES[i] for i in top5], probs)
        ax.set_ylabel('Probabilité (%)')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

# Pied de page
st.markdown(
    """
    <div class='footer'>
        © 2025 · Projet Reconnaissance Lettres Arabes · FST Mohammedia
    </div>
    """, unsafe_allow_html=True
)


