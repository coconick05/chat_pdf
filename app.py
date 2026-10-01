import os
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Configuración de la página (debe ir primero)
st.set_page_config(page_title="RAG con PDF", page_icon="💬")

# Colores
FUCSIA = "#FF00FF"
LILA = "#C8A2C8"

# Estilos globales: títulos en fucsia y párrafos en lila
st.markdown(
    f"""
    <style>
    h1, h2, h3, h4, h5, h6 {{
        color: {FUCSIA} !important;
    }}
    p, label, .stMarkdown, .stAlert, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p, [data-testid="stText"] {{
        color: {LILA} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la app
st.title("Generación Aumentada por Recuperación (RAG) 💬")

# Imagen debajo del primer título
try:
    image_chatt = Image.open("chatt.jpg")
    st.image(image_chatt, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen chatt.jpg: {e}")

# Información de la barra lateral
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Obtener la clave de API del usuario
ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Cargador de PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesar el PDF si fue cargado
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        st.info(f"Texto extraído: {len(text)} caracteres")

        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz de pregunta del usuario
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(
            "Tu pregunta",
            placeholder="Escribe tu pregunta aquí...",
            label_visibility="collapsed",
        )

        # Procesar la pregunta cuando se envía
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # Modelo actual
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")

            # Cargar la cadena de QA
            chain = load_qa_chain(llm, chain_type="stuff")

            # Ejecutar la cadena
            response = chain.run(input_documents=docs, question=user_question)

            # Mostrar la respuesta
            st.markdown("### Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
