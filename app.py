# MIT License
# 
# Copyright (c) 2024 Pedro Sánchez Alvarez (pedrosanchezal@gmail.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import asyncio  # Usar asyncio para operaciones asincrónicas
from datetime import datetime
import time
import json
import base64
import io
import re
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import openai
import google.generativeai as genai
import gradio as gr
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import aiofiles  # Importar aiofiles para I/O asincrónico

# Configuración de claves API desde variables de entorno
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

# Configuración de constantes globales
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8000,
    "response_mime_type": "application/json",
}
MODEL_NAME = "gemini-1.5-flash"

# Configurar el logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def upload_to_gemini(path, mime_type=None):
    """Sube el archivo dado a Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    logging.info(f"Archivo '{file.display_name}' subido como: {file.uri}")
    return file

async def wait_for_files_active(files, sleep_interval=5):
    """Espera a que los archivos dados estén activos."""
    logging.info("Esperando a que los archivos sean procesados...")
    tasks = [check_file_status(file.name, sleep_interval) for file in files]
    await asyncio.gather(*tasks)

async def check_file_status(name, sleep_interval):
    """Comprueba el estado de un archivo y espera hasta que esté activo."""
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
        await asyncio.sleep(sleep_interval)
        file = genai.get_file(name)
    if file.state.name != "ACTIVE":
        raise Exception(f"El archivo {file.name} falló al procesarse")
    logging.info(f"Archivo {file.name} está listo")

def process_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None

def get_chat_response(files, language):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=GENERATION_CONFIG,
    )

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    files[0],
                ],
            },
        ]
    )

    prompt = generate_prompt(language)
    data = None
    for attempt in range(2):
        response = chat_session.send_message(prompt)
        data = process_json_response(response.text)
        if data is not None:
            break
        logging.warning("Error al procesar el JSON, reintentando...")

    if data is None:
        raise Exception("Error al procesar el JSON después de dos intentos.")

    return data

def generate_prompt(language):
    language = language.lower()
    if language == "english":
        return (
            "Generate an in-depth and coherent interview in dialogue format between an interviewer (male) and a speaker (female) that accurately reflects the key aspects, "
            "insights, and nuances of the provided document. The interview should include a brief introduction by the interviewer, followed by a series of thoughtful questions "
            "and detailed responses, and conclude with a concise summary or conclusion by the interviewer. The output should be a single structured JSON array, where each element "
            "is a JSON object with two fields: 'Speaker' for identifying the person speaking, and 'Content' for capturing their dialogue. Ensure the entire dialogue is contained "
            "within this array, with each dialogue entry as an individual JSON object for clarity and easy parsing."
        )
    else:
        return (
            "Genera una entrevista en profundidad y coherente en formato de diálogo entre un entrevistador (hombre) y una ponente (mujer) que refleje con precisión los aspectos clave, "
            "ideas y matices del documento proporcionado. La entrevista debe incluir una breve introducción por parte del entrevistador, seguida de una serie de preguntas reflexivas "
            "y respuestas detalladas, y concluir con un resumen o conclusión concisa por parte del entrevistador. El resultado debe ser una única matriz JSON estructurada, donde cada "
            "elemento sea un objeto JSON con dos campos: 'Speaker' para identificar a la persona que habla, y 'Content' para capturar su diálogo. Asegúrate de que todo el diálogo "
            "esté contenido dentro de esta matriz, con cada entrada de diálogo como un objeto JSON individual para mayor claridad y facilidad de análisis."
        )

async def generate_and_combine_audio_files(df_interview, output_dir, base_name):
    voices = ["echo", "nova"]
    combined = AudioSegment.empty()

    with ThreadPoolExecutor() as executor:
        tasks = []

        for index, row in df_interview.iterrows():
            tasks.append(executor.submit(generate_audio, index, row, voices, output_dir))

        for task in tasks:
            speech_file_path = await asyncio.wrap_future(task)  # Corregido el uso para envolver el futuro
            audio = AudioSegment.from_file(speech_file_path)
            combined += audio

    combined_file_path = output_dir / f"{base_name}.mp3"
    combined.export(combined_file_path, format="mp3", bitrate="128k")
    logging.info(f"Archivo de audio combinado guardado en {combined_file_path}")

    return combined_file_path

def generate_audio(index, row, voices, output_dir):
    texto = row['Content']
    speech_file_path = output_dir / f"speech_{index + 1}.mp3"
    voice = voices[index % len(voices)]

    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        response_format="mp3",
        speed=1,
        input=texto
    )

    with open(speech_file_path, 'wb') as f:
        f.write(response.content)

    return speech_file_path

async def main_async(pdf_file, language):
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Guardar el contenido del archivo PDF en el directorio temporal
        temp_pdf_path = temp_dir_path / "input.pdf"
        pdf_content = pdf_file.read() if not isinstance(pdf_file, (str, Path)) else open(pdf_file, 'rb').read()
        
        async with aiofiles.open(temp_pdf_path, 'wb') as f:
            await f.write(pdf_content)
        
        files = [upload_to_gemini(temp_pdf_path)]
        await wait_for_files_active(files)

        data = get_chat_response(files, language)
        
        # Crear un DataFrame y guardarlo como CSV en el directorio temporal
        df_interview = pd.DataFrame(data)
        csv_file_path = temp_dir_path / "interview.csv"
        df_interview.to_csv(csv_file_path, index=False)

        # Generar y combinar archivos de audio en el directorio temporal
        audio_output_dir = temp_dir_path / "audio_files"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        base_name = "combined_interview"
        combined_audio_path = await generate_and_combine_audio_files(df_interview, audio_output_dir, base_name)

        # Leer el archivo de audio combinado
        async with aiofiles.open(combined_audio_path, 'rb') as f:
            audio_content = await f.read()

    return audio_content

def gradio_interface(pdf_file, language):
    try:
        # Llamada a la función principal que procesa el archivo y genera el audio
        audio_content = asyncio.run(main_async(pdf_file, language))
        
        # Guardar el contenido de audio directamente como MP3
        with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_audio.write(audio_content)

        return temp_audio.name
    except Exception as e:
        # Loguear el error para diagnóstico
        logging.error(f"Error en el procesamiento: {str(e)}")

        # Retornar un mensaje claro a la interfaz de usuario
        return f"Error en el procesamiento: {str(e)}"

# Ejemplos preconfigurados para la sección Examples
example_data = [
    {
        "report_name": "<a href='https://huggingface.co/spaces/peter020202/Doc-To-Interview-Generator/blob/main/Example/MckinseyTrends2024.pdf'>Mckinsey Trends 2024 <br><br>[Report in English - Interview in English]</a>",
        "image": "Example/MckinseyTrends2024.png",
        "interview": "Example/McKinsey_en.mp3",
        "report_pdf": "Example/MckinseyTrends2024.pdf",
    },
    {
        "report_name": "<a href='https://huggingface.co/spaces/peter020202/Doc-To-Interview-Generator/blob/main/Example/h2ogpt.pdf'>h2oGPT: Democratizing Large Language Models <br><br>[Report in English - Interview in Spanish]</a>",
        "image": "Example/h2ogpt.png",
        "interview": "Example/h2ogpt.mp3",
        "report_pdf": "Example/h2ogpt.pdf",
    },
    {
        "report_name": "<a href='https://huggingface.co/spaces/peter020202/Doc-To-Interview-Generator/blob/main/Example/RealEstateOutlookMS.pdf'>Real Estate Outlook 2024 Morgan Stanley <br><br>[Report in English - Interview in English]</a>",
        "image": "Example/RealEstateOutlookMS.png",
        "interview": "Example/RealEstateOutlookMS.mp3",
        "report_pdf": "Example/RealEstateOutlookMS.pdf",
    },
    {
        "report_name": "<a href='https://huggingface.co/spaces/peter020202/Doc-To-Interview-Generator/blob/main/Example/ManualPatrimonio2023_es_es.pdf'>Manual de declaración de la Renta en España 2023 <br><br>[Report in Spanish - Interview in Spanish]</a>",
        "image": "Example/ManualPatrimonio2023_es_es.png",
        "interview": "Example/ManualPatrimonio2023_es_es.mp3",
        "report_pdf": "Example/ManualPatrimonio2023_es_es.pdf",
    },
]

# Crear la interfaz

# Estilo CSS para centrar el contenido y limitar la altura de las filas
css = """
#title, #description {
    text-align: center;
    margin: auto;
    width: 60%; /* Ajusta el ancho para móviles */
}
.example-row {
    display: flex;
    flex-direction: row; /* Coloca los elementos en fila */
    align-items: center; /* Centra los elementos verticalmente */
    justify-content: center; /* Centra los elementos horizontalmente */
    margin-bottom: 20px; /* Espacio entre filas */
}
.example-row img {
    width: 80%; /* Ajusta el ancho de la imagen */
    max-width: 150px; /* Limita el ancho máximo */
    margin-right: 10px; /* Espacio a la derecha de la imagen */
}
.example-row audio {
    width: 60%; /* Ajusta el ancho del audio */
    max-width: 150px; /* Limita el ancho máximo */
}
"""

with gr.Blocks(css=css) as demo:
    # Centrar el título y la descripción con CSS
    gr.Markdown("# Doc-To-Dialogue", elem_id="title")
    gr.Markdown(
        "Turn your documents—whether market or research reports, user guides, manuals, or others—into an engaging interview or discussion where the most relevant insights are summarized. Just upload a PDF and let us generate the final audio for you.<br><br>If you have any questions or encounter any issues, don't hesitate to contact me directly [here](https://www.linkedin.com/in/psanchezal/).", 
        elem_id="description"
    )
    gr.Markdown("<hr>")  # Línea separadora en HTML

    # Contenido principal con filas y columnas
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(file_types=[".pdf"], label="Upload your PDF file")
            language_input = gr.Dropdown(choices=["English", "Spanish"], label="Select the language for the interview", value="English")
            generate_button = gr.Button("Generate Interview")
        
        with gr.Column():
            audio_output = gr.Audio(type="filepath", label="Audio-Generated Interview")

    generate_button.click(gradio_interface, inputs=[pdf_input, language_input], outputs=audio_output)

    gr.Markdown("<hr>")  # Línea separadora en HTML
    gr.Markdown("<hr>")  # Línea separadora en HTML

    gr.Markdown("## Examples")  # Título de la sección de ejemplos

    for example in example_data:
        with gr.Row(elem_classes="example-row"):
            gr.Markdown(f"**{example['report_name']}**")
            gr.Image(value=example["image"], label="Image")
            gr.Audio(value=example["interview"], label="Interview")

# Ejecutar la interfaz
demo.launch(share=True)
