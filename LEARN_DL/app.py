import os

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

personalities = {
    "Friendly":
    """You are a friendly, enthusiastic, and highly encouraging Study Assistant.
    Your goal is to break down complex concepts into simple, beginner-friendly explanations.
    Use analogies and real-world examples that beginners can relate to.
    Always ask a follow-up question to check understanding.""",

    "Academic":
    """You are a strictly academic, highly detailed, and professional university Professor.
    Use precise, formal terminology, cite key concepts and structure your response.
    Your goal is to break down complex concepts into simple, beginner-friendly explanations.
    Use analogies and real-world examples that beginners can relate to.
    Always ask a follow-up question to check understanding."""
}


def study_assistant(question, persona):

    if not question.strip():
        return "Please enter a question."

    system_prompt = personalities[persona]

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=2000
            )
        )

        return response.text

    except errors.ServerError:
        return "Gemini servers are busy. Please try again in a few minutes."


demo = gr.Interface(
    fn=study_assistant,

    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Example: Explain Transformers like I'm a beginner.",
            label="Question"
        ),

        gr.Radio(
            choices=list(personalities.keys()),
            value="Friendly",
            label="Personality"
        )
    ],

    outputs=gr.Textbox(
        lines=10,
        label="AI Response"
    ),

    title="📚 AI Study Assistant",

    description="""
Ask any study-related question.

Choose a personality before submitting your question.
""",

    submit_btn="Get Answer",
    clear_btn="Clear",
    # allow_flagging="never"
)

# demo.launch()
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)