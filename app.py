import gradio as gr
import os
import tempfile
from hybrid import predict_voice_attributes

def process_audio(audio_file):
    """Process audio file and return voice attributes"""
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
       
        if isinstance(audio_file, tuple):
            # When using microphone recording (sample_rate, audio_data)
            sample_rate, audio_data = audio_file
            
            # Create a temporary file for the audio
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "input.wav")
            
            # Import soundfile for saving audio
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
        else:
            # When using file upload (direct path)
            temp_path = audio_file
        
        print(f"Processing audio file: {temp_path}")
        
        # Make prediction
        results = predict_voice_attributes(temp_path)
        
        if results:
            gender = results["gender"]
            gender_confidence = results["gender_confidence"]
            age = results["age"]
            age_confidence = results["age_confidence"]
            
            output_text = f"## Voice Analysis Results\n\n"
            output_text += f"**Gender:** {gender} (Confidence: {gender_confidence:.2f})\n\n"
            output_text += f"**Age Group:** {age} (Confidence: {age_confidence:.2f})\n\n"
            output_text += f"**Speaker Profile:** {results['combined_label']}"
            
            return output_text
        else:
            return "Analysis failed. Please try a different audio sample."
    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return f"Error processing audio: {str(e)}\n\n```\n{trace}\n```"

# Create Gradio interface
with gr.Blocks(title="Voice Gender and Age Classifier") as demo:
    gr.Markdown("# Voice Gender and Age Classifier")
    gr.Markdown("Upload an audio file of a person speaking to classify their gender and approximate age group.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"], 
                type="filepath",
                label="Upload or Record Audio"
            )
            submit_btn = gr.Button("Analyze Voice")
        
        with gr.Column():
            output = gr.Markdown(label="Results")
    
    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=output
    )
    
    gr.Markdown("## How it works")
    gr.Markdown("""
    This model analyzes the acoustic properties of your voice to determine gender and approximate age group.
    
    The system:
    1. Extracts acoustic features (pitch, formants, spectral shape, etc.)
    2. First classifies gender (Male/Female)
    3. Uses gender-specific models to predict age group (Young 20s/Older 50s)
    
    For best results:
    - Use clear audio with minimal background noise
    - Provide at least 3-5 seconds of speech
    - Speak naturally in your normal voice
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()