import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Dictionary of model name → (tokenizer, model)
models_dict = {}

# Load model 1
model_name_or_path_1 = "t5_movies_base_20000"
tokenizer_1 = T5Tokenizer.from_pretrained(model_name_or_path_1)
model_1 = T5ForConditionalGeneration.from_pretrained(model_name_or_path_1)
models_dict["T5-base-20000"] = (tokenizer_1, model_1)

# Load model 2
model_name_or_path_2 = "t5_movies_full"
tokenizer_2 = T5Tokenizer.from_pretrained(model_name_or_path_2)
model_2 = T5ForConditionalGeneration.from_pretrained(model_name_or_path_2)
models_dict["T5-small-full"] = (tokenizer_2, model_2)

"""# 1. Load the tokenizer and model
model_name_or_path = "t5_movies"  # local folder with your fine-tuned T5 model
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
"""
# 2. Define a function that takes the user inputs and returns the model output
def generate_response(model_choice,slider_value, genres, keywords):
    # Create a prompt
    tokenizer, model = models_dict[model_choice]
    top_k_selected = 10 + int((slider_value / 100) * (50 - 10))
    
    prompt = (
        f"genres: {genres}\n"
        f"keywords: {keywords}\n"
        "Generate title and overview:"
    )

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_length=256,
        do_sample=True,
        #top_p=0.9,
        top_k=top_k_selected, # recommended 30
        temperature=1.0,
        repetition_penalty=1.2
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # -- Parse the generated output into "title" and "overview"
    #    We assume the model produces something like: "title: My Movie overview: This is about..."
    try:
        # Attempt to split on 'title:' and 'overview:'
        # This will handle the format: "title: X overview: Y"
        _, after_title = output_text.split("title:", 1)
        title_part, overview_part = after_title.split("overview:", 1)
        title = title_part.strip()
        overview = overview_part.strip()
    except ValueError:
        # If parsing fails, return the entire string as title and leave overview blank
        title = output_text
        overview = ""

    # Return them as two separate fields
    return title, overview

# 3. Creating a Gradio interface
interface = gr.Interface(
    fn=generate_response, 
    inputs=[
        # Let the user pick a model
        gr.Dropdown(
            choices=list(models_dict.keys()),  
            label="Select Model"
        ),
        gr.Slider(
            minimum=0, 
            maximum=100, 
            step=1, 
            value=50, 
            label="Genericness <-> Uniqueness (Recommended: 50)"
        ),
        gr.Textbox(label="Genres", value = "Science Fiction, Action, Adventure"),
        gr.Textbox(label="Keywords", value = "future, space, battleship, planet")
    ],
    outputs=[
        gr.Textbox(label="Title"),
        gr.Textbox(label="Overview")
    ],         # Two separate text outputs: title and overview
    title="PlotCraft: Generating Movie Titles and Storylines from Prompts",
    description="Choose a model, pick how 'unique' vs. 'generic' you'd like the output, enter values for genres and keywords to see your generated title and overview."
)

# 4. Launch the app
if __name__ == "__main__":
    interface.launch(share=True)


