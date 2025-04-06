import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load the tokenizer and model
model_name_or_path = "t5_movies"  # local folder with your fine-tuned T5 model
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# 2. Define a function that takes the user inputs and returns the model output
def generate_response(adult, genres, keywords):
    # Create a prompt
    prompt = (f"adult: {adult}\n"
            f"genres: {genres}\n"
            f"keywords: {keywords}\n"
            "Generate title and overview:")

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate
    output_ids = model.generate(
        input_ids,
        max_length=256,
        do_sample=True,
        top_p=0.9,       # nucleus/top-p sampling
        top_k=50,
        temperature=1.0,
        repetition_penalty=1.1
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# 3. Create a Gradio interface
interface = gr.Interface(
    fn=generate_response, 
    inputs=["text", "text", "text"],  # Three text boxes for 'adult', 'genres', 'keywords'
    outputs="text",
    title="T5 Movies Demo",
    description="Enter values for adult, genres, and keywords to see your generated title and overview."
)

# 4. Launch the app
if __name__ == "__main__":
    interface.launch(share=True)