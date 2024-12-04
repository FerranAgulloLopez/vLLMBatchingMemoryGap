from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "facebook/opt-1.3b"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("./opt-1.3b")
tokenizer.save_pretrained("./opt-1.3b")

