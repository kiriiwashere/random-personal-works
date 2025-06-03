import requests
from PIL import Image
import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

base_model_id = "Andres77872/SmolVLM-500M-anime-caption-v0.2"

processor = AutoProcessor.from_pretrained(base_model_id)
model = Idefics3ForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequence):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        max_keep = len(self.stop_sequence) + 10
        if len(new_text) > max_keep:
            new_text = new_text[-max_keep:]
        return self.stop_sequence in new_text

def prepare_inputs(image: Image.Image):
    # IMPORTANT: The question prompt must remain fixed as "describe the image".
    # This model is NOT designed for visual question answering.
    # It is strictly an image captioning model, not intended to answer arbitrary visual questions.
    question = "describe the image"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    max_image_size = processor.image_processor.max_image_size["longest_edge"]
    size = processor.image_processor.size.copy()
    if "longest_edge" in size and size["longest_edge"] > max_image_size:
        size["longest_edge"] = max_image_size
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[[image]], return_tensors='pt', padding=True, size=size)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return inputs

# Example: caption a sample anime image
image = Image.open(requests.get('https://img.arz.ai/5A7A-ckt', stream=True).raw).convert("RGB")
inputs = prepare_inputs(image)
stop_sequence = "</RATING>"
streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)
custom_stopping_criteria = StoppingCriteriaList([
    StopOnTokens(processor.tokenizer, stop_sequence)
])

with torch.no_grad():
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        do_sample=False,
        max_new_tokens=512,
        pad_token_id=processor.tokenizer.pad_token_id,
        stopping_criteria=custom_stopping_criteria,
    )

    import threading
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()

    for new_text in streamer:
        print(new_text, end="", flush=True)

    generation_thread.join()
s