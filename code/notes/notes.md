https://huggingface.co/docs/transformers/main_classes/tokenizer

- Quick Overview of the Arguments
Argument	Description
truncation	If True, cuts input that exceeds max_length. Needed when working with long sequences.
padding	If "max_length", pads sequences up to max_length. If "longest" or True, pads dynamically.
max_length	Max number of tokens (after tokenization). Default for BERT is usually 512, but you can reduce it (e.g. to 64).
return_tensors	If set (e.g. "pt"), returns PyTorch (torch.Tensor) or TensorFlow tensors ("tf"). Useful for batching or model input.


- Choose max_length based on:

Average sentence + context length in your data:

Tokenize some examples and inspect their length.

If most are under 64 or 128 tokens, that’s a good safe cap.

Memory constraints:

Longer sequences → more memory use and slower training.

Try 64, 128, or 256 as starting points. BERT can handle up to 512, but rarely necessary.

Truncation side-effects:

If you truncate too short, some important tokens (maybe APCs or context) may get cut off.

Print some examples of truncated input to ensure no critical info is lost.