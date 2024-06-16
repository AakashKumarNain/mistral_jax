# Common instructions
1. Clone the repo.
2. Install the required packages `pip install -r requirements.txt`
3. Copy the mistral weights and other downloaded files to `mistral_jax/model_files/` directory. You can find the instructions to download the original weights [here](https://mistral.ai/news/announcing-mistral-7b/)


# Running the simple model

1. `cd mistral_jax`
2. Run `python -m one_file_ref.py` to port the model, and generate text.


# Running the optimized model

1. `cd mistral_jax`
2. We need to modify the original torch weights `state_dict()` to adjust the params shape as per the optimized model. To do this, first run `python -m torch_weights_modification.py`
3. Once we have the modified state dict, we would like to load these weights into our Equinox model, and serialize them on the disk. To do this, run `python -m weights_utils.py`
4. To generate text with this Equinox model, run `python -m generation.py`
