
# Generate separate predictions for multiple choice tasks

Check two folders for long generation and multiple choice tasks. 

## Setup

1. Create env with conda: `conda create -n active-example-selection python=3.9`.
2. Install pytorch with version >= 1.10.2.
3. Run `pip install -r requirements.txt` (add the `-e` flag if you plan to make changes).

## Separate Predictions

The following command runs separate predictions on GPT-3 model, and make sure the field `api_key_file` points to a file with your OpenAI API key:
> `python src/prompting/main.py prompting_configs/zero-baseline-gpt3-sst.yaml`

**Note:** The GPT-3 model is not available. Therefore, you should choose other models to run. 

For example, you can run GPT-2 model:
> `python src/prompting/main.py prompting_configs/baseline-gpt2-sst-multi.yaml`

You can adjust the `yaml` file in `prompting_configs` to run more experiments. 

## Zero-shot Predictions
Example:
> `python src/prompting/main.py prompting_configs/zero-baseline-gpt3-da-sst.yaml`


## After Separate Predictions

After generating the results, you can then privately aggregate the results using report-noise-max.


## Reference

Our code is adapted from [Active Example Selection for In-Context Learning](https://github.com/ChicagoHAI/active-example-selection)