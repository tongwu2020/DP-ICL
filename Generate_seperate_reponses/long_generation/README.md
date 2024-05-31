
# Generate separate predictions for multiple choice tasks

Check two folders for long generation and multiple choice tasks. 

## Setup
Run `pip install openicl`

## Run Separate Predictions

Here is an example of running GPT-3 babbage model on the dialogue summarization task.  

`python -u run.py  --ice_num 4 --ds_size 100  --ensemble 100 --ojf ./output `

Please check the `run.py` file for more information and arguments. 

You data will be saved in `output_json_filename`

We save some data in the google colab. 

We use [OpenICL](https://github.com/Shark-NLP/OpenICL) for implementation. 

You can check [OpenICL Documentation](https://openicl.readthedocs.io/en/latest/index.html)

