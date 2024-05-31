
from datasets import load_dataset
from accelerate import Accelerator
from openicl import AccEvaluator, RougeEvaluator
from openicl import DatasetReader, PromptTemplate, TopkRetriever, PPLInferencer, GenInferencer, PAGenInferencer
from openicl import TopkRetriever, ZeroRetriever, RandomRetriever, PateRetriever, BM25Retriever
import openai
import os
import argparse

# set all seeds
import random
import torch
import numpy as np
random.seed(0)
np.random.seed(0)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True




def main():
     dataset = load_dataset(args.data_name,cache_dir='./data')
     data = DatasetReader(dataset, input_columns=['dialogue'], output_column='summary', ds_size=args.ds_size)
     print(len(data))
     template = PromptTemplate('</E>Dialogue:"\n </dialogue>" \nSummarize the above dialogue: </summary>', {'dialogue' : '</dialogue>', 'summary' : '</summary>'}, ice_token='</E>')

     # Select a piece of data from the dataset
     entry = dataset['train'][0]
     #print(f'entry:\n{entry}\n')

     # Generate output
     output = template.generate_item(entry)
     #print(f'output:\n{output}\n')

     # Generate masked output
     masked_output = template.generate_item(entry, output_field='summary')
     #print(f'masked output:\n{masked_output}')


     # TopK Retriever
     retriever = PateRetriever(data, ice_num=args.ice_num, ensemble = args.ensemble)

     # Define a Inferencer
     if "gpt3" not in args.model_name:
          inferencer = PAGenInferencer(model_name=args.model_name,batch_size = args.batch_size, args = args, generation_kwargs={"max_new_tokens": 200, "temperature":args.temp, "do_sample": True, "num_beams": args.nb})
     
     else:
          # export OPENAI_API_KEY="XXXX"
          inferencer = PAGenInferencer(api_name='gpt3', engine='text-babbage-001', sleep_time=0) #text-davinci-003 text-babbage-001

     # copy data.references to emsemble times
     data_references = []
     for data_refer in data.references:
          for i in range(args.ensemble):
               data_references.append(data_refer)

     predictions = inferencer.inference(retriever, ice_template=template, output_json_filename=args.output_json_filename)
     
     # predictions = inferencer.embedding(retriever, ice_template=template, output_json_filename=args.output_json_filename,
     #                                    input_json_filename=args.input_json_filename) # if you want to save the embeddings. 

     score = RougeEvaluator().score(predictions=predictions, references=data_references)
     print(score)


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='samsum') #dataset name
parser.add_argument('--model_name', type=str, default='gpt3') # model name
parser.add_argument('--batch_size', type=int, default=1) # batch size
parser.add_argument('--ice_num', type=int, default=0) # number of in-context examples
parser.add_argument('--ensemble', type=int, default=1) # number of ensemble
parser.add_argument('--ds_size', type=int, default=100) # dataset size
parser.add_argument('--output_json_filename','--ojf', type=str, default='gpt2-xl_samsum_zero') # output json filename
parser.add_argument('--temp', type=float, default=1.0) # temperature
parser.add_argument('--nb', type=int, default=1) # number of beams

args = parser.parse_args()
print(args)
main()

