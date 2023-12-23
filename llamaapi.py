import torch
import json
import argparse
import threading
import logging

import sys
from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from llama_index.prompts.prompts import SimpleInputPrompt
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import StoppingCriteria, StoppingCriteriaList
from loguru import logger
from typing import List, Union

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

logging.disable(logging.WARNING)
logging.disable(logging.INFO)

def comparator(q, obj1,obj2):
     obj1.ask(q)
     obj2.ask(q)

def finetune(llmobj, tune):
    llmobj.finetuneapi(tune)
    
    
    
def get_device_map(model_name, device, do_int8):
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {0: "18GiB"}
    for i in range(1, 6):
        d[i] = "26GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16,
        no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map



class LLM:

    def __init__(self, model_code="llama2", weight="7b",  temperature=0.6):
        self.weight = weight
        self.model_code = model_code
        self.temperature = temperature
        model_id = "{}/{}".format(model_code, weight)
        self.model =  HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            system_prompt="You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
,
            query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>"),
            tokenizer_name=model_id,
            model_name=model_id,
            device_map="auto",
            #trust_remote_code=True,
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                      use_fast="/opt" not in model_id)

        self.generate_kwargs = {
            "max_new_tokens": 400,
            "min_new_tokens": 100,
            "temperature": temperature,
            "do_sample": False,
            "top_k": 4,
            "penalty_alpha": 0.6,
        }

    #def compare(self, q, a,b):
    def finetuneapi(self,tune):
        documents = SimpleDirectoryReader("data").load_data()

        from llama_index.prompts.prompts import SimpleInputPrompt
        system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

        query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        from llama_index import LangchainEmbedding, ServiceContext

        embed_model = LangchainEmbedding(
          HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )

        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=self.model,
            embed_model=embed_model
        )

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)

        query_engine = index.as_query_engine()
        response = query_engine.query("Tell me about the movie Jailer")

        print(response)

        while True:
          query=input()
          response = query_engine.query(query)
          print(response)
        
        
        
    def ask(self, prompt):
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            assert len(input_ids) == 1, len(input_ids)
            if input_ids[0][-1] == 2:
                input_ids = input_ids[:, :-1]
            input_ids = input_ids.to(0)
            generated_ids = self.model.generate(
                input_ids,N
                **self.generate_kwargs
            )
            result = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            #print(len(result))
            #print("{} {}".format(model_code, weight))
            print("Question:")
            for i in result:
                print(i)
                print("Response")
           # print(result)
      

        
        



    
            
            
