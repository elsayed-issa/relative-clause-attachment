import pandas as pd
from datasets import Dataset
from prompts import prompt1, prompt2
from llms import llama, gpt, gemini, claude

def dataset(file_path):
    df = pd.read_csv(file_path, sep="\t")
    return Dataset.from_pandas(df)

def add_prompt_to_dataset(batch, prompt):
    if prompt == 1:
        batch['prompt'] = prompt1(batch['sentence'], batch['question'])
    elif prompt == 2:
        batch['prompt'] = prompt2(batch['sentence'], batch['question'], batch['choices'])
    return batch

def dataset_with_prompts(data):
    p1 = data.map(add_prompt_to_dataset, fn_kwargs={'prompt': 1})
    p2 = data.map(add_prompt_to_dataset, fn_kwargs={'prompt': 2})
    return p1, p2

class OllamaResponses:
    def __init__(self, path, model):
        self.data = dataset(path)
        self.model = model
        self.p1, self.p2 = dataset_with_prompts(self.data)
    
    def res(self, batch):
        batch['output'] = llama(prompt=batch['prompt'], model=self.model)
        return batch
    
    def get_responses(self, p):
        if p == 1:
            res = self.p1.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        elif p == 2:
            res = self.p2.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        return res

class GPTResponses:
    def __init__(self, path, model):
        self.data = dataset(path)
        self.model = model
        self.p1, self.p2 = dataset_with_prompts(self.data)
    
    def res(self, batch):
        batch['output'] = gpt(prompt=batch['prompt'], model=self.model)
        return batch
    
    def get_responses(self, p):
        if p == 1:
            res = self.p1.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        elif p == 2:
            res = self.p2.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        return res

class GeminiResponses:
    def __init__(self, path, model):
        self.data = dataset(path)
        self.model = model
        self.p1, self.p2 = dataset_with_prompts(self.data)
    
    def res(self, batch):
        batch['output'] = gemini(prompt=batch['prompt'], model=self.model)
        return batch
    
    def get_responses(self, p):
        if p == 1:
            res = self.p1.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        elif p == 2:
            res = self.p2.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        return res

class ClaudeResponses:
    def __init__(self, path, model):
        self.data = dataset(path)
        self.model = model
        self.p1, self.p2 = dataset_with_prompts(self.data)
    
    def res(self, batch):
        batch['output'] = claude(prompt=batch['prompt'], model=self.model)
        return batch
    
    def get_responses(self, p):
        if p == 1:
            res = self.p1.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        elif p == 2:
            res = self.p2.map(self.res)
            df = res.to_pandas()
            df.to_excel(f'{self.model}_{p}.xlsx', index=False)
        return res

#if __name__ == "__main__":
    #o = OllamaResponses("dataset.tsv", "llama3")
    #o.get_responses(2)

    #g3 = GPTResponses("dataset.tsv", "gpt-3.5-turbo")
    #g3.get_responses(2)
    
    #g = GPTResponses("dataset.tsv", "gpt-4o")
    #g.get_responses(2)

    #gem = GeminiResponses("dataset.tsv", "gemini-pro")
    #gem.get_responses(2)

    #c = ClaudeResponses("dataset.tsv", "claude-3-opus-20240229")
    #c.get_responses(2)

