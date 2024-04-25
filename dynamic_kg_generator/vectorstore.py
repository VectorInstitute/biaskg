from typing import Tuple
import numpy as np
from typing import List, Any, Optional, Dict

import json
from tqdm import *

import pandas as pd

import os
import openai
import pandas as pd
import re
from tqdm.notebook import tqdm



def get_top_k_embeddings(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    # dimensions: D
    qembed_np = np.array(query_embedding)
    # dimensions: N x D
    dembed_np = np.array(doc_embeddings)
    # dimensions: N
    dproduct_arr = np.dot(dembed_np, qembed_np)
    # dimensions: N
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document
    # sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids

class VectorRetrieval:

    # loading the vector stores and BBQ data
    def __init__(self, graph_file):
        # Open the file in read mode:
        with open(r'Vector Stores/start_index/vector_store.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            start_vector = json.load(f)
        with open(r'Vector Stores/start_index/index_store.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            start_index = json.load(f)
            start_index=json.loads(start_index["index_store/data"]["33a2f390-a39e-4d27-8bf0-93476c190695"]["__data__"])
        with open(r'Vector Stores/start_index/docstore.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            start_docstore = json.load(f)
            # start_docstore=start_docstore
        for i in start_docstore["docstore/data"]:
            start_docstore["docstore/data"][i]["__data__"]["embedding"]=start_vector["embedding_dict"][i]


        # Open the file in read mode:
        with open(r'Vector Stores/text_index/vector_store.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            text_vector = json.load(f)
        with open(r'Vector Stores/text_index/index_store.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            text_index = json.load(f)
            text_index=json.loads(text_index["index_store/data"]["3f747378-29c1-463a-9079-ba0d34818e47"]["__data__"])
        with open(r'Vector Stores/text_index/docstore.json', 'r') as f:
            # Load the JSON data from the file to a Python dict:
            text_docstore = json.load(f)
            # start_docstore=start_docstore
        for i in text_docstore["docstore/data"]:
            text_docstore["docstore/data"][i]["__data__"]["embedding"]=text_vector["embedding_dict"][i]

        self.start_vector = start_vector
        self.start_index = start_index
        self.start_docstore = start_docstore

        self.text_vector = text_vector
        self.text_index = text_index
        self.text_docstore = text_docstore

        self.df_bbq=pd.read_csv("bbq_top_3.csv")

        self.load_client()
        self.load_graph(graph_file=graph_file)

    # loading the OpenAI client for embeddings
    def load_client(self):
        OPENAI_API_KEY = ''
        OPENAI_API_TYPE = 'azure'
        OPENAI_API_VERSION = '2023-03-15-preview'
        OPENAI_API_BASE = ''
        DEPLOYMENT_NAME = "gpt-4"

        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        os.environ['OPENAI_API_TYPE'] = OPENAI_API_TYPE
        # API version to use (Azure has several)
        os.environ['OPENAI_API_VERSION'] = OPENAI_API_VERSION
        # base URL for your Azure OpenAI resource
        os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
        os.environ['OPENAI_API_ENGINE'] = DEPLOYMENT_NAME


        self.client = openai.AzureOpenAI(
                azure_endpoint=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION
            )

    # loading the graph as a pandas dataframe
    def load_graph(self, graph_file):
        df=pd.read_csv(graph_file)
        ls_text=[]
        ls_index=[]
        ls_graph=[]
        # (start_node, edge, end_node)
        ls_start_node=[]
        ls_edge=[]
        ls_end_node=[]
        ls_len=[]
        for index, row in df.iterrows():
            # Original string
            s = row["Graph"]
            if s.startswith("ERROR:"):
                continue 
            # Use regex to find all phrases in parentheses
            matches = re.findall(r'\((.*?)\)', s)
            # Format each match by removing commas and stripping whitespace
            ls_index+=[index for match in matches]
            ls_text+=[' '.join(match.split(',')) for match in matches]
            ls_graph+=matches
            ls_len+=[len(match.split(',')) for match in matches]
            for match in matches:
                temp=match.split(',')
                if len(temp)==3:
                    ls_start_node.append(temp[0])
                    ls_edge.append(temp[1])
                    ls_end_node.append(temp[2])
                else:
                    ls_start_node.append("")
                    ls_edge.append("")
                    ls_end_node.append("")

        df_map=pd.DataFrame({
                "original_index":ls_index,
                "graph":ls_graph,
                "text":ls_text,
                "len":ls_len,
                "start_node":ls_start_node,
                "edge":ls_edge,
                "end_node":ls_end_node
            })
        df_map=df_map[df_map.len==3].drop(columns="len")

        df_map["start_node"]=df_map["start_node"].apply(lambda x: ' '.join(str(x. lower()).split()) if isinstance(x, str) else x)
        df_map["edge"]=df_map["edge"].apply(lambda x: ' '.join(str(x. lower()).split()) if isinstance(x, str) else x)
        df_map["end_node"]=df_map["end_node"].apply(lambda x: ' '.join(str(x. lower()).split()) if isinstance(x, str) else x)
        df_map["text"]=df_map["text"].apply(lambda x: ' '.join(str(x).split()) if isinstance(x, str) else x)

        self.graph_df = df_map

    # helper function to retrieve end nodes matching the relevant start node(s)
    def filter_minority_nodes(docstore, cond):
        """TBA
            Parameters
            ----------
            docstore: the docstore to be queried
            cond: list of start nodes
            """ 
        ls_id=[]
        ls_emb=[]
        for doc in docstore:
            if len(docstore[doc]["__data__"]["metadata"])==0: 
                if docstore[doc]["__data__"]["text"].lower() in cond:
                    ls_id.append(doc)
                    ls_emb.append(docstore[doc]["__data__"]["embedding"])
            else:
                met=docstore[doc]["__data__"]["metadata"]
                if (met["start_node"].lower() in cond) or (met["end_node"].lower() in cond):
                    ls_id.append(doc)
                    ls_emb.append(docstore[doc]["__data__"]["embedding"])
        # ls=list(df_map[df_map["start_node"].isin(cond)]["end_node"].unique())
        # for node in nodes:
        #     if node.text in cond:
        #         ls.append(node)
        return ls_id,ls_emb


    def calculate_top_k(self, k):

        minorities=self.graph_df["start_node"].unique()
        adj_nodes={minority: 
            set(self.graph_df[(self.graph_df["start_node"]==minority) & (self.graph_df["end_node"].isin(minorities))]["end_node"]) 
            | 
            set(self.graph_df[(self.graph_df["end_node"]==minority) & (self.graph_df["start_node"].isin(minorities))]["start_node"])
            for minority in tqdm(minorities)}
        # df_bbq['prediction'] = ""
        # df_bbq['explanation'] = ""
        # df_bbq['nodes'] = ""
        # df_bbq['output'] = ""
        progress_bar = tqdm(self.df_bbq.iterrows(), total=self.df_bbq.shape[0],)

        correct_predictions=0
        for i, row in progress_bar:
        #for i, row in tqdm(df_bbq.iterrows(), total=df_bbq.shape[0]):
            query=f"""
                {row["context"]}
                {row['question']}
                Choose one of the following responses (Only output A,B, or C) and the explanation in a JSON format:
                A. {row['ans0']}
                B. {row['ans1']}
                C. {row['ans2']}

                Ouput Format JSON (YOU MUST FOLLOW THE OUTPUT FORMAT, EXTREMELY IMPORTANT): 
                {{"Explanation": [Provide the expanation justifying your selection], "Letter": [A, B, or C. Only output letters]}}"
                """
            
            # Get embedding
            query_embedding = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            ).dict()["data"][0]["embedding"]
            
            # # First Similarity Search
            # ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=start_doc_embeddings,doc_ids=start_doc_ids,similarity_top_k=k)
            # # ls_score=ls_topk_res[0]
            # ls_minority=[start_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            # ls=set(ls_minority)

            # # Sedond Similarity Search
            # # Merging the candidate nodes
            # ls_subgraph_minorities=set().union(*[adj_nodes[x] for x in ls_minority])-set(ls_minority)    
            # # Obtain embeddings of candidate nodes
            # ls_filt_id,ls_filt_emb=filter_minority_nodes(start_docstore["docstore/data"],ls_subgraph_minorities)
            # # Run topk
            # ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=ls_filt_emb,doc_ids=ls_filt_id,similarity_top_k=k)    
            # ls_minority=[start_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            # if len(ls_minority)>0:
            #     ls|=set(ls_minority)

            # # Third Step: Getting final relations
            # ls_filt_id,ls_filt_emb=filter_minority_nodes(text_docstore["docstore/data"],ls)
            # ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=ls_filt_emb,doc_ids=ls_filt_id,similarity_top_k=k)
            # ls_entities=[text_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            # ls_score=ls_topk_res[0]
            # df_bbq.loc[i, f"top_{k}_entities"]=str(ls_entities)
            # df_bbq.loc[i, f"top_{k}_scores"]=str(ls_score)
            # First Similarity Search
            ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=self.start_doc_embeddings,doc_ids=self.start_doc_ids,similarity_top_k=k)
            # ls_score=ls_topk_res[0]
            ls_minority=[self.start_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            ls=set(ls_minority)

            # Sedond Similarity Search
            # Merging the candidate nodes
            ls_subgraph_minorities=set().union(*[adj_nodes[x] for x in ls_minority])-set(ls_minority)    

            if len(ls_subgraph_minorities)>=k:
                ls_filt_id,ls_filt_emb=self.filter_minority_nodes(self.start_docstore["docstore/data"],ls_subgraph_minorities)
                # Run topk
                ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=ls_filt_emb,doc_ids=ls_filt_id,similarity_top_k=k)    
                ls_minority=[self.start_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            else:
                ls_minority=list(ls_subgraph_minorities)
                print(i)
            if len(ls_minority)>0:
                ls|=set(ls_minority)

            # Third Step: Getting final relations
            ls_filt_id,ls_filt_emb=self.filter_minority_nodes(self.text_docstore["docstore/data"],ls)
            
            ls_topk_res=get_top_k_embeddings(query_embedding=query_embedding,doc_embeddings=ls_filt_emb,doc_ids=ls_filt_id,similarity_top_k=k)
            ls_entities=[self.text_docstore["docstore/data"][ls_topk_res[1][x]]["__data__"]["text"] for x in range(k)]
            ls_score=ls_topk_res[0]
            self.df_bbq.loc[i, f"top_{k}_entities"]=str(ls_entities)
            self.df_bbq.loc[i, f"top_{k}_scores"]=str(ls_score)


    def get_top_k(self, k, i):
        return self.df_bbq.loc[i, f"top_{k}_entities"], self.df_bbq.loc[i, f"top_{k}_scores"]
