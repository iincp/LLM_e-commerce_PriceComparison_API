from django.shortcuts import render
from django.http import HttpResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from .models import *
import numpy as np
from django.db import connection
import torch 
import pandas as pd 
from django.db.models import Avg
from simpletransformers.ner import NERModel
from .word_dict import MyDictionary
import re
from pythainlp.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel

# Create your views here.
def search_prompt(request): 
    return render(request,"form.html")

def concatenate_keys_with_value_3(data,val):
    result = []

    for sublist in data:
        concatenated_string = ""
        for dictionary in sublist:
            for key, value in dictionary.items():
                if value == val:  # Checking if the value is 3
                    concatenated_string += key
        result.append(concatenated_string)

    return result

def extract_integer_from_list(lst):
    items = []
    for item in lst:
        if isinstance(item, int):  # Check if the item is already an integer
            items.append(item)
        elif isinstance(item, str):
            # Find all numbers in the string and convert them to integers
            found_numbers = re.findall(r'\d+', item)
            for number in found_numbers:
                items.append(int(number))
    return items  # Return the list of extracted integers

def fetch_data(query, condition = None):
    with connection.cursor() as cursor:
        cursor.execute(query,condition)
        rows = cursor.fetchall()
    return rows

def remove_space(input_string): 
    no_space_string = input_string.replace(" ", "")
    return no_space_string

def tokenize_and_predict(prompt, model, tokenizer):
    prompt_tokenize = word_tokenize(prompt, engine='newmm')
    join_tokenize = " ".join(prompt_tokenize)
    predictions, raw_outputs = model.predict([join_tokenize])
    return predictions

def get_embedding(text, tokenizer, model, device):
    input_tokens = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
    outputs = model(**input_tokens)
    embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
    return embedding

def fetch_cosine_similarity_results(sql, embed_list_str, ids):
    return fetch_data(sql, [embed_list_str, ids])

def form(request):
    if request.method != "POST":
        return HttpResponse("Invalid request method.")
    
    prompt = remove_space(request.POST["prompt"])
    
    # Initialize models and tokenizer
    model = NERModel("camembert", "C:/Users/natch/Desktop/Me/Hackatorn/model/Best", use_cuda=False)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_model_path = 'C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/'
    sentence_transformer_model = SentenceTransformer(embedding_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sentence_transformer_model.to(device)

    # Predictions and embeddings
    predictions = tokenize_and_predict(prompt, model, tokenizer)
    about_product_embed = get_embedding(''.join(concatenate_keys_with_value_3(predictions, '0')), tokenizer, sentence_transformer_model, device)

    # Database operations
    average_price = Product_ver2.objects.aggregate(Avg('price'))['price__avg']
    price_in_list = extract_integer_from_list(concatenate_keys_with_value_3(predictions, '3'))
    about_brand = concatenate_keys_with_value_3(predictions, '2')

    product_uuids, state = process_price_and_brand(price_in_list, about_brand, average_price)  # Needs implementation
    
    # Convert embedding to string for SQL query
    about_product_embed_list_str = '[' + ', '.join(map(str, about_product_embed)) + ']'
    
    # SQL query execution
    sql_query_productName = """
            WITH Cosine_Similarity_Calc AS (
                SELECT "vector_product_id",
                    ("productName_embed" <=> CAST(%s AS vector)) AS cosine_similarity
                FROM playground_product_ver2
                WHERE vector_product_id = ANY(%s)
            )
            SELECT vector_product_id, cosine_similarity
            FROM Cosine_Similarity_Calc
            ORDER BY cosine_similarity ASC
            LIMIT 100; """ 
    
    top_100_results = fetch_cosine_similarity_results(sql_query_productName, about_product_embed_list_str, product_uuids)
    top_100_ids = [result[0] for result in top_100_results]

    prompt_embed = get_embedding(prompt, tokenizer, sentence_transformer_model, device)
    prompt_embed_list_str = '[' + ', '.join(map(str, prompt_embed)) + ']'

    sql_query_productDes ="""
            WITH Cosine_Similarity_Calc AS (
                SELECT "product_id",
                    (embedding <=> CAST(%s AS vector)) AS cosine_similarity
                FROM playground_langchainpgembedding_ver2
                WHERE product_id = ANY(%s)
            ),
            Median_Calc AS (
                SELECT "product_id",
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cosine_similarity) AS median_cosine_similarity
                FROM Cosine_Similarity_Calc
                GROUP BY "product_id"
            )
            SELECT "product_id", median_cosine_similarity
            FROM Median_Calc
            ORDER BY median_cosine_similarity ASC
            LIMIT 5; """  
    
    top_5_results = fetch_cosine_similarity_results(sql_query_productDes, prompt_embed_list_str, top_100_ids)
    top_5_ids = [result[0] for result in top_5_results]

    # Get products
    products = Product_ver2.objects.filter(vector_product_id__in=top_5_ids)
    print(prompt_embed)
    print(prompt)
    print(top_5_ids)
    print(product_uuids)
    print('------------------')
    print(top_100_ids)
    print('-----------------' ) 
    print(price_in_list) 
    print(state)
    print(predictions)
    print(about_brand)
    print(prompt)

    return render(request, "result.html", {'prompt': prompt, 'top_5_distances': [result[1] for result in top_5_results], 'top_5_ids': top_5_ids, 'products': products})

# Example of additional helper functions
def process_price_and_brand(price_in_list, about_brand, average_price):
    state = ''
    product_uuids = []
    my_dict_instance = MyDictionary()

    if price_in_list:
        if my_dict_instance.is_cheaper(price_in_list) and len(price_in_list) == 1:
            filter_kwargs = {'price__lt': int(price_in_list[0])}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'cheaper'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)
            state = 'cheaper'

        elif my_dict_instance.is_expensive(price_in_list) and len(price_in_list) == 1:
            filter_kwargs = {'price__gt': int(price_in_list[0])}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'expensive'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)
            state = 'expensive'

        elif len(price_in_list) == 2:
            filter_kwargs = {'price__lt': int(max(price_in_list)), 'price__gt': int(min(price_in_list))}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'price between'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)
            state = 'price between'

        else:
            filter_kwargs = {}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)
                state = 'one for all'
            else:
                product_uuids = Product_ver2.objects.all().values_list('vector_product_id', flat=True)
                state = 'one for all'

    else:
        if my_dict_instance.is_cheaper(price_in_list):
            filter_kwargs = {'price__lt': int(average_price)}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'no price cheaper'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)

        elif my_dict_instance.is_expensive(price_in_list):
            filter_kwargs = {'price__gt': int(average_price)}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'no price expensive'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)

        elif my_dict_instance.is_expensive(price_in_list) and my_dict_instance.is_cheaper(price_in_list):
            filter_kwargs = {'price__lt': int(max(average_price - (average_price * 0.2))), 'price__gt': int(average_price + (average_price * 0.2))}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                state = 'no price price between'
            product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)

        else:
            filter_kwargs = {}
            if about_brand[0]:
                filter_kwargs['brand'] = about_brand[0]
                product_uuids = Product_ver2.objects.filter(**filter_kwargs).values_list('vector_product_id', flat=True)
                state = 'one for all'
            else:
                product_uuids = Product_ver2.objects.all().values_list('vector_product_id', flat=True)
                state = 'one for all'

    return product_uuids, state

