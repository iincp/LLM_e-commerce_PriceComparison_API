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
        result.append(concatenated_string.replace(" ",""))

    return result

def list_productName(data,val): 
    productName_entity = []
    for sublist in data:
        for dictionary in sublist:
            for key, value in dictionary.items():
                if value == val:  # Checking if the value is 3
                    productName_entity.append(key)
    return productName_entity

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

def form(request) : 
    if request.method == "POST": 
        prompt = request.POST["prompt"] 

        prompt= remove_space(prompt) 

        #catagirize the price 
        model = NERModel(
    "camembert", "C:/Users/natch/Desktop/Me/Hackatorn/model/Best", use_cuda=False
)
        prompt_tokenize = word_tokenize(prompt, engine='newmm')
        print(prompt_tokenize)
        join_tokenize = " ".join(prompt_tokenize)
        
        predictions, raw_outputs = model.predict([join_tokenize])

        about_price = concatenate_keys_with_value_3(predictions,'3') 

        about_brand = concatenate_keys_with_value_3(predictions,'2')

        about_product = ''.join(concatenate_keys_with_value_3(predictions,'0'))

        list_product = list_productName(predictions,'0')

        # about_product = ''.join(list_product)

        price_in_list = extract_integer_from_list(about_price)

        my_dict_intstace = MyDictionary()

        state = ''

        # Calculate the average price
        average_price = Product_ver2.objects.aggregate(Avg('price'))['price__avg']

        print(f'predictions : {predictions}')

        if (len(price_in_list ) > 0): 
            if my_dict_intstace.is_cheaper(about_price) & (len(price_in_list) == 1): 
                if about_brand[0] != '':
                    product_uuids = Product_ver2.objects.filter(price__lt=int(price_in_list[0]), brand=about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'cheaper'
                else : 
                    product_uuids = Product_ver2.objects.filter(price__lt=int(price_in_list[0])).values_list('vector_product_id', flat=True)
                    state = 'cheaper'

            elif my_dict_intstace.is_expensive(about_price) & (len(price_in_list) == 1):
                if about_brand[0] != '':
                    product_uuids = Product_ver2.objects.filter(price__gt=int(price_in_list[0]), brand=about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'expensive'
                else :
                    product_uuids = Product_ver2.objects.filter(price__gt=int(price_in_list[0])).values_list('vector_product_id', flat=True)
                    state = 'expensive'

            elif (len(price_in_list) == 2): 
                if about_brand[0] != '':
                    product_uuids = Product_ver2.objects.filter(price__lt=int(max(price_in_list)),price__gt=int(min(price_in_list)),brand = about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'price between'
                else : 
                    product_uuids = Product_ver2.objects.filter(price__lt=int(max(price_in_list)),price__gt=int(min(price_in_list))).values_list('vector_product_id', flat=True)
                    state = 'price between'

            else:
                if about_brand[0] != '':     
                    product_uuids = Product_ver2.objects.filter(brand = about_brand[0]).values_list('vector_product_id', flat=True) 
                    state = 'one for all'
                else : 
                    product_uuids = Product_ver2.objects.all().values_list('vector_product_id', flat=True) 
                    state = 'one for all'

        else:
            if my_dict_intstace.is_cheaper(about_price):
                if about_brand[0] != '' : 
                    product_uuids = Product_ver2.objects.filter(price__lt=int(average_price),brand = about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'no price cheaper'
                else:
                    product_uuids = Product_ver2.objects.filter(price__lt=int(average_price)).values_list('vector_product_id', flat=True)
            elif my_dict_intstace.is_expensive(about_price):
                if about_brand[0] != '' :
                    product_uuids = Product_ver2.objects.filter(price__gt=int(average_price),brand = about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'no price expensive'
                else : 
                    product_uuids = Product_ver2.objects.filter(price__gt=int(average_price)).values_list('vector_product_id', flat=True)
                    state = 'no price expensive'
            elif (my_dict_intstace.is_expensive(about_price) and my_dict_intstace.is_cheaper(about_price)): 
                if about_brand[0] != '' :
                    product_uuids = Product_ver2.objects.filter(price__lt=int(max(average_price - (average_price * 0.2))),price__gt=int(average_price + (average_price * 0.2)),brand = about_brand[0]).values_list('vector_product_id', flat=True)
                    state = 'no price price between'
                else : 
                    product_uuids = Product_ver2.objects.filter(price__lt=int(max(average_price - (average_price * 0.2))),price__gt=int(average_price + (average_price * 0.2))).values_list('vector_product_id', flat=True)
                    state = 'no price price between'

            else :   
                if about_brand[0] != '' :     
                    product_uuids = Product_ver2.objects.filter(brand =about_brand[0]).values_list('vector_product_id', flat=True) 
                    state = 'one for all'
                else : 
                    product_uuids = Product_ver2.objects.all().values_list('vector_product_id', flat=True) 
                    state = 'one for all'

    
 
        
        # Convert QuerySet to list of UUIDs
        product_uuids = list(product_uuids)

        #embed
        # embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

        # Provide the path to the directory containing your saved model
        model_path = 'C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/'

        embedding_model = SentenceTransformer(model_path)

        # Load the WangchanBERTa model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        model = AutoModel.from_pretrained("C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        input_tokens = tokenizer(about_product, padding="max_length",
                         truncation=True, max_length=256, return_tensors="pt")
        input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

        outputs = model(**input_tokens)
        about_product_embed = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]

        print(about_product_embed)

        # about_product_embed = np.array(embedding_model.encode(about_product))
        # about_product_embed_list = about_product_embed.tolist()
        about_product_embed_list_str = '[' + ', '.join(map(str, about_product_embed)) + ']'

        print('-------') 
        print(about_product_embed_list_str)

        sql_query_productName1 = """
            SELECT "vector_product_id" 
            FROM playground_product_ver2
            WHERE REPLACE("productName", '_', '') LIKE %s AND "vector_product_id" = ANY(%s)
        """
        product_name_like = '%' + '%'.join(list_product) + '%'

        filtered_by_name = fetch_data(sql_query_productName1,[product_name_like,product_uuids])
        filtered_by_name_ids = [id[0] for id in filtered_by_name] 

        
        sql_query_productName2 = """
            WITH Cosine_Similarity_Calc AS (
                SELECT "vector_product_id",
                    ("productName_embed" <=> CAST(%s AS vector)) AS cosine_similarity
                FROM playground_product_ver2
                WHERE vector_product_id = ANY(%s)
            )
            SELECT vector_product_id, cosine_similarity
            FROM Cosine_Similarity_Calc
            ORDER BY cosine_similarity ASC
            LIMIT 100;  

        """

        top_100_results = fetch_data(sql_query_productName2,[about_product_embed_list_str,filtered_by_name_ids])
        # Extract the distances and corresponding IDs
        top_100_distances = [result[1] for result in top_100_results]
        top_100_ids = [result[0] for result in top_100_results]
       
       
       
        input_tokens = tokenizer(prompt, padding="max_length",
                         truncation=True, max_length=256, return_tensors="pt")
        input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

        outputs = model(**input_tokens)
        prompt_embed = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
        # prompt_embed = np.array(embedding_model.encode(prompt))
        # Convert the prompt embedding to a list
        # prompt_embed_list = prompt_embed.tolist()
        # Convert the list to a string that PostgreSQL can understand
        prompt_embed_list_str = '[' + ', '.join(map(str, prompt_embed)) + ']'
       
       
       
        # Prepare the SQL query to find the top 5 closest vectors using L2 distance

        #vector_product_id
        #embedding
        sql_query_productDes = """
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
            LIMIT 5;
        """

        top_5_results = fetch_data(sql_query_productDes,[prompt_embed_list_str,top_100_ids])

        # Extract the distances and corresponding IDs
        top_5_distances = [result[1] for result in top_5_results]
        top_5_ids = [result[0] for result in top_5_results]

        # Query the database using the UUIDs
        products = Product_ver2.objects.filter(vector_product_id__in=top_5_ids)

        print(prompt_embed)
        print(prompt)
        print(top_5_distances)  # Print top 5 distances for debugging
        print(top_5_ids)
        print(product_uuids)
        print('------------------')
        print(top_100_ids)
        print('-----------------' ) 
        print(price_in_list) 
        print(about_price) 
        print(state)
        print(my_dict_intstace.is_cheaper(about_price))
        print(predictions)
        print(about_brand)
        print(list_product)
        print(about_product)
        print(prompt)
        print(filtered_by_name_ids)
        print(product_name_like)
        

        return render(request, "result.html", {'prompt': prompt, 'top_5_distances': top_5_distances, 'top_5_ids': top_5_ids,'products': products})
    else:
        print("gg")
