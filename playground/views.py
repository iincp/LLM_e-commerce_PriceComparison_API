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

def from_prediction(data,val):
    concate_string_result = []
    list_of_key = []

    for sublist in data:
        concatenated_string = ""
        for dictionary in sublist:
            for key, value in dictionary.items():
                if value == val:  # Checking if the value is 3
                    concatenated_string += key
                    list_of_key.append(key)
        concate_string_result.append(concatenated_string.replace(" ",""))

    return concate_string_result, list_of_key

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

def embed_process(prompt): 
    input_tokens = tokenizer(prompt, padding="max_length",
                         truncation=True, max_length=256, return_tensors="pt")
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

    outputs = model(**input_tokens)
    prompt_embed = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
    prompt_embed_list_str = '[' + ', '.join(map(str, prompt_embed)) + ']'

    return prompt_embed_list_str

def get_product_uuids(price_in_list, my_dict_instance, about_price, about_brand, average_price):
    """Main function to get product_uuids based on different conditions."""

    def update_product_uuids(price_condition, price_value=None):
        """Helper function to update product_uuids based on given conditions."""
        query_kwargs = {'brand': about_brand[0]} if about_brand and about_brand[0] else {}
        if price_condition == 'cheaper':
            query_kwargs['price__lt'] = int(price_value)
        elif price_condition == 'expensive':
            query_kwargs['price__gt'] = int(price_value)
        return Product_ver2.objects.filter(**query_kwargs).values_list('vector_product_id', flat=True)

    def determine_price_bounds():
        """Determine the appropriate query based on the number of prices in list and price conditions."""
        if len(price_in_list) == 1:
            price_cond = 'cheaper' if my_dict_instance.is_cheaper(about_price) else 'expensive'
            return update_product_uuids(price_cond, price_in_list[0])
        elif len(price_in_list) == 2:
            product_uuids = Product_ver2.objects.filter(
                price__lt=int(max(price_in_list)),
                price__gt=int(min(price_in_list)),
                **({'brand': about_brand[0]} if about_brand and about_brand[0] else {})
            ).values_list('vector_product_id', flat=True)
            return product_uuids
        else:
            return update_product_uuids(None)  # No specific price condition is applied here

    # Main logic of get_product_uuids
    if len(price_in_list) > 0:
        return determine_price_bounds()
    else:

        if my_dict_instance.is_cheaper(about_price):
            average_condition = 'cheaper'
            average_price_adjusted = average_price * 0.8
        elif my_dict_instance.is_expensive(about_price):
            average_condition = 'expensive'
            average_price_adjusted = average_price * 1.2
        else:
            return Product_ver2.objects.all().values_list('vector_product_id', flat=True)  # Default case when no condition matches
        
        return update_product_uuids(average_condition, average_price_adjusted)
#------------------------------LOAD MODEL --------------------------------#

# Provide the path to the directory containing your saved model
MAIN_PATH = 'C:/Users/natch/Desktop/Me/Hackatorn/model/'
EMBED_PATH = MAIN_PATH + 'embed/Embedding/'
NER_PATH = MAIN_PATH + '/Best'


#catagirize the price 
NER_model = NERModel(
    "camembert",NER_PATH, use_cuda=False
)

embedding_model = SentenceTransformer(EMBED_PATH)
# Load the WangchanBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = AutoModel.from_pretrained(EMBED_PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#-----------------------------------------------------------------------#

def form(request): 
    if request.method == "POST": 
        prompt = request.POST["prompt"].replace(" ", "") 
    #NER ON propmt
        #['word_1', 'word_2', 'word_3']#
        tokenize_prompt =  word_tokenize(prompt, engine='newmm')

        #'word_1 word_2 word_3'
        str_token_prompt = " ".join(tokenize_prompt)
        
        NER_prompt_prediction, raw_ouputs = NER_model.predict([str_token_prompt]) 

        words_about_price, _ = from_prediction(NER_prompt_prediction,'3') 
        price = extract_integer_from_list(words_about_price) 

        words_about_brand, _ = from_prediction(NER_prompt_prediction,'2') 

        words_about_product, list_product = from_prediction(NER_prompt_prediction,'0')
        str_about_product = " ".join(words_about_product) 

    #filter by price
        my_dict = MyDictionary()
        # Calculate the average price
        average_price = Product_ver2.objects.aggregate(Avg('price'))['price__avg']
        #productID filterd by price
        product_uuids = get_product_uuids(price, my_dict,words_about_price, words_about_brand, average_price)
        # Convert QuerySet to list of UUIDs
        product_uuids = list(product_uuids)

    #filter by productName 

        #elastic search 

        elastic_search_query = """
            SELECT "vector_product_id" 
            FROM playground_product_ver2
            WHERE REPLACE("productName", '_', '') LIKE %s AND "vector_product_id" = ANY(%s)
        """
        product_name = '%' + '%'.join(list_product) + '%'

        filtered_by_elastic = fetch_data(elastic_search_query,[product_name,product_uuids])

            #filterd by elastice search
        filtered_by_elastic_ids = [id[0] for id in filtered_by_elastic] 

        #similarity search on productName
        word_product_embed_list_str = embed_process(words_about_product)

        similarity_productName_query = """
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
    
        top_100_results = fetch_data(similarity_productName_query,[word_product_embed_list_str,filtered_by_elastic_ids])
        # top_100_distances = [result[1] for result in top_100_results]
        top_100_ids = [result[0] for result in top_100_results]

    #filter by productDes (using similarity search) 
        prompt_embed_list_str = embed_process(prompt)

        #similarity serach and median chunk measurement 
        productDes_query = """
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
        top_5_results = fetch_data(productDes_query,[prompt_embed_list_str,top_100_ids])

        # Extract the distances and corresponding IDs
        top_5_distances = [result[1] for result in top_5_results]
        top_5_ids = [result[0] for result in top_5_results]

        # Query the database using the UUIDs
        products = Product_ver2.objects.filter(vector_product_id__in=top_5_ids)

        #-------------debugging-------------#

        print(prompt)
        print(product_name)
        # print(filtered_by_elastic_ids) 
        # print(top_100_ids) 
        print(top_5_ids)
        print(tokenize_prompt)
        print('-----------------') 
        

        return render(request, "result.html", {'prompt': prompt, 'top_5_distances': top_5_distances, 'top_5_ids': top_5_ids,'products': products})
