import uuid
import json
from django.core.management.base import BaseCommand
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from playground.models import LangchainPgEmbedding, Product, Product_ver2
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class Command(BaseCommand):
    help = 'Embed productDes and store in LangchainPgEmbedding table'

    def handle(self, *args, **options):
        try:

            model_path = 'C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/'
            # Initialize the embedding model
            embedding_model = SentenceTransformer(model_path)

            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            model = AutoModel.from_pretrained("C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Fetch all products
            products = Product_ver2.objects.all()

            for product in products:
                try:
                    # Generate embedding
                    input_tokens = tokenizer(product.productName, padding="max_length",
                    truncation=True, max_length=256, return_tensors="pt")
                    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

                    outputs = model(**input_tokens)
                    embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]

                    product.productName_embed = embedding
                    product.save()
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Failed to embed for product {product.pk}: {str(e)}'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error initializing model: {str(e)}'))



