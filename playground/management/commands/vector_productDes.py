# import uuid
# import json
# from django.core.management.base import BaseCommand
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from playground.models import LangchainPgEmbedding, Product

# class Command(BaseCommand):
#     help = 'Embed productDes and store in LangchainPgEmbedding table'



#     def handle(self, *args, **options):
#         # Initialize the embedding model
#         embeddings = HuggingFaceEmbeddings(model_name="airesearch/wangchanberta-base-att-spm-uncased")

#         # Fetch all products
#         products = Product.objects.all()

#         # Text splitter for large descriptions
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=3)

#         for product in products:
#             documents = [Document(page_content=product.productDes)]
#             texts = text_splitter.split_documents(documents)

#             # Generate embeddings
#             text_embeddings = [embeddings.embed_documents([document.page_content]) for document in texts]

#             # Store embeddings and link to products
#             for text, embedding in zip(texts, text_embeddings):
#                 embedding_entry = LangchainPgEmbedding(
#                     id=str(uuid.uuid4()),  # Generate a unique ID for each embedding chunk
#                     product=product,  # Link to the product
#                     embedding=embedding[0],  # The generated embedding
#                     document=text.page_content,  # The chunked text
#                     cmetadata=json.dumps({'source': 'productDes'})  # Ensure valid JSON string
#                 )
#                 embedding_entry.save()

from pythainlp.tokenize import word_tokenize
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TextSplitter
from django.core.management.base import BaseCommand
from langchain_huggingface import HuggingFaceEmbeddings
from playground.models import Product, LangchainPgEmbedding, Product_ver2, LangchainPgEmbedding_ver2
import uuid
import json
from transformers import AutoTokenizer, AutoModel
import torch

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

class RecursiveWordTextSplitter(TextSplitter):
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        words = word_tokenize(text, engine='newmm') 
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk))
        return chunks

    def split_documents(self, documents):
        all_chunks = []
        for document in documents:
            chunks = self.split_text(document.page_content)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk))
        return all_chunks

class Command(BaseCommand):
    def handle(self, *args, **options):
        # Load the SentenceTransformer model
        model_path = 'C:/Users/natch/Desktop/Me/Hackatorn/model/embed/Embedding/'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Device configuration
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Fetch all products
        products = Product_ver2.objects.all()
        
        # Text splitter for large descriptions
        text_splitter = RecursiveWordTextSplitter(chunk_size=30, chunk_overlap=3)

        for product in products:
            documents = [Document(page_content=product.productDes)]
            texts = text_splitter.split_documents(documents)
            
            # Generate embeddings
            text_embeddings = []
            for document in texts:
                input_tokens = tokenizer(document.page_content, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
                input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
                with torch.no_grad():
                    outputs = model(**input_tokens)
                about_product_embed = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
                text_embeddings.append(about_product_embed)
            
            # Store embeddings and link to products
            for text, embedding in zip(texts, text_embeddings):
                embedding_entry = LangchainPgEmbedding_ver2(
                    id=str(uuid.uuid4()),  # Generate a unique ID for each embedding chunk
                    product=product,
                    embedding=embedding,
                    document=text.page_content,
                    cmetadata=json.dumps({'source': 'productDes'})
                )
                embedding_entry.save()


