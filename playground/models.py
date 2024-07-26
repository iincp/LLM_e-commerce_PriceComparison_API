import uuid
from django.db import models
from pgvector.django import VectorField

def default_vector():
    return [0]*384  # This creates a list of 768 zeros

class Product(models.Model): 
    vector_product_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    productName = models.CharField(max_length=255)
    productDes = models.TextField()
    image = models.URLField()
    price = models.FloatField()
    sold_units = models.IntegerField()
    rating = models.FloatField()
    no_review = models.IntegerField()
    link = models.URLField() 
    shipmentOrigin = models.CharField(max_length=255)
    brand = models.CharField(max_length=255)
    productDes_embed = VectorField(dimensions=384,default=default_vector)

class LangchainPgEmbedding(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    product = models.ForeignKey(Product, models.DO_NOTHING, blank=True, null=True)
    embedding = VectorField(dimensions=384)
    document = models.CharField(blank=True, null=True, max_length=1000)
    cmetadata = models.JSONField(blank=True, null=True)

class Product_ver2(models.Model): 
    vector_product_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    productName = models.CharField(max_length=255)
    productDes = models.TextField()
    image = models.URLField()
    price = models.FloatField()
    sold_units = models.IntegerField()
    rating = models.FloatField()
    no_review = models.IntegerField()
    link = models.URLField() 
    shipmentOrigin = models.CharField(max_length=255)
    brand = models.CharField(max_length=255)
    productName_embed = VectorField(dimensions=384,default=default_vector)

class LangchainPgEmbedding_ver2(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    product = models.ForeignKey(Product_ver2, models.DO_NOTHING, blank=True, null=True)
    embedding = VectorField(dimensions=384)
    document = models.CharField(blank=True, null=True, max_length=1000)
    cmetadata = models.JSONField(blank=True, null=True)
