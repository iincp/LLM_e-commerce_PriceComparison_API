from django.core.management.base import BaseCommand
from playground.models import Product
from playground.models import Product_ver2
import csv
import uuid

class Command(BaseCommand):
    help = 'Load a list of products from a CSV file into the database'

    def add_arguments(self, parser):
        parser.add_argument('csv_files', nargs='+', type=str, help='The CSV file(s) to import.')

    def handle(self, *args, **options):
        for file_path in options['csv_files']:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        sold_units = int(float(row['sold_units'])) if row['sold_units'] else 0
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid data for sold_units: {row['sold_units']}"))
                        continue

                    try:
                        price = float(row['price']) if row['price'] else 0.0
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid data for price: {row['price']}"))
                        continue

                    try:
                        rating = float(row['rating']) if row['rating'] else 0.0
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid data for rating: {row['rating']}"))
                        continue

                    try:
                        no_review = int(row['no_review']) if row['no_review'] else 0
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid data for no_review: {row['no_review']}"))
                        continue

                    product_data = {
                        'productName': row['productName'][:200],  # Truncate to 200 characters
                        'productDes': row['productDes'][:200],  # Truncate to 200 characters
                        'image': row['image'][:200],  # Truncate to 200 characters
                        'price': price,
                        'sold_units': sold_units,
                        'rating': rating,
                        'no_review': no_review,
                        'link': row['link'][:200],  # Truncate to 200 characters
                        'shipmentOrigin': row['shipmentOrigin'][:200],  # Truncate to 200 characters
                        'brand': row['brand'][:200],  # Truncate to 200 characters
                    }

                    product, created = Product_ver2.objects.update_or_create(
                        vector_product_id=uuid.uuid4(),
                        defaults=product_data
                    )

                    if created:
                        self.stdout.write(self.style.SUCCESS(f'Successfully added product {product.productName}'))
                    else:
                        self.stdout.write(self.style.SUCCESS(f'Updated product {product.productName}'))
