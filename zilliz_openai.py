import csv
import json
import random
import openai
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


# Extract the book titles
def csv_load(file):
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row[1]

