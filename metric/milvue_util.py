from pymilvus import MilvusClient
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

def initCollection(collectionName: str):
    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=256)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(collectionName, schema)

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index("embeddings", index=index)


def getClient():
    return MilvusClient(uri="http://localhost:19530")

def insert(client: MilvusClient, collection: str, data: list):
    client.insert(collection, data)


def topk(client: MilvusClient, collection: str, query: list, limit: int):
    return client.search(collection, anns_field="embeddings", data=query, limit=limit, search_params={"metric_type": "COSINE"})
