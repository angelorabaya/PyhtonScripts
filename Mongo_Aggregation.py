import pymongo
from datetime import datetime
from pymongo.errors import ConnectionFailure, OperationFailure

CONNECTION_STRING = "mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO"
DB_NAME = "ENRO"
COLLECTION_NAME = "Assessment_hdr"


def get_mongodb_collection():
    try:
        client = pymongo.MongoClient(CONNECTION_STRING)
        return client[DB_NAME][COLLECTION_NAME]
    except ConnectionFailure as e:
        raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")


def query_documents(target_date):
    pipeline = [
        {
            "$match": {
                "aop_ordate": {
                    "$gte": datetime.combine(target_date, datetime.min.time()),
                    "$lt": datetime.combine(target_date, datetime.max.time())
                }
            }
        }
    ]

    try:
        collection = get_mongodb_collection()
        return list(collection.aggregate(pipeline))
    except OperationFailure as e:
        print(f"Query operation failed: {e}")
        return []


def main():
    try:
        target_date = datetime(2024, 11, 5)
        results = query_documents(target_date)

        if not results:
            print("No documents found for the specified date.")
            return

        for doc in results:
            print(doc)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()