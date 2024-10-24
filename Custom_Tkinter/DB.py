import pymongo

def get_collection(colname):
    try:
        client = pymongo.MongoClient("mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO")
        db = client["ENRO"]
        collection = db[colname]
        return collection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_data(pcol,pquery):
    collection = get_collection(pcol)
    if collection is None:
        return []
    try:
        result = collection.find_one(pquery)
    except Exception as e:
        print(f"Error fetching data: {e}")
        result = []
    return result

def fetch_data_many(pcol, pquery):
    collection = get_collection(pcol)
    if collection is None:
        return []
    try:
        results = collection.find(pquery)
        return list(results)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []