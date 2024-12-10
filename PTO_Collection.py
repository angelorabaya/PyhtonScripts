import pymongo
import pandas as pd

def connect_to_mongo(uri, db_name):
    client = pymongo.MongoClient(uri)
    return client[db_name]

# Function to fetch aop data based on aop_control
def fetch_detail_data(collection, aop_control):
    #collection = connect_to_mongo()

    # Query to find documents with the specified aop_control
    query = {"aop_control": aop_control}

    # Fetch results
    results = collection.find(query)

    # Convert results to a DataFrame
    df = pd.DataFrame(list(results))

    # Drop the MongoDB's `_id` field if desired
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
        df.drop(columns=['aop_control'], inplace=True)
        df.drop(columns=['aop_charge'], inplace=True)
        df.drop(columns=['aop_share'], inplace=True)
    return df

def get_aop_details(collection, aop_control):
    # Query the database for the specific aop_control value
    query = {'aop_control': aop_control}
    result = collection.find_one(query)

    if result:
        # Display the fields related to the aop_control
        total = result.get('aop_total')
        formatted_total = f"{total:,}" if total > 1000 else str(total)

        print(f"Name: {result.get('ph_cname')}")
        print(f"Control: {result.get('aop_control')}")
        print(f"Date: {result.get('aop_date')}")
        print(f"Nature: {result.get('aop_nature')}")
        print(f"Municipality: {result.get('aop_mun')}")
        print(f"Barangay: {result.get('aop_brgy')}")
        print(f"Amount: {formatted_total}")
    else:
        print("No records found!")

if __name__ == "__main__":
    mongo_uri = "mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO"
    db_name = "ENRO"
    collection_name = "Assessment_hdr"
    collection_namedtl = "Assessment_dtl"

    # Establish a connection to the database
    db = connect_to_mongo(mongo_uri, db_name)
    collection = db[collection_name]
    collectiondtl = db[collection_namedtl]

    print("")

    # Input the aop_control value
    aop_control_value = input("Please enter control: ")

    # Get and display AOP details
    get_aop_details(collection, aop_control_value)

    print("")

    # Fetch data and display in tabular format
    data_frame = fetch_detail_data(collectiondtl, aop_control_value)

    if not data_frame.empty:
        print(data_frame.to_string())
    else:
        print("No results found!")