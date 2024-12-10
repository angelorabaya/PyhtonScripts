import pymongo
import pyodbc
#from datetime import datetime
from dateutil import parser
#import arrow

# Database connection settings
MONGO_URI = "mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO"
SQL_SERVER_CONNECTION_STRING = "Driver={ODBC Driver 17 for SQL Server};Server=DESKTOP-5NMUA8K;Database=ENRODB;UID=sa;PWD=iamgod3875;"

# Connect to MongoDB
mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_db = mongo_client["ENRO"]
mongo_collection = mongo_db["Assessment_hdr"]

# Function to handle change events
def handle_change(change):
    if change["operationType"] == "update":
        document_id = change["documentKey"]["_id"]
        updated_document = mongo_collection.find_one({"_id": document_id})
        print(updated_document["aop_orno"])
        print(updated_document["ph_cname"])
        print(updated_document["aop_control"])

        date_string = updated_document['aop_ordate'] 
        #print(date_string)
        #date_format = "%d/%m/%Y %I:%M:%S %p"
        #date_object = datetime.strptime(date_string, date_format)
        date_object = parser.parse(date_string)
        #date_object = arrow.get(date_string)
        output_format = "%Y-%m-%d"
        formatted_date = date_object.strftime(output_format)

        sql_server_conn = pyodbc.connect(SQL_SERVER_CONNECTION_STRING)
        sql_server_cursor = sql_server_conn.cursor()
        sql_update = "UPDATE tbl_assessmenthdr SET aop_orno = ?, aop_ordate = ? WHERE aop_control = ?"
        sql_server_cursor.execute(sql_update, (updated_document['aop_orno'], formatted_date, updated_document["aop_control"]))
        sql_server_conn.commit()
        sql_server_conn.close()
        print("SQL Server Updated!")

# Watch for changes in the collection
change_stream = mongo_collection.watch()

try:
    for change in change_stream:
        handle_change(change)
except KeyboardInterrupt:
    print("Stopped watching for changes.")
finally:
    mongo_client.close()
    