import pymongo
from datetime import datetime
import pandas as pd

# MongoDB connection setup
client = pymongo.MongoClient("mongodb+srv://jonathancruz2011:iamgod3875@clusterenro.qb6n0yq.mongodb.net/?retryWrites=true&w=majority&appName=ClusterENRO")  # Adjust connection string if needed
db = client["ENRO"]
collection = db["Assessment_hdr"]

# Get user input for the search string and field name
search_string = "11/06/2024"
field_name = "aop_ordate"

# Construct the MongoDB query using a regular expression
query = {
    field_name: { "$regex": search_string, "$options": "i" }
}

# Execute the query and retrieve results
results = collection.find(query)

# Process and display results
#for result in results:
#    print(result)

# Process and display results using Pandas
df = pd.DataFrame(list(results))  # Convert results to a DataFrame

# Set Pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Save results to a text file
if not df.empty:
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))  # Save DataFrame to file without index
    print("Results saved to result.txt")
else:
    print("No results found.")

# Close the MongoDB connection (good practice)
client.close()