# genie_agent/create_tools.py
from unitycatalog.ai.core import DatabricksFunctionClient
from tools import score_sql_query, review_complexity_score

# IMPORTANT: Replace these with your target catalog and schema
CATALOG = "sabrina"
SCHEMA = "agent"

def main():
    client = DatabricksFunctionClient()

    print("Registering `score_sql_query` function...")
    client.create_python_function(
        func=score_sql_query,
        catalog=CATALOG,
        schema=SCHEMA,
        replace=True
    )
    print("`score_sql_query` registered successfully.")

    print("Registering `review_complexity_score` function...")
    client.create_python_function(
        func=review_complexity_score,
        catalog=CATALOG,
        schema=SCHEMA,
        replace=True
    )
    print("`review_complexity_score` registered successfully.")
    print("\nAll tools have been registered in Unity Catalog.")

if __name__ == "__main__":
    main()
