# genie_agent/log_agent.py
import mlflow
from agent import GenieAgent

# The name we'll use to register the model in MLflow
MODEL_NAME = "genie_agent"

def main():
    # Ensure the Unity Catalog is the default registry
    mlflow.set_registry_uri("databricks-uc")

    # Create an instance of our agent
    agent = GenieAgent()

    # Log the agent to MLflow. This will package the agent and its dependencies.
    # The 'registered_model_name' will make it available in the Unity Catalog.
    mlflow.pyfunc.log_model(
        "agent",
        python_model=agent,
        registered_model_name=MODEL_NAME,
        # This is important for the ChatAgent interface
        input_example=[{"messages": [{"role": "user", "content": "What are the top 5 products by sales?"}]}],
        # We can simplify the requirements since we are not using the custom tools for now
        pip_requirements=["mlflow", "databricks-sdk", "databricks-langchain"]
    )

    print(f"Agent logged and registered as '{MODEL_NAME}' in Unity Catalog.")

if __name__ == "__main__":
    main()
