# genie_agent/agent.py
import mlflow
import json
from mlflow.pyfunc import ChatAgent, ChatAgentMessage, ChatAgentResponse

from databricks.sdk import WorkspaceClient
from databricks.sdk.oauth import ModelServingUserCredentials
from databricks_langchain.genie import GenieAgent as GenieTool

# --- The following imports are commented out for the simplified, Genie-only agent ---
# from databricks_langchain.tools import UCFunctionToolkit
# from unitycatalog.ai.core import DatabricksFunctionClient
# from databricks_langchain.chat_models import ChatDatabricks
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage

# --- The following constants are not needed for the simplified agent ---
# CATALOG = "sabrina"
# SCHEMA = "agent"
# LLM_ENDPOINT_NAME = "databricks-dbrx-instruct"

class GenieAgent(ChatAgent):
    def _initialize_agent(self):
        """
        Initializes the agent's tools and clients.
        For this simplified version, we only initialize the Genie tool.
        """
        user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())

        # Initialize the Genie tool for SQL generation
        self.genie_tool = GenieTool(
            genie_space_id="01f0722c73c61f12b1ab8023936c9fda",
            client=user_client,
        )

    def predict(self, history, **kwargs):
        """
        This is the main entry point for the agent.
        It directly calls the Genie tool and returns the result.
        """
        self._initialize_agent()
        
        question = history[-1]["content"]

        with mlflow.start_span("1_call_genie") as span:
            # Directly invoke the Genie tool
            genie_output = self.genie_tool.invoke({"question": question})
            
            # The output from GenieTool is a dictionary containing the result and the SQL query
            final_text = genie_output.get("result", "I could not find an answer.")
            sql_query = genie_output.get("sql_query", "")

            span.set_outputs({
                "final_text": final_text,
                "sql_query": sql_query,
            })
            
            response_message = f"""{final_text}

**Generated SQL Query:**
```sql
{sql_query}
```
"""

        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(role="assistant", content=response_message),
            ]
        )
