# genie_agent/agent.py
import mlflow
import json
from mlflow.pyfunc import ChatAgent, ChatAgentMessage, ChatAgentResponse

from databricks.sdk import WorkspaceClient
from databricks.sdk.oauth import ModelServingUserCredentials
from databricks_langchain.genie import GenieAgent as GenieTool
from databricks_langchain.tools import UCFunctionToolkit
from unitycatalog.ai.core import DatabricksFunctionClient
from databricks_langchain.chat_models import ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# Configuration for your Databricks environment
CATALOG = "sabrina"
SCHEMA = "agent"
LLM_ENDPOINT_NAME = "databricks-dbrx-instruct" # Using DBRX Instruct as the agent's brain

# The system prompt that instructs the LLM on how to behave.
SYSTEM_PROMPT = """
You are a helpful assistant that processes a user's query by following a strict workflow.
1. First, take the user's question and use the 'genie' tool to generate a SQL query and a natural language answer.
2. Next, take the generated SQL query and use the 'score_sql_query' tool to calculate its complexity score.
3. Finally, take the complexity score and use the 'review_complexity_score' tool to get a final decision.
4. If the decision is 'approved', present the final answer and the SQL query to the user.
5. If the decision is 'too_complex', inform the user that the query is too complex and that the Biorepo team has been notified, then show them the SQL query that was flagged.
Do not deviate from this workflow.
"""

class GenieAgent(ChatAgent):
    def _initialize_agent(self):
        """
        Initializes the agent's tools and clients.
        """
        user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        function_client = DatabricksFunctionClient()

        # Initialize the Genie tool for SQL generation
        genie_tool = GenieTool(
            genie_space_id="01f0722c73c61f12b1ab8023936c9fda",
            client=user_client,
        )

        # Initialize the toolkit for our custom UC functions
        uc_tools = UCFunctionToolkit(
            client=function_client,
            catalog=CATALOG,
            schema=SCHEMA,
            functions=["score_sql_query", "review_complexity_score"]
        )
        
        # The LLM that will act as the agent's "brain"
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, max_tokens=500)
        
        # Combine all tools for the agent
        tools = [genie_tool] + uc_tools.get_tools()

        # Create the prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create the agent itself
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Create the executor that runs the agent
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def predict(self, history, **kwargs):
        """
        This is the main entry point for the agent.
        It uses an AgentExecutor to decide which tools to call.
        """
        self._initialize_agent()
        
        question = history[-1]["content"]

        # The AgentExecutor runs the full workflow
        response = self.executor.invoke({"input": question})

        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(role="assistant", content=response["output"]),
            ]
        )
