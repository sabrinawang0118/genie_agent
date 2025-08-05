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

# IMPORTANT: Replace these with your target catalog, schema, and model endpoint
CATALOG = "sabrina"
SCHEMA = "agent"
LLM_ENDPOINT_NAME = "databricks-dbrx-instruct"

# The system prompt that instructs the LLM on how to behave.
SYSTEM_PROMPT = """
You are a helpful assistant that processes a user's question following a strict workflow.

Your workflow is as follows:
1.  First, you MUST use the 'genie' tool to generate a SQL query and a user-friendly answer from the user's question.
2.  Next, you MUST use the 'score_sql_query' tool with the generated SQL to get a complexity score.
3.  Then, you MUST use the 'review_complexity_score' tool with the complexity score to get a final decision.
4.  Finally, based on the decision, you will formulate a response to the user.
    - If the decision is 'approved', you MUST present the final answer and the SQL query to the user.
    - If the decision is 'too_complex', you MUST inform the user that their question is too complex and has been logged for review.

You must follow these steps in order for every user question. Do not skip any steps.
"""

class GenieAgent(ChatAgent):
    def _initialize_agent(self):
        """
        Initializes the agent's LLM and tools.
        """
        user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
        function_client = DatabricksFunctionClient()

        # The LLM that will act as the "brain" of the agent
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, temperature=0)

        # Initialize all available tools
        genie_tool = GenieTool(
            genie_space_id="01f0722c73c61f12b1ab8023936c9fda",
            client=user_client,
        )
        uc_function_toolkit = UCFunctionToolkit(
            function_names=[
                f"{CATALOG}.{SCHEMA}.score_sql_query",
                f"{CATALOG}.{SCHEMA}.review_complexity_score",
            ],
            client=function_client,
        )
        
        # Combine all tools into a single list
        all_tools = [genie_tool] + uc_function_toolkit.tools
        
        # Create the LangChain agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{input}")
        ])
        
        agent = create_tool_calling_agent(llm, all_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

        return agent_executor

    @mlflow.trace
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: mlflow.pyfunc.ChatAgent.ChatContext | None = None,
        params: dict[str, any] | None = None,
    ) -> ChatAgentResponse:
        """
        This is the main entry point for the agent.
        """
        agent_executor = self._initialize_agent()
        user_question = messages[-1]["content"]

        # Invoke the agent executor to run the full reasoning loop
        response = agent_executor.invoke({"input": user_question})

        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=response["output"])]
        )
