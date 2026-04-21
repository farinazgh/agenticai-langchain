from dotenv import load_dotenv
from langchain.tools import tool


from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: the query to search over
    Returns:
        the search result
    """
    print(f"searching {query}")
    return "Tokyo weather is sunny"


llm = ChatOpenAI(model="gpt-4o")
tools = [search]

agent = create_agent(
    model=llm,
    tools=tools,
)


def main():
    print("hello!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo")]}
    )
    print(result)


if __name__ == "__main__":
    main()
