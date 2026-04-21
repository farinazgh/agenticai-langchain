from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    user_prompt = "search for 3 casual dinner restaurants in Le Marais, Paris with rating above 4.5 and list their details"

    #  INPUT
    print("\n===  STEP 1: USER INPUT ===")
    print(user_prompt)

    # MODEL INFO
    print("\n===  STEP 2: MODEL CONFIG ===")
    print(f"Model: {llm.model_name}")

    # TOOLS AVAILABLE
    print("\n===  STEP 3: AVAILABLE TOOLS ===")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    print("\n===  STEP 4: RUNNING AGENT ===")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        }
    )

    # RAW RESULT
    print("\n===  STEP 5: RAW RESULT ===")
    print(result)

    # WHAT KEYS EXIST
    print("\n===  STEP 6: RESULT KEYS ===")
    print(result.keys())

    # STRUCTURED RESPONSE
    structured = result.get("structured_response", None)

    print("\n=== STEP 7: STRUCTURED RESPONSE ===")

    if structured:
        print(" Parsed Answer:")
        print(structured.answer)

        print("\n Sources:")
        for s in structured.sources:
            print(f"- {s.url}")
    else:
        print("⚠️ No structured response found, fallback to raw:")
        print(result)


if __name__ == "__main__":
    main()

# {
#    "query":"casual dinner restaurants Le Marais Paris rating above 4.5",
#    "include_domains":"None",
#    "exclude_domains":"None",
#    "search_depth":"advanced",
#    "include_images":"False",
#    "time_range":"None",
#    "topic":"general",
#    "start_date":"None",
#    "end_date":"None"
# }
