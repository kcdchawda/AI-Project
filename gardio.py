import gradio as gr

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# TOOLS
# --------------------------------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


@tool
def get_flight_prices(departure: str, arrival: str) -> str:
    """Get available mock flight prices between two cities."""

    mock_flights = {
        ("Kolkata", "Delhi"): [
            {
                "airline": "IndiGo",
                "price": 5200,
                "duration": "2h 15m"
            },
            {
                "airline": "Air India",
                "price": 6100,
                "duration": "2h 30m"
            }
        ],

        ("Kolkata", "Mumbai"): [
            {
                "airline": "Vistara",
                "price": 800,
                "duration": "3h 05m"
            },
            {
                "airline": "Akasa Air",
                "price": 600,
                "duration": "3h 10m"
            },
            {
                "airline": "Paul Air",
                "price": 600,
                "duration": "3h"
            }
        ],

        ("Delhi", "Bangalore"): [
            {
                "airline": "IndiGo",
                "price": 6400,
                "duration": "2h 45m"
            }
        ]
    }

    flights = mock_flights.get((departure, arrival))

    if not flights:
        return (
            f"No mock flight data found for "
            f"{departure} → {arrival}"
        )

    result = []

    for flight in flights:
        result.append(
            f"""
Airline: {flight['airline']}
Price: ₹{flight['price']}
Duration: {flight['duration']}
"""
        )

    return "\n".join(result)


# --------------------------------------------------
# OLLAMA
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.1",
    base_url="http://127.0.0.1:11434",
    temperature=0
)


# --------------------------------------------------
# AGENT
# --------------------------------------------------

agent = create_agent(
    model=llm,

    # Include both tools if you want arithmetic too
    tools=[
        get_flight_prices,
        add_numbers
    ],

    system_prompt="""
You are a helpful AI travel assistant.

Rules:
- If the user asks about flights or ticket prices,
  ALWAYS use the get_flight_prices tool.
- If the user asks you to add numbers,
  use the add_numbers tool.
- Present flight information clearly.
- If multiple flights are available, compare them.
- When appropriate, mention the cheapest option.
"""
)


# --------------------------------------------------
# GRADIO -> LANGCHAIN
# --------------------------------------------------

def chat(message, history):
    """
    message = current Gradio user message
    history = previous Gradio conversation
    """

    try:
        messages = []

        # Convert Gradio history to LangChain-compatible messages
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")

                if role in ("user", "assistant") and isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": content
                    })

        # Current user message
        messages.append({
            "role": "user",
            "content": message
        })

        # Run LangChain Agent
        response = agent.invoke({
            "messages": messages
        })

        # Final AI response
        return response["messages"][-1].content

    except Exception as e:
        return f"Error: {str(e)}"


# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------

demo = gr.ChatInterface(
    fn=chat,
    title="✈️ AI Flight Assistant",
    description=(
        "Ask me about available flights, ticket prices, "
        "durations, or simple calculations."
    ),
    examples=[
        "Find flight prices from Delhi to Bangalore",
        "Show flights from Kolkata to Mumbai",
        "What is the cheapest flight from Kolkata to Mumbai?",
        "Find flights from Kolkata to Delhi",
        "Add 25 and 37"
    ]
)


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if _name_ == "_main_":
    demo.launch()