from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


@tool
def get_flight_prices(departure: str,arrival: str) -> str:
    """
    Mock flight price lookup tool.
    """

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
        return f"No mock flight data found for {departure} → {arrival}"

    result = []

    for f in flights:
        result.append(
            f"""
Airline: {f['airline']}
Price: ₹{f['price']}
Duration: {f['duration']}
"""
        )

    return "\n".join(result)


llm = ChatOllama(
    model="llama3.1",
    base_url="http://127.0.0.1:11434"
)

agent = create_agent(
    model=llm,
    tools=[get_flight_prices],
    system_prompt='''
    You are a helpful AI agent.
    Use tools whenever needed.
If user asks for flights or ticket prices,
use get_flight_prices tool.
'''
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find flight prices from Delhi to Bangalore"}
    ]
})

print(response["messages"][-1].content)