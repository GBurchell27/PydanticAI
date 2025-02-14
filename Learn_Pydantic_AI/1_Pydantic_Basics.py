import os
from dotenv import load_dotenv

load_dotenv()
print("API Key loaded:", bool(os.getenv('OPENAI_API_KEY')))

##################################################    
# 1. BASIC AGENT SETUP
# This is the simplest agent that answers questions using default settings
from pydantic_ai import Agent
# Create agent with GPT-4 model
basic_agent = Agent('openai:gpt-4o')
# Synchronous execution
result = basic_agent.run_sync('What is the capital of France?')
print(result.data)  # Output: Paris


##################################################
# 2. ADDING SYSTEM PROMPT
# System prompts guide the agent's behavior
from pydantic_ai import Agent
# Create agent with custom system prompt
chef_agent = Agent(
    'openai:gpt-4o',
    system_prompt="You are a master chef. Always answer with recipes and cooking tips."
)
# Get cooking advice
result = chef_agent.run_sync('How do I make perfect scrambled eggs?')
print(result.data)



##################################################
# 3. CREATING TOOLS
# Tools are functions the agent can use during execution
from pydantic_ai import Agent, RunContext

# Create calculator agent
calc_agent = Agent(
    'openai:gpt-4o',
    system_prompt="Use calculator tools for math operations."
)
# Register tool with agent
@calc_agent.tool
async def calculator(ctx: RunContext, expression: str) -> float:
    """Evaluate mathematical expressions"""
    return eval(expression)

# Use tool through agent
result = calc_agent.run_sync('What is 15 cubed?')
print(result.data)  # Output: 3375



##################################################
# 4. STRUCTURED OUTPUT
# Enforce structured responses using Pydantic models
from pydantic import BaseModel
from pydantic_ai import Agent

# Define response structure
class WeatherResponse(BaseModel):
    city: str
    temperature: float
    unit: str
    conditions: str
# Create weather agent
weather_agent = Agent(
    'openai:gpt-4o',
    result_type=WeatherResponse, # this is the structure of the response
    system_prompt="Provide weather information in structured format."
)
# Get structured weather data
result = weather_agent.run_sync('What is the weather in London?')
print(result.data.dict())
# Output: {'city': 'London', 'temperature': 18.5, 'unit': 'C', 'conditions': 'Partly cloudy'}



##################################################
# 5. DEPENDENCY INJECTION
# Pass runtime dependencies to agent tools
from pydantic_ai import Agent, RunContext

# Create database-aware agent
db_agent = Agent(
    'openai:gpt-4o',
    deps_type=dict,  # Dependency type (simulated database)
    system_prompt="Use user_db tool to access user information."
)
@db_agent.tool
async def user_db(ctx: RunContext[dict], user_id: int) -> dict:
    """Access user database"""
    return ctx.deps.get(user_id, {})
# Simulated database
user_database = {
    1: {'name': 'Alice', 'email': 'alice@example.com'},
    2: {'name': 'Bob', 'email': 'bob@example.com'}
}
# Query user information
result = db_agent.run_sync('Get email for user ID 2', deps=user_database)
print(result.data)  # Output: bob@example.com

         
            
##################################################
# 6. DYNAMIC SYSTEM PROMPTS
# Generate context-aware system prompts
from datetime import datetime
from pydantic_ai import Agent, RunContext

time_agent = Agent(
    'openai:gpt-4o',
    deps_type=datetime,
    system_prompt="Include current time in responses."
)

@time_agent.system_prompt
def add_current_time(ctx: RunContext[datetime]) -> str:
    return f"Current time is {ctx.deps.strftime('%H:%M')}"

# Get time-aware response
result = time_agent.run_sync('What time is it?', deps=datetime.now())
print(result.data)  # Output depends on current time




##################################################
# 7. ERROR HANDLING & RETRIES
# Implement error recovery and retry logic
from pydantic_ai import Agent, ModelRetry, RunContext

retry_agent = Agent(
    'openai:gpt-4o',
    retries=2,
    system_prompt="Handle errors gracefully and retry when needed."
)

@retry_agent.tool(retries=3)
async def unstable_api(ctx: RunContext) -> str:
    """Simulate flaky API call"""
    if ctx.retry < 2:
        raise ModelRetry("API temporarily unavailable")
    return "Success on retry 3"

result = retry_agent.run_sync('Call unstable API')
print(result.data)  # Output: Success on retry 3




##################################################
# 8. USAGE LIMITS & MODEL SETTINGS
# Control resource usage and model behavior
from pydantic_ai import Agent, UsageLimits

limit_agent = Agent('openai:gpt-4o')

# Enforce usage limits
result = limit_agent.run_sync(
    'Explain quantum physics in simple terms',
    usage_limits=UsageLimits(response_tokens_limit=100),
    model_settings={'temperature': 0.2}
)
print(result.data)  # Concise explanation within token limit




##################################################
# 9. CONVERSATION HISTORY
# Maintain context across multiple interactions
from pydantic_ai import Agent

chat_agent = Agent('openai:gpt-4o')

# First interaction
result1 = chat_agent.run_sync('Who invented the telephone?')
print(result1.data)  # Output: Alexander Graham Bell

# Follow-up with history
result2 = chat_agent.run_sync(
    'What year was that?',
    message_history=result1.new_messages()
)
print(result2.data)  # Output: 1876

    
    
    
##################################################
# 10. COMPLEX WORKFLOW
# Combine multiple features in real-world scenario
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry

# Define data structures
class Customer(BaseModel):
    id: int
    name: str
    premium: bool

class SupportResponse(BaseModel):
    response: str
    priority: int
    follow_up: bool

# Create customer support agent
support_agent = Agent(
    'openai:gpt-4o',
    deps_type=list[Customer],
    result_type=SupportResponse,
    system_prompt="Handle support tickets using customer data.",
    retries=2
)

# Customer database tool
@support_agent.tool
async def find_customer(ctx: RunContext[list[Customer]], name: str) -> Customer:
    """Find customer by name"""
    for customer in ctx.deps:
        if customer.name.lower() == name.lower():
            return customer
    raise ModelRetry(f"Customer {name} not found")

# Complex query with error handling
customers = [
    Customer(id=1, name="Alice", premium=True),
    Customer(id=2, name="Bob", premium=False)
]

result = support_agent.run_sync(
    "Help Alice with her premium account issue",
    deps=customers
)

print(result.data.dict())
# Output example: {'response': '...', 'priority': 1, 'follow_up': True}