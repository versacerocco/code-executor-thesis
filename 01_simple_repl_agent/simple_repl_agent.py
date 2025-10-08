# simple_repl_agent.py

# This script defines an AI agent using LangGraph and a local llama-cpp model.
# The agent can use two tools: a summation tool and a Python REPL tool for executing simple code snippets.
# The model is Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf -> https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF
# The .gguf model file must be in the same directory as this script.

import os
import json
import re
from typing import Literal
from llama_cpp import Llama  
from langchain_core.tools import tool  
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  
from langgraph.graph import StateGraph, MessagesState, START, END  
from langgraph.prebuilt import ToolNode  
from langchain_experimental.utilities import PythonREPL 

# --- TOOL DEFINITIONS ---
# Tools are functions the AI agent can decide to use to answer a user's query.

# Create a single instance of the Python REPL utility to be reused.
python_repl = PythonREPL()

@tool
def execute_python(code: str) -> str:
    """
    Executes a string of Python code in a REPL environment and returns the output.
    Use this for calculations, string manipulations, or any other task that requires code execution.
    The code should be a valid Python snippet. For example, to calculate 5 to the power of 8,
    the code would be 'print(5**8)'.
    """
    
    print(f"\n[PYTHON REPL TOOL HAS BEEN CALLED] execute_python(code='{code}')")
    try:
        # Execute the provided Python code string.
        result = python_repl.run(code)
        return f"Execution successful. Result:\n{result}"
    except Exception as e:
        # If the code fails, return a descriptive error message to the AI.
        return f"Error executing code: {e}"

@tool
def sum_tool(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"\n[SUM TOOL HAS BEEN CALLED] sum_tool(a={a}, b={b})")
    return a + b

# A list containing all the functions the agent is allowed to use.
tools = [sum_tool, execute_python]

# --- CUSTOM LLM WRAPPER ---
# This class acts as a bridge between LangGraph and the local llama-cpp model.
# It formats the prompts EXACTLY as Hermes-2-Pro expects and parses its specific output format.

class LlamaCppWrapper:
    def __init__(self, llm: Llama, tools: list, max_tokens: int = 512):
        self.llm = llm
        self.max_tokens = max_tokens

        # Convert the LangChain tool format into the JSON schema that Hermes-2-Pro requires.
        # This structure is critical for the model to understand the available functions.
        tools_list = []
        for t in tools:
            schema = t.args_schema.model_json_schema()
            tools_list.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": schema.get('properties', {}),
                        "required": schema.get('required', [])
                    }
                }
            })

        # The model expects the tool definitions as a JSON string.
        tools_json = json.dumps(tools_list, indent=2)

        # This system prompt follows the official Hermes-2-Pro function-calling documentation.
        # It instructs the model on its role and how to format its response when it wants to call a tool.

        self.system = f"""You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You must use the provided tools to answer the user's query.

<tools>
{tools_json}
</tools>

When you need to call a tool, you must respond with a JSON object enclosed in <tool_call></tool_call> tags. The JSON object must contain "name" and "arguments" keys.

For example, to answer "what is 5 plus 3", you would respond ONLY with:
<tool_call>
{{'name': 'sum_tool', 'arguments': {{'a': 5, 'b': 3}}}}
</tool_call>

Do not under any circumstances reply with any other text, explanation, or conversational content.
"""

    def invoke(self, messages: list):
        # This method builds the full conversation history into the ChatML format that the model was trained on.
        # Format: <|im_start|>role\ncontent<|im_end|>
        conv = f"<|im_start|>system\n{self.system}<|im_end|>\n"

        for msg in messages:
            if isinstance(msg, HumanMessage):
                conv += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif isinstance(msg, AIMessage):
                # Reconstruct the AI's previous turn, including any tool calls it made.
                conv += f"<|im_start|>assistant\n{msg.content}"
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Recreate the exact <tool_call> tag string. The model uses single quotes.
                        args_str = str(tc['args']).replace('"', "'")
                        tool_call_str = f"<tool_call>\n{{'arguments': {args_str}, 'name': '{tc['name']}'}}\n</tool_call>"
                        conv += tool_call_str
                conv += "<|im_end|>\n"
            elif isinstance(msg, ToolMessage):
                # Add the result from a tool execution to the conversation history.
                tool_response = json.dumps({'name': msg.name, 'content': msg.content})
                conv += f"<|im_start|>tool\n<tool_response>\n{tool_response}\n</tool_response>\n<|im_end|>\n"
        
        # This special token signals to the model that it's now its turn to generate a response.
        conv += "<|im_start|>assistant\n"

        # Call the local llama-cpp model with the complete prompt and generation settings.
        response = self.llm(
            conv,
            max_tokens=self.max_tokens,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|im_end|>"],  # Stop generating text when the model finishes its turn.
            echo=False
        )

        text = response['choices'][0]['text'].strip()

        # After getting the model's response, parse it to find any tool call requests.
        tool_calls = []
        # Use regex to find all occurrences of the <tool_call> tag.
        for i, match in enumerate(re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)):
            try:
                # The model outputs a Python-style dict string (with single quotes), so we clean it up for JSON parsing.
                match_normalized = match.replace("'", '"').replace("True", "true").replace("False", "false")
                tc = json.loads(match_normalized)
                
                # Format the parsed tool call into the structure LangGraph expects.
                tool_calls.append({
                    "name": tc["name"],
                    "args": tc["arguments"],
                    "id": f"call_{i}",
                    "type": "tool_call"
                })
                print(f"[PARSED TOOL CALL] {tc['name']} with args {tc['arguments']}")
            except Exception as e:
                print(f"[WARNING] Failed to parse tool call: {e}")
                print(f"[DEBUG] Raw match: {match}")
        
        # Clean the final content by removing the tool call tags, so it's not displayed to the user.
        content_without_tool_calls = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
        
        # Return an AIMessage. If tool calls were found, include them. Otherwise, return a standard message.
        return AIMessage(content=content_without_tool_calls, tool_calls=tool_calls) if tool_calls else AIMessage(content=text)

# --- LANGGRAPH AGENT DEFINITION ---
# A graph represents the agent's reasoning loop as a series of steps (nodes) and choices (edges).

# Agent Node: This is the "brain" of the agent. It calls the LLM to decide what to do next.
def call_model(state: MessagesState, config):
    print("\n[NODE: agent] Calling LLM...")
    # The 'config' dictionary is used to pass the llm_wrapper to this node during runtime.
    response = config["configurable"]["llm_wrapper"].invoke(state["messages"])
    print("[NODE: agent] Response generated")
    
    # Check if the response contains a decision to use a tool.
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[NODE: agent] Tool calls detected: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print("[NODE: agent] No tool calls detected")
        print(f"[DEBUG] Response content: {response.content[:200]}...")
        
    # The node must return a dictionary with the key "messages" to update the graph's state.
    return {"messages": [response]}

# Conditional Edge: This function determines the next step in the graph after the agent node runs.
def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    # If the last message from the AI contains tool calls, route to the "tools" node.
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("\n[EDGE DECISION] Routing to: tools")
        return "tools"
    # Otherwise, if there are no tool calls, the AI has its final answer, so we can end.
    else:
        print("\n[EDGE DECISION] Routing to: end")
        return "end"

# Function to build and compile the graph.
def create_agent(llm: Llama, tools: list, max_tokens: int = 512):
    wrapper = LlamaCppWrapper(llm, tools, max_tokens)
    
    # Define the state machine graph. MessagesState keeps track of the conversation history.
    workflow = StateGraph(MessagesState)

    # Add the nodes to the graph.
    workflow.add_node("agent", call_model)  # The node that calls the LLM.
    workflow.add_node("tools", ToolNode(tools))  # The node that executes tools.

    # Define the flow of the graph.
    workflow.set_entry_point("agent")  # The graph always starts at the "agent" node.
    
    # After the "agent" node, call the `should_continue` function to decide where to go next.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If it returns "tools", go to the "tools" node.
            "end": END,      # If it returns "end", finish the execution.
        }
    )
    
    # After the "tools" node runs, loop back to the "agent" node to process the tool results.
    workflow.add_edge("tools", "agent")

    # Compile the graph into a runnable object.
    graph = workflow.compile()
    
    # Attach the wrapper to the graph object for easy access later.
    graph.llm_wrapper = wrapper
    return graph

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define the local model file to use, the .gguf file must be in the same directory as this script.
    model_filename = "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"
    # Construct the full path to the model file.
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    # Load the GGUF model into memory using llama-cpp-python.
    # n_gpu_layers=-1 attempts to offload all layers to the GPU for faster inference.
    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1, verbose=False)

    # Create the agent graph.
    agent = create_agent(llm, tools, max_tokens=2048)
    
    print("="*60)
    print("HERMES-2-PRO FUNCTION CALLING AGENT")
    print("="*60)
    print("GRAPH ARCHITECTURE:")
    print("   START → agent → [conditional] → tools → agent → END")
    print("AVAILABLE TOOLS:", [tool.name for tool in tools])
    print("="*60)


    # ===== DEFINE THE USER'S QUESTION =====
    example_query = "can you square each element in this list: [1, 2, 3, 4, 5] and then sum the results?"
    # ======================================


    print(f"EXECUTING QUERY: \"{example_query}\"")
    
    final_state = None
    # Stream the execution of the graph. This runs the agent and prints updates for each step.
    for step in agent.stream(
        {"messages": [HumanMessage(content=example_query)]},  # The initial input.
        config={"configurable": {"llm_wrapper": agent.llm_wrapper}}, # Pass the wrapper to the nodes.
        stream_mode="updates"  # Get updates as each node finishes.
    ):
        node_name = list(step.keys())[0]
        print(f"\n[STEP] Node '{node_name}' executed")
        final_state = step

    print("\n" + "="*60)
    print("FINAL RESPONSE:")
    print("="*60)
    
    # Extract and print the final AI-generated message from the last step.
    if final_state:
        for node, data in final_state.items():
            if "messages" in data:
                # Find the last AIMessage in the history, which will be the final answer.
                final_message = next((msg for msg in reversed(data["messages"]) if isinstance(msg, AIMessage)), None)
                if final_message and not final_message.tool_calls:
                    print(final_message.content)

    print("\n" + "="*60)
    print("[FINISHED] Graph execution complete")
    print("="*60)

   # Clean up resources, I added this to avoid some warnings on the terminal. 
    if hasattr(llm, 'close'):
        llm.close()

