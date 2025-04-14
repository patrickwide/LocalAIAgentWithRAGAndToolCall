# -*- coding: utf-8 -*-
import json
import ollama
import asyncio
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import pandas as pd

# --- RAG Components (from scripts 2 & 3) ---
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever # Explicit import

# Global variable for the retriever (or pass it around)
retriever: Optional[VectorStoreRetriever] = None

# --- Configuration ---
DB_LOCATION = "./chrome_langchain_db_reviews"
CSV_PATH = "realistic_restaurant_reviews.csv" # Make sure this file exists or a dummy will be created
ORDER_FILE = "orders.txt" # File to save orders
OLLAMA_MODEL = "llama3.2" # Specify your desired Ollama model
EMBEDDING_MODEL = "mxbai-embed-large" # Specify your desired embedding model

def setup_vector_store():
    """Initializes the vector store and retriever."""
    global retriever

    print("Setting up vector store...")

    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Creating a dummy file.")
        # Create a dummy CSV if it doesn't exist for the code to run
        dummy_data = {
            'Title': ['Great Place', 'Okay Food', 'Loved the Pepperoni', 'Service issues', 'Vegan Delight'],
            'Date': ['2024-01-01', '2024-01-02', '2024-02-10', '2024-03-05', '2024-04-15'],
            'Rating': [5, 3, 5, 2, 4],
            'Review': ['Loved the pizza! The crust was perfect.', 'Service was slow but the pizza was decent.', 'Best pepperoni pizza I have had in ages! Crispy crust.', 'Had to wait forever for our order. Food was cold.', 'The vegan options are amazing! Tried the Vegan Supreme.']
        }
        pd.DataFrame(dummy_data).to_csv(CSV_PATH, index=False)
        print(f"Created dummy {CSV_PATH}.")

    try:
        df = pd.read_csv(CSV_PATH)
        if not {'Title', 'Review', 'Rating', 'Date'}.issubset(df.columns):
            raise ValueError("CSV must contain 'Title', 'Review', 'Rating', 'Date' columns.")

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Decide if we need to add documents (fresh DB or empty)
        add_documents = True
        if os.path.exists(DB_LOCATION):
             # A more robust check for an existing, potentially populated Chroma DB
             try:
                 if any(fname.endswith(('.db', '.sqlite', '.duckdb', '.parquet')) for fname in os.listdir(DB_LOCATION)):
                    # Attempt to load to see if it's valid, but don't create retriever yet
                    test_load = Chroma(
                        collection_name="restaurant_reviews_agent_v2",
                        persist_directory=DB_LOCATION,
                        embedding_function=embeddings
                    )
                    # Simple check if collection exists (may vary based on Chroma version)
                    if test_load._collection.count() > 0:
                         print(f"Found existing collection with {test_load._collection.count()} documents.")
                         add_documents = False
                    else:
                         print("Found existing directory, but collection seems empty or invalid. Will repopulate.")
                         add_documents = True # Treat as needing population
                 else:
                    print(f"Directory {DB_LOCATION} exists but contains no apparent DB files. Will populate.")
             except Exception as load_err:
                 print(f"Error checking existing DB, will attempt to re-initialize: {load_err}")
                 add_documents = True # Force re-population on error
        else:
            print(f"Directory {DB_LOCATION} not found. Creating new vector store.")


        vector_store = Chroma(
            collection_name="restaurant_reviews_agent_v2", # Use a distinct collection name
            persist_directory=DB_LOCATION,
            embedding_function=embeddings
        )

        if add_documents:
            print(f"Creating/Populating vector store at {DB_LOCATION}")
            documents = []
            ids = []
            count = 0
            for i, row in df.iterrows():
                content = f"Title: {row.get('Title', '')}. Review: {row.get('Review', '')}"
                metadata = {
                    "rating": int(row.get('Rating', 0)) if pd.notna(row.get('Rating')) else 0,
                    "date": str(row.get('Date', '')) if pd.notna(row.get('Date')) else ''
                }
                if not isinstance(metadata["rating"], int): metadata["rating"] = 0
                if not isinstance(metadata["date"], str): metadata["date"] = ""

                doc_id = f"review_{i}"
                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
                ids.append(doc_id)
                count += 1

            if documents:
                vector_store.add_documents(documents=documents, ids=ids)
                print(f"Added {count} documents to the vector store.")
                # vector_store.persist() # <<< REMOVED THIS LINE - FIX 1
                print("Vector store persistence is handled by Chroma initialization.")
            else:
                 print("No documents found in CSV to add.")
        else:
            print(f"Loading existing vector store from {DB_LOCATION}")

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("Retriever setup complete.")

    except Exception as e:
        print(f"Error setting up vector store: {e}")
        retriever = None # Ensure retriever is None if setup fails

# --- Agent Tools ---

def query_restaurant_reviews(query: str) -> str:
    """
    Searches the restaurant review database for reviews relevant to the user's query.
    Use this to answer questions about specific aspects of the restaurant like food quality,
    service, specific dishes (e.g., pepperoni pizza, gluten-free options), or general sentiment.

    Args:
        query: The user's question or topic to search for in the reviews.

    Returns:
        A JSON string containing the retrieved reviews or an error message.
    """
    global retriever
    if retriever is None:
        # Provide a more informative message if the retriever failed setup
        return json.dumps({"error": "Review database could not be initialized. Cannot search reviews."})

    print(f"\n--- Tool Call: query_restaurant_reviews ---")
    print(f"--- Query: {query}")
    try:
        results: List[Document] = retriever.invoke(query)
        if not results:
            return json.dumps({"message": "No relevant reviews found for your query."})

        formatted_results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        print(f"--- Found {len(results)} reviews.")
        return json.dumps(formatted_results, indent=2)

    except Exception as e:
        print(f"Error during review query: {e}")
        # Also inform the user the search failed
        return json.dumps({"error": f"Failed to query reviews: {str(e)}"})

# --- Agent Tools ---

# [query_restaurant_reviews function remains the same]
# ...

def tool_place_order(
    pizza_type: str,
    size: str,
    quantity: Any, # Accept Any initially, as LLM might send string or int
    delivery_address: str,
    customer_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    crust_type: Optional[str] = "Regular",
    extra_toppings: Optional[List[str]] = None,
) -> str:
    """
    Places a pizza order with the specified details and saves it to a file.
    Use this tool *only* when the user explicitly confirms they want to place an order
    and has provided the necessary details (pizza type, size, quantity, delivery address).

    Args:
        pizza_type: The type of pizza being ordered (e.g., 'Pepperoni', 'Margherita').
        size: The size of the pizza (e.g., 'Large', 'Medium').
        quantity: The number of pizzas of this type and size (should be an integer).
        delivery_address: The full address for delivery.
        customer_name: The name of the customer placing the order (optional).
        phone_number: The contact phone number for the order (optional).
        crust_type: The desired crust type (e.g., 'Thin', 'Stuffed', defaults to 'Regular').
        extra_toppings: A list of any extra toppings requested (optional).

    Returns:
        A JSON string confirming the order details placed or an error message.
    """
    print(f"\n--- Tool Call: tool_place_order ---")
    # Print raw quantity received for debugging
    print(f"--- Raw Args: pizza_type='{pizza_type}', size='{size}', quantity='{quantity}' (type: {type(quantity)}), address='{delivery_address}', "
          f"name='{customer_name}', phone='{phone_number}', crust='{crust_type}', extras={extra_toppings}")

    # --- FIX: Convert quantity to integer and validate ---
    quantity_int: Optional[int] = None
    try:
        quantity_int = int(quantity) # Attempt conversion
        if quantity_int <= 0:
             raise ValueError("Quantity must be positive.") # Add check for non-positive numbers
    except (ValueError, TypeError) as e:
        # Handle cases: quantity is not a valid integer string, is None, or <= 0
        error_msg = f"Invalid quantity received: '{quantity}'. Quantity must be a positive whole number. Error: {e}"
        print(f"--- Validation Error: {error_msg}")
        return json.dumps({"error": error_msg})
    # --- END FIX ---

    # --- Use the validated integer quantity_int moving forward ---
    print(f"--- Validated Args: pizza_type='{pizza_type}', size='{size}', quantity={quantity_int}, address='{delivery_address}', "
          f"name='{customer_name}', phone='{phone_number}', crust='{crust_type}', extras={extra_toppings}")


    # Simplified validation now relies on successful conversion above and checks other fields
    if not all([pizza_type, size, delivery_address]): # quantity_int > 0 already checked
         error_msg = "Missing required order details: pizza_type, size, and delivery_address are mandatory."
         print(f"--- Validation Error: {error_msg}")
         return json.dumps({"error": error_msg})

    order_details = {
        "order_timestamp": datetime.now().isoformat(),
        "pizza_type": pizza_type,
        "size": size,
        "quantity": quantity_int, # <<< Use the integer version here
        "crust_type": crust_type if crust_type else "Regular",
        "extra_toppings": extra_toppings if extra_toppings is not None else [],
        "delivery_address": delivery_address,
        "customer_name": customer_name,
        "phone_number": phone_number,
        "status": "Received"
    }

    try:
        # Append the order as a JSON line to the order file
        with open(ORDER_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(order_details) + '\n')
        print(f"--- Order successfully saved to {ORDER_FILE}")

        confirmation_message = {
            "status": "Order Placed Successfully",
            "confirmation": f"OK. Your order for {quantity_int} x {size} {pizza_type} pizza(s) " # Use quantity_int
                            f"{('with crust ' + order_details['crust_type']) if order_details['crust_type'] != 'Regular' else ''} "
                            f"{('and extra ' + ', '.join(order_details['extra_toppings']) + ' ') if order_details['extra_toppings'] else ''}"
                            f"to be delivered to '{delivery_address}' has been received.",
        }
        return json.dumps(confirmation_message, indent=2)

    except IOError as e:
        print(f"Error saving order to {ORDER_FILE}: {e}")
        return json.dumps({"error": f"Failed to save the order due to a file system error: {str(e)}"})
    except Exception as e:
        print(f"An unexpected error occurred during order placement: {e}")
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

# --- [Rest of the script: AgentMemory, run_agent, main, etc.] ---
# Make sure to replace the old tool_place_order function definition
# with this corrected version in your restaurant_agent.py file.

# --- Agent Memory ---
class AgentMemory:
    """Class to maintain conversation history and context for the agent"""
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.function_call_attempts: Dict[str, int] = {} # Track attempts per function/args

    def add_message(self, message: Dict[str, Any]) -> None:
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.max_history:
            # A simple trim keeping the last max_history items
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        if count is None:
            return self.conversation_history
        return self.conversation_history[-count:]

    def record_function_attempt(self, function_name: str, args: Dict[str, Any]) -> int:
        key = f"{function_name}:{json.dumps(args, sort_keys=True)}"
        self.function_call_attempts[key] = self.function_call_attempts.get(key, 0) + 1
        return self.function_call_attempts[key]

# --- Agent Runner ---
async def run_agent(model: str, user_input: str, memory: AgentMemory):
    client = ollama.AsyncClient()

    memory.add_message({"role": "user", "content": user_input})
    messages = memory.get_recent_messages()

    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_restaurant_reviews",
                "description": "Searches and retrieves relevant customer reviews for the pizza restaurant based on a query. Useful for answering questions about food (pizza, crust, toppings, vegan, gluten-free), service, price, atmosphere, or overall experiences. Do NOT use this to place an order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific question or topic to search for in the reviews (e.g., 'pepperoni pizza quality', 'service speed', 'are there vegan options?', 'comments about the crust').",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_place_order",
                "description": "Places a pizza order with the specified details and saves it. Use this tool ONLY when the user explicitly confirms they want to place an order and has provided AT LEAST the pizza type, size, quantity, and delivery address. Ask clarifying questions first if details are missing. Do not invent details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pizza_type": {"type": "string", "description": "The type of pizza (e.g., 'Pepperoni', 'Margherita', 'Vegan Supreme')."},
                        "size": {"type": "string", "description": "The size of the pizza (e.g., 'Large', 'Medium', 'Small')."},
                        "quantity": {"type": "integer", "description": "The number of pizzas."},
                        "delivery_address": {"type": "string", "description": "The full delivery address."},
                        # --- FIX 2: Changed type for optional parameters ---
                        "customer_name": {"type": "string", "description": "Customer's name (optional). Omit if not provided."},
                        "phone_number": {"type": "string", "description": "Customer's phone number (optional). Omit if not provided."},
                        # --- End Fix 2 ---
                        "crust_type": {"type": "string", "description": "Desired crust type (e.g., 'Thin', 'Regular', 'Stuffed'). Defaults to 'Regular' if not specified.", "default": "Regular"},
                        "extra_toppings": {"type": "array", "items": {"type": "string"}, "description": "List of extra toppings (optional). Provide as an empty list if none."},
                    },
                    # Optional parameters are implicitly defined by NOT being in 'required'
                    "required": ["pizza_type", "size", "quantity", "delivery_address"],
                },
            },
        },
    ]

    print("\n--- Agent --- Sending request to LLM...")
    # print("--- Sending Messages:", json.dumps(messages, indent=2)) # DEBUG: See messages sent
    # print("--- Sending Tools:", json.dumps(tools, indent=2)) # DEBUG: See tools sent
    try:
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )
        # print("--- LLM Raw Response:", json.dumps(response, indent=2)) # DEBUG: See raw response
    except Exception as e:
        print(f"Error calling Ollama chat API: {e}")
        memory.add_message({"role": "system", "content": f"Error contacting LLM: {e}"})
        return

    memory.add_message(response["message"])

    if not response["message"].get("tool_calls"):
        print("\n--- Agent --- LLM Response (no tool call):")
        print(response["message"]["content"])
        return

    print("\n--- Agent --- LLM decided to use a tool.")
    available_functions = {
        "query_restaurant_reviews": query_restaurant_reviews,
        "tool_place_order": tool_place_order,
    }

    tool_calls_to_process = list(response["message"].get("tool_calls", []))

    for tool_call in tool_calls_to_process:
        # Robustly access potentially missing keys
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name")
        raw_function_args = function_info.get("arguments")
        tool_call_id = tool_call.get("id") # Use this if provided by Ollama

        if not function_name or raw_function_args is None:
             print(f"--- Agent --- ERROR: Invalid tool call structure received: {tool_call}")
             # Add minimal error message if possible
             err_content = json.dumps({"error": "Received invalid tool call structure from LLM."})
             memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name or "unknown", "content": err_content})
             continue

        function_args = None
        print(f"--- Agent --- Attempting Tool Call: {function_name}")

        if isinstance(raw_function_args, dict):
            print("--- Agent --- Arguments received as dict.")
            function_args = raw_function_args
        elif isinstance(raw_function_args, str):
             print("--- Agent --- Arguments received as string, attempting JSON parse.")
             try:
                 function_args = json.loads(raw_function_args)
             except json.JSONDecodeError:
                 print(f"--- Agent --- ERROR: Could not parse function arguments string: {raw_function_args}")
                 err_content = json.dumps({"error": f"Malformed arguments JSON received: {raw_function_args}"})
                 memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": err_content})
                 continue
        else:
            print(f"--- Agent --- ERROR: Unexpected type for function arguments: {type(raw_function_args)}")
            err_content = json.dumps({"error": f"Unexpected argument type: {type(raw_function_args)}"})
            memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": err_content})
            continue

        if function_args is None:
             continue # Skip if args are invalid

        attempt_count = memory.record_function_attempt(function_name, function_args)
        print(f"--- Agent --- Calling: {function_name} (attempt {attempt_count}) | Args: {function_args}")

        if function_name in available_functions:
            function_to_call = available_functions[function_name]
            try:
                function_response = function_to_call(**function_args)
                print(f"--- Agent --- Function {function_name} executed.")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": function_response, # Should be JSON string
                }
                memory.add_message(tool_message)
                print(f"--- Agent --- Added tool response to memory.")

            except TypeError as e:
                 # More specific error for bad arguments passed to the python function
                 print(f"Error calling function {function_name} due to argument mismatch: {e}")
                 error_content = json.dumps({"error": f"Argument mismatch calling {function_name}: {str(e)}. Check if LLM provided all required args correctly. Provided: {function_args}"})
                 memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": error_content})
            except Exception as e:
                print(f"Error executing function {function_name}: {e}")
                error_content = json.dumps({"error": f"Runtime error calling function {function_name}: {str(e)}"})
                memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": error_content})
        else:
            print(f"Error: Function '{function_name}' not found in available_functions.")
            error_content = json.dumps({"error": f"Function '{function_name}' is not implemented."})
            memory.add_message({"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": error_content})


    # After processing ALL tool calls, send updated history back to LLM
    print("\n--- Agent --- Sending updated conversation history to LLM for final response...")
    # print("--- Sending Messages for Final Response:", json.dumps(memory.get_recent_messages(), indent=2)) # DEBUG
    try:
        # Ensure messages list is not empty before calling chat
        if not memory.get_recent_messages():
            print("--- Agent --- Error: No messages in history to send for final response.")
            return

        final_response = await client.chat(
            model=model,
            messages=memory.get_recent_messages(),
            # No tools needed here usually
        )
        # print("--- Final LLM Raw Response:", json.dumps(final_response, indent=2)) # DEBUG
        # Add the final response only if it contains content
        if final_response["message"].get("content"):
             memory.add_message(final_response["message"])
             print("\n--- Agent --- Final LLM Response:")
             print(final_response["message"]["content"])
        else:
             print("\n--- Agent --- LLM provided no final content response (might happen after tool error).")


    except Exception as e:
        print(f"Error getting final response from Ollama: {e}")
        memory.add_message({"role": "system", "content": f"Error getting final LLM response: {e}"})


# --- Main Execution Block ---
async def main():
    setup_vector_store() # Setup DB first

    if retriever is None:
       print("Warning: Failed to initialize retriever. Review search functionality will be unavailable.")
       # Proceeding without retriever functionality

    memory = AgentMemory(max_history=15) # Increased history slightly

    print("\nWelcome to the Pizza Restaurant Assistant!")
    print("Ask about reviews or place an order.")
    print("(e.g., 'How is the pepperoni pizza?', 'Tell me about the service',")
    print("      'I want to order 1 large veggie pizza to 456 Oak Avenue', 'exit' to quit)")

    while True:
        try:
            user_input = input("\nPlease ask or order=> ")
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            await run_agent(OLLAMA_MODEL, user_input, memory)

        except KeyboardInterrupt:
             print("\nExiting...")
             break
        except Exception as e:
             print(f"\nAn unexpected error occurred in the main loop: {e}")
             print("Restarting loop...")
             await asyncio.sleep(1)


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
         print(f"Warning: {CSV_PATH} not found. A dummy file will be created on first run.")
    asyncio.run(main())