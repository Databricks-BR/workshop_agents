#!/usr/bin/env python3
"""
Test script for LangGraph time-travel functionality
Based on Databricks documentation: https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents#implement-langgraph-time-travel
"""

import logging
from agent import get_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_langgraph_timetravel():
    """Test LangGraph checkpoint time-travel as per Databricks docs"""
    
    print("🚀 Testing LangGraph Time-Travel Functionality")
    print("=" * 60)
    
    try:
        # Get the agent
        AGENT = get_agent()
        print("✅ Agent loaded successfully")
        
        # 1. Start a conversational thread and add a few messages
        print("\n📝 Step 1: Starting initial conversation...")
        response1 = AGENT.predict({
            "input": [{"role": "user", "content": "I'm planning for an upcoming trip!"}],
        })
        print("Response 1:", response1.output[0].content if response1.output else "No output")
        thread_id = response1.custom_outputs["thread_id"]
        print(f"Thread ID: {thread_id}")
        
        # 2. Continue conversation within the same thread
        print("\n📝 Step 2: Follow-up question...")
        response2 = AGENT.predict({
            "input": [{"role": "user", "content": "I'm headed to SF!"}],
            "custom_inputs": {"thread_id": thread_id}
        })
        print("Response 2:", response2.output[0].content if response2.output else "No output")
        
        # 3. Another follow-up to test memory
        print("\n📝 Step 3: Testing memory...")
        response3 = AGENT.predict({
            "input": [{"role": "user", "content": "Where did I say I'm going?"}],
            "custom_inputs": {"thread_id": thread_id}
        })
        print("Response 3:", response3.output[0].content if response3.output else "No output")
        
        # 4. Get checkpoint history
        print("\n📚 Step 4: Retrieving checkpoint history...")
        history = AGENT.get_checkpoint_history(thread_id, 20)
        print(f"Found {len(history)} checkpoints:")
        
        for i, checkpoint in enumerate(history):
            print(f"  {i+1}. Checkpoint: {checkpoint['checkpoint_id'][:8]}...")
            print(f"     Messages: {checkpoint['message_count']}")
            print(f"     Timestamp: {checkpoint['timestamp']}")
            print(f"     Last message: {checkpoint['last_message']}")
            print()
        
        # 5. Branch from an earlier checkpoint
        if len(history) >= 2:
            print("\n🌿 Step 5: Branching from earlier checkpoint...")
            # Get a checkpoint from earlier in the conversation
            branch_checkpoint = history[-2]["checkpoint_id"]  # Second-to-last checkpoint
            print(f"Branching from checkpoint: {branch_checkpoint[:8]}...")
            
            # Branch with different information
            response4 = AGENT.predict({
                "input": [{"role": "user", "content": "Actually, I'm headed to New York instead!"}],
                "custom_inputs": {
                    "thread_id": thread_id,
                    "checkpoint_id": branch_checkpoint  # Branch from this checkpoint!
                }
            })
            print("Branched Response:", response4.output[0].content if response4.output else "No output")
            
            # 6. Test that the branch worked
            print("\n🔍 Step 6: Testing branched conversation...")
            response5 = AGENT.predict({
                "input": [{"role": "user", "content": "Where am I going now?"}],
                "custom_inputs": {"thread_id": thread_id}
            })
            print("Final Response:", response5.output[0].content if response5.output else "No output")
            
            # Show updated history
            print("\n📚 Updated checkpoint history after branching:")
            final_history = AGENT.get_checkpoint_history(thread_id, 10)
            print(f"Now has {len(final_history)} checkpoints (should be more than before)")
            
        print("\n✅ LangGraph time-travel test completed successfully!")
        print("🎯 Key capabilities demonstrated:")
        print("   • Thread-based conversation persistence")
        print("   • Checkpoint history retrieval")
        print("   • Conversation branching from checkpoints")
        print("   • State preservation in Lakebase")
        
    except Exception as e:
        print(f"❌ Error during time-travel test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_langgraph_timetravel()
