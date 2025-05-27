import sys
import asyncio

# Adjust path to import from the correct location if necessary
# This assumes the script is run from the root of the repository
sys.path.insert(0, './')

try:
    from openmanus_rl.agentgym.agentenv.agentenv.envs.openmanus_env import OpenManusLocalEnvClient, LocalToolExecutor
    from openmanus_rl.agentgym.agentenv.agentenv.controller import StepOutput # For type hinting, not strictly necessary for runtime
except ModuleNotFoundError as e:
    print(f"Error: Could not import necessary modules. Make sure you are in the repository root. Details: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"Error: Import failed. Details: {e}")
    sys.exit(1)

async def main():
    print("Starting minimal test for OpenManusLocalEnvClient...")

    # Configuration for the client and executor
    # We use a small max_steps to test the termination condition quickly.
    env_config = {"max_steps": 3, "str_editor_workspace_root": "/tmp"} 
    client = OpenManusLocalEnvClient(
        env_server_base="local_test_server", # Dummy value, not used by local client
        data_len=1,                          # Dummy value
        env_specific_config=env_config
    )

    print("Resetting environment...")
    try:
        # The reset method in the provided code is synchronous.
        reset_output = client.reset(idx=0, task_description="Minimal test task")
        print(f"Reset output observation: {reset_output['observation'][:100]}...") # Print first 100 chars
        print(f"Reset output next_prompt: {reset_output['next_prompt'][:100]}...")
    except Exception as e:
        print(f"Error during client.reset(): {e}")
        await client.close()
        return

    max_steps_to_run = env_config["max_steps"]
    print(f"Attempting to run {max_steps_to_run} steps with empty actions ('[]')...")

    try:
        for i in range(max_steps_to_run + 1): # Run one more step than max_steps to ensure termination
            print(f"\n--- Step {i+1} ---")
            action_str = "[]" # Empty list of actions
            
            # The step method in the provided code is synchronous and calls asyncio.run internally.
            step_result: StepOutput = client.step(action_str)
            
            print(f"Step {i+1} - Action: '{action_str}'")
            print(f"Step {i+1} - Done: {step_result.done}")
            print(f"Step {i+1} - Reward: {step_result.reward}")
            print(f"Step {i+1} - Observation: {step_result.state[:200]}...") # Print first 200 chars

            if step_result.done:
                print(f"\nEnvironment indicated 'done' at step {i+1}.")
                if i == max_steps_to_run -1: # Done on the last expected step
                     print("Termination at max_steps seems correct.")
                elif i < max_steps_to_run -1 :
                     print(f"Warning: Done flagged earlier than max_steps ({max_steps_to_run}).")
                else: # i == max_steps_to_run (i.e. one step *after* max_steps)
                     # This is also acceptable if the termination logic flags done *after* the max_steps action is processed.
                     print("Termination after processing max_steps action seems correct.")
                break
        else:
            # This else block executes if the loop completes without a 'break'.
            print(f"Warning: Environment did not indicate 'done' after {max_steps_to_run + 1} steps.")

        # Check the final state message for max_steps termination
        final_obs = client.observe()
        if "Max steps reached. Episode terminated." in final_obs:
            print("\nSUCCESS: 'Max steps reached' message found in final observation.")
            print("The re-indented code block was likely executed without a NameError.")
        else:
            print("\nFAILURE: 'Max steps reached' message NOT found in final observation.")
            print(f"Final observation: {final_obs}")

    except Exception as e:
        print(f"Error during step execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing client...")
        await client.close()
        print("Test finished.")

if __name__ == "__main__":
    # If running in an environment that already has an asyncio loop (e.g. Jupyter notebook),
    # asyncio.run() might not be suitable. However, for a standalone script, it's fine.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("Error: asyncio.run() cannot be called from a running event loop. Try running in a different environment or adjusting the script's async execution.")
        else:
            raise
