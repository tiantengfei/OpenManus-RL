import sys
import json

# Adjust path to import OpenManusAgent from the correct location
# This assumes the script is run from the root of the repository (/app)
sys.path.insert(0, './')
from openmanus_rl.llm_agent.openmanus import OpenManusAgent

def run_test_case(test_name, prediction_input, expected_action_type, expected_content_str):
    print(f"Running {test_name}...")
    try:
        action_types, contents = OpenManusAgent.postprocess_predictions(None, prediction_input)
        actual_action_type = action_types[0]
        actual_content_str = contents[0]

        assert actual_action_type == expected_action_type, \
            f"Action type mismatch: Expected '{expected_action_type}', Got '{actual_action_type}'"
        
        # For content, if expected is JSON, compare parsed objects to be robust to key order/whitespace.
        # Otherwise, direct string comparison.
        if (expected_content_str.startswith('{') and expected_content_str.endswith('}')) or \
           (expected_content_str.startswith('[') and expected_content_str.endswith(']')):
            try:
                expected_obj = json.loads(expected_content_str)
                actual_obj = json.loads(actual_content_str)
                assert actual_obj == expected_obj, \
                    f"Content JSON mismatch: Expected '{expected_content_str}', Got '{actual_content_str}'"
            except json.JSONDecodeError:
                # If either is not valid JSON, fall back to string comparison for this specific case
                assert actual_content_str == expected_content_str, \
                    f"Content string mismatch (JSON parse failed for one): Expected '{expected_content_str}', Got '{actual_content_str}'"
        else:
            assert actual_content_str == expected_content_str, \
                f"Content string mismatch: Expected '{expected_content_str}', Got '{actual_content_str}'"
        
        print(f"{test_name} PASSED.")
        return True
    except AssertionError as e:
        print(f"{test_name} FAILED: {e}")
        return False
    except Exception as e:
        print(f"{test_name} FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    test_results = []

    # Test Case 1
    tc1_input = ['为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。\n<tool_call>\n{"name": "travel_web_search", "arguments": {"keyword": "大理理想邦圣托里尼 最佳拍照时间段 取景角度", "page_size": 10, "page_num": 0}}\n</tool_call>']
    tc1_exp_action = 'natural_language_and_tool_call'
    tc1_exp_content = json.dumps({
        "natural_language": "为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。",
        "tool_call_content": '{"name": "travel_web_search", "arguments": {"keyword": "大理理想邦圣托里尼 最佳拍照时间段 取景角度", "page_size": 10, "page_num": 0}}'
    })
    test_results.append(run_test_case("Test Case 1: NL and Single Tool Call", tc1_input, tc1_exp_action, tc1_exp_content))

    # Test Case 2
    tc2_input = ['<tool_call>{"name": "calculator", "expression": "2+2"}</tool_call>']
    tc2_exp_action = 'tool_call'
    tc2_exp_content = '{"name": "calculator", "expression": "2+2"}'
    test_results.append(run_test_case("Test Case 2: Only Single Tool Call", tc2_input, tc2_exp_action, tc2_exp_content))

    # Test Case 3
    tc3_input = ['Hello, how are you today?']
    tc3_exp_action = 'natural_language_response'
    tc3_exp_content = 'Hello, how are you today?'
    test_results.append(run_test_case("Test Case 3: Only Natural Language", tc3_input, tc3_exp_action, tc3_exp_content))

    # Test Case 4
    tc4_input = ['']
    tc4_exp_action = None
    tc4_exp_content = ''
    test_results.append(run_test_case("Test Case 4: Empty Input String", tc4_input, tc4_exp_action, tc4_exp_content))

    # Test Case 5a: Non-String Input (None)
    tc5a_input = [None]
    tc5a_exp_action = None
    tc5a_exp_content = ''
    test_results.append(run_test_case("Test Case 5a: Non-String Input (None)", tc5a_input, tc5a_exp_action, tc5a_exp_content))

    # Test Case 5b: Non-String Input (int)
    tc5b_input = [123]
    tc5b_exp_action = None
    tc5b_exp_content = ''
    test_results.append(run_test_case("Test Case 5b: Non-String Input (int)", tc5b_input, tc5b_exp_action, tc5b_exp_content))

    # Test Case 6
    tc6_input = ['First do this:\n<tool_call>{"name": "tool1", "arg": "A"}</tool_call>\nThen do that:\n<tool_call>{"name": "tool2", "arg": "B"}</tool_call>']
    tc6_exp_action = 'natural_language_and_tool_call'
    tc6_exp_content = json.dumps({
        "natural_language": "First do this:",
        "tool_call_content": '[{"name": "tool1", "arg": "A"}, {"name": "tool2", "arg": "B"}]'
    })
    test_results.append(run_test_case("Test Case 6: NL and Multiple Tool Calls", tc6_input, tc6_exp_action, tc6_exp_content))

    # Test Case 7
    tc7_input = ['<tool_call>{"name": "intermediate_step"}</tool_call>\nThis is some trailing text.']
    tc7_exp_action = 'tool_call'
    tc7_exp_content = '{"name": "intermediate_step"}'
    test_results.append(run_test_case("Test Case 7: Tool Call with Trailing Text", tc7_input, tc7_exp_action, tc7_exp_content))

    # Test Case 8
    tc8_input = ['   ']
    tc8_exp_action = None 
    tc8_exp_content = ''
    test_results.append(run_test_case("Test Case 8: Prediction with only Whitespace", tc8_input, tc8_exp_action, tc8_exp_content))

    print("\n--- Summary ---")
    if all(test_results):
        print("All tests PASSED!")
    else:
        failed_count = len([r for r in test_results if not r])
        print(f"{failed_count} test(s) FAILED.")

if __name__ == "__main__":
    main()
