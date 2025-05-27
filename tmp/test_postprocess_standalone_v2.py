import re
import json # Not used by postprocess_predictions, but kept for potential test case definition needs.
import sys
from typing import Tuple, List, Optional # Ensure List is imported

# This is the "Fourth Revision" of postprocess_predictions
def postprocess_predictions_for_test(prediction: str) -> Tuple[str, List[str]]:
    content = ""
    actions_list = []

    if not isinstance(prediction, str):
        return ("", [])

    opening_tag = "<tool_call>"
    
    first_tag_idx = prediction.find(opening_tag)

    if first_tag_idx != -1:
        # At least one tool call tag is found
        content = prediction[:first_tag_idx].strip()
        
        # The part of the string to search for all tool calls starts from the first tag
        prediction_after_content = prediction[first_tag_idx:]
        
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(tool_call_pattern, prediction_after_content, re.DOTALL)
        for match in matches:
            actions_list.append(match.group(1).strip())
    else:
        # No <tool_call> tag found
        content = prediction.strip()
        # actions_list remains empty
            
    return content, actions_list

def run_test_case(test_name, prediction_input_str, expected_content_str, expected_actions_list):
    print(f"Running {test_name}...")
    try:
        actual_content_str, actual_actions_list = postprocess_predictions_for_test(prediction_input_str)

        assert actual_content_str == expected_content_str, \
            f"Content Part mismatch: Expected '{expected_content_str}', Got '{actual_content_str}'"
        
        assert actual_actions_list == expected_actions_list, \
            f"Actions List mismatch: Expected '{expected_actions_list}', Got '{actual_actions_list}'"
        
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

    # Test Case 1 (User's Example Format)
    tc1_input = "content_part\n<tool_call>action_str1</tool_call>\n<tool_call>action_str2</tool_call>"
    tc1_exp_content = "content_part"
    tc1_exp_actions = ["action_str1", "action_str2"]
    test_results.append(run_test_case("Test Case 1: User's Example", tc1_input, tc1_exp_content, tc1_exp_actions))

    # Test Case 2 (User's Original Example)
    tc2_input = '为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。\n<tool_call>\n{"name": "travel_web_search", "arguments": {"keyword": "大理理想邦圣托里尼 最佳拍照时间段 取景角度", "page_size": 10, "page_num": 0}}\n</tool_call>'
    tc2_exp_content = "为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。"
    tc2_exp_actions = ['{"name": "travel_web_search", "arguments": {"keyword": "大理理想邦圣托里尼 最佳拍照时间段 取景角度", "page_size": 10, "page_num": 0}}']
    test_results.append(run_test_case("Test Case 2: Original User Example", tc2_input, tc2_exp_content, tc2_exp_actions))

    # Test Case 3 (Only Single Tool Call)
    tc3_input = '<tool_call>{"name": "calculator", "expression": "2+2"}</tool_call>'
    tc3_exp_content = ""
    tc3_exp_actions = ['{"name": "calculator", "expression": "2+2"}']
    test_results.append(run_test_case("Test Case 3: Only Single Tool Call", tc3_input, tc3_exp_content, tc3_exp_actions))

    # Test Case 4 (Only Natural Language - No Tool Calls)
    tc4_input = 'Hello, how are you today?'
    tc4_exp_content = 'Hello, how are you today?'
    tc4_exp_actions = []
    test_results.append(run_test_case("Test Case 4: Only Natural Language", tc4_input, tc4_exp_content, tc4_exp_actions))

    # Test Case 5 (Empty Input String)
    tc5_input = ''
    tc5_exp_content = ""
    tc5_exp_actions = []
    test_results.append(run_test_case("Test Case 5: Empty Input String", tc5_input, tc5_exp_content, tc5_exp_actions))

    # Test Case 6a: Non-String Input (None)
    tc6a_input = None
    tc6a_exp_content = ""
    tc6a_exp_actions = []
    test_results.append(run_test_case("Test Case 6a: Non-String Input (None)", tc6a_input, tc6a_exp_content, tc6a_exp_actions))

    # Test Case 6b: Non-String Input (int)
    tc6b_input = 123
    tc6b_exp_content = ""
    tc6b_exp_actions = []
    test_results.append(run_test_case("Test Case 6b: Non-String Input (int)", tc6b_input, tc6b_exp_content, tc6b_exp_actions))
    
    # Test Case 7 (Natural Language and Multiple Tool Calls with varied spacing)
    tc7_input = 'Action plan:\n<tool_call>  tool_A_content  </tool_call>  Some text between. <tool_call>tool_B_content</tool_call>'
    tc7_exp_content = "Action plan:"
    tc7_exp_actions = ["tool_A_content", "tool_B_content"]
    test_results.append(run_test_case("Test Case 7: NL and Multiple Spaced Calls", tc7_input, tc7_exp_content, tc7_exp_actions))

    # Test Case 8 (Tool call with text after it, no preceding text)
    tc8_input = '<tool_call>{"name": "intermediate_step"}</tool_call>\nThis is some trailing text.'
    tc8_exp_content = ""
    tc8_exp_actions = ['{"name": "intermediate_step"}'] # Trailing text is ignored by finditer for the first tool call.
                                                       # And not captured as subsequent tool calls.
    test_results.append(run_test_case("Test Case 8: Tool Call with Trailing Text", tc8_input, tc8_exp_content, tc8_exp_actions))

    # Test Case 9 (Input with only whitespace)
    tc9_input = '   '
    tc9_exp_content = "" # After strip
    tc9_exp_actions = []
    test_results.append(run_test_case("Test Case 9: Prediction with only Whitespace", tc9_input, tc9_exp_content, tc9_exp_actions))
    
    # Test Case 10 (Malformed Tool Call - Missing Closing Tag)
    # The regex finditer will not find a match for an unclosed tag.
    tc10_input = 'Text before <tool_call>tool_content_no_close'
    tc10_exp_content = "Text before" 
    tc10_exp_actions = [] 
    test_results.append(run_test_case("Test Case 10: Malformed (No Closing Tag)", tc10_input, tc10_exp_content, tc10_exp_actions))

    # Test Case 11 (Tool Call with only whitespace content)
    tc11_input = 'NL part <tool_call>   </tool_call>'
    tc11_exp_content = "NL part"
    tc11_exp_actions = [""] # Content is "   ", which strips to ""
    test_results.append(run_test_case("Test Case 11: Tool Call with Whitespace Content", tc11_input, tc11_exp_content, tc11_exp_actions))

    # Test Case 12 (No text before first tool call)
    tc12_input = '<tool_call>action1</tool_call><tool_call>action2</tool_call>'
    tc12_exp_content = ""
    tc12_exp_actions = ["action1", "action2"]
    test_results.append(run_test_case("Test Case 12: No text before first tool call", tc12_input, tc12_exp_content, tc12_exp_actions))


    print("\n--- Summary ---")
    passed_count = sum(1 for r in test_results if r)
    total_count = len(test_results)
    if passed_count == total_count:
        print(f"All {total_count} tests PASSED!")
    else:
        print(f"{passed_count}/{total_count} tests PASSED. {total_count - passed_count} test(s) FAILED.")

if __name__ == "__main__":
    main()
