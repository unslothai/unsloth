import pytest
import subprocess
import sys
import os

# Add the module path to sys.path to import from unsloth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Payloads that attempt to inject shell commands through os.system calls
PAYLOADS = [
    # Exact exploit case: command injection with shell metacharacters
    "llama.cpp; echo 'INJECTED' > /tmp/exploit.txt",
    # Boundary case: input with spaces and quotes that could break command parsing
    "llama.cpp' || ls #",
    # Valid input: normal expected path
    "llama.cpp",
]

@pytest.mark.parametrize("payload", PAYLOADS)
def test_shell_command_injection_mitigation(payload):
    """Invariant: User-controlled input passed to os.system() must not execute arbitrary shell commands."""
    
    # Import the module here to avoid import issues if payloads cause early failures
    import unsloth.save
    
    # Mock os.system to capture and inspect the command being executed
    original_system = os.system
    captured_command = None
    
    def mock_system(cmd):
        nonlocal captured_command
        captured_command = cmd
        # Return success to avoid breaking the function flow
        return 0
    
    os.system = mock_system
    
    try:
        # Trigger the vulnerable code path by calling a function that uses os.system
        # We'll call a function that likely uses the vulnerable pattern
        # Since we can't directly call internal functions, we'll check if the module
        # has a function that would trigger the os.system call
        if hasattr(unsloth.save, 'save_model'):
            # This is a common function in save modules that might trigger the vulnerable code
            try:
                # Pass the payload as a potential path parameter
                unsloth.save.save_model(payload, "dummy_output")
            except Exception:
                pass  # We expect some calls to fail due to invalid inputs
        
        # Security assertion: The captured command must not contain the injected payload as executable code
        if captured_command is not None:
            # Check that the payload isn't directly executable in the shell command
            # The command should sanitize or properly handle the input
            assert ";" not in captured_command, f"Shell command injection via semicolon detected: {captured_command}"
            assert "||" not in captured_command, f"Shell command injection via OR operator detected: {captured_command}"
            assert "&&" not in captured_command, f"Shell command injection via AND operator detected: {captured_command}"
            assert "`" not in captured_command, f"Shell command injection via backticks detected: {captured_command}"
            assert "$(" not in captured_command, f"Shell command injection via command substitution detected: {captured_command}"
            
    finally:
        # Restore original os.system
        os.system = original_system