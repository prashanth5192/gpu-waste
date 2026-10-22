import os

# Importing gpu_waste_meter auto-starts in-process recording. Keep the test
# runner itself un-instrumented; tests that need it run in subprocesses and
# clear this variable explicitly.
os.environ["GWM_DISABLE"] = "1"
