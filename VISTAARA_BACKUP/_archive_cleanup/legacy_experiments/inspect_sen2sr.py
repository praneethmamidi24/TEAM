import sen2sr
import inspect

print("=== sen2sr dir ===")
print(dir(sen2sr))

print("\n=== sen2sr submodules / files ===")
import os
sen2sr_path = os.path.dirname(sen2sr.__file__)
for root, dirs, files in os.walk(sen2sr_path):
    rel = os.path.relpath(root, sen2sr_path)
    print(f"{rel}: files={files}")

try:
    from sen2sr import lam
    print("Found sen2sr.lam")
except ImportError as e:
    print("No sen2sr.lam:", e)

try:
    from sen2sr import robustness
    print("Found sen2sr.robustness")
except ImportError as e:
    print("No sen2sr.robustness:", e)

try:
    from sen2sr import attention
    print("Found sen2sr.attention")
except ImportError as e:
    print("No sen2sr.attention:", e)
