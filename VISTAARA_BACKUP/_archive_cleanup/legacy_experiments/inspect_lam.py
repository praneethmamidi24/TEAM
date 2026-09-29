import inspect
import sen2sr
from sen2sr import lam
from sen2sr.xai import lam as xai_lam

print("=== sen2sr.lam dir ===")
print(dir(lam))

print("\n=== sen2sr.xai.lam dir ===")
print(dir(xai_lam))

for name in dir(xai_lam):
    obj = getattr(xai_lam, name)
    if inspect.isfunction(obj) or inspect.isclass(obj):
        try:
            sig = inspect.signature(obj)
            print(f"  {name}: {sig}")
            doc = inspect.getdoc(obj)
            if doc:
                first_line = doc.split('\n')[0]
                print(f"     doc: {first_line}")
        except Exception as e:
            print(f"  {name}: {e}")
