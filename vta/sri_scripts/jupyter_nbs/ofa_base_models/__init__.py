# Local shim that forwards to the external OFA_Obfs package under the same module name.
# It executes the external package __init__.py in this module's namespace and
# points __path__ to the external package directory so submodules resolve.
import sys
import os
import importlib.util

external_repo_root = "/home/srchand/Desktop/research/OFA_Obfs"
external_pkg_dir = os.path.join(external_repo_root, "ofa_base_models")
external_init = os.path.join(external_pkg_dir, "__init__.py")

if not os.path.isfile(external_init):
    raise ImportError(f"External ofa_base_models not found at {external_init}")

# Ensure the external repo root is searchable for any cross-package refs
if external_repo_root not in sys.path:
    sys.path.insert(0, external_repo_root)

# Prepare a spec that uses the current module name and external package path
spec = importlib.util.spec_from_file_location(
    __name__, external_init, submodule_search_locations=[external_pkg_dir]
)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load external ofa_base_models from {external_init}")

# Use the current module object; set its package file and path to the external ones
mod = sys.modules[__name__]
setattr(mod, "__file__", external_init)
setattr(mod, "__path__", [external_pkg_dir])  # mark as package pointing to external dir

# Execute the external __init__.py in-place so all symbols are defined here
spec.loader.exec_module(mod)
