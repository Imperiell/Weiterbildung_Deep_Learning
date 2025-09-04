from tensorflow.python.keras.saving.saved_model.load import layers_module
from torch import nn

class InteractiveModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._name_to_module = {}
        self._build_name_mapping(module)

    """def _build_name_mapping(self, module, prefix=""):
        # Recursively add submodules to the name mapping
        # Each submodule is assigned to one key
        for name, sub in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self._name_to_module[full_name] = sub # Map layer names to submodules

            if len(list(sub.children())) > 0:
                self._build_name_mapping(sub, prefix=full_name)"""

    def _build_name_mapping(self, module, prefix=""):
        # Recursively add submodules to the name mapping
        # Each submodule is assigned to one key
        for name, sub in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer_type = type(sub).__name__
            full_name = f"{layer_type}_{full_name}"

            self._name_to_module[full_name] = sub  # Map layer names to submodules

            if len(list(sub.children())) > 0:
                self._build_name_mapping(sub, prefix=full_name)

    def set_freeze(self, freeze=True, layer_names=None):
        # Freeze or unfreeze all or selected layers
        if layer_names is None:
            # Target all submodules, if no specific layers are chosen
            targets = self._name_to_module.values()
        else:
            # Target only specified layers that exist
            targets = []
            for name in layer_names:
                if name in self._name_to_module:
                    targets.append(self._name_to_module[name])

        for module in targets:
            for param in module.parameters(recurse=True):
                param.requires_grad = not freeze # evtl besser explizit setzen

    def list_layer_names(self):
        # Return a list of all submodule names
        return list(self._name_to_module.keys())

    def list_module_names(self):
        return list(self._name_to_module.values())

    def before_forward(self, x):
        # Hook to modify input before main module
        return x

    def after_forward(self, x):
        # Hook to modigy output after main module
        return x

    def forward(self, x):
        # Apply hooks before and after the main module
        x = self.before_forward(x)
        x = self.module(x)
        x = self.after_forward(x)
        return x


class LatentLayerModule(nn.Module):
    def __init__(self, module: InteractiveModule):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)
