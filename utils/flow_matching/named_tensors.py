import torch

class NamedTensor:
    def __init__(self, tensor: torch.Tensor, names: tuple[str]):
        if tensor.ndim != len(names):
            raise ValueError(f"Expected {len(names)} dimensions, got {tensor.ndim}")
        self.tensor = tensor
        self.names = names

    def __repr__(self):
        return f"NamedTensor(names={self.names}, shape={tuple(self.tensor.shape)})"

    def __getattr__(self, attr):
        """
        Unbekannte Attribute werden an den inneren Tensor weitergeleitet.
        Ist das Ergebnis ein Tensor, wrappe es neu.
        """
        t_attr = getattr(self.tensor, attr)

        if callable(t_attr):
            def wrapper(*args, **kwargs):
                result = t_attr(*args, **kwargs)

                # Fallback: Namen beibehalten
                if isinstance(result, torch.Tensor):
                    return NamedTensor(result, self.names[:result.ndim])
                return result

            return wrapper
        else:
            return t_attr

    # Zugriff auf rohe Tensoren
    def raw(self):
        return self.tensor

    # Hilfsfunktionen für Namen
    def dim_index(self, name: str) -> int:
        return self.names.index(name)

    def rename(self, **rename_map):
        new_names = tuple(rename_map.get(n, n) for n in self.names)
        return NamedTensor(self.tensor, new_names)

    def permute(self, *order):
        # permute nach Namen oder Indizes
        if all(isinstance(x, str) for x in order):
            indices = [self.dim_index(n) for n in order]
            new_names = order
        else:
            indices = order
            new_names = tuple(self.names[i] for i in indices)

        result = self.tensor.permute(*indices)
        return NamedTensor(result, new_names)

    def mean(self, dim=None, *args, **kwargs):
        # Mean mit Namen
        if isinstance(dim, str):
            dim = self.dim_index(dim)
        result = self.tensor.mean(dim=dim, *args, **kwargs)
        if dim is None:
            return NamedTensor(result, ())
        else:
            new_names = self.names[:dim] + self.names[dim+1:]
            return NamedTensor(result, new_names)

    # Operatoren
    def __add__(self, other):
        if isinstance(other, NamedTensor):
            # naive Variante: gleiche Reihenfolge annehmen
            return NamedTensor(self.tensor + other.tensor, self.names)
        return NamedTensor(self.tensor + other, self.names)

    def __sub__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor - other.tensor, self.names)
        return NamedTensor(self.tensor - other, self.names)

    def __mul__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor * other.tensor, self.names)
        return NamedTensor(self.tensor * other, self.names)

    def __truediv__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor / other.tensor, self.names)
        return NamedTensor(self.tensor / other, self.names)

    def __getitem__(self, item):
        result = self.tensor[item]
        new_names = self.names[:result.ndim]
        return NamedTensor(result, new_names)

"""
# --- Beispiel ---
x = NamedTensor(torch.randn(10, 20, 32), ("batch", "time", "features"))
print(x)

# Mittelwert über Dimension
m = x.mean("time")
print(m)

# Permute nach Namen
p = x.permute("features", "batch", "time")
print(p)

# Addition mit normalem Tensor
y = torch.randn(10, 20, 32)
z = x + y
print(z)
"""