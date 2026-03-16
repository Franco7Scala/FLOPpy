import torch

from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode


class UniversalFlopCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.flops = 0

    def _get_numel(self, obj):
        """Calcola in modo sicuro il numero totale di elementi, gestendo anche liste di tensori."""
        if isinstance(obj, torch.Tensor):
            return obj.numel()
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_numel(x) for x in obj)
        return 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Esegue la reale operazione di PyTorch e ne cattura l"output
        out = func(*args, **kwargs)

        # Convertiamo in stringa e calcoliamo sia gli elementi in entrata che in uscita
        func_str = str(func).lower()
        out_elements = self._get_numel(out)
        in_elements = self._get_numel(args[0]) if len(args) > 0 else 0

        # --- REGOLE DI CALCOLO DEI FLOPS ---

        # 1. OPTIMIZER: Operazioni Foreach e Fused
        # Usiamo "in_elements" perché l"output di queste op in-place è spesso None
        if "foreach" in func_str:
            if "addcdiv" in func_str or "addcmul" in func_str:
                self.flops += in_elements * 3
            else:
                self.flops += in_elements

        elif "aten.addcdiv" in func_str or "aten.addcmul" in func_str:
            self.flops += in_elements * 3

        # 2. OPERAZIONI ELEMENT-WISE (Standard e In-Place)
        elif any(func_str.startswith(f"aten.{op}") for op in ["add", "sub", "mul", "div", "exp", "log", "pow", "neg", "abs", "relu", "sigmoid", "tanh"]):
            # Se out_elements è 0 (operazione in-place), ripieghiamo su in_elements
            elements = out_elements if out_elements > 0 else in_elements
            self.flops += elements

        # 2. OPERAZIONI ELEMENT-WISE (Standard e In-Place)
        # Usiamo startswith per catturare "aten.add", "aten.add_", "aten.add.Tensor" ecc.
        elif any(func_str.startswith(f"aten.{op}") for op in ["add", "sub", "mul", "div", "exp", "log", "pow", "neg", "abs", "relu", "sigmoid", "tanh"]):
            self.flops += out_elements

        # 3. RIDUZIONI (Somme, Medie, Norme)
        elif any(op in func_str for op in ["aten.sum", "aten.mean", "aten.max", "aten.min", "aten.norm", "aten.var"]):
            in_elements = self._get_numel(args[0]) if len(args) > 0 else 0
            self.flops += in_elements
            # Le medie e le varianze richiedono una divisione finale extra
            if "aten.mean" in func_str or "aten.var" in func_str:
                self.flops += out_elements

        # 4. ALGEBRA LINEARE BASE (Moltiplicazioni tra matrici)
        elif "aten.mm" in func_str or "aten.addmm" in func_str:
            mat1 = args[1] if "addmm" in func_str else args[0]
            mat2 = args[2] if "addmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 2:
                m, k = mat1.shape
                k2, n = mat2.shape
                self.flops += 2 * m * n * k

        elif "aten.bmm" in func_str or "aten.baddbmm" in func_str:
            mat1 = args[1] if "baddbmm" in func_str else args[0]
            mat2 = args[2] if "baddbmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 3:
                b, m, k = mat1.shape
                b2, k2, n = mat2.shape
                self.flops += 2 * b * m * n * k

        elif "aten.matmul" in func_str:
            mat1, mat2 = args[0], args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() >= 2 and mat2.dim() >= 2:
                m, k = mat1.shape[-2], mat1.shape[-1]
                k2, n = mat2.shape[-2], mat2.shape[-1]
                batch_elements = mat1.numel() // (m * k)
                self.flops += 2 * batch_elements * m * n * k

        # 5. CONVOLUZIONI
        elif "aten.convolution" in func_str or "aten.conv" in func_str:
            weight_t = args[1]
            if isinstance(weight_t, torch.Tensor):
                # FLOPs = 2 * elementi_nel_kernel * elementi_in_output
                flops_per_element = 2 * (weight_t.numel() / weight_t.shape[0])
                self.flops += int(out_elements * flops_per_element)

        # 6. ALGEBRA LINEARE AVANZATA (es. Optimizer del 2nd Ordine)
        elif "aten.inverse" in func_str or "aten.linalg_inv" in func_str:
            mat = args[0]
            if isinstance(mat, torch.Tensor) and mat.dim() >= 2:
                n = mat.shape[-1]
                batch_elements = mat.numel() // (n * n)
                self.flops += batch_elements * 2 * (n ** 3)

        # 7. BACKEND LOSS STANDARD
        elif "aten._log_softmax" in func_str:
            in_elements = self._get_numel(args[0]) if len(args) > 0 else 0
            self.flops += in_elements * 3
        elif "aten.nll_loss" in func_str:
            self.flops += out_elements

        return out


def custom_loss(predictions, targets):
    diff = (predictions - targets) ** 2
    penalization = torch.exp(diff)
    return torch.sum(penalization)


if __name__ == "__main__":
    batch_size = 10
    input_features = 10
    num_classes = 10

    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, num_classes)
    )

    model.train()

    x = torch.randn(batch_size, input_features)
    targets_custom = torch.randn(batch_size, num_classes)
    targets_ce = torch.randint(0, num_classes, (batch_size,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #TODO esempio di utilizzo ma da spostare dentro gli hook per rendere trasparente l'utilizzo del contatore di FLOP
    logits = model(x)
    with UniversalFlopCounter() as loss_counter:
        loss_val = custom_loss(logits, targets_custom)
        print(f"custom loss FLOPs after loss computation: {loss_counter.flops}")
        loss_val.backward()
        print(f"custom loss FLOPs after backward: {loss_counter.flops}")

    logits = model(x)
    with UniversalFlopCounter() as loss_counter:
        loss_val = criterion(logits, targets_custom)
        print(f"CE loss FLOPs after loss computation: {loss_counter.flops}")
        loss_val.backward()
        print(f"CE loss FLOPs after backward: {loss_counter.flops}")

    with UniversalFlopCounter() as opt_counter:
        optimizer.step()
        print(f"opt FLOPs: {opt_counter.flops}")
