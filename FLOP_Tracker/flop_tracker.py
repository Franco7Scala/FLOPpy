from __future__ import annotations

from typing import Optional, Any


class FlopTracker:
    """
    Entry point pubblico della libreria.
    """

    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name

        # FLOP del modello
        self._raw_flop: int = 0
        self._total_flop: float = 0.0

        # operazioni di preprocessing/tokenizer
        self._preproc_ops: int = 0

        # FLOPs della loss 
        self._loss_flop: int = 0

        # Totale aggregato: FLOP modello + operazioni preprocessing/tokenizer
        self._total_operations: float = 0.0

        self._history: dict[str, Any] = {}

    # -------------------- PROPRIETÀ DI LETTURA -------------------- #

    @property
    def raw_flop(self) -> int:
        """FLOP (somma dei layer del modello, senza preprocessing)."""
        return self._raw_flop

    @property
    def total_flop(self) -> float:
        """FLOP totali del modello (include anche i FLOP della loss se abilitata)."""
        return self._total_flop

    @property
    def total_preproc_ops(self) -> int:
        """Operazioni totali di preprocessing/tokenizer registrate dal Tracker."""
        return self._preproc_ops

    @property
    def total_loss_flop(self) -> int:
        """FLOP totali attribuiti alla loss."""
        return self._loss_flop

    @property
    def total_operations(self) -> float:
        """
        Metrica aggregata:
            FLOP del modello + operazioni di preprocessing/tokenizer.
        """
        return self._total_operations

    @property
    def history(self) -> dict[str, Any]:
        return self._history

    # -------------------- STIMA LOSS FLOPs -------------------- #

    def _estimate_loss_flop(self, loss_fn, preds, targets) -> int:
        """
        Stima teorica e hardware-agnostica dei FLOP della loss (forward). 
        """
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return 0

        if loss_fn is None or preds is None:
            return 0

        if not hasattr(preds, "numel"):
            return 0

        N = int(preds.numel())

        # --- Regression losses ---
        if isinstance(loss_fn, nn.MSELoss):
            # (p-t)^2: sub + mul; reduce: (N-1) add
            return max(0, 2 * N + (N - 1))

        if isinstance(loss_fn, nn.L1Loss):
            # |p-t|: sub + abs; reduce
            return max(0, 2 * N + (N - 1))

        # --- Probabilistic/classification losses ---
        if isinstance(loss_fn, nn.NLLLoss):
            # input log-prob (B,C), target (B,)
            # gather B + reduce (B-1)
            B = int(preds.shape[0]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            return max(0, B + (B - 1))

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # CE = LogSoftmax + NLL
            B = int(preds.shape[0]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            C = int(preds.shape[-1]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            # logsoftmax approx per sample: ~4C
            return max(0, B * (4 * C) + B + (B - 1))

        if isinstance(loss_fn, nn.KLDivLoss):
            # approx 3 ops/elem + reduce
            return max(0, 3 * N + (N - 1))

        if isinstance(loss_fn, nn.BCELoss):
            # approx 6 ops/elem + reduce
            return max(0, 6 * N + (N - 1))

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            # sigmoid + BCE approx ~10 ops/elem + reduce
            return max(0, 10 * N + (N - 1))

        if isinstance(loss_fn, (nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss)):
            # difficile stimare senza vedere le triple; fallback conservativo
            return max(0, 5 * N)

        # fallback (se non riconosciuta)
        return 0

    # -------------------- TORCH BIND -------------------- #

    def torch_bind(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        device: Optional[str] = None,
        *,
        epochs: int = 1,
        backend: str = "torch",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopTracker":
        """
        Esegue training + tracking FLOP per un modello PyTorch.
        Include (opzionalmente) anche una stima dei FLOP della loss.
        """

        from tracker import Tracker

        if device is not None:
            model.to(device)

        # reset contatori
        self._loss_flop = 0

        with Tracker(
            model=model,
            backend=backend,
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            for epoch in range(epochs):
                if hasattr(tr.backend, "set_epoch"):
                    tr.backend.set_epoch(epoch)

                for xb, yb in train_loader:
                    if device is not None:
                        xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    output = model(xb)

                    if loss_fn is not None:
                        loss = loss_fn(output, yb)

                        # stima FLOP loss (forward) e aggiunta al batch corrente
                        loss_flop = self._estimate_loss_flop(loss_fn, output, yb)
                        self._loss_flop += int(loss_flop)

                        if hasattr(tr.backend, "add_extra_flop"):
                            tr.backend.add_extra_flop(int(loss_flop))

                        loss.backward()
                        optimizer.step()

                # log per epoch (se abilitato)
                if tr.logger is not None and hasattr(tr.logger, "log_epoch"):
                    tr.logger.log_epoch(
                        epoch=epoch,
                        flop=tr.total_flop,
                        cumulative_flop=tr.total_flop,
                    )

            # metriche dal Tracker
            self._raw_flop = tr.total_flop
            self._total_flop = float(tr.total_flop)
            self._preproc_ops = tr.total_preproc_ops
            self._total_operations = tr.total_operations

            self._history["backend"] = backend
            self._history["export_path"] = export_path
            self._history["use_wandb"] = use_wandb
            self._history["wandb_project"] = wandb_project
            self._history["epochs"] = epochs
            self._history["loss_flop"] = self._loss_flop
            self._history["loss_name"] = loss_fn.__class__.__name__ if loss_fn is not None else None

        # STAMPA AUTOMATICA DEI FLOP TOTALI
        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        print(
            f"[FlopTracker{run_label}] FLOP totali (modello): {self._total_flop:.0f} "
            f"(raw: {self._raw_flop})"
        )

        if loss_fn is not None:
            print(
                f"[FlopTracker{run_label}] FLOP loss (forward): {self._loss_flop}"
            )

        if self._preproc_ops > 0:
            print(
                f"[FlopTracker{run_label}] Operazioni preprocessing/tokenizer: "
                f"{self._preproc_ops} (totale aggregato: {self._total_operations:.0f})"
            )

        return self

    # -------------------- HF BIND -------------------- #

    def hf_bind(
        self,
        model,
        dataloader,
        optimizer=None,
        device: Optional[str] = None,
        *,
        epochs: int = 1,
        backend: str = "hf",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopTracker":
        """
        Esegue training / inferenza per un modello HuggingFace (transformers),
        stimando i FLOP del modello (e se abilitato il wrapper, le operazioni di tokenizer).
        Nota: qui non stimiamo FLOP della loss separatamente, perché in HF la loss è
        generalmente interna al forward del modello e quindi già inclusa negli hook.
        """

        from tracker import Tracker

        if device is not None:
            model.to(device)

        with Tracker(
            model=model,
            backend=backend,
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            for epoch in range(epochs):
                if hasattr(tr.backend, "set_epoch"):
                    tr.backend.set_epoch(epoch)

                for batch in dataloader:
                    if device is not None:
                        batch = {
                            k: (v.to(device) if hasattr(v, "to") else v)
                            for k, v in batch.items()
                        }

                    if optimizer is not None:
                        optimizer.zero_grad()

                    output = model(**batch)

                    loss = getattr(output, "loss", None)
                    if loss is not None and optimizer is not None:
                        loss.backward()
                        optimizer.step()

                if tr.logger is not None and hasattr(tr.logger, "log_epoch"):
                    tr.logger.log_epoch(
                        epoch=epoch,
                        flop=tr.total_flop,
                        cumulative_flop=tr.total_flop,
                    )

            self._raw_flop = tr.total_flop
            self._total_flop = float(tr.total_flop)
            self._preproc_ops = tr.total_preproc_ops
            self._total_operations = tr.total_operations

            self._history["backend"] = backend
            self._history["export_path"] = export_path
            self._history["use_wandb"] = use_wandb
            self._history["wandb_project"] = wandb_project
            self._history["epochs"] = epochs
            self._history["hf_mode"] = "train" if optimizer is not None else "inference"

        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        mode_label = "train" if optimizer is not None else "inference"
        print(
            f"[FlopTracker{run_label}] FLOP totali (HF, mode={mode_label}): "
            f"{self._total_flop:.0f} (raw: {self._raw_flop})"
        )
        if self._preproc_ops > 0:
            print(
                f"[FlopTracker{run_label}] Operazioni preprocessing/tokenizer: "
                f"{self._preproc_ops} (totale aggregato: {self._total_operations:.0f})"
            )

        return self

    # -------------------- SKLEARN BIND -------------------- #

    def sklearn_bind(
        self,
        model,
        X,
        y=None,
        *,
        mode: str = "fit",
        backend: str = "sklearn",
        log_per_call: bool = True,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopTracker":
        """
        Esegue fit/predict per un modello sklearn e traccia i FLOP.
        """

        from tracker import Tracker

        log_per_batch = log_per_call
        log_per_epoch = False

        with Tracker(
            model=model,
            backend=backend,
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            if mode == "fit":
                model.fit(X, y)
            elif mode == "predict":
                _ = model.predict(X)
            elif mode == "fit_predict":
                model.fit(X, y)
                _ = model.predict(X)
            else:
                raise ValueError(f"Modo sklearn_bind non supportato: {mode}")

            self._raw_flop = tr.total_flop
            self._total_flop = float(self._raw_flop)
            self._preproc_ops = tr.total_preproc_ops
            self._total_operations = tr.total_operations

            self._history["backend"] = backend
            self._history["export_path"] = export_path
            self._history["use_wandb"] = use_wandb
            self._history["wandb_project"] = wandb_project
            self._history["mode"] = mode

        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        print(
            f"[FlopTracker{run_label}] FLOP totali (sklearn, mode={mode}): "
            f"{self._total_flop:.0f} (raw: {self._raw_flop})"
        )
        if self._preproc_ops > 0:
            print(
                f"[FlopTracker{run_label}] Operazioni preprocessing/tokenizer: "
                f"{self._preproc_ops} (totale aggregato: {self._total_operations:.0f})"
            )

        return self
