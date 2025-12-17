from __future__ import annotations

from typing import Optional, Any

from tracker import Tracker


class FlopTracker:
    """
    Entry point pubblico della libreria.

    Uso:
        ft = FlopTracker(run_name="dp_cnn").torch_bind(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            device=device,
            epochs=10,
            log_per_batch=True,
            export_path="flop.csv",
            use_wandb=False,
        )
    """

    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name
        # FLOP del modello
        self._raw_flop: int = 0
        self._total_flop: float = 0.0
        # operazioni di preprocessing/tokenizer
        self._preproc_ops: int = 0
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
        """FLOP totali del modello"""
        return self._total_flop

    @property
    def total_preproc_ops(self) -> int:
        """
        Operazioni totali di preprocessing/tokenizer registrate dal Tracker.
        """
        return self._preproc_ops

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
        Questa funzione incapsula loop per batch/epoch e stampa i FLOP totali
        (totalmente trasparente per l'utente).
        """

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

                for xb, yb in train_loader:
                    if device is not None:
                        xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    output = model(xb)

                    if loss_fn is not None:
                        loss = loss_fn(output, yb)
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

        # STAMPA AUTOMATICA DEI FLOP TOTALI
        run_label = f"[{self.run_name}]" if self.run_name is not None else ""
        print(
            f"[FlopTracker{run_label}] FLOP totali (modello): {self._total_flop:.0f} "
            f"(raw: {self._raw_flop})"
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

        Assunzioni:
        - dataloader restituisce dict con chiavi tipo:
          "input_ids", "attention_mask", "labels", ecc.
        - se il model restituisce un oggetto con attributo .loss
          e optimizer non è None, facciamo training:
            loss = output.loss; loss.backward(); optimizer.step().
        - se optimizer è None, facciamo solo forward (inferenza).
        """

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
                    # spostiamo tutti i tensori su device
                    if device is not None:
                        batch = {
                            k: (v.to(device) if hasattr(v, "to") else v)
                            for k, v in batch.items()
                        }

                    if optimizer is not None:
                        optimizer.zero_grad()

                    output = model(**batch)

                    # se l'output ha 'loss' e c’è un optimizer, facciamo training
                    loss = getattr(output, "loss", None)
                    if loss is not None and optimizer is not None:
                        loss.backward()
                        optimizer.step()

                # log per epoch HF (se richiesto)
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
        mode: str = "fit",  # "fit", "predict" o "fit_predict"
        backend: str = "sklearn",
        log_per_call: bool = True,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> "FlopTracker":
        """
        Esegue fit/predict per un modello sklearn e traccia i FLOP.

        Esempi:
            ft = FlopTracker(run_name="lr").sklearn_bind(
                model=clf,
                X=X_train,
                y=y_train,
                mode="fit",
                log_per_call=True,
                export_path="flop_lr.csv",
            )

            ft = FlopTracker(run_name="knn").sklearn_bind(
                model=knn,
                X=X_test,
                mode="predict",
            )

        Ogni chiamata a fit/predict viene trattata come un "batch"
        nei log del backend sklearn.
        """

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

            # eseguiamo fit/predict secondo la modalità scelta
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
