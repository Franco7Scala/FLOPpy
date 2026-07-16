import torch
import time
from torchvision.models import resnet18, resnet34, resnet50
from codecarbon import EmissionsTracker

# --- Import FLOPpy ---
from floppy import FLOPpyTracker

print("\n" + "=" * 90)
print(" 🔬 FLOPpy: Hardware Correlation & Analytical Validation Test (with BOPs)")
print("=" * 90)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Utilizziamo 3 modelli della famiglia ResNet con complessità crescente
models_to_test = {
    "ResNet-18": resnet18,
    "ResNet-34": resnet34,
    "ResNet-50": resnet50
}

# Parametri dell'esperimento
batch_size = 32
cycles = 20  # Numero di step per dare a CodeCarbon il tempo di registrare consumi stabili
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
dummy_labels = torch.randint(0, 1000, (batch_size,)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()

results = []

for name, model_class in models_to_test.items():
    print(f"\n[*] Esecuzione {name}...")
    model = model_class().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------------------------
    # 1. WARMUP DELLA GPU E MISURAZIONE HARDWARE (Time & Energy)
    # ---------------------------------------------------------
    # Warmup per stabilizzare la GPU e le metriche di tempo
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = loss_fn(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()  # Sincronizzazione fondamentale

    # Inizializziamo CodeCarbon spegnendo i log a schermo per pulizia
    tracker_hw = EmissionsTracker(log_level="error", project_name=f"hw_{name}")
    tracker_hw.start()

    start_time = time.time()

    # Eseguiamo il loop di training reale
    for _ in range(cycles):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = loss_fn(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()  # Sincronizzazione fondamentale
    end_time = time.time()
    emissions = tracker_hw.stop()  # Ritorna le emissioni/energia stimata

    execution_time_ms = (end_time - start_time) * 1000
    energy_consumed_wh = tracker_hw._total_energy.kWh * 1000  # Convertiamo in Watt-ora (Wh)

    # ---------------------------------------------------------
    # 2. MISURAZIONE ALGORITMICA E PRECISION-AWARE (FLOPpy)
    # ---------------------------------------------------------
    tracker_algo = FLOPpyTracker(run_name=f"algo_{name}")
    tracker_algo.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

    # Facciamo fare un solo step al tracker per estrarre la firma matematica esatta
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = loss_fn(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    tracker_algo.stop()
    report = tracker_algo.report()

    # Estraiamo FLOPs e BOPs e li moltiplichiamo per i cicli eseguiti
    total_flops = getattr(report, 'overall_flop', 0) * cycles
    total_bops = getattr(report, 'overall_bop', 0) * cycles

    results.append({
        "Model": name,
        "FLOPs": total_flops,
        "BOPs": total_bops,
        "Time (ms)": execution_time_ms,
        "Energy (Wh)": energy_consumed_wh
    })

    del model
    del optimizer
    torch.cuda.empty_cache()

# ============================================================
# 3. STAMPA TABELLA DI CORRELAZIONE
# ============================================================
print("\n" + "=" * 90)
print(f"{'MODELLO':<15} | {'FLOPs (Tera)':<15} | {'BOPs (Tera)':<15} | {'TEMPO (ms)':<15} | {'ENERGIA (mWh)':<15}")
print("-" * 90)

for res in results:
    flops_str = f"{res['FLOPs'] / 1e12:.3f} T"
    bops_str = f"{res['BOPs'] / 1e12:.3f} T"
    time_str = f"{res['Time (ms)']:.1f} ms"
    energy_str = f"{res['Energy (Wh)'] * 1000:.3f} mWh"
    print(f"{res['Model']:<15} | {flops_str:<15} | {bops_str:<15} | {time_str:<15} | {energy_str:<15}")

print("=" * 90)
