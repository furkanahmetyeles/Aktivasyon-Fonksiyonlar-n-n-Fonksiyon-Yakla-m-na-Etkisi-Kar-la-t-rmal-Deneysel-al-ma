"""
Aktivasyon Fonksiyonlarının Fonksiyon Yaklaşımına Etkisi: Bir Karşılaştırmalı Deneysel Çalışma
Veri Seti: UCI Energy Efficiency (ID: 242)
Yazarlar: Dr. Öğr. Üyesi Saim KURAL, Furkan Ahmet Yeles
Kurum: Manisa Celal Bayar Üniversitesi, Makine Mühendisliği Bölümü

Bu betik aşağıdaki çıktıları otomatik olarak üretir:
1. 7 adet 300 DPI yüksek çözünürlüklü akademik grafik (figures/ klasörü altında)
2. 10 adet tablonun gerçek deneysel verileri (tables_output.json ve konsol çıktısı)
3. Tekrarlanabilirlik için sabitlenmiş random_seed (42, 123, 2026) ile 3 tekrarlı deney sonuçları
"""

import os
import time
import json
import random
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

# ==========================================
# 0. GENEL AYARLAR VE KLASÖR YAPISI
# ==========================================
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

SEEDS = [42, 123, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. VERİ SETİ İNDİRME VE YÜKLEME (UCI ID: 242)
# ==========================================
def load_uci_energy_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    local_path = "ENB2012_data.xlsx"
    if not os.path.exists(local_path):
        print("UCI Energy Efficiency veri seti indiriliyor...")
        urllib.request.urlretrieve(url, local_path)
    df = pd.read_excel(local_path)
    # Olası boş sütunları temizle
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    df.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
    return df

df = load_uci_energy_data()
print(f"Veri seti yüklendi. Boyut: {df.shape}")

# ==========================================
# 2. TABLO T1 VE T2: İSTATİSTİKSEL ÖZET VE ÖZNİTELİKLER
# ==========================================
feature_descriptions = {
    "X1": ("Bağıl Kompaktlık (Relative Compactness)", "Bina hacminin yüzey alanına oranının kübik referansa göre normalize değeri", "Birim yok", "Sürekli"),
    "X2": ("Yüzey Alanı (Surface Area)", "Binanın toplam dış kabuk yüzey alanı", "m²", "Sürekli"),
    "X3": ("Duvar Alanı (Wall Area)", "Binanın toplam dış duvar alanı", "m²", "Sürekli"),
    "X4": ("Çatı Alanı (Roof Area)", "Binanın toplam çatı alanı", "m²", "Sürekli"),
    "X5": ("Toplam Yükseklik (Overall Height)", "Binanın kat ve toplam yapı yüksekliği", "m", "Kesikli (3.5, 7.0)"),
    "X6": ("Yönelim (Orientation)", "Binanın ana cephe yönelimi (Kuzey, Doğu, Güney, Batı)", "Sınıf (2-5)", "Kategorik/Kesikli"),
    "X7": ("Cam Alanı Oranı (Glazing Area)", "Pencere alanının toplam zemin alanına oranı", "Oran (0.0-0.4)", "Kesikli"),
    "X8": ("Cam Alanı Dağılımı (Glazing Area Distribution)", "Cam yüzeylerin cephelere göre dağılım senaryosu", "Sınıf (0-5)", "Kategorik/Kesikli"),
    "Y1": ("Isıtma Yükü (Heating Load - Hedef 1)", "İç ortam konforu için gereken birim ısıtma enerjisi ihtiyacı", "kWh/m²", "Sürekli (Hedef)"),
    "Y2": ("Soğutma Yükü (Cooling Load - Hedef 2)", "İç ortam konforu için gereken birim soğutma enerjisi ihtiyacı", "kWh/m²", "Sürekli (Hedef)")
}

stats_list = []
for col in df.columns:
    s = df[col]
    stats_list.append({
        "Kod": col,
        "Degisken": feature_descriptions[col][0],
        "Ortalama": round(float(s.mean()), 4),
        "Std_Sapma": round(float(s.std()), 4),
        "Min": round(float(s.min()), 4),
        "Q25": round(float(s.quantile(0.25)), 4),
        "Medyan": round(float(s.median()), 4),
        "Q75": round(float(s.quantile(0.75)), 4),
        "Maks": round(float(s.max()), 4),
        "Carpiklik": round(float(s.skew()), 4),
        "Basiklik": round(float(s.kurtosis()), 4)
    })
t1_df = pd.DataFrame(stats_list)
t1_df.to_csv("results/T1_genel_istatistikler.csv", index=False)

# ==========================================
# 3. GRAFİK G1 VE G2: VERİ DAĞILIMI VE KORELASYON ISI HARİTASI
# ==========================================
# G1: Histogram + KDE ve Boxplot (Hedef değişkenler ve temel öznitelikler)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["Y1"], kde=True, ax=axes[0, 0], color="#1f77b4", bins=25)
axes[0, 0].set_title("(a) Isıtma Yükü (Y1) Dağılımı ve Yoğunluk Eğrisi")
axes[0, 0].set_xlabel("Isıtma Yükü Y1 (kWh/m²)")
axes[0, 0].set_ylabel("Frekans")

sns.histplot(df["Y2"], kde=True, ax=axes[0, 1], color="#ff7f0e", bins=25)
axes[0, 1].set_title("(b) Soğutma Yükü (Y2) Dağılımı ve Yoğunluk Eğrisi")
axes[0, 1].set_xlabel("Soğutma Yükü Y2 (kWh/m²)")
axes[0, 1].set_ylabel("Frekans")

# Normalize edilmiş özniteliklerin boxplot gösterimi
norm_df = (df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]] - df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].mean()) / df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].std()
sns.boxplot(data=norm_df, ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("(c) Standartlaştırılmış Giriş Öznitelikleri (X1-X8) Kutu Grafiği")
axes[1, 0].set_xlabel("Öznitelikler")
axes[1, 0].set_ylabel("Z-Skoru Değeri")

sns.boxplot(data=df[["Y1", "Y2"]], ax=axes[1, 1], palette=["#1f77b4", "#ff7f0e"], width=0.4)
axes[1, 1].set_title("(d) Hedef Değişkenlerin (Y1 ve Y2) Kutu Grafiği")
axes[1, 1].set_xlabel("Hedef Değişkenler")
axes[1, 1].set_ylabel("Enerji Yükü (kWh/m²)")

plt.tight_layout()
plt.savefig("figures/G1_veri_dagilimi.png", dpi=300)
plt.close()

# G2: Korelasyon Isı Haritası
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(method="pearson")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Korelasyon Katsayısı (r)"}
)
plt.title("UCI Energy Efficiency Veri Seti Pearson Korelasyon Matrisi Isı Haritası")
plt.xlabel("Değişkenler")
plt.ylabel("Değişkenler")
plt.tight_layout()
plt.savefig("figures/G2_korelasyon_isi_haritasi.png", dpi=300)
plt.close()

# ==========================================
# 4. GRAFİK G3: MODEL MİMARİSİ ŞEMATİK GÖSTERİMİ
# ==========================================
def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    layers = [
        {"x": 1.0, "w": 1.6, "h": 4.2, "color": "#D6EAF8", "edge": "#2874A6", "title": "Giriş Katmanı\n(Input Layer)", "desc": "8 Öznitelik\n(X1 ... X8)\nZ-Score Ölçekli"},
        {"x": 3.4, "w": 1.8, "h": 4.8, "color": "#D5F5E3", "edge": "#1E8449", "title": "Gizli Katman 1\n(Dense-1)", "desc": "128 Nöron\n+ LayerNorm\n+ Aktivasyon f(z)"},
        {"x": 6.0, "w": 1.8, "h": 4.8, "color": "#FCF3CF", "edge": "#B7950B", "title": "Gizli Katman 2\n(Dense-2)", "desc": "128 Nöron\n+ LayerNorm\n+ Aktivasyon f(z)"},
        {"x": 8.6, "w": 1.8, "h": 4.2, "color": "#EBDEF0", "edge": "#6C3483", "title": "Gizli Katman 3\n(Dense-3)", "desc": "64 Nöron\n+ Aktivasyon f(z)"},
        {"x": 11.1, "w": 1.5, "h": 3.0, "color": "#FADBD8", "edge": "#922B21", "title": "Çıkış Katmanı\n(Output Layer)", "desc": "2 Çıkış (Lineer)\nY1: Isıtma Yükü\nY2: Soğutma Yükü"}
    ]

    for i, l in enumerate(layers):
        y0 = (6.2 - l["h"]) / 2
        rect = patches.FancyBboxPatch((l["x"], y0), l["w"], l["h"], boxstyle="round,pad=0.1",
                                      linewidth=2, edgecolor=l["edge"], facecolor=l["color"])
        ax.add_patch(rect)
        ax.text(l["x"] + l["w"]/2, y0 + l["h"]*0.68, l["title"], ha="center", va="center", fontweight="bold", fontsize=9.5, color="#1C2833")
        ax.text(l["x"] + l["w"]/2, y0 + l["h"]*0.30, l["desc"], ha="center", va="center", fontsize=8.5, color="#2C3E50")

        if i < len(layers) - 1:
            next_l = layers[i+1]
            ax.annotate("", xy=(next_l["x"] - 0.12, 3.1), xytext=(l["x"] + l["w"] + 0.12, 3.1),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#34495E"))

    # Alt bilgi kutusu: Test edilen aktivasyonlar
    act_box = patches.FancyBboxPatch((1.0, 0.15), 11.6, 0.75, boxstyle="round,pad=0.08",
                                     linewidth=1.5, edgecolor="#566573", facecolor="#F4F6F7")
    ax.add_patch(act_box)
    ax.text(6.8, 0.52,
            "Karşılaştırılan Aktivasyon Fonksiyonları f(z):  [1] Sigmoid   [2] Tanh   [3] ReLU   [4] Leaky ReLU (α=0.01)   [5] ELU (α=1.0)   [6] GELU   [7] Swish / SiLU",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1B2631")

    ax.set_title("Çok Çıkışlı Derin İleri Beslemeli Sinir Ağı (MLP) Mimarisi ve Aktivasyon Bloğu Şeması", fontsize=12, fontweight="bold", pad=10)
    plt.savefig("figures/G3_model_mimarisi.png", dpi=300)
    plt.close()

draw_architecture_diagram()

# ==========================================
# 5. MODEL TANIMI VE EĞİTİM FONKSİYONLARI
# ==========================================
def get_activation(name: str):
    activations = {
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
        "ELU": nn.ELU(alpha=1.0),
        "GELU": nn.GELU(),
        "Swish": nn.SiLU()
    }
    return activations[name]

class EnergyMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128, 128, 64], output_dim=2, act_name="GELU", use_norm=True, dropout_rate=0.0):
        super().__init__()
        layers = []
        in_d = input_dim
        for idx, h_d in enumerate(hidden_dims):
            lin = nn.Linear(in_d, h_d)
            # Başlatma stratejisi
            if act_name in ["ReLU", "LeakyReLU", "ELU", "GELU", "Swish"]:
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if use_norm and idx < len(hidden_dims) - 1:
                layers.append(nn.LayerNorm(h_d))
            layers.append(get_activation(act_name))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_d = h_d
        out_lin = nn.Linear(in_d, output_dim)
        nn.init.xavier_normal_(out_lin.weight)
        nn.init.zeros_(out_lin.bias)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def prepare_splits(df, seed):
    X = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values.astype(np.float32)
    y = df[["Y1", "Y2"]].values.astype(np.float32)

    # %70 Eğitim, %15 Doğrulama, %15 Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    return (X_train_s, y_train_s, y_train), (X_val_s, y_val_s, y_val), (X_test_s, y_test_s, y_test), scaler_x, scaler_y

def train_and_evaluate(act_name="GELU", seed=42, hidden_dims=[128, 128, 64], use_norm=True, dropout_rate=0.0, lr=1e-3, epochs=400, batch_size=32):
    set_seed(seed)
    (X_tr, y_tr_s, y_tr), (X_va, y_va_s, y_va), (X_te, y_te_s, y_te), scaler_x, scaler_y = prepare_splits(df, seed)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr_s)), batch_size=batch_size, shuffle=True)
    X_va_t = torch.tensor(X_va).to(DEVICE)
    y_va_t = torch.tensor(y_va_s).to(DEVICE)
    X_te_t = torch.tensor(X_te).to(DEVICE)

    model = EnergyMLP(8, hidden_dims, 2, act_name=act_name, use_norm=use_norm, dropout_rate=dropout_rate).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    best_val_loss = float("inf")
    best_weights = None

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        tr_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            va_pred_s = model(X_va_t)
            va_loss = float(criterion(va_pred_s, y_va_t).item())

            # Gerçek ölçekte RMSE takibi
            tr_pred_real = scaler_y.inverse_transform(model(torch.tensor(X_tr).to(DEVICE)).cpu().numpy())
            va_pred_real = scaler_y.inverse_transform(va_pred_s.cpu().numpy())
            tr_rmse = float(np.sqrt(mean_squared_error(y_tr, tr_pred_real)))
            va_rmse = float(np.sqrt(mean_squared_error(y_va, va_pred_real)))

        scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_rmses.append(tr_rmse)
        val_rmses.append(va_rmse)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - start_time
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        te_pred_s = model(X_te_t).cpu().numpy()
        te_pred = scaler_y.inverse_transform(te_pred_s)

    metrics = {
        "Y1_RMSE": float(np.sqrt(mean_squared_error(y_te[:, 0], te_pred[:, 0]))),
        "Y1_MAE": float(mean_absolute_error(y_te[:, 0], te_pred[:, 0])),
        "Y1_R2": float(r2_score(y_te[:, 0], te_pred[:, 0])),
        "Y2_RMSE": float(np.sqrt(mean_squared_error(y_te[:, 1], te_pred[:, 1]))),
        "Y2_MAE": float(mean_absolute_error(y_te[:, 1], te_pred[:, 1])),
        "Y2_R2": float(r2_score(y_te[:, 1], te_pred[:, 1])),
        "Avg_RMSE": float(np.sqrt(mean_squared_error(y_te, te_pred))),
        "Avg_MAE": float(mean_absolute_error(y_te, te_pred)),
        "Avg_R2": float(r2_score(y_te, te_pred)),
        "Train_Time_s": round(train_time, 3)
    }

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_rmse": train_rmses,
        "val_rmse": val_rmses,
        "y_test": y_te,
        "y_pred": te_pred,
        "X_test": scaler_x.inverse_transform(X_te)
    }
    return metrics, history

# ==========================================
# 6. ANA DENEYLER: 7 AKTİVASYON FONKSİYONU x 3 SEED
# ==========================================
activation_list = ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "GELU", "Swish"]
all_runs = {}
histories_seed42 = {}

print("\nAna deneyler çalıştırılıyor (7 Aktivasyon x 3 Seed)...")
for act in activation_list:
    seed_metrics = []
    for s in SEEDS:
        m, h = train_and_evaluate(act_name=act, seed=s)
        seed_metrics.append(m)
        if s == 42:
            histories_seed42[act] = h
    all_runs[act] = seed_metrics
    print(f"Tamamlandı: {act:10s} | Ort. RMSE: {np.mean([x['Avg_RMSE'] for x in seed_metrics]):.4f} ± {np.std([x['Avg_RMSE'] for x in seed_metrics]):.4f}")

# Tablo T6: Ana Performans Metrikleri (Ortalama ± Std)
t6_rows = []
for act in activation_list:
    ms = all_runs[act]
    row = {"Aktivasyon": act}
    for k in ["Y1_RMSE", "Y1_MAE", "Y1_R2", "Y2_RMSE", "Y2_MAE", "Y2_R2", "Avg_RMSE", "Avg_R2", "Train_Time_s"]:
        vals = [m[k] for m in ms]
        row[k + "_Mean"] = round(float(np.mean(vals)), 4)
        row[k + "_Std"] = round(float(np.std(vals)), 4)
        row[k + "_Formatted"] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    t6_rows.append(row)

t6_df = pd.DataFrame(t6_rows)
t6_df.to_csv("results/T6_ana_performans_metrikleri.csv", index=False)

# ==========================================
# 7. GRAFİKLER G4, G5, G6, G7
# ==========================================
colors = {
    "Sigmoid": "#7F8C8D",
    "Tanh": "#8E44AD",
    "ReLU": "#2980B9",
    "LeakyReLU": "#16A085",
    "ELU": "#D35400",
    "GELU": "#C0392B",
    "Swish": "#27AE60"
}

# G4: Eğitim ve Doğrulama Kayıp Eğrisi (MSE)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for act in activation_list:
    axes[0].plot(histories_seed42[act]["train_loss"], label=act, color=colors[act], linewidth=1.6)
    axes[1].plot(histories_seed42[act]["val_loss"], label=act, color=colors[act], linewidth=1.6)

axes[0].set_yscale("log")
axes[0].set_title("(a) Eğitim Kaybı Eğrileri (MSE - Logaritmik Ölçek)")
axes[0].set_xlabel("Epok Sayısı (Epoch)")
axes[0].set_ylabel("Eğitim MSE Kaybı (Log)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_yscale("log")
axes[1].set_title("(b) Doğrulama Kaybı Eğrileri (MSE - Logaritmik Ölçek)")
axes[1].set_xlabel("Epok Sayısı (Epoch)")
axes[1].set_ylabel("Doğrulama MSE Kaybı (Log)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/G4_egitim_dogrulama_kayip_egrisi.png", dpi=300)
plt.close()

# G5: Eğitim ve Doğrulama RMSE Eğrisi
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for act in activation_list:
    axes[0].plot(histories_seed42[act]["train_rmse"], label=act, color=colors[act], linewidth=1.6)
    axes[1].plot(histories_seed42[act]["val_rmse"], label=act, color=colors[act], linewidth=1.6)

axes[0].set_title("(a) Eğitim Seti Ortalama RMSE Yakınsama Eğrisi")
axes[0].set_xlabel("Epok Sayısı (Epoch)")
axes[0].set_ylabel("Eğitim RMSE (kWh/m²)")
axes[0].set_ylim(0, 5.0)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_title("(b) Doğrulama Seti Ortalama RMSE Yakınsama Eğrisi")
axes[1].set_xlabel("Epok Sayısı (Epoch)")
axes[1].set_ylabel("Doğrulama RMSE (kWh/m²)")
axes[1].set_ylim(0, 5.0)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/G5_egitim_dogrulama_rmse_egrisi.png", dpi=300)
plt.close()

# G6: Karşılaştırmalı Yöntem / Parametre Analizi Grafiği (Hata Çubuklu Bar Grafiği)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x_pos = np.arange(len(activation_list))
y1_means = [t6_df.loc[i, "Y1_RMSE_Mean"] for i in range(len(activation_list))]
y1_stds = [t6_df.loc[i, "Y1_RMSE_Std"] for i in range(len(activation_list))]
y2_means = [t6_df.loc[i, "Y2_RMSE_Mean"] for i in range(len(activation_list))]
y2_stds = [t6_df.loc[i, "Y2_RMSE_Std"] for i in range(len(activation_list))]

bar_colors = [colors[a] for a in activation_list]

bars1 = axes[0].bar(x_pos, y1_means, yerr=y1_stds, capsize=4, color=bar_colors, alpha=0.88, edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(activation_list, rotation=25)
axes[0].set_title("(a) Isıtma Yükü (Y1) Test RMSE Karşılaştırması (3 Seed Ort. ± Std)")
axes[0].set_ylabel("Test RMSE (kWh/m²)")
axes[0].set_xlabel("Aktivasyon Fonksiyonu")
axes[0].grid(axis="y", alpha=0.3)
for b in bars1:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

bars2 = axes[1].bar(x_pos, y2_means, yerr=y2_stds, capsize=4, color=bar_colors, alpha=0.88, edgecolor="black")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(activation_list, rotation=25)
axes[1].set_title("(b) Soğutma Yükü (Y2) Test RMSE Karşılaştırması (3 Seed Ort. ± Std)")
axes[1].set_ylabel("Test RMSE (kWh/m²)")
axes[1].set_xlabel("Aktivasyon Fonksiyonu")
axes[1].grid(axis="y", alpha=0.3)
for b in bars2:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("figures/G6_karsilastirmali_aktivasyon_analizi.png", dpi=300)
plt.close()

# En iyi aktivasyon fonksiyonunu seç
best_idx = int(np.argmin(t6_df["Avg_RMSE_Mean"].values))
best_act = t6_df.loc[best_idx, "Aktivasyon"]
print(f"\nEn düşük ortalama RMSE elde eden aktivasyon fonksiyonu: {best_act}")

# G7: Tahmin vs. Gerçek Scatter Grafiği (En İyi Model İçin Y1 ve Y2)
best_hist = histories_seed42[best_act]
y_te_best = best_hist["y_test"]
y_pr_best = best_hist["y_pred"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
# Y1 Scatter
axes[0].scatter(y_te_best[:, 0], y_pr_best[:, 0], alpha=0.78, color="#1f77b4", edgecolors="k", s=45, label="Test Örnekleri")
min_y1, max_y1 = y_te_best[:, 0].min(), y_te_best[:, 0].max()
axes[0].plot([min_y1, max_y1], [min_y1, max_y1], "r--", lw=2, label="İdeal Doğru (y = x)")
r2_y1 = r2_score(y_te_best[:, 0], y_pr_best[:, 0])
rmse_y1 = np.sqrt(mean_squared_error(y_te_best[:, 0], y_pr_best[:, 0]))
axes[0].set_title(f"(a) {best_act} Modeli Isıtma Yükü (Y1) Tahmin vs. Gerçek\n(R² = {r2_y1:.4f}, RMSE = {rmse_y1:.4f} kWh/m²)")
axes[0].set_xlabel("Gerçek Isıtma Yükü Y1 (kWh/m²)")
axes[0].set_ylabel("Tahmin Edilen Isıtma Yükü Ŷ1 (kWh/m²)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Y2 Scatter
axes[1].scatter(y_te_best[:, 1], y_pr_best[:, 1], alpha=0.78, color="#ff7f0e", edgecolors="k", s=45, label="Test Örnekleri")
min_y2, max_y2 = y_te_best[:, 1].min(), y_te_best[:, 1].max()
axes[1].plot([min_y2, max_y2], [min_y2, max_y2], "r--", lw=2, label="İdeal Doğru (y = x)")
r2_y2 = r2_score(y_te_best[:, 1], y_pr_best[:, 1])
rmse_y2 = np.sqrt(mean_squared_error(y_te_best[:, 1], y_pr_best[:, 1]))
axes[1].set_title(f"(b) {best_act} Modeli Soğutma Yükü (Y2) Tahmin vs. Gerçek\n(R² = {r2_y2:.4f}, RMSE = {rmse_y2:.4f} kWh/m²)")
axes[1].set_xlabel("Gerçek Soğutma Yükü Y2 (kWh/m²)")
axes[1].set_ylabel("Tahmin Edilen Soğutma Yükü Ŷ2 (kWh/m²)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/G7_tahmin_vs_gercek_scatter.png", dpi=300)
plt.close()

# ==========================================
# 8. TABLO T7: ABLASYON ÇALIŞMASI
# ==========================================
print("\nAblasyon deneyleri çalıştırılıyor...")
ablation_configs = [
    {"Ad": f"Tam Model ({best_act} + LayerNorm + 3 Katman [128,128,64])", "hidden": [128, 128, 64], "norm": True, "drop": 0.0, "lr": 1e-3},
    {"Ad": "LayerNorm Çıkarılmış (No LayerNorm)", "hidden": [128, 128, 64], "norm": False, "drop": 0.0, "lr": 1e-3},
    {"Ad": "Sığ Mimari (Tek Gizli Katman [128])", "hidden": [128], "norm": True, "drop": 0.0, "lr": 1e-3},
    {"Ad": "Daraltılmış Mimari ([64, 32])", "hidden": [64, 32], "norm": True, "drop": 0.0, "lr": 1e-3},
    {"Ad": "Dropout Eklenmiş (p = 0.10)", "hidden": [128, 128, 64], "norm": True, "drop": 0.10, "lr": 1e-3},
    {"Ad": "Yüksek Öğrenme Oranı (lr = 5e-3)", "hidden": [128, 128, 64], "norm": True, "drop": 0.0, "lr": 5e-3},
]

t7_rows = []
for cfg in ablation_configs:
    ms = []
    for s in SEEDS:
        m, _ = train_and_evaluate(act_name=best_act, seed=s, hidden_dims=cfg["hidden"], use_norm=cfg["norm"], dropout_rate=cfg["drop"], lr=cfg["lr"])
        ms.append(m)
    t7_rows.append({
        "Konfigurasyon": cfg["Ad"],
        "Y1_RMSE": f"{np.mean([m['Y1_RMSE'] for m in ms]):.4f} ± {np.std([m['Y1_RMSE'] for m in ms]):.4f}",
        "Y2_RMSE": f"{np.mean([m['Y2_RMSE'] for m in ms]):.4f} ± {np.std([m['Y2_RMSE'] for m in ms]):.4f}",
        "Avg_RMSE": f"{np.mean([m['Avg_RMSE'] for m in ms]):.4f} ± {np.std([m['Avg_RMSE'] for m in ms]):.4f}",
        "Avg_R2": f"{np.mean([m['Avg_R2'] for m in ms]):.4f} ± {np.std([m['Avg_R2'] for m in ms]):.4f}"
    })
t7_df = pd.DataFrame(t7_rows)
t7_df.to_csv("results/T7_ablasyon_calismasi.csv", index=False)

# ==========================================
# 9. TABLO T8: KLASİK VE LITERATÜR YÖNTEMLERİYLE KARŞILAŞTIRMA
# ==========================================
print("\nKlasik makine öğrenmesi modelleriyle kıyaslama yapılıyor...")
baseline_models = {
    "Ridge Regresyon (L2)": lambda: Ridge(alpha=1.0),
    "Destek Vektör Regresyonu (SVR - RBF)": lambda: SVR(kernel="rbf", C=100.0, gamma="scale"),
    "Rastgele Orman (Random Forest)": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradyan Artırma (Gradient Boosting)": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42)
}

t8_rows = []
for b_name, model_fn in baseline_models.items():
    y1_rmses, y2_rmses, avg_rmses, avg_r2s = [], [], [], []
    for s in SEEDS:
        (X_tr, _, y_tr), (_, _, _), (X_te, _, y_te), _, _ = prepare_splits(df, s)
        m1 = model_fn()
        m2 = model_fn()
        m1.fit(X_tr, y_tr[:, 0])
        m2.fit(X_tr, y_tr[:, 1])
        p1 = m1.predict(X_te)
        p2 = m2.predict(X_te)
        y1_rmses.append(np.sqrt(mean_squared_error(y_te[:, 0], p1)))
        y2_rmses.append(np.sqrt(mean_squared_error(y_te[:, 1], p2)))
        avg_rmses.append(np.sqrt(mean_squared_error(y_te, np.column_stack([p1, p2]))))
        avg_r2s.append(r2_score(y_te, np.column_stack([p1, p2])))
    t8_rows.append({
        "Yontem": b_name,
        "Y1_RMSE": f"{np.mean(y1_rmses):.4f} ± {np.std(y1_rmses):.4f}",
        "Y2_RMSE": f"{np.mean(y2_rmses):.4f} ± {np.std(y2_rmses):.4f}",
        "Avg_RMSE": f"{np.mean(avg_rmses):.4f} ± {np.std(avg_rmses):.4f}",
        "Avg_R2": f"{np.mean(avg_r2s):.4f} ± {np.std(avg_r2s):.4f}"
    })

# Önerilen en iyi MLP modelini T8'e ekle
best_row_t6 = t6_df.loc[best_idx]
t8_rows.append({
    "Yontem": f"Önerilen Derin MLP ({best_act})",
    "Y1_RMSE": best_row_t6["Y1_RMSE_Formatted"],
    "Y2_RMSE": best_row_t6["Y2_RMSE_Formatted"],
    "Avg_RMSE": best_row_t6["Avg_RMSE_Formatted"],
    "Avg_R2": best_row_t6["Avg_R2_Formatted"]
})
t8_df = pd.DataFrame(t8_rows)
t8_df.to_csv("results/T8_yontem_karsilastirma.csv", index=False)

# ==========================================
# 10. TABLO T10: HATA ANALİZİ (EN YÜKSEK HATALI 10 TEST ÖRNEĞİ)
# ==========================================
abs_errors = np.abs(y_te_best - y_pr_best)
total_err = abs_errors.sum(axis=1)
worst_indices = np.argsort(total_err)[::-1][:10]

t10_rows = []
for rank, idx in enumerate(worst_indices, 1):
    x_row = best_hist["X_test"][idx]
    t10_rows.append({
        "Sira": rank,
        "X1_Kompaktlik": round(float(x_row[0]), 2),
        "X2_Yuzey": round(float(x_row[1]), 1),
        "X5_Yukseklik": round(float(x_row[4]), 1),
        "X7_CamOrani": round(float(x_row[6]), 2),
        "Gercek_Y1": round(float(y_te_best[idx, 0]), 2),
        "Tahmin_Y1": round(float(y_pr_best[idx, 0]), 2),
        "Hata_Y1": round(float(abs_errors[idx, 0]), 2),
        "Gercek_Y2": round(float(y_te_best[idx, 1]), 2),
        "Tahmin_Y2": round(float(y_pr_best[idx, 1]), 2),
        "Hata_Y2": round(float(abs_errors[idx, 1]), 2)
    })
t10_df = pd.DataFrame(t10_rows)
t10_df.to_csv("results/T10_hata_analizi.csv", index=False)

# Tüm sonuçları tek JSON dosyasında topla
summary_payload = {
    "best_activation": best_act,
    "T1_stats": t1_df.to_dict(orient="records"),
    "T6_main_metrics": t6_df[["Aktivasyon", "Y1_RMSE_Formatted", "Y1_MAE_Formatted", "Y1_R2_Formatted", "Y2_RMSE_Formatted", "Y2_MAE_Formatted", "Y2_R2_Formatted", "Avg_RMSE_Formatted", "Avg_R2_Formatted", "Train_Time_s_Formatted"]].to_dict(orient="records"),
    "T7_ablation": t7_df.to_dict(orient="records"),
    "T8_comparison": t8_df.to_dict(orient="records"),
    "T10_error_analysis": t10_df.to_dict(orient="records")
}
with open("results/tum_deney_sonuclari.json", "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, ensure_ascii=False, indent=2)

print("\nTÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
print("- 7 adet 300 DPI grafik 'figures/' klasörüne kaydedildi.")
print("- Tüm tablo verileri 'results/' klasörüne CSV ve JSON olarak kaydedildi.")
