import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# HÀM CHUẨN HÓA
# =========================
def min_max_scaling(df):
    return (df - df.min()) / (df.max() - df.min())

def z_score_scaling(df):
    return (df - df.mean()) / df.std()

# =========================
# HÀM VẼ
# =========================
def plot_hist_box(df, title):
    df.hist(figsize=(10,6))
    plt.suptitle(f"Histogram - {title}")
    plt.show()

    df.plot(kind='box', subplots=True, layout=(1,len(df.columns)), figsize=(12,4))
    plt.suptitle(f"Boxplot - {title}")
    plt.show()

def compare_distribution(original, minmax, zscore, title):
    for col in original.columns:
        plt.figure(figsize=(10,4))
        
        plt.subplot(1,3,1)
        plt.hist(original[col])
        plt.title(f"{col} - Gốc")
        
        plt.subplot(1,3,2)
        plt.hist(minmax[col])
        plt.title("Min-Max")
        
        plt.subplot(1,3,3)
        plt.hist(zscore[col])
        plt.title("Z-score")
        
        plt.suptitle(title)
        plt.show()

# =========================
# BÀI 1: VẬN ĐỘNG VIÊN
# =========================

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_3_Sports.csv")

cols1 = ['chieu_cao_cm','can_nang_kg','toc_do_100m_s','so_ban_thang','so_phut_thi_dau']

df1 = df[cols1]

print("=== BÀI 1 ===")
print(df1.info())
print(df1.isnull().sum())
print(df1.describe())

plot_hist_box(df1, "Athlete")

df1_minmax = min_max_scaling(df1)
df1_zscore = z_score_scaling(df1)

compare_distribution(df1, df1_minmax, df1_zscore, "Athlete Scaling")


# =========================
# BÀI 2: BỆNH NHÂN
# =========================
df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_3_Health.csv")

cols2 = ['BMI','huyet_ap_mmHg','nhip_tim_bpm','cholesterol_mg_dl']
df2 = df[cols2]

print("=== BÀI 2 ===")
print(df2.describe())

plot_hist_box(df2, "Patient")

# phát hiện outlier bằng IQR
def detect_outlier_iqr(df):
    out = pd.DataFrame()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        out[col] = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
    return out

outliers2 = detect_outlier_iqr(df2)
print("Outliers:\n", outliers2.sum())

df2_minmax = min_max_scaling(df2)
df2_zscore = z_score_scaling(df2)

compare_distribution(df2, df2_minmax, df2_zscore, "Patient Scaling")


# =========================
# BÀI 3: CÔNG TY
# =========================
df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_3_Finance.csv")

cols3 = ['doanh_thu_musd','loi_nhuan_musd','so_nhan_vien','EPS']
df3 = df[cols3]

print("=== BÀI 3 ===")
print(df3.describe())

plot_hist_box(df3, "Company")

df3_minmax = min_max_scaling(df3)
df3_zscore = z_score_scaling(df3)

# scatter trước
plt.scatter(df3['doanh_thu_musd'], df3['loi_nhuan_musd'])
plt.title("Before Scaling")
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.show()

# scatter MinMax
plt.scatter(df3_minmax['doanh_thu_musd'], df3_minmax['loi_nhuan_musd'])
plt.title("Min-Max Scaling")
plt.show()

# scatter Z-score
plt.scatter(df3_zscore['doanh_thu_musd'], df3_zscore['loi_nhuan_musd'])
plt.title("Z-score Scaling")
plt.show()


# =========================
# BÀI 4: NGƯỜI CHƠI
# =========================
df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_3_Gaming.csv")

cols4 = ['gio_choi','diem_tich_luy','so_level','so_vat_pham']
df4 = df[cols4]

print("=== BÀI 4 ===")
print(df4.info())
print(df4.isnull().sum())

plot_hist_box(df4, "Player")

df4_minmax = min_max_scaling(df4)
df4_zscore = z_score_scaling(df4)

compare_distribution(df4, df4_minmax, df4_zscore, "Player Scaling")