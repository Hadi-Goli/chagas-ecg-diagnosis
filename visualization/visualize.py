import matplotlib.pyplot as plt
import wfdb
import numpy as np
import os
import glob
import scipy.signal

# تنظیمات گرافیکی برای نمایش هماهنگ و فضای کافی برای ۴ ردیف
plt.rcParams['figure.figsize'] = [25, 16]
plt.style.use('seaborn-v0_8-whitegrid')

def get_records_from_folder(folder_path, limit=5):
    """گرفتن لیست فایل‌های رکورد با حذف پسوند"""
    header_files = glob.glob(os.path.join(folder_path, "*.hea"))
    header_files = sorted(header_files)
    records = [os.path.splitext(f)[0] for f in header_files]
    return records[:limit]

def normalize_ecg(raw_lead, fs):
    """
    نرمال‌سازی با الهام از رویکرد پیشرفته دوست شما:
    ۱. جایگزینی مقادیر خالی با صفر
    ۲. استفاده از فیلتر باترورث (High-pass) برای حذف نوسانات بیس‌لاین
    ۳. اسکیل کردن داده‌ها بین ۰ و ۱
    """
    processed = np.nan_to_num(raw_lead)
    
    # مرحله ۱: فیلتر برای حذف Baseline Wander تا سیگنال در مرکز ثابت شود
    try:
        sos = scipy.signal.butter(1, 0.5, 'hp', fs=fs, output='sos')
        cleaned = scipy.signal.sosfilt(sos, processed)
    except Exception:
        # در صورت بروز خطای فیلتر به دلیل باگ فرکانس، سیگنال خام استفاده می‌شود
        cleaned = processed
        
    # مرحله ۲: اسکیل کردن Min-Max (بین 0 و 1) بر اساس کد قبلی
    min_val = np.min(cleaned)
    max_val = np.max(cleaned)
    
    if max_val - min_val > 0:
        clean_norm = (cleaned - min_val) / (max_val - min_val)
    else:
        clean_norm = cleaned
        
    return clean_norm

def plot_chagas_comparison(pos_folder, neg_folder):
    pos_records = get_records_from_folder(pos_folder, limit=5)
    neg_records = get_records_from_folder(neg_folder, limit=5)

    if not pos_records or not neg_records:
        print("خطا: فایل‌های کافی در پوشه‌ها پیدا نشد.")
        return

    # ساخت گرید ۴ در ۵ (ردیف خام و نرمال برای مثبت، سپس ردیف خام و نرمال برای منفی)
    fig, axes = plt.subplots(nrows=4, ncols=5, constrained_layout=True)
    fig.suptitle('ECG Signal (Lead II): Raw vs Normalized\nShowing Correction of Baseline and Extreme Peaks', fontsize=22, weight='bold')

    # تنظیمات هر ردیف (نمونه‌ها، عنوان، شماره ردیف خام، شماره ردیف نرمال، رنگ خام، رنگ نرمال)
    configs = [
        (pos_records, "POSITIVE\n(SaMi-Trop)", 0, 1, 'lightcoral', 'crimson'),
        (neg_records, "NEGATIVE\n(PTB-XL)", 2, 3, 'mediumaquamarine', 'teal')
    ]

    for records, label, raw_row, norm_row, raw_c, norm_c in configs:
        for col_idx, record_path in enumerate(records):
            ax_raw = axes[raw_row, col_idx]
            ax_norm = axes[norm_row, col_idx]
            
            try:
                # خواندن داده
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
                fs = record.fs
                
                # استخراج لید II
                lead_idx = 1
                raw_data = signal[:, lead_idx]
                
                # اعمال تابع نرمال‌سازی ترکیبی
                norm_data = normalize_ecg(raw_data, fs)
                time = np.arange(len(raw_data)) / fs
                file_name = os.path.basename(record_path)
                
                # ----- 1. رسم سیگنال خام در ردیف بالایی -----
                ax_raw.plot(time, raw_data, color=raw_c, linewidth=1)
                ax_raw.set_title(f"{file_name} - Raw", fontsize=11)
                if col_idx == 0:
                    ax_raw.set_ylabel(f"{label}\nRaw Amplitude", fontsize=11, weight='bold')
                
                # ----- 2. رسم سیگنال نرمال شده در ردیف پایینی -----
                ax_norm.plot(time, norm_data, color=norm_c, linewidth=1.2)
                ax_norm.set_title(f"{file_name} - Normalized", fontsize=11)
                if col_idx == 0:
                    ax_norm.set_ylabel(f"{label}\nNorm (0-1)", fontsize=11, weight='bold')
                    
                # فقط برای ردیف آخر محور X تنظیم شود
                if norm_row == 3:
                    ax_norm.set_xlabel("Time (s)", fontsize=11)
                    
                # زیباسازی پلات (حذف قاب‌های اضافی)
                for ax in [ax_raw, ax_norm]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

            except Exception as e:
                for ax in [ax_raw, ax_norm]:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', color='red')
                print(f"Error processing {record_path}: {e}")

    # ذخیره فایل در مسیر اجرایی
    output_file = "raw_vs_normalized_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"Visualization completed! Saved as: {output_file}")

if __name__ == "__main__":
    # به دست آوردن مسیر کامل به پوشه‌های old برای فراخوانی صحیح
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_dir = os.path.join(script_dir, "sami-trop")
    neg_dir = os.path.join(script_dir, "ptb-xl")
    
    plot_chagas_comparison(pos_dir, neg_dir)
