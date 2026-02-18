import os
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from helper_code import get_label, load_header

# ==========================================
# ⚙️ تنظیمات و مسیرها (تنظیم شده برای لینوکس)
# ==========================================
BASE_DIR = Path(".")
DATA_DIR = Path("/home/hadi/Coding/ML/chagas_dataset/python-example-2025")

SOURCE_DIRS = {
    'code15': DATA_DIR / 'code15_output',
    'samitrop': DATA_DIR / 'samitrop_output',
    'ptbxl': DATA_DIR / 'ptbxl_output'
}

# مسیر پوشه‌های خروجی نهایی
TRAIN_DIR = BASE_DIR / 'training_data'
VAL_DIR = BASE_DIR / 'validation_data'
HOLDOUT_DIR = BASE_DIR / 'holdout_data'

# نسبت‌های تقسیم دیتا
TEST_SIZE = 0.10  # 10% برای تست نهایی (Holdout)
VAL_SIZE = 0.10   # 10% از کل دیتا برای اعتبارسنجی (Validation)
# (مابقی که 80% است برای آموزش (Train) در نظر گرفته می‌شود)

# ==========================================
# 🛠 توابع کمکی (Helper Functions)
# ==========================================
def create_symlink(src: Path, dst: Path):
    """ایجاد لینک نمادین بسیار سریع و جایگزینی در صورت وجود"""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)

def clean_directory(dir_path: Path):
    """پاکسازی و ساخت مجدد پوشه‌ها برای جلوگیری از تداخل دیتای قدیمی"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 🚀 خط لوله اصلی پردازش داده (Main Pipeline)
# ==========================================
def prepare_data_pipeline():
    start_time = time.time()

    print("="*65)
    print("🚀 Starting AI Data Pipeline: Stratified Split & Symlink")
    print("="*65)
    
    # ---------------------------------------------------------
    # فاز ۱: اسکن فایل‌ها و استخراج لیبل‌ها (مانند کد نوت‌بوک شما)
    # ---------------------------------------------------------
    print("\n⏳ Phase 1: Scanning files and extracting labels (rglob)...")
    records = []
    
    for source_name, source_path in SOURCE_DIRS.items():
        if not source_path.exists():
            print(f"   ⚠️ Warning: Source path not found -> {source_path}")
            continue
            
        print(f"   🔍 Scanning '{source_name}' recursively...")
        
        # پیدا کردن تمام فایل‌های هدر در تمامی زیرپوشه‌ها
        hea_files = list(source_path.rglob('*.hea'))
        
        for hea_path in hea_files:
            record_base = hea_path.with_suffix('') # مسیر بدون پسوند
            try:
                # خواندن هدر برای استخراج لیبل (جهت Stratified Split)
                header = load_header(str(record_base))
                label = get_label(header)
                
                records.append({
                    'record_name': hea_path.stem,
                    'base_path': record_base,
                    'header_path': hea_path,
                    'label': label,
                    'source': source_name
                })
            except Exception as e:
                print(f"      ⚠️ Error processing {hea_path.name}: {e}")

    df = pd.DataFrame(records)
    
    if df.empty:
        print("❌ No records found! Please check your source paths.")
        return

    # --- نمایش آمار قبل از تقسیم (Pre-Split) ---
    print("\n📊 Pre-Split Statistics (Total Data Found):")
    print("-" * 45)
    for src, count in df['source'].value_counts().items():
         print(f"   - {src:<15}: {count} records")
    print("-" * 45)
    print(f"   - {'TOTAL':<15}: {len(df)} records")
    
    # ---------------------------------------------------------
    # فاز ۲: تقسیم لایه‌بندی شده (Stratified Splits)
    # ---------------------------------------------------------
    print("\n✂️ Phase 2: Performing Stratified Splits (Train/Val/Holdout)...")
    
    # مرحله اول: جدا کردن Test (Holdout) از بقیه
    train_val_df, holdout_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        stratify=df['label'], 
        random_state=42
    )
    
    # مرحله دوم: جدا کردن Val از Train
    # چون test_size از دیتای باقی‌مانده حساب می‌شود، باید نسبت آن را حساب کنیم
    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        stratify=train_val_df['label'], 
        random_state=42
    )
    
    # --- نمایش آمار دقیق و جدولی بعد از تقسیم (Post-Split) ---
    print("\n📊 Post-Split Statistics:")
    print(f"{'Source':<15} | {'Train':<8} | {'Val':<8} | {'Holdout':<8} | {'Total':<8}")
    print("-" * 65)
    for source in SOURCE_DIRS.keys():
        n_train = len(train_df[train_df['source'] == source])
        n_val = len(val_df[val_df['source'] == source])
        n_hold = len(holdout_df[holdout_df['source'] == source])
        n_total = n_train + n_val + n_hold
        if n_total > 0:
            print(f"{source:<15} | {n_train:<8} | {n_val:<8} | {n_hold:<8} | {n_total:<8}")
    print("-" * 65)
    print(f"{'ALL':<15} | {len(train_df):<8} | {len(val_df):<8} | {len(holdout_df):<8} | {len(df):<8}\n")

    # ---------------------------------------------------------
    # فاز ۳: ساخت پوشه‌ها و لینک کردن فایل‌ها (Symlinks)
    # ---------------------------------------------------------
    print("📁 Phase 3: Cleaning directories and creating Symlinks...")
    
    splits = [
        ('Train', train_df, TRAIN_DIR),
        ('Validation', val_df, VAL_DIR),
        ('Holdout', holdout_df, HOLDOUT_DIR)
    ]
    
    for split_name, split_df, target_dir in splits:
        clean_directory(target_dir) # اطمینان از خالی بودن پوشه
        print(f"   🚀 Populating {split_name} data into '{target_dir.name}'...")
        
        count = 0
        for _, row in split_df.iterrows():
            base_src = row['base_path']
            record_name = row['record_name']
            
            # 1. لینک فایل هدر (.hea)
            create_symlink(row['header_path'], target_dir / f"{record_name}.hea")
            
            # 2. لینک فایل سیگنال (.dat)
            dat_src = base_src.with_suffix('.dat')
            if dat_src.exists():
                create_symlink(dat_src, target_dir / f"{record_name}.dat")
                
            # 3. لینک فایل متلب (در صورت وجود .mat)
            mat_src = base_src.with_suffix('.mat')
            if mat_src.exists():
                create_symlink(mat_src, target_dir / f"{record_name}.mat")
                
            count += 1
            if count % 15000 == 0:
                print(f"      ... linked {count} files")
                
        # --- ذخیره فایل متادیتا (Best Practice مهندسی داده) ---
        # یک فایل CSV در هر پوشه ذخیره می‌کنیم تا بدانیم دقیقا چه فایل‌هایی با چه لیبلی داخلش هستند
        csv_path = target_dir / f"{split_name.lower()}_metadata.csv"
        # مسیرهای طولانی سیستم را حذف میکنیم که فقط دیتای تمیز در CSV بماند
        clean_df = split_df.drop(columns=['base_path', 'header_path'])
        clean_df.to_csv(csv_path, index=False)

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n⏱️ Total execution time: {int(duration // 60)}m {duration % 60:.2f}s")
    print("\n✅ Success! The data pipeline is complete. Data is ready for AI training.")
    print("="*65)

if __name__ == '__main__':
    prepare_data_pipeline()