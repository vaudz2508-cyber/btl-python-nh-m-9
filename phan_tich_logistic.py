# -*- coding: utf-8 -*-
"""Đọc Excel có sẵn 210 dòng, tiền xử lý, thống kê, Logistic Regression và đánh giá."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

FOLDER=Path(__file__).resolve().parent
INPUT=FOLDER/'Bảng dữ liệu đánh giá của người dùng.xlsx'
OUTPUT=FOLDER/'Ket_qua_phan_tich.xlsx'
NUMERICAL=['Age','Flight Distance','Inflight wifi service']
CATEGORICAL=['Gender','Class']
TARGET='satisfaction'

# 1. Đọc file Excel có dữ liệu sẵn. KHÔNG cần Internet.
if not INPUT.exists():
    raise FileNotFoundError('Không tìm thấy file Excel trong cùng thư mục: '+INPUT.name)
df=pd.read_excel(INPUT, sheet_name='Du lieu goc')
assert len(df)==210, f'Cần 210 bản ghi, hiện có {len(df)}'
print('Đã đọc',len(df),'bản ghi từ Excel')

# 2. Kiểm tra dữ liệu trước xử lý
print('Số ô thiếu:',int(df.isna().sum().sum()))
print('Dòng trùng lặp:',int(df.duplicated().sum()))
print('Chỉ số dòng nguồn trùng:',int(df['Source row index'].duplicated().sum()))
errors=[]
for col in NUMERICAL:
    converted=pd.to_numeric(df[col], errors='coerce')
    count=int((df[col].notna() & converted.isna()).sum())
    errors.append([col,'Sai kiểu',count])
    df[col]=converted
for col, lower, upper in [('Age',0,110),('Flight Distance',1,np.inf),('Inflight wifi service',0,5)]:
    invalid=(df[col]<lower)|(df[col]>upper)
    errors.append([col,'Giá trị ngoài khoảng',int(invalid.sum())])
    df.loc[invalid,col]=np.nan
for col, accepted in {'Gender':['Female','Male'], 'Class':['Eco','Eco Plus','Business']}.items():
    invalid=df[col].notna() & ~df[col].isin(accepted)
    errors.append([col,'Nhãn không hợp lệ',int(invalid.sum())])
    df.loc[invalid,col]=np.nan
bad=df[TARGET].isna()|~df[TARGET].isin(['satisfied','neutral or dissatisfied'])
errors.append([TARGET,'Nhãn mục tiêu không hợp lệ',int(bad.sum())])
df=df.loc[~bad].drop_duplicates().drop_duplicates(subset=['Source row index']).copy()

# 3. IQR để phát hiện ngoại lệ, chỉ cảnh báo vì có thể là dữ liệu thật
for col in NUMERICAL:
    x=df[col].dropna();q1,q3=x.quantile([0.25,0.75]);iqr=q3-q1
    count=int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum())
    print('Ngoại lệ IQR',col,':',count)
    errors.append([col,'Ngoại lệ IQR (giữ lại)',count])

# 4. Chia dữ liệu trước khi điền thiếu/mã hóa/chuẩn hóa để tránh data leakage
X=df[NUMERICAL+CATEGORICAL];y=df[TARGET]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
preprocess=ColumnTransformer([
  ('number',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERICAL),
  ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                       ('encode',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)
])
model=Pipeline([('preprocess',preprocess),('logistic',LogisticRegression(max_iter=2000))])
model.fit(X_train,y_train)
pred=model.predict(X_test)
print('Accuracy:',round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred,zero_division=0))

# 5. Thống kê mô tả và tương quan
stats=df[NUMERICAL].agg(['mean','median','var','std','min','max']).T
corr=df[NUMERICAL].corr()
results=X_test.copy();results['Thực tế']=y_test;results['Dự đoán']=pred
results['Dự đoán đúng']=results['Thực tế']==results['Dự đoán']
with pd.ExcelWriter(OUTPUT,engine='openpyxl') as writer:
    df.to_excel(writer,sheet_name='Du lieu sau kiem tra',index=False)
    stats.to_excel(writer,sheet_name='Thong ke mo ta')
    corr.to_excel(writer,sheet_name='Tuong quan')
    pd.DataFrame(errors,columns=['Thuộc tính','Kiểm tra','Số lượng']).to_excel(writer,sheet_name='Kiem tra du lieu',index=False)
    results.to_excel(writer,sheet_name='Ket qua du doan',index=False)
    pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.to_excel(writer,sheet_name='Danh gia mo hinh')

# 6. Ba biểu đồ + ma trận nhầm lẫn
plt.figure(figsize=(7,4));df[TARGET].value_counts().plot(kind='bar');plt.title('Phan bo muc do hai long');plt.ylabel('So luong');plt.xticks(rotation=10);plt.tight_layout();plt.savefig(FOLDER/'01_phan_bo_hai_long.png');plt.close()
plt.figure(figsize=(7,4));df.groupby(TARGET)['Age'].mean().plot(kind='bar');plt.title('Tuoi trung binh theo nhom');plt.ylabel('Tuoi');plt.xticks(rotation=10);plt.tight_layout();plt.savefig(FOLDER/'02_tuoi_trung_binh.png');plt.close()
plt.figure(figsize=(7,4));df.boxplot(column='Flight Distance',by=TARGET);plt.suptitle('');plt.title('Khoang cach chuyen bay');plt.tight_layout();plt.savefig(FOLDER/'03_khoang_cach.png');plt.close()
labels=['satisfied','neutral or dissatisfied']
cm=confusion_matrix(y_test,pred,labels=labels)
fig,ax=plt.subplots(figsize=(8,5));ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Greys',colorbar=False,xticks_rotation=15);plt.tight_layout();plt.savefig(FOLDER/'04_ma_tran_nham_lan.png');plt.close()
print('Xong! Kết quả đã được tạo tại:',FOLDER)
