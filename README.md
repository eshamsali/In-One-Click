# StatLab — دليل التشغيل

Contribution by Sara Fathy

## المتطلبات
- Python 3.8 أو أحدث → https://www.python.org/downloads/

---

## خطوات التشغيل

### Windows
انقر دابل كليك على `start.bat`

### Mac / Linux
```bash
chmod +x start.sh
./start.sh
```

### يدوياً (أي نظام)
```bash
pip install -r requirements.txt
python analysis.py
```

---

## بعد تشغيل السيرفر
افتح ملف **`index.html`** في المتصفح (Chrome/Firefox/Edge).

> السيرفر يعمل على: http://localhost:5000

---

## هيكل الملفات

```
StatLab/
├── index.html          ← الصفحة الرئيسية
├── workspace.html      ← بيئة التحليل
├── auth.html           ← تسجيل الدخول
├── profile.html        ← صفحة المستخدم
├── styles.css          ← التصميم
│
├── analysis.py         ← Flask Backend (API Server)
├── predict.py          ← نموذج التنبؤ بالأمراض
│
├── diabetes_model.joblib    ← نموذج السكري المدرّب
├── diabetes_scaler.joblib   ← معالج البيانات
├── diabetes_meta.json       ← معلومات النموذج
├── diabetes_database.csv    ← قاعدة بيانات التدريب
│
├── requirements.txt    ← المكتبات المطلوبة
├── start.bat           ← تشغيل Windows
└── start.sh            ← تشغيل Mac/Linux
```

---

## API Endpoints

| Endpoint | Method | الوصف |
|----------|--------|-------|
| `/health` | GET | حالة السيرفر |
| `/upload` | POST | رفع ملف CSV/Excel وتحليله |
| `/predict` | POST | التنبؤ بالأمراض |
| `/analyze/ttest` | POST | اختبار T |
| `/analyze/ztest` | POST | اختبار Z |
| `/analyze/chisquare` | POST | اختبار Chi-Square |
| `/diseases` | GET | قائمة الأمراض المتاحة |
