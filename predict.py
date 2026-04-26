import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional

# ══════════════════════════════════════════════════════════════
#  StatLab — وحدة التنبؤ بالأمراض
#  predict.py
#
#  المبدأ:
#    1. عندنا نموذج Logistic Regression اتدرّب على داتا بيز
#       لكل مرض (السكر مثلاً)
#    2. الطالب يرفع بياناته → النظام يكتشف المرض تلقائياً
#       من أسماء الأعمدة → يشغّل النموذج المناسب
#    3. النتيجة: احتمال الإصابة (0-100%) + تفسير + أهم العوامل
#
#  إضافة مرض جديد:
#    - اعمل ملف: {disease}_model.joblib
#    - اعمل ملف: {disease}_scaler.joblib
#    - اعمل ملف: {disease}_meta.json
#    - ضيفه في DISEASE_REGISTRY
# ══════════════════════════════════════════════════════════════

# المسار اللي فيه ملفات النماذج
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── سجل الأمراض المتاحة ──────────────────────────────────────
#
#  لكل مرض:
#    features      : الأعمدة المطلوبة بالترتيب (نفس ترتيب التدريب)
#    aliases       : أسماء بديلة للأعمدة (الفرونت ممكن يبعت أسماء مختلفة)
#    model_file    : ملف النموذج
#    scaler_file   : ملف الـ Scaler
#    meta_file     : ملف الـ metadata
#    target_col    : اسم عمود النتيجة في الداتا (لو موجود)
#    display_name  : الاسم اللي يظهر في الـ output
#
DISEASE_REGISTRY = {
    "diabetes": {
        "display_name": "Diabetes (السكر)",
        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        # كلمات مفتاحية للكشف التلقائي من أسماء الأعمدة
        "detection_keywords": [
            "glucose", "insulin", "bmi", "diabetes",
            "pregnancies", "pedigree", "diabetic",
        ],
        "aliases": {
            "blood_pressure": "BloodPressure",
            "bloodpressure":  "BloodPressure",
            "bp":             "BloodPressure",
            "skin":           "SkinThickness",
            "skin_thickness": "SkinThickness",
            "pedigree":       "DiabetesPedigreeFunction",
            "dpf":            "DiabetesPedigreeFunction",
            "preg":           "Pregnancies",
        },
        "model_file":  "diabetes_model.joblib",
        "scaler_file": "diabetes_scaler.joblib",
        "meta_file":   "diabetes_meta.json",
        "target_col":  "Outcome",
        "labels":      {0: "Non-Diabetic", 1: "Diabetic"},
        "risk_labels": {
            "low":      "Low Risk — منخفض الخطورة",
            "moderate": "Moderate Risk — متوسط الخطورة",
            "high":     "High Risk — عالي الخطورة",
        },
        # نصائح طبية بناءً على أهم العوامل
        "feature_advice": {
            "Glucose":                  "مستوى الجلوكوز هو المؤشر الأهم — القيمة الطبيعية < 140 mg/dL",
            "Insulin":                  "مستوى الإنسولين مرتفع — قد يشير لمقاومة الإنسولين",
            "BMI":                      "مؤشر كتلة الجسم — تقليل الوزن يقلل الخطورة بشكل كبير",
            "Pregnancies":              "عدد الحمل عامل خطورة لسكر الحمل",
            "Age":                      "العمر عامل خطورة — تزداد الخطورة بعد 45 سنة",
            "DiabetesPedigreeFunction": "التاريخ العائلي له دور مهم",
            "BloodPressure":            "ضغط الدم مرتبط بمتلازمة الأيض",
            "SkinThickness":            "سماكة الجلد مؤشر لمقاومة الإنسولين",
        },
    },

    # ── يمكن إضافة أمراض جديدة هنا بنفس الشكل ──
    # "pcos": { ... },
    # "heart_disease": { ... },
}


# ══════════════════════════════════════════════════════════════
#  1. تحميل النموذج
# ══════════════════════════════════════════════════════════════

_loaded_models = {}   # cache عشان مانحملش النموذج أكتر من مرة


def load_disease_model(disease: str) -> dict:
    """
    يحمّل نموذج المرض من الـ cache أو من الملفات

    الاستجابة:
        { "model", "scaler", "meta", "config" }
    """
    if disease in _loaded_models:
        return _loaded_models[disease]

    config = DISEASE_REGISTRY.get(disease)
    if not config:
        raise ValueError(f"المرض '{disease}' غير موجود في السجل. المتاح: {list(DISEASE_REGISTRY.keys())}")

    model_path  = os.path.join(MODELS_DIR, config["model_file"])
    scaler_path = os.path.join(MODELS_DIR, config["scaler_file"])
    meta_path   = os.path.join(MODELS_DIR, config["meta_file"])

    for path in [model_path, scaler_path, meta_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ملف النموذج مش موجود: {path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    result = {"model": model, "scaler": scaler, "meta": meta, "config": config}
    _loaded_models[disease] = result
    return result


# ══════════════════════════════════════════════════════════════
#  2. الكشف التلقائي عن المرض من أسماء الأعمدة
# ══════════════════════════════════════════════════════════════

def detect_disease(columns: list) -> Optional[str]:
    """
    يكتشف المرض تلقائياً من أسماء الأعمدة في الداتا

    مثال:
        columns = ["Glucose", "BMI", "Age", "Outcome"]
        → يرجع "diabetes"
    """
    cols_lower = {c.lower().replace(" ", "_") for c in columns}

    best_disease = None
    best_score   = 0

    for disease, config in DISEASE_REGISTRY.items():
        keywords = config.get("detection_keywords", [])
        score    = sum(1 for kw in keywords if kw in cols_lower)
        if score > best_score:
            best_score   = score
            best_disease = disease

    # لازم يكون في تطابق على الأقل عشان نكون واثقين
    return best_disease if best_score >= 2 else None


# ══════════════════════════════════════════════════════════════
#  3. تجهيز بيانات صف واحد للتنبؤ
# ══════════════════════════════════════════════════════════════

def prepare_row(row_data: dict, config: dict, meta: dict) -> np.ndarray:
    """
    يحوّل dict (صف واحد من البيانات) لـ numpy array جاهز للنموذج

    بيعمل:
      - تطبيق الـ aliases (أسماء بديلة)
      - ملء القيم الناقصة بالـ mean من التدريب
      - ترتيب الأعمدة صح
    """
    features = config["features"]
    aliases  = config.get("aliases", {})
    ranges   = meta.get("feature_ranges", {})

    # normalize أسماء المفاتيح
    normalized = {}
    for k, v in row_data.items():
        key = k.strip().replace(" ", "_")
        # ابحث في الـ aliases
        canonical = aliases.get(key.lower(), key)
        normalized[canonical] = v

    values = []
    for feat in features:
        if feat in normalized and normalized[feat] is not None:
            try:
                values.append(float(normalized[feat]))
            except (ValueError, TypeError):
                # لو مش رقم → استخدم الـ mean
                values.append(ranges.get(feat, {}).get("mean", 0.0))
        else:
            # قيمة ناقصة → استخدم الـ mean من التدريب
            values.append(ranges.get(feat, {}).get("mean", 0.0))

    return np.array(values).reshape(1, -1)


# ══════════════════════════════════════════════════════════════
#  4. التنبؤ لصف واحد
# ══════════════════════════════════════════════════════════════

def predict_single(row_data: dict, disease: str = None) -> dict:
    """
    يعمل تنبؤ لحالة واحدة (صف واحد من البيانات)

    المعاملات:
        row_data : dict فيه القيم، مثال:
                   {"Glucose": 148, "BMI": 33.6, "Age": 50, ...}
        disease  : اسم المرض (لو None بيكتشف تلقائياً)

    الاستجابة:
        {
          "disease", "probability", "risk_level", "prediction",
          "prediction_label", "top_factors", "advice",
          "model_accuracy", "interpretation"
        }
    """
    # كشف المرض تلقائياً لو مش محدد
    if not disease:
        disease = detect_disease(list(row_data.keys()))
        if not disease:
            return {"error": "لم يتم التعرف على نوع المرض من أسماء الأعمدة"}

    # تحميل النموذج
    bundle = load_disease_model(disease)
    model  = bundle["model"]
    scaler = bundle["scaler"]
    meta   = bundle["meta"]
    config = bundle["config"]

    # تجهيز البيانات
    X = prepare_row(row_data, config, meta)
    X_scaled = scaler.transform(X)

    # التنبؤ
    probability  = float(model.predict_proba(X_scaled)[0][1])
    prediction   = int(model.predict(X_scaled)[0])
    pred_label   = config["labels"][prediction]

    # مستوى الخطورة
    thresholds = meta.get("risk_thresholds", {"low": 0.3, "moderate": 0.5, "high": 0.7})
    if probability < thresholds["low"]:
        risk_level = "low"
    elif probability < thresholds["moderate"]:
        risk_level = "moderate"
    elif probability < thresholds["high"]:
        risk_level = "high"
    else:
        risk_level = "very_high"

    risk_label = config["risk_labels"].get(
        risk_level,
        config["risk_labels"].get("high", "High Risk")
    )

    # أهم العوامل المؤثرة (Feature Importance)
    coefs   = meta.get("coefficients", {})
    ranges  = meta.get("feature_ranges", {})
    factors = []

    for feat in config["features"]:
        val     = X[0][config["features"].index(feat)]
        mean    = ranges.get(feat, {}).get("mean", val)
        std     = ranges.get(feat, {}).get("std",  1.0) or 1.0
        coef    = coefs.get(feat, 0)

        # Z-score للقيمة هذه
        z_score = (val - mean) / std

        # تأثير هذا العامل على الاحتمال
        impact  = round(float(coef * z_score), 4)

        factors.append({
            "feature":   feat,
            "value":     round(float(val), 2),
            "mean":      round(float(mean), 2),
            "z_score":   round(float(z_score), 2),
            "impact":    impact,
            "direction": "↑ يرفع الخطورة" if impact > 0 else "↓ يخفض الخطورة",
        })

    # ترتيب حسب التأثير المطلق
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_factors = factors[:4]   # أهم 4 عوامل

    # نصائح للعوامل الأعلى تأثيراً
    advice = []
    for f in top_factors:
        tip = config["feature_advice"].get(f["feature"])
        if tip and abs(f["impact"]) > 0.05:
            advice.append({"factor": f["feature"], "tip": tip})

    # تفسير النتيجة
    pct = round(probability * 100, 1)
    if probability < 0.3:
        interpretation = f"احتمال الإصابة منخفض جداً ({pct}%). المؤشرات الحيوية في النطاق الطبيعي."
    elif probability < 0.5:
        interpretation = f"احتمال إصابة معتدل ({pct}%). يُنصح بمتابعة دورية مع الطبيب."
    elif probability < 0.7:
        interpretation = f"احتمال إصابة مرتفع ({pct}%). يُنصح بإجراء تحاليل مؤكدة."
    else:
        interpretation = f"احتمال إصابة مرتفع جداً ({pct}%). يُنصح بمراجعة الطبيب فوراً."

    return {
        "disease":         disease,
        "display_name":    config["display_name"],
        "probability":     round(probability, 4),
        "probability_pct": pct,
        "risk_level":      risk_level,
        "risk_label":      risk_label,
        "prediction":      prediction,
        "prediction_label": pred_label,
        "top_factors":     top_factors,
        "all_factors":     factors,
        "advice":          advice,
        "model_accuracy":  round(meta.get("cv_accuracy", 0) * 100, 1),
        "model_auc":       meta.get("auc_roc", 0),
        "interpretation":  interpretation,
        "disclaimer":      "هذا تحليل إحصائي للأغراض البحثية فقط وليس تشخيصاً طبياً",
    }


# ══════════════════════════════════════════════════════════════
#  5. التنبؤ لـ DataFrame كامل (كل الصفوف)
# ══════════════════════════════════════════════════════════════

def predict_dataframe(df: pd.DataFrame, disease: str = None) -> dict:
    """
    يعمل تنبؤ لكل صفوف الـ DataFrame ويرجع:
      - نتائج كل صف
      - إحصائيات مجمّعة (توزيع الخطورة، متوسط الاحتمال)
      - المقارنة بين الحالات الفعلية (لو Outcome موجود)
    """
    # كشف المرض
    if not disease:
        disease = detect_disease(list(df.columns))
        if not disease:
            return {"error": "لم يتم التعرف على نوع المرض من أسماء الأعمدة"}

    bundle = load_disease_model(disease)
    model  = bundle["model"]
    scaler = bundle["scaler"]
    meta   = bundle["meta"]
    config = bundle["config"]
    labels = config["labels"]

    features    = config["features"]
    target_col  = config.get("target_col")
    thresholds  = meta.get("risk_thresholds", {"low": 0.3, "moderate": 0.5, "high": 0.7})

    # استخراج الأعمدة المتاحة فقط
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]

    # ملء الناقص بالـ mean
    X_df = pd.DataFrame(index=df.index)
    for feat in features:
        if feat in df.columns:
            X_df[feat] = pd.to_numeric(df[feat], errors='coerce')
        else:
            X_df[feat] = meta.get("feature_ranges", {}).get(feat, {}).get("mean", 0.0)

    X_df.fillna(X_df.mean(), inplace=True)
    X = X_df[features].values
    X_scaled = scaler.transform(X)

    # التنبؤ
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions   = model.predict(X_scaled)

    # تحديد مستوى الخطورة لكل صف
    def get_risk(p):
        if p < thresholds["low"]:      return "low"
        elif p < thresholds["moderate"]: return "moderate"
        elif p < thresholds["high"]:    return "high"
        else:                           return "very_high"

    risk_levels = [get_risk(p) for p in probabilities]

    # نتائج كل صف
    row_results = []
    for i in range(len(df)):
        row_results.append({
            "row":             i + 1,
            "probability":     round(float(probabilities[i]), 4),
            "probability_pct": round(float(probabilities[i]) * 100, 1),
            "prediction":      int(predictions[i]),
            "prediction_label": labels[int(predictions[i])],
            "risk_level":      risk_levels[i],
        })

    # إحصائيات مجمّعة
    risk_counts = {
        "low":       risk_levels.count("low"),
        "moderate":  risk_levels.count("moderate"),
        "high":      risk_levels.count("high"),
        "very_high": risk_levels.count("very_high"),
    }

    predicted_positive = int(predictions.sum())
    total              = len(predictions)

    summary = {
        "total_cases":          total,
        "predicted_diabetic":   predicted_positive,
        "predicted_non_diabetic": total - predicted_positive,
        "prevalence_pct":       round(predicted_positive / total * 100, 1),
        "avg_probability":      round(float(probabilities.mean()) * 100, 1),
        "risk_distribution":    risk_counts,
        "missing_features":     missing,
        "available_features":   available,
    }

    # مقارنة بالنتائج الفعلية لو Outcome موجود
    actual_comparison = None
    if target_col and target_col in df.columns:
        actual = pd.to_numeric(df[target_col], errors='coerce').dropna().astype(int)
        if len(actual) == total:
            from sklearn.metrics import (accuracy_score, roc_auc_score,
                                         confusion_matrix)
            acc = accuracy_score(actual, predictions)
            auc = roc_auc_score(actual, probabilities)
            cm  = confusion_matrix(actual, predictions).tolist()
            actual_comparison = {
                "accuracy":          round(float(acc) * 100, 1),
                "auc_roc":           round(float(auc), 3),
                "confusion_matrix":  cm,
                "true_positive":     cm[1][1],
                "true_negative":     cm[0][0],
                "false_positive":    cm[0][1],
                "false_negative":    cm[1][0],
            }

    return {
        "disease":            disease,
        "display_name":       config["display_name"],
        "summary":            summary,
        "row_results":        row_results,
        "actual_comparison":  actual_comparison,
        "model_accuracy":     round(meta.get("cv_accuracy", 0) * 100, 1),
        "model_auc":          meta.get("auc_roc", 0),
        "disclaimer":         "هذا تحليل إحصائي للأغراض البحثية فقط وليس تشخيصاً طبياً",
    }


# ══════════════════════════════════════════════════════════════
#  6. قائمة الأمراض المتاحة
# ══════════════════════════════════════════════════════════════

def list_available_diseases() -> list:
    """يرجع قائمة بالأمراض المتاحة للتنبؤ"""
    result = []
    for disease, config in DISEASE_REGISTRY.items():
        model_path = os.path.join(MODELS_DIR, config["model_file"])
        result.append({
            "id":           disease,
            "name":         config["display_name"],
            "features":     config["features"],
            "available":    os.path.exists(model_path),
        })
    return result
