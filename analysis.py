import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional

# ══════════════════════════════════════════════════════════════
#  StatLab — analysis.py
#  التحليل الإحصائي + التنبؤ بالأمراض
#
#  Endpoints:
#    POST /upload            → descriptive + frequency + correlation
#                              + كشف تلقائي + تنبؤ لو الداتا طبية
#    POST /analyze/ttest     → Independent T-Test
#    POST /analyze/ztest     → Z-Test للنسب (JSON)
#    POST /analyze/chisquare → Chi-Square
#    POST /predict           → تنبؤ صريح (يقدر يبعت مرض محدد)
#    GET  /diseases          → قائمة الأمراض المتاحة
#    GET  /health            → التحقق من الـ backend
# ══════════════════════════════════════════════════════════════


def load_data(filepath: str) -> pd.DataFrame:
    """قراءة ملف Excel أو CSV"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, encoding='utf-8')
    return pd.read_excel(filepath)


def load_data_from_dict(headers: list, rows: list) -> pd.DataFrame:
    """تحويل البيانات القادمة من الفرونت لـ DataFrame"""
    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df


def get_numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ──────────────────────────────────────────────
#  1. Descriptive Statistics
# ──────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame) -> dict:
    results = {}
    for col in get_numeric_cols(df):
        data = df[col].dropna()
        if len(data) == 0:
            continue
        n  = len(data)
        sd = float(data.std())
        results[col] = {
            "N":         n,
            "Mean":      round(float(data.mean()),   4),
            "Median":    round(float(data.median()), 4),
            "Std_Dev":   round(sd,                   4),
            "Std_Error": round(sd / np.sqrt(n),      4),
            "Min":       round(float(data.min()),    4),
            "Max":       round(float(data.max()),    4),
            "Range":     round(float(data.max() - data.min()), 4),
            "Skewness":  round(float(data.skew()),      4),
            "Kurtosis":  round(float(data.kurtosis()),  4),
            "Missing":   int(df[col].isna().sum()),
        }
    return results


# ──────────────────────────────────────────────
#  2. Frequency Analysis
# ──────────────────────────────────────────────

def frequency_analysis(df: pd.DataFrame) -> dict:
    results = {}
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if 2 <= len(unique_vals) <= 15:
            freq  = df[col].value_counts()
            total = freq.sum()
            results[col] = [
                {
                    "value":      str(val),
                    "count":      int(count),
                    "percentage": round(count / total * 100, 2),
                    "cumulative": round(freq[:i+1].sum() / total * 100, 2),
                }
                for i, (val, count) in enumerate(freq.items())
            ]
    return results


# ──────────────────────────────────────────────
#  3. Correlation
# ──────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> dict:
    numeric_df = df[get_numeric_cols(df)].dropna()
    cols = numeric_df.columns.tolist()
    if len(cols) < 2:
        return {"columns": cols, "correlation_matrix": {}, "pvalue_matrix": {}}

    corr_matrix, pval_matrix = {}, {}
    for c1 in cols:
        corr_matrix[c1], pval_matrix[c1] = {}, {}
        for c2 in cols:
            if c1 == c2:
                corr_matrix[c1][c2] = 1.0
                pval_matrix[c1][c2] = 0.0
            else:
                r, p = stats.pearsonr(numeric_df[c1], numeric_df[c2])
                corr_matrix[c1][c2] = round(float(r), 4)
                pval_matrix[c1][c2] = round(float(p), 4)

    return {"columns": cols, "correlation_matrix": corr_matrix, "pvalue_matrix": pval_matrix}


# ──────────────────────────────────────────────
#  4. Independent T-Test
# ──────────────────────────────────────────────

def independent_ttest(df, column, group_column, group1, group2):
    g1 = df[df[group_column] == group1][column].dropna()
    g2 = df[df[group_column] == group2][column].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "حجم المجموعة صغير جداً (أقل من 2)"}

    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
    levene_stat, levene_p = stats.levene(g1, g2)
    significant = bool(p_value < 0.05)

    return {
        "column": column,
        "group1": {"name": str(group1), "N": int(len(g1)),
                   "Mean": round(float(g1.mean()), 4), "SD": round(float(g1.std()), 4),
                   "SE":   round(float(g1.std() / np.sqrt(len(g1))), 4)},
        "group2": {"name": str(group2), "N": int(len(g2)),
                   "Mean": round(float(g2.mean()), 4), "SD": round(float(g2.std()), 4),
                   "SE":   round(float(g2.std() / np.sqrt(len(g2))), 4)},
        "t_statistic":    round(float(t_stat),  4),
        "p_value":        round(float(p_value), 4),
        "df":             len(g1) + len(g2) - 2,
        "significant":    significant,
        "interpretation": "يوجد فرق دال إحصائياً" if significant else "لا يوجد فرق دال إحصائياً",
        "levene": {"statistic": round(float(levene_stat), 4),
                   "p_value":   round(float(levene_p),    4)},
    }


# ──────────────────────────────────────────────
#  5. Z-Test للنسب
# ──────────────────────────────────────────────

def _binary_series(series, sorted_vals):
    return (series == sorted_vals[1]).astype(int)


def z_test_one_sample(df, column, p0=0.5, alternative="two-sided"):
    col_data    = df[column].dropna()
    unique_vals = col_data.unique()
    if len(unique_vals) != 2:
        return {"error": f"العمود '{column}' لازم يكون ثنائي (قيمتين فقط)"}

    sorted_vals = sorted(unique_vals)
    binary = _binary_series(col_data, sorted_vals)
    n, p_hat = len(binary), float(binary.mean())
    se = np.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return {"error": "الانحراف المعياري = صفر"}

    z_stat = (p_hat - p0) / se
    if alternative == "two-sided": p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    elif alternative == "greater":  p_value = float(1 - stats.norm.cdf(z_stat))
    else:                           p_value = float(stats.norm.cdf(z_stat))

    significant = bool(p_value < 0.05)
    z_crit      = 1.96
    return {
        "test_type": "one_sample", "alternative": alternative, "column": column,
        "categories": [str(v) for v in sorted_vals],
        "n": int(n), "p_hat": round(p_hat, 4), "p0": round(p0, 4),
        "z_statistic": round(float(z_stat), 4), "p_value": round(p_value, 4),
        "significant": significant,
        "confidence_interval_95": {
            "lower": round(p_hat - z_crit * np.sqrt(p_hat*(1-p_hat)/n), 4),
            "upper": round(p_hat + z_crit * np.sqrt(p_hat*(1-p_hat)/n), 4),
        },
        "interpretation": "النسبة تختلف معنوياً عن p₀" if significant else "النسبة لا تختلف معنوياً عن p₀",
    }


def z_test_two_sample(df, group_column, success_column, alternative="two-sided"):
    groups = df[group_column].dropna().unique()
    if len(groups) != 2:
        return {"error": f"عمود المجموعة لازم يكون فيه قيمتين فقط"}

    sorted_groups = sorted(groups)
    g1_data = df[df[group_column] == sorted_groups[0]][success_column].dropna()
    g2_data = df[df[group_column] == sorted_groups[1]][success_column].dropna()

    all_vals    = pd.concat([g1_data, g2_data])
    unique_vals = all_vals.unique()
    if len(unique_vals) != 2:
        return {"error": "عمود النجاح لازم يكون ثنائي"}

    sorted_vals = sorted(unique_vals)
    g1_bin, g2_bin = _binary_series(g1_data, sorted_vals), _binary_series(g2_data, sorted_vals)
    n1, n2 = len(g1_bin), len(g2_bin)
    p1, p2 = float(g1_bin.mean()), float(g2_bin.mean())
    p_pool = (g1_bin.sum() + g2_bin.sum()) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return {"error": "الانحراف المعياري = صفر"}

    z_stat = (p1 - p2) / se
    if alternative == "two-sided": p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    elif alternative == "greater":  p_value = float(1 - stats.norm.cdf(z_stat))
    else:                           p_value = float(stats.norm.cdf(z_stat))

    significant = bool(p_value < 0.05)
    return {
        "test_type": "two_sample", "alternative": alternative,
        "group1": {"name": str(sorted_groups[0]), "N": int(n1),
                   "proportion": round(p1, 4), "successes": int(g1_bin.sum())},
        "group2": {"name": str(sorted_groups[1]), "N": int(n2),
                   "proportion": round(p2, 4), "successes": int(g2_bin.sum())},
        "pooled_proportion": round(float(p_pool), 4),
        "z_statistic": round(float(z_stat), 4), "p_value": round(p_value, 4),
        "significant": significant,
        "interpretation": "يوجد فرق دال إحصائياً بين النسبتين" if significant else "لا يوجد فرق دال إحصائياً بين النسبتين",
    }


# ──────────────────────────────────────────────
#  6. Chi-Square
# ──────────────────────────────────────────────

def chi_square_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {
        "column1": col1, "column2": col2,
        "chi2_statistic": round(float(chi2), 4), "p_value": round(float(p), 4),
        "degrees_of_freedom": int(dof), "significant": bool(p < 0.05),
        "interpretation": "توجد علاقة دالة إحصائياً" if p < 0.05 else "لا توجد علاقة دالة إحصائياً",
        "contingency_table": contingency.to_dict(),
    }


# ──────────────────────────────────────────────
#  7. Variable Summary
# ──────────────────────────────────────────────

def variable_summary(df: pd.DataFrame) -> list:
    summary = []
    for i, col in enumerate(df.columns):
        data, non_null = df[col], df[col].dropna()
        is_num = pd.api.types.is_numeric_dtype(data)
        summary.append({
            "index": i+1, "name": col,
            "type":    "Numeric" if is_num else "String",
            "count":   int(non_null.count()),
            "missing": int(data.isna().sum()),
            "width":   int(data.astype(str).str.len().max()) if len(non_null) > 0 else 0,
            "min":  round(float(non_null.min()),  4) if is_num and len(non_null) else None,
            "max":  round(float(non_null.max()),  4) if is_num and len(non_null) else None,
            "mean": round(float(non_null.mean()), 4) if is_num and len(non_null) else None,
        })
    return summary


# ──────────────────────────────────────────────
#  Helper: تحميل predict.py بأمان
# ──────────────────────────────────────────────

def _get_predictor():
    """يحمّل وحدة predict.py لو متاحة"""
    try:
        import predict
        return predict
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════
#  Flask App
# ══════════════════════════════════════════════════════════════

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import io

    app = Flask(__name__)
    CORS(app)

    # ─────────────────────────────────────────
    #  Helper داخلي — قراءة الملف من request
    # ─────────────────────────────────────────
    def _read_file_from_request() -> pd.DataFrame:
        file = request.files.get('file')
        if not file:
            raise ValueError("لم يتم إرسال ملف")
        fname = file.filename or 'data.csv'
        if fname.endswith('.csv'):
            return pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        return pd.read_excel(io.BytesIO(file.read()))


    # ─────────────────────────────────────────
    #  /upload — تحليل كامل + تنبؤ تلقائي
    # ─────────────────────────────────────────
    @app.route('/upload', methods=['POST'])
    def upload():
        try:
            df     = _read_file_from_request()
            domain = request.form.get('domain', 'medical')

            data = {
                "domain":      domain,
                "shape":       {"rows": len(df), "columns": len(df.columns)},
                "variables":   variable_summary(df),
                "descriptive": descriptive_statistics(df),
                "frequency":   frequency_analysis(df),
                "correlation": correlation_analysis(df),
            }

            # ── تنبؤ تلقائي لو الداتا طبية ──
            predictor = _get_predictor()
            if predictor:
                disease = predictor.detect_disease(list(df.columns))
                if disease:
                    pred_result = predictor.predict_dataframe(df, disease=disease)
                    data["prediction"] = pred_result

            return jsonify({"success": True, "data": data})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /analyze/ttest
    # ─────────────────────────────────────────
    @app.route('/analyze/ttest', methods=['POST'])
    def analyze_ttest():
        try:
            df           = _read_file_from_request()
            column       = request.form.get('column')
            group_column = request.form.get('group_column')
            group1       = request.form.get('group1')
            group2       = request.form.get('group2')

            # اكتشاف تلقائي للأعمدة لو مش محددة
            if not column or not group_column:
                num_cols = get_numeric_cols(df)
                cat_cols = [c for c in df.columns if c not in num_cols]
                if not num_cols or not cat_cols:
                    return jsonify({"success": False, "error": "لازم يكون في عمود رقمي وعمود فئوي"}), 400
                column, group_column = num_cols[0], cat_cols[0]

            if not group1 or not group2:
                groups = df[group_column].dropna().unique()
                if len(groups) < 2:
                    return jsonify({"success": False, "error": f"العمود '{group_column}' يحتاج قيمتين على الأقل"}), 400
                group1, group2 = str(sorted(groups)[0]), str(sorted(groups)[1])

            result = independent_ttest(df, column, group_column, group1, group2)
            if "error" in result:
                return jsonify({"success": False, "error": result["error"]}), 400

            return jsonify({"success": True, "data": {"ttest": result}})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /analyze/ztest — يستقبل JSON مباشرة
    # ─────────────────────────────────────────
    @app.route('/analyze/ztest', methods=['POST'])
    def analyze_ztest():
        try:
            payload     = request.get_json()
            if not payload:
                return jsonify({"success": False, "error": "لم يتم إرسال بيانات"}), 400

            test_type   = payload.get('test_type',   'one_sample')
            alternative = payload.get('alternative', 'two-sided')
            dataset     = payload.get('dataset', {})
            df          = load_data_from_dict(dataset.get('headers', []), dataset.get('rows', []))

            if test_type == 'one_sample':
                column = payload.get('column')
                p0     = float(payload.get('p0', 0.5))
                if not column:
                    return jsonify({"success": False, "error": "حدد اسم العمود"}), 400
                result = z_test_one_sample(df, column, p0, alternative)
            else:
                group_col   = payload.get('group_column')
                success_col = payload.get('success_column')
                if not group_col or not success_col:
                    return jsonify({"success": False, "error": "حدد عمود المجموعة وعمود النجاح"}), 400
                result = z_test_two_sample(df, group_col, success_col, alternative)

            if "error" in result:
                return jsonify({"success": False, "error": result["error"]}), 400
            return jsonify({"success": True, "data": {"ztest": result}})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /analyze/chisquare
    # ─────────────────────────────────────────
    @app.route('/analyze/chisquare', methods=['POST'])
    def analyze_chisquare():
        try:
            df   = _read_file_from_request()
            col1 = request.form.get('col1')
            col2 = request.form.get('col2')

            if not col1 or not col2:
                num_cols = get_numeric_cols(df)
                cat_cols = [c for c in df.columns if c not in num_cols]
                if len(cat_cols) < 2:
                    return jsonify({"success": False, "error": "لازم يكون في عمودين فئويين"}), 400
                col1, col2 = cat_cols[0], cat_cols[1]

            result = chi_square_test(df, col1, col2)
            return jsonify({"success": True, "data": {"chisquare": result}})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /predict — تنبؤ صريح
    #
    #  يقبل:
    #    FormData: file + disease (اختياري)
    #    أو JSON:  { disease, row: {...} }
    #              لتنبؤ حالة واحدة بدون ملف
    # ─────────────────────────────────────────
    @app.route('/predict', methods=['POST'])
    def predict_endpoint():
        try:
            predictor = _get_predictor()
            if not predictor:
                return jsonify({"success": False,
                                "error": "predict.py غير موجود — ضع الملف في نفس المجلد"}), 500

            # ── حالة 1: JSON (صف واحد بدون ملف)
            if request.content_type and 'application/json' in request.content_type:
                payload = request.get_json()
                row     = payload.get('row', {})
                disease = payload.get('disease')
                if not row:
                    return jsonify({"success": False, "error": "أرسل row بها القيم"}), 400

                result = predictor.predict_single(row, disease=disease)
                if "error" in result:
                    return jsonify({"success": False, "error": result["error"]}), 400
                return jsonify({"success": True, "data": {"prediction": result}})

            # ── حالة 2: FormData (ملف كامل)
            df      = _read_file_from_request()
            disease = request.form.get('disease')

            result  = predictor.predict_dataframe(df, disease=disease)
            if "error" in result:
                return jsonify({"success": False, "error": result["error"]}), 400
            return jsonify({"success": True, "data": {"prediction": result}})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /diseases — قائمة الأمراض المتاحة
    # ─────────────────────────────────────────
    @app.route('/diseases', methods=['GET'])
    def diseases_endpoint():
        try:
            predictor = _get_predictor()
            if not predictor:
                return jsonify({"success": True, "data": []})
            diseases = predictor.list_available_diseases()
            return jsonify({"success": True, "data": diseases})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


    # ─────────────────────────────────────────
    #  /health
    # ─────────────────────────────────────────
    @app.route('/health', methods=['GET'])
    def health():
        predictor = _get_predictor()
        diseases  = []
        if predictor:
            try:
                diseases = [d["id"] for d in predictor.list_available_diseases() if d["available"]]
            except Exception:
                pass
        return jsonify({
            "status":             "ok",
            "message":            "StatLab backend is running",
            "prediction_enabled": predictor is not None,
            "available_diseases": diseases,
        })


except ImportError:
    print("تحذير: Flask غير مثبت.")
    print("  pip install flask flask-cors")


# ══════════════════════════════════════════════
#  للاستخدام المباشر بدون Flask
# ══════════════════════════════════════════════

def run_full_analysis(filepath: str, domain: str = "medical") -> dict:
    df = load_data(filepath)
    data = {
        "domain":      domain,
        "shape":       {"rows": len(df), "columns": len(df.columns)},
        "variables":   variable_summary(df),
        "descriptive": descriptive_statistics(df),
        "frequency":   frequency_analysis(df),
        "correlation": correlation_analysis(df),
    }
    predictor = _get_predictor()
    if predictor:
        disease = predictor.detect_disease(list(df.columns))
        if disease:
            data["prediction"] = predictor.predict_dataframe(df, disease=disease)
    return data


if __name__ == "__main__":
    import json, sys

    # ── تشغيل Flask السيرفر (الوضع الافتراضي)
    try:
        from flask import Flask
        print("=" * 50)
        print("  StatLab Backend — جاهز")
        print("  http://localhost:5000")
        print("=" * 50)
        app.run(debug=True, port=5000)
    except NameError:
        # Flask مش مثبت — تشغيل مباشر من command line
        if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
            filepath = sys.argv[1]
            print(f"جاري تحليل: {filepath}\n")
            results = run_full_analysis(filepath)
            print(f"الصفوف:   {results['shape']['rows']}")
            print(f"الأعمدة:  {results['shape']['columns']}")
            for col, s in results['descriptive'].items():
                print(f"  {col}: Mean={s['Mean']}, SD={s['Std_Dev']}")
            if "prediction" in results:
                pred = results["prediction"]
                print(f"\nتنبؤ ({pred['display_name']}):")
                print(f"  متوسط الاحتمال:  {pred['summary']['avg_probability']}%")
                print(f"  مصابون متوقعون:  {pred['summary']['predicted_diabetic']}/{pred['summary']['total_cases']}")
            with open("results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("\nتم حفظ النتايج في results.json")
        else:
            print("الاستخدام: python analysis.py [filepath.csv]")
