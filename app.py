
from flask import Flask, render_template, request, Response
import pandas as pd
import joblib
import os
import sqlite3
import json
import csv
import io
from html import escape
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

app = Flask(__name__)

# =====================================
# PATHS AND MODEL
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model_compressed.pkl")
DB_PATH = os.path.join(BASE_DIR, "churn_history.db")

model = joblib.load(model_path)

# =====================================
# CATEGORIES
# =====================================

categories = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
}

# =====================================
# DATABASE
# =====================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_database():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_data TEXT NOT NULL,
            prediction TEXT NOT NULL,
            churn_probability REAL NOT NULL,
            stay_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()

# =====================================
# HOME
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# PREDICTION
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        customer = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": float(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # Validate input
        if customer["SeniorCitizen"] not in [0, 1]:
            raise ValueError("Invalid Senior Citizen value.")

        if not 0 <= customer["tenure"] <= 1000:
            raise ValueError("Invalid tenure.")

        if customer["MonthlyCharges"] < 0 or customer["TotalCharges"] < 0:
            raise ValueError("Charges cannot be negative.")

        for column, levels in categories.items():
            if customer[column] not in levels:
                raise ValueError(f"Invalid value for {column}.")

        # Convert to DataFrame
        customer_df = pd.DataFrame([customer])

        for column, levels in categories.items():
            customer_df[column] = pd.Categorical(
                customer_df[column],
                categories=levels
            )

        # Encoding
        customer_encoded = pd.get_dummies(
            customer_df,
            columns=list(categories.keys()),
            drop_first=True,
            dtype=int
        )

        # Match model training columns
        customer_encoded = customer_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(customer_encoded)[0]
        probabilities = model.predict_proba(customer_encoded)[0]

        classes = list(model.classes_)

        churn_probability = round(
            float(probabilities[classes.index(1)]) * 100, 2
        )

        stay_probability = round(
            float(probabilities[classes.index(0)]) * 100, 2
        )

        if prediction == 1:
            result = "Customer is likely to leave."
            status = "churn"
            prediction_label = "Churn"
        else:
            result = "Customer is likely to stay."
            status = "stay"
            prediction_label = "Stay"

        # Save to database
        conn = get_db_connection()

        conn.execute("""
            INSERT INTO predictions
            (customer_data, prediction, churn_probability,
             stay_probability, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            json.dumps(customer),
            prediction_label,
            churn_probability,
            stay_probability,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            prediction=result,
            status=status,
            churn_probability=churn_probability,
            stay_probability=stay_probability
        )

    except (ValueError, KeyError, TypeError) as e:
        return f"Invalid customer information: {str(e)}", 400

    except Exception:
        app.logger.exception("Prediction failed")
        return "An unexpected prediction error occurred.", 500


# =====================================
# PREDICTION HISTORY
# =====================================

@app.route("/history")
def history():

    conn = get_db_connection()

    records = conn.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template("history.html", records=records)


# =====================================
# DOWNLOAD CSV REPORT
# =====================================

@app.route("/download_report")
def download_report():

    conn = get_db_connection()

    records = conn.execute("""
        SELECT id, customer_data, prediction,
               churn_probability, stay_probability, created_at
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    output = io.StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Customer Data",
        "Prediction",
        "Churn Probability (%)",
        "Stay Probability (%)",
        "Created At"
    ])

    for record in records:
        writer.writerow([
            record["id"],
            record["customer_data"],
            record["prediction"],
            record["churn_probability"],
            record["stay_probability"],
            record["created_at"]
        ])

    csv_data = output.getvalue().encode("utf-8-sig")

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=churn_prediction_report.csv"
        }
    )


# =====================================
# DOWNLOAD COMPLETE PDF REPORT
# =====================================

@app.route("/download_pdf")
def download_pdf():

    conn = get_db_connection()

    records = conn.execute("""
        SELECT id, customer_data, prediction,
               churn_probability, stay_probability, created_at
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#192957"),
        fontSize=19,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#243b80"),
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        wordWrap="CJK"
    )

    elements = []

    # 1. Report title and generation date
    elements.append(
        Paragraph(
            "CUSTOMER CHURN PREDICTION REPORT",
            title_style
        )
    )

    generation_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    elements.append(
        Paragraph(
            f"Report Generated: {generation_date}",
            normal_style
        )
    )

    elements.append(Spacer(1, 15))

    # 2. Total predictions
    total = len(records)

    elements.append(
        Paragraph("Report Overview", heading_style)
    )

    elements.append(
        Paragraph(
            f"Total Customer Predictions: <b>{total}</b>",
            normal_style
        )
    )

    elements.append(Spacer(1, 10))

    # 3. Churn count and Stay count
    churn_count = sum(
        1 for record in records
        if record["prediction"] == "Churn"
    )

    stay_count = sum(
        1 for record in records
        if record["prediction"] == "Stay"
    )

    summary_data = [
        ["Total Predictions", "Churn", "Stay"],
        [str(total), str(churn_count), str(stay_count)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[55 * mm, 55 * mm, 55 * mm]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#192957")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1),
         colors.HexColor("#edf1fa")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white)
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 18))

    # 4, 5 and 6. Customer details, prediction,
    # probabilities and prediction date/time

    elements.append(
        Paragraph(
            "Individual Customer Prediction Records",
            heading_style
        )
    )

    if not records:
        elements.append(
            Paragraph(
                "No prediction history available.",
                normal_style
            )
        )

    for record in records:

        elements.append(
            Paragraph(
                f"Customer Prediction #{record['id']}",
                heading_style
            )
        )

        # 5. Prediction and probabilities
        prediction_data = [
            ["Prediction", "Churn Probability", "Stay Probability"],
            [
                record["prediction"],
                f"{record['churn_probability']}%",
                f"{record['stay_probability']}%"
            ]
        ]

        prediction_table = Table(
            prediction_data,
            colWidths=[55 * mm, 55 * mm, 55 * mm]
        )

        prediction_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#243b80")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1),
             colors.HexColor("#f1f4f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.white)
        ]))

        elements.append(prediction_table)
        elements.append(Spacer(1, 8))

        # 4. Customer details
        try:
            customer = json.loads(record["customer_data"])
        except (ValueError, TypeError):
            customer = {}

        customer_rows = [
            ["Customer Field", "Customer Value"]
        ]

        for key, value in customer.items():
            customer_rows.append([
                Paragraph(escape(str(key)), small_style),
                Paragraph(escape(str(value)), small_style)
            ])

        customer_table = Table(
            customer_rows,
            colWidths=[55 * mm, 110 * mm],
            repeatRows=1
        )

        customer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#192957")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1),
             colors.HexColor("#f7f8fb")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4,
             colors.HexColor("#dce2ed")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ]))

        elements.append(
            Paragraph("Customer Details", styles["Heading3"])
        )

        elements.append(customer_table)
        elements.append(Spacer(1, 8))

        # 6. Prediction date and time
        elements.append(
            Paragraph(
                "<b>Prediction Date and Time:</b> "
                + escape(str(record["created_at"])),
                normal_style
            )
        )

        elements.append(Spacer(1, 18))

    # Page number
    def add_page_number(canvas, document):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)

        canvas.drawRightString(
            195 * mm,
            10 * mm,
            f"Page {document.page}"
        )

        canvas.restoreState()

    # Generate PDF
    pdf.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={
            "Content-Disposition":
            "attachment; filename=customer_churn_full_report.pdf"
        }
    )


# =====================================
# INTERACTIVE DASHBOARD
# =====================================

@app.route("/dashboard")
def dashboard():

    conn = get_db_connection()

    # Summary statistics
    total = conn.execute("""
        SELECT COUNT(*) FROM predictions
    """).fetchone()[0]

    churn_count = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Churn'
    """).fetchone()[0]

    stay_count = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE prediction = 'Stay'
    """).fetchone()[0]

    churn_rate = round(
        (churn_count / total) * 100, 2
    ) if total else 0

    # Monthly prediction statistics
    monthly_records = conn.execute("""
        SELECT
            substr(created_at, 1, 7) AS month,
            COUNT(*) AS total,
            SUM(CASE WHEN prediction = 'Churn' THEN 1 ELSE 0 END) AS churn,
            SUM(CASE WHEN prediction = 'Stay' THEN 1 ELSE 0 END) AS stay
        FROM predictions
        GROUP BY month
        ORDER BY month
    """).fetchall()

    # Churn probability distribution
    probability_records = conn.execute("""
        SELECT
            CASE
                WHEN churn_probability < 20 THEN '0-19%'
                WHEN churn_probability < 40 THEN '20-39%'
                WHEN churn_probability < 60 THEN '40-59%'
                WHEN churn_probability < 80 THEN '60-79%'
                ELSE '80-100%'
            END AS probability_range,
            COUNT(*) AS total
        FROM predictions
        GROUP BY
            CASE
                WHEN churn_probability < 20 THEN 1
                WHEN churn_probability < 40 THEN 2
                WHEN churn_probability < 60 THEN 3
                WHEN churn_probability < 80 THEN 4
                ELSE 5
            END
        ORDER BY
            CASE
                WHEN churn_probability < 20 THEN 1
                WHEN churn_probability < 40 THEN 2
                WHEN churn_probability < 60 THEN 3
                WHEN churn_probability < 80 THEN 4
                ELSE 5
            END
    """).fetchall()

    # Recent predictions
    recent_records = conn.execute("""
        SELECT id, customer_data, prediction,
               churn_probability, stay_probability, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    # Prepare chart data
    monthly_labels = [row["month"] for row in monthly_records]
    monthly_total = [row["total"] for row in monthly_records]
    monthly_churn = [row["churn"] for row in monthly_records]
    monthly_stay = [row["stay"] for row in monthly_records]

    probability_labels = [
        row["probability_range"] for row in probability_records
    ]

    probability_totals = [
        row["total"] for row in probability_records
    ]

    return render_template(
        "dashboard.html",
        total=total,
        churn_count=churn_count,
        stay_count=stay_count,
        churn_rate=churn_rate,
        recent_records=recent_records,
        monthly_labels=monthly_labels,
        monthly_total=monthly_total,
        monthly_churn=monthly_churn,
        monthly_stay=monthly_stay,
        probability_labels=probability_labels,
        probability_totals=probability_totals
    )


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":
    app.run(debug=True)
