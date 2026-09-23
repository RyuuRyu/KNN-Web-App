import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = ROOT_DIR / "training_report.json"
DEFAULT_PDF_PATH = ROOT_DIR / "training_report.pdf"


def percent(value):
	return f"{value * 100:.2f}%"


def make_table(data, widths=None):
	table = Table(data, colWidths=widths, repeatRows=1)
	table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
		("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
		("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c9d6")),
		("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
		("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
		("LEFTPADDING", (0, 0), (-1, -1), 6),
		("RIGHTPADDING", (0, 0), (-1, -1), 6),
		("TOPPADDING", (0, 0), (-1, -1), 6),
		("BOTTOMPADDING", (0, 0), (-1, -1), 6),
	]))
	return table


def generate_pdf_report(report_data, output_path=DEFAULT_PDF_PATH):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	styles = getSampleStyleSheet()
	styles.add(ParagraphStyle(
		name="ReportTitle",
		parent=styles["Title"],
		alignment=TA_CENTER,
		textColor=colors.HexColor("#1f4e79"),
		spaceAfter=18,
	))
	styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))

	story = [
		Paragraph("Laporan Training Model KNN", styles["ReportTitle"]),
		Paragraph(
			f"Dibuat: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
			styles["Small"],
		),
		Spacer(1, 0.2 * inch),
		Paragraph("Ringkasan", styles["Heading2"]),
	]

	summary = [
		["Metrik", "Hasil"],
		["Dataset", report_data["dataset"]],
		["Jumlah baris setelah pembersihan", str(report_data["total_rows_after_cleaning"])],
		["Jumlah fitur", str(report_data["feature_count"])],
		["Akurasi rata-rata cross-validation", percent(report_data["mean_cross_validation_accuracy"])],
		["Akurasi data testing", percent(report_data["test_accuracy"])],
	]
	story.append(make_table(summary, [3.5 * inch, 3.2 * inch]))
	story.extend([Spacer(1, 0.2 * inch), Paragraph("Akurasi Cross-Validation", styles["Heading2"])])
	scores = ", ".join(percent(score) for score in report_data["cross_validation_scores"])
	story.append(Paragraph(f"Skor setiap fold: {scores}", styles["BodyText"]))

	story.extend([Spacer(1, 0.2 * inch), Paragraph("Classification Report", styles["Heading2"])])
	classification = report_data["classification_report"]
	classification_rows = [["Kelas", "Precision", "Recall", "F1-score", "Support"]]
	for label in ("TIDAK LULUS", "LULUS"):
		values = classification[label]
		classification_rows.append([
			label,
			percent(values["precision"]),
			percent(values["recall"]),
			percent(values["f1-score"]),
			str(int(values["support"])),
		])
	classification_rows.append([
		"Accuracy", "", "", percent(classification["accuracy"]),
		str(int(classification["macro avg"]["support"])),
	])
	for label, display in (("macro avg", "Macro average"), ("weighted avg", "Weighted average")):
		values = classification[label]
		classification_rows.append([
			display,
			percent(values["precision"]),
			percent(values["recall"]),
			percent(values["f1-score"]),
			str(int(values["support"])),
		])
	story.append(make_table(classification_rows, [1.6 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch]))

	story.extend([Spacer(1, 0.2 * inch), Paragraph("Distribusi Kelas Data Training", styles["Heading2"])])
	before = report_data["training_class_distribution_before_smote"]
	after = report_data["training_class_distribution_after_smote"]
	distribution = [
		["Kelas", "Sebelum SMOTE", "Sesudah SMOTE"],
		["TIDAK LULUS (0)", str(before["0"]), str(after["0"])],
		["LULUS (1)", str(before["1"]), str(after["1"])],
	]
	story.append(make_table(distribution, [2.4 * inch, 1.8 * inch, 1.8 * inch]))

	story.extend([Spacer(1, 0.2 * inch), Paragraph("Fitur Model", styles["Heading2"])])
	story.append(Paragraph(", ".join(report_data["features"]), styles["BodyText"]))

	image_paths = [
		ROOT_DIR / "image" / "weighted_ipk_distribution.png",
		ROOT_DIR / "image" / "distribusi_kelas_sebelum_smote.png",
		ROOT_DIR / "image" / "distribusi_kelas_sesudah_smote.png",
	]
	existing_images = [path for path in image_paths if path.exists()]
	if existing_images:
		story.append(PageBreak())
		story.append(Paragraph("Visualisasi Training", styles["Heading2"]))
		for image_path in existing_images:
			story.append(Image(str(image_path), width=6.4 * inch, height=3.8 * inch))
			story.append(Spacer(1, 0.1 * inch))

	SimpleDocTemplate(
		str(output_path),
		pagesize=A4,
		rightMargin=0.55 * inch,
		leftMargin=0.55 * inch,
		topMargin=0.55 * inch,
		bottomMargin=0.55 * inch,
		title="Laporan Training Model KNN",
	).build(story)
	return output_path


def create_pdf_from_json(json_path=DEFAULT_JSON_PATH, output_path=DEFAULT_PDF_PATH):
	with Path(json_path).open("r", encoding="utf-8") as file:
		report_data = json.load(file)
	return generate_pdf_report(report_data, output_path)


if __name__ == "__main__":
	pdf_path = create_pdf_from_json()
	print(f"PDF report saved to: {pdf_path}")
