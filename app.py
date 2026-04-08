from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

nltk.download('stopwords')

app = Flask(__name__)

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- PREPROCESSING ----------------
def preprocess(text):
    words = text.lower().split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# ---------------- KEYWORDS ----------------
def get_keywords(text):
    return set(text.split())

# ---------------- PDF REPORT ----------------
def create_pdf(match, match_skills, missing_skills):
    doc = SimpleDocTemplate("static/report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Resume Match Report", styles['Title']))
    content.append(Paragraph(f"Match Percentage: {match}%", styles['Normal']))

    content.append(Paragraph("Matching Skills:", styles['Heading2']))
    content.append(Paragraph(", ".join(match_skills), styles['Normal']))

    content.append(Paragraph("Missing Skills:", styles['Heading2']))
    content.append(Paragraph(", ".join(missing_skills), styles['Normal']))

    doc.build(content)

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    file = request.files['resume']
    jd = request.form['jd']

    # Extract text
    resume_text = extract_pdf(file)

    # Preprocess
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd)

    # Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])

    # Similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    match_percent = round(similarity * 100, 2)

    # Skills
    resume_words = get_keywords(resume_clean)
    jd_words = get_keywords(jd_clean)

    match_skills = resume_words.intersection(jd_words)
    missing_skills = jd_words - resume_words

    # Graph
    labels = ['Match', 'Missing']
    values = [len(match_skills), len(missing_skills)]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Skill Comparison")
    plt.savefig('static/graph.png')
    plt.close()

    # PDF Report
    create_pdf(match_percent, match_skills, missing_skills)

    return render_template('index.html',
                           match=match_percent,
                           match_skills=match_skills,
                           missing_skills=missing_skills)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)