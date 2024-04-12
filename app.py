from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def calculate_similarity(correct_answer, student_answer):
    vectorizer = TfidfVectorizer().fit_transform([correct_answer, student_answer])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

def convert_to_marks(similarity_score, marks):
    if marks == 5:
        if similarity_score >= 81:
            return 5
        elif similarity_score >= 61:
            return 4
        elif similarity_score >= 51:
            return 3
        elif similarity_score >= 41:
            return 2
        elif similarity_score >= 25:
            return 1
        else:
            return 0
    elif marks == 7:
        if similarity_score >= 95:
            return 7
        elif similarity_score >= 86:
            return 6
        elif similarity_score >= 71:
            return 5
        elif similarity_score >= 61:
            return 4
        elif similarity_score >= 51:
            return 3
        elif similarity_score >= 35:
            return 2
        elif similarity_score >= 15:
            return 1
        else:
            return 0
    elif marks == 8:
        if similarity_score >= 86:
            return 8
        elif similarity_score >= 76:
            return 7
        elif similarity_score >= 61:
            return 6
        elif similarity_score >= 51:
            return 5
        elif similarity_score >= 41:
            return 4
        elif similarity_score >= 35:
            return 3
        elif similarity_score >= 25:
            return 2
        elif similarity_score >= 15:
            return 1
        else:
            return 0
    elif marks == 10:
        if similarity_score >= 91:
            return 10
        elif similarity_score >= 86:
            return 9
        elif similarity_score >= 76:
            return 8
        elif similarity_score >= 61:
            return 7
        elif similarity_score >= 51:
            return 6
        elif similarity_score >= 41:
            return 5
        elif similarity_score >= 31:
            return 4
        elif similarity_score >= 21:
            return 3
        elif similarity_score >= 11:
            return 2
        elif similarity_score >= 5:
            return 1
        else:
            return 0
    elif marks == 12:
        if similarity_score >= 96:
            return 12
        elif similarity_score >= 91:
            return 11
        elif similarity_score >= 86:
            return 10
        elif similarity_score >= 76:
            return 9
        elif similarity_score >= 70:
            return 8
        elif similarity_score >= 61:
            return 7
        elif similarity_score >= 50:
            return 6
        elif similarity_score >= 41:
            return 5
        elif similarity_score >= 31:
            return 4
        elif similarity_score >= 25:
            return 3
        elif similarity_score >= 15:
            return 2
        elif similarity_score >= 10:
            return 1
        else:
            return 0
    else:
        return 0

def compare_answers(docx_file):
    doc = Document(docx_file)
    data = []
    total_marks = 0
    marks_obtained = 0
    for table in doc.tables:
        for row in table.rows[1:]:  # Assuming first row contains headers
            srno, question, correct_answer, student_answer, marks = [cell.text for cell in row.cells]
            total_marks += int(marks)
            similarity_score = calculate_similarity(correct_answer, student_answer) * 100
            converted_marks = convert_to_marks(similarity_score, int(marks))
            marks_obtained += converted_marks
            data.append([srno, similarity_score, marks, converted_marks])

    df = pd.DataFrame(data, columns=['Sr.No', 'Similarity Score', 'Total Marks', 'Marks Obtained'])
    percentage_obtained = (marks_obtained / total_marks) * 100
    return df, total_marks, marks_obtained, percentage_obtained

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result_df, total_marks, marks_obtained, percentage_obtained = compare_answers(file_path)
            return render_template('result.html', result=result_df.to_html(), total_marks=total_marks, marks_obtained=marks_obtained, percentage_obtained=percentage_obtained)

    return 'Error in file upload'

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
