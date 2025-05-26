from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_mysqldb import MySQL
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from MySQLdb import IntegrityError

from speech_analysis import( transcribe_audio_with_whisper_triple, 
                            analyze_speech_triple, 
                            extract_mfcc_features_triple)
from llm_feedback import( generate_topic_triple, 
                        generate_distractor_words,
                        get_llama_feedback_triple)
from conductor import (
    generate_topic,
    generate_expected_scores,
    split_audio,
    predict_emotion_per_chunk,
    calculate_weighted_accuracy,
    get_llama_feedback,
)

from analogy import (
    generate_questions,  
    transcribe_audio_with_whisper, 
    analyze_speech, 
    extract_mfcc_features, 
    get_llama_feedback_analogy)

app = Flask(__name__)
app.secret_key = "your_secret_key"  


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'rithika_1609'
app.config['MYSQL_DB'] = 'speechassistantdb'
mysql = MySQL(app)

with app.app_context():
    cur = mysql.connection.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        )
    ''')
    mysql.connection.commit()
    cur.close()

@app.route("/", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        cur = mysql.connection.cursor()
        cur.execute("SELECT email FROM users WHERE email = %s AND password = %s", (email, password))
        user = cur.fetchone()
        cur.close()

        if user:
            session["user"] = user[0]  
            return redirect(url_for("choose")) 
        else:
            message = "Invalid email or password! Please sign up first."

    return render_template("login.html", message=message)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name")
        email = request.form.get("email")
        password = request.form.get("password")

        # Prevent empty fields
        if not full_name or not email or not password:
            return render_template("login.html", message="All fields are required!")

        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)", (full_name, email, password))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for("login"))  
        except IntegrityError:
            return render_template("login.html", message="Account already exists! Please log in.")

    return render_template("login.html")

@app.route("/choose", methods=["GET", "POST"])
def choose():
    if "user" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        game_mode = request.form.get("game_mode")
        if game_mode == "conductor":
            return redirect(url_for("conductor"))
        elif game_mode == "triple_step":
            return redirect(url_for("triple_step"))
        elif game_mode == "rapid_fire":
            return redirect(url_for("rapid_fire"))
    
    return render_template("choose.html")

@app.route("/conductor", methods=["GET", "POST"])
def conductor():
    if "user" not in session:
        return redirect(url_for("login"))   

    topic = generate_topic()
    
    
    expected_scores = session.get("expected_scores", None)
    duration = session.get("duration", None)

    if request.method == "POST":
        if "duration" in request.form:
            duration = int(request.form.get("duration"))
            expected_scores = generate_expected_scores(duration)
            session["expected_scores"] = expected_scores
            session["duration"] = duration

            return render_template("conductor.html", topic=topic, expected_scores=expected_scores, duration=duration)

        if "audio" in request.files:
            if expected_scores is None:
                return jsonify({"error": "Expected scores are not available. Please select a duration first."}), 400

            audio_file = request.files["audio"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "recording.wav")
            audio_file.save(file_path)

            audio_chunks, sampling_rate = split_audio(file_path)
            print("audio splitting")

            model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
            model = AutoModelForAudioClassification.from_pretrained(model_id)
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
            id2label = model.config.id2label

            emotion_results = predict_emotion_per_chunk(audio_chunks, sampling_rate, model, feature_extractor, id2label, expected_scores)
            print("running emotional test")

            accuracy_score = calculate_weighted_accuracy(emotion_results)
            print("accuracy generation")

            feedback = get_llama_feedback(topic, emotion_results, accuracy_score)
            print("feedback generating")
            print(feedback)

            return jsonify({
                "message": "Audio processed successfully",
                "accuracy_score": accuracy_score,
                "feedback": feedback
            })

    return render_template("conductor.html", topic=topic, expected_scores=expected_scores, duration=duration)


@app.route("/triple_step", methods=["GET", "POST"])  
def triple_step():
    if "user" not in session:
        return redirect(url_for("login"))

    topic = generate_topic_triple()
    distractorWords = generate_distractor_words(topic)

    if request.method == "POST":
        if "audio" in request.files:
            audio_file = request.files["audio"]
            save_path = f"uploads/{audio_file.filename}"
            audio_file.save(save_path)
            print("Audio file saved:", save_path)

            trans = transcribe_audio_with_whisper_triple(save_path)
            print("Transcribed:", trans)

            analysis = analyze_speech_triple(trans, distractorWords)
            print("Analyzed:", analysis)

            mfcc = extract_mfcc_features_triple(save_path)
            print("MFCC extracted")

            feedback = get_llama_feedback_triple(topic, trans, mfcc,analysis)

            return jsonify({"feedback": feedback})

    return render_template("triple_step.html", topic=topic, distractorWords=distractorWords)


@app.route("/rapid_fire", methods=["GET", "POST"]) 
def rapid_fire():
    if "user" not in session:
        return redirect(url_for("login"))

    questions = session.get('questions', []) 

    if request.method == "POST":
        if "duration" in request.form:
            duration = int(request.form.get("duration"))
            questions = generate_questions(duration)
            session['questions'] = questions  
            session['duration'] = duration
            return jsonify({"questions": questions})  

        if "audio" in request.files:
            audio_file = request.files["audio"]
            audio_file.save(f"uploads/{audio_file.filename}")

            transcription = transcribe_audio_with_whisper(f"uploads/{audio_file.filename}")
            print("transcribed")
            filler_count = analyze_speech(transcription)
            print("analyzed")
            mfcc_features = extract_mfcc_features(f"uploads/{audio_file.filename}")
            print("mfcc features")
            feedback = get_llama_feedback_analogy(questions, transcription, mfcc_features, filler_count)
            print("feedback")

            return jsonify({"feedback": feedback})  
    return render_template("analogy.html", questions=questions)


if __name__ == "__main__":
    app.run(debug=True)
