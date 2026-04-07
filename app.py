from flask import Flask, render_template, request, redirect, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"
# Load dataset
df = pd.read_csv("coursera.csv")
df = df.fillna("")

# Combine useful columns
df["combined"] = df["course"] + " " + df["skills"] + " " + df["level"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)


# Function to recommend courses
def recommend_from_input(user_profile):

    user_vec = vectorizer.transform([user_profile])

    sim_scores = cosine_similarity(user_vec, tfidf_matrix)

    top_indices = sim_scores.argsort()[0][-5:][::-1]

    recommended = df.iloc[top_indices][["course", "rating"]]

    return recommended.to_dict(orient="records")




# Home page
@app.route("/")
def home():
    if 'user' in session:
        return render_template("index.html")
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # simple demo login
        if email == "admin@gmail.com" and password == "1234":
            session["user"] = email
            return redirect("/")
        else:
            return "Invalid login"

    return render_template("login.html")
# Recommendation route
@app.route("/recommend", methods=["POST"])
def recommend():

    desired_skill = request.form["desired_skill"]
    current_skills = request.form["current_skills"]
    level = request.form["level"]
    duration = request.form["duration"]

    # Create user profile text
    user_profile = f"{desired_skill} {current_skills} {level} {duration}"

    courses = recommend_from_input(user_profile)

    return render_template("index.html", courses=courses)


if __name__ == "__main__":
    app.run(debug=True)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")