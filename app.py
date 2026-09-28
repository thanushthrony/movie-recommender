"""Flask front end for the content-based movie recommender.

Run with:  python app.py   (then open http://127.0.0.1:5000)
"""
import os
import secrets

from flask import Flask, abort, flash, redirect, render_template, request, url_for
from flask_login import (LoginManager, current_user, login_required, login_user,
                         logout_user)
from werkzeug.security import check_password_hash, generate_password_hash

from data_utils import GENRES
from models import User, db
from recommender import Recommender

DEMO_URL = "https://thanushthrony.github.io/projects/recommender-demo.html"


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///users.db"),
    )
    if config:
        app.config.update(config)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    login_manager = LoginManager(app)
    login_manager.login_view = "login"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Load the data and trained weights once, not on every request.
    recommender = Recommender()

    @app.context_processor
    def globals_for_templates():
        return {"demo_url": DEMO_URL}

    # ---- accounts ---------------------------------------------------------

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm", "")

            if User.query.filter_by(email=email).first():
                flash("An account with that email already exists.", "error")
            elif User.query.filter_by(username=username).first():
                flash("That username is taken.", "error")
            elif "@" not in email:
                flash("Please enter a valid email address.", "error")
            elif len(username) < 2:
                flash("Username must be at least 2 characters.", "error")
            elif len(password) < 8:
                flash("Password must be at least 8 characters.", "error")
            elif password != confirm:
                flash("Passwords don't match.", "error")
            else:
                db.session.add(User(email=email, username=username,
                                    password_hash=generate_password_hash(password)))
                db.session.commit()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
        return render_template("signup.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, request.form.get("password", "")):
                login_user(user, remember=True)
                return redirect(url_for("index"))
            flash("Incorrect email or password.", "error")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    # ---- recommendations --------------------------------------------------

    @app.route("/")
    @login_required
    def index():
        return render_template("index.html", genres=GENRES, user=current_user)

    @app.route("/recommend")
    @login_required
    def recommend():
        ratings = {}
        for genre in GENRES:
            try:
                ratings[genre] = min(max(float(request.args.get(genre, 0)), 0.0), 5.0)
            except ValueError:
                ratings[genre] = 0.0
        if not any(ratings.values()):
            flash("Rate at least one genre to get recommendations.", "error")
            return redirect(url_for("index"))
        results = recommender.recommend(ratings, n=10)
        return render_template("results.html", results=results, ratings=ratings)

    @app.route("/movie/<int:movie_id>")
    @login_required
    def movie(movie_id):
        if movie_id not in recommender.row_of:
            abort(404)
        return render_template("movie.html", movie=recommender.movie(movie_id),
                               similar=recommender.similar(movie_id, n=8))

    @app.route("/search")
    @login_required
    def search():
        query = request.args.get("q", "")
        hits = recommender.search(query) if query.strip() else []
        return render_template("search.html", query=query, hits=hits)

    @app.route("/users/<int:user_id>")
    @login_required
    def movielens_user(user_id):
        try:
            genre_ratings, rated = recommender.training_user(user_id)
        except KeyError:
            abort(404)
        errors = [abs(m["predicted"] - m["actual"]) for m in rated]
        return render_template("user.html", user_id=user_id, genre_ratings=genre_ratings,
                               rated=rated, mae=sum(errors) / len(errors),
                               user_ids=recommender.user_ids()[:40])

    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
