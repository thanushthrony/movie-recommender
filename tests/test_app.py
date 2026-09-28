import pytest

from app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://"})
    return app.test_client()


def signup_and_login(client):
    client.post("/signup", data={"email": "ada@example.com", "username": "ada",
                                 "password": "correct horse", "confirm": "correct horse"})
    return client.post("/login", data={"email": "ada@example.com", "password": "correct horse"})


def test_pages_require_login(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_wrong_password_is_rejected(client):
    signup_and_login(client)
    client.get("/logout")
    response = client.post("/login", data={"email": "ada@example.com", "password": "nope"},
                           follow_redirects=True)
    assert b"Incorrect email or password" in response.data


def test_recommendation_flow(client):
    assert signup_and_login(client).status_code == 302

    assert client.get("/").status_code == 200

    response = client.get("/recommend?Action=5&Sci-Fi=5")
    assert response.status_code == 200
    assert b"Spider-Man" in response.data

    # No genres rated: back to the form with a message.
    response = client.get("/recommend", follow_redirects=True)
    assert b"Rate at least one genre" in response.data


def test_movie_search_and_users(client):
    signup_and_login(client)
    assert b"Spider-Man 2" in client.get("/search?q=spider").data
    assert client.get("/movie/5349").status_code == 200  # Spider-Man (2002)
    assert client.get("/movie/1").status_code == 404
    response = client.get("/users/2")
    assert response.status_code == 200
    assert b"mean absolute error" in response.data
