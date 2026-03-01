# Aligned Backend

FastAPI backend with MySQL: quiz, profile, and diagnostics APIs.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or: `make install` / `.\build.ps1 install`

2. **MySQL**
   - Create DB and tables: `make db-setup` or `.\build.ps1 db-setup`
   - Uses `.env` or env vars: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

3. **Run server**
   ```bash
   make run
   # or
   .\build.ps1 run
   ```
   API: **http://127.0.0.1:8000**  
   Docs: **http://127.0.0.1:8000/docs**

---

## Authentication

After Google login, the client must send the current user id on protected routes via header:

- **`X-User-Id`** — User id = `providerSub + provider` (e.g. `123456789google`).

---

## API Routes

### Quiz (`/quiz`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/quiz/` | No | Health / info |
| GET | `/quiz/getQuestion` | Yes | Get one question (standard JSON format) |
| POST | `/quiz/sendResponse` | Yes | Save answer for a question |
| POST | `/quiz/submit` | Yes | Submit quiz and save personality vector |

#### Quiz question JSON (GET `/quiz/getQuestion`)

Responses use this shape:

```json
{
  "id": 294029492059020,
  "question_type": 0,
  "question": {
    "number": 1,
    "text": "What is the middle of the world?"
  },
  "answers": [
    { "id": 0, "text": "foo" },
    { "id": 1, "text": "bar" },
    { "id": 2, "text": "baz" }
  ],
  "prior_response": null
}
```

- **`id`** — Unique question id (integer).
- **`question_type`** — `0`–`6`: e.g. `0` = single select, `1` = multi-select, `2` = scale, `3` = free text, `4` = ranking, `5` = yes/no, `6` = other.
- **`question`** — `number`: 1-based index; `text`: question text.
- **`answers`** — List of `{ "id": number, "text": string }`. Omit for free-text/scale types if not used.
- **`prior_response`** — If the user already answered this question, their saved `response_data`; otherwise `null`.

**Query params**

- `question_id` (optional) — Return this question by id.
- `index` (optional) — When `question_id` is omitted, 0-based index into the question list (default `0`).

**Example: send response (POST `/quiz/sendResponse`)**

```json
{
  "question_id": 294029492059020,
  "response_data": {
    "selected_ids": [1]
  }
}
```

For multi-select, `selected_ids` can be an array. For free text, e.g. `{ "text": "user answer" }`.

**Example: submit quiz (POST `/quiz/submit`)**

```json
{
  "personality_vector": [0.1, -0.2, 0.5, 0.0]
}
```

Stores the vector in the **characteristics** table (`personality_vector` trait).

---

### Characteristics

The **personality vector** is still a **vector of floats** (stored as BLOB). All other traits (star sign, Myers-Briggs, attachment style, etc.) are text and **all are populatable** via the API.

- **GET `/profile/getCharacteristics`** (auth) — Returns your characteristics: each `{ trait_key, value, is_public }`. `personality_vector` value is `list[float]`; others are strings.
- **POST `/profile/updateCharacteristics`** (auth) — Set/update traits. Body: `{ "traits": [ { "trait_key": "star_sign", "value": "Leo", "is_public": true }, { "trait_key": "personality_vector", "value": [0.1, -0.2], "is_public": false } ] }`. Use `value` as **list of floats** only for `personality_vector`; for all others use a string.

When `showPersonality` is true, **GET `/profile/getProfile`** returns a `characteristics` object with only **public** traits. Trait keys include: `personality_vector`, `star_sign`, `myers_briggs`, `attachment_style`, `enneagram_type`, `love_language`, `moral_foundation`, `political_leaning`, `humor_style`, `conflict_style`, `learning_style`, `big_five`, `chronotype`, `spirituality`, `life_philosophy`, `core_values`, `communication_style`, and more (see `db.CHARACTERISTIC_KEYS`).

---

### Profile (`/profile`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/profile/` | No | Health / info |
| POST | `/profile/register` | No | Register/update user after Google login |
| GET | `/profile/my` | Yes | Current user’s full profile |
| GET | `/profile/getSettings` | Yes | Current user’s settings |
| POST | `/profile/updateSettings` | Yes | Update user settings |
| POST | `/profile/updatePrivacy` | Yes | Update privacy (is_hidden, privacy_settings) |
| GET | `/profile/getProfile` | No | Public profile by `user_id` (respects privacy) |
| GET | `/profile/getCharacteristics` | Yes | Your characteristics (vector + traits, each with is_public) |
| POST | `/profile/updateCharacteristics` | Yes | Set/update traits (personality_vector = list[float]; others = string) |
| POST | `/profile/addFriend` | Yes | Add a friend |

**POST `/profile/register`** (body)

```json
{
  "provider_sub": "123456789",
  "provider": "google",
  "email": "user@example.com"
}
```

Returns `user_id` (= `providerSub + provider`) and full `user` object.

**GET `/profile/getProfile`** — Query: `?user_id=<id>`. Returns only fields allowed by target user’s privacy (e.g. `showEmail`, `showAge`, `showBirthday`, `showPersonality`). 404 if not found or profile hidden.

**POST `/profile/updateSettings`** (body)

```json
{
  "user_settings": { "theme": "dark", "notifications": true }
}
```

**POST `/profile/updatePrivacy`** (body)

```json
{
  "is_hidden": false,
  "privacy_settings": {
    "showEmail": false,
    "showAge": true,
    "showBirthday": false,
    "showPersonality": true
  }
}
```

**POST `/profile/addFriend`** (body)

```json
{
  "friend_id": "987654321google"
}
```

---

### Diagnostics (`/diagnostics`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/diagnostics/` | No | Health / info |
| GET | `/diagnostics/getDiagnostics` | Yes | List diagnostics for current user |

Returns `{ "diagnostics": [ { "id", "diagnostic_data", "created_at" }, ... ] }`.

---

## Build scripts

- **Makefile** (Git Bash / WSL): `make install`, `make db-setup`, `make run`, `make mysql`
- **PowerShell**: `.\build.ps1 install`, `.\build.ps1 db-setup`, `.\build.ps1 run`, `.\build.ps1 mysql`
