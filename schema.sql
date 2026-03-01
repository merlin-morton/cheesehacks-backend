-- Align Database Schema
-- Run this to create the database and tables.

CREATE DATABASE IF NOT EXISTS align;
USE align;

-- If you had personality_vector on users before: INSERT INTO characteristics (user_id, trait_key, value_blob, is_public)
--   SELECT id, 'personality_vector', personality_vector, FALSE FROM users WHERE personality_vector IS NOT NULL;
-- then drop the column from users.

-- Users table: core identity + profile + privacy (personality/characteristics in characteristics table)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY COMMENT 'providerSub + provider concatenated',
    provider VARCHAR(32) NOT NULL,
    provider_sub VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    birthday DATE NULL,
    age INT NULL,
    -- User preferences (JSON for flexibility: theme, notifications, etc.)
    user_settings JSON NULL DEFAULT ('{}'),
    -- Privacy: whether account appears in lookups
    is_hidden BOOLEAN NOT NULL DEFAULT FALSE,
    -- What to show when profile is looked up (JSON: showEmail, showAge, showBirthday, showPersonality)
    privacy_settings JSON NULL DEFAULT ('{"showEmail":false,"showAge":false,"showBirthday":false,"showPersonality":true}'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_provider_sub (provider, provider_sub)
);

-- Characteristics: personality vector (vector of floats, stored as BLOB) + other traits (text); each has is_public; all populatable
-- trait_key examples: personality_vector (floats), star_sign, myers_briggs, attachment_style, enneagram_type,
--   love_language, moral_foundation, political_leaning, humor_style, conflict_style, learning_style,
--   big_five, optimism_level, introvert_extrovert, chronotype, decision_style, creativity_style,
--   spirituality, life_philosophy, mindset, core_values, top_strengths, communication_style,
--   stress_response, motivation_style, risk_tolerance, perfectionism_level, empathy_style,
--   leadership_style, learning_orientation, time_orientation, self_monitoring, need_for_closure,
--   cognitive_style, emotional_expressiveness, assertiveness, emotional_intelligence, curiosity_level
CREATE TABLE IF NOT EXISTS characteristics (
    user_id VARCHAR(255) NOT NULL,
    trait_key VARCHAR(64) NOT NULL,
    value_text TEXT NULL,
    value_blob BLOB NULL,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    manually_overridden BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Set TRUE when set via POST /profile/updateCharacteristics; MLP callback will not overwrite',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, trait_key),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- If upgrading: ALTER TABLE characteristics ADD COLUMN manually_overridden BOOLEAN NOT NULL DEFAULT FALSE;

-- Friends: many-to-many relationship
CREATE TABLE IF NOT EXISTS friends (
    user_id VARCHAR(255) NOT NULL,
    friend_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, friend_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (friend_id) REFERENCES users(id) ON DELETE CASCADE,
    CHECK (user_id != friend_id)
);

-- Questions cache: MLP-generated or preloaded questions by question_id (avoids repeated MLP calls)
CREATE TABLE IF NOT EXISTS questions (
    question_id BIGINT NOT NULL PRIMARY KEY,
    question_data JSON NOT NULL COMMENT 'Full question: id, question_type, question: {number, text}, answers: [{id, text}]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quiz responses: store per-question responses before final submit (for getQuestion context)
CREATE TABLE IF NOT EXISTS quiz_responses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    question_id VARCHAR(64) NOT NULL,
    response_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_question (user_id, question_id)
);

-- Diagnostics: user-specific diagnostic data
CREATE TABLE IF NOT EXISTS diagnostics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    diagnostic_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
